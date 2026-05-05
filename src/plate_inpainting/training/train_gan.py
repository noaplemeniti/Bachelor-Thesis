import os
import random
import time
import shutil
from argparse import ArgumentParser
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT_DIR = PROJECT_ROOT / "src" / "plate_inpainting"
if str(PACKAGE_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_DIR))

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from trainer_gan import Trainer
from data.dataset import InpaintingDataset
from utils.tools import get_config
from utils.logger import get_logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

parser = ArgumentParser()
parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'config' / 'gan.yaml'),
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')

def main():
    args = parser.parse_args() 
    config = get_config(args.config)

    # CUDA configuration
    requested_cuda = config['cuda']
    device_ids = config['gpu_ids']
    if requested_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)

    cuda = requested_cuda and torch.cuda.is_available()
    config['cuda'] = cuda
    device = torch.device('cuda:0' if cuda else 'cpu')

    if cuda:
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call

    logger.info("Arguments: {}".format(args))
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Log the configuration
    logger.info("Configuration: {}".format(config))

    try:  # for unexpected error logging
        # Load the dataset
        logger.info("Training on dataset: {}".format(config['dataset_name']))
        train_dataset = InpaintingDataset(root_dir=config['train_data_path'])
        # val_dataset = InpaintingDataset(data_path=config['val_data_path'])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'],
                                                   pin_memory=cuda,
                                                   persistent_workers=config['num_workers'] > 0)
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
        #                                           batch_size=config['batch_size'],
        #                                           shuffle=False,
        #                                           num_workers=config['num_workers'])

        # Define the trainer
        trainer = Trainer(config)
        logger.info("\n{}".format(trainer.netG))
        logger.info("\n{}".format(trainer.localD))
        logger.info("\n{}".format(trainer.globalD))

        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        # Get the resume iteration to restart training
        start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1

        iterable_train_loader = iter(train_loader)

        time_count = time.time()

        for iteration in range(start_iteration, config['niter'] + 1):
            try:
                x, mask, ground_truth = next(iterable_train_loader)
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                x, mask, ground_truth = next(iterable_train_loader)

            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            ground_truth = ground_truth.to(device, non_blocking=True)

            ###### Forward pass ######
            compute_g_loss = iteration % config['n_critic'] == 0

            # Run D and G on separate forward passes. Reusing one graph across
            # both optimizers breaks once D is stepped before G backward.
            losses, inpainted_result, offset_flow = trainer(x, mask, ground_truth, False)
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()
            trainer_module.optimizer_d.step()

            # Update G
            if compute_g_loss:
                losses_g, inpainted_result, offset_flow = trainer(x, mask, ground_truth, True)
                for k in losses_g.keys():
                    if not losses_g[k].dim() == 0:
                        losses_g[k] = torch.mean(losses_g[k])
                losses.update(losses_g)
                trainer_module.optimizer_g.zero_grad()
                losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                              + losses['ae'] * config['ae_loss_alpha'] \
                              + losses['wgan_g'] * config['gan_loss_alpha']
                losses['g'].backward()
                trainer_module.optimizer_g.step()

            # Log and visualization
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = losses.get(k, 0.)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)

            if iteration % (config['viz_iter']) == 0:
                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
                                              offset_flow[:viz_max_out]], dim=1)
                else:
                    viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
                viz_images = viz_images.view(-1, *list(x.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (checkpoint_path, iteration),
                                  nrow=3 * 4,
                                  normalize=True)

            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)

    except Exception as e:  # for unexpected error logging
        logger.error("{}".format(e))
        raise e

if __name__ == '__main__':
    main()
