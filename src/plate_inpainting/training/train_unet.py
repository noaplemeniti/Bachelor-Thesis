import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import os
import sys
import csv  #
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT_DIR = PROJECT_ROOT / "src" / "plate_inpainting"
if str(PACKAGE_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_DIR))


from models.unet import get_unet
from data.dataset import InpaintingDataset
from utils.tools import get_config
from evaluation.image_metrics import compute_image_metrics  #
from visualization import get_random_visual_sample, save_epoch_quadrant, should_save_epoch_visual

parser = ArgumentParser(description="Train a UNet model for license plate inpainting")
parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / 'config' / 'unet.yaml'),help="Path to a config file.")

args = parser.parse_args()
config = get_config(args.config)

def masked_l1_loss(pred, target, mask):
    hole_loss = (torch.abs(pred - target) * mask).sum() / (
        mask.sum() * pred.shape[1] + 1e-8
    )

    valid_loss = (torch.abs(pred - target) * (1 - mask)).sum() / (
        (1 - mask).sum() * pred.shape[1] + 1e-8
    )

    return 6.0 * hole_loss + 1.0 * valid_loss


def save_unet_epoch_visual(model, dataset, device, output_dir, epoch, timing):
    was_training = model.training
    model.eval()

    with torch.no_grad():
        masked_image, mask, image = get_random_visual_sample(dataset, device)
        model_input = torch.cat([masked_image, mask], dim=1)
        generated_image = torch.clamp(model(model_input), 0.0, 1.0)

    if was_training:
        model.train()

    return save_epoch_quadrant(
        image,
        masked_image,
        mask,
        generated_image,
        output_dir,
        epoch,
        timing,
        "unet",
    )


def train_unet():
    requested_device = config.get("device", "cuda")
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    pin_memory = device.type == "cuda"

    model = get_unet(
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
    ).to(device)

    checkpoint_path = config["resume"]
    if checkpoint_path and checkpoint_path.lower() != "none":
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Checkpoint not found at {checkpoint_path}; training from scratch.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    training_dataset = InpaintingDataset(
        root_dir=config["train_data_path"],
        random_color_flag=config.get("random_color_flag", True),
        mask_color=config.get("mask_color", (0, 0, 0)),
        mask_colors=config.get("mask_colors"),
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    val_loader = None
    val_path = config["val_data_path"]
    if val_path is not None:
        val_dataset = InpaintingDataset(
            root_dir=val_path,
            random_color_flag=config.get("random_color_flag", True),
            mask_color=config.get("mask_color", (0, 0, 0)),
            mask_colors=config.get("mask_colors"),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            persistent_workers=True if config["num_workers"] > 0 else False,
        )

    best_loss = float("inf")
    improvement_counter = 0

    print("Starting training on device:", device)

    epochs = config["epochs"]
    stopping_patience = config["stopping_patience"]
    checkpoint_root = PROJECT_ROOT / "checkpoints" / "Unet"
    model_dir = checkpoint_root / "models"
    visual_dir = checkpoint_root / "epoch_quadrants"
    metrics_dir = checkpoint_root / "metrics"
    for output_dir in (model_dir, visual_dir, metrics_dir):
        os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(checkpoint_root, os.path.basename(args.config)))
    metrics_path = os.path.join(metrics_dir, "unet_epoch_metrics.csv")  #
    if not os.path.exists(metrics_path):  #
        with open(metrics_path, "w", newline="") as metrics_file:  #
            metrics_writer = csv.writer(metrics_file)  #
            metrics_writer.writerow(["epoch", "train_loss", "train_ssim", "train_psnr", "val_loss", "val_ssim", "val_psnr"])  #

    for epoch in range(epochs):
        if should_save_epoch_visual(config, "start"):
            visual_path = save_unet_epoch_visual(model, training_dataset, device, visual_dir, epoch + 1, "start")
            print(f"Saved epoch visualization: {visual_path}")

        model.train()
        running_loss = 0.0
        running_psnr = 0.0  #
        running_ssim = 0.0  #

        for masked_images, masks, images in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            mininterval=5,
        ):
            masked_images = masked_images.float().to(device, non_blocking=True)
            masks = masks.float().to(device, non_blocking=True)
            images = images.float().to(device, non_blocking=True)

            model_input = torch.cat([masked_images, masks], dim=1)

            optimizer.zero_grad()
            outputs = model(model_input)

            loss = masked_l1_loss(outputs, images, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            metric_outputs = torch.clamp(outputs.detach(), 0.0, 1.0)
            batch_psnr, batch_ssim = compute_image_metrics(metric_outputs.cpu(), images.detach().cpu())  #
            running_psnr += batch_psnr * images.size(0)  #
            running_ssim += batch_ssim * images.size(0)  #

        train_loss = running_loss / len(train_loader.dataset)
        train_psnr = running_psnr / len(train_loader.dataset)  #
        train_ssim = running_ssim / len(train_loader.dataset)  #

        val_loss = None
        val_psnr = None  #
        val_ssim = None  #
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_running_psnr = 0.0  #
            val_running_ssim = 0.0  #

            with torch.no_grad():
                for masked_images, masks, images in val_loader:
                    masked_images = masked_images.float().to(device, non_blocking=True)
                    masks = masks.float().to(device, non_blocking=True)
                    images = images.float().to(device, non_blocking=True)

                    model_input = torch.cat([masked_images, masks], dim=1)

                    outputs = model(model_input)

                    loss = masked_l1_loss(outputs, images, masks)

                    val_running_loss += loss.item() * images.size(0)
                    metric_outputs = torch.clamp(outputs.detach(), 0.0, 1.0)
                    batch_psnr, batch_ssim = compute_image_metrics(metric_outputs.cpu(), images.detach().cpu())  #
                    val_running_psnr += batch_psnr * images.size(0)  #
                    val_running_ssim += batch_ssim * images.size(0)  #

            val_loss = val_running_loss / len(val_loader.dataset)
            val_psnr = val_running_psnr / len(val_loader.dataset)  #
            val_ssim = val_running_ssim / len(val_loader.dataset)  #

        metric = val_loss if val_loss is not None else train_loss

        if metric < best_loss:
            best_loss = metric
            improvement_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f"{epoch}_unet_model.pth"))
        else:
            improvement_counter += 1

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss:.4f} "
            f"Train SSIM: {train_ssim:.4f} "  #
            f"Train PSNR: {train_psnr:.4f} "  #
            + (f"Val Loss: {val_loss:.4f}" if val_loss is not None else "")
            + (f" Val SSIM: {val_ssim:.4f} Val PSNR: {val_psnr:.4f}" if val_loss is not None else "")  #
        )
        with open(metrics_path, "a", newline="") as metrics_file:  #
            metrics_writer = csv.writer(metrics_file)  #
            metrics_writer.writerow([epoch + 1, train_loss, train_ssim, train_psnr, val_loss, val_ssim, val_psnr])  #
        if should_save_epoch_visual(config, "end"):
            visual_path = save_unet_epoch_visual(model, training_dataset, device, visual_dir, epoch + 1, "end")
            print(f"Saved epoch visualization: {visual_path}")

        if improvement_counter >= stopping_patience:
            print("Early stopping triggered")
            break

    return model


if __name__ == "__main__":
    save_dir = PROJECT_ROOT / "checkpoints" / "Unet" / "models"
    os.makedirs(save_dir, exist_ok=True)
    model = train_unet()
    torch.save(model.state_dict(), os.path.join(save_dir, f"final_{config['dataset_name']}_unet_model.pth"))
