import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT_DIR = PROJECT_ROOT / "src" / "plate_inpainting"
if str(PACKAGE_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_DIR))


from models.unet import get_unet
from data.dataset import InpaintingDataset
from utils.tools import get_config

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

    training_dataset = InpaintingDataset(root_dir=config["train_data_path"])

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
        val_dataset = InpaintingDataset(root_dir=val_path)

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

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

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
            outputs = torch.clamp(outputs, 0.0, 1.0)

            loss = masked_l1_loss(outputs, images, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0

            with torch.no_grad():
                for masked_images, masks, images in val_loader:
                    masked_images = masked_images.float().to(device, non_blocking=True)
                    masks = masks.float().to(device, non_blocking=True)
                    images = images.float().to(device, non_blocking=True)

                    model_input = torch.cat([masked_images, masks], dim=1)

                    outputs = model(model_input)
                    outputs = torch.clamp(outputs, 0.0, 1.0)

                    loss = masked_l1_loss(outputs, images, masks)

                    val_running_loss += loss.item() * images.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)

        metric = val_loss if val_loss is not None else train_loss

        if metric < best_loss:
            best_loss = metric
            improvement_counter = 0
            save_dir = os.path.join("checkpoints",config['dataset_name'])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, f"{epoch}_unet_model.pth"))
        else:
            improvement_counter += 1

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss:.4f} "
            + (f"Val Loss: {val_loss:.4f}" if val_loss is not None else "")
        )

        if improvement_counter >= stopping_patience:
            print("Early stopping triggered")
            break

    return model


if __name__ == "__main__":
    save_dir = os.path.join("checkpoints", config['dataset_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = train_unet()
    torch.save(model.state_dict(), os.path.join(save_dir, f"final_{config['dataset_name']}_unet_model.pth"))
