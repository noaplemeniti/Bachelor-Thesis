from argparse import ArgumentParser
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT_DIR = PROJECT_ROOT_DIR / "src" / "plate_inpainting"
if str(PACKAGE_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_DIR))

from utils.tools import get_config, tensor_to_pil_image
from models.unet import get_unet
import tqdm


parser = ArgumentParser(description="Train a UNet model for license plate inpainting")
parser.add_argument("--config", type=Path, default=PROJECT_ROOT_DIR / "config" / "unet.yaml", help="Path to a config file.")
parser.add_argument("--model", type=Path, default=PROJECT_ROOT_DIR / "checkpoints" / "unet_model.pth", help="Path to a model file.")
parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT_DIR / "data" / "license_plates_synth" / "test", help="Path to the data for generating predictions.")

def generate_unet_predictions():
    args = parser.parse_args()
    config = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet(encoder_name=config["encoder_name"], encoder_weights=config["encoder_weights"], in_channels=config["in_channels"], num_classes=config["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    mask_dir = args.data_path / "masks"
    masked_image_dir = args.data_path / "masked_images"
    prediction_dir = args.data_path / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()

    masked_image_paths = sorted(
        p for p in masked_image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    with torch.no_grad():
        for masked_image_path in tqdm.tqdm(masked_image_paths, desc="Generating predictions"):
            mask_path = mask_dir / masked_image_path.name
            if not mask_path.exists():
                print(f"Missing mask for {masked_image_path.name}")
                continue

            mask = to_tensor(Image.open(mask_path).convert("L")).unsqueeze(0)
            masked_image = to_tensor(Image.open(masked_image_path).convert("RGB")).unsqueeze(0)
            mask = mask.float().to(device, non_blocking=True)
            masked_image = masked_image.float().to(device, non_blocking=True)

            input_tensor = torch.cat([masked_image, mask], dim=1)

            prediction = model(input_tensor)
            prediction = torch.clamp(prediction, 0.0, 1.0)

            prediction_image = tensor_to_pil_image(prediction.squeeze(0))
            prediction_image.save(prediction_dir / masked_image_path.name)

if __name__ == "__main__":
    generate_unet_predictions()
