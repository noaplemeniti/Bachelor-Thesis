from argparse import ArgumentParser
from pathlib import Path
import sys

from PIL import Image
import torch
from torchvision import transforms
import tqdm

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT_DIR = PROJECT_ROOT_DIR / "src" / "plate_inpainting"
if str(PACKAGE_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_DIR))

from training.trainer_gan import Trainer
from utils.tools import get_config, tensor_to_pil_image


parser = ArgumentParser(description="Generate GAN predictions for license plate inpainting")
parser.add_argument("--config", type=Path, default=PROJECT_ROOT_DIR / "config" / "gan.yaml", help="Path to a config file.")
parser.add_argument("--model", type=Path, default=PROJECT_ROOT_DIR / "checkpoints" / "GAN" / "models", help="Path to a gen_*.pt file or a directory containing generator checkpoints.")
parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT_DIR / "data" / "license_plates_synth" / "test", help="Path to the data for generating predictions.")
parser.add_argument("--predictions_dir_name", type=str, default="predictions", help="Name of the output predictions directory inside data_path.")


def resolve_generator_checkpoint(model_path):
    if model_path.is_dir():
        checkpoints = sorted(model_path.glob("gen_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No generator checkpoint found in: {model_path}")
        return checkpoints[-1]

    if not model_path.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {model_path}")

    return model_path


def generate_gan_predictions():
    args = parser.parse_args()
    config = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["cuda"] = device.type == "cuda"
    config["gpu_ids"] = config.get("gpu_ids", [0])

    trainer = Trainer(config).to(device)
    generator_checkpoint = resolve_generator_checkpoint(args.model)
    trainer.netG.load_state_dict(torch.load(generator_checkpoint, map_location=device))
    trainer.eval()

    mask_dir = args.data_path / "masks"
    masked_image_dir = args.data_path / "masked_images"
    prediction_dir = args.data_path / args.predictions_dir_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()

    masked_image_paths = sorted(
        p for p in masked_image_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    with torch.no_grad():
        for masked_image_path in tqdm.tqdm(masked_image_paths, desc="Generating GAN predictions"):
            mask_path = mask_dir / masked_image_path.name
            if not mask_path.exists():
                print(f"Missing mask for {masked_image_path.name}")
                continue

            mask = to_tensor(Image.open(mask_path).convert("L")).unsqueeze(0)
            masked_image = to_tensor(Image.open(masked_image_path).convert("RGB")).unsqueeze(0)
            mask = mask.float().to(device, non_blocking=True)
            masked_image = masked_image.float().to(device, non_blocking=True)

            prediction, _ = trainer.inference(masked_image, mask)
            prediction = torch.clamp(prediction, 0.0, 1.0)

            prediction_image = tensor_to_pil_image(prediction.squeeze(0))
            prediction_image.save(prediction_dir / masked_image_path.name)


if __name__ == "__main__":
    generate_gan_predictions()
