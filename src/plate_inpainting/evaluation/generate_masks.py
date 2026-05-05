import sys
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT_DIR = PROJECT_ROOT / "src" / "plate_inpainting"
if str(PACKAGE_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_DIR))

from data.masks import brush_strokes

parser = ArgumentParser(description="Generate masks for license plate inpainting")
parser.add_argument("--data_path", type=Path, default=PROJECT_ROOT / "data" / "license_plates_synth" / "test", help="Path to the data for generating masks.")
parser.add_argument("--length", type=int, default=50, help="Length of the stroke.")
parser.add_argument("--width", type=int, default=15, help="Width of the stroke.")
parser.add_argument("--count", type=int, default=5, help="Number of strokes.")
parser.add_argument("--color", type=int, nargs=3, default=(255, 255, 255), help="RGB color applied to the masked image.")
parser.add_argument("--padding_color", type=int, nargs=3, default=(127, 127, 127), help="RGB color of the padding.")

args = parser.parse_args()

def generate_masks():
    data_path = args.data_path
    image_dir = data_path / "images"
    mask_dir = data_path / "masks"
    masked_image_dir = data_path / "masked_images"

    for image in image_dir.iterdir():
        image_path = image.resolve()
        mask_path = mask_dir / image.name
        masked_image_path = masked_image_dir / image.name
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        masked_image, mask = brush_strokes(image, args.length, args.width, args.count, args.color, args.padding_color)
        mask = Image.fromarray(mask)
        masked_image = Image.fromarray(masked_image)
        mask.save(mask_path)
        masked_image.save(masked_image_path)
        print(f"Processed image: {image_path}")


if __name__ == "__main__":
    generate_masks()
