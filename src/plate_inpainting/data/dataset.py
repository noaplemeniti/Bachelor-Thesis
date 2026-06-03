import os
import random
from PIL import Image
from pathlib import Path
import sys
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT_DIR = PROJECT_ROOT / "src" / "plate_inpainting"
if str(PACKAGE_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_DIR))

from data.masks import brush_strokes


def _normalize_rgb_color(color):
    if len(color) != 3:
        raise ValueError(f"Mask color must contain exactly 3 values, got: {color}")

    normalized = tuple(int(channel) for channel in color)
    if any(channel < 0 or channel > 255 for channel in normalized):
        raise ValueError(f"Mask color values must be between 0 and 255, got: {color}")

    return normalized


def _normalize_mask_colors(mask_colors):
    if mask_colors is None:
        return None

    if len(mask_colors) == 0:
        raise ValueError("mask_colors must contain at least one RGB color")

    return [_normalize_rgb_color(color) for color in mask_colors]


class InpaintingDataset(data.Dataset):
    def __init__(self, root_dir, random_color_flag=True, mask_color=(0, 0, 0), mask_colors=None):
        self.root_dir = root_dir
        self.random_color_flag = random_color_flag
        self.mask_color = _normalize_rgb_color(mask_color)
        self.mask_colors = _normalize_mask_colors(mask_colors)
        self.image_dir = os.path.join(root_dir, "images")
        self.images = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def _sample_mask_color(self):
        if self.mask_colors is not None:
            return random.choice(self.mask_colors)

        return self.mask_color

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask_color = self._sample_mask_color()

        masked_image, mask = brush_strokes(
            image,
            random_color_flag=self.random_color_flag,
            mask_color=mask_color,
        )
        if masked_image is None or mask is None:
            raise ValueError(f"Could not generate mask for image: {image_path}")
        
        # Convert arrays back into standard object formats for the ToTensor transform pipelines
        masked_image = Image.fromarray(masked_image)
        image = Image.fromarray(image)

        # PyTorch Tensor Formats
        image = self.transform(image)
        masked_image = self.transform(masked_image)
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return masked_image, mask, image
