import os
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


class InpaintingDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.images = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])

        image = np.array(Image.open(image_path).convert("RGB"))

        masked_image, mask = brush_strokes(image)
        if masked_image is None or mask is None:
            raise ValueError(f"Could not generate mask for image: {image_path}")
        
        masked_image = Image.fromarray(masked_image)

        image = self.transform(image)
        masked_image = self.transform(masked_image)
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return masked_image, mask, image
