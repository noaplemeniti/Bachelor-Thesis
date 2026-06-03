import os
import random

import torch
import torchvision.utils as vutils


def should_save_epoch_visual(config, timing):
    configured_timing = config.get("epoch_visualization_timing", "end")
    if configured_timing is None:
        return False

    configured_timing = str(configured_timing).lower()
    return configured_timing == timing or configured_timing == "both"


def get_random_visual_sample(dataset, device):
    masked_image, mask, image = dataset[random.randrange(len(dataset))]
    return (
        masked_image.unsqueeze(0).float().to(device),
        mask.unsqueeze(0).float().to(device),
        image.unsqueeze(0).float().to(device),
    )


def save_epoch_quadrant(image, masked_image, mask, generated_image, output_dir, epoch, timing, prefix):
    os.makedirs(output_dir, exist_ok=True)

    image = image.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    masked_image = masked_image.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    generated_image = generated_image.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    mask = mask.detach().cpu().squeeze(0).clamp(0.0, 1.0)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.size(0) == 1:
        mask = mask.repeat(3, 1, 1)

    quadrants = torch.stack([image, masked_image, mask, generated_image], dim=0)
    output_path = os.path.join(output_dir, f"{prefix}_epoch_{epoch:04d}_{timing}.png")
    vutils.save_image(quadrants, output_path, nrow=2, padding=2, normalize=False)
    return output_path
