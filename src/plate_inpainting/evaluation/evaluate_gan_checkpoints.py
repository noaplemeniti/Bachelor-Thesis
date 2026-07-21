"""Evaluate every saved GAN generator checkpoint on a deterministic validation set.

The output deliberately stays separate from the training-epoch CSV: checkpoints are
saved at fixed iterations, which generally do not coincide with epoch boundaries.
"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "plate_inpainting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from data.dataset import InpaintingDataset
from evaluation.image_metrics import compute_image_metrics
from training.trainer_gan import Trainer
from utils.tools import get_config


class DeterministicInpaintingDataset(InpaintingDataset):
    """Generate a stable random mask for each validation image."""

    def __init__(self, *args, seed=2026, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def __getitem__(self, idx):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        random.seed(self.seed + idx)
        np.random.seed((self.seed + idx) % (2**32))
        try:
            return super().__getitem__(idx)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def checkpoint_iteration(path):
    return int(path.stem.rsplit("_", 1)[1])


def resolve_device(requested):
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda"), "cuda"

    if requested in {"auto", "directml"}:
        try:
            import torch_directml
        except ImportError:
            if requested == "directml":
                raise RuntimeError(
                    "DirectML was requested but torch-directml is not installed"
                ) from None
        else:
            return torch_directml.device(), "directml"

    return torch.device("cpu"), "cpu"


def evaluate_checkpoint(trainer, checkpoint, loader, device):
    # Loading on CPU first is supported by every backend, including DirectML's
    # PrivateUse1 device. load_state_dict then copies parameters to the device.
    trainer.netG.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    trainer.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for masked_images, masks, truths in loader:
            masked_images = masked_images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            predictions, _ = trainer.inference(masked_images, masks)
            predictions = predictions.clamp(0.0, 1.0).cpu()
            batch_psnr, batch_ssim = compute_image_metrics(predictions, truths)
            batch_size = truths.size(0)
            psnr_sum += batch_psnr * batch_size
            ssim_sum += batch_ssim * batch_size
            sample_count += batch_size

    return ssim_sum / sample_count, psnr_sum / sample_count, sample_count


def main():
    parser = ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device", choices=("auto", "directml", "cuda", "cpu"), default="auto",
        help="Compute backend (auto prefers CUDA, then DirectML, then CPU)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = get_config(run_dir / "gan.yaml")
    device, device_name = resolve_device(args.device)
    config["cuda"] = device_name == "cuda"
    config["gpu_ids"] = config.get("gpu_ids", [0])

    validation_path = (PROJECT_ROOT / config["val_data_path"]).resolve()
    dataset = DeterministicInpaintingDataset(
        validation_path,
        random_color_flag=config.get("random_color_flag", True),
        mask_colors=config.get("mask_colors"),
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device_name == "cuda",
    )
    checkpoints = sorted((run_dir / "models").glob("gen_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No generator checkpoints in {run_dir / 'models'}")

    output = run_dir / "metrics" / f"{run_dir.name.lower()}_validation_checkpoint_metrics.csv"
    completed = {}
    if output.exists() and not args.force:
        with output.open(newline="", encoding="utf-8") as handle:
            completed = {int(row["iteration"]): row for row in csv.DictReader(handle)}

    trainer = Trainer(config).to(device)
    rows = [] if args.force else list(completed.values())
    for checkpoint in tqdm(checkpoints, desc=f"Validating {run_dir.name} on {device_name}"):
        iteration = checkpoint_iteration(checkpoint)
        if iteration in completed and not args.force:
            continue
        val_ssim, val_psnr, count = evaluate_checkpoint(trainer, checkpoint, loader, device)
        rows.append({
            "iteration": iteration,
            "checkpoint": checkpoint.name,
            "val_ssim": f"{val_ssim:.10f}",
            "val_psnr": f"{val_psnr:.10f}",
            "validation_samples": count,
            "mask_seed": args.seed,
        })
        rows.sort(key=lambda row: int(row["iteration"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"iteration {iteration}: val_ssim={val_ssim:.4f}, val_psnr={val_psnr:.2f} dB")

    print(f"Saved {output}")


if __name__ == "__main__":
    main()
