"""Plot deterministic GAN validation SSIM and PSNR at saved checkpoints."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "final_results")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir / "metric_curves" / "gan_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.results_dir.glob("GAN_*/metrics/*_validation_checkpoint_metrics.csv"))
    if not paths:
        parser.error("No GAN validation checkpoint CSV files found")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        label = path.parents[1].name.replace("GAN_", "").replace("_", " ").title()
        iterations = [int(row["iteration"]) for row in rows]
        if path.parents[1].name.lower() == "gan_finetune" and iterations:
            iterations = [iteration - iterations[0] for iteration in iterations]
        ssim = [float(row["val_ssim"]) for row in rows]
        psnr = [float(row["val_psnr"]) for row in rows]
        axes[0].plot(iterations, ssim, marker="o", markersize=3, label=label)
        axes[1].plot(iterations, psnr, marker="o", markersize=3, label=label)

    axes[0].set(title="Validation SSIM", xlabel="Training iteration", ylabel="SSIM")
    axes[1].set(title="Validation PSNR", xlabel="Training iteration", ylabel="PSNR (dB)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle("GAN validation quality at saved checkpoints")
    figure.tight_layout()
    output = output_dir / "gan_validation_quality_comparison.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
