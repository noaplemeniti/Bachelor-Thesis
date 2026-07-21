"""Generate GAN training curves from final_results/GAN_*/metrics/*.csv."""

import argparse
from pathlib import Path

from plot_training_metrics import (
    DEFAULT_RESULTS_DIR,
    experiment_name,
    plot_experiment,
    plot_quality_comparison,
    print_summary,
    read_metrics,
)


def rebase_finetune_timeline(name, rows):
    """Show fine-tuning progress from zero instead of the resumed iteration."""
    if name.lower() != "gan_finetune" or not rows:
        return rows

    first_iteration = rows[0]["iteration"]
    first_epoch = rows[0].get("epoch")
    rebased = []
    for row in rows:
        adjusted = dict(row)
        adjusted["iteration"] -= first_iteration
        if first_epoch is not None:
            adjusted["epoch"] -= first_epoch
        rebased.append(adjusted)
    return rebased


def sample_every(rows, interval=10):
    """Keep every nth recorded value while always retaining the final point."""
    sampled = rows[::interval]
    if rows and sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Plot GAN training metrics.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "metric_curves" / "gan").resolve()
    csv_paths = sorted(results_dir.glob("GAN_*/metrics/*_epoch_metrics.csv"))
    if not csv_paths:
        parser.error(f"No GAN metric CSV files found under {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    for csv_path in csv_paths:
        fields, rows = read_metrics(csv_path)
        required = {"l1", "ae", "wgan_g", "wgan_d", "wgan_gp", "g", "d", "ssim", "psnr"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Missing GAN columns in {csv_path}: {sorted(missing)}")
        name = experiment_name(csv_path, results_dir)
        rows = rebase_finetune_timeline(name, rows)
        rows = sample_every(rows, interval=10)
        experiments.append((name, fields, rows))
        plot_experiment(name, fields, rows, output_dir, x_column="iteration")

    plot_quality_comparison(
        experiments,
        output_dir,
        filename="gan_quality_comparison.png",
        title="GAN metrics comparison",
        x_column="iteration",
    )
    print_summary(experiments, x_column="iteration")
    print(f"\nGAN plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
