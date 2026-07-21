"""Generate U-Net training curves from final_results/Unet_*/metrics/*.csv."""

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


UNET_L1_WEIGHT_SUM = 7.0  # 6 * hole MAE + 1 * valid-region MAE


def normalize_l1_for_plot(fields, rows):
    """Return plot-only rows; never mutate values read from the source CSV."""
    normalized_fields = [
        "normalized_train_loss" if field == "train_loss"
        else "normalized_val_loss" if field == "val_loss"
        else field
        for field in fields
    ]
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized["normalized_train_loss"] = normalized.pop("train_loss") / UNET_L1_WEIGHT_SUM
        normalized["normalized_val_loss"] = normalized.pop("val_loss") / UNET_L1_WEIGHT_SUM
        normalized_rows.append(normalized)
    return normalized_fields, normalized_rows


def main():
    parser = argparse.ArgumentParser(description="Plot U-Net training metrics.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "metric_curves" / "unet_normalized_l1").resolve()
    csv_paths = sorted(results_dir.glob("Unet_*/metrics/*.csv"))
    if not csv_paths:
        parser.error(f"No U-Net metric CSV files found under {results_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    for csv_path in csv_paths:
        fields, rows = read_metrics(csv_path)
        required = {"train_loss", "train_ssim", "train_psnr", "val_loss", "val_ssim", "val_psnr"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Missing U-Net columns in {csv_path}: {sorted(missing)}")
        name = experiment_name(csv_path, results_dir)
        fields, rows = normalize_l1_for_plot(fields, rows)
        experiments.append((name, fields, rows))
        plot_experiment(name, fields, rows, output_dir)

    plot_quality_comparison(
        experiments,
        output_dir,
        filename="unet_quality_comparison.png",
        title="U-Net metrics comparison (L1 normalized by 6 + 1 weights)",
    )
    print_summary(experiments)
    print(f"\nU-Net plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
