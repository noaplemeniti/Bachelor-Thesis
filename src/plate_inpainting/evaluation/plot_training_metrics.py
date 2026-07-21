"""Generate training curves from every metrics CSV under ``final_results``.

The script understands both metric formats used by this project:

* U-Net: training and validation loss, SSIM, and PSNR.
* GAN: reconstruction/adversarial losses, SSIM, and PSNR.

Run from the project root with::

    uv run python src/plate_inpainting/evaluation/plot_training_metrics.py
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "final_results"
NON_METRIC_COLUMNS = {"epoch", "iteration"}

METRIC_LABELS = {
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
    "normalized_train_loss": "Normalized training L1 loss",
    "normalized_val_loss": "Normalized validation L1 loss",
    "train_ssim": "Training SSIM",
    "val_ssim": "Validation SSIM",
    "train_psnr": "Training PSNR",
    "val_psnr": "Validation PSNR",
    "l1": "L1 loss",
    "ae": "AE loss",
    "wgan_g": "WGAN generator loss",
    "wgan_d": "WGAN discriminator loss",
    "wgan_gp": "Gradient penalty",
    "g": "Total generator loss",
    "d": "Total discriminator loss",
    "ssim": "SSIM",
    "psnr": "PSNR",
}


def display_name(name):
    suffix = name.lower().split("_", maxsplit=1)[-1]
    return {
        "synth": "Synthetic images",
        "real": "Real images",
        "finetune": "Finetune",
    }.get(suffix, name)


def experiment_title(name):
    model, _, suffix = name.partition("_")
    model = "U-Net" if model.lower() == "unet" else model.upper()
    dataset = {
        "real": "Stvarne slike",
        "synth": "Sintetičke slike",
        "finetune": "Finetune",
    }.get(suffix.lower(), suffix)
    return f"{model}: {dataset}"


def axis_label(column):
    return {"epoch": "Epoch", "iteration": "Iteration"}.get(column, column)


def read_metrics(csv_path):
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError(f"Missing an 'epoch' column in {csv_path}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append({name: float(value) for name, value in row.items() if value != ""})
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {csv_path}, line {line_number}"
                ) from error

    if not rows:
        raise ValueError(f"No metric rows found in {csv_path}")
    return reader.fieldnames, rows


def experiment_name(csv_path, results_dir):
    relative_path = csv_path.relative_to(results_dir)
    return relative_path.parts[0]


def plot_experiment(name, fields, rows, output_dir, x_column="epoch"):
    metric_names = [field for field in fields if field not in NON_METRIC_COLUMNS]
    column_count = 3
    row_count = math.ceil(len(metric_names) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 3.6 * row_count),
        squeeze=False,
    )
    x_values = [row[x_column] for row in rows]
    x_label = axis_label(x_column)

    for axis, metric_name in zip(axes.flat, metric_names):
        axis.plot(x_values, [row[metric_name] for row in rows], linewidth=1.5)
        axis.set_title(METRIC_LABELS.get(metric_name, metric_name.replace("_", " ")))
        axis.set_xlabel(x_label)
        axis.set_ylabel(METRIC_LABELS.get(metric_name, metric_name.replace("_", " ")))
        axis.grid(True, alpha=0.3)

    for axis in list(axes.flat)[len(metric_names):]:
        axis.set_visible(False)

    figure.suptitle(experiment_title(name), fontsize=15)
    figure.tight_layout()
    output_path = output_dir / f"{name.lower()}_metrics.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def quality_columns(fields):
    """Use validation quality for U-Net and the logged quality for GAN."""
    if "val_ssim" in fields and "val_psnr" in fields:
        return "val_ssim", "val_psnr"
    if "ssim" in fields and "psnr" in fields:
        return "ssim", "psnr"
    return None


def loss_column(fields):
    if "normalized_val_loss" in fields:
        return "normalized_val_loss"
    if "val_loss" in fields:
        return "val_loss"
    if "g" in fields:
        return "g"
    return None


def plot_quality_comparison(
    experiments, output_dir, filename="quality_metrics_comparison.png",
    title="Image-quality metrics across experiments", x_column="epoch",
):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, fields, rows in experiments:
        columns = quality_columns(fields)
        if columns is None:
            continue
        ssim_column, psnr_column = columns
        selected_loss = loss_column(fields)
        x_values = [row[x_column] for row in rows]
        label = display_name(name)
        if selected_loss:
            axes[0].plot(x_values, [row[selected_loss] for row in rows], label=label, linewidth=1.4)
        axes[1].plot(x_values, [row[ssim_column] for row in rows], label=label, linewidth=1.4)
        axes[2].plot(x_values, [row[psnr_column] for row in rows], label=label, linewidth=1.4)

    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss value")
    axes[1].set_title("SSIM")
    axes[1].set_ylabel("SSIM")
    axes[2].set_title("PSNR")
    axes[2].set_ylabel("PSNR (dB)")
    for axis in axes:
        axis.set_xlabel(axis_label(x_column))
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    output_path = output_dir / filename
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def print_summary(experiments, x_column="epoch"):
    print(f"\nMetric summary (best {x_column} in each recorded run):")
    for name, fields, rows in experiments:
        columns = quality_columns(fields)
        if columns is None:
            continue
        ssim_column, psnr_column = columns
        best_ssim = max(rows, key=lambda row: row[ssim_column])
        best_psnr = max(rows, key=lambda row: row[psnr_column])
        print(
            f"  {name}: best {ssim_column.upper()}={best_ssim[ssim_column]:.4f} "
            f"({x_column} {best_ssim[x_column]:g}); best {psnr_column.upper()}="
            f"{best_psnr[psnr_column]:.2f} dB ({x_column} {best_psnr[x_column]:g})"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot all final training metric CSV files.")
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory containing experiment folders (default: final_results)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Plot output directory (default: <results-dir>/metric_curves)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "metric_curves").resolve()
    csv_paths = sorted(results_dir.glob("*/metrics/*.csv"))
    if not csv_paths:
        parser.error(f"No CSV files found under {results_dir}/*/metrics/")

    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    generated_paths = []
    for csv_path in csv_paths:
        fields, rows = read_metrics(csv_path)
        name = experiment_name(csv_path, results_dir)
        experiments.append((name, fields, rows))
        generated_paths.append(plot_experiment(name, fields, rows, output_dir))

    generated_paths.append(plot_quality_comparison(experiments, output_dir))
    print_summary(experiments)
    print("\nGenerated plots:")
    for path in generated_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
