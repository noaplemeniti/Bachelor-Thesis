from argparse import ArgumentParser
from pathlib import Path
import sys

from PIL import Image
import torch
from torchvision import transforms
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PACKAGE_ROOT_DIR = PROJECT_ROOT / "src" / "plate_inpainting"
if str(PACKAGE_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_DIR))

from evaluation.image_metrics import compute_image_metrics


parser = ArgumentParser(description="Evaluate GAN predictions for license plate inpainting")
parser.add_argument("--test_data_path", type=Path, default=PROJECT_ROOT / "data" / "license_plates_synth" / "test", help="Path to the test dataset for evaluation.")
parser.add_argument("--predictions_dir_name", type=str, default="predictions", help="Name of the predictions directory inside test_data_path.")
parser.add_argument("--ocr", action="store_true", help="Also compute OCR character error rate using image filename stems as labels.")


def load_rgb_tensor(path, to_tensor):
    return to_tensor(Image.open(path).convert("RGB"))


def evaluate_gan():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    gt_dir = args.test_data_path / "images"
    masked_image_dir = args.test_data_path / "masked_images"
    prediction_dir = args.test_data_path / args.predictions_dir_name

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth image directory not found: {gt_dir}")
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    prediction_paths = sorted(
        p for p in prediction_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not prediction_paths:
        raise ValueError(f"No prediction images found in: {prediction_dir}")

    predictions = []
    truths = []
    masked_images = []
    masked_truths = []
    masked_labels = []
    labels = []

    for pred_path in tqdm.tqdm(prediction_paths, desc="Loading GAN predictions"):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"Missing ground truth for {pred_path.name}")
            continue

        predictions.append(load_rgb_tensor(pred_path, to_tensor))
        truths.append(load_rgb_tensor(gt_path, to_tensor))
        labels.append(pred_path.stem)

        masked_path = masked_image_dir / pred_path.name
        if masked_path.exists():
            masked_images.append(load_rgb_tensor(masked_path, to_tensor))
            masked_truths.append(load_rgb_tensor(gt_path, to_tensor))
            masked_labels.append(pred_path.stem)

    if not predictions:
        raise ValueError("No predictions with matching ground-truth images were found.")

    predictions = torch.stack(predictions).to(device)
    truths = torch.stack(truths).to(device)

    psnr_value, ssim_value = compute_image_metrics(predictions, truths)
    print(f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB")

    if masked_images:
        masked_images = torch.stack(masked_images).to(device)
        masked_truths = torch.stack(masked_truths).to(device)
        masked_psnr, masked_ssim = compute_image_metrics(masked_images, masked_truths)
        print(f"Masked SSIM: {masked_ssim:.4f}, Masked PSNR: {masked_psnr:.2f} dB")
    else:
        masked_truths = None

    if args.ocr:
        from evaluation.ocr_metrics import compute_ocr_metrics

        ocr_value_truths = compute_ocr_metrics(truths.cpu(), labels)
        ocr_value_preds = compute_ocr_metrics(predictions.cpu(), labels)
        print(f"OCR (truths): {ocr_value_truths:.2f}")
        print(f"OCR (preds): {ocr_value_preds:.2f}")
        if masked_truths is not None:
            ocr_value_masked = compute_ocr_metrics(masked_images.cpu(), masked_labels)
            print(f"OCR (masked): {ocr_value_masked:.2f}")


if __name__ == "__main__":
    evaluate_gan()
