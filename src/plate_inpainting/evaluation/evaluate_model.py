from argparse import ArgumentParser
from pathlib import Path
import torch
from utils.tools import get_config
from models.unet import get_unet
import tqdm
from evaluation.image_metrics import compute_image_metrics
from evaluation.ocr_metrics import compute_ocr_metrics

"""
This script expects existing grount truth images, masks, masked images and predictions all placed in
the test dir in subdirs names as images/; masks/; masked_images/; predictions/.
"""


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

parser = ArgumentParser(description="Train a UNet model for license plate inpainting")
parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / 'config' / 'unet.yaml'),help="Path to a config file.")
parser.add_argument("--test_data_path", type=str, help="Path to the test dataset for evaluation.")

args = parser.parse_args()

def evaluate_unet():
    config = get_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet(encoder_name=config["encoder_name"], encoder_weights=config["encoder_weights"], in_channels=config["in_channels"], num_classes=config["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    img_names = []
    truths = {}
    masks = {}
    masked_images = {}

    for img_file in tqdm.tqdm((args.test_data_path / "images").iterdir(), desc="Loading images"):
        img_names.append(img_file.stem)
        truths[img_file] = torch.load(args.test_data_path / "truths" / img_file).to(device)

    gt_path = args.test_data_path / "images"
    mask_path = args.test_data_path / "masks"
    masked_images_path = args.test_data_path / "masked_images"
    for img_file in tqdm.tqdm(gt_path, desc="Loading masked images"):
        try:
            masks[img_file] = torch.load(mask_path / img_file).to(device)
            masked_images[img_file] = torch.load(masked_images_path / img_file).to(device)
        except FileNotFoundError:
            print(f"File not found: {img_file}")

    predictions = {}
    for pred_file in tqdm.tqdm((args.test_data_path / "predictions").iterdir(), desc="Loading predictions"):
        try:
            predictions[img_file] = (torch.load(args.test_data_path / "predictions" / pred_file).to(device))
        except FileNotFoundError:
            print(f"File not found: {pred_file}")
    
    ssim_value, psnr_value = compute_image_metrics(torch.cat(list(predictions.values())), torch.cat(list(truths.values())))
    ocr_value_truths = compute_ocr_metrics(torch.cat(list(truths.values())), torch.cat(img_names))
    ocr_value_preds = compute_ocr_metrics(torch.cat(list(predictions.values())), torch.cat(img_names))
    ocr_value_masked = compute_ocr_metrics(torch.cat(list(masked_images.values())), torch.cat(img_names))

    print(f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB")
    print(f"OCR (truths): {ocr_value_truths:.2f}")
    print(f"OCR (preds): {ocr_value_preds:.2f}")
    print(f"OCR (masked): {ocr_value_masked:.2f}")

if __name__ == "__main__":
    evaluate_unet()








