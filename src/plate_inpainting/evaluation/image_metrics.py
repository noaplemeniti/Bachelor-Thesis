from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def compute_image_metrics(preds, targets):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    psnr = psnr_metric(preds, targets)
    ssim = ssim_metric(preds, targets)

    return psnr.item(), ssim.item()