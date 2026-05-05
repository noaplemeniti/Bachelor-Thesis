from torchmetrics.text import CharErrorRate
from fast_plate_ocr import LicensePlateRecognizer

def normalize_plate(plate):
    return plate.replace(" ", "").upper().strip()

def compute_ocr_metrics(images, truths, recognizer = "cct-s-v2-global-model"):
    cer_metric = CharErrorRate()
    recognizer = LicensePlateRecognizer(model_name=recognizer)
    preds = []
    for img in images:
        pred = recognizer.recognize(img)
        preds.append(normalize_plate(pred))
    truths = [normalize_plate(t) for t in truths]
    cer = cer_metric(preds, truths)
    return cer.item()
    