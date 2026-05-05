import segmentation_models_pytorch as smp

def get_unet(encoder_name="resnet34", encoder_weights=None, in_channels=4 ,num_classes=3):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )