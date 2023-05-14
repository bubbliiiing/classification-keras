from .mobilenetv1 import MobileNetV1
from .mobilenetv2 import MobileNetV2
from .swin_transformer import swin_transformer_tiny, swin_transformer_base, swin_transformer_small
from .resnet import ResNet50
from .vgg import VGG16
from .vision_transformer import VisionTransformer

get_model_from_name = {
    "mobilenetv1"   : MobileNetV1,
    "mobilenetv2"   : MobileNetV2,
    "resnet50"      : ResNet50,
    "vgg16"         : VGG16,
    "vit_b_16"      : VisionTransformer,
    "swin_transformer_tiny"     : swin_transformer_tiny,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_base"     : swin_transformer_base
}

freeze_layers = {
    "mobilenetv1"   : 81,
    "mobilenetv2"   : 151,
    "resnet50"      : 173,
    "vgg16"         : 19,
    "vit_b_16"      : 130,
    "swin_transformer_tiny"     : 181,
    "swin_transformer_small"    : 350,
    "swin_transformer_base"     : 350
}
