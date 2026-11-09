"""DeepLabV3+ on a torchvision ResNet, the network the GVI checkpoint holds.

Built from torchvision's ResNet, layer getter and ASPP, with the DeepLabV3+
decoder (Chen et al. 2018) on top, at output stride 16 (dilation in the last
ResNet stage, ASPP rates 6 / 12 / 18). Parameter names match the checkpoints
of the DeepLabV3Plus-Pytorch training code, so a checkpoint loads as is.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import ASPP

#: ResNet depth by the number of blocks in its third stage.
_DEPTHS = {6: "resnet50", 23: "resnet101", 36: "resnet152"}


class DeepLabV3Plus(nn.Module):
    """Backbone features ``low_level`` (stride 4) and ``out`` (stride 16), the
    ASPP on ``out`` upsampled onto the projected ``low_level``, a 3 × 3
    classifier, and the logits upsampled to the input size."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = IntermediateLayerGetter(
            backbone, return_layers={"layer1": "low_level", "layer4": "out"}
        )
        self.classifier = _Head(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.backbone(x))
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


class _Head(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )
        self.aspp = ASPP(2048, [6, 12, 18])
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, features: dict) -> torch.Tensor:
        low = self.project(features["low_level"])
        out = F.interpolate(
            self.aspp(features["out"]),
            size=low.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.classifier(torch.cat([low, out], dim=1))


def depth_of(state_dict: dict) -> str:
    """The ResNet a DeepLabV3+ checkpoint was trained on, from its blocks."""
    if "classifier.aspp.convs.0.0.weight" not in state_dict or not any(
        k.startswith("backbone.layer4.") for k in state_dict
    ):
        raise ValueError(
            "The checkpoint is not a DeepLabV3+ with a ResNet backbone; GVI "
            "segmentation supports ResNet-50 / 101 / 152 checkpoints."
        )
    blocks = {
        int(k.split(".")[2]) for k in state_dict if k.startswith("backbone.layer3.")
    }
    n = max(blocks) + 1
    if n not in _DEPTHS:
        raise ValueError(f"No ResNet has {n} blocks in its third stage.")
    return _DEPTHS[n]


def build(state_dict: dict, num_classes: int = 19) -> DeepLabV3Plus:
    """The network for ``state_dict``, untrained; load the weights next."""
    backbone = getattr(resnet, depth_of(state_dict))(
        weights=None, replace_stride_with_dilation=[False, False, True]
    )
    return DeepLabV3Plus(backbone, num_classes)
