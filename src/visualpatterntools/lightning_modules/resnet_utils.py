"""Utility Enums, Classes, and Methods for ResNet Neural Network Modules

References:
    - https://docs.pytorch.org/vision/main/models/resnet.html
    - https://github.com/kenshohara/video-classification-3d-cnn-pytorch
    - [He2015] "Deep Residual Learning for Image Recognition": https://arxiv.org/abs/1512.03385
"""

import enum
import logging

import torch
import torchvision


module_logger = logging.getLogger(__name__)


class ResNetDepth(enum.IntEnum):
    """Integer-Enum for ResNet Implementations by their "depth" Characteristic

    This enum is meant to be used with the `resnet_modules_map` (`dict`) defined below,
    for selection of the class/callable that can be used to instantiate a new instance
    of a ResNet PyTorch Module.
    """

    resnet10: int = 10
    resnet18: int = 18
    resnet34: int = 34
    resnet50: int = 50
    resnet101: int = 101
    resnet152: int = 152
    resnet200: int = 200


def resnet10(**kwargs) -> torch.nn.Module:
    """Factory | Create a ResNet10 Instance

    ResNet10 is a ResNet implementation with a "BasicBlock" and the Layers:
        - Layer 1: (64, 1), stride=1
        - Layer 2: (128, 1), stride=2
        - Layer 3: (256, 1), stride=2
        - Layer 4: (512, 1), stride=2

    Args:
        **kwargs: Keyword Arguments to pass through to the `torchvision.models.ResNet`
            constructor

    Returns:
        torch.nn.Module: ResNet Neural Net Module instance
    """
    return torchvision.models.ResNet(
        torchvision.models.resnet.BasicBlock,
        [1, 1, 1, 1],
        **kwargs,
    )


def resnet200(**kwargs) -> torch.nn.Module:
    """Factory | Create a ResNet200 Instance

    ResNet10 is a ResNet implementation with a "Bottleneck" and the Layers:
        - Layer 1: (64, 3), stride=1
        - Layer 2: (128, 24), stride=2
        - Layer 3: (256, 36), stride=2
        - Layer 4: (512, 3), stride=2

    Args:
        **kwargs: Keyword Arguments to pass through to the `torchvision.models.ResNet`
            constructor

    Returns:
        torch.nn.Module: ResNet Neural Net Module instance
    """
    return torchvision.models.ResNet(
        torchvision.models.resnet.Bottleneck,
        [3, 24, 36, 3],
        **kwargs,
    )


resnet_modules_map: dict[ResNetDepth, torch.nn.Module] = {
    ResNetDepth.resnet10: resnet10,
    ResNetDepth.resnet18: torchvision.models.resnet18,
    ResNetDepth.resnet34: torchvision.models.resnet34,
    ResNetDepth.resnet50: torchvision.models.resnet50,
    ResNetDepth.resnet101: torchvision.models.resnet101,
    ResNetDepth.resnet152: torchvision.models.resnet152,
    ResNetDepth.resnet200: resnet200,
}
