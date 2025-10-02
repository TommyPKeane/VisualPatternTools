import logging

import pytest

from visualpatterntools.lightning_modules.resnet_utils import (
    ResNetDepth,
    resnet10,
    resnet200,
    resnet_modules_map,
)


module_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "given_value, expected_good",
    (
        (10, True),
        (18, True),
        (34, True),
        (50, True),
        (101, True),
        (152, True),
        (200, True),
        (ResNetDepth.resnet10, True),
        (ResNetDepth.resnet18, True),
        (ResNetDepth.resnet34, True),
        (ResNetDepth.resnet50, True),
        (ResNetDepth.resnet101, True),
        (ResNetDepth.resnet152, True),
        (ResNetDepth.resnet200, True),
        (201, False),
        ("200", False),
    ),
)
def test_ResNetDepth(given_value: int, expected_good: bool) -> None:
    pytest.helpers.simple_enum_validation(
        given_value=given_value,
        enum_class=ResNetDepth,
        expected_good=expected_good,
    )
    return None


def test_resnet10() -> None:
    _ = resnet10()
    return None


def test_resnet200() -> None:
    _ = resnet200()
    return None
