"""PyTest Configuration | Unit-Tests

Helper methods are decorated with the `@pytest.helpers.register` decorator through the
installation of the `pytest-helpers-namespace` PyPI Package.

The decorator allows those decorated methods to be accessed in any Python modules in any
subdirectories at or below this `conftest.py` module by prefacing the function alias
with `pytest.helper.`.

This pattern allows for common utility functions used for testing that you don't need
to include in the Package/Project code, and so that you don't need to copy-paste the
definition into multiple modules.

References:
    - https://pypi.org/project/pytest-helpers-namespace/
    - https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
"""

import enum
import logging

import pytest


module_logger = logging.getLogger(__name__)


@pytest.helpers.register
def simple_enum_validation(
    given_value: object,
    enum_class: enum.Enum,
    expected_good: bool,
) -> None:
    """PyTest Helper | Validate an Enum Class by conditionally checking given Values

    Python Enums all have a protected member `dict` that inverts the Enum relationship
    as the `_value2member_map_`. This is a `dict` where the keys are the values of each
    member in the enum.

    This helper function checks whether the `given_value` is one of the member-values of
    the Enum. If you want a positive check then you use `expected_good=True`, but if you
    want to make sure a value is _not_ defined in the Enum, then you would call this
    helper with `expected_good=False`.

    Args:
        given_value (object): Value to check
        enum_class (enum.Enum): Enum Class to check
        expected_good (bool): Should the check succeed?

    Returns:
        None: Nothing returned, assertions would be raised if the test check has failed.
    """
    enum_values = tuple(enum_class._value2member_map_.keys())

    if expected_good:
        assert given_value in enum_values
    else:
        assert given_value not in enum_values

    return None
