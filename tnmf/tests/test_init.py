"""
Test if `from tnmf import *` does not fail
"""

# pylint: disable=broad-except, wildcard-import, unused-wildcard-import

_import_error = None

try:
    from tnmf import *
except Exception as e:
    _import_error = e


def test_import_tinmf():
    assert _import_error is None
