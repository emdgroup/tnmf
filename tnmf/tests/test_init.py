"""
Test if `from tnmf import *` does not fail
"""


_import_error = None

try:
    from tnmf import *
except Exception as e:
    _import_error = e


def test_import_tinmf():
    assert _import_error is None
