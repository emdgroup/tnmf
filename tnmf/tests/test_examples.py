"""
Test if all examples can be run without error, i.e.
   `python example.py` does not yield a nonzero return code
"""

import sys
import subprocess  # noqa: S404
import pkgutil
import pytest


EXAMPLES = [name for _, name, _ in pkgutil.iter_modules(['examples'])]


EXAMPLE_FRAME = """
import matplotlib.pyplot as plt
plt.ion()
from examples import {}
"""


@pytest.mark.parametrize('example', EXAMPLES)
def test_example(example: str):
    example_run = subprocess.run([sys.executable,  # noqa: S603
                                  "-c", EXAMPLE_FRAME.format(example)],
                                 capture_output=True,
                                 check=False)

    assert example_run.returncode == 0
