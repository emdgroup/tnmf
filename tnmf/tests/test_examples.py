"""
Test if all examples can be run without error, i.e. `python <example>.py` yields a zero return code.
"""

import sys
import os
import subprocess  # noqa: S404
from tempfile import mkstemp
import pkgutil
import logging
import pytest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

EXAMPLES = [name for _, name, _ in pkgutil.iter_modules(['examples'])]

EXAMPLE_FRAME = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
from examples import {}
"""


@pytest.mark.parametrize('example', EXAMPLES)
def test_example(example: str):

    logger = logging.getLogger(example)
    logger.setLevel(logging.INFO)

    fd, path = mkstemp(suffix='.py')
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(EXAMPLE_FRAME.format(example))

        with subprocess.Popen([sys.executable, '-m', 'coverage', 'run', path],  # noqa: S603, DUO106
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as example_run:
            stdout, stderr = example_run.communicate()
            for stream in [stdout, stderr]:
                if len(stream) > 0:
                    for line in stream.decode().split('\n'):
                        logger.info(line)

            assert example_run.returncode == 0
    finally:
        os.remove(path)
