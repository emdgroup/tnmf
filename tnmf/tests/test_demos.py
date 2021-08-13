"""
Test if all demos can be run without error, i.e. `python demo_selector.py <demo_name>` yields a zero return code.
"""

import sys
import os
import subprocess  # noqa: S404
from tempfile import mkstemp
import logging
import pytest
from demos.demo_selector import DEMO_NAME_DICT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

DEMOS = list(DEMO_NAME_DICT.keys())

DEMO_FRAME = """
# Work around a ModuleNotFoundError when running the demo inside coverage.py
import sys
sys.path[0] = '{}'
# Finally run the selected demo
from demos.demo_selector import main
main('{}')
"""


@pytest.mark.parametrize('demo_name', DEMOS)
def test_demo(demo_name: str):

    logger = logging.getLogger(demo_name)
    logger.setLevel(logging.INFO)

    fd, path = mkstemp(suffix='.py')
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(DEMO_FRAME.format(os.path.abspath('./demos'), demo_name))

        with subprocess.Popen([sys.executable, '-m', 'coverage', 'run', path],  # noqa: S603
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
