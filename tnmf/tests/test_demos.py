"""
Test if all demos can be run without error, i.e.
   `python demo_selector.py demo_name` does not
yield a nonzero return code
"""

import sys
import subprocess  # noqa: S404
import logging
import pytest
from demos.demo_selector import DEMO_NAME_DICT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

DEMOS = list(DEMO_NAME_DICT.keys())


@pytest.mark.parametrize('demo_name', DEMOS)
def test_demo(demo_name: str):

    logger = logging.getLogger(demo_name)
    logger.setLevel(logging.INFO)

    with subprocess.Popen([sys.executable,  # noqa: S603
                           #  TODO: '-m', 'coverage', 'run',
                           'demos/demo_selector.py', demo_name],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as demo_run:
        stdout, stderr = demo_run.communicate()
        for stream in [stdout, stderr]:
            if len(stream) > 0:
                for line in stream.decode().split('\n'):
                    logger.info(line)

        assert demo_run.returncode == 0
