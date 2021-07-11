import argparse
import os
from glob import glob
from pathlib import Path
from typing import List

DEMO_FILE = Path(__file__).absolute().parent.parent / 'demos' / 'demo_selector.py'
EXAMPLES_FOLDER = Path(__file__).absolute().parent.parent / 'examples'


def get_examples() -> List[Path]:
    """Returns a list containing the absolute file paths of all examples."""
    file_list = glob(str(EXAMPLES_FOLDER / '*.py'))
    examples = []
    for p in file_list:
        abs_file_path = Path(p)
        if abs_file_path.name != '__init__.py':
            examples.append(abs_file_path)
    return examples


def main():
    """Entry point for the `tnmf` command."""

    # create parser for the `tnmf` command
    parser = argparse.ArgumentParser()

    # create subparsers for the `example` and `demo` commands
    subparsers = parser.add_subparsers(dest='command')
    demo_parser = subparsers.add_parser("demo", help='Runs the streamlit demo.')  # noqa: F841
    example_parser = subparsers.add_parser("example", help='Runs a specific example. Type `tnmf example -h` to show the list '
                                                           'of available examples.')

    examples = get_examples()
    example_parser.add_argument("example_name", choices=[e.stem for e in examples], help='Name of the example.')

    # parse the command type
    args = parser.parse_args()

    # parse the example name and run the corresponding script
    if args.command == 'example':
        args_example = parser.parse_args()
        example = [e for e in examples if e.stem == args_example.example_name][0]
        os.system(f'python {example}')  # noqa: S605

    # run the demo
    elif args.command == 'demo':
        os.system(f'streamlit run {DEMO_FILE}')  # noqa: S605
