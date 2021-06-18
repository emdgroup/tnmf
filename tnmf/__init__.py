# pylint: skip-file
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tnmf")
except PackageNotFoundError:
    pass

del version
del PackageNotFoundError
