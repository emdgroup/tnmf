# pylint: skip-file
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # for Python<3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("tnmf")
except PackageNotFoundError:
    pass

del version
del PackageNotFoundError
