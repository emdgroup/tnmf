from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tnmf")
except PackageNotFoundError:
    pass
