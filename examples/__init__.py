# TODO: remove path extension once everything has been packaged
import sys
from pathlib import Path

p = Path(__file__).absolute().parent.parent
sys.path.append(str(p))

del sys
del Path
