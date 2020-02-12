import glob

from os.path import join, basename, isfile
from pathlib import Path

modules = glob.glob(join(Path(__file__).parent, "ps1_*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f)]
