from .database import PadDatabase
from .file import PadFile

# to fix sphinx warnings of not able to find classes, when path is shortened
PadDatabase.__module__ = "database"
PadFile.__module__ = "database"
# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
