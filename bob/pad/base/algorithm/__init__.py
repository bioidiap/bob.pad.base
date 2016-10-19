from .Algorithm import Algorithm

# to fix sphinx warnings of not able to find classes, when path is shortened
Algorithm.__module__ = "bob.pad.base.algorithm"
# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
