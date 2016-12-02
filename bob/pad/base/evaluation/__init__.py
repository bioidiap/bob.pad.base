from .PadIsoMetrics import PadIsoMetrics

# to fix sphinx warnings of not able to find classes, when path is shortened
PadIsoMetrics.__module__ = "bob.pad.base.evaluation"
# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]

