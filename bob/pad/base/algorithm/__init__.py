from .Algorithm import Algorithm
from .SVM import SVM
from .OneClassGMM import OneClassGMM
from .LogRegr import LogRegr
from .SVMCascadePCA import SVMCascadePCA
from .Predictions import Predictions

from .MLP import MLP

# to fix sphinx warnings of not able to find classes, when path is shortened
def __appropriate__(*args):
    """Says object was actually declared here, and not in the import module.
    Fixing sphinx warnings of not being able to find classes, when path is
    shortened.

    Parameters
    ----------
    *args
        The objects that you want sphinx to believe that are defined here.

    Resolves `Sphinx referencing issues <https//github.com/sphinx-
    doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    Algorithm,
    SVM,
    OneClassGMM,
    LogRegr,
    SVMCascadePCA,
    Predictions,
)

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
