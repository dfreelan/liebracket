r"""liebracket — exact Lie algebra and representation-theory numerics.

Pure-Python, standard-library only.  Structure constants, Killing forms,
representations, root systems, Cartan matrices, highest-weight theory and the
Baker--Campbell--Hausdorff series, computed over :math:`\mathbb Q` and
:math:`\mathbb Q(i)` without rounding.
"""
__version__ = "0.1.0"

from .algebra import LieAlgebra
from .bch import bch_in_algebra, bch_lie_terms, bch_matrices, exp_matrix, format_lie_terms
from .cartan import cartan_matrix, dynkin_diagram, is_finite_type, symmetrizer
from .chevalley import simply_laced_algebra
from .classical import borel, gl, heisenberg, nilpotent_upper, sl, sl2, so, so3, sp, su2, su3
from .linalg import Matrix, commutator
from .representations import (
    Representation,
    adjoint_representation,
    defining_representation,
    sl2_irrep,
    su2_spin,
)
from .roots import RootSystem, root_system
from .scalars import GaussianRational, I
from .weights import WeightSpace


# Stable fixture catalog (internal golden IDs)
from ._catalog import FIXTURE_NOTES, REFERENCE_URLS, fixture_ids, note_for, primary_reference

__all__ = ["FIXTURE_NOTES", "REFERENCE_URLS", "fixture_ids", "note_for", "primary_reference", 
    "__version__",
    "LieAlgebra", "Matrix", "commutator", "GaussianRational", "I",
    "sl2", "su2", "so3", "su3", "sl", "so", "sp", "gl", "heisenberg", "nilpotent_upper", "borel",
    "Representation", "adjoint_representation", "defining_representation", "sl2_irrep", "su2_spin",
    "RootSystem", "root_system", "cartan_matrix", "dynkin_diagram", "is_finite_type", "symmetrizer",
    "WeightSpace", "simply_laced_algebra",
    "bch_lie_terms", "bch_matrices", "bch_in_algebra", "exp_matrix", "format_lie_terms",
]
