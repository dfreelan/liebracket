r"""Worked examples for the classical algebras.

Each function returns a multi-line report string (used by ``liebracket demo``)
and doubles as executable documentation of the API.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, List

from .bch import bch_lie_terms, bch_matrices, exp_matrix, format_lie_terms
from .cartan import cartan_matrix, dynkin_diagram
from .chevalley import simply_laced_algebra
from .classical import heisenberg, sl, so3, su2, su3
from .linalg import Matrix
from .representations import adjoint_representation, casimir_eigenspaces, sl2_irrep, su2_spin
from .roots import root_system
from .scalars import format_scalar
from .weights import WeightSpace

__all__ = ["DEMOS", "run_demo"]


def demo_su2_spins() -> str:
    r""":math:`\mathfrak{su}(2)` spin-:math:`j` modules: homomorphism check and :math:`\sum_k\rho(e_k)^2 = -j(j+1)`."""
    g = su2()
    lines = ["su(2) spin representations  ([e_i, e_j] = eps_ijk e_k)"]
    for twice_j in range(0, 5):
        j = Fraction(twice_j, 2)
        rho = su2_spin(j, g)
        cas = sum((m @ m for m in rho.matrices[1:]), rho.matrices[0] @ rho.matrices[0])
        lines.append(
            "  j = %-4s dim %d  homomorphism: %s  irreducible: %s  sum rho(e_k)^2 = %s * 1"
            % (j, rho.dim, rho.is_homomorphism(), rho.is_irreducible(), format_scalar(cas[0, 0]))
        )
    return "\n".join(lines)


def demo_su2_so3() -> str:
    r"""The isomorphism :math:`\mathfrak{su}(2)\cong\mathfrak{so}(3)` and the shared Killing form :math:`\kappa = -2\,\mathbb 1`."""
    a, b = su2(), so3()
    ok = a.is_isomorphism(b, Matrix.identity(3))
    return "\n".join(
        [
            "su(2) and so(3): identical structure constants eps_ijk",
            "  e_k -> L_k is a Lie algebra isomorphism: %s" % ok,
            "  Killing form of su(2):\n" + _indent(a.killing_form().pretty(), 4),
            "  Killing form of so(3):\n" + _indent(b.killing_form().pretty(), 4),
        ]
    )


def demo_su3() -> str:
    r""":math:`\mathfrak{su}(3)`: Jacobi identity, Killing form, adjoint Casimir."""
    g = su3()
    ad = adjoint_representation(g)
    lines = [
        "su(3) with T_a = -(i/2) lambda_a, T_8 = -(i/2) sqrt(3) lambda_8",
        "  Jacobi identity holds: %s" % g.satisfies_jacobi(),
        "  adjoint is a homomorphism: %s" % ad.is_homomorphism(),
        "  Killing form diagonal: %s" % [format_scalar(x) for x in g.killing_form().diagonal()],
        "  Casimir in the adjoint (Killing-normalised) = identity: %s" % (g.casimir_in_adjoint() == Matrix.identity(8)),
        "  brackets:",
    ]
    lines += ["    " + s for s in g.bracket_table()]
    return "\n".join(lines)


def demo_sl2_clebsch_gordan() -> str:
    r"""Clebsch--Gordan for :math:`\mathfrak{sl}_2`: :math:`V_1\otimes V_2 \cong V_3 \oplus V_1`, read off from Casimir eigenspaces."""
    lines = ["sl(2) tensor products decomposed by Casimir eigenvalue n(n+2)/8 (Killing form)"]
    for a, b in ((1, 1), (1, 2), (2, 2)):
        rep = sl2_irrep(a).tensor(sl2_irrep(b))
        spaces = casimir_eigenspaces(rep)
        parts = []
        for c, mult in sorted(spaces.items(), reverse=True):
            n = _sl2_highest_from_casimir(c)
            parts.append("V_%d" % n + ("" if mult == n + 1 else " x%d" % (mult // (n + 1))))
        lines.append("  V_%d (x) V_%d = %s" % (a, b, " + ".join(parts)))
    return "\n".join(lines)


def _sl2_highest_from_casimir(c: Fraction) -> int:
    n = 0
    while Fraction(n * (n + 2), 8) < c:
        n += 1
    return n


def demo_root_systems() -> str:
    """Summary table for a sample of root systems."""
    lines = ["type  rank  |Phi+|  dim g  h    |W|          exponents"]
    for spec in ("A2", "A3", "B2", "B3", "C3", "D4", "G2", "F4", "E6", "E7", "E8"):
        rs = root_system(spec)
        lines.append(
            "%-5s %-5d %-7d %-6d %-4d %-12d %s"
            % (spec, rs.rank, rs.num_positive_roots, rs.lie_algebra_dimension(), rs.coxeter_number(),
               rs.weyl_group_order(), rs.exponents())
        )
    return "\n".join(lines)


def demo_g2() -> str:
    r""":math:`G_2`: Cartan matrix, Dynkin diagram, roots, and its two fundamental modules (7 and 14)."""
    rs = root_system("G2")
    W = WeightSpace(rs)
    long_, short = rs.long_and_short()
    lines = [
        "G2 Cartan matrix:",
        _indent(cartan_matrix("G2").pretty(), 2),
        "Dynkin diagram:",
        _indent(dynkin_diagram(cartan_matrix("G2")), 2),
        "positive roots (simple-root coords): %s" % rs.positive_roots,
        "  long: %s   short: %s" % (long_, short),
        "dim V(1,0) = %d, dim V(0,1) = %d (adjoint)" % (W.weyl_dimension((1, 0)), W.weyl_dimension((0, 1))),
        "V(1,0) x V(1,0) = %s" % W.format_decomposition(W.tensor_product((1, 0), (1, 0))),
    ]
    return "\n".join(lines)


def demo_sl3_weights() -> str:
    r"""Weights of :math:`\mathfrak{sl}_3` modules two ways: diagonal Cartan action vs. Freudenthal."""
    g = sl(3)
    ad = adjoint_representation(g)
    W = WeightSpace.of_type("A2")
    direct = ad.weights([0, 1])
    freud = sorted((mu for mu, m in W.character((1, 1)).items() for _ in range(m)), reverse=True)
    lines = [
        "sl(3) adjoint weights from [H_i, .] eigenvalues: %s" % direct,
        "Freudenthal weights of V(1,1):                 %s" % freud,
        "agree: %s" % (direct == freud),
        "3 x 3bar = %s" % W.format_decomposition(W.tensor_product((1, 0), (0, 1))),
        "3 x 3    = %s" % W.format_decomposition(W.tensor_product((1, 0), (1, 0))),
        "8 x 8    = %s" % W.format_decomposition(W.tensor_product((1, 1), (1, 1))),
    ]
    return "\n".join(lines)


def demo_bch_heisenberg() -> str:
    r"""BCH is exact on the Heisenberg group: :math:`e^Ae^B = e^{A+B+\frac12[A,B]}`."""
    h = heisenberg(1)
    assert h.matrices is not None
    A = h.matrices[0] * 2 + h.matrices[2] * Fraction(1, 3)
    B = h.matrices[1] * (-5)
    Z = bch_matrices(A, B, 2)
    lines = [
        "BCH series to degree 4: " + format_lie_terms(bch_lie_terms(4)),
        "Heisenberg: A = 2x + z/3, B = -5p",
        "  exp(A) exp(B) == exp(BCH_2(A,B)): %s" % (exp_matrix(A) @ exp_matrix(B) == exp_matrix(Z)),
        "  BCH_2(A,B) =\n" + _indent(Z.pretty(), 4),
    ]
    return "\n".join(lines)


def demo_e_series() -> str:
    """Build :math:`\\mathfrak g(D_4)` from its roots and check Jacobi, Killing non-degeneracy and the adjoint Casimir."""
    rs = root_system("D4")
    g = simply_laced_algebra(rs)
    return "\n".join(
        [
            "g(D4) from the root lattice sign cocycle: dim %d" % g.dim,
            "  Jacobi identity verified on all basis triples: %s" % g.satisfies_jacobi(),
            "  semisimple (det Killing != 0): %s" % g.is_semisimple(),
            "  Killing form on h: %s" % [format_scalar(x) for x in g.killing_form().diagonal()[:4]],
        ]
    )


def _indent(text: str, n: int) -> str:
    return "\n".join(" " * n + line for line in text.splitlines())


DEMOS: Dict[str, Callable[[], str]] = {
    "su2": demo_su2_spins,
    "su2-so3": demo_su2_so3,
    "su3": demo_su3,
    "sl2-cg": demo_sl2_clebsch_gordan,
    "roots": demo_root_systems,
    "g2": demo_g2,
    "sl3-weights": demo_sl3_weights,
    "bch": demo_bch_heisenberg,
    "d4": demo_e_series,
}


def run_demo(name: str) -> str:
    if name not in DEMOS:
        raise KeyError("unknown demo %r; choose from %s" % (name, ", ".join(DEMOS)))
    return DEMOS[name]()


def all_demos() -> List[str]:
    return [DEMOS[k]() for k in DEMOS]
