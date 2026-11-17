r"""Constructing a simple Lie algebra from its root system.

For a simply-laced root system (types :math:`A`, :math:`D`, :math:`E`) the algebra
can be written down directly from the root lattice :math:`Q` using a sign cocycle
:math:`\varepsilon : Q \times Q \to \{\pm 1\}` (Frenkel--Kac / Kac §7.8):
:math:`\varepsilon` is bimultiplicative with
:math:`\varepsilon(\alpha,\alpha) = (-1)^{(\alpha,\alpha)/2}`, and

.. math::

    [h, h'] = 0,\qquad [h, E_\alpha] = (h, \alpha)\,E_\alpha,\qquad
    [E_\alpha, E_{-\alpha}] = -\alpha,\qquad
    [E_\alpha, E_\beta] = \begin{cases}\varepsilon(\alpha,\beta)E_{\alpha+\beta} & \alpha+\beta\in\Phi\\ 0 & \text{otherwise.}\end{cases}

On simple roots we take :math:`\varepsilon(\alpha_i,\alpha_j) = -1` if :math:`i = j` or
(:math:`i < j` and :math:`A_{ij} = -1`), and :math:`+1` otherwise.  Every structure
constant is an integer; the Jacobi identity is *verified*, not assumed.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .algebra import LieAlgebra
from .roots import RootSystem

__all__ = ["simply_laced_algebra", "sign_cocycle"]


def sign_cocycle(rs: RootSystem) -> "callable":  # type: ignore[valid-type]
    r"""The bimultiplicative cocycle :math:`\varepsilon(\alpha,\beta) = \prod_{i,j}\varepsilon_{ij}^{a_i b_j}`."""
    r = rs.rank
    neg = [[(i == j) or (i < j and rs.cartan[i, j] == -1) for j in range(r)] for i in range(r)]

    def eps(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        parity = 0
        for i in range(r):
            if a[i] % 2 == 0:
                continue
            for j in range(r):
                if neg[i][j] and b[j] % 2:
                    parity ^= 1
        return -1 if parity else 1

    return eps


def simply_laced_algebra(rs: RootSystem, check: bool = True) -> LieAlgebra:
    r"""The split simple Lie algebra of a simply-laced root system.

    Basis order: :math:`h_1,\dots,h_r` (identified with :math:`\alpha_i`), then
    :math:`E_\alpha` for positive roots, then :math:`E_{-\alpha}`.
    """
    if not rs.is_simply_laced():
        raise ValueError("the sign-cocycle construction needs a simply-laced root system")
    r = rs.rank
    roots: List[Tuple[int, ...]] = rs.roots
    index: Dict[Tuple[int, ...], int] = {beta: r + k for k, beta in enumerate(roots)}
    eps = sign_cocycle(rs)
    br: Dict[Tuple[int, int], Dict[int, int]] = {}
    # [h_i, E_beta] = (alpha_i, beta) E_beta
    for i in range(r):
        alpha_i = tuple(1 if k == i else 0 for k in range(r))
        for beta in roots:
            c = int(rs.inner(alpha_i, beta))
            if c:
                br[(i, index[beta])] = {index[beta]: c}
    for a_pos, alpha in enumerate(roots):
        for beta in roots[a_pos + 1:]:
            s = tuple(x + y for x, y in zip(alpha, beta))
            key = (index[alpha], index[beta])
            if not any(s):
                br[key] = {i: -alpha[i] for i in range(r) if alpha[i]}
            elif s in index:
                br[key] = {index[s]: eps(alpha, beta)}
    names = ["h%d" % (i + 1) for i in range(r)]
    for beta in roots:
        sign = "-" if sum(beta) < 0 else ""
        names.append("E%s[%s]" % (sign, "".join(str(abs(b)) for b in beta)))
    return LieAlgebra(r + len(roots), br, names=names, name="g(%s)" % (rs.label or "?"), check=check)
