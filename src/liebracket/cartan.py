r"""Cartan matrices and Dynkin diagrams of the finite-type root systems.

Conventions (Bourbaki numbering) — the Cartan matrix is

.. math::

    A_{ij} = \langle \alpha_i^\vee, \alpha_j \rangle = \frac{2(\alpha_i, \alpha_j)}{(\alpha_i, \alpha_i)} ,

so :math:`A_{ii} = 2`, :math:`A_{ij} \le 0` for :math:`i \ne j`, and row :math:`i`
records how the simple reflection :math:`s_i` acts.  Matrices for every type are
*derived* from explicit Euclidean simple roots (see :mod:`liebracket.roots`)
rather than typed in by hand.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

from .linalg import Matrix

__all__ = [
    "parse_type",
    "cartan_matrix",
    "cartan_from_simple_roots",
    "symmetrizer",
    "symmetrized_cartan",
    "is_cartan_matrix",
    "is_finite_type",
    "dynkin_edges",
    "dynkin_diagram",
    "CARTAN_DETERMINANTS",
]


def parse_type(spec: str) -> Tuple[str, int]:
    """Parse ``'A3'``, ``'b_4'``, ``'G2'`` -> ``('A', 3)`` etc., validating the rank."""
    m = re.fullmatch(r"\s*([A-Ga-g])_?(\d+)\s*", spec)
    if not m:
        raise ValueError("cannot parse root system type %r" % spec)
    t, n = m.group(1).upper(), int(m.group(2))
    valid = {
        "A": n >= 1,
        "B": n >= 2,
        "C": n >= 2,
        "D": n >= 3,
        "E": n in (6, 7, 8),
        "F": n == 4,
        "G": n == 2,
    }
    if not valid[t]:
        raise ValueError("no finite root system of type %s%d" % (t, n))
    return t, n


def _dot(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(u, v)), Fraction(0))


def cartan_from_simple_roots(simple: Sequence[Sequence[Fraction]]) -> Matrix:
    r""":math:`A_{ij} = 2(\alpha_i,\alpha_j)/(\alpha_i,\alpha_i)` for Euclidean vectors :math:`\alpha_i`."""
    n = len(simple)
    rows = []
    for i in range(n):
        ii = _dot(simple[i], simple[i])
        rows.append([2 * _dot(simple[i], simple[j]) / ii for j in range(n)])
    return Matrix(rows)


def cartan_matrix(spec: str) -> Matrix:
    """Cartan matrix of a finite type, e.g. ``cartan_matrix('G2')``."""
    from .roots import euclidean_simple_roots

    return cartan_from_simple_roots(euclidean_simple_roots(spec))


def symmetrizer(A: Matrix) -> List[Fraction]:
    r"""Positive :math:`d_i` with :math:`d_i A_{ij} = d_j A_{ji}`, normalised so :math:`\min d_i = 1`.

    Then :math:`(\alpha_i, \alpha_j) = d_i A_{ij}` is a symmetric bilinear form and
    :math:`d_i = (\alpha_i,\alpha_i)/2` (short roots have length squared 2).
    Raises ``ValueError`` if :math:`A` is not symmetrisable.
    """
    n = A.nrows
    d: List = [None] * n
    for start in range(n):
        if d[start] is not None:
            continue
        d[start] = Fraction(1)
        stack = [start]
        while stack:
            i = stack.pop()
            for j in range(n):
                if j == i or A[i, j] == 0:
                    continue
                if A[j, i] == 0:
                    raise ValueError("A is not symmetrisable (zero pattern)")
                val = d[i] * Fraction(A[i, j]) / Fraction(A[j, i])
                if d[j] is None:
                    d[j] = val
                    stack.append(j)
                elif d[j] != val:
                    raise ValueError("A is not symmetrisable")
    # normalise each connected component separately is unnecessary for orthogonality,
    # but we normalise globally so that the shortest simple root has d = 1.
    m = min(d)
    return [x / m for x in d]


def symmetrized_cartan(A: Matrix) -> Matrix:
    r"""Gram matrix :math:`G_{ij} = (\alpha_i, \alpha_j) = d_i A_{ij}`."""
    d = symmetrizer(A)
    return Matrix.from_function(A.nrows, A.ncols, lambda i, j: d[i] * A[i, j])


def is_cartan_matrix(A: Matrix) -> bool:
    r"""Generalised Cartan matrix axioms: :math:`A_{ii}=2`, :math:`A_{ij}\in\mathbb Z_{\le 0}`, :math:`A_{ij}=0 \iff A_{ji}=0`."""
    if not A.is_square:
        return False
    n = A.nrows
    for i in range(n):
        if A[i, i] != 2:
            return False
        for j in range(n):
            if i == j:
                continue
            x = Fraction(A[i, j])
            if x > 0 or x.denominator != 1:
                return False
            if (A[i, j] == 0) != (A[j, i] == 0):
                return False
    return True


def is_finite_type(A: Matrix) -> bool:
    r"""A generalised Cartan matrix is of finite type iff its symmetrisation is positive definite
    (checked with Sylvester's criterion on leading principal minors)."""
    if not is_cartan_matrix(A):
        return False
    try:
        G = symmetrized_cartan(A)
    except ValueError:
        return False
    n = G.nrows
    for k in range(1, n + 1):
        minor = Matrix([[G[i, j] for j in range(k)] for i in range(k)])
        if minor.det() <= 0:
            return False
    return True


#: :math:`\det A` for each finite type, equal to the index :math:`[P : Q]` of the root lattice in the weight lattice.
CARTAN_DETERMINANTS: Dict[str, object] = {
    "A": lambda n: n + 1,
    "B": lambda n: 2,
    "C": lambda n: 2,
    "D": lambda n: 4,
    "E": lambda n: 9 - n,
    "F": lambda n: 1,
    "G": lambda n: 1,
}


def dynkin_edges(A: Matrix) -> List[Tuple[int, int, int, int]]:
    r"""Edges :math:`(i, j, m, \text{arrow})` of the Dynkin diagram.

    :math:`m = A_{ij}A_{ji}` is the bond multiplicity; ``arrow`` is ``+1`` if the arrow points
    from :math:`i` to :math:`j` (:math:`\alpha_i` longer), ``-1`` for the reverse, ``0`` if simply laced.
    """
    edges = []
    n = A.nrows
    for i in range(n):
        for j in range(i + 1, n):
            m = A[i, j] * A[j, i]
            if m:
                arrow = 0
                if abs(A[i, j]) < abs(A[j, i]):
                    arrow = 1
                elif abs(A[i, j]) > abs(A[j, i]):
                    arrow = -1
                edges.append((i, j, int(m), arrow))
    return edges


def dynkin_diagram(A: Matrix) -> str:
    r"""ASCII Dynkin diagram.

    Chains are drawn on one line (``o---o==>o``); nodes off the main chain
    (as in :math:`D_n` and :math:`E_n`) are attached on a second line.  Node labels are 1-based.
    """
    n = A.nrows
    edges = dynkin_edges(A)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    info: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i, j, m, arrow in edges:
        adj[i].append(j)
        adj[j].append(i)
        info[(i, j)] = (m, arrow)
        info[(j, i)] = (m, -arrow)
    # longest path in the tree = main chain
    def farthest(src: int) -> Tuple[int, Dict[int, int]]:
        parent = {src: -1}
        order = [src]
        for v in order:
            for w in adj[v]:
                if w not in parent:
                    parent[w] = v
                    order.append(w)
        return order[-1], parent

    a, _ = farthest(0)
    b, parent = farthest(a)
    chain = [b]
    while parent[chain[-1]] != -1:
        chain.append(parent[chain[-1]])
    if chain[0] > chain[-1]:
        chain.reverse()
    bonds = {1: "---", 2: "==", 3: "≡≡"}
    line, pos = "", {}
    for k, v in enumerate(chain):
        pos[v] = len(line)
        line += "o"
        if k + 1 < len(chain):
            m, arrow = info[(v, chain[k + 1])]
            if m == 1:
                line += "---"
            else:
                line += bonds[m] + (">" if arrow == 1 else "<")
    labels = [" "] * (len(line) + 2)
    for v, p in pos.items():
        for t, ch in enumerate(str(v + 1)):
            if p + t < len(labels):
                labels[p + t] = ch
    out = [line, "".join(labels).rstrip()]
    extra = [v for v in range(n) if v not in pos]
    for v in extra:
        u = next(w for w in adj[v] if w in pos)
        out.append(" " * pos[u] + "|")
        out.append(" " * pos[u] + "o %d" % (v + 1))
    return "\n".join(out)
