r"""Root systems of the finite-dimensional simple Lie algebras.

A :class:`RootSystem` is built from a Cartan matrix alone.  Positive roots are
generated in simple-root coordinates by the root-string algorithm: if
:math:`\beta - p\alpha_i, \dots, \beta + q\alpha_i` is the :math:`\alpha_i`-string
through :math:`\beta`, then

.. math::

    p - q = \langle \beta, \alpha_i^\vee \rangle = \sum_j \beta_j A_{ij},

so :math:`\beta + \alpha_i` is a root iff :math:`q = p - \langle\beta,\alpha_i^\vee\rangle > 0`.

Derived data: highest root, heights, exponents (via Kostant's height partition),
Coxeter number, Weyl group order :math:`|W| = \prod_i (m_i + 1)`, simple reflections,
and explicit Weyl group enumeration for small rank.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .cartan import cartan_from_simple_roots, parse_type, symmetrizer
from .linalg import Matrix

__all__ = ["RootSystem", "euclidean_simple_roots", "root_system"]

Root = Tuple[int, ...]
F = Fraction


def _e(n: int, *entries: Tuple[int, object]) -> List[Fraction]:
    v = [F(0)] * n
    for i, c in entries:
        v[i] = F(c)  # type: ignore[arg-type]
    return v


def euclidean_simple_roots(spec: str) -> List[List[Fraction]]:
    r"""Bourbaki's realisation of the simple roots in :math:`\mathbb{R}^m`.

    =====  ==============  =====================================================
    type   ambient         simple roots
    =====  ==============  =====================================================
    A_n    R^{n+1}         :math:`e_i - e_{i+1}`
    B_n    R^n             :math:`e_i - e_{i+1}`, :math:`e_n`
    C_n    R^n             :math:`e_i - e_{i+1}`, :math:`2e_n`
    D_n    R^n             :math:`e_i - e_{i+1}`, :math:`e_{n-1} + e_n`
    E_8    R^8             :math:`\tfrac12(e_1 - e_2 - \dots - e_7 + e_8)`, :math:`e_1+e_2`, :math:`e_{i-2}-e_{i-3}`
    E_6/7  R^8             first 6 / 7 roots of :math:`E_8`
    F_4    R^4             :math:`e_2-e_3,\ e_3-e_4,\ e_4,\ \tfrac12(e_1-e_2-e_3-e_4)`
    G_2    R^3             :math:`e_1 - e_2,\ -2e_1 + e_2 + e_3`
    =====  ==============  =====================================================
    """
    t, n = parse_type(spec)
    if t == "A":
        return [_e(n + 1, (i, 1), (i + 1, -1)) for i in range(n)]
    chain = [_e(n, (i, 1), (i + 1, -1)) for i in range(n - 1)]
    if t == "B":
        return chain + [_e(n, (n - 1, 1))]
    if t == "C":
        return chain + [_e(n, (n - 1, 2))]
    if t == "D":
        return chain + [_e(n, (n - 2, 1), (n - 1, 1))]
    if t == "E":
        h = F(1, 2)
        e8 = [[h, -h, -h, -h, -h, -h, -h, h], _e(8, (0, 1), (1, 1))]
        e8 += [_e(8, (k - 2, -1), (k - 1, 1)) for k in range(2, 8)]
        return e8[:n]
    if t == "F":
        h = F(1, 2)
        return [_e(4, (1, 1), (2, -1)), _e(4, (2, 1), (3, -1)), _e(4, (3, 1)), [h, -h, -h, -h]]
    return [_e(3, (0, 1), (1, -1)), _e(3, (0, -2), (1, 1), (2, 1))]


class RootSystem:
    r"""The root system :math:`\Phi` attached to a finite-type Cartan matrix :math:`A`.

    Roots are integer tuples of coordinates with respect to the simple roots
    :math:`\alpha_1,\dots,\alpha_r`.

    >>> rs = RootSystem.of_type("G2")
    >>> len(rs.positive_roots), rs.highest_root, rs.weyl_group_order()
    (6, (3, 2), 12)
    """

    def __init__(self, cartan: Matrix, label: str = "") -> None:
        if not cartan.is_square:
            raise ValueError("Cartan matrix must be square")
        self.cartan = cartan.map(lambda x: int(x))
        self.label = label
        self.rank = cartan.nrows
        self.d = symmetrizer(self.cartan)
        self.positive_roots: List[Root] = self._generate_positive_roots()
        self._positive_set: Set[Root] = set(self.positive_roots)

    @classmethod
    def of_type(cls, spec: str) -> "RootSystem":
        t, n = parse_type(spec)
        simple = euclidean_simple_roots(spec)
        rs = cls(cartan_from_simple_roots(simple), label="%s%d" % (t, n))
        rs.euclidean_simple = simple  # type: ignore[attr-defined]
        return rs

    # -- generation -------------------------------------------------------
    def pairing(self, beta: Sequence[int], i: int) -> int:
        r""":math:`\langle \beta, \alpha_i^\vee \rangle = \sum_j \beta_j A_{ij}`."""
        return sum(b * self.cartan[i, j] for j, b in enumerate(beta))

    def _generate_positive_roots(self, max_roots: int = 100000) -> List[Root]:
        r = self.rank
        simple = [tuple(1 if k == i else 0 for k in range(r)) for i in range(r)]
        roots: List[Root] = list(simple)
        known: Set[Root] = set(simple)
        layer = list(simple)
        while layer:
            nxt: List[Root] = []
            for beta in layer:
                for i in range(r):
                    # p = number of times alpha_i can be subtracted staying in Phi+
                    p = 0
                    cur = list(beta)
                    while True:
                        cur[i] -= 1
                        if tuple(cur) in known:
                            p += 1
                        else:
                            break
                    q = p - self.pairing(beta, i)
                    if q > 0:
                        new = tuple(b + (1 if k == i else 0) for k, b in enumerate(beta))
                        if new not in known:
                            known.add(new)
                            nxt.append(new)
                            roots.append(new)
                            if len(roots) > max_roots:
                                raise ValueError("Cartan matrix is not of finite type")
            layer = nxt
        return sorted(roots, key=lambda b: (sum(b), b))

    # -- basic data -------------------------------------------------------
    @property
    def simple_roots(self) -> List[Root]:
        return [tuple(1 if k == i else 0 for k in range(self.rank)) for i in range(self.rank)]

    @property
    def roots(self) -> List[Root]:
        """All roots :math:`\\Phi = \\Phi^+ \\sqcup -\\Phi^+`."""
        return self.positive_roots + [tuple(-b for b in beta) for beta in self.positive_roots]

    def is_root(self, beta: Sequence[int]) -> bool:
        t = tuple(beta)
        return t in self._positive_set or tuple(-b for b in t) in self._positive_set

    @property
    def num_positive_roots(self) -> int:
        return len(self.positive_roots)

    def lie_algebra_dimension(self) -> int:
        r""":math:`\dim\mathfrak g = r + |\Phi|`."""
        return self.rank + 2 * len(self.positive_roots)

    @staticmethod
    def height(beta: Sequence[int]) -> int:
        return sum(beta)

    @property
    def highest_root(self) -> Root:
        return max(self.positive_roots, key=lambda b: (sum(b), b))

    def coxeter_number(self) -> int:
        r""":math:`h = \operatorname{ht}(\theta) + 1`."""
        return self.height(self.highest_root) + 1

    def height_distribution(self) -> Dict[int, int]:
        return dict(sorted(Counter(sum(b) for b in self.positive_roots).items()))

    def exponents(self) -> List[int]:
        r"""Exponents :math:`m_1 \le \dots \le m_r`.

        By Kostant's theorem the partition of :math:`|\Phi^+|` by heights is dual to the
        partition formed by the exponents: :math:`\#\{i : m_i = k\} = r_k - r_{k+1}`, where
        :math:`r_k` counts positive roots of height :math:`k`.
        """
        dist = self.height_distribution()
        out: List[int] = []
        for k in range(1, max(dist) + 1):
            out += [k] * (dist.get(k, 0) - dist.get(k + 1, 0))
        return sorted(out)

    def weyl_group_order(self) -> int:
        r""":math:`|W| = \prod_i (m_i + 1)`."""
        out = 1
        for m in self.exponents():
            out *= m + 1
        return out

    # -- metric -----------------------------------------------------------
    def inner(self, a: Sequence[object], b: Sequence[object]) -> Fraction:
        r"""Invariant form in simple-root coordinates: :math:`(\alpha,\beta) = \sum_{ij} a_i b_j d_i A_{ij}`."""
        total = Fraction(0)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    if bj:
                        total += Fraction(ai) * Fraction(bj) * self.d[i] * self.cartan[i, j]
        return total

    def norm2(self, a: Sequence[object]) -> Fraction:
        return self.inner(a, a)

    def gram_matrix(self) -> Matrix:
        return Matrix.from_function(self.rank, self.rank, lambda i, j: self.d[i] * self.cartan[i, j])

    def coroot_pairing(self, beta: Sequence[int], alpha: Sequence[int]) -> Fraction:
        r""":math:`\langle \beta, \alpha^\vee\rangle = 2(\beta,\alpha)/(\alpha,\alpha)`."""
        return 2 * self.inner(beta, alpha) / self.inner(alpha, alpha)

    def long_and_short(self) -> Tuple[List[Root], List[Root]]:
        """Positive roots split into (long, short); short is empty if simply laced."""
        lengths = {beta: self.norm2(beta) for beta in self.positive_roots}
        top = max(lengths.values())
        return ([b for b, l in lengths.items() if l == top], [b for b, l in lengths.items() if l != top])

    def is_simply_laced(self) -> bool:
        return not self.long_and_short()[1]

    def root_string(self, beta: Sequence[int], alpha: Sequence[int]) -> List[Root]:
        r"""The :math:`\alpha`-string :math:`\beta - p\alpha, \dots, \beta + q\alpha` through :math:`\beta`."""
        beta, alpha = tuple(beta), tuple(alpha)

        def shift(k: int) -> Root:
            return tuple(b + k * a for b, a in zip(beta, alpha))

        p = 0
        while self.is_root(shift(-(p + 1))):
            p += 1
        q = 0
        while self.is_root(shift(q + 1)):
            q += 1
        return [shift(k) for k in range(-p, q + 1)]

    # -- Weyl group -------------------------------------------------------
    def reflect(self, i: int, beta: Sequence[int]) -> Root:
        r"""Simple reflection :math:`s_i(\beta) = \beta - \langle\beta,\alpha_i^\vee\rangle\alpha_i`."""
        c = self.pairing(beta, i)
        return tuple(b - (c if k == i else 0) for k, b in enumerate(beta))

    def reflection_matrix(self, i: int) -> Matrix:
        r"""Matrix of :math:`s_i` on simple-root coordinates (columns are images of :math:`\alpha_j`)."""
        cols = [list(self.reflect(i, a)) for a in self.simple_roots]
        return Matrix.from_columns(cols)

    def weyl_group(self, limit: int = 200000) -> List[Matrix]:
        r"""All elements of :math:`W` as integer matrices, by breadth-first search on words in :math:`s_i`."""
        gens = [self.reflection_matrix(i) for i in range(self.rank)]
        ident = Matrix.identity(self.rank)
        key = lambda m: tuple(m.flatten())  # noqa: E731
        seen = {key(ident): ident}
        frontier = [ident]
        while frontier:
            nxt = []
            for w in frontier:
                for s in gens:
                    ws = w @ s
                    k = key(ws)
                    if k not in seen:
                        seen[k] = ws
                        nxt.append(ws)
                        if len(seen) > limit:
                            raise ValueError("Weyl group larger than limit=%d" % limit)
            frontier = nxt
        return list(seen.values())

    def orbit(self, beta: Sequence[int]) -> List[Root]:
        r"""Weyl orbit :math:`W\beta` of a vector in simple-root coordinates."""
        start = tuple(beta)
        seen = {start}
        frontier = [start]
        while frontier:
            nxt = []
            for v in frontier:
                for i in range(self.rank):
                    w = self.reflect(i, v)
                    if w not in seen:
                        seen.add(w)
                        nxt.append(w)
            frontier = nxt
        return sorted(seen)

    def longest_element_length(self) -> int:
        r""":math:`\ell(w_0) = |\Phi^+|`."""
        return len(self.positive_roots)

    # -- Euclidean realisation ---------------------------------------------
    def to_euclidean(self, beta: Sequence[int]) -> List[Fraction]:
        """Coordinates in Bourbaki's ambient space (only for systems built with :meth:`of_type`)."""
        simple = getattr(self, "euclidean_simple", None)
        if simple is None:
            raise ValueError("no Euclidean realisation attached")
        dim = len(simple[0])
        return [sum((b * s[k] for b, s in zip(beta, simple)), Fraction(0)) for k in range(dim)]

    def summary(self) -> str:
        long_, short = self.long_and_short()
        lines = [
            "Root system %s (rank %d)" % (self.label or "?", self.rank),
            "  positive roots: %d   (|Phi| = %d)" % (len(self.positive_roots), 2 * len(self.positive_roots)),
            "  dim g = %d" % self.lie_algebra_dimension(),
            "  highest root: %s" % (self.highest_root,),
            "  Coxeter number h = %d" % self.coxeter_number(),
            "  exponents: %s" % self.exponents(),
            "  |W| = %d" % self.weyl_group_order(),
            "  long/short positive roots: %d/%d" % (len(long_), len(short)),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "RootSystem(%s)" % (self.label or self.cartan.rows())


def root_system(spec: str) -> RootSystem:
    """Shorthand for :meth:`RootSystem.of_type`."""
    return RootSystem.of_type(spec)
