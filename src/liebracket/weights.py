r"""Highest-weight theory: Weyl's dimension formula and Freudenthal's multiplicity formula.

Weights are written in the basis of fundamental weights :math:`\omega_i`
(Dynkin labels): :math:`\lambda = \sum_i \lambda_i\omega_i` with
:math:`\lambda_i = \langle\lambda,\alpha_i^\vee\rangle`.  A simple root has labels
given by a column of the Cartan matrix, :math:`\alpha_j = \sum_i A_{ij}\,\omega_i`.

*Weyl dimension formula*

.. math::

    \dim V(\lambda) = \prod_{\alpha > 0} \frac{(\lambda + \rho, \alpha)}{(\rho, \alpha)},
    \qquad \rho = \sum_i \omega_i .

*Freudenthal's formula*

.. math::

    \bigl((\lambda+\rho,\lambda+\rho) - (\mu+\rho,\mu+\rho)\bigr)\, m_\lambda(\mu)
    = 2 \sum_{\alpha>0}\sum_{k\ge1} (\mu + k\alpha, \alpha)\, m_\lambda(\mu + k\alpha).
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

from .roots import RootSystem

__all__ = ["WeightSpace", "Character"]

Weight = Tuple[int, ...]
Character = Dict[Weight, int]


class WeightSpace:
    r"""Weight lattice :math:`P` of a root system, with dominant-weight representation theory."""

    def __init__(self, rs: RootSystem) -> None:
        self.rs = rs
        self.rank = rs.rank
        A = rs.cartan
        self._Ainv = A.map(Fraction).inverse()
        # simple roots and positive roots in Dynkin labels
        self.simple_root_labels: List[Weight] = [tuple(int(A[i, j]) for i in range(self.rank)) for j in range(self.rank)]
        self.positive_root_labels: List[Weight] = [self.root_to_weight(b) for b in rs.positive_roots]
        self._char_cache: Dict[Weight, Character] = {}

    @classmethod
    def of_type(cls, spec: str) -> "WeightSpace":
        return cls(RootSystem.of_type(spec))

    # -- coordinates ------------------------------------------------------
    def root_to_weight(self, beta: Sequence[int]) -> Weight:
        r"""Simple-root coordinates :math:`\to` Dynkin labels (:math:`\lambda = A c`)."""
        A = self.rs.cartan
        return tuple(sum(int(A[i, j]) * beta[j] for j in range(self.rank)) for i in range(self.rank))

    def weight_to_root_coords(self, lam: Sequence[int]) -> List[Fraction]:
        r"""Dynkin labels :math:`\to` (rational) simple-root coordinates :math:`c = A^{-1}\lambda`."""
        return [sum((self._Ainv[j, k] * lam[k] for k in range(self.rank)), Fraction(0)) for j in range(self.rank)]

    def inner(self, lam: Sequence[int], mu: Sequence[int]) -> Fraction:
        r""":math:`(\lambda,\mu) = \sum_j c_j\, d_j\, \mu_j` with :math:`c = A^{-1}\lambda`."""
        c = self.weight_to_root_coords(lam)
        return sum((c[j] * self.rs.d[j] * mu[j] for j in range(self.rank)), Fraction(0))

    def pair_with_root(self, lam: Sequence[int], beta: Sequence[int]) -> Fraction:
        r""":math:`(\lambda, \beta) = \sum_j \beta_j d_j \lambda_j` for a root :math:`\beta` in simple-root coordinates."""
        return sum((Fraction(beta[j]) * self.rs.d[j] * lam[j] for j in range(self.rank)), Fraction(0))

    @property
    def rho(self) -> Weight:
        r"""Weyl vector :math:`\rho = \sum_i \omega_i = \tfrac12\sum_{\alpha>0}\alpha`."""
        return tuple([1] * self.rank)

    def fundamental_weights(self) -> List[Weight]:
        return [tuple(1 if k == i else 0 for k in range(self.rank)) for i in range(self.rank)]

    @staticmethod
    def is_dominant(lam: Sequence[int]) -> bool:
        return all(x >= 0 for x in lam)

    def reflect(self, i: int, lam: Sequence[int]) -> Weight:
        r""":math:`s_i\lambda = \lambda - \lambda_i\alpha_i`."""
        a = self.simple_root_labels[i]
        return tuple(x - lam[i] * y for x, y in zip(lam, a))

    def weyl_orbit(self, lam: Sequence[int]) -> List[Weight]:
        seen = {tuple(lam)}
        frontier = [tuple(lam)]
        while frontier:
            nxt = []
            for v in frontier:
                for i in range(self.rank):
                    w = self.reflect(i, v)
                    if w not in seen:
                        seen.add(w)
                        nxt.append(w)
            frontier = nxt
        return sorted(seen, reverse=True)

    def dominant_representative(self, lam: Sequence[int]) -> Weight:
        """The unique dominant weight in the Weyl orbit of ``lam``."""
        v = tuple(lam)
        while not self.is_dominant(v):
            i = next(k for k, x in enumerate(v) if x < 0)
            v = self.reflect(i, v)
        return v

    # -- dimensions -------------------------------------------------------
    def weyl_dimension(self, lam: Sequence[int]) -> int:
        r"""Weyl's dimension formula :math:`\prod_{\alpha>0}(\lambda+\rho,\alpha)/(\rho,\alpha)`."""
        if len(lam) != self.rank or not self.is_dominant(lam):
            raise ValueError("highest weight must be dominant with %d labels" % self.rank)
        lr = [x + 1 for x in lam]
        num, den = Fraction(1), Fraction(1)
        for beta in self.rs.positive_roots:
            num *= self.pair_with_root(lr, beta)
            den *= self.pair_with_root(self.rho, beta)
        q = num / den
        if q.denominator != 1:
            raise ArithmeticError("non-integral Weyl dimension (bug)")
        return int(q)

    def casimir_eigenvalue(self, lam: Sequence[int]) -> Fraction:
        r""":math:`(\lambda, \lambda + 2\rho)`, the Casimir eigenvalue on :math:`V(\lambda)` for the normalised form."""
        l2r = [x + 2 for x in lam]
        return self.inner(lam, l2r)

    # -- Freudenthal --------------------------------------------------------
    def character(self, lam: Sequence[int]) -> Character:
        r"""Formal character :math:`\operatorname{ch}V(\lambda) = \sum_\mu m_\lambda(\mu)\, e^\mu` via Freudenthal's formula."""
        lam = tuple(lam)
        if lam in self._char_cache:
            return dict(self._char_cache[lam])
        if not self.is_dominant(lam):
            raise ValueError("highest weight must be dominant")
        rho = self.rho
        lr = tuple(a + b for a, b in zip(lam, rho))
        top = self.inner(lr, lr)
        mult: Dict[Weight, int] = {lam: 1}
        # depth = height of lam - mu in simple roots; process level by level
        level = [lam]
        while level:
            candidates = set()
            for mu in level:
                for a in self.simple_root_labels:
                    candidates.add(tuple(x - y for x, y in zip(mu, a)))
            nxt = []
            for mu in sorted(candidates, reverse=True):
                if mu in mult:
                    continue
                m = self._freudenthal(mu, mult, top, self.depth(lam, mu))
                if m:
                    mult[mu] = m
                    nxt.append(mu)
            level = nxt
        self._char_cache[lam] = dict(mult)
        return mult

    def depth(self, lam: Sequence[int], mu: Sequence[int]) -> int:
        r"""Height of :math:`\lambda - \mu` in simple roots (non-negative for weights of :math:`V(\lambda)`)."""
        c = self.weight_to_root_coords([a - b for a, b in zip(lam, mu)])
        return int(sum(c))

    def _freudenthal(self, mu: Weight, mult: Dict[Weight, int], top: Fraction, depth: int) -> int:
        rho = self.rho
        mr = tuple(a + b for a, b in zip(mu, rho))
        denom = top - self.inner(mr, mr)
        if denom == 0:
            return 0
        total = Fraction(0)
        for beta, a in zip(self.rs.positive_roots, self.positive_root_labels):
            ht = sum(beta)
            k = 1
            while k * ht <= depth:
                nu = tuple(x + k * y for x, y in zip(mu, a))
                m = mult.get(nu)
                if m:
                    total += self.pair_with_root(nu, beta) * m
                k += 1
        value = 2 * total / denom
        if value.denominator != 1 or value < 0:
            raise ArithmeticError("non-integral multiplicity %s at %s (bug)" % (value, mu))
        return int(value)

    def multiplicity(self, lam: Sequence[int], mu: Sequence[int]) -> int:
        return self.character(lam).get(tuple(mu), 0)

    def dominant_character(self, lam: Sequence[int]) -> Dict[Weight, int]:
        """Multiplicities of the dominant weights only (the rest follow by Weyl symmetry)."""
        return {mu: m for mu, m in sorted(self.character(lam).items(), reverse=True) if self.is_dominant(mu)}

    # -- tensor products ----------------------------------------------------
    @staticmethod
    def multiply_characters(a: Character, b: Character) -> Character:
        out: Dict[Weight, int] = defaultdict(int)
        for mu, m in a.items():
            for nu, n in b.items():
                out[tuple(x + y for x, y in zip(mu, nu))] += m * n
        return dict(out)

    def decompose(self, char: Character) -> Dict[Weight, int]:
        r"""Write a Weyl-invariant character as :math:`\sum_\lambda n_\lambda \operatorname{ch}V(\lambda)` (peeling highest weights)."""
        rest = {k: v for k, v in char.items() if v}
        out: Dict[Weight, int] = {}
        while rest:
            # a maximal weight w.r.t. the dominance order: maximise the rho-height
            hw = max(rest, key=lambda w: (self.inner(w, self.rho), w))
            n = rest[hw]
            if n < 0 or not self.is_dominant(hw):
                raise ValueError("character is not a non-negative combination of irreducibles")
            out[hw] = n
            for mu, m in self.character(hw).items():
                left = rest.get(mu, 0) - n * m
                if left:
                    rest[mu] = left
                else:
                    rest.pop(mu, None)
        return dict(sorted(out.items(), key=lambda t: (-self.weyl_dimension(t[0]), t[0])))

    def tensor_product(self, lam: Sequence[int], mu: Sequence[int]) -> Dict[Weight, int]:
        r"""Clebsch--Gordan decomposition :math:`V(\lambda)\otimes V(\mu) = \bigoplus_\nu N_{\lambda\mu}^{\nu}\, V(\nu)`."""
        return self.decompose(self.multiply_characters(self.character(lam), self.character(mu)))

    def adjoint_highest_weight(self) -> Weight:
        """Dynkin labels of the highest root :math:`\\theta`."""
        return self.root_to_weight(self.rs.highest_root)

    def irreps_up_to(self, max_label: int) -> List[Tuple[Weight, int]]:
        """All dominant weights with labels :math:`\\le` ``max_label`` and their dimensions, sorted by dimension."""
        out: List[Tuple[Weight, int]] = []

        def rec(prefix: List[int]) -> None:
            if len(prefix) == self.rank:
                w = tuple(prefix)
                out.append((w, self.weyl_dimension(w)))
                return
            for k in range(max_label + 1):
                rec(prefix + [k])

        rec([])
        return sorted(out, key=lambda t: (t[1], t[0]))

    def format_weight(self, lam: Sequence[int]) -> str:
        return "(" + ",".join(str(x) for x in lam) + ")"

    def format_decomposition(self, dec: Dict[Weight, int]) -> str:
        parts = []
        for w, n in dec.items():
            term = "V%s[%d]" % (self.format_weight(w), self.weyl_dimension(w))
            parts.append(term if n == 1 else "%d*%s" % (n, term))
        return " + ".join(parts)

