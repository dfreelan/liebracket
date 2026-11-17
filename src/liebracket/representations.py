r"""Finite-dimensional representations :math:`\rho: \mathfrak g \to \mathfrak{gl}(V)`.

A representation is stored as the list of images :math:`\rho(e_i)` of the basis
of :math:`\mathfrak g`.  :meth:`Representation.is_homomorphism` verifies

.. math::

    \rho([e_i, e_j]) = [\rho(e_i), \rho(e_j)] \qquad \forall\, i < j

exactly.  Constructions provided: adjoint, defining, the irreducible
:math:`\mathfrak{sl}_2`/:math:`\mathfrak{su}(2)` modules :math:`V_n` / spin :math:`j`,
direct sums, tensor products, duals, trace forms and Casimir operators.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .algebra import LieAlgebra
from .classical import sl2, su2
from .linalg import Matrix, commutator
from .scalars import I, is_zero, to_exact

__all__ = [
    "Representation",
    "adjoint_representation",
    "defining_representation",
    "trivial_representation",
    "sl2_irrep",
    "su2_spin",
    "casimir_eigenspaces",
]


class Representation:
    r"""A linear representation of a :class:`~liebracket.algebra.LieAlgebra`.

    Parameters
    ----------
    algebra:
        The Lie algebra :math:`\mathfrak g`.
    matrices:
        Images :math:`\rho(e_1), \dots, \rho(e_n)`, all square of the same size.
    name:
        Optional label (``"adjoint"``, ``"V_3"``, ...).
    """

    def __init__(self, algebra: LieAlgebra, matrices: Sequence[Matrix], name: str = "rho") -> None:
        if len(matrices) != algebra.dim:
            raise ValueError("need one matrix per basis element (%d)" % algebra.dim)
        sizes = {m.shape for m in matrices}
        if len(sizes) != 1 or not matrices[0].is_square:
            raise ValueError("representation matrices must be square and of equal size")
        self.algebra = algebra
        self.matrices = list(matrices)
        self.name = name

    @property
    def dim(self) -> int:
        """Dimension of the representation space :math:`V`."""
        return self.matrices[0].nrows

    def __call__(self, x: Sequence[Any]) -> Matrix:
        r""":math:`\rho(x) = \sum_i x_i\,\rho(e_i)`."""
        total = Matrix.zeros(self.dim)
        for c, m in zip(x, self.matrices):
            if not is_zero(c):
                total = total + m * c
        return total

    # -- verification -----------------------------------------------------
    def homomorphism_defects(self) -> List[Tuple[int, int]]:
        r"""Index pairs :math:`(i,j)` with :math:`\rho([e_i,e_j]) \ne [\rho(e_i),\rho(e_j)]`."""
        bad = []
        for i, j in combinations(range(self.algebra.dim), 2):
            lhs = self(self.algebra.basis_bracket(i, j))
            rhs = commutator(self.matrices[i], self.matrices[j])
            if lhs != rhs:
                bad.append((i, j))
        return bad

    def is_homomorphism(self) -> bool:
        return not self.homomorphism_defects()

    def is_faithful(self) -> bool:
        r"""Injectivity of :math:`\rho` (the images are linearly independent)."""
        flat = Matrix([m.flatten() for m in self.matrices])
        return flat.rank() == self.algebra.dim

    def is_skew_hermitian(self) -> bool:
        r"""Unitarity at the Lie-algebra level: :math:`\rho(e_i)^\dagger = -\rho(e_i)`."""
        return all(m.dagger() == -m for m in self.matrices)

    # -- invariants -------------------------------------------------------
    def trace_form(self) -> Matrix:
        r"""Gram matrix :math:`B_\rho(e_i,e_j) = \operatorname{tr}(\rho(e_i)\rho(e_j))`; invariant by construction."""
        n = self.algebra.dim
        return Matrix.from_function(n, n, lambda i, j: (self.matrices[i] @ self.matrices[j]).trace())

    def dynkin_index(self) -> Fraction:
        r"""Ratio :math:`B_\rho / \kappa` for a simple algebra (:math:`B_\rho = \ell_\rho\,\kappa`)."""
        B = self.trace_form()
        K = self.algebra.killing_form()
        ratio: Optional[Any] = None
        for i in range(K.nrows):
            for j in range(K.ncols):
                if is_zero(K[i, j]):
                    if not is_zero(B[i, j]):
                        raise ValueError("trace form is not proportional to the Killing form")
                    continue
                r = to_exact(B[i, j] / Fraction(K[i, j]) if isinstance(K[i, j], int) else B[i, j] / K[i, j])
                if ratio is None:
                    ratio = r
                elif not is_zero(ratio - r):
                    raise ValueError("trace form is not proportional to the Killing form")
        if ratio is None:
            raise ValueError("Killing form vanishes")
        return Fraction(ratio)

    def casimir(self, form: Optional[Matrix] = None) -> Matrix:
        r"""Casimir operator :math:`C_\rho = \sum_{ij} B^{ij}\rho(e_i)\rho(e_j)` for a non-degenerate invariant form :math:`B`.

        By default :math:`B = \kappa` (Killing form).  :math:`C_\rho` commutes with every
        :math:`\rho(x)`, hence (Schur) is scalar on an irreducible module.
        """
        B = self.algebra.killing_form() if form is None else form
        Binv = B.inverse()
        total = Matrix.zeros(self.dim)
        n = self.algebra.dim
        for i in range(n):
            for j in range(n):
                if not is_zero(Binv[i, j]):
                    total = total + (self.matrices[i] @ self.matrices[j]) * Binv[i, j]
        return total

    def commutant_dimension(self) -> int:
        r""":math:`\dim \operatorname{End}_{\mathfrak g}(V)`; equals 1 iff :math:`V` is absolutely irreducible (in char 0, for semisimple :math:`\mathfrak g`)."""
        d = self.dim
        rows: List[List[Any]] = []
        for m in self.matrices:
            # linear map X -> mX - Xm on vec(X) (row-major)
            for a in range(d):
                for b in range(d):
                    row = [0] * (d * d)
                    for c in range(d):
                        row[c * d + b] = row[c * d + b] + m[a, c]
                        row[a * d + c] = row[a * d + c] - m[c, b]
                    rows.append(row)
        return len(Matrix(rows).nullspace())

    def is_irreducible(self) -> bool:
        return self.commutant_dimension() == 1

    def weights(self, cartan: Sequence[int]) -> Optional[List[Tuple[Any, ...]]]:
        r"""Weights :math:`(\lambda(H_1), \dots, \lambda(H_r))` if the given Cartan elements act diagonally, else ``None``."""
        diag = [self.matrices[c] for c in cartan]
        if not all(m.is_diagonal() for m in diag):
            return None
        return sorted((tuple(m[k, k] for m in diag) for k in range(self.dim)), reverse=True)

    # -- constructions ----------------------------------------------------
    def direct_sum(self, other: "Representation") -> "Representation":
        r""":math:`\rho \oplus \sigma`."""
        self._same_algebra(other)
        mats = [a.direct_sum(b) for a, b in zip(self.matrices, other.matrices)]
        return Representation(self.algebra, mats, name="(%s+%s)" % (self.name, other.name))

    def tensor(self, other: "Representation") -> "Representation":
        r""":math:`(\rho\otimes\sigma)(x) = \rho(x)\otimes 1 + 1\otimes\sigma(x)`."""
        self._same_algebra(other)
        I1, I2 = Matrix.identity(self.dim), Matrix.identity(other.dim)
        mats = [a.kron(I2) + I1.kron(b) for a, b in zip(self.matrices, other.matrices)]
        return Representation(self.algebra, mats, name="(%s x %s)" % (self.name, other.name))

    __matmul__ = tensor

    def dual(self) -> "Representation":
        r"""Contragredient :math:`\rho^*(x) = -\rho(x)^{\mathsf T}`."""
        return Representation(self.algebra, [-m.T for m in self.matrices], name=self.name + "*")

    def conjugated(self, P: Matrix) -> "Representation":
        r"""Equivalent representation :math:`P^{-1}\rho(x)P`."""
        Pinv = P.inverse()
        return Representation(self.algebra, [Pinv @ m @ P for m in self.matrices], name=self.name)

    def intertwines(self, other: "Representation", T: Matrix) -> bool:
        r"""Check :math:`T\rho(x) = \sigma(x)T` for all :math:`x`."""
        return all(T @ a == b @ T for a, b in zip(self.matrices, other.matrices))

    def _same_algebra(self, other: "Representation") -> None:
        if other.algebra.dim != self.algebra.dim:
            raise ValueError("representations of different algebras")

    def __repr__(self) -> str:
        return "Representation(%s of %s, dim=%d)" % (self.name, self.algebra.name, self.dim)


def adjoint_representation(algebra: LieAlgebra) -> Representation:
    r""":math:`\operatorname{ad}: \mathfrak g \to \mathfrak{gl}(\mathfrak g)`; a homomorphism precisely because of the Jacobi identity."""
    return Representation(algebra, [algebra.ad_basis(i) for i in range(algebra.dim)], name="ad")


def defining_representation(algebra: LieAlgebra) -> Representation:
    """The matrix realisation the algebra was constructed from."""
    if algebra.matrices is None:
        raise ValueError("%s has no stored matrix realisation" % algebra.name)
    return Representation(algebra, algebra.matrices, name="defining")


def trivial_representation(algebra: LieAlgebra, dim: int = 1) -> Representation:
    return Representation(algebra, [Matrix.zeros(dim)] * algebra.dim, name="trivial")


def _sl2_irrep_matrices(n: int) -> Tuple[Matrix, Matrix, Matrix]:
    r"""Matrices of :math:`e, f, h` on :math:`V_n = \operatorname{span}(v_0,\dots,v_n)`:
    :math:`h v_k = (n-2k)v_k,\ f v_k = v_{k+1},\ e v_k = k(n-k+1)v_{k-1}`."""
    d = n + 1
    E = Matrix.from_function(d, d, lambda i, j: j * (n - j + 1) if i == j - 1 else 0)
    F = Matrix.from_function(d, d, lambda i, j: 1 if i == j + 1 else 0)
    H = Matrix.diag([n - 2 * k for k in range(d)])
    return E, F, H


def sl2_irrep(n: int, algebra: Optional[LieAlgebra] = None) -> Representation:
    r"""The irreducible :math:`\mathfrak{sl}_2`-module :math:`V_n` of highest weight :math:`n` (dimension :math:`n+1`)."""
    if n < 0:
        raise ValueError("highest weight must be non-negative")
    g = algebra or sl2()
    E, F, H = _sl2_irrep_matrices(n)
    images = {"e": E, "f": F, "h": H}
    return Representation(g, [images[nm] for nm in g.names], name="V_%d" % n)


def su2_spin(j: Any, algebra: Optional[LieAlgebra] = None) -> Representation:
    r"""Spin-:math:`j` representation of :math:`\mathfrak{su}(2)` (dimension :math:`2j+1`), exact over :math:`\mathbb{Q}(i)`.

    Uses :math:`\sigma_1 \mapsto e+f,\ \sigma_2 \mapsto -i(e-f),\ \sigma_3 \mapsto h` and
    :math:`e_k = -\tfrac i2 \sigma_k`.  The basis is not orthonormal, so the matrices are
    skew-Hermitian only for :math:`j = 1/2`; :math:`\sum_k \rho(e_k)^2 = -j(j+1)\,\mathbb 1`.
    """
    j = Fraction(j)
    n = 2 * j
    if n.denominator != 1 or n < 0:
        raise ValueError("spin must be a non-negative half-integer")
    E, F, H = _sl2_irrep_matrices(int(n))
    half = -I * Fraction(1, 2)
    mats = [(E + F) * half, (E - F) * (-I) * half, H * half]
    return Representation(algebra or su2(), mats, name="spin_%s" % j)


def casimir_eigenspaces(rep: Representation, form: Optional[Matrix] = None) -> Dict[Any, int]:
    r"""Decompose :math:`V` by Casimir eigenvalue: returns ``{eigenvalue: dimension}``.

    Eigenvalues are found as rational roots of the characteristic polynomial; this
    suffices for all representations built from :func:`sl2_irrep` / :func:`su2_spin`
    / adjoints of split algebras.
    """
    C = rep.casimir(form)
    poly = C.characteristic_polynomial()
    candidates = _rational_roots(poly)
    out: Dict[Any, int] = {}
    n = rep.dim
    for lam in candidates:
        k = len((C - Matrix.identity(n) * lam).nullspace())
        if k:
            out[lam] = k
    if sum(out.values()) != n:
        raise ValueError("Casimir is not diagonalisable over Q")
    return out


def _rational_roots(coeffs: Sequence[Any]) -> List[Fraction]:
    r"""Rational roots of :math:`\sum c_k t^k` (rational coefficients) by the rational root theorem."""
    from math import gcd

    cs = [Fraction(to_exact(c)) for c in coeffs]
    while cs and cs[-1] == 0:
        cs.pop()
    roots: List[Fraction] = []
    shift = 0
    while cs and cs[0] == 0:
        cs.pop(0)
        shift += 1
    if shift:
        roots.append(Fraction(0))
    if len(cs) <= 1:
        return roots
    cs = _squarefree(cs)
    lcm = 1
    for c in cs:
        lcm = lcm * c.denominator // gcd(lcm, c.denominator)
    ints = [int(c * lcm) for c in cs]
    a0, an = abs(ints[0]), abs(ints[-1])

    def divisors(m: int) -> List[int]:
        small = [d for d in range(1, int(m ** 0.5) + 1) if m % d == 0]
        return sorted(set(small + [m // d for d in small]))

    for p in divisors(a0):
        for q in divisors(an):
            for s in (1, -1):
                r = Fraction(s * p, q)
                if r in roots:
                    continue
                if sum(c * r ** k for k, c in enumerate(ints)) == 0:
                    roots.append(r)
    return sorted(roots)


def _poly_divmod(a: List[Fraction], b: List[Fraction]) -> Tuple[List[Fraction], List[Fraction]]:
    """Quotient and remainder of polynomials (coefficient lists, lowest degree first)."""
    a = list(a)
    q = [Fraction(0)] * max(len(a) - len(b) + 1, 1)
    while len(a) >= len(b) and any(a):
        shift = len(a) - len(b)
        f = a[-1] / b[-1]
        q[shift] = f
        for i, c in enumerate(b):
            a[i + shift] -= f * c
        while a and a[-1] == 0:
            a.pop()
    return q, a


def _squarefree(p: List[Fraction]) -> List[Fraction]:
    r"""Square-free part :math:`p / \gcd(p, p')`; has the same roots with multiplicity one."""
    dp = [k * c for k, c in enumerate(p)][1:]
    a, b = list(p), dp
    while b and any(b):
        _, r = _poly_divmod(a, b)
        a, b = b, r
    g = a
    if len(g) <= 1:
        return p
    q, _ = _poly_divmod(p, g)
    while q and q[-1] == 0:
        q.pop()
    return q
