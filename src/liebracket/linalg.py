r"""Dense exact linear algebra over an arbitrary field of scalars.

The :class:`Matrix` type is a small immutable dense matrix whose entries may be
``int``, :class:`~fractions.Fraction`, :class:`~liebracket.scalars.GaussianRational`
or inexact ``float``/``complex``.  Elimination routines (rank, determinant,
inverse, null space) are fraction-exact whenever the entries are exact, which
is what makes the structure-constant and Killing-form computations of this
package *proofs* rather than approximations.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from .scalars import format_scalar, is_zero, to_exact

__all__ = [
    "Matrix",
    "Vector",
    "commutator",
    "rref",
    "solve_in_span",
    "span_basis",
    "vec_add",
    "vec_scale",
    "vec_sub",
    "vec_is_zero",
    "dot",
]

Vector = List[Any]


def _exactify(x: Any) -> Any:
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    if isinstance(x, bool):
        return int(x)
    return to_exact(x)


# -- vector helpers --------------------------------------------------------
def vec_add(u: Sequence[Any], v: Sequence[Any]) -> Vector:
    """Componentwise sum :math:`u + v`."""
    return [a + b for a, b in zip(u, v)]


def vec_sub(u: Sequence[Any], v: Sequence[Any]) -> Vector:
    """Componentwise difference :math:`u - v`."""
    return [a - b for a, b in zip(u, v)]


def vec_scale(c: Any, v: Sequence[Any]) -> Vector:
    """Scalar multiple :math:`c\\,v`."""
    return [c * a for a in v]


def vec_is_zero(v: Sequence[Any]) -> bool:
    return all(is_zero(a) for a in v)


def dot(u: Sequence[Any], v: Sequence[Any]) -> Any:
    r"""Bilinear (non-conjugating) pairing :math:`\sum_i u_i v_i`."""
    total: Any = 0
    for a, b in zip(u, v):
        total = total + a * b
    return total


class Matrix:
    r"""Immutable dense matrix :math:`A \in \mathbb{F}^{m \times n}`.

    >>> A = Matrix([[1, 2], [3, 4]])
    >>> A.det()
    -2
    >>> (A @ A.inverse()) == Matrix.identity(2)
    True
    """

    __slots__ = ("_rows", "nrows", "ncols")

    def __init__(self, rows: Iterable[Iterable[Any]]) -> None:
        data = [tuple(_exactify(x) for x in r) for r in rows]
        ncols = len(data[0]) if data else 0
        if any(len(r) != ncols for r in data):
            raise ValueError("ragged matrix rows")
        self._rows: Tuple[Tuple[Any, ...], ...] = tuple(data)
        self.nrows = len(data)
        self.ncols = ncols

    # -- constructors -----------------------------------------------------
    @classmethod
    def zeros(cls, m: int, n: Optional[int] = None) -> "Matrix":
        n = m if n is None else n
        return cls([[0] * n for _ in range(m)])

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        return cls([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @classmethod
    def diag(cls, entries: Sequence[Any]) -> "Matrix":
        n = len(entries)
        return cls([[entries[i] if i == j else 0 for j in range(n)] for i in range(n)])

    @classmethod
    def from_function(cls, m: int, n: int, f: Callable[[int, int], Any]) -> "Matrix":
        return cls([[f(i, j) for j in range(n)] for i in range(m)])

    @classmethod
    def unit(cls, n: int, i: int, j: int) -> "Matrix":
        r"""Matrix unit :math:`E_{ij}` of size :math:`n`."""
        return cls.from_function(n, n, lambda a, b: 1 if (a, b) == (i, j) else 0)

    @classmethod
    def from_columns(cls, columns: Sequence[Sequence[Any]]) -> "Matrix":
        if not columns:
            return cls([])
        return cls([[col[i] for col in columns] for i in range(len(columns[0]))])

    # -- access -----------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nrows, self.ncols)

    @property
    def is_square(self) -> bool:
        return self.nrows == self.ncols

    def __getitem__(self, key: Tuple[int, int]) -> Any:
        i, j = key
        return self._rows[i][j]

    def row(self, i: int) -> Vector:
        return list(self._rows[i])

    def column(self, j: int) -> Vector:
        return [r[j] for r in self._rows]

    def rows(self) -> List[Vector]:
        return [list(r) for r in self._rows]

    def flatten(self) -> Vector:
        return [x for r in self._rows for x in r]

    def map(self, f: Callable[[Any], Any]) -> "Matrix":
        return Matrix([[f(x) for x in r] for r in self._rows])

    # -- algebra ----------------------------------------------------------
    def _check_same_shape(self, other: "Matrix") -> None:
        if self.shape != other.shape:
            raise ValueError("shape mismatch %s vs %s" % (self.shape, other.shape))

    def __add__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            return NotImplemented
        self._check_same_shape(other)
        return Matrix([[a + b for a, b in zip(r, s)] for r, s in zip(self._rows, other._rows)])

    def __sub__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            return NotImplemented
        self._check_same_shape(other)
        return Matrix([[a - b for a, b in zip(r, s)] for r, s in zip(self._rows, other._rows)])

    def __neg__(self) -> "Matrix":
        return self.map(lambda x: -x)

    def __mul__(self, c: Any) -> "Matrix":
        if isinstance(c, Matrix):
            raise TypeError("use @ for matrix multiplication")
        return self.map(lambda x: c * x)

    __rmul__ = __mul__

    def __truediv__(self, c: Any) -> "Matrix":
        if isinstance(c, int):
            c = Fraction(c)
        return self.map(lambda x: x / c)

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, Matrix):
            if self.ncols != other.nrows:
                raise ValueError("cannot multiply %s by %s" % (self.shape, other.shape))
            cols = list(zip(*other._rows)) if other.nrows else [()] * other.ncols
            return Matrix([[dot(r, c) for c in cols] for r in self._rows])
        if isinstance(other, (list, tuple)):
            return [dot(r, other) for r in self._rows]
        return NotImplemented

    def __pow__(self, k: int) -> "Matrix":
        if not self.is_square or k < 0:
            raise ValueError("power requires a square matrix and k >= 0")
        result = Matrix.identity(self.nrows)
        base = self
        while k:
            if k & 1:
                result = result @ base
            base = base @ base
            k >>= 1
        return result

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.shape == other.shape and all(
            is_zero(a - b) for a, b in zip(self.flatten(), other.flatten())
        )

    __hash__ = None  # type: ignore[assignment]

    def transpose(self) -> "Matrix":
        return Matrix(zip(*self._rows)) if self.nrows else Matrix([])

    @property
    def T(self) -> "Matrix":
        return self.transpose()

    def conjugate(self) -> "Matrix":
        return self.map(lambda x: x.conjugate() if hasattr(x, "conjugate") else x)

    def dagger(self) -> "Matrix":
        r"""Hermitian adjoint :math:`A^\dagger = \bar A^{\mathsf T}`."""
        return self.conjugate().transpose()

    def trace(self) -> Any:
        if not self.is_square:
            raise ValueError("trace of a non-square matrix")
        total: Any = 0
        for i in range(self.nrows):
            total = total + self._rows[i][i]
        return _exactify(total)

    def is_zero(self) -> bool:
        return all(is_zero(x) for x in self.flatten())

    def is_symmetric(self) -> bool:
        return self.is_square and self == self.transpose()

    def is_antisymmetric(self) -> bool:
        return self.is_square and self == -self.transpose()

    def is_diagonal(self) -> bool:
        return self.is_square and all(
            is_zero(self[i, j]) for i in range(self.nrows) for j in range(self.ncols) if i != j
        )

    def diagonal(self) -> Vector:
        return [self._rows[i][i] for i in range(min(self.nrows, self.ncols))]

    def kron(self, other: "Matrix") -> "Matrix":
        r"""Kronecker product :math:`A \otimes B`."""
        m, n = other.shape
        return Matrix.from_function(
            self.nrows * m, self.ncols * n,
            lambda i, j: self[i // m, j // n] * other[i % m, j % n],
        )

    def direct_sum(self, other: "Matrix") -> "Matrix":
        r"""Block diagonal :math:`A \oplus B`."""
        m, n = self.shape
        p, q = other.shape

        def entry(i: int, j: int) -> Any:
            if i < m and j < n:
                return self[i, j]
            if i >= m and j >= n:
                return other[i - m, j - n]
            return 0

        return Matrix.from_function(m + p, n + q, entry)

    # -- elimination ------------------------------------------------------
    def rref(self) -> Tuple["Matrix", List[int]]:
        """Reduced row echelon form and the list of pivot columns."""
        return rref(self.rows())

    def rank(self) -> int:
        return len(self.rref()[1])

    def det(self) -> Any:
        """Determinant via fraction-exact Gaussian elimination."""
        if not self.is_square:
            raise ValueError("determinant of a non-square matrix")
        n = self.nrows
        a = [[Fraction(x) if isinstance(x, int) else x for x in r] for r in self._rows]
        det: Any = 1
        for c in range(n):
            pivot = next((r for r in range(c, n) if not is_zero(a[r][c])), None)
            if pivot is None:
                return 0
            if pivot != c:
                a[c], a[pivot] = a[pivot], a[c]
                det = -det
            p = a[c][c]
            det = det * p
            for r in range(c + 1, n):
                if not is_zero(a[r][c]):
                    f = a[r][c] / p
                    a[r] = [x - f * y for x, y in zip(a[r], a[c])]
        return _exactify(det)

    def inverse(self) -> "Matrix":
        if not self.is_square:
            raise ValueError("inverse of a non-square matrix")
        n = self.nrows
        aug = [list(r) + [1 if i == j else 0 for j in range(n)] for i, r in enumerate(self._rows)]
        red, pivots = rref(aug)
        if pivots[:n] != list(range(n)):
            raise ZeroDivisionError("matrix is singular")
        return Matrix([red.row(i)[n:] for i in range(n)])

    def solve(self, b: Sequence[Any]) -> Optional[Vector]:
        r"""A particular solution :math:`x` of :math:`A x = b`, or ``None``."""
        aug = [list(r) + [b[i]] for i, r in enumerate(self._rows)]
        red, pivots = rref(aug)
        if self.ncols in pivots:
            return None
        x: Vector = [0] * self.ncols
        for r, c in enumerate(pivots):
            x[c] = red[r, self.ncols]
        return x

    def nullspace(self) -> List[Vector]:
        r"""A basis of :math:`\ker A` (as column vectors)."""
        red, pivots = self.rref()
        free = [c for c in range(self.ncols) if c not in pivots]
        basis = []
        for f in free:
            v: Vector = [0] * self.ncols
            v[f] = 1
            for r, c in enumerate(pivots):
                v[c] = _exactify(-red[r, f])
            basis.append(v)
        return basis

    def characteristic_polynomial(self) -> List[Any]:
        r"""Coefficients :math:`[c_0, \dots, c_n]` of :math:`\det(t I - A) = \sum c_k t^k`.

        Uses the Faddeev--LeVerrier recursion, exact over any field of
        characteristic zero.
        """
        n = self.nrows
        coeffs: List[Any] = [0] * (n + 1)
        coeffs[n] = 1
        M = Matrix.zeros(n)
        ident = Matrix.identity(n)
        for k in range(1, n + 1):
            M = self @ M + ident * coeffs[n - k + 1]
            coeffs[n - k] = _exactify(-Fraction(1, k) * (self @ M).trace())
        return coeffs

    # -- presentation -----------------------------------------------------
    def pretty(self) -> str:
        cells = [[format_scalar(x) for x in r] for r in self._rows]
        if not cells:
            return "[]"
        width = max(len(c) for r in cells for c in r)
        lines = ["[" + "  ".join(c.rjust(width) for c in r) + "]" for r in cells]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "Matrix(%r)" % ([[format_scalar(x) for x in r] for r in self._rows],)

    def __str__(self) -> str:
        return self.pretty()


def commutator(a: Matrix, b: Matrix) -> Matrix:
    r"""Matrix commutator :math:`[A, B] = AB - BA`."""
    return a @ b - b @ a


def rref(rows: Sequence[Sequence[Any]]) -> Tuple[Matrix, List[int]]:
    """Reduced row echelon form of a list of rows; returns ``(R, pivot_columns)``."""
    a = [[Fraction(x) if isinstance(x, int) else x for x in r] for r in rows]
    m = len(a)
    n = len(a[0]) if m else 0
    pivots: List[int] = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        p = next((i for i in range(r, m) if not is_zero(a[i][c])), None)
        if p is None:
            continue
        a[r], a[p] = a[p], a[r]
        pv = a[r][c]
        a[r] = [x / pv for x in a[r]]
        for i in range(m):
            if i != r and not is_zero(a[i][c]):
                f = a[i][c]
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
        pivots.append(c)
        r += 1
    return Matrix(a) if a else Matrix([]), pivots


def span_basis(vectors: Sequence[Sequence[Any]]) -> List[Vector]:
    """A basis (in reduced echelon form) of the span of ``vectors``."""
    vectors = [list(v) for v in vectors]
    if not vectors:
        return []
    red, pivots = rref(vectors)
    return [[_exactify(x) for x in red.row(i)] for i in range(len(pivots))]


def solve_in_span(basis: Sequence[Sequence[Any]], v: Sequence[Any]) -> Optional[Vector]:
    r"""Coefficients :math:`c` with :math:`\sum_k c_k b_k = v`, or ``None`` if :math:`v \notin \operatorname{span}(b)`."""
    if not basis:
        return [] if vec_is_zero(v) else None
    A = Matrix.from_columns(basis)
    x = A.solve(v)
    return None if x is None else [_exactify(c) for c in x]
