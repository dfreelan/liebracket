r"""The Baker--Campbell--Hausdorff series, computed exactly.

For non-commuting :math:`X, Y`,

.. math::

    \log(e^X e^Y) = X + Y + \tfrac12[X,Y] + \tfrac1{12}[X,[X,Y]] - \tfrac1{12}[Y,[X,Y]]
    - \tfrac1{24}[Y,[X,[X,Y]]] + \cdots

This module computes the series to any degree :math:`N` in the free associative
algebra :math:`\mathbb Q\langle X, Y\rangle` (truncated at degree :math:`N`) and
converts each homogeneous component to Lie form with the
Dynkin--Specht--Wever projection

.. math::

    P(x_{i_1}\cdots x_{i_n}) = [x_{i_1},[x_{i_2},[\cdots,x_{i_n}]\cdots]],\qquad
    P(L) = n\,L \text{ for a Lie polynomial } L \text{ of degree } n .

Coefficients are then expressed in a basis of right-normed brackets chosen by exact
elimination, so the output is canonical.  The result can be evaluated on matrices
(:func:`bch_matrices`) or inside any :class:`~liebracket.algebra.LieAlgebra`
(:func:`bch_in_algebra`).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .linalg import Matrix, commutator, rref

__all__ = [
    "FreeElement",
    "bch_series",
    "bch_associative",
    "bch_lie_terms",
    "right_normed",
    "lie_basis",
    "witt_dimension",
    "is_lie_element",
    "format_lie_terms",
    "evaluate_terms",
    "bch_matrices",
    "bch_in_algebra",
    "exp_matrix",
    "log_unipotent",
]

Word = Tuple[int, ...]


class FreeElement:
    r"""Element of the free associative algebra :math:`\mathbb Q\langle x_0, \dots, x_{k-1}\rangle`, truncated at ``degree``.

    Stored as ``{word: coefficient}``; the empty word is the unit.
    """

    __slots__ = ("terms", "degree")

    def __init__(self, terms: Optional[Dict[Word, Fraction]] = None, degree: int = 10) -> None:
        self.degree = degree
        self.terms: Dict[Word, Fraction] = {
            w: Fraction(c) for w, c in (terms or {}).items() if c != 0 and len(w) <= degree
        }

    @classmethod
    def generator(cls, i: int, degree: int) -> "FreeElement":
        return cls({(i,): Fraction(1)}, degree)

    @classmethod
    def one(cls, degree: int) -> "FreeElement":
        return cls({(): Fraction(1)}, degree)

    def __add__(self, other: "FreeElement") -> "FreeElement":
        out = dict(self.terms)
        for w, c in other.terms.items():
            out[w] = out.get(w, Fraction(0)) + c
        return FreeElement(out, min(self.degree, other.degree))

    def __sub__(self, other: "FreeElement") -> "FreeElement":
        return self + other.scale(-1)

    def scale(self, c: Any) -> "FreeElement":
        return FreeElement({w: c * v for w, v in self.terms.items()}, self.degree)

    def __mul__(self, other: Any) -> "FreeElement":
        if not isinstance(other, FreeElement):
            return self.scale(other)
        deg = min(self.degree, other.degree)
        out: Dict[Word, Fraction] = {}
        for u, a in self.terms.items():
            for v, b in other.terms.items():
                if len(u) + len(v) <= deg:
                    w = u + v
                    out[w] = out.get(w, Fraction(0)) + a * b
        return FreeElement(out, deg)

    __rmul__ = scale

    def homogeneous(self, n: int) -> "FreeElement":
        return FreeElement({w: c for w, c in self.terms.items() if len(w) == n}, self.degree)

    def constant(self) -> Fraction:
        return self.terms.get((), Fraction(0))

    def is_zero(self) -> bool:
        return not self.terms

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FreeElement) and self.terms == other.terms

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return "FreeElement(%d terms, degree<=%d)" % (len(self.terms), self.degree)


def free_exp(x: FreeElement) -> FreeElement:
    r""":math:`\exp(x) = \sum_{k=0}^{N} x^k / k!` for :math:`x` without constant term."""
    if x.constant() != 0:
        raise ValueError("exp needs an element with zero constant term")
    result = FreeElement.one(x.degree)
    power = FreeElement.one(x.degree)
    for k in range(1, x.degree + 1):
        power = (power * x).scale(Fraction(1, k))
        if power.is_zero():
            break
        result = result + power
    return result


def free_log(y: FreeElement) -> FreeElement:
    r""":math:`\log(1 + z) = \sum_{k\ge1} (-1)^{k+1} z^k / k` for :math:`y = 1 + z`."""
    if y.constant() != 1:
        raise ValueError("log needs constant term 1")
    z = y - FreeElement.one(y.degree)
    result = FreeElement({}, y.degree)
    power = FreeElement.one(y.degree)
    for k in range(1, y.degree + 1):
        power = power * z
        if power.is_zero():
            break
        result = result + power.scale(Fraction((-1) ** (k + 1), k))
    return result


def bch_associative(degree: int) -> FreeElement:
    r""":math:`\log(e^{X}e^{Y})` in :math:`\mathbb Q\langle X,Y\rangle` up to total degree ``degree`` (:math:`X=x_0, Y=x_1`)."""
    X = FreeElement.generator(0, degree)
    Y = FreeElement.generator(1, degree)
    return free_log(free_exp(X) * free_exp(Y))


def right_normed(word: Sequence[int], degree: Optional[int] = None) -> FreeElement:
    r"""Expansion of :math:`[x_{i_1},[x_{i_2},[\cdots,x_{i_n}]]]` as an associative polynomial."""
    deg = degree if degree is not None else len(word)
    cur = FreeElement.generator(word[-1], deg)
    for letter in reversed(word[:-1]):
        g = FreeElement.generator(letter, deg)
        cur = g * cur - cur * g
    return cur


def witt_dimension(n: int, k: int = 2) -> int:
    r"""Witt's formula: :math:`\dim \mathcal L_n(k) = \frac1n\sum_{d\mid n}\mu(d)\,k^{n/d}` (free Lie algebra on :math:`k` generators)."""

    def mobius(m: int) -> int:
        result, p = 1, 2
        while p * p <= m:
            if m % p == 0:
                m //= p
                if m % p == 0:
                    return 0
                result = -result
            p += 1
        return -result if m > 1 else result

    total = sum(mobius(d) * k ** (n // d) for d in range(1, n + 1) if n % d == 0)
    return total // n


_BASIS_CACHE: Dict[Tuple[int, int], Tuple[List[Word], List[Word], Matrix]] = {}


def lie_basis(n: int, k: int = 2) -> List[Word]:
    r"""Words :math:`w` whose right-normed brackets :math:`[w]` form a basis of the degree-:math:`n` part of the free Lie algebra.

    Chosen greedily in lexicographic order among words ending in an increasing
    pair (other words give zero or a repeat up to sign).  Its size equals :func:`witt_dimension`.
    """
    return _basis_data(n, k)[0]


def _basis_data(n: int, k: int) -> Tuple[List[Word], List[Word], Matrix]:
    key = (n, k)
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]
    all_words = [tuple(w) for w in product(range(k), repeat=n)]
    col = {w: i for i, w in enumerate(all_words)}
    if n == 1:
        cands = [(i,) for i in range(k)]
    else:
        cands = [w for w in all_words if w[-2] < w[-1]]
    chosen: List[Word] = []
    rows: List[List[Fraction]] = []
    rank = 0
    for w in cands:
        vec = [Fraction(0)] * len(all_words)
        for u, c in right_normed(w).terms.items():
            vec[col[u]] = c
        trial = rows + [vec]
        if len(rref(trial)[1]) > rank:
            rows, rank = trial, rank + 1
            chosen.append(w)
    # column j of B is the expansion of the j-th chosen bracket in the word basis
    B = Matrix.from_columns(rows) if rows else Matrix([])
    data = (chosen, all_words, B)
    _BASIS_CACHE[key] = data
    return data


def is_lie_element(p: FreeElement) -> bool:
    r"""Dynkin--Specht--Wever criterion: a homogeneous-by-degree :math:`p` is Lie iff :math:`P(p_n) = n\,p_n` for every :math:`n`."""
    degs = {len(w) for w in p.terms}
    for n in degs:
        if n == 0:
            return False
        comp = p.homogeneous(n)
        proj = FreeElement({}, p.degree)
        for w, c in comp.terms.items():
            proj = proj + right_normed(w, p.degree).scale(c)
        if not (proj - comp.scale(n)).is_zero():
            return False
    return True


def bch_lie_terms(degree: int, series: Optional[FreeElement] = None) -> List[Tuple[Fraction, Word]]:
    r"""BCH series up to ``degree`` as ``[(c, w), ...]`` with :math:`\log(e^Xe^Y) = \sum c\,[w]`."""
    z = series if series is not None else bch_associative(degree)
    out: List[Tuple[Fraction, Word]] = []
    for n in range(1, degree + 1):
        comp = z.homogeneous(n)
        if comp.is_zero():
            continue
        words, all_words, B = _basis_data(n, 2)
        target = [comp.terms.get(w, Fraction(0)) for w in all_words]
        x = B.solve(target)
        if x is None:
            raise ArithmeticError("degree-%d component is not a Lie polynomial" % n)
        out += [(Fraction(c), w) for c, w in zip(x, words) if c != 0]
    return out


def bch_series(degree: int) -> List[Tuple[Fraction, Word]]:
    """Alias for :func:`bch_lie_terms`."""
    return bch_lie_terms(degree)


def format_word(word: Sequence[int], letters: Sequence[str] = ("X", "Y")) -> str:
    if len(word) == 1:
        return letters[word[0]]
    return "[%s,%s]" % (letters[word[0]], format_word(word[1:], letters))


def format_lie_terms(terms: Iterable[Tuple[Fraction, Word]], letters: Sequence[str] = ("X", "Y")) -> str:
    r"""Pretty-print, e.g. ``X + Y + 1/2 [X,Y] + 1/12 [X,[X,Y]] - ...``."""
    pieces: List[str] = []
    for c, w in terms:
        body = format_word(w, letters)
        sign = "-" if c < 0 else "+"
        a = abs(c)
        coef = "" if a == 1 else ("%s " % a)
        pieces.append("%s %s%s" % (sign, coef, body))
    if not pieces:
        return "0"
    s = " ".join(pieces)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]


def evaluate_terms(
    terms: Iterable[Tuple[Fraction, Word]],
    gens: Sequence[Any],
    bracket: Callable[[Any, Any], Any],
    add: Callable[[Any, Any], Any],
    scale: Callable[[Any, Any], Any],
    zero: Any,
) -> Any:
    """Evaluate a Lie series in any Lie algebra given by ``bracket``/``add``/``scale``.

    Nested brackets are memoised by suffix, so shared tails are computed once.
    """
    cache: Dict[Word, Any] = {}

    def val(w: Word) -> Any:
        if w not in cache:
            cache[w] = gens[w[0]] if len(w) == 1 else bracket(gens[w[0]], val(w[1:]))
        return cache[w]

    total = zero
    for c, w in terms:
        total = add(total, scale(c, val(w)))
    return total


def bch_matrices(A: Matrix, B: Matrix, degree: int) -> Matrix:
    r"""Truncated :math:`Z = \log(e^A e^B)` for square matrices, via nested commutators."""
    return evaluate_terms(
        bch_lie_terms(degree), [A, B], commutator, lambda u, v: u + v, lambda c, m: m * c, Matrix.zeros(A.nrows)
    )


def bch_in_algebra(algebra: Any, x: Sequence[Any], y: Sequence[Any], degree: int) -> List[Any]:
    """Truncated BCH product of two vectors of a :class:`~liebracket.algebra.LieAlgebra`, using its structure constants."""
    return evaluate_terms(
        bch_lie_terms(degree), [list(x), list(y)], algebra.bracket,
        lambda u, v: [a + b for a, b in zip(u, v)],
        lambda c, v: [c * a for a in v],
        [0] * algebra.dim,
    )


def exp_matrix(A: Matrix, terms: Optional[int] = None) -> Matrix:
    r""":math:`e^A = \sum_k A^k/k!`.  For nilpotent :math:`A` (default) the sum is exact and stops at :math:`A^k = 0`."""
    n = A.nrows
    result = Matrix.identity(n)
    power = Matrix.identity(n)
    limit = terms if terms is not None else n
    for k in range(1, limit + 1):
        power = (power @ A) / k
        if power.is_zero():
            break
        result = result + power
    return result


def log_unipotent(U: Matrix) -> Matrix:
    r""":math:`\log U = \sum_{k\ge1} (-1)^{k+1}(U - 1)^k/k`, exact when :math:`U - 1` is nilpotent."""
    n = U.nrows
    N = U - Matrix.identity(n)
    result = Matrix.zeros(n)
    power = Matrix.identity(n)
    for k in range(1, n + 2):
        power = power @ N
        if power.is_zero():
            return result
        if k > n:
            break
        result = result + power * Fraction((-1) ** (k + 1), k)
    raise ValueError("U - 1 is not nilpotent")
