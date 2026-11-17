r"""Exact scalar arithmetic.

Compact real forms such as :math:`\mathfrak{su}(2)` are most naturally written
with matrices over :math:`\mathbb{C}`, yet every structure constant we care about
is rational.  The :class:`GaussianRational` type implements the field
:math:`\mathbb{Q}(i) = \{a + b i : a, b \in \mathbb{Q}\}` exactly, so that
brackets, traces and Killing forms of the defining representations can be
computed without floating point error.

The helpers :func:`is_zero`, :func:`to_exact` and :func:`format_scalar` work
uniformly over ``int``, :class:`fractions.Fraction`, :class:`GaussianRational`
and (for interoperability) ``float``/``complex``.
"""
from __future__ import annotations

from fractions import Fraction
from numbers import Rational
from typing import Any, Union

__all__ = [
    "GaussianRational",
    "I",
    "Scalar",
    "is_zero",
    "to_exact",
    "format_scalar",
    "as_fraction",
    "FLOAT_TOLERANCE",
]

#: Absolute tolerance used when inexact (float / complex) scalars are tested for zero.
FLOAT_TOLERANCE = 1e-10


class GaussianRational:
    r"""An element :math:`a + b i` of the Gaussian rationals :math:`\mathbb{Q}(i)`.

    Instances are immutable and hashable.  Arithmetic with ``int`` and
    :class:`~fractions.Fraction` promotes automatically; an element whose
    imaginary part vanishes compares equal to the corresponding rational.

    >>> z = GaussianRational(1, 2)
    >>> z * z.conjugate()
    GaussianRational(5, 0)
    >>> z * z.conjugate() == 5
    True
    """

    __slots__ = ("_re", "_im")

    def __init__(self, re: Any = 0, im: Any = 0) -> None:
        self._re = Fraction(re)
        self._im = Fraction(im)

    # -- accessors --------------------------------------------------------
    @property
    def real(self) -> Fraction:
        """Real part :math:`\\operatorname{Re} z`."""
        return self._re

    @property
    def imag(self) -> Fraction:
        """Imaginary part :math:`\\operatorname{Im} z`."""
        return self._im

    def conjugate(self) -> "GaussianRational":
        r"""Complex conjugate :math:`\bar z = a - b i`."""
        return GaussianRational(self._re, -self._im)

    def norm(self) -> Fraction:
        r"""Field norm :math:`N(z) = z \bar z = a^2 + b^2`."""
        return self._re * self._re + self._im * self._im

    def is_real(self) -> bool:
        return self._im == 0

    # -- coercion ---------------------------------------------------------
    @staticmethod
    def _coerce(other: Any) -> "GaussianRational":
        if isinstance(other, GaussianRational):
            return other
        if isinstance(other, (int, Fraction, Rational)):
            return GaussianRational(other, 0)
        raise TypeError("cannot coerce %r to GaussianRational" % (other,))

    # -- arithmetic -------------------------------------------------------
    def __add__(self, other: Any) -> "GaussianRational":
        try:
            o = self._coerce(other)
        except TypeError:
            return NotImplemented
        return GaussianRational(self._re + o._re, self._im + o._im)

    __radd__ = __add__

    def __neg__(self) -> "GaussianRational":
        return GaussianRational(-self._re, -self._im)

    def __pos__(self) -> "GaussianRational":
        return self

    def __sub__(self, other: Any) -> "GaussianRational":
        try:
            o = self._coerce(other)
        except TypeError:
            return NotImplemented
        return GaussianRational(self._re - o._re, self._im - o._im)

    def __rsub__(self, other: Any) -> "GaussianRational":
        try:
            o = self._coerce(other)
        except TypeError:
            return NotImplemented
        return o - self

    def __mul__(self, other: Any) -> "GaussianRational":
        try:
            o = self._coerce(other)
        except TypeError:
            return NotImplemented
        return GaussianRational(
            self._re * o._re - self._im * o._im,
            self._re * o._im + self._im * o._re,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "GaussianRational":
        try:
            o = self._coerce(other)
        except TypeError:
            return NotImplemented
        n = o.norm()
        if n == 0:
            raise ZeroDivisionError("division by zero in Q(i)")
        num = self * o.conjugate()
        return GaussianRational(num._re / n, num._im / n)

    def __rtruediv__(self, other: Any) -> "GaussianRational":
        try:
            o = self._coerce(other)
        except TypeError:
            return NotImplemented
        return o / self

    def __pow__(self, k: int) -> "GaussianRational":
        if not isinstance(k, int):
            return NotImplemented
        if k < 0:
            return GaussianRational(1) / (self ** (-k))
        result = GaussianRational(1)
        base = self
        while k:
            if k & 1:
                result = result * base
            base = base * base
            k >>= 1
        return result

    # -- comparison -------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GaussianRational):
            return self._re == other._re and self._im == other._im
        if isinstance(other, (int, Fraction)):
            return self._im == 0 and self._re == other
        if isinstance(other, complex):
            return complex(self) == other
        if isinstance(other, float):
            return self._im == 0 and float(self._re) == other
        return NotImplemented

    def __hash__(self) -> int:
        if self._im == 0:
            return hash(self._re)
        return hash((self._re, self._im))

    def __bool__(self) -> bool:
        return self._re != 0 or self._im != 0

    def __complex__(self) -> complex:
        return complex(float(self._re), float(self._im))

    def __repr__(self) -> str:
        return "GaussianRational(%s, %s)" % (self._re, self._im)

    def __str__(self) -> str:
        return format_scalar(self)


#: The imaginary unit :math:`i \in \mathbb{Q}(i)`.
I = GaussianRational(0, 1)

Scalar = Union[int, Fraction, GaussianRational, float, complex]


def is_zero(x: Any, tol: float = FLOAT_TOLERANCE) -> bool:
    """Return ``True`` if ``x`` is zero (exactly, or within ``tol`` for inexact types)."""
    if isinstance(x, (float, complex)):
        return abs(x) <= tol
    return x == 0


def to_exact(x: Any) -> Any:
    """Normalise a scalar to the simplest exact type representing it.

    Gaussian rationals with vanishing imaginary part become :class:`Fraction`,
    and integral fractions become ``int``.  Inexact values are returned unchanged.
    """
    if isinstance(x, GaussianRational):
        if x.imag != 0:
            return x
        x = x.real
    if isinstance(x, Fraction):
        return int(x) if x.denominator == 1 else x
    return x


def as_fraction(x: Any) -> Fraction:
    """Convert an exact real scalar to :class:`Fraction`, raising if it is not real."""
    x = to_exact(x)
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    raise ValueError("%r is not an exact rational number" % (x,))


def _fmt_rational(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else "%d/%d" % (q.numerator, q.denominator)


def format_scalar(x: Any) -> str:
    """Human-readable form of a scalar, e.g. ``'1/2'``, ``'-3i'``, ``'1/2 + i/3'``."""
    x = to_exact(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, Fraction):
        return _fmt_rational(x)
    if isinstance(x, GaussianRational):
        re, im = x.real, x.imag
        if im == 1:
            ims = "i"
        elif im == -1:
            ims = "-i"
        elif im.denominator == 1:
            ims = "%di" % im.numerator
        else:
            num = {1: "", -1: "-"}.get(im.numerator, str(im.numerator))
            ims = "%si/%d" % (num, im.denominator)
        if re == 0:
            return ims
        sign = " - " if im < 0 else " + "
        return _fmt_rational(re) + sign + ims.lstrip("-")
    if isinstance(x, float):
        return "%.6g" % x
    if isinstance(x, complex):
        return "(%.6g%+.6gj)" % (x.real, x.imag)
    return str(x)
