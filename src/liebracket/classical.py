r"""The classical Lie algebras and a few standard non-semisimple examples.

Every constructor returns a :class:`~liebracket.algebra.LieAlgebra` whose
structure constants were *computed* from an explicit matrix basis (stored in
``algebra.matrices``), so the defining representation is always available and
every structure constant is exact.

=====================  ======================  ================================
constructor            algebra                 basis
=====================  ======================  ================================
:func:`sl2`            :math:`\mathfrak{sl}_2`  Chevalley :math:`e, f, h`
:func:`su2`            :math:`\mathfrak{su}(2)` :math:`-\tfrac{i}{2}\sigma_k`
:func:`so3`            :math:`\mathfrak{so}(3)` :math:`(L_i)_{jk} = -\varepsilon_{ijk}`
:func:`su3`            :math:`\mathfrak{su}(3)` rescaled Gell-Mann
:func:`sl`             :math:`\mathfrak{sl}_n`  :math:`H_i, E_{ij}`
:func:`so`             :math:`\mathfrak{so}_n`  :math:`E_{ab} - E_{ba}`
:func:`sp`             :math:`\mathfrak{sp}_{2n}` block form
:func:`gl`             :math:`\mathfrak{gl}_n`  matrix units
:func:`heisenberg`     :math:`\mathfrak{h}_{2n+1}` :math:`x_i, p_i, z`
:func:`nilpotent_upper`:math:`\mathfrak n_n`   :math:`E_{ij},\ i<j`
:func:`borel`          :math:`\mathfrak b_n`   :math:`E_{ij},\ i\le j`
=====================  ======================  ================================
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Tuple

from .algebra import LieAlgebra
from .linalg import Matrix
from .scalars import I

__all__ = [
    "pauli_matrices",
    "gell_mann_matrices",
    "GELL_MANN_F",
    "levi_civita",
    "sl2",
    "su2",
    "so3",
    "su3",
    "sl",
    "so",
    "sp",
    "gl",
    "heisenberg",
    "nilpotent_upper",
    "borel",
    "by_name",
]

HALF = Fraction(1, 2)


def levi_civita(i: int, j: int, k: int) -> int:
    r"""The totally antisymmetric symbol :math:`\varepsilon_{ijk}` on indices :math:`\{0,1,2\}`."""
    if len({i, j, k}) < 3:
        return 0
    return 1 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1


def pauli_matrices() -> List[Matrix]:
    r"""The Pauli matrices :math:`\sigma_1, \sigma_2, \sigma_3` over :math:`\mathbb{Q}(i)`."""
    return [
        Matrix([[0, 1], [1, 0]]),
        Matrix([[0, -I], [I, 0]]),
        Matrix([[1, 0], [0, -1]]),
    ]


def gell_mann_matrices(rational: bool = True) -> List[Matrix]:
    r"""The Gell-Mann matrices :math:`\lambda_1, \dots, \lambda_8`.

    With ``rational=True`` the eighth matrix is replaced by
    :math:`\sqrt 3\,\lambda_8 = \operatorname{diag}(1, 1, -2)`, keeping every entry
    in :math:`\mathbb{Q}(i)`.  With ``rational=False`` the usual normalisation
    :math:`\operatorname{tr}(\lambda_a\lambda_b) = 2\delta_{ab}` is used and
    entries are Python ``complex``.
    """
    z = 0
    lam = [
        [[z, 1, z], [1, z, z], [z, z, z]],
        [[z, -I, z], [I, z, z], [z, z, z]],
        [[1, z, z], [z, -1, z], [z, z, z]],
        [[z, z, 1], [z, z, z], [1, z, z]],
        [[z, z, -I], [z, z, z], [I, z, z]],
        [[z, z, z], [z, z, 1], [z, 1, z]],
        [[z, z, z], [z, z, -I], [z, I, z]],
        [[1, z, z], [z, 1, z], [z, z, -2]],
    ]
    mats = [Matrix(m) for m in lam]
    if rational:
        return mats
    out = [m.map(complex) for m in mats[:7]]
    out.append(mats[7].map(lambda x: complex(x) / math.sqrt(3)))
    return out


_R3 = math.sqrt(3) / 2
#: Totally antisymmetric :math:`\mathfrak{su}(3)` structure constants
#: :math:`[\lambda_a/2, \lambda_b/2] = i f_{abc}\, \lambda_c/2` (1-based indices).
GELL_MANN_F: Dict[Tuple[int, int, int], float] = {
    (1, 2, 3): 1.0,
    (1, 4, 7): 0.5,
    (1, 5, 6): -0.5,
    (2, 4, 6): 0.5,
    (2, 5, 7): 0.5,
    (3, 4, 5): 0.5,
    (3, 6, 7): -0.5,
    (4, 5, 8): _R3,
    (6, 7, 8): _R3,
}


def sl2() -> LieAlgebra:
    r""":math:`\mathfrak{sl}_2` in the Chevalley basis: :math:`[h,e]=2e,\ [h,f]=-2f,\ [e,f]=h`."""
    mats = [Matrix([[0, 1], [0, 0]]), Matrix([[0, 0], [1, 0]]), Matrix([[1, 0], [0, -1]])]
    return LieAlgebra.from_matrices(mats, names=["e", "f", "h"], name="sl2")


def su2() -> LieAlgebra:
    r""":math:`\mathfrak{su}(2)` with basis :math:`e_k = -\tfrac{i}{2}\sigma_k`, so that
    :math:`[e_i, e_j] = \varepsilon_{ijk} e_k`."""
    mats = [p * (-I * HALF) for p in pauli_matrices()]
    return LieAlgebra.from_matrices(mats, names=["e1", "e2", "e3"], name="su2")


def so3() -> LieAlgebra:
    r""":math:`\mathfrak{so}(3)` with :math:`(L_i)_{jk} = -\varepsilon_{ijk}`, so that
    :math:`[L_i, L_j] = \varepsilon_{ijk} L_k`."""
    mats = [Matrix.from_function(3, 3, lambda j, k, i=i: -levi_civita(i, j, k)) for i in range(3)]
    return LieAlgebra.from_matrices(mats, names=["L1", "L2", "L3"], name="so3")


def su3() -> LieAlgebra:
    r""":math:`\mathfrak{su}(3)` with :math:`T_a = -\tfrac{i}{2}\lambda_a` (:math:`a \le 7`) and
    :math:`T_8 = -\tfrac{i}{2}\sqrt3\,\lambda_8`; all structure constants are rational."""
    mats = [m * (-I * HALF) for m in gell_mann_matrices(rational=True)]
    return LieAlgebra.from_matrices(mats, names=["T%d" % a for a in range(1, 9)], name="su3")


def sl(n: int) -> LieAlgebra:
    r""":math:`\mathfrak{sl}_n`: Cartan elements :math:`H_i = E_{ii} - E_{i+1,i+1}` first, then
    :math:`E_{ij}` for :math:`i \neq j` (1-based names)."""
    if n < 2:
        raise ValueError("sl(n) needs n >= 2")
    mats, names = [], []
    for i in range(n - 1):
        mats.append(Matrix.unit(n, i, i) - Matrix.unit(n, i + 1, i + 1))
        names.append("H%d" % (i + 1))
    for i in range(n):
        for j in range(n):
            if i != j:
                mats.append(Matrix.unit(n, i, j))
                names.append("E%d%d" % (i + 1, j + 1))
    return LieAlgebra.from_matrices(mats, names=names, name="sl%d" % n)


def gl(n: int) -> LieAlgebra:
    r""":math:`\mathfrak{gl}_n` with the matrix units :math:`E_{ij}` (reductive, not semisimple)."""
    mats = [Matrix.unit(n, i, j) for i in range(n) for j in range(n)]
    names = ["E%d%d" % (i + 1, j + 1) for i in range(n) for j in range(n)]
    return LieAlgebra.from_matrices(mats, names=names, name="gl%d" % n)


def so(n: int) -> LieAlgebra:
    r""":math:`\mathfrak{so}_n = \{X : X^{\mathsf T} = -X\}` with basis :math:`M_{ab} = E_{ab} - E_{ba}`, :math:`a<b`."""
    if n < 2:
        raise ValueError("so(n) needs n >= 2")
    mats, names = [], []
    for a in range(n):
        for b in range(a + 1, n):
            mats.append(Matrix.unit(n, a, b) - Matrix.unit(n, b, a))
            names.append("M%d%d" % (a + 1, b + 1))
    return LieAlgebra.from_matrices(mats, names=names, name="so%d" % n)


def sp(n: int) -> LieAlgebra:
    r""":math:`\mathfrak{sp}_{2n} = \{X : X^{\mathsf T}J + JX = 0\}`, :math:`J = \begin{pmatrix}0&I\\-I&0\end{pmatrix}`.

    Basis: :math:`\begin{pmatrix}A&0\\0&-A^{\mathsf T}\end{pmatrix}` for matrix units :math:`A`,
    and :math:`\begin{pmatrix}0&S\\0&0\end{pmatrix}`, :math:`\begin{pmatrix}0&0\\S&0\end{pmatrix}`
    for elementary symmetric :math:`S`; :math:`\dim = n(2n+1)`.
    """
    if n < 1:
        raise ValueError("sp(2n) needs n >= 1")
    N = 2 * n
    mats, names = [], []
    for i in range(n):
        for j in range(n):
            mats.append(Matrix.unit(N, i, j) - Matrix.unit(N, n + j, n + i))
            names.append("A%d%d" % (i + 1, j + 1))
    for i in range(n):
        for j in range(i, n):
            sym = Matrix.unit(N, i, n + j) + (Matrix.unit(N, j, n + i) if i != j else Matrix.zeros(N))
            mats.append(sym)
            names.append("B%d%d" % (i + 1, j + 1))
            mats.append(sym.T)
            names.append("C%d%d" % (i + 1, j + 1))
    return LieAlgebra.from_matrices(mats, names=names, name="sp%d" % N)


def heisenberg(n: int = 1) -> LieAlgebra:
    r"""The Heisenberg algebra :math:`\mathfrak h_{2n+1}`: :math:`[x_i, p_j] = \delta_{ij} z`, :math:`z` central.

    Realised by strictly upper-triangular :math:`(n+2)\times(n+2)` matrices, hence nilpotent of step 2.
    """
    N = n + 2
    mats = [Matrix.unit(N, 0, i + 1) for i in range(n)]
    mats += [Matrix.unit(N, i + 1, N - 1) for i in range(n)]
    mats.append(Matrix.unit(N, 0, N - 1))
    names = ["x%d" % (i + 1) for i in range(n)] + ["p%d" % (i + 1) for i in range(n)] + ["z"]
    if n == 1:
        names = ["x", "p", "z"]
    return LieAlgebra.from_matrices(mats, names=names, name="heis%d" % (2 * n + 1))


def nilpotent_upper(n: int) -> LieAlgebra:
    r"""Strictly upper-triangular :math:`n \times n` matrices :math:`\mathfrak n_n` (nilpotent)."""
    mats = [Matrix.unit(n, i, j) for i in range(n) for j in range(i + 1, n)]
    names = ["E%d%d" % (i + 1, j + 1) for i in range(n) for j in range(i + 1, n)]
    return LieAlgebra.from_matrices(mats, names=names, name="n%d" % n)


def borel(n: int) -> LieAlgebra:
    r"""Upper-triangular :math:`n \times n` matrices :math:`\mathfrak b_n` (solvable, not nilpotent)."""
    mats = [Matrix.unit(n, i, j) for i in range(n) for j in range(i, n)]
    names = ["E%d%d" % (i + 1, j + 1) for i in range(n) for j in range(i, n)]
    return LieAlgebra.from_matrices(mats, names=names, name="b%d" % n)


def by_name(spec: str) -> LieAlgebra:
    """Parse names like ``'su2'``, ``'so3'``, ``'su3'``, ``'sl4'``, ``'so5'``, ``'sp4'``, ``'gl2'``, ``'heis3'``."""
    s = spec.strip().lower().replace("(", "").replace(")", "").replace("_", "")
    fixed = {"sl2": sl2, "su2": su2, "so3": so3, "su3": su3}
    if s in fixed:
        return fixed[s]()
    for prefix, fn in (("heis", None), ("sl", sl), ("so", so), ("sp", sp), ("gl", gl), ("n", nilpotent_upper), ("b", borel)):
        if s.startswith(prefix) and s[len(prefix):].isdigit():
            k = int(s[len(prefix):])
            if prefix == "heis":
                if k % 2 == 0:
                    raise ValueError("Heisenberg algebras have odd dimension")
                return heisenberg((k - 1) // 2)
            if prefix == "sp":
                if k % 2:
                    raise ValueError("sp(N) needs even N")
                return sp(k // 2)
            assert fn is not None
            return fn(k)
    raise ValueError("unknown algebra %r" % spec)
