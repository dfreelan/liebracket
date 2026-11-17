import cmath
from fractions import Fraction
from itertools import product

import pytest

from liebracket.classical import (
    GELL_MANN_F,
    by_name,
    gell_mann_matrices,
    levi_civita,
    pauli_matrices,
    sl,
    so,
    sp,
    su2,
    su3,
)
from liebracket.linalg import Matrix, commutator
from liebracket.scalars import I


def test_levi_civita():
    assert levi_civita(0, 1, 2) == 1 and levi_civita(2, 1, 0) == -1 and levi_civita(0, 0, 1) == 0


def test_pauli_algebra():
    s = pauli_matrices()
    for a, b in product(range(3), repeat=2):
        expected = Matrix.identity(2) * (1 if a == b else 0)
        for c in range(3):
            expected = expected + s[c] * (I * levi_civita(a, b, c))
        assert s[a] @ s[b] == expected


def test_su2_structure_constants_are_epsilon():
    g = su2()
    for i, j, k in product(range(3), repeat=3):
        assert g.structure_constant(i, j, k) == levi_civita(i, j, k)


def test_su2_basis_is_skew_hermitian_traceless():
    for m in su2().matrices:
        assert m.dagger() == -m and m.trace() == 0


@pytest.mark.parametrize("n,dim", [(2, 3), (3, 8), (4, 15), (5, 24)])
def test_sl_dimensions(n, dim):
    g = sl(n)
    assert g.dim == dim
    assert all(m.trace() == 0 for m in g.matrices)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_so_dimensions_and_antisymmetry(n):
    g = so(n)
    assert g.dim == n * (n - 1) // 2
    assert all(m.is_antisymmetric() for m in g.matrices)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_sp_preserves_symplectic_form(n):
    g = sp(n)
    assert g.dim == n * (2 * n + 1)
    Z, Id = Matrix.zeros(n), Matrix.identity(n)
    J = Matrix([Z.row(i) + Id.row(i) for i in range(n)] + [(-Id).row(i) + Z.row(i) for i in range(n)])
    for X in g.matrices:
        assert (X.T @ J + J @ X).is_zero()


def test_su3_rational_structure_constants():
    g = su3()
    for c in g.structure_constants().values():
        assert isinstance(c, (int, Fraction))
    assert g.structure_constant(0, 1, 2) == 1  # [T1, T2] = T3
    assert g.structure_constant(3, 4, 7) == Fraction(1, 2)  # [T4, T5] contains T8' / 2


def test_gell_mann_float_table():
    lam = gell_mann_matrices(rational=False)
    for a, b in product(range(8), repeat=2):
        comm = commutator(lam[a], lam[b])
        for c in range(8):
            # f_abc = -(i/4) tr([lambda_a, lambda_b] lambda_c)
            f = (-0.25j * (comm @ lam[c]).trace())
            key = tuple(sorted((a + 1, b + 1, c + 1)))
            perm = (a + 1, b + 1, c + 1)
            expected = 0.0
            if key in GELL_MANN_F and len(set(perm)) == 3:
                sign = _perm_sign(perm, key)
                expected = sign * GELL_MANN_F[key]
            assert cmath.isclose(f, expected, abs_tol=1e-12)


def _perm_sign(perm, sorted_key):
    idx = [sorted_key.index(p) for p in perm]
    inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if idx[i] > idx[j])
    return -1 if inversions % 2 else 1


def test_gell_mann_normalisation():
    lam = gell_mann_matrices(rational=False)
    for a, b in product(range(8), repeat=2):
        assert cmath.isclose((lam[a] @ lam[b]).trace(), 2.0 if a == b else 0.0, abs_tol=1e-12)


def test_by_name():
    assert by_name("su(2)").dim == 3
    assert by_name("sl_4").dim == 15
    assert by_name("sp4").dim == 10
    assert by_name("heis5").dim == 5
    assert by_name("n4").dim == 6
    assert by_name("b3").dim == 6
    for bad in ("sp3", "heis4", "foo"):
        with pytest.raises(ValueError):
            by_name(bad)


def test_low_rank_isomorphisms_by_dimension_and_killing_rank():
    # so(5) ~ sp(4), so(6) ~ sl(4): same dimension, both semisimple
    assert so(5).dim == sp(2).dim and so(6).dim == sl(4).dim
    assert so(5).is_semisimple() and sp(2).is_semisimple()


def test_invalid_sizes():
    for fn, arg in ((sl, 1), (so, 1), (sp, 0)):
        with pytest.raises(ValueError):
            fn(arg)
