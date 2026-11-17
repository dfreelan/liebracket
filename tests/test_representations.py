from fractions import Fraction

import pytest

from liebracket.classical import gl, heisenberg, sl, sl2, so, so3, sp, su2, su3
from liebracket.linalg import Matrix
from liebracket.representations import (
    Representation,
    adjoint_representation,
    casimir_eigenspaces,
    defining_representation,
    sl2_irrep,
    su2_spin,
    trivial_representation,
)


@pytest.mark.parametrize("make", [su2, so3, sl2, su3, lambda: sl(4), lambda: so(5), lambda: sp(2), lambda: heisenberg(1)])
def test_adjoint_is_homomorphism(make):
    g = make()
    assert adjoint_representation(g).is_homomorphism()


@pytest.mark.parametrize("make", [su2, su3, lambda: sl(3), lambda: so(4), lambda: sp(3), lambda: gl(2)])
def test_defining_is_faithful_homomorphism(make):
    rho = defining_representation(make())
    assert rho.is_homomorphism() and rho.is_faithful()


def test_defining_requires_matrices():
    from liebracket.algebra import LieAlgebra
    with pytest.raises(ValueError):
        defining_representation(LieAlgebra(1, {}))


def test_bad_representation_detected():
    g = sl2()
    E, F, H = Matrix([[0, 1], [0, 0]]), Matrix([[0, 0], [1, 0]]), Matrix([[1, 0], [0, -1]])
    rho = Representation(g, [E, F, H * 2])
    assert not rho.is_homomorphism()
    assert rho.homomorphism_defects()
    with pytest.raises(ValueError):
        Representation(g, [E, F])


@pytest.mark.parametrize("n", range(0, 6))
def test_sl2_irreps(n):
    rho = sl2_irrep(n)
    assert rho.dim == n + 1
    assert rho.is_homomorphism()
    assert rho.is_irreducible()
    # Killing-normalised Casimir acts by n(n+2)/8
    assert rho.casimir() == Matrix.identity(n + 1) * Fraction(n * (n + 2), 8)


@pytest.mark.parametrize("twice_j", range(0, 6))
def test_su2_spin(twice_j):
    j = Fraction(twice_j, 2)
    rho = su2_spin(j)
    assert rho.dim == twice_j + 1
    assert rho.is_homomorphism()
    total = sum((m @ m for m in rho.matrices[1:]), rho.matrices[0] @ rho.matrices[0])
    assert total == Matrix.identity(rho.dim) * (-j * (j + 1))


def test_spin_half_is_unitary_and_equals_defining():
    rho = su2_spin(Fraction(1, 2))
    assert rho.is_skew_hermitian()
    assert rho.matrices == su2().matrices


def test_spin_one_equivalent_to_adjoint():
    g = su2()
    ad = adjoint_representation(g)
    spin1 = su2_spin(1, g)
    assert ad.casimir() == spin1.casimir()
    assert ad.is_irreducible() and spin1.is_irreducible()


def test_invalid_spin():
    with pytest.raises(ValueError):
        su2_spin(Fraction(1, 3))
    with pytest.raises(ValueError):
        sl2_irrep(-1)


def test_casimir_commutes_with_action():
    for rho in (adjoint_representation(su3()), defining_representation(so(5)), sl2_irrep(3)):
        C = rho.casimir()
        for m in rho.matrices:
            assert C @ m == m @ C


def test_tensor_product_homomorphism_and_clebsch_gordan():
    rep = sl2_irrep(1).tensor(sl2_irrep(2))
    assert rep.dim == 6 and rep.is_homomorphism()
    spaces = casimir_eigenspaces(rep)
    assert spaces == {Fraction(15, 8): 4, Fraction(3, 8): 2}
    rep22 = sl2_irrep(2) @ sl2_irrep(2)
    assert casimir_eigenspaces(rep22) == {Fraction(0): 1, Fraction(1): 3, Fraction(3): 5}
    assert rep22.commutant_dimension() == 3


def test_direct_sum_and_dual():
    a, b = sl2_irrep(1), sl2_irrep(2)
    s = a.direct_sum(b)
    assert s.dim == 5 and s.is_homomorphism() and not s.is_irreducible()
    assert s.commutant_dimension() == 2
    d = defining_representation(sl(3)).dual()
    assert d.is_homomorphism()
    assert d.weights([0, 1]) == [(1, -1), (0, 1), (-1, 0)]


def test_dual_of_sl2_is_self_dual_via_intertwiner():
    rho = sl2_irrep(1)
    dual = rho.dual()
    T = Matrix([[0, 1], [-1, 0]])
    assert rho.intertwines(dual, T) and dual.intertwines(rho, T)
    assert not rho.intertwines(dual, Matrix.identity(2))


def test_conjugated_representation():
    rho = sl2_irrep(2)
    P = Matrix([[1, 1, 0], [0, 1, 2], [0, 0, 1]])
    sigma = rho.conjugated(P)
    assert sigma.is_homomorphism()
    assert sigma.intertwines(rho, P)


def test_trace_form_and_dynkin_index():
    for g in (su3(), sl(4)):
        assert g.is_invariant_form(defining_representation(g).trace_form())
    # su(n): tr_fund / Killing = 1/(2n)
    assert defining_representation(su3()).dynkin_index() == Fraction(1, 6)
    assert defining_representation(sl(4)).dynkin_index() == Fraction(1, 8)
    assert adjoint_representation(su3()).dynkin_index() == 1


def test_weights_of_sl3_defining_and_adjoint():
    g = sl(3)
    assert defining_representation(g).weights([0, 1]) == [(1, 0), (0, -1), (-1, 1)]
    ad_weights = adjoint_representation(g).weights([0, 1])
    assert ad_weights.count((0, 0)) == 2 and len(ad_weights) == 8
    assert adjoint_representation(su2()).weights([2]) is None


def test_trivial_representation():
    rho = trivial_representation(su3(), 2)
    assert rho.is_homomorphism() and rho.dim == 2 and not rho.is_faithful()


def test_call_linear():
    rho = sl2_irrep(2)
    x = [1, 2, Fraction(1, 2)]
    assert rho(x) == rho.matrices[0] + rho.matrices[1] * 2 + rho.matrices[2] * Fraction(1, 2)
