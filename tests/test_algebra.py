from fractions import Fraction
from itertools import product

import pytest

from liebracket.algebra import LieAlgebra
from liebracket.classical import borel, gl, heisenberg, nilpotent_upper, sl, sl2, so, so3, sp, su2, su3
from liebracket.linalg import Matrix


def eps_algebra():
    br = {(0, 1): {2: 1}, (1, 2): {0: 1}, (2, 0): {1: 1}}
    return LieAlgebra(3, br, names=["a", "b", "c"], name="eps")


class TestConstruction:
    def test_antisymmetry_filled_in(self):
        g = eps_algebra()
        assert g.structure_constant(1, 0, 2) == -1
        assert g.basis_bracket(2, 1) == [-1, 0, 0]

    def test_bad_jacobi_rejected(self):
        # [a,b]=a, [b,c]=a, [a,c]=b violates Jacobi
        br = {(0, 1): {0: 1}, (1, 2): {0: 1}, (0, 2): {1: 1}}
        with pytest.raises(ValueError, match="Jacobi"):
            LieAlgebra(3, br)
        g = LieAlgebra(3, br, check=False)
        assert g.jacobi_violations()
        assert not g.satisfies_jacobi()

    def test_inconsistent_antisymmetry_rejected(self):
        with pytest.raises(ValueError):
            LieAlgebra(2, {(0, 1): {0: 1}, (1, 0): {0: 1}})

    def test_self_bracket_must_vanish(self):
        with pytest.raises(ValueError):
            LieAlgebra(1, {(0, 0): {0: 1}})

    def test_dense_constructor(self):
        g = eps_algebra()
        n = 3
        c = [[[g.structure_constant(i, j, k) for k in range(n)] for j in range(n)] for i in range(n)]
        h = LieAlgebra.from_structure_constants(c)
        assert h.structure_constants() == g.structure_constants()

    def test_from_matrices_requires_closure(self):
        with pytest.raises(ValueError, match="closed"):
            LieAlgebra.from_matrices([Matrix.unit(2, 0, 1), Matrix.unit(2, 1, 0)])
        with pytest.raises(ValueError, match="dependent"):
            LieAlgebra.from_matrices([Matrix.unit(2, 0, 1), Matrix.unit(2, 0, 1) * 2])

    def test_element_helpers(self):
        g = sl2()
        x = g.element(e=1, h=Fraction(1, 2))
        assert x == [1, 0, Fraction(1, 2)]
        assert g.format_element(x) == "e + (1/2) h"
        with pytest.raises(KeyError):
            g.index("nope")


class TestBracketAxioms:
    @pytest.mark.parametrize("make", [su2, so3, sl2, su3, lambda: sl(3), lambda: so(4), lambda: sp(2), lambda: gl(2),
                                      lambda: heisenberg(2), lambda: borel(3)])
    def test_jacobi_and_antisymmetry_on_random_elements(self, make):
        g = make()
        assert g.satisfies_jacobi()
        n = g.dim
        xs = [[Fraction((3 * i + 7 * s) % 5 - 2, 1 + s) for i in range(n)] for s in range(3)]
        x, y, z = xs
        assert g.bracket(x, y) == [-c for c in g.bracket(y, x)]
        assert all(c == 0 for c in g.jacobiator(x, y, z))
        # bilinearity
        two_x = [2 * a for a in x]
        assert g.bracket(two_x, y) == [2 * c for c in g.bracket(x, y)]

    def test_ad_is_derivation(self):
        g = su3()
        x, y = g.basis_vector(3), g.basis_vector(7)
        assert g.ad(g.bracket(x, y)) == g.ad(x) @ g.ad(y) - g.ad(y) @ g.ad(x)

    def test_ad_basis_matches_ad(self):
        g = sl(3)
        for i in range(g.dim):
            assert g.ad_basis(i) == g.ad(g.basis_vector(i))


class TestKilling:
    def test_su2_so3(self):
        for g in (su2(), so3()):
            assert g.killing_form() == Matrix.identity(3) * -2

    def test_sl2_values(self):
        g = sl2()
        e, f, h = (g.index(n) for n in "efh")
        K = g.killing_form()
        assert K[h, h] == 8 and K[e, f] == 4 and K[e, e] == 0

    def test_su3_values(self):
        K = su3().killing_form()
        assert K == Matrix.diag([-3] * 7 + [-9])

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_sl_n_killing_is_2n_trace(self, n):
        g = sl(n)
        mats = g.matrices
        K = g.killing_form()
        for i, j in product(range(g.dim), repeat=2):
            assert K[i, j] == 2 * n * (mats[i] @ mats[j]).trace()

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_so_n_killing_is_n_minus_2_trace(self, n):
        g = so(n)
        K = g.killing_form()
        mats = g.matrices
        for i, j in product(range(g.dim), repeat=2):
            assert K[i, j] == (n - 2) * (mats[i] @ mats[j]).trace()

    def test_invariance(self):
        for g in (su3(), sp(2), so(5)):
            assert g.is_invariant_form(g.killing_form())

    def test_killing_bilinear_scalar(self):
        g = sl2()
        assert g.killing(g.element(h=1), g.element(h=1)) == 8

    @pytest.mark.parametrize("make", [su2, su3, lambda: sl(4), lambda: so(5), lambda: sp(2)])
    def test_casimir_in_adjoint_is_identity(self, make):
        g = make()
        assert g.casimir_in_adjoint() == Matrix.identity(g.dim)


class TestStructureTheory:
    def test_semisimple(self):
        for g in (su2(), su3(), sl(4), so(5), so(6), sp(3)):
            assert g.is_semisimple()
        for g in (gl(2), heisenberg(1), borel(2), nilpotent_upper(3)):
            assert not g.is_semisimple()

    def test_heisenberg(self):
        h = heisenberg(2)
        assert h.dim == 5
        assert h.is_nilpotent() and h.is_solvable()
        assert len(h.center()) == 1
        assert h.lower_central_series() == [5, 1, 0]

    def test_nilpotent_upper(self):
        n4 = nilpotent_upper(4)
        assert n4.lower_central_series() == [6, 3, 1, 0]
        assert n4.is_nilpotent()

    def test_borel(self):
        b = borel(3)
        assert b.is_solvable() and not b.is_nilpotent()
        assert b.derived_series() == [6, 3, 1, 0]

    def test_gl_centre(self):
        g = gl(3)
        Z = g.center()
        assert len(Z) == 1
        # centre is spanned by the identity matrix E11 + E22 + E33
        z = Z[0]
        idx = [g.index("E%d%d" % (k, k)) for k in (1, 2, 3)]
        assert len({z[i] for i in idx}) == 1 and z[idx[0]] != 0
        assert g.derived_series() == [9, 8]

    def test_simple_algebras_are_perfect(self):
        for g in (su3(), so(5)):
            assert g.derived_series() == [g.dim]
            assert g.center() == []

    def test_abelian(self):
        g = LieAlgebra(2, {})
        assert g.is_abelian() and g.is_nilpotent()
        assert not su2().is_abelian()


class TestHomomorphisms:
    def test_su2_iso_so3(self):
        assert su2().is_isomorphism(so3(), Matrix.identity(3))

    def test_su2_to_sl2_complexification(self):
        # e_k -> -(i/2) sigma_k written in the (e, f, h) basis of sl2
        from liebracket.scalars import I
        half = Fraction(1, 2)
        phi = Matrix([[-I * half, -half, 0], [-I * half, half, 0], [0, 0, -I * half]])
        assert su2().is_homomorphism(sl2(), phi)
        assert su2().is_isomorphism(sl2(), phi)

    def test_non_homomorphism_detected(self):
        assert not su2().is_homomorphism(so3(), Matrix.diag([1, 1, 2]))
        with pytest.raises(ValueError):
            su2().is_homomorphism(so3(), Matrix.identity(2))

    def test_direct_sum(self):
        g = su2().direct_sum(sl2())
        assert g.dim == 6 and g.satisfies_jacobi() and g.is_semisimple()
        assert all(g.structure_constant(i, j, k) == 0 for i in range(3) for j in range(3, 6) for k in range(6))

    def test_rescaled_basis_is_isomorphic(self):
        g = sl2()
        s = [2, 3, Fraction(1, 5)]
        h = g.rescaled(s)
        assert h.satisfies_jacobi()
        assert h.is_isomorphism(g, Matrix.diag(s))

    def test_summary_and_table(self):
        text = su2().summary()
        assert "Jacobi identity: ok" in text and "semisimple" in text
        assert "[e1, e2] = e3" in su2().bracket_table()
