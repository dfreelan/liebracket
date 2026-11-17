from fractions import Fraction

import pytest

from liebracket import cli
from liebracket.bch import (
    FreeElement,
    bch_associative,
    bch_in_algebra,
    bch_lie_terms,
    bch_matrices,
    evaluate_terms,
    exp_matrix,
    format_lie_terms,
    free_exp,
    free_log,
    is_lie_element,
    lie_basis,
    log_unipotent,
    right_normed,
    witt_dimension,
)
from liebracket.chevalley import sign_cocycle, simply_laced_algebra
from liebracket.classical import heisenberg, nilpotent_upper, su2
from liebracket.examples import DEMOS, run_demo
from liebracket.linalg import Matrix, commutator
from liebracket.representations import adjoint_representation
from liebracket.roots import root_system

X, Y = 0, 1
F = Fraction


class TestFreeAlgebra:
    def test_exp_log_inverse(self):
        x = FreeElement.generator(X, 6) + FreeElement.generator(Y, 6).scale(2)
        assert free_log(free_exp(x)) == x

    def test_commuting_case(self):
        x = FreeElement.generator(X, 5)
        z = free_log(free_exp(x) * free_exp(x.scale(3)))
        assert z == x.scale(4)

    def test_right_normed(self):
        assert right_normed((X, Y)).terms == {(X, Y): 1, (Y, X): -1}
        assert right_normed((X, X)).is_zero()

    def test_lie_criterion(self):
        assert is_lie_element(right_normed((X, Y, X, Y)))
        assert not is_lie_element(FreeElement({(X, Y): 1}, 3))
        assert not is_lie_element(FreeElement.one(3))

    def test_errors(self):
        with pytest.raises(ValueError):
            free_exp(FreeElement.one(3))
        with pytest.raises(ValueError):
            free_log(FreeElement.generator(X, 3))


class TestBCHCoefficients:
    def test_low_degree_terms(self):
        terms = dict((w, c) for c, w in bch_lie_terms(4))
        assert terms[(X,)] == 1 and terms[(Y,)] == 1
        assert terms[(X, Y)] == F(1, 2)
        assert terms[(X, X, Y)] == F(1, 12)
        assert terms[(Y, X, Y)] == F(-1, 12)
        assert terms[(X, Y, X, Y)] == F(-1, 24)
        assert len(terms) == 6

    def test_textbook_degree_four_form(self):
        # -1/24 [Y,[X,[X,Y]]] equals the canonical -1/24 [X,[Y,[X,Y]]]
        z4 = bch_associative(4).homogeneous(4)
        assert z4 == right_normed((Y, X, X, Y)).scale(F(-1, 24))

    @pytest.mark.parametrize("deg", range(1, 8))
    def test_series_is_lie_and_reconstructs(self, deg):
        z = bch_associative(deg)
        assert is_lie_element(z)
        rebuilt = FreeElement({}, deg)
        for c, w in bch_lie_terms(deg, z):
            rebuilt = rebuilt + right_normed(w, deg).scale(c)
        assert rebuilt == z

    def test_symmetry_z_of_minus(self):
        # log(e^{-Y} e^{-X}) = -log(e^X e^Y)
        deg = 6
        z = bch_associative(deg)
        swapped = FreeElement({tuple(1 - a for a in w): c * (-1) ** len(w) for w, c in z.terms.items()}, deg)
        assert swapped == z.scale(-1)

    def test_even_degree_terms_vanish_on_symmetric_pair(self):
        # Z(X, X) = 2X: all brackets vanish when Y = X
        A = Matrix([[1, 2], [3, 4]])
        assert bch_matrices(A, A, 5) == A * 2

    @pytest.mark.parametrize("n", range(1, 9))
    def test_witt_formula(self, n):
        assert len(lie_basis(n)) == witt_dimension(n)

    def test_witt_three_generators(self):
        assert [witt_dimension(n, 3) for n in range(1, 6)] == [3, 3, 8, 18, 48]

    def test_formatting(self):
        s = format_lie_terms(bch_lie_terms(3))
        assert s == "X + Y + 1/2 [X,Y] + 1/12 [X,[X,Y]] - 1/12 [Y,[X,Y]]"
        assert format_lie_terms([]) == "0"
        assert format_lie_terms([(F(-1), (X,))]) == "-X"


class TestBCHOnGroups:
    def test_exp_log_nilpotent(self):
        N = Matrix([[0, 1, F(1, 2)], [0, 0, 3], [0, 0, 0]])
        assert log_unipotent(exp_matrix(N)) == N
        with pytest.raises(ValueError):
            log_unipotent(Matrix([[2, 0], [0, 1]]))

    def test_heisenberg_exact(self):
        h = heisenberg(1)
        x, p, z = h.matrices
        A = x * 3 + z * F(1, 7)
        B = p * F(-2, 5) + x
        assert exp_matrix(A) @ exp_matrix(B) == exp_matrix(bch_matrices(A, B, 2))
        assert bch_matrices(A, B, 2) == A + B + commutator(A, B) * F(1, 2)

    @pytest.mark.parametrize("n,deg", [(4, 3), (5, 4)])
    def test_strictly_upper_triangular_exact(self, n, deg):
        g = nilpotent_upper(n)
        mats = g.matrices
        A = sum((m * F(k + 1, 3) for k, m in enumerate(mats)), Matrix.zeros(n))
        B = sum((m * (-1) ** k for k, m in enumerate(mats[::-1])), Matrix.zeros(n))
        Z = bch_matrices(A, B, deg)
        assert exp_matrix(A) @ exp_matrix(B) == exp_matrix(Z)
        assert Z == log_unipotent(exp_matrix(A) @ exp_matrix(B))

    def test_truncation_error_decreases_su2(self):
        g = su2()
        a = [F(1, 20), F(1, 50), 0]
        b = [0, F(1, 30), F(1, 40)]
        A, B = (sum((m * c for m, c in zip(g.matrices, v)), Matrix.zeros(2)) for v in (a, b))
        exact = exp_matrix(A, 25) @ exp_matrix(B, 25)

        def err(deg):
            Z = bch_matrices(A, B, deg)
            diff = exp_matrix(Z, 25) - exact
            return sum(abs(complex(x)) ** 2 for x in diff.flatten())

        errors = [err(d) for d in (1, 2, 3, 4)]
        assert errors[0] > errors[1] > errors[2] > errors[3]

    def test_bch_in_algebra_matches_matrices(self):
        g = su2()
        x, y = [1, F(1, 2), 0], [0, 2, -1]
        v = bch_in_algebra(g, x, y, 4)
        M = sum((m * c for m, c in zip(g.matrices, v)), Matrix.zeros(2))
        X_, Y_ = (sum((m * c for m, c in zip(g.matrices, w)), Matrix.zeros(2)) for w in (x, y))
        assert M == bch_matrices(X_, Y_, 4)

    def test_evaluate_terms_generic(self):
        val = evaluate_terms([(F(1), (X,)), (F(2), (X, Y))], [3, 5], lambda a, b: a * b - b * a,
                             lambda a, b: a + b, lambda c, a: c * a, 0)
        assert val == 3


class TestChevalley:
    @pytest.mark.parametrize("spec,dim", [("A1", 3), ("A2", 8), ("A3", 15), ("D4", 28)])
    def test_jacobi_dimension_semisimple(self, spec, dim):
        g = simply_laced_algebra(root_system(spec))
        assert g.dim == dim
        assert g.satisfies_jacobi()
        assert g.is_semisimple()
        assert adjoint_representation(g).is_homomorphism()

    def test_cocycle_properties(self):
        rs = root_system("A3")
        eps = sign_cocycle(rs)
        for a in rs.roots:
            assert eps(a, a) == (-1) ** (int(rs.norm2(a)) // 2)
            for b in rs.roots:
                assert eps(a, b) * eps(b, a) == (-1) ** int(rs.inner(a, b))

    def test_killing_on_cartan_is_multiple_of_gram(self):
        rs = root_system("A3")
        g = simply_laced_algebra(rs)
        K = g.killing_form()
        G = rs.gram_matrix()
        h = rs.coxeter_number()
        for i in range(3):
            for j in range(3):
                assert K[i, j] == 2 * h * G[i, j]

    def test_rejects_non_simply_laced(self):
        with pytest.raises(ValueError):
            simply_laced_algebra(root_system("B2"))


class TestCLI:
    @pytest.mark.parametrize("argv,needle", [
        (["algebra", "su2", "--brackets", "--killing"], "[e1, e2] = e3"),
        (["cartan", "G2"], "det = 1"),
        (["roots", "B2", "--list"], "ht 3"),
        (["dim", "E8", "0,0,0,0,0,0,0,1"], "= 248"),
        (["weights", "A2", "1,1", "--dominant"], "(0,0)"),
        (["weights", "G2", "1,0"], "dim 7"),
        (["tensor", "A2", "1,0", "0,1"], "V(1,1)[8] + V(0,0)[1]"),
        (["bch", "--degree", "3"], "1/12 [X,[X,Y]]"),
        (["bch", "--degree", "2", "--table"], "1/2"),
        (["demo", "su2"], "j = 1/2"),
    ])
    def test_commands(self, argv, needle, capsys):
        assert cli.main(argv) == 0
        assert needle in capsys.readouterr().out

    def test_errors(self, capsys):
        assert cli.main(["cartan", "E9"]) == 2
        with pytest.raises(SystemExit):
            cli.main(["dim", "A2", "1"])

    def test_all_demos_run(self):
        for name in DEMOS:
            text = run_demo(name)
            assert "False" not in text and "FAILS" not in text
        with pytest.raises(KeyError):
            run_demo("nope")
