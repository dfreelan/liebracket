from fractions import Fraction

import pytest

from liebracket.linalg import Matrix, commutator, solve_in_span, span_basis
from liebracket.scalars import GaussianRational, I, as_fraction, format_scalar, is_zero, to_exact


class TestGaussianRational:
    def test_field_axioms_sample(self):
        a = GaussianRational(Fraction(1, 2), 3)
        b = GaussianRational(-2, Fraction(1, 5))
        c = GaussianRational(7, -1)
        assert (a + b) + c == a + (b + c)
        assert (a * b) * c == a * (b * c)
        assert a * (b + c) == a * b + a * c
        assert a * b == b * a
        assert (a / b) * b == a

    def test_i_squared(self):
        assert I * I == -1
        assert I ** 4 == 1
        assert I ** -1 == -I

    def test_conjugate_and_norm(self):
        z = GaussianRational(3, 4)
        assert z * z.conjugate() == 25
        assert z.norm() == 25

    def test_mixed_arithmetic_and_hash(self):
        z = GaussianRational(Fraction(1, 2), 0)
        assert z == Fraction(1, 2)
        assert hash(z) == hash(Fraction(1, 2))
        assert 1 - I == GaussianRational(1, -1)
        assert 2 / I == -2 * I

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            I / GaussianRational(0, 0)

    def test_formatting(self):
        assert format_scalar(Fraction(3, 6)) == "1/2"
        assert format_scalar(-I) == "-i"
        assert format_scalar(GaussianRational(1, Fraction(-1, 3))) == "1 - i/3"
        assert format_scalar(GaussianRational(0, 2)) == "2i"

    def test_to_exact_and_as_fraction(self):
        assert to_exact(GaussianRational(4, 0)) == 4 and isinstance(to_exact(GaussianRational(4, 0)), int)
        assert as_fraction(GaussianRational(Fraction(1, 3))) == Fraction(1, 3)
        with pytest.raises(ValueError):
            as_fraction(I)
        assert is_zero(1e-14) and not is_zero(Fraction(1, 10**20))


class TestMatrix:
    def test_det_inverse(self):
        A = Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
        assert A.det() == 18
        assert A @ A.inverse() == Matrix.identity(3)
        assert A.inverse()[0, 0] == Fraction(11, 18)

    def test_singular(self):
        A = Matrix([[1, 2], [2, 4]])
        assert A.det() == 0
        assert A.rank() == 1
        with pytest.raises(ZeroDivisionError):
            A.inverse()

    def test_nullspace(self):
        A = Matrix([[1, 1, 1], [1, 2, 3]])
        ns = A.nullspace()
        assert len(ns) == 1
        assert A @ ns[0] == [0, 0]

    def test_solve(self):
        A = Matrix([[1, 2], [3, 4]])
        x = A.solve([5, 6])
        assert A @ x == [5, 6]
        assert Matrix([[1, 1], [1, 1]]).solve([1, 2]) is None

    def test_complex_det(self):
        A = Matrix([[I, 1], [1, I]])
        assert A.det() == -2

    def test_kron_and_direct_sum(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[0, 1], [1, 0]])
        K = A.kron(B)
        assert K.shape == (4, 4)
        assert K[0, 1] == 1 and K[2, 3] == 4 and K[3, 1] == 0 and K[3, 0] == 3
        assert A.kron(B).trace() == A.trace() * B.trace()
        assert A.direct_sum(B).det() == A.det() * B.det()

    def test_mixed_product_property(self):
        A, B = Matrix([[1, 2], [0, 1]]), Matrix([[0, 1], [1, 1]])
        C, D = Matrix([[2, 0], [1, 1]]), Matrix([[1, -1], [0, 3]])
        assert A.kron(B) @ C.kron(D) == (A @ C).kron(B @ D)

    def test_characteristic_polynomial(self):
        A = Matrix([[2, 1], [1, 2]])
        assert A.characteristic_polynomial() == [3, -4, 1]
        M = Matrix([[1, 2, 0], [0, 1, 3], [4, 0, 1]])
        cp = M.characteristic_polynomial()
        assert cp[0] == -M.det() and cp[2] == -M.trace()

    def test_commutator_identities(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[0, 1], [-1, 0]])
        C = Matrix([[2, 0], [5, -1]])
        assert commutator(A, A).is_zero()
        assert commutator(A, B) == -commutator(B, A)
        jac = commutator(A, commutator(B, C)) + commutator(B, commutator(C, A)) + commutator(C, commutator(A, B))
        assert jac.is_zero()
        assert commutator(A, B).trace() == 0

    def test_dagger_and_power(self):
        A = Matrix([[1, I], [0, 2]])
        assert A.dagger() == Matrix([[1, 0], [-I, 2]])
        assert A ** 3 == A @ A @ A
        assert A ** 0 == Matrix.identity(2)

    def test_ragged_rejected(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3]])
        with pytest.raises(ValueError):
            Matrix([[1, 2]]) @ Matrix([[1, 2]])

    def test_span_helpers(self):
        basis = span_basis([[1, 0, 1], [2, 0, 2], [0, 1, 0]])
        assert len(basis) == 2
        assert solve_in_span([[1, 0, 1], [0, 1, 0]], [3, -2, 3]) == [3, -2]
        assert solve_in_span([[1, 0, 1]], [1, 0, 0]) is None
        assert solve_in_span([], [0, 0]) == []
