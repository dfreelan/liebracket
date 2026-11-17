from fractions import Fraction
from math import factorial

import pytest

from liebracket.cartan import (
    CARTAN_DETERMINANTS,
    cartan_matrix,
    dynkin_diagram,
    dynkin_edges,
    is_cartan_matrix,
    is_finite_type,
    parse_type,
    symmetrized_cartan,
    symmetrizer,
)
from liebracket.linalg import Matrix
from liebracket.roots import RootSystem, euclidean_simple_roots, root_system

TYPES = ["A1", "A2", "A3", "A4", "B2", "B3", "B4", "C2", "C3", "C4", "D4", "D5", "E6", "E7", "E8", "F4", "G2"]

NUM_POSITIVE = {"A": lambda n: n * (n + 1) // 2, "B": lambda n: n * n, "C": lambda n: n * n,
                "D": lambda n: n * (n - 1), "E": lambda n: {6: 36, 7: 63, 8: 120}[n], "F": lambda n: 24,
                "G": lambda n: 6}
WEYL_ORDER = {"A": lambda n: factorial(n + 1), "B": lambda n: 2 ** n * factorial(n), "C": lambda n: 2 ** n * factorial(n),
              "D": lambda n: 2 ** (n - 1) * factorial(n), "E": lambda n: {6: 51840, 7: 2903040, 8: 696729600}[n],
              "F": lambda n: 1152, "G": lambda n: 12}
COXETER = {"A": lambda n: n + 1, "B": lambda n: 2 * n, "C": lambda n: 2 * n, "D": lambda n: 2 * n - 2,
           "E": lambda n: {6: 12, 7: 18, 8: 30}[n], "F": lambda n: 12, "G": lambda n: 6}


def test_parse_type():
    assert parse_type("a3") == ("A", 3)
    assert parse_type("E_8") == ("E", 8)
    for bad in ("E5", "G3", "B1", "X2", "D2", "A0", "F"):
        with pytest.raises(ValueError):
            parse_type(bad)


@pytest.mark.parametrize("spec", TYPES)
def test_cartan_axioms_and_determinant(spec):
    t, n = parse_type(spec)
    A = cartan_matrix(spec)
    assert is_cartan_matrix(A)
    assert is_finite_type(A)
    assert A.det() == CARTAN_DETERMINANTS[t](n)
    G = symmetrized_cartan(A)
    assert G.is_symmetric()


def test_specific_cartan_matrices():
    assert cartan_matrix("G2") == Matrix([[2, -3], [-1, 2]])
    assert cartan_matrix("B2") == Matrix([[2, -1], [-2, 2]])
    assert cartan_matrix("C2") == cartan_matrix("B2").T
    assert cartan_matrix("F4") == Matrix([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -2, 2, -1], [0, 0, -1, 2]])
    E8 = cartan_matrix("E8")
    assert E8[0, 2] == -1 and E8[1, 3] == -1 and E8[0, 1] == 0


def test_affine_and_indefinite_not_finite():
    affine_a1 = Matrix([[2, -2], [-2, 2]])
    assert is_cartan_matrix(affine_a1) and not is_finite_type(affine_a1)
    hyperbolic = Matrix([[2, -3], [-3, 2]])
    assert not is_finite_type(hyperbolic)
    assert not is_cartan_matrix(Matrix([[2, 1], [1, 2]]))
    assert not is_cartan_matrix(Matrix([[2, -1], [0, 2]]))
    with pytest.raises(ValueError):
        RootSystem(affine_a1)


def test_symmetrizer():
    assert symmetrizer(cartan_matrix("G2")) == [1, 3]
    assert symmetrizer(cartan_matrix("B3")) == [2, 2, 1]
    assert symmetrizer(cartan_matrix("C3")) == [1, 1, 2]
    with pytest.raises(ValueError):
        symmetrizer(Matrix([[2, -1, 0], [-1, 2, -1], [-2, -1, 2]]))


def test_dynkin():
    assert dynkin_diagram(cartan_matrix("A3")).splitlines()[0] == "o---o---o"
    assert dynkin_diagram(cartan_matrix("B3")).splitlines()[0] == "o---o==>o"
    assert dynkin_diagram(cartan_matrix("C3")).splitlines()[0] == "o---o==<o"
    assert "o 2" in dynkin_diagram(cartan_matrix("E6"))
    assert dynkin_edges(cartan_matrix("G2")) == [(0, 1, 3, -1)]
    assert len(dynkin_edges(cartan_matrix("D4"))) == 3


@pytest.mark.parametrize("spec", TYPES)
def test_root_counts_weyl_order_coxeter(spec):
    t, n = parse_type(spec)
    rs = root_system(spec)
    assert rs.num_positive_roots == NUM_POSITIVE[t](n)
    assert rs.weyl_group_order() == WEYL_ORDER[t](n)
    assert rs.coxeter_number() == COXETER[t](n)
    ex = rs.exponents()
    assert len(ex) == n and ex[0] == 1 and ex[-1] == rs.coxeter_number() - 1
    # exponents are symmetric: m_i + m_{r+1-i} = h
    assert all(a + b == rs.coxeter_number() for a, b in zip(ex, reversed(ex)))
    # |Phi| = r h
    assert 2 * rs.num_positive_roots == n * rs.coxeter_number()


@pytest.mark.parametrize("spec", TYPES)
def test_euclidean_realisation_consistent(spec):
    rs = root_system(spec)
    for beta in rs.positive_roots:
        v = rs.to_euclidean(beta)
        norm = sum(x * x for x in v)
        # invariant form in simple coords is a constant multiple of the Euclidean one
        ratio = rs.norm2(beta) / norm
        assert ratio == rs.norm2(rs.simple_roots[0]) / sum(x * x for x in euclidean_simple_roots(spec)[0])


def test_explicit_root_sets():
    # B2 roots in Bourbaki coordinates: ±e1±e2, ±e1, ±e2
    rs = root_system("B2")
    vecs = {tuple(rs.to_euclidean(b)) for b in rs.roots}
    expected = {(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}
    assert vecs == {tuple(Fraction(x) for x in e) for e in expected}
    # G2 has 6 short and 6 long roots, length ratio sqrt 3
    g2 = root_system("G2")
    long_, short = g2.long_and_short()
    assert len(long_) == len(short) == 3
    assert g2.norm2(long_[0]) == 3 * g2.norm2(short[0])
    assert g2.highest_root == (3, 2)


@pytest.mark.parametrize("spec", ["A1", "A2", "A3", "B2", "B3", "C3", "G2", "D4", "F4"])
def test_weyl_group_enumeration_matches_formula(spec):
    rs = root_system(spec)
    W = rs.weyl_group()
    assert len(W) == rs.weyl_group_order()
    # W permutes the roots
    roots = set(rs.roots)
    w = W[len(W) // 2]
    assert {tuple(w @ list(b)) for b in roots} == roots


def test_reflections():
    rs = root_system("A2")
    a1, a2 = rs.simple_roots
    assert rs.reflect(0, a1) == (-1, 0)
    assert rs.reflect(0, a2) == (1, 1)
    s = rs.reflection_matrix(0)
    assert s @ s == Matrix.identity(2)
    # (s1 s2)^3 = 1 for A2, (s1 s2)^6 = 1 for G2
    t = rs.reflection_matrix(0) @ rs.reflection_matrix(1)
    assert t ** 3 == Matrix.identity(2)
    g2 = root_system("G2")
    t = g2.reflection_matrix(0) @ g2.reflection_matrix(1)
    assert t ** 6 == Matrix.identity(2) and t ** 3 != Matrix.identity(2)


def test_root_strings_and_pairings():
    g2 = root_system("G2")
    string = g2.root_string((0, 1), (1, 0))
    assert string == [(0, 1), (1, 1), (2, 1), (3, 1)]
    for beta in g2.roots:
        for alpha in g2.roots:
            c = g2.coroot_pairing(beta, alpha)
            assert c.denominator == 1 and abs(c) <= 3
    assert g2.orbit((1, 0)) == sorted(set(g2.long_and_short()[1]) | {tuple(-x for x in b) for b in g2.long_and_short()[1]})


def test_highest_root_dominant_and_summary():
    for spec in TYPES:
        rs = root_system(spec)
        theta = rs.highest_root
        assert all(rs.pairing(theta, i) >= 0 for i in range(rs.rank))
        assert rs.is_root(theta) and not rs.is_root(tuple(2 * x for x in theta))
    assert "|W| = 1152" in root_system("F4").summary()
    assert root_system("E8").lie_algebra_dimension() == 248
    assert root_system("E7").is_simply_laced() and not root_system("F4").is_simply_laced()
    assert root_system("D4").longest_element_length() == 12
