from fractions import Fraction

import pytest

from liebracket.classical import sl, so, sp
from liebracket.representations import adjoint_representation, defining_representation
from liebracket.roots import root_system
from liebracket.weights import WeightSpace

ADJOINT_DIMS = {"A2": 8, "A3": 15, "B2": 10, "B3": 21, "C3": 21, "D4": 28, "G2": 14, "F4": 52, "E6": 78}


@pytest.mark.parametrize("spec,dim", sorted(ADJOINT_DIMS.items()))
def test_adjoint_dimension_and_zero_weight(spec, dim):
    W = WeightSpace.of_type(spec)
    theta = W.adjoint_highest_weight()
    assert W.weyl_dimension(theta) == dim
    char = W.character(theta)
    assert sum(char.values()) == dim
    assert char[tuple([0] * W.rank)] == W.rank


@pytest.mark.parametrize("spec,dims", [
    ("A2", [3, 3]), ("A3", [4, 6, 4]), ("B2", [5, 4]), ("B3", [7, 21, 8]), ("C3", [6, 14, 14]),
    ("D4", [8, 28, 8, 8]), ("G2", [7, 14]), ("F4", [52, 1274, 273, 26]), ("E6", [27, 78, 351, 2925, 351, 27]),
])
def test_fundamental_dimensions(spec, dims):
    W = WeightSpace.of_type(spec)
    assert [W.weyl_dimension(w) for w in W.fundamental_weights()] == dims


@pytest.mark.parametrize("spec,lam", [("A2", (2, 1)), ("A3", (1, 0, 1)), ("B2", (1, 1)), ("C3", (0, 1, 0)),
                                      ("G2", (1, 1)), ("B3", (0, 0, 2)), ("D4", (0, 0, 1, 1))])
def test_freudenthal_sums_to_weyl_dimension(spec, lam):
    W = WeightSpace.of_type(spec)
    char = W.character(lam)
    assert sum(char.values()) == W.weyl_dimension(lam)
    # Weyl invariance of multiplicities
    for mu, m in char.items():
        for i in range(W.rank):
            assert char.get(W.reflect(i, mu), 0) == m


def test_known_multiplicities():
    A2 = WeightSpace.of_type("A2")
    assert A2.multiplicity((2, 2), (0, 0)) == 3  # 27 of su(3)
    assert A2.multiplicity((3, 0), (0, 0)) == 1  # decuplet
    G2 = WeightSpace.of_type("G2")
    assert G2.multiplicity((1, 0), (0, 0)) == 1
    assert G2.multiplicity((0, 1), (0, 0)) == 2


def test_tensor_products_su3():
    W = WeightSpace.of_type("A2")
    assert W.tensor_product((1, 0), (0, 1)) == {(1, 1): 1, (0, 0): 1}
    assert W.tensor_product((1, 1), (1, 1)) == {(2, 2): 1, (3, 0): 1, (0, 3): 1, (1, 1): 2, (0, 0): 1}
    three_cubed = W.decompose(W.multiply_characters(W.character((1, 0)), W.character((1, 1))))
    assert three_cubed == {(2, 1): 1, (0, 2): 1, (1, 0): 1}


def test_tensor_dimensions_add_up():
    for spec, a, b in (("B2", (1, 0), (0, 1)), ("G2", (1, 0), (1, 0)), ("C3", (1, 0, 0), (0, 0, 1))):
        W = WeightSpace.of_type(spec)
        dec = W.tensor_product(a, b)
        assert sum(n * W.weyl_dimension(w) for w, n in dec.items()) == W.weyl_dimension(a) * W.weyl_dimension(b)


def test_g2_seven_squared():
    W = WeightSpace.of_type("G2")
    assert W.tensor_product((1, 0), (1, 0)) == {(2, 0): 1, (0, 1): 1, (1, 0): 1, (0, 0): 1}


def test_decompose_rejects_non_characters():
    W = WeightSpace.of_type("A1")
    with pytest.raises(ValueError):
        W.decompose({(1,): 1})


@pytest.mark.parametrize("n", [3, 4])
def test_sl_n_weights_match_freudenthal(n):
    g = sl(n)
    cartan = list(range(n - 1))
    W = WeightSpace.of_type("A%d" % (n - 1))
    fund = tuple([1] + [0] * (n - 2))
    defining = defining_representation(g).weights(cartan)
    expected = sorted((mu for mu, m in W.character(fund).items() for _ in range(m)), reverse=True)
    assert defining == expected
    adj = adjoint_representation(g).weights(cartan)
    theta = W.adjoint_highest_weight()
    expected = sorted((mu for mu, m in W.character(theta).items() for _ in range(m)), reverse=True)
    assert adj == expected


def test_casimir_eigenvalue_ratio_matches_matrix_casimir():
    # For a simple algebra, Killing-normalised Casimir on V(lambda) is (lambda, lambda+2rho)/(theta, theta+2rho)
    W = WeightSpace.of_type("A2")
    theta = W.adjoint_highest_weight()
    ratio = W.casimir_eigenvalue((1, 0)) / W.casimir_eigenvalue(theta)
    C = defining_representation(sl(3)).casimir()
    assert C[0, 0] == ratio == Fraction(4, 9)
    W = WeightSpace.of_type("B2")
    ratio = W.casimir_eigenvalue((1, 0)) / W.casimir_eigenvalue(W.adjoint_highest_weight())
    assert defining_representation(so(5)).casimir()[0, 0] == ratio
    W = WeightSpace.of_type("C2")
    ratio = W.casimir_eigenvalue((1, 0)) / W.casimir_eigenvalue(W.adjoint_highest_weight())
    assert defining_representation(sp(2)).casimir()[0, 0] == ratio


def test_weyl_orbit_and_dominant_representative():
    W = WeightSpace.of_type("A2")
    assert len(W.weyl_orbit((1, 0))) == 3
    assert len(W.weyl_orbit((1, 1))) == 6
    assert W.dominant_representative((-1, -1)) == (1, 1)
    rs = root_system("B3")
    assert len(WeightSpace(rs).weyl_orbit((1, 0, 0))) == 6


def test_root_weight_round_trip_and_rho():
    W = WeightSpace.of_type("F4")
    for beta in W.rs.positive_roots[:10]:
        lab = W.root_to_weight(beta)
        assert W.weight_to_root_coords(lab) == [Fraction(b) for b in beta]
    # rho = half sum of positive roots
    total = [sum(b[i] for b in W.rs.positive_roots) for i in range(W.rank)]
    assert W.root_to_weight([Fraction(t, 2) for t in total]) == W.rho


def test_invalid_highest_weight_and_listing():
    W = WeightSpace.of_type("A2")
    with pytest.raises(ValueError):
        W.weyl_dimension((1, -1))
    with pytest.raises(ValueError):
        W.character((-1, 0))
    table = W.irreps_up_to(2)
    assert table[0] == ((0, 0), 1) and len(table) == 9
    assert W.format_decomposition({(1, 1): 2}) == "2*V(1,1)[8]"
