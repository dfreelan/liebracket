# liebracket

**Exact Lie algebra and representation-theory numerics in pure Python.**

`liebracket` computes structure constants, Killing forms, representations, root
systems, Cartan matrices, weight multiplicities and the Baker–Campbell–Hausdorff
series. It does this over $\mathbb{Q}$ and the Gaussian rationals $\mathbb{Q}(i)$, so
there is no rounding. Every identity the library reports (the Jacobi identity,
homomorphism conditions, $e^{A}e^{B} = e^{Z}$) is checked by exact arithmetic,
not up to a tolerance.

It runs on the standard library alone and has no runtime dependencies. It suits
work where you need trusted group-theory data: building equivariant layers
(Clebsch–Gordan decompositions, Casimir operators), physics-informed models with
$\mathfrak{su}(2)$/$\mathfrak{su}(3)$ symmetry, or checking hand-derived
structure constants.

```console
$ pip install -e .
$ liebracket tensor A2 1,1 1,1
V(1,1) x V(1,1) = V(2,2)[27] + V(0,3)[10] + V(3,0)[10] + 2*V(1,1)[8] + V(0,0)[1]
```

---

## Mathematical content

### Lie algebras from structure constants

A Lie algebra with basis $e_1,\dots,e_n$ is stored through

$$[e_i, e_j] = \sum_k c_{ij}^{\;k}\, e_k ,\qquad c_{ij}^{\;k} = -c_{ji}^{\;k}.$$

Construction verifies the Jacobi identity on every basis triple:

$$[e_i,[e_j,e_k]] + [e_j,[e_k,e_i]] + [e_k,[e_i,e_j]] = 0 .$$

| quantity | formula | method |
|---|---|---|
| adjoint | $(\operatorname{ad}_{e_i})_{kj} = c_{ij}^{\;k}$ | `ad`, `ad_basis` |
| Killing form | $\kappa_{ij} = \operatorname{tr}(\operatorname{ad}_{e_i}\operatorname{ad}_{e_j}) = \sum_{k,l} c_{ik}^{\;l}c_{jl}^{\;k}$ | `killing_form` |
| Cartan's criterion | $\mathfrak g$ semisimple $\iff \det\kappa \ne 0$ | `is_semisimple` |
| adjoint Casimir | $\sum_{ij}\kappa^{ij}\operatorname{ad}_{e_i}\operatorname{ad}_{e_j} = \mathbb 1$ (simple $\mathfrak g$) | `casimir_in_adjoint` |
| derived / lower central series | $\mathfrak g^{(k+1)} = [\mathfrak g^{(k)},\mathfrak g^{(k)}]$, $\mathfrak g_{k+1} = [\mathfrak g,\mathfrak g_k]$ | `derived_series`, `lower_central_series` |
| centre, solvable, nilpotent | | `center`, `is_solvable`, `is_nilpotent` |
| homomorphisms | $\varphi[x,y] = [\varphi x,\varphi y]$ | `is_homomorphism`, `is_isomorphism` |

### Classical algebras (`liebracket.classical`)

Each algebra is built from an explicit matrix basis. Its structure constants are
found by solving $[A_i,A_j] = \sum_k c_{ij}^{\;k}A_k$ exactly, and the matrices
are kept as the defining representation.

* $\mathfrak{su}(2)$ with $e_k = -\tfrac{i}{2}\sigma_k$, so that $[e_i,e_j] = \varepsilon_{ijk}e_k$ and $\kappa = -2\cdot\mathbb 1$.
* $\mathfrak{so}(3)$ with $(L_i)_{jk} = -\varepsilon_{ijk}$. It has the same constants, and `su2().is_isomorphism(so3(), I)` returns `True`.
* $\mathfrak{su}(3)$ with $T_a = -\tfrac{i}{2}\lambda_a$. The eighth generator is rescaled by $\sqrt3$, which makes every $c_{ij}^{\;k}$ rational. The standard irrational table $f_{abc}$ is also provided, as `GELL_MANN_F`.
* $\mathfrak{sl}_n$, $\mathfrak{so}_n$, $\mathfrak{sp}_{2n}$, $\mathfrak{gl}_n$, the Heisenberg algebras $\mathfrak h_{2n+1}$, and the nilpotent and Borel subalgebras $\mathfrak n_n \subset \mathfrak b_n$.

### Representations (`liebracket.representations`)

A representation is a list of matrices $\rho(e_i)$. `is_homomorphism()` checks
$\rho([e_i,e_j]) = [\rho(e_i),\rho(e_j)]$ for every pair.

* adjoint, defining and trivial representations
* $\mathfrak{sl}_2$ irreducibles $V_n$: $h v_k = (n-2k)v_k$, $f v_k = v_{k+1}$, $e v_k = k(n-k+1)v_{k-1}$
* $\mathfrak{su}(2)$ spin $j$, exact over $\mathbb Q(i)$, with $\sum_k\rho(e_k)^2 = -j(j+1)\,\mathbb 1$
* $\oplus$, $\otimes$ (where $\rho\otimes\sigma = \rho\otimes 1 + 1\otimes\sigma$), the dual $-\rho^{\mathsf T}$, and conjugation
* trace forms, Dynkin index, Casimir operators, commutant dimension (Schur), and Casimir eigenspace decomposition

### Root systems and Cartan matrices (`liebracket.roots`, `liebracket.cartan`)

All finite types are covered: $A_n, B_n, C_n, D_n, E_{6,7,8}, F_4, G_2$. Cartan
matrices use Bourbaki numbering and are *derived* from Bourbaki's Euclidean
simple roots, with
$A_{ij} = \langle\alpha_i^\vee,\alpha_j\rangle = 2(\alpha_i,\alpha_j)/(\alpha_i,\alpha_i)$.

Positive roots are generated from the Cartan matrix alone, by root strings. If
$\beta - p\alpha_i,\dots,\beta+q\alpha_i$ is the $\alpha_i$-string through
$\beta$, then $p - q = \langle\beta,\alpha_i^\vee\rangle$. From the roots the
library computes:

* the highest root $\theta$, the Coxeter number $h = \operatorname{ht}\theta + 1$, and the exponents (Kostant's height partition)
* $|W| = \prod_i(m_i+1)$, with brute-force enumeration of $W$ as a cross-check for small rank
* simple reflections, Weyl orbits, root strings, long and short roots
* the symmetrizer $d_i$ with $d_iA_{ij} = d_jA_{ji}$, finite-type detection via Sylvester's criterion, and ASCII Dynkin diagrams

```console
$ liebracket cartan F4
Cartan matrix of F4  (A_ij = <alpha_i^vee, alpha_j>):
[ 2  -1   0   0]
[-1   2  -1   0]
[ 0  -2   2  -1]
[ 0   0  -1   2]
det = 1, finite type: True
Dynkin diagram:
o---o==>o---o
1   2   3   4
```

### Highest-weight modules (`liebracket.weights`)

Weights are written in Dynkin labels. **Weyl's dimension formula**:

$$\dim V(\lambda) = \prod_{\alpha>0}\frac{(\lambda+\rho,\alpha)}{(\rho,\alpha)}.$$

**Freudenthal's multiplicity formula**:

$$\bigl(\|\lambda+\rho\|^2 - \|\mu+\rho\|^2\bigr)\,m_\lambda(\mu) = 2\sum_{\alpha>0}\sum_{k\ge 1}(\mu+k\alpha,\alpha)\,m_\lambda(\mu+k\alpha).$$

Tensor products are decomposed by peeling highest weights off products of
characters. Results such as $\mathbf 7\otimes\mathbf 7 = \mathbf{27}\oplus\mathbf{14}\oplus\mathbf 7\oplus\mathbf 1$
for $G_2$, and the zero-weight multiplicity 8 in the 248-dimensional adjoint of
$E_8$, come straight out of these formulas.

### From roots back to an algebra (`liebracket.chevalley`)

For simply-laced types the library builds the split simple Lie algebra
directly from the root lattice. It uses a bimultiplicative sign cocycle with
$\varepsilon(\alpha,\alpha) = (-1)^{(\alpha,\alpha)/2}$, and sets
$[E_\alpha,E_\beta] = \varepsilon(\alpha,\beta)E_{\alpha+\beta}$ and
$[E_\alpha,E_{-\alpha}] = -\alpha$. The Jacobi identity is verified rather than
assumed. For $\mathfrak g(D_4)$ the result is 28-dimensional and semisimple.

### Baker–Campbell–Hausdorff (`liebracket.bch`)

$$\log(e^Xe^Y) = X + Y + \tfrac12[X,Y] + \tfrac1{12}[X,[X,Y]] - \tfrac1{12}[Y,[X,Y]] - \tfrac1{24}[Y,[X,[X,Y]]] + \cdots$$

The series is computed to any degree in the truncated free associative algebra
$\mathbb Q\langle X,Y\rangle$. It is then projected to Lie form with the
Dynkin–Specht–Wever map $P(L) = n L$, and written in a basis of right-normed
brackets chosen by exact elimination. The basis size matches Witt's formula
$\tfrac1n\sum_{d\mid n}\mu(d)2^{n/d}$, and this is tested. The truncated series
can be evaluated on matrices or inside any `LieAlgebra`. On nilpotent groups
(Heisenberg, strictly upper-triangular matrices) the truncation is exact, and
the tests check $e^Ae^B = e^{Z}$ entry by entry.

```console
$ liebracket bch --degree 5
log(e^X e^Y) = X + Y + 1/2 [X,Y] + 1/12 [X,[X,Y]] - 1/12 [Y,[X,Y]] - 1/24 [X,[Y,[X,Y]]] - 1/720 [X,[X,[X,[X,Y]]]] - ...
```

---

## Python API

```python
from fractions import Fraction
from liebracket import su3, sl2_irrep, adjoint_representation, root_system, WeightSpace, bch_matrices, heisenberg, exp_matrix

g = su3()
assert g.satisfies_jacobi() and g.is_semisimple()
print(g.killing_form().diagonal())          # [-3, -3, -3, -3, -3, -3, -3, -9]
assert adjoint_representation(g).is_homomorphism()

V = sl2_irrep(1).tensor(sl2_irrep(2))      # V_1 (x) V_2 = V_3 + V_1
print(V.casimir().characteristic_polynomial())

rs = root_system("E8")
print(rs.num_positive_roots, rs.coxeter_number(), rs.weyl_group_order())   # 120 30 696729600

W = WeightSpace.of_type("G2")
print(W.tensor_product((1, 0), (1, 0)))    # {(2,0): 1, (0,1): 1, (1,0): 1, (0,0): 1}

x, p, z = heisenberg(1).matrices
A, B = 2 * x + Fraction(1, 3) * z, -5 * p
assert exp_matrix(A) @ exp_matrix(B) == exp_matrix(bch_matrices(A, B, 2))
```

## Command line

| command | description |
|---|---|
| `liebracket algebra su3 --brackets --killing` | summary, bracket table, Killing form |
| `liebracket cartan G2` | Cartan matrix, determinant, Dynkin diagram |
| `liebracket roots E7 --list` | root data and positive roots by height |
| `liebracket dim B3 0,0,1` | Weyl dimension |
| `liebracket weights A2 2,1 [--dominant]` | Freudenthal multiplicities |
| `liebracket tensor G2 1,0 1,0` | tensor product decomposition |
| `liebracket bch --degree 6 [--table]` | BCH series |
| `liebracket demo {su2,su2-so3,su3,sl2-cg,roots,g2,sl3-weights,bch,d4,all}` | worked examples |

`python -m liebracket ...` works too.

## Design notes

* **Exactness first.** Entries are `int`, `Fraction` or `GaussianRational`. Elimination (rank, determinant, inverse, null space, characteristic polynomial via Faddeev–LeVerrier) never rounds. `float`/`complex` entries also work, with a small zero tolerance.
* **Derived, not tabulated.** Cartan matrices come from Euclidean simple roots. Roots come from Cartan matrices. Structure constants come from matrices or from root data. Weight data comes from Freudenthal's formula. The tests check each of these routes against the others: weights of `sl(n)` read from the diagonal Cartan action agree with Freudenthal, and matrix Casimirs agree with $(\lambda,\lambda+2\rho)$.
* **Conventions.** Bourbaki numbering, $A_{ij} = \langle\alpha_i^\vee,\alpha_j\rangle$, short roots of squared length 2 in the symmetrised form, and Killing-form normalisation for Casimirs.

## Development

```console
$ python -m venv .venv && .venv/bin/pip install -e '.[test]'
$ .venv/bin/pytest -q
$ .venv/bin/python tools/count_loc.py
```

## License

MIT, see [LICENSE](LICENSE).
