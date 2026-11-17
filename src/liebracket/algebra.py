r"""Finite-dimensional Lie algebras given by structure constants.

A Lie algebra :math:`\mathfrak g` with basis :math:`e_1, \dots, e_n` is
determined by its structure constants

.. math::

    [e_i, e_j] = \sum_k c_{ij}^{\;k}\, e_k ,

subject to antisymmetry :math:`c_{ij}^{\;k} = -c_{ji}^{\;k}` and the Jacobi identity

.. math::

    [e_i,[e_j,e_k]] + [e_j,[e_k,e_i]] + [e_k,[e_i,e_j]] = 0 .

:class:`LieAlgebra` stores the constants sparsely and exposes the adjoint
map, the Killing form :math:`\kappa(x,y) = \operatorname{tr}(\operatorname{ad}_x
\operatorname{ad}_y)`, Cartan's criterion, centres, derived and lower central
series, and homomorphism checks.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .linalg import Matrix, Vector, commutator, solve_in_span, span_basis, vec_is_zero
from .scalars import format_scalar, is_zero, to_exact

__all__ = ["LieAlgebra", "JacobiViolation"]

SparseTable = Dict[Tuple[int, int], Dict[int, Any]]


class JacobiViolation(Tuple[int, int, int, Vector]):
    """A basis triple :math:`(i,j,k)` together with its non-zero Jacobiator."""


class LieAlgebra:
    r"""A Lie algebra over a field of characteristic zero, presented by structure constants.

    Parameters
    ----------
    dim:
        Dimension :math:`n = \dim \mathfrak g`.
    brackets:
        Mapping ``(i, j) -> {k: c}`` encoding :math:`[e_i, e_j] = \sum_k c\, e_k`.
        Only one of ``(i, j)`` / ``(j, i)`` needs to be supplied; the other is
        filled in by antisymmetry (supplying both is allowed and is checked).
    names:
        Optional labels of the basis vectors.
    matrices:
        Optional faithful matrix realisation of the basis (the defining representation).
    check:
        If true (default) verify antisymmetry and the Jacobi identity on construction.
    """

    def __init__(
        self,
        dim: int,
        brackets: Mapping[Tuple[int, int], Mapping[int, Any]],
        names: Optional[Sequence[str]] = None,
        name: str = "g",
        matrices: Optional[Sequence[Matrix]] = None,
        check: bool = True,
    ) -> None:
        self.dim = dim
        self.name = name
        self.names: List[str] = list(names) if names is not None else ["e%d" % (i + 1) for i in range(dim)]
        if len(self.names) != dim:
            raise ValueError("expected %d basis names" % dim)
        self.matrices: Optional[List[Matrix]] = list(matrices) if matrices is not None else None
        table: SparseTable = {}
        for (i, j), terms in brackets.items():
            if not (0 <= i < dim and 0 <= j < dim):
                raise IndexError("bracket index out of range: %r" % ((i, j),))
            clean = {k: to_exact(c) for k, c in terms.items() if not is_zero(c)}
            if i == j:
                if clean:
                    raise ValueError("[e_%d, e_%d] must vanish" % (i, i))
                continue
            neg = {k: -c for k, c in clean.items()}
            for key, val in (((i, j), clean), ((j, i), neg)):
                if key in table and table[key] != val:
                    raise ValueError("structure constants are not antisymmetric at %r" % (key,))
                table[key] = dict(val)
        self._table = {k: v for k, v in table.items() if v}
        self._ad_cache: Dict[int, Matrix] = {}
        if check:
            bad = self.jacobi_violations(limit=1)
            if bad:
                i, j, k, _ = bad[0]
                raise ValueError(
                    "Jacobi identity fails for (%s, %s, %s)" % (self.names[i], self.names[j], self.names[k])
                )

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_structure_constants(
        cls, c: Sequence[Sequence[Sequence[Any]]], **kwargs: Any
    ) -> "LieAlgebra":
        r"""Build from a dense array ``c[i][j][k]`` :math:`= c_{ij}^{\;k}`."""
        n = len(c)
        br = {(i, j): {k: c[i][j][k] for k in range(n)} for i in range(n) for j in range(n) if i < j}
        return cls(n, br, **kwargs)

    @classmethod
    def from_matrices(
        cls, matrices: Sequence[Matrix], names: Optional[Sequence[str]] = None, name: str = "g",
        check: bool = True,
    ) -> "LieAlgebra":
        r"""The matrix Lie algebra spanned by ``matrices``, which must be closed under :math:`[A,B]=AB-BA`.

        Structure constants are obtained by solving :math:`[A_i, A_j] = \sum_k c_{ij}^{\;k} A_k` exactly.
        """
        flats = [m.flatten() for m in matrices]
        if len(span_basis(flats)) != len(flats):
            raise ValueError("matrices are linearly dependent")
        br: Dict[Tuple[int, int], Dict[int, Any]] = {}
        for i, j in combinations(range(len(matrices)), 2):
            comm = commutator(matrices[i], matrices[j]).flatten()
            coeffs = solve_in_span(flats, comm)
            if coeffs is None:
                raise ValueError("span is not closed under the commutator: [%d, %d]" % (i, j))
            br[(i, j)] = {k: to_exact(c) for k, c in enumerate(coeffs)}
        return cls(len(matrices), br, names=names, name=name, matrices=matrices, check=check)

    # -- elements ---------------------------------------------------------
    def basis_vector(self, i: int) -> Vector:
        return [1 if k == i else 0 for k in range(self.dim)]

    def element(self, **coefficients: Any) -> Vector:
        """Vector from named coefficients, e.g. ``g.element(e=1, f=-2)``."""
        v: Vector = [0] * self.dim
        for key, c in coefficients.items():
            v[self.index(key)] = c
        return v

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError("no basis element named %r" % name) from None

    def format_element(self, v: Sequence[Any]) -> str:
        terms = []
        for c, nm in zip(v, self.names):
            if is_zero(c):
                continue
            s = format_scalar(c)
            terms.append(nm if s == "1" else ("-" + nm if s == "-1" else "(%s) %s" % (s, nm)))
        return " + ".join(terms) if terms else "0"

    # -- brackets ---------------------------------------------------------
    def structure_constant(self, i: int, j: int, k: int) -> Any:
        r""":math:`c_{ij}^{\;k}`."""
        return self._table.get((i, j), {}).get(k, 0)

    def basis_bracket(self, i: int, j: int) -> Vector:
        v: Vector = [0] * self.dim
        for k, c in self._table.get((i, j), {}).items():
            v[k] = c
        return v

    def structure_constants(self) -> Dict[Tuple[int, int, int], Any]:
        r"""All non-zero :math:`c_{ij}^{\;k}` with :math:`i < j`."""
        return {(i, j, k): c for (i, j), t in sorted(self._table.items()) if i < j for k, c in sorted(t.items())}

    def bracket(self, x: Sequence[Any], y: Sequence[Any]) -> Vector:
        r"""The Lie bracket :math:`[x, y] = \sum_{i,j,k} x_i y_j c_{ij}^{\;k} e_k`."""
        out: Vector = [0] * self.dim
        xs = [(i, a) for i, a in enumerate(x) if not is_zero(a)]
        ys = [(j, b) for j, b in enumerate(y) if not is_zero(b)]
        for i, a in xs:
            for j, b in ys:
                t = self._table.get((i, j))
                if t:
                    ab = a * b
                    for k, c in t.items():
                        out[k] = out[k] + ab * c
        return [to_exact(v) for v in out]

    def jacobiator(self, x: Sequence[Any], y: Sequence[Any], z: Sequence[Any]) -> Vector:
        r""":math:`J(x,y,z) = [x,[y,z]] + [y,[z,x]] + [z,[x,y]]`."""
        b = self.bracket
        terms = (b(x, b(y, z)), b(y, b(z, x)), b(z, b(x, y)))
        return [to_exact(p + q + r) for p, q, r in zip(*terms)]

    def jacobi_violations(self, limit: Optional[int] = None) -> List[Tuple[int, int, int, Vector]]:
        """Basis triples :math:`i<j<k` where the Jacobi identity fails (empty for a Lie algebra)."""
        bad = []
        e = self.basis_vector
        for i, j, k in combinations(range(self.dim), 3):
            if not ((i, j) in self._table or (j, k) in self._table or (i, k) in self._table):
                continue
            jac = self.jacobiator(e(i), e(j), e(k))
            if not vec_is_zero(jac):
                bad.append((i, j, k, jac))
                if limit is not None and len(bad) >= limit:
                    break
        return bad

    def satisfies_jacobi(self) -> bool:
        return not self.jacobi_violations(limit=1)

    # -- adjoint and Killing form ----------------------------------------
    def ad(self, x: Sequence[Any]) -> Matrix:
        r"""Matrix of :math:`\operatorname{ad}_x = [x, \cdot\,]`; column :math:`j` is :math:`[x, e_j]`."""
        cols = [self.bracket(x, self.basis_vector(j)) for j in range(self.dim)]
        return Matrix.from_columns(cols)

    def ad_basis(self, i: int) -> Matrix:
        r""":math:`\operatorname{ad}_{e_i}`, with :math:`(\operatorname{ad}_{e_i})_{kj} = c_{ij}^{\;k}` (cached)."""
        if i not in self._ad_cache:
            rows = [[0] * self.dim for _ in range(self.dim)]
            for j in range(self.dim):
                for k, c in self._table.get((i, j), {}).items():
                    rows[k][j] = c
            self._ad_cache[i] = Matrix(rows)
        return self._ad_cache[i]

    def killing_form(self) -> Matrix:
        r"""Gram matrix :math:`\kappa_{ij} = \operatorname{tr}(\operatorname{ad}_{e_i}\operatorname{ad}_{e_j})
        = \sum_{k,l} c_{ik}^{\;l} c_{jl}^{\;k}`."""
        n = self.dim
        by_first: Dict[int, List[Tuple[int, int, Any]]] = {i: [] for i in range(n)}
        for (a, k), t in self._table.items():
            for l, c in t.items():
                by_first[a].append((k, l, c))
        K = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                total: Any = 0
                for k, l, c in by_first[i]:
                    d = self._table.get((j, l), {}).get(k)
                    if d is not None:
                        total = total + c * d
                K[i][j] = K[j][i] = to_exact(total)
        return Matrix(K)

    def killing(self, x: Sequence[Any], y: Sequence[Any]) -> Any:
        r""":math:`\kappa(x, y)`."""
        K = self.killing_form()
        return to_exact(sum((a * v for a, v in zip(x, K @ list(y))), 0))

    def is_semisimple(self) -> bool:
        r"""Cartan's criterion: :math:`\mathfrak g` is semisimple iff :math:`\det \kappa \neq 0`."""
        return self.dim > 0 and not is_zero(self.killing_form().det())

    def casimir_in_adjoint(self) -> Matrix:
        r"""Quadratic Casimir :math:`\Omega = \sum_{ij} \kappa^{ij} \operatorname{ad}_{e_i}\operatorname{ad}_{e_j}`.

        For a simple algebra this is the identity matrix (Freudenthal's normalisation).
        """
        Kinv = self.killing_form().inverse()
        total = Matrix.zeros(self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                if not is_zero(Kinv[i, j]):
                    total = total + (self.ad_basis(i) @ self.ad_basis(j)) * Kinv[i, j]
        return total

    def is_invariant_form(self, B: Matrix) -> bool:
        r"""Check :math:`B([x,y],z) + B(y,[x,z]) = 0`, i.e. :math:`\operatorname{ad}_x^{\mathsf T} B + B\operatorname{ad}_x = 0`."""
        return all((self.ad_basis(i).T @ B + B @ self.ad_basis(i)).is_zero() for i in range(self.dim))

    # -- structure theory -------------------------------------------------
    def bracket_span(self, A: Sequence[Sequence[Any]], B: Sequence[Sequence[Any]]) -> List[Vector]:
        r"""A basis of :math:`[A, B] = \operatorname{span}\{[a, b]\}`."""
        return span_basis([self.bracket(a, b) for a in A for b in B]) if A and B else []

    def full_basis(self) -> List[Vector]:
        return [self.basis_vector(i) for i in range(self.dim)]

    def derived_series(self) -> List[int]:
        r"""Dimensions of :math:`\mathfrak g \supseteq [\mathfrak g,\mathfrak g] \supseteq \cdots` until it stabilises."""
        cur = self.full_basis()
        dims = [len(cur)]
        while cur:
            nxt = self.bracket_span(cur, cur)
            if len(nxt) == len(cur):
                break
            cur = nxt
            dims.append(len(cur))
        return dims

    def lower_central_series(self) -> List[int]:
        r"""Dimensions of :math:`\mathfrak g^{(1)} = \mathfrak g,\ \mathfrak g^{(k+1)} = [\mathfrak g, \mathfrak g^{(k)}]`."""
        g = self.full_basis()
        cur = g
        dims = [len(cur)]
        while cur:
            nxt = self.bracket_span(g, cur)
            if len(nxt) == len(cur):
                break
            cur = nxt
            dims.append(len(cur))
        return dims

    def is_solvable(self) -> bool:
        return self.derived_series()[-1] == 0

    def is_nilpotent(self) -> bool:
        return self.lower_central_series()[-1] == 0

    def is_abelian(self) -> bool:
        return not self._table

    def center(self) -> List[Vector]:
        r"""A basis of :math:`Z(\mathfrak g) = \{z : [z, e_j] = 0\ \forall j\}`."""
        rows: List[List[Any]] = []
        for j in range(self.dim):
            # coefficient of e_k in [z, e_j] is sum_i z_i c_{ij}^k
            for k in range(self.dim):
                rows.append([self.structure_constant(i, j, k) for i in range(self.dim)])
        return Matrix(rows).nullspace()

    def is_homomorphism(self, other: "LieAlgebra", phi: Matrix) -> bool:
        r"""Check that the linear map :math:`\varphi: \mathfrak g \to \mathfrak h` (an
        :math:`\dim\mathfrak h \times \dim\mathfrak g` matrix) satisfies
        :math:`\varphi[e_i,e_j] = [\varphi e_i, \varphi e_j]`."""
        if phi.shape != (other.dim, self.dim):
            raise ValueError("phi must be %d x %d" % (other.dim, self.dim))
        images = [phi.column(i) for i in range(self.dim)]
        for i, j in combinations(range(self.dim), 2):
            lhs = phi @ self.basis_bracket(i, j)
            rhs = other.bracket(images[i], images[j])
            if not vec_is_zero([a - b for a, b in zip(lhs, rhs)]):
                return False
        return True

    def is_isomorphism(self, other: "LieAlgebra", phi: Matrix) -> bool:
        return self.dim == other.dim and phi.is_square and not is_zero(phi.det()) and self.is_homomorphism(other, phi)

    def direct_sum(self, other: "LieAlgebra") -> "LieAlgebra":
        r""":math:`\mathfrak g \oplus \mathfrak h` with :math:`[\mathfrak g, \mathfrak h] = 0`."""
        n = self.dim
        br: Dict[Tuple[int, int], Dict[int, Any]] = {k: dict(v) for k, v in self._table.items()}
        for (i, j), t in other._table.items():
            br[(i + n, j + n)] = {k + n: c for k, c in t.items()}
        names = self.names + [nm + "'" if nm in self.names else nm for nm in other.names]
        return LieAlgebra(n + other.dim, br, names=names, name="%s+%s" % (self.name, other.name), check=False)

    def rescaled(self, scales: Sequence[Any], names: Optional[Sequence[str]] = None) -> "LieAlgebra":
        r"""Same algebra in the basis :math:`e_i' = s_i e_i`:  :math:`c'^{\,k}_{ij} = s_i s_j c_{ij}^{\;k} / s_k`."""
        br = {
            (i, j): {k: Fraction(1) * scales[i] * scales[j] * c / scales[k] for k, c in t.items()}
            for (i, j), t in self._table.items() if i < j
        }
        return LieAlgebra(self.dim, br, names=names or self.names, name=self.name, check=False)

    # -- presentation -----------------------------------------------------
    def bracket_table(self) -> List[str]:
        """Lines ``[a, b] = ...`` for every non-zero bracket with ``a`` before ``b``."""
        lines = []
        for (i, j), _ in sorted(self._table.items()):
            if i < j:
                lines.append("[%s, %s] = %s" % (self.names[i], self.names[j], self.format_element(self.basis_bracket(i, j))))
        return lines

    def summary(self) -> str:
        K = self.killing_form()
        det = K.det()
        out = [
            "Lie algebra %s, dim %d" % (self.name, self.dim),
            "  Jacobi identity: %s" % ("ok" if self.satisfies_jacobi() else "FAILS"),
            "  det(Killing) = %s  (%s)" % (format_scalar(det), "semisimple" if not is_zero(det) else "not semisimple"),
            "  derived series dims: %s" % self.derived_series(),
            "  lower central series dims: %s" % self.lower_central_series(),
            "  centre dim: %d" % len(self.center()),
        ]
        return "\n".join(out)

    def __repr__(self) -> str:
        return "LieAlgebra(%r, dim=%d)" % (self.name, self.dim)
