"""Command-line interface: ``liebracket <command> ...``.

Examples::

    liebracket algebra su3
    liebracket cartan G2
    liebracket roots F4
    liebracket dim A2 1,1
    liebracket weights B2 0,1
    liebracket tensor A2 1,0 0,1
    liebracket bch --degree 5
    liebracket demo g2
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from . import __version__
from .bch import bch_lie_terms, format_lie_terms
from .cartan import cartan_matrix, dynkin_diagram, is_finite_type
from .classical import by_name
from .examples import DEMOS, run_demo
from .roots import root_system
from .scalars import format_scalar
from .weights import WeightSpace


def _parse_weight(text: str, rank: int) -> List[int]:
    try:
        w = [int(x) for x in text.replace("(", "").replace(")", "").split(",") if x.strip()]
    except ValueError:
        raise SystemExit("error: weight %r must be comma-separated integers" % text)
    if len(w) != rank:
        raise SystemExit("error: weight %r needs %d Dynkin labels" % (text, rank))
    return w


def cmd_algebra(args: argparse.Namespace) -> str:
    g = by_name(args.name)
    out = [g.summary()]
    if args.brackets:
        out.append("brackets:")
        out += ["  " + s for s in g.bracket_table()]
    if args.killing:
        out.append("Killing form:")
        out.append(g.killing_form().pretty())
    return "\n".join(out)


def cmd_cartan(args: argparse.Namespace) -> str:
    A = cartan_matrix(args.type)
    return "\n".join(
        [
            "Cartan matrix of %s  (A_ij = <alpha_i^vee, alpha_j>):" % args.type.upper(),
            A.pretty(),
            "det = %s, finite type: %s" % (format_scalar(A.det()), is_finite_type(A)),
            "Dynkin diagram:",
            dynkin_diagram(A),
        ]
    )


def cmd_roots(args: argparse.Namespace) -> str:
    rs = root_system(args.type)
    out = [rs.summary()]
    if args.list:
        out.append("positive roots (simple-root coordinates, by height):")
        for beta in rs.positive_roots:
            out.append("  ht %-2d %s" % (sum(beta), beta))
    return "\n".join(out)


def cmd_dim(args: argparse.Namespace) -> str:
    W = WeightSpace.of_type(args.type)
    lam = _parse_weight(args.weight, W.rank)
    return "dim V%s = %d" % (W.format_weight(lam), W.weyl_dimension(lam))


def cmd_weights(args: argparse.Namespace) -> str:
    W = WeightSpace.of_type(args.type)
    lam = _parse_weight(args.weight, W.rank)
    char = W.character(lam)
    out = ["V%s of %s: dim %d, %d distinct weights" % (W.format_weight(lam), args.type.upper(), sum(char.values()), len(char))]
    items = W.dominant_character(lam).items() if args.dominant else sorted(char.items(), key=lambda t: (W.depth(lam, t[0]), t[0]))
    for mu, m in items:
        out.append("  %-20s mult %d" % (W.format_weight(mu), m))
    return "\n".join(out)


def cmd_tensor(args: argparse.Namespace) -> str:
    W = WeightSpace.of_type(args.type)
    a = _parse_weight(args.left, W.rank)
    b = _parse_weight(args.right, W.rank)
    dec = W.tensor_product(a, b)
    return "V%s x V%s = %s" % (W.format_weight(a), W.format_weight(b), W.format_decomposition(dec))


def cmd_bch(args: argparse.Namespace) -> str:
    terms = bch_lie_terms(args.degree)
    if args.table:
        return "\n".join("%-10s %s" % (c, _fmt(w)) for c, w in terms)
    return "log(e^X e^Y) = " + format_lie_terms(terms) + " + O(%d)" % (args.degree + 1)


def _fmt(w: Sequence[int]) -> str:
    from .bch import format_word

    return format_word(w)


def cmd_demo(args: argparse.Namespace) -> str:
    if args.name == "all":
        return "\n\n".join(run_demo(k) for k in DEMOS)
    return run_demo(args.name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="liebracket", description="Exact Lie algebra and representation theory numerics.")
    p.add_argument("--version", action="version", version="liebracket " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("algebra", help="structure of a matrix Lie algebra (su2, so3, su3, slN, soN, spN, glN, heisN)")
    s.add_argument("name")
    s.add_argument("--brackets", action="store_true", help="print the bracket table")
    s.add_argument("--killing", action="store_true", help="print the Killing form")
    s.set_defaults(func=cmd_algebra)

    s = sub.add_parser("cartan", help="Cartan matrix and Dynkin diagram of a type such as G2")
    s.add_argument("type")
    s.set_defaults(func=cmd_cartan)

    s = sub.add_parser("roots", help="root system data")
    s.add_argument("type")
    s.add_argument("--list", action="store_true", help="list positive roots")
    s.set_defaults(func=cmd_roots)

    s = sub.add_parser("dim", help="Weyl dimension of an irreducible module")
    s.add_argument("type")
    s.add_argument("weight", help="Dynkin labels, e.g. 1,0,2")
    s.set_defaults(func=cmd_dim)

    s = sub.add_parser("weights", help="weight multiplicities (Freudenthal)")
    s.add_argument("type")
    s.add_argument("weight")
    s.add_argument("--dominant", action="store_true", help="show dominant weights only")
    s.set_defaults(func=cmd_weights)

    s = sub.add_parser("tensor", help="decompose a tensor product of two irreducibles")
    s.add_argument("type")
    s.add_argument("left")
    s.add_argument("right")
    s.set_defaults(func=cmd_tensor)

    s = sub.add_parser("bch", help="Baker-Campbell-Hausdorff series")
    s.add_argument("--degree", type=int, default=4)
    s.add_argument("--table", action="store_true", help="one coefficient per line")
    s.set_defaults(func=cmd_bch)

    s = sub.add_parser("demo", help="run a worked example")
    s.add_argument("name", choices=sorted(DEMOS) + ["all"])
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        print(args.func(args))
    except (ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
