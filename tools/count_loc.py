"""Count physical and code-only (non-blank, non-comment, non-docstring) lines."""
import io, sys, tokenize
from pathlib import Path

def code_lines(path):
    src = path.read_text()
    lines = set()
    prev = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING and prev in (tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT, tokenize.NL):
            prev = tok.type
            continue  # docstring / bare string statement
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            lines.update(range(tok.start[0], tok.end[0] + 1))
        if tok.type not in (tokenize.COMMENT, tokenize.NL):
            prev = tok.type
    return len(lines)

for group in ("src", "tests"):
    files = sorted(Path(group).rglob("*.py"))
    phys = sum(len(f.read_text().splitlines()) for f in files)
    code = sum(code_lines(f) for f in files)
    print("%-6s files=%2d physical=%5d code=%5d" % (group, len(files), phys, code))
