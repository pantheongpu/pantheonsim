#!/usr/bin/env python3
"""Rewrite the `op == "name"_op` if-ladders of amd/src/gcn_exec.cpp as switches on the name's hash.

The interpreter used to tell an instruction from the others by testing it against every rung of a
chain of `if (op == "a"_op || op == "b"_op) {...} else if (...)` -- 217 rungs in vector_alu, 118 more
in more_alu in front of it. This turns each such chain into

    switch (op.hash()) {
      case "a"_op.hash: case "b"_op.hash:
        if (!(op == "a"_op || op == "b"_op)) goto L_n;   // the original condition, verbatim
        {BODY}
        break;
      ...
      default: L_n: ELSE-BODY
    }

so an instruction costs one jump. Semantics are those of the ladder:

  * the first rung that names a hash gets its case label; a name that an earlier rung already has is
    dead code in the ladder and gets no label (the rung is left out if all its names are dead);
  * the original condition is kept and tested in the case, so a hash collision between two different
    names sends the instruction to the else-branch, as the ladder's text comparison does;
  * only a rung whose condition is nothing but `op == "..."_op` joined by `||` is converted; the first
    rung that is anything else (and everything after it) stays an if/else chain, in `default`;
  * `if` statements in a row (not `else if`) are one chain only where each body ends in `return`;
  * a chain whose body has a `break`/`continue`-sensitive shape (a `break` that a switch would capture)
    is left alone.

Usage: ladder-to-switch.py FILE [--min-labels N]   (rewrites FILE in place, prints what it did)
"""
import re
import sys

PURE = re.compile(r'^\s*op\s*==\s*"[^"\\]*"_op(\s*\|\|\s*op\s*==\s*"[^"\\]*"_op)*\s*$')
NAME = re.compile(r'op\s*==\s*"([^"\\]*)"_op')


class Src:
    def __init__(self, t):
        self.t = t

    def skip_ws(self, i):
        """Index of the next code character at or after i (whitespace and comments skipped)."""
        t = self.t
        while i < len(t):
            if t[i].isspace():
                i += 1
            elif t.startswith('//', i):
                i = t.index('\n', i) + 1 if '\n' in t[i:] else len(t)
            elif t.startswith('/*', i):
                i = t.index('*/', i) + 2
            else:
                break
        return i

    def skip_lit(self, i):
        """If a string or char literal starts at i, the index after it; else i."""
        t = self.t
        c = t[i]
        if c == '"':
            if i > 0 and t[i - 1] == 'R':
                raise RuntimeError('raw string')
            i += 1
            while t[i] != '"':
                i += 2 if t[i] == '\\' else 1
            return i + 1
        if c == "'":
            # a digit separator (0x1000'0000) is not a char literal
            if i > 0 and (t[i - 1].isalnum() or t[i - 1] == '_') and i + 1 < len(t) and t[i + 1].isalnum():
                return i
            i += 1
            while t[i] != "'":
                i += 2 if t[i] == '\\' else 1
            return i + 1
        return i

    def match(self, i, open_c, close_c):
        """i is at open_c; the index just past its match."""
        t = self.t
        depth = 0
        while i < len(t):
            if t.startswith('//', i):
                i = t.index('\n', i) + 1
                continue
            if t.startswith('/*', i):
                i = t.index('*/', i) + 2
                continue
            j = self.skip_lit(i)
            if j != i:
                i = j
                continue
            c = t[i]
            if c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise RuntimeError('unbalanced')


def code_only(s):
    """The text with comments and literals blanked, same length (for scanning bodies)."""
    out = list(s)
    src = Src(s)
    i = 0
    while i < len(s):
        if s.startswith('//', i):
            j = s.find('\n', i)
            j = len(s) if j < 0 else j
            for k in range(i, j):
                out[k] = ' '
            i = j
            continue
        if s.startswith('/*', i):
            j = s.index('*/', i) + 2
            for k in range(i, j):
                out[k] = ' ' if s[k] != '\n' else '\n'
            i = j
            continue
        j = src.skip_lit(i)
        if j != i:
            for k in range(i + 1, j - 1):
                out[k] = ' '
            i = j
            continue
        i += 1
    return ''.join(out)


def ends_in_return(body):
    """body is the text between a rung's braces."""
    c = code_only(body).rstrip()
    if not c.endswith(';'):
        return False
    depth = 0
    last = -1   # start of the last depth-0 statement
    for i, ch in enumerate(c[:-1]):
        if ch in '({[':
            depth += 1
        elif ch in ')}]':
            depth -= 1
            if depth == 0 and ch == '}':
                last = i + 1
        elif ch == ';' and depth == 0:
            last = i + 1
    return re.match(r'\s*return\b', c[last:] if last >= 0 else c) is not None


def break_captured(body):
    """True if the body has a `break` not inside a loop/switch of its own (a switch would capture it)."""
    c = code_only(body)
    if not re.search(r'\bbreak\b', c):
        return False
    # stack of brace kinds: 'L' loop/switch, 'O' other
    stack = []
    i = 0
    n = len(c)
    pending = None   # kind for the next '{'
    while i < n:
        m = re.compile(r'\b(for|while|switch|do|break|if|else)\b|[{}]').search(c, i)
        if not m:
            break
        w = m.group(0)
        i = m.end()
        if w in ('for', 'while', 'switch'):
            j = i
            while j < n and c[j].isspace():
                j += 1
            if j < n and c[j] == '(':
                depth = 0
                while j < n:
                    if c[j] == '(':
                        depth += 1
                    elif c[j] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                k = j + 1
                while k < n and c[k].isspace():
                    k += 1
                if k < n and c[k] == '{':
                    pending = 'L'
                elif k < n and c[k] == ';':
                    pass
                else:
                    return True   # an unbraced loop body: not analysed
                i = j + 1
        elif w == 'do':
            pending = 'L'
        elif w == '{':
            stack.append(pending or 'O')
            pending = None
        elif w == '}':
            stack.pop()
        elif w == 'break':
            if 'L' not in stack:
                return True
    return False


def stmt_end(src, j):
    """Index just past the simple statement starting at j (up to its `;` at depth 0), or None."""
    t = src.t
    if re.compile(r'(if|for|while|do|switch|try)\b|\{').match(t, j):
        return None
    depth = 0
    i = j
    while i < len(t):
        if t.startswith('//', i):
            i = t.index('\n', i) + 1
            continue
        if t.startswith('/*', i):
            i = t.index('*/', i) + 2
            continue
        k = src.skip_lit(i)
        if k != i:
            i = k
            continue
        c = t[i]
        if c in '({[':
            depth += 1
        elif c in ')}]':
            depth -= 1
        elif c == ';' and depth == 0:
            return i + 1
        i += 1
    return None


def parse_chain(src, start):
    """Parse an if-chain at `start` (the 'i' of `if`). Returns (rungs, else_part, chain_end, else_end) or None.

    A rung is dict(cond, body, gap, seq, pure, at, end). `else_part` is (comments, "{...}") or None.
    """
    t = src.t
    rungs = []
    i = start
    gap = ''
    seq = False
    ifre = re.compile(r'if\s*\(')
    while True:
        m = ifre.match(t, i)
        p = m.end() - 1
        pe = src.match(p, '(', ')')
        cond = t[p + 1:pe - 1]
        j = src.skip_ws(pe)
        if t[j] == '{':
            be = src.match(j, '{', '}')
            body = t[j + 1:be - 1]
        else:
            be = stmt_end(src, j)
            if be is None:
                return None
            body = t[j:be]
        rungs.append(dict(cond=cond, body=body, gap=gap, seq=seq, pure=bool(PURE.match(cond)), at=i, end=be))
        k2 = src.skip_ws(be)
        if t.startswith('else', k2) and not (t[k2 + 4].isalnum() or t[k2 + 4] == '_'):
            after = src.skip_ws(k2 + 4)
            comments = t[be:k2].strip()
            if ifre.match(t, after):
                gap = (comments + '\n') if comments else ''
                seq = False
                i = after
                continue
            if t[after] == '{':
                ee = src.match(after, '{', '}')
                return rungs, (comments, t[after:ee]), be, ee
            ee = stmt_end(src, after)
            if ee is None:
                return None
            return rungs, (comments, '{' + t[after:ee] + '}'), be, ee
        if ifre.match(t, k2) and ends_in_return(body):
            gap = t[be:k2].strip()
            gap = (gap + '\n') if gap else ''
            seq = True
            i = k2
            continue
        return rungs, None, be, be


LABEL = [0]
STATS = dict(chains=0, rungs=0, labels=0, dead=0, skipped_break=0, skipped_small=0, nested=0)
MIN_LABELS = 4


def indent(text, n):
    pad = ' ' * n
    return text.replace('\n', '\n' + pad)


def gen(rungs, else_part):
    """C++ for a chain: a list of rung dicts and an optional else body ("{...}")."""
    if not rungs:
        return else_part[1] if else_part else ''
    k = 0
    while k < len(rungs) and rungs[k]['pure']:
        k += 1
    labels_in_run = sum(len(NAME.findall(r['cond'])) for r in rungs[:k])
    if k and labels_in_run >= MIN_LABELS and not any(break_captured(r['body']) for r in rungs[:k]):
        LABEL[0] += 1
        lab = f'L_ladder_{LABEL[0]}'
        seen = set()
        out = []
        for r in rungs[:k]:
            labels = []
            for n in NAME.findall(r['cond']):
                if n not in seen:
                    seen.add(n)
                    labels.append(n)
                else:
                    STATS['dead'] += 1
            if r['gap'].strip():
                out.append('  ' + indent(r['gap'].strip(), 2))
            if not labels:
                out.append('  // (a rung whose names are all an earlier rung\'s: it could not be reached)')
                continue
            out.append('  ' + ' '.join(f'case "{n}"_op.hash:' for n in labels))
            out.append(f'    if (!({r["cond"].strip()})) goto {lab};')
            out.append('    {' + r['body'] + '}')
            out.append('    break;')
            STATS['labels'] += len(labels)
        rest = gen(rungs[k:], else_part)
        if rest.strip():
            gap = rungs[k]['gap'].strip() if k < len(rungs) else ''
            out.append(f'  default: {lab}:')
            if gap:
                out.append('    ' + indent(gap, 4))
            out.append('    ' + indent(rest, 4) + '\n    break;')
        else:
            out.append(f'  default: {lab}: break;')
        STATS['rungs'] += k
        return 'switch (op.hash()) {\n' + '\n'.join(out) + '\n}'
    # the first rung stays an `if`; the rest of the chain is its else
    r = rungs[0]
    head = 'if (' + r['cond'] + ') {' + r['body'] + '}'
    if len(rungs) == 1 and not else_part:
        return head
    rest = gen(rungs[1:], else_part)
    gap = rungs[1]['gap'].strip() if len(rungs) > 1 else ''
    if len(rungs) > 1 and not rest.startswith('switch'):
        # keep a plain `else if` chain flat
        return head + (' ' + gap.replace('\n', ' ') + '\n' if False else '') + ' else ' + rest
    STATS['nested'] += 1
    return head + ' else {\n' + (gap + '\n' if gap else '') + '  ' + indent(rest, 2) + '\n}'


def main():
    global MIN_LABELS
    path = sys.argv[1]
    if '--min-labels' in sys.argv:
        MIN_LABELS = int(sys.argv[sys.argv.index('--min-labels') + 1])
    text = open(path).read()
    starts = []
    for m in re.finditer(r'\bif\s*\(', text):
        if text[:m.start()].rstrip().endswith('else'):
            continue
        ls = text.rfind('\n', 0, m.start()) + 1
        if text[ls:m.start()].lstrip().startswith('//'):
            continue
        starts.append(m.start())
    # a rung that continues a chain (`if` statements in a row, each ending in return) is not a start
    covered = set()
    for s in starts:
        try:
            parsed = parse_chain(Src(text), s)
        except (RuntimeError, ValueError, IndexError):
            continue
        if parsed:
            covered.update(r['at'] for r in parsed[0][1:])
    starts = [s for s in starts if s not in covered]
    for s in sorted(starts, reverse=True):
        try:
            parsed = parse_chain(Src(text), s)
        except (RuntimeError, ValueError, IndexError):
            continue
        if not parsed:
            continue
        rungs, else_part, chain_end, else_end = parsed
        if not rungs[0]['pure'] and not any(r['pure'] for r in rungs[1:]):
            continue
        before = STATS['labels']
        new = gen(rungs, else_part)
        if STATS['labels'] == before:
            continue
        STATS['chains'] += 1
        end = else_end if else_part else chain_end
        text = text[:s] + new + text[end:]
    open(path, 'w').write(text)
    print(STATS)


if __name__ == '__main__':
    main()
