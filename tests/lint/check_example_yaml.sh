#!/usr/bin/env bash
# pantheonsim.example.yaml and the README's settings table list every settings key, once, and names no key that
# docs/config.md does not document. Checked without building or running anything.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
f=pantheonsim.example.yaml
[[ -f $f ]] || { echo "FAIL  no $f at the repository root"; exit 1; }
fail=0
doc=$(sed -n 's/^| `\([a-z_]*\)` | [^|]* | `VGPU_[A-Z_]*` |.*/\1/p' docs/config.md | sort)
[[ -n $doc ]] || { echo "FAIL  found no keys in docs/config.md"; exit 1; }
# Only lines that are a key (active or commented out once): "key: value" or "# key: value".
ex=$(grep -E '^(# )?[a-z_]+: ' "$f" | sed -E 's/^(# )?([a-z_]+): .*/\2/' | sort)
if [[ $doc == "$ex" ]]; then echo "ok    the example has exactly the documented keys"
else echo "FAIL  keys differ between docs/config.md (<) and $f (>)"; diff <(echo "$doc") <(echo "$ex"); fail=1; fi
dup=$(echo "$ex" | uniq -d)
[[ -z $dup ]] && echo "ok    no key twice" || { echo "FAIL  twice: $dup"; fail=1; }
active=$(grep -E '^[a-z_]+: ' "$f" | sed -E 's/: .*//' | tr '\n' ' ')
[[ $active == "gpu count " ]] && echo "ok    only gpu and count are switched on" || { echo "FAIL  active keys: $active"; fail=1; }
# The README's settings table has exactly the documented keys too.
rd=$(sed -n 's/^| `\([a-z_]*\)` | .*/\1/p' README.md | sort)
if [[ $doc == "$rd" ]]; then echo "ok    the README table has exactly the documented keys"
else echo "FAIL  keys differ between docs/config.md (<) and the README table (>)"; diff <(echo "$doc") <(echo "$rd"); fail=1; fi
exit $fail
