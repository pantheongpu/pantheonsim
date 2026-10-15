#!/usr/bin/env bash
# The README is the front door, so every command, name and link it shows must be real.
# Checked without building or running anything:
#   - `make <target>` names a target the Makefile defines, and `make ... VAR=value`
#     uses a variable the Makefile reads;
#   - `vgpu <command>` (and `vgpu config <word>`) is a command src/cli/main.cpp
#     dispatches (and src/cli/config_cmd.cpp);
#   - a `--flag` on a line that runs make or vgpu is documented in docs/cli.md or docs/config.md;
#   - a VGPU_* variable is documented in docs/cli.md or docs/config.md, and a settings-file
#     key (in a yaml block, as `key: value`, or an underscored name like `vram_mb`) is in
#     the key tables of docs/config.md;
#   - a link to a file in the repository points at a file that exists, and a `#anchor`
#     on a .md file is a heading (or <a id>) of that file.
# Usage: tests/lint/check_readme_commands.sh [README.md]   (links resolve from the repository root)
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
readme="${1:-README.md}"
[[ -f $readme ]] || { echo "FAIL  no such file: $readme"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fail=0
bad() { echo "FAIL  $*"; fail=1; }

# --- what the README shows as code -----------------------------------------------------
# code.txt: every line of a fenced block, and every `inline span` (spans that wrap over a
# line break are joined first). yaml.txt: the lines of the fenced yaml blocks.
awk -v code="$tmp/code.txt" -v yaml="$tmp/yaml.txt" '
  function flush(   t, a) {
    t = para; para = ""
    while (match(t, /`[^`]+`/)) { print substr(t, RSTART + 1, RLENGTH - 2) > code; t = substr(t, RSTART + RLENGTH) }
  }
  /^[ \t]*```/ {
    flush()
    if (infence) infence = 0
    else { infence = 1; lang = $0; sub(/^[ \t]*```/, "", lang) }
    next
  }
  infence { print > code; if (lang == "yaml") print > yaml; next }
  /^[ \t]*$/ { flush(); next }
  { para = para " " $0 }
  END { flush() }
' "$readme"
touch "$tmp/code.txt" "$tmp/yaml.txt"

# --- make targets and variables ---------------------------------------------------------
targets=$(sed -n 's/^\([a-z][a-z-]*\):\([^=]\|$\).*/\1/p' Makefile)
while IFS= read -r line; do
  [[ $line =~ (^|[^[:alnum:]_/.-])make([[:space:]]|$) ]] || continue
  while read -r t; do
    [[ -z $t ]] && continue
    grep -qxF -- "$t" <<<"$targets" || bad "README shows 'make $t', but the Makefile has no target '$t'  (in: $line)"
  done < <(grep -oE '(^|[^[:alnum:]_/.-])make +[a-z][a-z-]*' <<<"$line" | awk '{print $NF}')
  while read -r v; do
    [[ -z $v ]] && continue
    grep -qw -- "$v" Makefile || bad "README gives '$v=' to make, but the Makefile does not read $v  (in: $line)"
  done < <(grep -oE '(^|[[:space:]])[A-Z][A-Z_]*=' <<<"$line" | tr -d ' \t=')
done < "$tmp/code.txt"
# a span that is just VAR=value is a make variable
while IFS= read -r line; do
  v=${line%%=*}
  grep -qw -- "$v" Makefile || bad "README shows '$line', but the Makefile does not read $v"
done < <(grep -E '^[A-Z][A-Z_]*=' "$tmp/code.txt")

# --- vgpu commands ----------------------------------------------------------------------
cmds=$(sed -n 's/.*cmd == "\([a-z][a-z-]*\)".*/\1/p' src/cli/main.cpp | sort -u)
cfg=$(sed -n 's/.*sub == "\([a-z][a-z-]*\)".*/\1/p' src/cli/config_cmd.cpp | sort -u)
[[ -n $cmds && -n $cfg ]] || bad "could not read the command list from src/cli/main.cpp or src/cli/config_cmd.cpp"
while read -r c; do
  [[ -z $c ]] && continue
  grep -qxF -- "$c" <<<"$cmds" || bad "README shows 'vgpu $c', but src/cli/main.cpp has no such command"
done < <(grep -oE '(^|[^[:alnum:]_-])vgpu +[a-z][a-z-]*' "$tmp/code.txt" | awk '{print $NF}')
while read -r c; do
  [[ -z $c ]] && continue
  grep -qxF -- "$c" <<<"$cfg" || bad "README shows 'vgpu config $c', but src/cli/config_cmd.cpp has no such subcommand"
done < <(grep -oE '(^|[^[:alnum:]_-])vgpu +config +[a-z]+' "$tmp/code.txt" | awk '{print $NF}')

# --- flags on make and vgpu lines ----------------------------------------------------------
while read -r f; do
  [[ -z $f ]] && continue
  grep -qF -- "$f" docs/cli.md docs/config.md || bad "README shows the flag '$f' with make or vgpu, but docs/cli.md and docs/config.md do not mention it"
done < <(awk 'match($0, /(^|[^[:alnum:]_\/.-])(make|vgpu)([ \t]|$)/) { print substr($0, RSTART) }' "$tmp/code.txt" |
         grep -oE '(^|[[:space:]])--[a-z][a-z-]*' | tr -d ' \t' | sort -u)

# --- environment variables and settings-file keys ---------------------------------------------
while read -r v; do
  [[ -z $v ]] && continue
  grep -qF -- "$v" docs/cli.md docs/config.md || bad "README names $v, but docs/cli.md and docs/config.md do not document it"
done < <(grep -oE 'VGPU_[A-Z0-9_]+' "$readme" | sort -u)

keys=$(sed -n 's/^| `\([a-z_]*\)` | .*/\1/p' docs/config.md | sort -u)
[[ -n $keys ]] || bad "could not read the key tables from docs/config.md"
{ sed -n 's/^\([a-z_][a-z_]*\):.*/\1/p' "$tmp/yaml.txt"
  sed -n 's/^\([a-z_][a-z_]*\): .*/\1/p' "$tmp/code.txt"
  grep -E '^[a-z]+(_[a-z]+)+$' "$tmp/code.txt"
} | sort -u | while read -r k; do
  [[ -z $k ]] && continue
  grep -qxF -- "$k" <<<"$keys" || echo "FAIL  README names the settings key '$k', but docs/config.md has no such key" >&2
done 2> "$tmp/keyerr"
if [[ -s $tmp/keyerr ]]; then cat "$tmp/keyerr"; fail=1; fi

# --- links ---------------------------------------------------------------------------------
# slugs <file>: GitHub's anchor for each heading outside code fences, plus every <a id="...">.
slugs() {
  awk '/^```/ { f = !f; next } !f && /^#+ / { sub(/^#+ +/, ""); print }' "$1" |
    tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9 _-]//g; s/ /-/g'
  grep -oE '<a id="[^"]+"' "$1" | sed -E 's/<a id="([^"]+)"/\1/'
}
while read -r target; do
  [[ -z $target ]] && continue
  case $target in http://*|https://*|mailto:*) continue ;; esac
  path=${target%%#*}
  anchor=""
  [[ $target == *'#'* ]] && anchor=${target#*#}
  file=$path
  [[ -z $path ]] && file=$readme
  if [[ ! -e $file ]]; then bad "README links to '$target', but $file does not exist"; continue; fi
  if [[ -n $anchor && $file == *.md ]]; then
    slugs "$file" | grep -qxF -- "$anchor" || bad "README links to '$target', but $file has no heading or anchor '$anchor'"
  fi
done < <(grep -oE '\]\([^)[:space:]]+\)' "$readme" | sed -E 's/^\]\((.*)\)$/\1/' | sort -u)

if (( fail )); then echo "check_readme_commands: FAILED"; exit 1; fi
echo "ok    $readme: every make target, vgpu command, flag, variable, settings key and link exists"
