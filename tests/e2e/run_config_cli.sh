#!/usr/bin/env bash
# The settings file end to end: a pantheonsim.yaml in the current directory sets
# the GPU and the device count for `vgpu info`, `vgpu run` and `vgpu shell`; a
# real environment variable overrides it, a flag overrides both; `vgpu config
# show` says which is which; a broken file stops the command with its message;
# `vgpu config init` writes a starter and will not overwrite one.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Nothing from the caller's environment may decide the answers.
for v in $(env | sed -n 's/^\(VGPU_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset XDG_CONFIG_HOME
export HOME="$tmp/home"
mkdir -p "$HOME" "$tmp/work"
cd "$tmp/work"

fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
contains() {  # contains <name> <needle> <haystack>
  if [[ "$3" == *"$2"* ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected to contain: $2"; echo "      actual: $3"; fail=1; fi
}
row() { awk -v k="$1" '$1 == k { $1 = $1; sub(/^[^ ]+ /, ""); print }'; }   # a `config show` row, minus its key, single-spaced

# ---- no file -----------------------------------------------------------------
out=$("$vgpu" config show 2>&1); rc=$?
expect "no settings file is fine" "0" "$rc"
contains "show says there is no file and where it looked" "settings file: none (looked for pantheonsim.yaml" "$out"
expect "everything is a default" "0" "$(sed -n '/^key /,$p' <<<"$out" | tail -n +2 | grep -c -E ' (file|environment|flag) ' | tr -d ' ')"

# ---- a file ------------------------------------------------------------------
cat > pantheonsim.yaml <<'EOF'
# a comment
gpu: nvidia/t4          # inline comment
count: 2

strict: true
cuda_version: 12.4
EOF
out=$("$vgpu" config show 2>&1)
contains "show reports the file" "settings file: pantheonsim.yaml" "$out"
contains "gpu comes from line 2" "nvidia/t4 file pantheonsim.yaml:2" "$(row gpu <<<"$out")"
contains "count comes from line 3" "file pantheonsim.yaml:3" "$(row count <<<"$out")"
contains "an untouched key is a default" "default" "$(row threads <<<"$out")"

expect "info uses the file's gpu" "VirtualGPU device: nvidia/t4" "$("$vgpu" info 2>&1 | head -1)"
expect "run puts the file's settings in the program's environment" "VGPU_DEVICE_COUNT=2 VGPU_GPU=nvidia/t4 VGPU_STRICT=1" \
  "$("$vgpu" run env 2>&1 | grep -E '^VGPU_(GPU|DEVICE_COUNT|STRICT)=' | sort | paste -sd' ')"
expect "shell starts the machine the file describes" "nvidia/t4 2 12.4" \
  "$(timeout 60 "$vgpu" shell -y --no-isolate -c 'echo $VGPU_GPU $VGPU_DEVICE_COUNT $VGPU_CUDA_VERSION' </dev/null 2>/dev/null | tail -1)"

# ---- the environment beats the file ------------------------------------------
out=$(VGPU_DEVICE_COUNT=4 VGPU_GPU=nvidia/h100 "$vgpu" config show 2>&1)
contains "show names the variable" "environment VGPU_DEVICE_COUNT" "$(row count <<<"$out")"
contains "and what it overrode" "pantheonsim.yaml:3 says 2" "$(row count <<<"$out")"
expect "run: an environment variable beats the file" "VGPU_DEVICE_COUNT=4 VGPU_GPU=nvidia/h100" \
  "$(VGPU_DEVICE_COUNT=4 VGPU_GPU=nvidia/h100 "$vgpu" run env 2>&1 | grep -E '^VGPU_(GPU|DEVICE_COUNT)=' | sort | paste -sd' ')"
expect "info: an environment variable beats the file" "VirtualGPU device: nvidia/h100" \
  "$(VGPU_GPU=nvidia/h100 "$vgpu" info 2>&1 | head -1)"
expect "shell: an environment variable beats the file" "nvidia/h100 2" \
  "$(VGPU_GPU=nvidia/h100 timeout 60 "$vgpu" shell -y --no-isolate -c 'echo $VGPU_GPU $VGPU_DEVICE_COUNT' </dev/null 2>/dev/null | tail -1)"

# ---- a flag beats both ---------------------------------------------------------
expect "run: a flag beats the environment and the file" "VGPU_DEVICE_COUNT=3 VGPU_GPU=amd/mi300x" \
  "$(VGPU_DEVICE_COUNT=4 VGPU_GPU=nvidia/h100 "$vgpu" run --count 3 --gpu amd/mi300x env 2>&1 | grep -E '^VGPU_(GPU|DEVICE_COUNT)=' | sort | paste -sd' ')"
expect "info: a flag beats the file" "VirtualGPU device: nvidia/h100" \
  "$("$vgpu" info --gpu nvidia/h100 2>&1 | head -1)"
expect "shell: a flag beats the environment and the file" "amd/mi300x 5" \
  "$(VGPU_GPU=nvidia/h100 timeout 60 "$vgpu" shell -y --no-isolate --gpu amd/mi300x --count 5 -c 'echo $VGPU_GPU $VGPU_DEVICE_COUNT' </dev/null 2>/dev/null | tail -1)"
out=$(VGPU_DEVICE_COUNT=4 "$vgpu" config show --count 6 --strict 2>&1)
contains "show: a flag is reported as one" "flag --count" "$(row count <<<"$out")"
contains "show: a switch flag too" "flag --strict" "$(row strict <<<"$out")"
expect "show: a bad flag value is refused" "2" "$("$vgpu" config show --count 99 >/dev/null 2>&1; echo $?)"
contains "show: and says why" "a whole number from 1 to 16" "$("$vgpu" config show --count 99 2>&1)"

# ---- naming a file -------------------------------------------------------------
printf 'gpu: nvidia/a100\ncount: 7\n' > "$tmp/other.yaml"
contains "--config before the command picks the file" "settings file: $tmp/other.yaml" "$("$vgpu" --config "$tmp/other.yaml" config show 2>&1)"
expect "--config applies to the command" "VirtualGPU device: nvidia/a100" "$("$vgpu" --config "$tmp/other.yaml" info 2>&1 | head -1)"
expect "VGPU_CONFIG picks the file" "VirtualGPU device: nvidia/a100" "$(VGPU_CONFIG="$tmp/other.yaml" "$vgpu" info 2>&1 | head -1)"
out=$("$vgpu" --config "$tmp/nope.yaml" info 2>&1); rc=$?
expect "a file named but missing stops the command" "1" "$rc"
contains "and says so" "nope.yaml does not exist" "$out"
mkdir -p "$HOME/.config/pantheonsim"
printf 'gpu: nvidia/a10\n' > "$HOME/.config/pantheonsim/config.yaml"
cd "$tmp"
expect "the user-level file is used when there is no local one" "VirtualGPU device: nvidia/a10" "$("$vgpu" info 2>&1 | head -1)"
cd "$tmp/work"
rm -rf "$HOME/.config"

# ---- a broken file stops the command, loudly -----------------------------------
printf 'gpu: nvidia/t4\ncount: lots\n' > pantheonsim.yaml
out=$("$vgpu" info 2>&1); rc=$?
expect "an invalid value stops the command" "1" "$rc"
contains "with the file and line" "pantheonsim.yaml:2: count:" "$out"
contains "and what is allowed" "a whole number from 1 to 16" "$out"
expect "run does not start the program" "1" "$("$vgpu" run env >/dev/null 2>&1; echo $?)"
expect "list-gpus is stopped too" "1" "$("$vgpu" list-gpus >/dev/null 2>&1; echo $?)"
expect "--help still works" "0" "$("$vgpu" --help >/dev/null 2>&1; echo $?)"
expect "config keys still works" "0" "$("$vgpu" config keys >/dev/null 2>&1; echo $?)"
out=$("$vgpu" config show 2>&1); rc=$?
expect "config show reports the break" "1" "$rc"
contains "with the same message" "pantheonsim.yaml:2: count:" "$out"

printf 'gpuu: nvidia/t4\n' > pantheonsim.yaml
contains "a typo in a key suggests the key" "unknown key 'gpuu'; did you mean 'gpu'?" "$("$vgpu" info 2>&1)"
printf 'gpu: nvidia/h1000\n' > pantheonsim.yaml
out=$("$vgpu" info 2>&1)
contains "a gpu that does not exist is named" "'nvidia/h1000' is not a GPU profile" "$out"
contains "with a near one" "did you mean 'nvidia/" "$out"
printf 'threads 4\n' > pantheonsim.yaml
contains "a line that is not key: value is refused" "pantheonsim.yaml:1: expected \`key: value\`" "$("$vgpu" info --gpu nvidia/t4 2>&1)"
rm -f pantheonsim.yaml

# ---- init ----------------------------------------------------------------------
out=$("$vgpu" config init 2>&1); rc=$?
expect "init writes pantheonsim.yaml" "0 yes" "$rc $([[ -f pantheonsim.yaml ]] && echo yes || echo no)"
cp pantheonsim.yaml starter.copy
contains "the starter is for the default gpu" "gpu: nvidia/h100" "$(cat pantheonsim.yaml)"
expect "the starter is a valid file" "0" "$("$vgpu" config show >/dev/null 2>&1; echo $?)"
contains "and show reads it" "file pantheonsim.yaml:" "$("$vgpu" config show 2>&1 | grep -E '^gpu ')"
echo "gpu: nvidia/t4" > pantheonsim.yaml
out=$("$vgpu" config init 2>&1); rc=$?
expect "init refuses to overwrite" "1" "$rc"
contains "and says how" "--force" "$out"
expect "and leaves the file alone" "gpu: nvidia/t4" "$(cat pantheonsim.yaml)"
"$vgpu" config init --force >/dev/null 2>&1
expect "--force replaces it" "same" "$(cmp -s pantheonsim.yaml starter.copy && echo same || echo different)"
"$vgpu" config init "$tmp/elsewhere/my.yaml" >/dev/null 2>&1
expect "init takes a path" "yes" "$([[ -f $tmp/elsewhere/my.yaml ]] && echo yes || echo no)"
rm -f pantheonsim.yaml

# ---- keys, and the docs that describe them -----------------------------------------
keys=$("$vgpu" config keys 2>&1)
names=$(sed -n 's/^\([a-z_]*\): .*/\1/p' <<<"$keys")
expect "config keys lists the keys" "yes" "$([[ $(wc -w <<<"$names") -ge 20 ]] && echo yes || echo no)"
missing=""
for k in $names; do grep -q "\`$k\`" "$root/docs/config.md" || missing+=" $k"; done
expect "docs/config.md documents every key" "" "$missing"
expect "every key's variable is in docs/cli.md or docs/config.md" "" \
  "$(for e in $(sed -n 's/^ *sets \(VGPU_[A-Z_]*\);.*/\1/p' <<<"$keys"); do grep -q "$e" "$root/docs/config.md" || echo -n " $e"; done)"

exit $fail
