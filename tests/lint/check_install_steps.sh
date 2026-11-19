#!/usr/bin/env bash
# docs/install.md is run by the clean-install workflow (tests/install/run_doc_steps.sh). Checked without installing anything:
#   - the page has every step and file the workflow names, each a non-empty block;
#   - the script the steps make passes `bash -n`, and the extractor takes the right block from a fixture
#     (a step, a file, a block it must not take);
#   - every step name the workflow runs exists, and the default toolkit versions the page installs are in the range
#     the version matrix checks;
#   - the workflow runs on a schedule.
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)
runner=$here/tests/install/run_doc_steps.sh
fail=0
check() { if [[ $2 == ok ]]; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

names=$("$runner" --list)
for s in prereqs build docker cuda hello-cuda rocm hello-hip; do
  check "install.md has step $s" "$([[ $names == *"step $s"* ]] && echo ok)"
  check "step $s is not empty" "$([[ -n $(DOC=$here/docs/install.md "$runner" --print "$s" 2>/dev/null | grep -v "^set \|^cd \|^if \|::") ]] && echo ok)"
done
for f in hello.cu hello.hip; do check "install.md has file $f" "$([[ $names == *"file $f"* ]] && echo ok)"; done

script=$("$runner" --print prereqs build cuda hello-cuda rocm hello-hip 2>&1)
check "the steps make a script bash accepts" "$(bash -n <(printf '%s\n' "$script") 2>/dev/null && echo ok)"
rm -f "$here/hello.cu" "$here/hello.hip"

wf=$here/.github/workflows/clean-install.yml
used=$(sed -n 's/^ *run: tests\/install\/run_doc_steps.sh //p' "$wf")
for s in $used; do check "the workflow's step $s exists" "$([[ $names == *"step $s"* ]] && echo ok)"; done
check "the workflow is nightly" "$(grep -q '^  schedule:' "$wf" && echo ok)"

cuda=$(sed -n 's/^CUDA=\${CUDA:-\(.*\)}$/\1/p' "$here/docs/install.md")
rocm=$(sed -n 's/^ROCM=\${ROCM:-\(.*\)}$/\1/p' "$here/docs/install.md")
check "the default CUDA ($cuda) is one the matrix checks" "$(grep -q "cuda: '$cuda'" "$here/.github/workflows/version-matrix.yml" && echo ok)"
check "the default ROCm ($rocm) is one the matrix checks" "$(grep -q "rocm: '$rocm'" "$here/.github/workflows/version-matrix.yml" && echo ok)"

# The extractor, on a fixture.
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/doc.md" <<'MD'
text
```bash
not_a_step
```
<!-- install-step: one -->
```bash
echo first
```
<!-- install-file: f.txt -->
```
contents
```
<!-- install-step: two -->
```bash
echo second
```
MD
out=$(DOC=$tmp/doc.md "$runner" --print one two 2>&1)
check "the extractor takes the marked block" "$([[ $out == *"echo first"* && $out == *"echo second"* && $out != *not_a_step* ]] && echo ok)"
out=$(DOC=$tmp/doc.md "$runner" --print three 2>&1); rc=$?
check "a step the page lacks fails" "$([[ $rc != 0 && $out == *"no install-step 'three'"* ]] && echo ok)"
rm -f "$here/f.txt"
exit $fail
