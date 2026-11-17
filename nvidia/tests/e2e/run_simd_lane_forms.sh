#!/usr/bin/env bash
# The plain ALU and tensor-core forms the simulator's SIMD kernels run (simd_lane_forms.cu), hashed over values that
# break arithmetic, on both engines at every SIMD level (VGPU_SIMD unset, =base, =off), against what the simulator
# printed before the kernels existed:
#
#   nvidia/tests/data/simd_lane_forms.sass.txt, .ptx.txt   the simulator's own hashes, pinned (a kernel that changes
#                                                           a bit anywhere fails here)
#   nvidia/tests/data/simd_lane_forms.rtx3060.txt          an RTX 3060's hashes. The simulator agrees with it on every line, on both
#                                                           engines: the integer, float, half and double forms, and the f16, bf16 and
#                                                           tf32 tensor core over full-range random data and over the special values
#                                                           (the tensor core sums in chunks with truncation, include/vgpu/exec/
#                                                           tc_ampere.hpp; the sequential f32 sum this simulator used before differed
#                                                           from the card on the random lines). So the pinned files ARE the card's
#                                                           lines, and this script requires it.
#
#   nvidia/tests/data/simd_lane_forms.sass.tcseq.txt, .ptx.tcseq.txt   what VGPU_TC_EXACT=0 prints (the tensor cores' earlier sequential f32
#                                                           sum, on sm_86 too): the sass file is r8-a's pinned file byte for byte; the ptx
#                                                           file is r8-a's with its two wmma tf32 lines replaced (random, edges), because
#                                                           the PTX engine no longer rounds a tf32 operand when wmma.load reads it -- it
#                                                           truncates, as the SASS executor and the card do -- a fix that is not part of
#                                                           the sum model and is not switched. Every other line is the same in both modes.
#
#   nvidia/tests/data/simd_lane_forms.t4.txt               a Tesla T4's lines (sm_75, built with -DNO80: 56 lines, no bf16 or tf32 tensor form).
#                                                           They are the RTX 3060's lines, which this script checks.
#
#   nvidia/tests/data/simd_lane_forms.h100.txt             an H100 PCIe's lines (sm_90; the same from sm_80, sm_86 and sm_89 builds run on it).
#                                                           The 53 lines that are not tensor-core sums are the RTX 3060's, and so is
#                                                           wmma_f16_f32acc edges; the other seven wmma lines are its own: an H100
#                                                           sums in chunks of 16 products and cuts one bit lower (tc_ampere.hpp).
#
# The other architectures (docs/matrix-numerics.md): an A10G (sm_86), an A100 (sm_80), an L4 and an L40S (sm_86 and sm_89 builds) print
# the RTX 3060's 60 lines and a Tesla T4 its 56, measured 2026-11-15. So the same program built for sm_75 (-DNO80), sm_80 and sm_89 must
# print the card's lines on the t4 / rtx2080ti, a100 and l4 / l40s profiles on both engines, and the sequential pins with
# VGPU_TC_EXACT=0. An sm_90 build on the h100 profile must print the H100's lines, and so must an sm_86 build there: the device chooses
# the sum, not what the cubin was built for. The b200 and gb200 profiles sum by the Hopper rule by default (inferred from data recorded by
# others on a B200, 2026-11-16) and print the H100's lines. So do the compute capability 12.0 profiles since 2026-11-17: an RTX PRO 6000
# Blackwell Server Edition (AWS g7e.2xlarge, driver 595.112.07, nvcc 13.0 and 13.2) printed the H100's 60 lines, the same from sm_86, sm_90,
# sm_100 and sm_120 builds (rtx-pro-6000-server: measured; rtx5090, rtx-pro-4500, rtx-pro-6000, rtx-pro-6000-max-q: inferred from it).
# Only b300, vr200, thor and gb10 keep the sequential sum (no recorded vectors, no card) and print the sequential pins by default.
#
#   run_simd_lane_forms.sh                  simulator vs the pinned files, with the tensor-core model on (the default) and off
#   run_simd_lane_forms.sh --card [--update]   NVIDIA's runtime on a real GPU vs the card file (or rewrite it). The program is built for
#                                              the card's own architecture (nvidia-smi's compute_cap; VGPU_CARD_ARCH=86 overrides), with
#                                              -DNO80 below sm_80, where the four bf16 and tf32 lines are left out of the comparison;
#                                              an sm_90 or sm_120 card is held to the H100's file. $VGPU_CARD_OUT, if set, receives the card's lines.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
data="$root/nvidia/tests/data"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/simd_lane_forms.cu"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_simd_lane_forms.XXXXXX")"
trap 'rm -rf "$out"' EXIT
build() { nvcc -std=c++17 -O1 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$@" "$src" -o "$out/app"; }
# The same program for another architecture: build_arch <sm> <output> [nvcc flags]; below sm_80 without the bf16 and tf32 tensor forms.
build_arch() {
  local a="$1" o="$2"; shift 2
  local no80=(); (( a < 80 )) && no80=(-DNO80)
  nvcc -std=c++17 -O1 -cudart shared -arch="sm_$a" -Wno-deprecated-gpu-targets "${no80[@]}" "$@" "$src" -o "$o"
}
# A pinned file as a build for <sm> prints it: without the four bf16 and tf32 tensor lines below sm_80.
want_for() { if (( $1 < 80 )); then grep -v -e '^wmma_bf16_' -e '^wmma_tf32_' "$2"; else cat "$2"; fi; }
if [[ "${1:-}" == --card ]]; then
  cap="${VGPU_CARD_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')}"
  [[ -n "$cap" ]] || cap=86
  build_arch "$cap" "$out/app"
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  LD_LIBRARY_PATH="$cuda_root/lib64" "$out/app" > "$out/card.txt"
  if [[ "${2:-}" == --update ]]; then
    (( cap == 86 )) || { echo "--update rewrites the RTX 3060's file: run it on an sm_86 card (this one is sm_$cap)"; exit 1; }
    cp "$out/card.txt" "$data/simd_lane_forms.rtx3060.txt"; echo "wrote simd_lane_forms.rtx3060.txt"; exit 0
  fi
  [[ -n "${VGPU_CARD_OUT:-}" ]] && cp "$out/card.txt" "$VGPU_CARD_OUT"
  card_file="$data/simd_lane_forms.rtx3060.txt"; (( cap == 90 || cap == 120 )) && card_file="$data/simd_lane_forms.h100.txt"
  want_for "$cap" "$card_file" > "$out/want.txt"
  diff "$out/want.txt" "$out/card.txt" && { echo "PASS (card, sm_$cap, $(wc -l < "$out/card.txt") lines, $(basename "$card_file"))"; exit 0; }
  echo "FAIL: the card (sm_$cap) disagrees with $(basename "$card_file")"; exit 1
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
build $(shim_sanitizer_nvcc_flags "$shim")
if ! require_shim_libs "$shim" "$out/app"; then exit 0; fi
fails=0
for engine in sass ptx; do
  for level in default base off; do
    case "$level" in default) lv=();; *) lv=(VGPU_SIMD="$level");; esac
    env "${lv[@]}" VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) \
        LD_LIBRARY_PATH="$shim" "$out/app" > "$out/sim.txt" 2>&1 || true
    cp "$data/simd_lane_forms.$engine.txt" "$out/want.txt"
    if diff "$out/want.txt" "$out/sim.txt" > "$out/d.txt"; then
      echo "ok   $engine VGPU_SIMD=$level"
    else
      echo "FAIL $engine VGPU_SIMD=$level: differs from the pinned hashes"; head -10 "$out/d.txt"; fails=1
    fi
  done
done
# VGPU_TC_EXACT=0: the earlier sequential sums, every engine and level; and a mutation guard: the two modes must differ on exactly the
# tensor-core lines (a switch an engine ignores would print the card's lines here).
for engine in sass ptx; do
  for level in default base off; do
    case "$level" in default) lv=();; *) lv=(VGPU_SIMD="$level");; esac
    env "${lv[@]}" VGPU_TC_EXACT=0 VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) \
        LD_LIBRARY_PATH="$shim" "$out/app" > "$out/sim.txt" 2>&1 || true
    if diff "$data/simd_lane_forms.$engine.tcseq.txt" "$out/sim.txt" > "$out/d.txt"; then
      echo "ok   $engine VGPU_SIMD=$level VGPU_TC_EXACT=0"
    else
      echo "FAIL $engine VGPU_SIMD=$level VGPU_TC_EXACT=0: differs from the sequential-sum pins"; head -10 "$out/d.txt"; fails=1
    fi
  done
  # (the lines that differ between the two modes are the wmma and mma tensor lines, and only those)
  if diff "$data/simd_lane_forms.$engine.tcseq.txt" "$data/simd_lane_forms.$engine.txt" | grep '^[<>]' | grep -v -e 'wmma_' -e 'mma_' > "$out/d.txt"; then
    echo "FAIL: the modes differ on lines that are not tensor-core sums (sass/ptx $engine):"; head -5 "$out/d.txt"; fails=1
  fi
done
# The pinned files are the card's lines: SASS == PTX == card.
for engine in sass ptx; do
  if ! diff "$data/simd_lane_forms.rtx3060.txt" "$data/simd_lane_forms.$engine.txt" > "$out/d.txt"; then
    echo "FAIL: simd_lane_forms.$engine.txt differs from the RTX 3060's lines:"; head -10 "$out/d.txt"; fails=1
  fi
done
# The T4's own lines are the RTX 3060's without the four bf16 and tf32 tensor lines.
if ! diff <(want_for 75 "$data/simd_lane_forms.rtx3060.txt") "$data/simd_lane_forms.t4.txt" > "$out/d.txt"; then
  echo "FAIL: simd_lane_forms.t4.txt differs from the RTX 3060's lines:"; head -10 "$out/d.txt"; fails=1
fi
# The other architectures. run_case <label> <app> <sm> <profile> <engine> <pinned file> [ENV=...]
run_case() {
  local label="$1" app="$2" a="$3" gpu="$4" engine="$5" pin="$6"; shift 6
  env "$@" VGPU_QUIET=1 VGPU_GPU="nvidia/$gpu" VGPU_SASS=$([[ $engine == sass ]] && echo 1 || echo 0) LD_LIBRARY_PATH="$shim" "$app" > "$out/sim.txt" 2>&1 || true
  want_for "$a" "$pin" > "$out/want.txt"
  # An sm_75 cubin's f16 accumulator, summed sequentially: wmma 16x16x16 is two HMMA.1688.F16 there, so the sum is rounded to f16 after
  # eight products as well as after sixteen (sm_80 and later: one HMMA.16816.F16, rounded once, which is what the pin holds). Rounded
  # every eight, the sequential sum's two f16-accumulator lines are the card's.
  if [[ $a == 75 && $engine == sass && $pin == *tcseq* ]]; then
    { grep -v '^wmma_f16_f16acc' "$out/want.txt"; grep '^wmma_f16_f16acc' "$data/simd_lane_forms.rtx3060.txt"; } > "$out/want2.txt"
    # (back in the file's order: the line order of the card file)
    awk 'NR == FNR { n[$1 " " $2] = NR; next } { print n[$1 " " $2], $0 }' "$data/simd_lane_forms.rtx3060.txt" "$out/want2.txt" | sort -n | cut -d' ' -f2- > "$out/want.txt"
  fi
  if diff "$out/want.txt" "$out/sim.txt" > "$out/d.txt"; then
    echo "ok   sm_$a $gpu $engine $label"
  else
    echo "FAIL sm_$a $gpu $engine $label: differs from $(basename "$pin")"; head -10 "$out/d.txt"; fails=1
  fi
}
cp "$out/app" "$out/app86"
for a in 75 80 89 90 100 120; do
  if ! build_arch "$a" "$out/app$a" $(shim_sanitizer_nvcc_flags "$shim") 2> "$out/build$a.txt"; then
    echo "SKIP sm_$a: this toolkit does not build it ($(head -1 "$out/build$a.txt"))"; rm -f "$out/app$a"
  fi
done
# Measured on a card: the card's lines by default, the sequential pins with the model off.
for pair in 75:t4 75:rtx2080ti 80:a100 86:a10g 89:l4 89:l40s 80:rtx3060 86:l4 90:h100 86:h100 89:h100; do
  a="${pair%%:*}"; gpu="${pair##*:}"
  [[ -x "$out/app$a" ]] || continue
  card="$data/simd_lane_forms.rtx3060.txt"
  [[ $gpu == h100 ]] && card="$data/simd_lane_forms.h100.txt"
  for engine in sass ptx; do
    run_case "card lines" "$out/app$a" "$a" "$gpu" "$engine" "$card"
    run_case "VGPU_TC_EXACT=0" "$out/app$a" "$a" "$gpu" "$engine" "$data/simd_lane_forms.$engine.tcseq.txt" VGPU_TC_EXACT=0
    [[ $gpu == h100 ]] && run_case "VGPU_TC_EXACT=force (the H100's own rule still)" "$out/app$a" "$a" "$gpu" "$engine" "$card" VGPU_TC_EXACT=force
  done
done
# The H100's lines differ from the RTX 3060's on seven tensor-core lines and on no other line.
changed="$( { diff "$data/simd_lane_forms.rtx3060.txt" "$data/simd_lane_forms.h100.txt" || true; } | { grep '^>' || true; } | awk '{print $2 "." $3}' | tr '\n' ' ')"
want_changed="wmma_f16_f32acc.random wmma_f16_f16acc.random wmma_bf16_f32acc.random wmma_tf32_f32acc.random wmma_f16_f16acc.edges wmma_bf16_f32acc.edges wmma_tf32_f32acc.edges "
if [[ "$changed" != "$want_changed" ]]; then echo "FAIL: simd_lane_forms.h100.txt differs from the RTX 3060's on: $changed"; fails=1; fi
# RE-RECORDED 2026-11-16: the b200 profile (and gb200) now sums by the Hopper rule by default, inferred from the 20,000 vectors recorded on a
# B200 by others (docs/numerics/tcpapers.md; not a card run of ours): an sm_100 or sm_86 build prints the H100's lines by default and with
# VGPU_TC_EXACT=force, and the sequential pins with VGPU_TC_EXACT=0 (the way back to the earlier results). Before: the sequential pins by default.
for pair in 100:b200 86:b200 100:gb200; do
  a="${pair%%:*}"; gpu="${pair##*:}"
  [[ -x "$out/app$a" ]] || continue
  for engine in sass ptx; do
    run_case "hopper by default (inferred from a B200's recorded vectors)" "$out/app$a" "$a" "$gpu" "$engine" "$data/simd_lane_forms.h100.txt"
    run_case "VGPU_TC_EXACT=0" "$out/app$a" "$a" "$gpu" "$engine" "$data/simd_lane_forms.$engine.tcseq.txt" VGPU_TC_EXACT=0
    run_case "VGPU_TC_EXACT=force (its own rule still)" "$out/app$a" "$a" "$gpu" "$engine" "$data/simd_lane_forms.h100.txt" VGPU_TC_EXACT=force
  done
done
# Compute capability 12.0 (2026-11-17): measured on an RTX PRO 6000 Blackwell Server Edition (rtx-pro-6000-server), inferred for the other
# profiles of the compute capability. The H100's lines by default and with VGPU_TC_EXACT=force, the sequential pins with VGPU_TC_EXACT=0. Before
# 2026-11-17 these profiles printed the sequential pins by default (6 of the 60 lines and 13 of the 169 of float_half_forms were not the card's).
for pair in 120:rtx-pro-6000-server 86:rtx-pro-6000-server 120:rtx-pro-6000 120:rtx5090 120:rtx-pro-4500 120:rtx-pro-6000-max-q; do
  a="${pair%%:*}"; gpu="${pair##*:}"
  [[ -x "$out/app$a" ]] || continue
  for engine in sass ptx; do
    run_case "hopper by default (card: an RTX PRO 6000 Blackwell Server Edition)" "$out/app$a" "$a" "$gpu" "$engine" "$data/simd_lane_forms.h100.txt"
    run_case "VGPU_TC_EXACT=0" "$out/app$a" "$a" "$gpu" "$engine" "$data/simd_lane_forms.$engine.tcseq.txt" VGPU_TC_EXACT=0
    run_case "VGPU_TC_EXACT=force (its own rule still)" "$out/app$a" "$a" "$gpu" "$engine" "$data/simd_lane_forms.h100.txt" VGPU_TC_EXACT=force
  done
done

(( fails == 0 )) && echo PASS || { echo FAIL; exit 1; }
