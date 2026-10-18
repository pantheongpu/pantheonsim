#!/usr/bin/env bash
# The counters rocprofv3 can collect on a Radeon card, and what the simulated card offers of them.
#
# The real RX 6800 and RX 6700 XT offer about a hundred (`rocprofv3-avail info --pmc`): cycle, stall, cache and memory
# counters that need a model of time, instruction counts, waves, and the agent's properties as constants. The simulator
# offers the ones it counts exactly, under the card's names, with the card's descriptions and expressions, and refuses the
# others the way rocprofv3 refuses a counter a GPU lacks. This script checks that every counter the simulator offers is
# on the card and printed the same (the expected file holds the card's own entries), that the simulator offers
# the whole set, and that SQ_WAVES counts what the geometry says.
#
#   run_rocprof_counters.sh [--gpu rx6800|rx6700xt]          the simulated card
#   run_rocprof_counters.sh --card [--update] [--gpu ...]    the real card (ROCm's libraries): check, or rewrite
#                                                            amd/tests/data/rocprof_counters[.rx6700xt].expected
#
# Not checked against the card: the values of a collected counter. `rocprofv3 --pmc` aborts on the machine the
# card files were recorded on (std::out_of_range in the tool, with the iGPU's agent in the list), so SQ_WAVES's value is
# what the launch's geometry says (vecadd: 4 groups of 256 work-items, 8 waves of 32 each, 32 waves).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
usage="usage: $0 [--gpu rx6800|rx6700xt] [--card [--update]]"
gpu=rx6800; card=0; update=0
while (( $# )); do
  case "$1" in
    --card) card=1 ;;
    --update) update=1 ;;
    --gpu) shift; gpu="${1:-}" ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
  shift
done
case "$gpu" in rx6800|rx6700xt) ;; *) echo "$usage" >&2; exit 2 ;; esac
isa=gfx1030; [[ "$gpu" == rx6700xt ]] && isa=gfx1031
suffix=""; [[ "$gpu" == rx6700xt ]] && suffix=".rx6700xt"
expected="$root/amd/tests/data/rocprof_counters${suffix}.expected"
if (( update )) && (( ! card )); then echo "--update needs --card: the expected files come from a real card" >&2; exit 2; fi
# What the simulated card offers: the checked set.
names="SQ_WAVES SQ_WAVES_sum Wavefronts CU_NUM SIMD_NUM SE_NUM MAX_WAVE_SIZE
  num_xcc array_count num_gws simd_id_base max_waves_per_simd gfx_target_version cpu_core_id_base location_id wave_front_size
  lds_size_in_kb simd_count simd_per_cu mem_banks_count cpu_cores_count io_links_count domain max_engine_clk_fcompute
  caches_count simd_arrays_per_engine gds_size_in_kb cu_per_simd_array max_slots_scratch_cu vendor_id device_id
  drm_render_minor num_sdma_engines num_sdma_xgmi_engines num_sdma_queues_per_engine hive_id num_cp_queues
  max_engine_clk_ccompute local_mem_size"
command -v python3 >/dev/null || { echo "SKIP: python3 needed"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/bin/rocprofv3" && -x "$c/bin/hipcc" && -x "$c/bin/rocprofv3-avail" ]] && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm with rocprofv3, rocprofv3-avail and hipcc found (set VGPU_ROCM_PATH)"; exit 0; }
rocprof_version=$("$rocm/bin/rocprofv3" --version 2>&1 | sed -n 's/^ *version *: *\([0-9][0-9.]*\).*/\1/p' | head -1)
if (( ! update )); then
  [[ -f "$expected" ]] || { echo "FAIL: no expected file $expected"; exit 1; }
  recorded=$(sed -n 's/^# recorded on:.* rocprofv3 \([0-9][0-9.]*\).*/\1/p' "$expected" | head -1)
  if [[ -n "$recorded" && "$recorded" != "$rocprof_version" && -z "${ROCPROF_ANY_VERSION:-}" ]]; then
    echo "SKIP: the expected file is from rocprofv3 $recorded, this ROCm's is ${rocprof_version:-unknown} (ROCPROF_ANY_VERSION=1 compares anyway)"; exit 0
  fi
fi
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_rocprof_counters.XXXXXX")"
trap 'rm -rf "$work"' EXIT
export ROCM_PATH="$rocm"
py() { python3 -I "$root/amd/tests/rocprof_counters.py" "$@"; }

if (( card )); then
  [[ -e /dev/kfd ]] || { echo "SKIP: no AMD GPU (/dev/kfd) for --card"; exit 0; }
  dev="${ROCPROF_CARD_DEVICE:-0}"; override=""
  if [[ "$gpu" == rx6700xt ]]; then dev="${ROCPROF_CARD_DEVICE:-1}"; override="10.3.0"; fi
  env ${override:+HSA_OVERRIDE_GFX_VERSION=$override} HIP_VISIBLE_DEVICES="$dev" \
    timeout -k 5 120 "$rocm/bin/rocprofv3-avail" info --pmc >"$work/info.txt" 2>"$work/info.err" \
    || { tail -5 "$work/info.err"; echo "FAIL: rocprofv3-avail failed"; exit 1; }
  { echo "[offered by the card]"; py "$work/info.txt" $isa offers; echo "[entries]"; py "$work/info.txt" $isa blocks $names; } >"$work/card.txt"
  if (( update )); then
    { echo "# the counters of the $gpu ($isa) from rocprofv3-avail info --pmc, reduced by amd/tests/rocprof_counters.py"
      echo "# recorded on: AMD Radeon $gpu ($isa), $(basename "$(readlink -f "$rocm")"), rocprofv3 ${rocprof_version:-?}"
      echo "# regenerate: amd/tests/e2e/run_rocprof_counters.sh --card --update${suffix:+ --gpu $gpu}"
      cat "$work/card.txt"; } >"$expected"
    echo "wrote $expected"; exit 0
  fi
  grep -v '^#' "$expected" >"$work/expected.txt"
  diff -u --label expected --label "the real $gpu" "$work/expected.txt" "$work/card.txt" && echo "the real $gpu's counters are the recorded ones" && exit 0
  echo "FAIL: the counters the real $gpu offers differ from the recorded ones"; exit 1
fi

vgpu="${VGPU_BUILD_DIR:-$root/build}/vgpu"
build="$(dirname "$vgpu")"
[[ -x "$vgpu" && -e "$build/shim/librocprofiler-sdk.so.1" ]] || { echo "SKIP: build vgpu and vgpurocprof first"; exit 0; }
if objdump -p "$build/shim/libamdhip64.so.7" 2>/dev/null | awk '/NEEDED/ {print $2}' | grep -qE 'lib(a|t)san'; then
  echo "SKIP: the shim is built with a sanitizer, and ROCm's rocprofv3 tool is not"; exit 0
fi
unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_SESSION
[[ "$gpu" == rx6700xt ]] && export HSA_OVERRIDE_GFX_VERSION=10.3.0
export VGPU_TELEMETRY_PATH="$work/telemetry"; mkdir -p "$VGPU_TELEMETRY_PATH"
if ! "$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1030 -w "$root/amd/tests/rocprof/vecadd.cpp" -o "$work/vecadd" >"$work/build.log" 2>&1; then
  cat "$work/build.log"; echo "FAIL: hipcc could not build vecadd"; exit 1
fi
sim() { timeout -k 5 300 "$vgpu" shell -y --no-isolate --gpu "amd/$gpu" --count 1 -c "$1" </dev/null 2>&1; }
sim "rocprofv3-avail info --pmc" >"$work/info.txt"
fail=0
# 1. Everything the simulator offers is the card's, printed as the card prints it.
sed -n '/^\[offered by the card\]/,/^\[entries\]/p' "$expected" | grep -v '^\[' >"$work/card_offers.txt"
py "$work/info.txt" $isa offers >"$work/sim_offers.txt"
extra=$(LC_ALL=C comm -13 "$work/card_offers.txt" "$work/sim_offers.txt")
[[ -z "$extra" ]] || { echo "FAIL: the simulator offers counters the card does not: $extra"; fail=1; }
# 2. The checked set is what it offers, and the entries are the card's.
want=$(echo $names | tr ' ' '\n' | LC_ALL=C sort)
[[ "$want" == "$(cat "$work/sim_offers.txt")" ]] || { echo "FAIL: the simulator offers a different set than the checked one"; diff <(echo "$want") "$work/sim_offers.txt"; fail=1; }
{ echo "[entries]"; py "$work/info.txt" $isa blocks $names; } >"$work/sim_entries.txt"
sed -n '/^\[entries\]/,$p' "$expected" >"$work/card_entries.txt"
diff -u --label "expected (card)" --label "the simulated $gpu" "$work/card_entries.txt" "$work/sim_entries.txt" || fail=1
# 3. A counter that needs a model of time or of the caches is refused, as on a GPU that lacks it.
out=$(sim "rocprofv3 --pmc SQ_INSTS_VALU --kernel-trace -d '$work/refused' -- '$work/vecadd'")
grep -q 'Missing: \[SQ_INSTS_VALU\]' <<<"$out" || { echo "FAIL: SQ_INSTS_VALU should be refused, not collected or invented"; fail=1; }
# 4. SQ_WAVES, per dispatch, is the geometry's: 1024 work-items in groups of 256, wave32.
out=$(sim "rocprofv3 --pmc SQ_WAVES --kernel-trace --output-format csv -d '$work/pmc' -- '$work/vecadd'")
csv=$(find "$work/pmc" -name '*_counter_collection.csv' | head -1)
got=$([[ -n "$csv" ]] && python3 -I -c '
import csv, sys
print(" ".join(str(int(float(r["Counter_Value"]))) for r in csv.DictReader(open(sys.argv[1])) if r["Counter_Name"] == "SQ_WAVES"))' "$csv")
[[ "$got" == "32" ]] || { echo "FAIL: SQ_WAVES of vecadd should be 32, got '${got}'"; echo "$out" | tail -5; fail=1; }
(( fail )) && exit 1
echo "the simulated $gpu offers the card's counters it counts exactly, as the card prints them"
