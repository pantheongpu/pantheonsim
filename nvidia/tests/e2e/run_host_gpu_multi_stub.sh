#!/usr/bin/env bash
# Turbo mode on a machine with several cards (docs/perf-design/12-host-gpu-turbo.md section 17; include/vgpu/host_gpu_cards.hpp, src/runtime/host_gpu.cpp), tested with the
# stand-in driver (host_gpu_stub_driver.c, told to show several cards) so that it runs on every machine, card or not. The stand-in card keeps the memory it is
# sent and "computes" host_gpu_multi.cu's `bump` (STUB_CARD_BUMP=1), so the program prints the same lines whether a launch ran on a card or in the simulator, and
# any memory mixed up between devices, cards or the simulator changes them. What it pins (every output equals the mode-off output of the same program):
#   1. four simulated devices on four cards: each device's launches run on its own card (the stand-in's context of that card), in its own window
#      (cuMemAddressReserve at kDeviceVaBase + slot * 1 TiB on the card of that slot), and each card gets its device's memory;
#   2. four devices on two cards: devices 2 and 3 stay in the simulator, said once each, and nothing reaches a card for them;
#   3. one card: only device 0 uses it, as before multi-card;
#   4. CUDA_VISIBLE_DEVICES=2,3: the program sees two devices, which are the machine's cards 2 and 3 on the real driver's devices 0 and 1;
#   5. VGPU_HOST_GPU_DEVICES=3,1 picks the real devices; a list that repeats a card turns the mode off with a message and nothing reaches the driver;
#   6. cudaMemcpyPeer between devices on two cards: card 0's result reaches the copy (the simulator pulls it first), and card 1 is sent the copy;
#   7. a launch that passes another device's pointer (no peer access) never reaches a card;
#   8. peer access between devices 0 and 1 keeps their launches in the simulator (devices 2 and 3 still use their cards);
#   9. a copy from a device with no card to one with a card (4 devices, 3 cards);
#  10. exact mode serves device 0 only, on the driver's device 0, whatever the machine holds;
#  11. the summary: one host_gpu_card line for each device that used a card;
#  12. launches on the cards of several devices are queued, not waited for (so the cards run at the same time); a single device's launches are each waited for.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
profile="${VGPU_HOST_GPU_TEST_PROFILE:-nvidia/a100}"
yaml="$root/nvidia/profiles/${profile#nvidia/}.yaml"
[[ -f "$yaml" ]] || { echo "SKIP: no profile $profile"; exit 0; }
cc="$(sed -n 's/^compute_capability: "\([0-9]*\)\.\([0-9]*\)".*/\1\2/p' "$yaml" | head -1)"
mp="$(sed -n 's/^  multiprocessors: \([0-9]*\).*/\1/p' "$yaml" | head -1)"
[[ -n "$cc" && -n "$mp" ]] || { echo "FAIL: cannot read the compute capability and SM count of $yaml"; exit 1; }
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
cc_bin="$(command -v cc || command -v gcc)"
[[ -n "$cc_bin" ]] || { echo "SKIP: no C compiler"; exit 0; }
[[ -z "$(shim_sanitizer "$shim")" ]] || { echo "SKIP: the sanitizer runtimes refuse a RTLD_DEEPBIND dlopen"; exit 0; }
nvcc_host_compiler_fix
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-host-gpu-multi.XXXXXX")"
trap 'rm -rf "$work"' EXIT
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$root/nvidia/tests/e2e/host_gpu_multi.cu" -o "$work/prog" || { echo "FAIL: host_gpu_multi.cu does not compile"; exit 1; }
"$cc_bin" -shared -fPIC -O1 -o "$work/libstubcuda.so" "$root/nvidia/tests/e2e/host_gpu_stub_driver.c" || { echo "FAIL: the stand-in driver does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$work/prog"; then exit 0; fi
fails=0
# The simulated machine: $1 devices. The program's own CUDA_VISIBLE_DEVICES, when a step sets one, is passed to the mode-off run and the mode-on run alike.
off() {   # <tag> <program argument> <devices> [env...]
  local tag=$1 arg=$2 devs=$3
  shift 3
  env VGPU_GPU="$profile" VGPU_DEVICE_COUNT="$devs" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$@" "$work/prog" "$arg" > "$work/off_$tag.out" 2> "$work/off_$tag.err" || { echo "FAIL: the $tag run with the mode off failed"; tail -5 "$work/off_$tag.err"; exit 1; }
}
run() {   # <tag> <program argument> <devices> <stand-in cards> [env...]: the turbo run; stdout $work/<tag>.out, stderr .err, driver calls .log
  local tag=$1 arg=$2 devs=$3 cards=$4
  shift 4
  : > "$work/$tag.log"
  env VGPU_GPU="$profile" VGPU_DEVICE_COUNT="$devs" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" STUB_LOG="$work/$tag.log" STUB_SM="$cc" STUB_MP="$mp" STUB_DEVICES="$cards" \
      STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 STUB_CARD_BUMP=1 VGPU_HOST_GPU_LIBCUDA="$work/libstubcuda.so" VGPU_HOST_GPU=turbo "$@" "$work/prog" "$arg" > "$work/$tag.out" 2> "$work/$tag.err" || { echo "FAIL: the $tag run failed"; tail -5 "$work/$tag.err"; fails=1; }
}
same() { diff -q "$work/off_$2.out" "$work/$1.out" > /dev/null || { echo "FAIL: $1: the output is not the mode-off output:"; diff "$work/off_$2.out" "$work/$1.out" | head -6; fails=1; }; }
has() { grep -qx -- "$2" "$work/$1.log" || { echo "FAIL: $1: the driver was never told '$2': $(sort "$work/$1.log" | uniq -c | tr -s ' ' | tr '\n' ';')"; fails=1; }; }
lacks() { ! grep -q -- "$2" "$work/$1.log" || { echo "FAIL: $1: the driver was told '$2' ($3)"; fails=1; }; }
base=0x200000000000
win() { printf '0x%x' $(( base + $1 * 0x10000000000 )); }   # slot N's window
off dp dp 4
off copy copy 4
off peerptr peerptr 4
off peeron peeron 4
off mixed mixed 4
# 1. four devices, four cards.
run dp4 dp 4 4
same dp4 dp
for d in 0 1 2 3; do
  has dp4 "ctx_retain $d"; has dp4 "launch on $d"; has dp4 "reserve $(win $d) on $d"; has dp4 "mem_create on $d for $d"; has dp4 "module_load on $d"
done
lacks dp4 "reserve $(win 1) on 0" "device 1's window was reserved on card 0"
# 11. the summary (the first device's lines are the ones scripts read; with several devices each one that started a card adds a host_gpu_card line).
n="$(grep -c '^host_gpu_card: ' "$work/dp4.err")"
(( n == 4 )) || { echo "FAIL: expected 4 host_gpu_card lines (one per device), got $n: $(grep -m4 '^host_gpu' "$work/dp4.err")"; fails=1; }
grep -q '^host_gpu_card: device_slot=0 card=0 launches_card=3 ' "$work/dp4.err" || { echo "FAIL: device 0's summary line is wrong: $(grep '^host_gpu_card: device_slot=0' "$work/dp4.err")"; fails=1; }
grep -q '^host_gpu_card: device_slot=2 card=2 launches_card=3 ' "$work/dp4.err" || { echo "FAIL: device 2's summary line is wrong: $(grep '^host_gpu_card: device_slot=2' "$work/dp4.err")"; fails=1; }
# 2. four devices, two cards.
run dp2 dp 4 2
same dp2 dp
has dp2 "launch on 0"; has dp2 "launch on 1"
lacks dp2 "launch on 2" "a device with no card"; lacks dp2 "ctx_retain 2" "a device with no card"; lacks dp2 "ctx_retain 3" "a device with no card"
for d in 2 3; do grep -q "simulated device $d has no host card" "$work/dp2.err" || { echo "FAIL: dp2: no message for device $d, which has no card: $(grep host_gpu "$work/dp2.err" | head -3)"; fails=1; }; done
[[ "$(grep -c 'has no host card' "$work/dp2.err")" == 2 ]] || { echo "FAIL: dp2: the no-card message is not said once per device"; fails=1; }
# 3. one card.
run dp1 dp 4 1
same dp1 dp
has dp1 "launch on 0"; lacks dp1 "launch on 1" "one card serves device 0 only"; lacks dp1 "ctx_retain 1" "one card serves device 0 only"
# 4. CUDA_VISIBLE_DEVICES=2,3 on a machine of four cards (the simulated machine too has four, the program sees two).
off cvd dp 4 CUDA_VISIBLE_DEVICES=2,3
run cvd dp 4 4 CUDA_VISIBLE_DEVICES=2,3
same cvd cvd
grep -qx 'devices 2' "$work/cvd.out" || { echo "FAIL: CUDA_VISIBLE_DEVICES=2,3 did not show two devices"; fails=1; }
has cvd "ctx_retain 2"; has cvd "ctx_retain 3"; lacks cvd "ctx_retain 0" "a card CUDA_VISIBLE_DEVICES hides"; lacks cvd "ctx_retain 1" "a card CUDA_VISIBLE_DEVICES hides"
has cvd "reserve $(win 2) on 2"; has cvd "reserve $(win 3) on 3"; has cvd "launch on 2"; has cvd "launch on 3"
# 5. VGPU_HOST_GPU_DEVICES.
run list dp 4 4 VGPU_HOST_GPU_DEVICES=3,1
same list dp
has list "reserve $(win 0) on 3"; has list "reserve $(win 1) on 1"; has list "launch on 3"; has list "launch on 1"; lacks list "launch on 0" "a card the list does not name"; lacks list "ctx_retain 2" "a card the list does not name"
run listbad dp 4 4 VGPU_HOST_GPU_DEVICES=1,1
same listbad dp
grep -q 'VGPU_HOST_GPU_DEVICES=1,1 is not a list of different device numbers' "$work/listbad.err" || { echo "FAIL: a repeated card in VGPU_HOST_GPU_DEVICES was not refused"; fails=1; }
[[ ! -s "$work/listbad.log" ]] || { echo "FAIL: a refused VGPU_HOST_GPU_DEVICES reached the driver"; fails=1; }
# 6. cudaMemcpyPeer between devices on two cards.
run copy copy 4 2
same copy copy
has copy "launch on 0"; has copy "launch on 1"
# 7. a peer pointer without peer access.
run peerptr peerptr 4 4
same peerptr peerptr
lacks peerptr "launch on" "a launch whose argument is another device's pointer"
# 8. peer access between devices 0 and 1.
run peeron peeron 4 4
same peeron peeron
lacks peeron "launch on 0" "device 0 has a peer link"; lacks peeron "launch on 1" "device 1 has a peer link"; has peeron "launch on 2"; has peeron "launch on 3"
# 9. a copy from the device with no card (device 3 of 4 with three cards) to device 0.
run mixed mixed 4 3
same mixed mixed
has mixed "launch on 0"; has mixed "launch on 2"; lacks mixed "launch on 3" "a device with no card"
# 10. exact mode.
run ex dp 4 4 VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0
same ex dp
lacks ex "ctx_retain 1" "exact mode serves device 0 only"; lacks ex "launch on 1" "exact mode serves device 0 only"; lacks ex "reserve $(win 1)" "exact mode serves device 0 only"
# 12. several devices queue their launches on the cards and do not wait (the cards run at the same time); one device waits for each launch, as before. The stand-in logs
# every cuCtxSynchronize: with four devices none comes before the last launch (the read-backs, at the end, are where the queue is waited for), with one device each launch is
# followed by its own.
first_sync="$(grep -n '^sync on' "$work/dp4.log" | head -1 | cut -d: -f1)"
last_launch="$(grep -n '^launch on' "$work/dp4.log" | tail -1 | cut -d: -f1)"
[[ -n "$first_sync$last_launch" && "${first_sync:-0}" -gt "${last_launch:-0}" ]] || { echo "FAIL: dp4: a launch was waited for although four devices queue theirs (first sync at line ${first_sync:-none}, last launch at line ${last_launch:-none})"; fails=1; }
off solo solo 1
run solo solo 1 1
same solo solo
seq="$(grep -E '^(launch|sync) on' "$work/solo.log" | tr '\n' ',')"
[[ "$seq" == "launch on 0,sync on 0,launch on 0,sync on 0,launch on 0,sync on 0," ]] || { echo "FAIL: a single device's launches were not each waited for: $seq"; fails=1; }
if (( fails )); then echo "FAIL: the multi-card stand-in tests"; exit 1; fi
echo "PASS: turbo mode on several cards (stand-in driver)"
