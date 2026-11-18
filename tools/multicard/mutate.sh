#!/usr/bin/env bash
# Mutation evidence for the multi-card change (docs/perf-design/12-host-gpu-turbo.md section 17): applies each mutation to a COPY of the sources, rebuilds, runs the unit
# test and the stand-in e2e script, and prints which of them fail. Usage: mutate.sh <source dir> <build dir>   (the build dir configured on the source dir; nvcc 13 in VGPU_NVCC)
set -u
src=$1 build=$2
cd "$src"
cp src/runtime/host_gpu.cpp /tmp/hg.orig; cp include/vgpu/host_gpu_cards.hpp /tmp/cards.orig
run() {
  local name=$1
  ( cd "$build" && ninja test_host_gpu_cards vgpucudart > /tmp/mut_build.log 2>&1 ) || { echo "$name: BUILD FAILED"; tail -5 /tmp/mut_build.log; return; }
  local u e
  u="$("$build/test_host_gpu_cards" 2>&1 | grep -c 'FAIL')"
  e="$(VGPU_BUILD_DIR=$build bash nvidia/tests/e2e/run_host_gpu_multi_stub.sh 2>&1 | grep -c '^FAIL')"
  echo "$name: unit failures=$u e2e FAIL lines=$e"
  VGPU_BUILD_DIR=$build bash nvidia/tests/e2e/run_host_gpu_multi_stub.sh 2>&1 | grep '^FAIL' | head -3 | cut -c1-220
  "$build/test_host_gpu_cards" 2>&1 | grep 'FAIL' | head -3
}
restore() { cp /tmp/hg.orig src/runtime/host_gpu.cpp; cp /tmp/cards.orig include/vgpu/host_gpu_cards.hpp; }
patch() {   # <file> <python expression old> <new>
  python3 - "$1" "$2" "$3" <<'PY'
import sys
f,a,b=sys.argv[1:4]
s=open(f).read()
assert s.count(a)==1,(a,s.count(a))
open(f,'w').write(s.replace(a,b))
PY
}
echo "== M0 unmutated"; run M0
echo "== M1 every device slot uses window 0 (the window of slot 0)"
patch src/runtime/host_gpu.cpp 'mode == Mode::Turbo ? kDeviceVaBase + static_cast<uint64_t>(slot) * kDeviceVaStride : kDeviceVaBase' 'kDeviceVaBase'; run M1; restore
echo "== M2 every device uses the driver's device 0"
patch src/runtime/host_gpu.cpp 'dev = pick.index;' 'dev = 0; (void)pick;'; run M2; restore
echo "== M3 a peer pointer argument is not looked at"
patch src/runtime/host_gpu.cpp 'if (multi && addr_in_other_window(v, slot, kDeviceVaBase, kDeviceVaStride, kDeviceVaWindows))' 'if (false)'; run M3; restore
echo "== M4 one runtime for every simulated device (device 0's)"
patch src/runtime/host_gpu.cpp 'if (auto it = made.find(slot); it != made.end()) return it->second;' 'if (auto it = made.begin(); it != made.end()) return it->second;'; run M4; restore
echo "== M5 CUDA_VISIBLE_DEVICES is not read (the slot is taken for the driver's number)"
patch include/vgpu/host_gpu_cards.hpp 'const std::vector<int> cards = driver_cards(cvd, real_count, &decided);' 'const std::vector<int> cards = driver_cards(nullptr, real_count, &decided);'; run M5; restore
echo "== M6 exact mode serves every device like turbo"
patch include/vgpu/host_gpu_cards.hpp 'if (!turbo) return ordinal == 0 ? 0 : -1;' 'if (!turbo) return ordinal;'; run M6; restore
echo "== M7 a device without a card is served by the first card instead of staying simulated"
patch include/vgpu/host_gpu_cards.hpp '  p.why = (cvd && *cvd) ? CardWhy::Hidden : CardWhy::NoCard;
  return p;' '  p.index = 0; p.why = CardWhy::Ok;
  return p;'; run M7; restore
echo "== M8 every launch is waited for (the queue is off)"
patch src/runtime/host_gpu.cpp 'g_runtimes.load(std::memory_order_relaxed) > 1;' 'false;'; run M8; restore
echo "== M9 launches are queued with one device too"
patch src/runtime/host_gpu.cpp 'g_runtimes.load(std::memory_order_relaxed) > 1;' 'true;'; run M9; restore
echo "== done"; run RESTORED
