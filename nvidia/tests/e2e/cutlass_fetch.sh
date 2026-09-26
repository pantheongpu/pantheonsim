# Sourced by the e2e tests that check Hopper features against NVIDIA's own
# code. Sets $cutlass to a CUTLASS checkout at a pinned release -- $CUTLASS_DIR
# if set, otherwise fetched once into the build directory -- and, with
# `need_gtest=1`, $gtest to a built googletest. Prints SKIP and exits 0 when
# either cannot be had (no network, say), since that is an unrunnable
# combination rather than a defect.
cutlass_tag="v4.8.0"
gtest_tag="v1.15.2"
cutlass="${CUTLASS_DIR:-$build/_deps/cutlass-$cutlass_tag}"
fetch_into() {  # fetch_into <dir> <url> <tar members...>
  local dir="$1" url="$2" tmp="$1.tmp.$$"
  shift 2
  mkdir -p "$tmp"
  if ! curl -fsSL "$url" | tar -xz -C "$tmp" --strip-components=1 --no-wildcards-match-slash \
       --wildcards "$@" 2>/dev/null; then
    rm -rf "$tmp"
    return 1
  fi
  rm -rf "$dir"
  mv "$tmp" "$dir"
}
# The tests that source this run in parallel under ctest -j, and a fetch ends by
# replacing the directory: without a lock, the second one to finish deleted the
# checkout the first was already compiling against ("cute/tensor.hpp: No such
# file"). So one fetches at a time and the rest find its result; the lock is let
# go as soon as the fetching is done, not held for the test.
mkdir -p "$build/_deps"
exec {fetch_lock}>"$build/_deps/.fetch.lock"
flock "$fetch_lock"
# The GEMM tests run_cutlass_gemm.sh compiles are the newest things fetched, so
# a checkout without them is from before and is fetched again.
if [[ ! -f "$cutlass/test/unit/conv/device_3x/testbed_conv.hpp" ||
      ! -f "$cutlass/test/unit/gemm/device/sm90_gemm_s8_s8_s8_tensor_op_s32.cu" ]]; then
  if [[ -n "${CUTLASS_DIR:-}" ]]; then
    echo "FAIL: CUTLASS_DIR=$CUTLASS_DIR is not a CUTLASS checkout with its unit tests"; exit 1
  fi
  # The headers, the Hopper unit tests with what they include (the conv
  # tests' testbed includes the GEMM one), and the GEMM tests
  # run_cutlass_gemm.sh runs: the rest of the release is examples, Python
  # and tools.
  if ! fetch_into "$cutlass" "https://github.com/NVIDIA/cutlass/archive/refs/tags/$cutlass_tag.tar.gz" \
       'cutlass-*/include/*' 'cutlass-*/LICENSE.txt' 'cutlass-*/tools/util/include/*' \
       'cutlass-*/test/unit/common/*' 'cutlass-*/test/unit/cute/hopper/*' \
       'cutlass-*/test/unit/conv/cache_testbed_output.h' 'cutlass-*/test/unit/conv/device_3x/*.hpp' \
       'cutlass-*/test/unit/conv/device_3x/fprop/sm90_*' \
       'cutlass-*/test/unit/gemm/device/*.h' 'cutlass-*/test/unit/gemm/device/*.hpp' \
       'cutlass-*/test/unit/gemm/device/gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80.cu' \
       'cutlass-*/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_group_gemm.cu' \
       'cutlass-*/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_pingpong.cu' \
       'cutlass-*/test/unit/gemm/device/sm90_gemm_s8_s8_s8_tensor_op_s32.cu' \
       'cutlass-*/test/unit/test_unit.cpp'; then
    echo "SKIP: could not download CUTLASS $cutlass_tag (set CUTLASS_DIR to a checkout)"; exit 0
  fi
fi
if [[ "${need_gtest:-0}" == 1 ]]; then
  gtest="$build/_deps/googletest-$gtest_tag"
  if [[ ! -f "$gtest/libgtest.a" ]]; then
    if ! fetch_into "$gtest" "https://github.com/google/googletest/archive/refs/tags/$gtest_tag.tar.gz" \
         'googletest-*/googletest/*'; then
      echo "SKIP: could not download googletest $gtest_tag"; exit 0
    fi
    ( cd "$gtest/googletest" && ${CXX:-g++} -std=c++17 -O1 -Iinclude -I. -c src/gtest-all.cc -o gtest-all.o &&
      ar rcs ../libgtest.a gtest-all.o ) || { echo "FAIL: googletest did not build"; exit 1; }
  fi
fi
flock -u "$fetch_lock"
exec {fetch_lock}>&-

# How many CUTLASS compiles to run at once, at most $1: as many as the free
# memory holds at ${VGPU_NVCC_COMPILE_MB:-10240} MB each (a compile peaks near
# 10 GB), keeping 2 GB back, and never fewer than one. VGPU_NVCC_JOBS sets it
# outright. "Free" is the smaller of the machine's MemAvailable and what this
# process's memory cgroup still allows: inside a container /proc/meminfo
# reports the host, and sizing to that ran four 10 GB compiles in a 10 GB
# container on a shared machine, which ended in the out-of-memory killer.
cutlass_compile_jobs() {
  local max="$1" jobs
  if [[ -n "${VGPU_NVCC_JOBS:-}" ]]; then
    jobs="$VGPU_NVCC_JOBS"
  else
    local per_kb=$(( ${VGPU_NVCC_COMPILE_MB:-10240} * 1024 ))
    local free_kb limit cur cg
    free_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    cg="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2>/dev/null)"
    if [[ -n "$cg" && -r "/sys/fs/cgroup$cg/memory.max" ]]; then           # cgroup v2
      limit="$(cat "/sys/fs/cgroup$cg/memory.max")"
      cur="$(cat "/sys/fs/cgroup$cg/memory.current" 2>/dev/null || echo 0)"
    elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then          # cgroup v1
      limit="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
      cur="$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)"
    fi
    if [[ "${limit:-max}" =~ ^[0-9]+$ ]] && (( limit / 1024 - cur / 1024 < free_kb )); then
      free_kb=$(( limit / 1024 - cur / 1024 ))
    fi
    jobs=$(( (free_kb - 2 * 1024 * 1024) / per_kb ))
  fi
  (( jobs < 1 )) && jobs=1
  (( jobs > max )) && jobs=$max
  echo "$jobs"
}
