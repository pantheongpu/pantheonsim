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
