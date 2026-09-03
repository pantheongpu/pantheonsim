# Shared guard for tests that compile an app with nvcc and run it against the
# shims in one build directory.
#
# The shim's soname major follows the toolkit it was built against, and nvcc
# stamps the app with the major *it* ships. Those can differ on one host -- a
# CUDA 12 build directory next to a CUDA 13 nvcc -- and then the app asks for a
# library this build does not provide, quietly falls through to the real
# libcudart, and fails against the real driver. That is an unbuildable
# combination rather than a defect, so it skips.
#
# Skipping only matters if it cannot hide a genuine failure, so the check is on
# what the linker actually recorded, not on a guess about versions.

# require_shim_libs <shim_dir> <binary>
# Returns 1 (and prints a SKIP line) when the binary needs a CUDA library this
# shim directory does not supply.
require_shim_libs() {
  local shim="$1" bin="$2" need
  command -v objdump >/dev/null 2>&1 || return 0
  for need in $(objdump -p "$bin" 2>/dev/null | awk '/NEEDED/ {print $2}'); do
    case "$need" in
      libcudart.so.*|libcublas.so.*|libcublasLt.so.*|libnccl.so.*|libcuda.so.*|libcudnn.so.*)
        if [[ ! -e "$shim/$need" ]]; then
          echo "SKIP: app needs $need, which $shim does not provide" \
               "(nvcc's toolkit major differs from this build's)"
          return 1
        fi
        ;;
    esac
  done
  return 0
}
