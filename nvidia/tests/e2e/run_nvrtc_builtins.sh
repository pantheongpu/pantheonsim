#!/usr/bin/env bash
# The shim compiles through the toolkit's real libnvrtc when it can find one (VGPU_NVRTC_LIB, nvcc's toolkit,
# /usr/local/cuda, ...). NVIDIA's pip wheels (nvidia-cuda-nvrtc, a dependency of PyTorch) keep libnvrtc.so.13 and
# libnvrtc-builtins.so.13.x in one directory that is on no search path; the real library dlopens the builtins by
# name at its first compile, so with the shim loading it RTLD_LOCAL every compile (every jiterator kernel in
# PyTorch: complex abs, det, slogdet, special functions...) failed with "failed to open libnvrtc-builtins".
# This compiles a kernel through the shim with only the wheel's libnvrtc named, and needs no nvcc and no GPU.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
[[ -e "$shim/libnvrtc.so.13" ]] || { echo "SKIP: no NVRTC shim in $shim"; exit 0; }
lib="${VGPU_NVRTC_LIB:-}"
if [[ -z "$lib" ]]; then
  for c in "$HOME"/.local/share/*/lib/python3*/site-packages/nvidia/cu13/lib/libnvrtc.so.13 \
           /usr/lib/python3*/site-packages/nvidia/cu13/lib/libnvrtc.so.13; do
    [[ -e "$c" ]] && { lib="$c"; break; }
  done
fi
# Where the globs find nothing, ask the Pythons that have the wheel installed (the one run_pytorch.sh uses, then any
# python3): the package's own location is the one thing that is right whatever the virtual environment is called.
if [[ -z "$lib" || ! -e "$lib" ]]; then
  for py in "${VGPU_TORCH_CUDA_PYTHON:-}" "$HOME"/.local/share/torch-cu13*/bin/python python3; do
    [[ -n "$py" ]] && command -v "$py" >/dev/null 2>&1 || continue
    dir=$("$py" -I -c 'import importlib.util as u; s = u.find_spec("nvidia.cu13"); print(list(s.submodule_search_locations)[0] if s and s.submodule_search_locations else "")' 2>/dev/null)
    [[ -n "$dir" && -e "$dir/lib/libnvrtc.so.13" ]] && { lib="$dir/lib/libnvrtc.so.13"; break; }
  done
fi
[[ -n "$lib" && -e "$lib" ]] || { echo "SKIP: no pip-wheel libnvrtc.so.13 (set VGPU_NVRTC_LIB)"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: python3 not found"; exit 0; }
out=$(env -u LD_LIBRARY_PATH VGPU_NVRTC_LIB="$lib" python3 -I - "$shim/libnvrtc.so.13" <<'PY'
import ctypes, sys
n = ctypes.CDLL(sys.argv[1])
prog = ctypes.c_void_p()
src = b'extern "C" __global__ void k(float* p) { p[threadIdx.x] = 2.0f * p[threadIdx.x]; }'
assert n.nvrtcCreateProgram(ctypes.byref(prog), src, b"k.cu", 0, None, None) == 0
opts = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_80")
rc = n.nvrtcCompileProgram(prog, 1, opts)
size = ctypes.c_size_t()
n.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
log = ctypes.create_string_buffer(size.value + 1)
n.nvrtcGetProgramLog(prog, log)
n.nvrtcGetPTXSize(prog, ctypes.byref(size))
print("rc", rc, "ptx", size.value, "log", log.value.decode(errors="replace").strip()[:300])
sys.exit(0 if rc == 0 and size.value > 100 else 1)
PY
)
rc=$?
echo "$out"
if (( rc == 0 )); then echo "PASS nvrtc builtins ($lib)"; else echo "FAIL nvrtc compile through $lib"; exit 1; fi
