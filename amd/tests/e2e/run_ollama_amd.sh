#!/usr/bin/env bash
# Ollama, unmodified, running a model on every simulated AMD GPU through its
# ROCm backend: llama.cpp's HIP kernels, with rocBLAS and hipBLAS, the ones
# Ollama ships. It finds the GPU by the name and target the simulator gives,
# puts every layer of the model on it, and generates the same text its CPU
# backend does.
#
# Needs `ollama` with its ROCm backend beside it (lib/ollama/rocm, from
# ollama-linux-amd64-rocm.tar.zst) and the model (default smollm2:135m,
# VGPU_OLLAMA_MODEL); VGPU_OLLAMA_PULL=1 pulls it. Without either, it skips.
# VGPU_OLLAMA_PROFILES narrows the profiles (space-separated).
#
# Ollama puts its own libamdhip64 and libhsa-runtime64 first for the runner it
# starts, so the simulator's are preloaded: a library already loaded under a
# soname satisfies every later need for it. The host's NVIDIA GPUs, if any, are
# hidden, so the simulated GPU is the only one Ollama finds.
#
# Two settings Ollama's own package needs here, not the simulator:
#   - ROCBLAS_USE_HIPBLASLT=0. On gfx12 rocBLAS hands its GEMMs to hipBLASLt,
#     and Ollama ships hipBLASLt without its kernel library, so the runner
#     fails ("rocblaslt error: Cannot read TensileLibrary_lazy_gfx1201.dat");
#     this keeps them in rocBLAS, whose kernels Ollama does ship.
#   - OLLAMA_LOAD_TIMEOUT: a simulated GPU takes minutes to warm a model up
#     (an RX 9070 XT some seven), past Ollama's default of five.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
model="${VGPU_OLLAMA_MODEL:-smollm2:135m}"
command -v ollama >/dev/null || { echo "SKIP: ollama is not installed"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: python3 is needed to read ollama's replies"; exit 0; }
rocm="$(dirname "$(readlink -f "$(command -v ollama)")")/../lib/ollama/rocm"
[[ -f "$rocm/libggml-hip.so" ]] || { echo "SKIP: no ROCm backend beside ollama (ollama-linux-amd64-rocm.tar.zst)"; exit 0; }
[[ -e "$shim/libamdhip64.so.7" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
# A copy of the shim with its links resolved: they point into the build tree.
tmp=$(mktemp -d)
cp -L "$shim"/libamdhip64.so.7 "$shim"/libhsa-runtime64.so.1 "$tmp"/
mkdir -p "$tmp/ask"
serve_pid=""
cleanup() { [[ -n "$serve_pid" ]] && kill "$serve_pid" 2>/dev/null; wait 2>/dev/null; rm -rf "$tmp"; }
trap cleanup EXIT
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

# The preload hides a gap: a function the simulator's library lacks is found
# in Ollama's own libamdhip64, loaded beside it, and the test passes. The
# pantheonsim.com playground puts the simulator's library in place of Ollama's,
# and there a gap stops the ROCm backend loading at all -- llama.cpp's backend
# is loaded with every symbol bound up front, and hipMemAdvise was missing. So
# every HIP and HSA function Ollama's ROCm libraries ask for, at its version,
# has to be the simulator's own.
if command -v nm >/dev/null; then
  have=$(nm -D --defined-only "$tmp/libamdhip64.so.7" | awk '{print $3}' | sed 's/@@/@/' | sort -u)
  need=$(for f in "$rocm"/*.so*; do
           case "$(basename "$f")" in libamdhip64.so*|libhsa-runtime64.so*) continue ;; esac
           [[ -L "$f" ]] || nm -D --undefined-only "$f" 2>/dev/null
         done | awk '{print $NF}' | grep -E '^_*(hip|hsa)[A-Za-z0-9_]*@(hip_|ROCR_)' | sort -u)
  expect "the simulator's HIP runtime has every function Ollama's ROCm libraries use" "" \
    "$(comm -23 <(echo "$need") <(echo "$have") | tr '\n' ' ' | sed 's/ $//')"
fi

port=$((20000 + RANDOM % 20000))
api="http://127.0.0.1:$port"
# serve <log> [env...]: an ollama server with the given environment, ready.
serve() {
  local log=$1; shift
  env OLLAMA_HOST="127.0.0.1:$port" OLLAMA_DEBUG=1 CUDA_VISIBLE_DEVICES=-1 ROCBLAS_USE_HIPBLASLT=0 \
      OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-60m}" "$@" ollama serve > "$log" 2>&1 &
  serve_pid=$!
  for _ in $(seq 150); do curl -s -m 2 "$api/api/version" >/dev/null && return 0; sleep 0.2; done
  echo "FAIL  ollama serve did not come up"; tail -20 "$log"; exit 1
}
stop() { kill "$serve_pid" 2>/dev/null; wait "$serve_pid" 2>/dev/null; serve_pid=""; }
# generate [num_gpu]: the reply to a fixed prompt, greedy, three tokens.
generate() {
  local opts='"num_predict":3,"temperature":0,"seed":1'
  [[ -n "${1:-}" ]] && opts="$opts,\"num_gpu\":$1"
  curl -s -m 5400 "$api/api/generate" \
    -d "{\"model\":\"$model\",\"prompt\":\"The capital of France is\",\"stream\":false,\"options\":{$opts}}" |
    python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("response") if "response" in d else "ERROR: " + d.get("error", "?"))'
}
# resident: "<size> <size_vram>" of the loaded model.
resident() {
  curl -s -m 10 "$api/api/ps" | python3 -c '
import json, sys
m = json.load(sys.stdin).get("models", [])
print("%d %d" % (m[0]["size"], m[0]["size_vram"]) if m else "none")'
}

# The model, and the CPU backend's answer, which every GPU must reproduce.
serve "$tmp/cpu.log"
if ! OLLAMA_HOST="127.0.0.1:$port" ollama list 2>/dev/null | grep -q "^$model "; then
  if [[ "${VGPU_OLLAMA_PULL:-0}" == 1 ]]; then
    OLLAMA_HOST="127.0.0.1:$port" ollama pull "$model" >/dev/null || { echo "FAIL  could not pull $model"; exit 1; }
  else
    echo "SKIP: $model is not in $OLLAMA_MODELS (VGPU_OLLAMA_PULL=1 pulls it)"; exit 0
  fi
fi
reference=$(generate 0)
stop
[[ -n "$reference" && "$reference" != ERROR* ]] || { echo "FAIL  no reference reply from the CPU backend: $reference"; exit 1; }
echo "      reference (CPU backend): $(printf %q "$reference")"

preload="$tmp/libamdhip64.so.7:$tmp/libhsa-runtime64.so.1"
profiles="${VGPU_OLLAMA_PROFILES:-$("$build/vgpu" list-gpus 2>/dev/null | grep -o 'amd/[a-z0-9-]*' | sort -u | tr '\n' ' ')}"
for gpu in $profiles; do
  # The name and target the simulator gives the GPU, asked of a machine of
  # its own (not whatever else this host is simulating).
  name=$(VGPU_GPU=$gpu VGPU_TELEMETRY_PATH="$tmp/ask" "$build/vgpu" smi --query-gpu=name --format=csv,noheader 2>/dev/null)
  target=$(VGPU_GPU=$gpu VGPU_TELEMETRY_PATH="$tmp/ask" "$build/vgpu" smi --agents -t GPU 2>/dev/null | head -1)
  log="$tmp/${gpu//\//-}.log"
  # Not quiet: a kernel the simulator refuses says so in the log.
  serve "$log" LD_PRELOAD="$preload" VGPU_GPU="$gpu" \
        VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state"
  # Ollama quotes the description only where it has a space ("AMD Instinct
  # MI300X", but MI325X).
  found=$(grep -o 'msg="inference compute".*library=ROCm compute=[a-z0-9]* .*' "$log" | head -1 |
          sed -E 's/.*compute=([a-z0-9]*) .*description=("([^"]*)"|([^ ]*)).*/\1 \3\4/')
  expect "$gpu: ollama finds the GPU on its ROCm backend" "$target $name" "$found"
  reply=$(generate)
  read -r size vram < <(resident)
  layers=$(grep -o 'offloaded [0-9]*/[0-9]* layers to GPU' "$log" | tail -1)
  expect "$gpu: every layer on the GPU, the whole model resident" "yes yes" \
    "$([[ "$layers" =~ offloaded\ ([0-9]+)/([0-9]+) && ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" ]] && echo yes || echo "no ($layers)") $([[ -n "$size" && "$size" == "$vram" ]] && echo yes || echo "no ($size in, $vram on the GPU)")"
  expect "$gpu: generates what the CPU backend does" "$reference" "$reply"
  # A kernel the simulator could not run fails the launch; nothing may.
  expect "$gpu: every kernel ran" "0" "$(grep -c 'VirtualGPU error' "$log")"
  stop
done
exit $fail
