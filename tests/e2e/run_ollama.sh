#!/usr/bin/env bash
# Ollama, unmodified, running a model on every simulated NVIDIA GPU: it finds
# the GPU with the profile's memory, puts every layer of a model that fits on
# it, and generates the same text its CPU backend does. A GPU whose memory the
# model does not fit gets only part of it -- the placement follows each
# profile's VRAM, as on real cards.
#
# Needs `ollama` and the model (default smollm2:135m, VGPU_OLLAMA_MODEL);
# VGPU_OLLAMA_PULL=1 pulls it. Without either, it skips. VGPU_OLLAMA_PROFILES
# narrows the profiles (space-separated). AMD profiles are not run: Ollama's
# ROCm backend needs HIP kernels to execute, and the simulator runs CUDA only.
#
# Ollama bundles its own libcudart and libcublas and puts its library directory
# first for the runner it starts, so the simulator's are preloaded: a library
# already loaded under a soname satisfies every later need for it.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
model="${VGPU_OLLAMA_MODEL:-smollm2:135m}"
command -v ollama >/dev/null || { echo "SKIP: ollama is not installed"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: python3 is needed to read ollama's replies"; exit 0; }
# The CUDA major the shims were built for, which picks Ollama's backend too.
major=$(ls "$shim"/libcudart.so.[0-9]* 2>/dev/null | head -1 | sed 's/.*\.so\.//')
[[ -n "$major" ]] || { echo "SKIP: no libcudart in $shim (built without CUDA headers)"; exit 0; }
tmp=$(mktemp -d)
serve_pid=""
cleanup() { [[ -n "$serve_pid" ]] && kill "$serve_pid" 2>/dev/null; wait 2>/dev/null; rm -rf "$tmp"; }
trap cleanup EXIT
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

port=$((20000 + RANDOM % 20000))
api="http://127.0.0.1:$port"
# serve <log> [env...]: an ollama server with the given environment, ready.
serve() {
  local log=$1; shift
  env OLLAMA_HOST="127.0.0.1:$port" OLLAMA_DEBUG=1 "$@" ollama serve > "$log" 2>&1 &
  serve_pid=$!
  for _ in $(seq 100); do curl -s -m 2 "$api/api/version" >/dev/null && return 0; sleep 0.2; done
  echo "FAIL  ollama serve did not come up"; cat "$log" | tail -20; exit 1
}
stop() { kill "$serve_pid" 2>/dev/null; wait "$serve_pid" 2>/dev/null; serve_pid=""; }
# generate [num_gpu]: the reply to a fixed prompt, greedy, three tokens.
generate() {
  local opts='"num_predict":3,"temperature":0,"seed":1'
  [[ -n "${1:-}" ]] && opts="$opts,\"num_gpu\":$1"
  curl -s -m 1800 "$api/api/generate" \
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

preload="$shim/libcuda.so.1:$shim/libcudart.so.$major:$shim/libcublas.so.$major:$shim/libcublasLt.so.$major"
profiles="${VGPU_OLLAMA_PROFILES:-$("$build/vgpu" list-gpus 2>/dev/null | grep -o 'nvidia/[a-z0-9-]*' | sort -u | tr '\n' ' ')}"
for gpu in $profiles; do
  vram_mb=$(VGPU_GPU=$gpu "$build/vgpu" smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
  name=$(VGPU_GPU=$gpu "$build/vgpu" smi --query-gpu=name --format=csv,noheader 2>/dev/null)
  log="$tmp/${gpu//\//-}.log"
  serve "$log" LD_PRELOAD="$preload" LD_LIBRARY_PATH="$shim" VGPU_GPU="$gpu" VGPU_QUIET=1 \
        VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state"
  found=$(grep -o 'msg="inference compute".*library=CUDA.*description="[^"]*"' "$log" | grep -o 'description="[^"]*"')
  expect "$gpu: ollama finds the GPU" "description=\"$name\"" "$found"
  reply=$(generate)
  read -r size vram < <(resident)
  layers=$(grep -o 'offloaded [0-9]*/[0-9]* layers to GPU' "$log" | tail -1)
  expect "$gpu ($vram_mb MiB): every layer on the GPU, the whole model resident" "yes yes" \
    "$([[ "$layers" =~ offloaded\ ([0-9]+)/([0-9]+) && ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" ]] && echo yes || echo "no ($layers)") $([[ -n "$size" && "$size" == "$vram" ]] && echo yes || echo "no ($size in, $vram on the GPU)")"
  expect "$gpu: generates what the CPU backend does" "$reference" "$reply"
  stop
done

# A GPU the model does not fit: its VRAM below the model's size, so only part
# of the model goes on it.
fit_gpu="${profiles%% *}"
if [[ -n "$fit_gpu" ]]; then
  serve "$tmp/small.log" LD_PRELOAD="$preload" LD_LIBRARY_PATH="$shim" VGPU_GPU="$fit_gpu" VGPU_VRAM_MB=320 \
        VGPU_QUIET=1 VGPU_TELEMETRY_PATH="$tmp/run2" VGPU_STATE_DIR="$tmp/state2"
  reply=$(generate)
  read -r size vram < <(resident)
  expect "$fit_gpu with 320 MiB: the model does not fit, so only part of it is on the GPU" "yes" \
    "$([[ -n "$size" && -n "$vram" && "$vram" -lt "$size" ]] && echo yes || echo "no ($size in, $vram on the GPU)")"
  expect "and it still generates what the CPU backend does" "$reference" "$reply"
  stop
fi
exit $fail
