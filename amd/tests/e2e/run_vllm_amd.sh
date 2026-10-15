#!/usr/bin/env bash
# vLLM, unmodified, generating on a simulated AMD GPU: the same greedy
# continuation Hugging Face's transformers gives on the CPU.
#
# The Python it runs has vLLM for ROCm installed: VGPU_VLLM_PYTHON. vLLM's
# ROCm wheels (wheels.vllm.ai/rocm) come with their own PyTorch, which is
# linked against a ROCm installed on the machine (its RUNPATH names
# /opt/rocm-<version>/lib), so the machine has that ROCm's libraries; the
# simulator's HIP runtime, ROCm SMI and AMD SMI libraries go in front of them
# on the library path. Without VGPU_VLLM_PYTHON the test skips.
#
#   amd/tests/e2e/run_vllm_amd.sh [gpu] [model] [prompt]
#
# VGPU_VLLM_MODE=graphs runs vLLM as it runs by default on a card, not eager:
# torch.compile (Dynamo and Inductor, whose Triton kernels are compiled for the
# device at run time) and the model's decode step captured into HIP graphs
# (piecewise and full) for batch sizes 1 and 2, and replayed per token. A
# second prompt is generated together with the first, so the batch of two
# takes the size-2 graph. The log must show that the model was compiled and the
# graphs captured: vLLM falls back to eager when it cannot, and the text alone
# would still be right.
#
# VGPU_VLLM_TP=2 (with graphs) shards the model over two simulated GPUs, one
# worker process each, as vLLM's tensor parallelism runs: RCCL's all-reduces
# between the two processes, vLLM's custom all-reduce sharing the buffers its
# graphs capture through HIP's IPC handles, and each graph replayed with them.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
gpu="${1:-amd/mi300x}"
model="${2:-facebook/opt-125m}"
prompt="${3:-The capital of France is}"
mode="${VGPU_VLLM_MODE:-eager}"
tp="${VGPU_VLLM_TP:-1}"
[[ "$tp" =~ ^[12]$ ]] || { echo "FAIL  VGPU_VLLM_TP is 1 or 2, not $tp"; exit 1; }
[[ "$tp" == 1 || "$mode" == graphs ]] || { echo "FAIL  VGPU_VLLM_TP=2 runs in graphs mode"; exit 1; }
case "$mode" in eager|graphs) ;; *) echo "FAIL  VGPU_VLLM_MODE is eager or graphs, not $mode"; exit 1 ;; esac
prompts="$prompt"
[[ "$mode" == graphs ]] && prompts="$prompt|One plus one equals"
python="${VGPU_VLLM_PYTHON:-}"
[[ -n "$python" && -x "$python" ]] || { echo "SKIP: no Python with vLLM for ROCm (set VGPU_VLLM_PYTHON)"; exit 0; }
# Found, not imported: its PyTorch loads only with the simulator's libraries
# in front (below).
"$python" -c 'import importlib.util as u, sys; sys.exit(u.find_spec("vllm") is None)' 2>/dev/null ||
  { echo "SKIP: $python has no vllm"; exit 0; }
[[ -e "$build/shim/libamdhip64.so.7" ]] || { echo "SKIP: no HIP shim in $build/shim"; exit 0; }

tmp="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_vllm.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/telemetry"
cat > "$tmp/run.py" <<'EOF'
import sys
model, prompts, where, mode, tp = sys.argv[1], sys.argv[2].split("|"), sys.argv[3], sys.argv[4], int(sys.argv[5])
if where == "cpu":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids
        out = m.generate(ids, max_new_tokens=8, do_sample=False)
        print("TEXT", repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)), flush=True)
else:
    from vllm import LLM, SamplingParams
    # Short sequences: what a simulator can run in minutes. The KV cache takes
    # half the device.
    if mode == "graphs":
        # The default: torch.compile and CUDA graphs, for batch sizes 1 and 2.
        from vllm.config import CompilationConfig
        llm = LLM(model=model, enforce_eager=False, max_model_len=64, max_num_seqs=2,
                  max_num_batched_tokens=64, gpu_memory_utilization=0.5, dtype="float16", tensor_parallel_size=tp,
                  compilation_config=CompilationConfig(cudagraph_capture_sizes=[1, 2]))
    else:
        # Eager: no graphs, no torch.compile, one sequence.
        llm = LLM(model=model, enforce_eager=True, max_model_len=128, max_num_seqs=1,
                  max_num_batched_tokens=128, gpu_memory_utilization=0.5, dtype="float16")
    for out in llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0)):
        print("TEXT", repr(out.outputs[0].text), flush=True)
EOF

# vLLM's PyTorch, with the simulator's HIP runtime and ROCm SMI in its lib
# directory, as the PyTorch tests do it (run_pytorch.sh).
package=$("$python" -c 'import os, importlib.util as u; print(os.path.dirname(u.find_spec("torch").origin))')
mkdir -p "$tmp/torch/lib"
for e in "$package"/*; do [[ "$(basename "$e")" == lib ]] || ln -s "$e" "$tmp/torch/"; done
for f in "$package"/lib/*; do
  case "$(basename "$f")" in libamdhip64.so|librocm_smi64.so) ;; *) ln -s "$f" "$tmp/torch/lib/" ;; esac
done
ln -s "$(readlink -f "$build/shim/libamdhip64.so.7")" "$tmp/torch/lib/libamdhip64.so"
ln -s "$(readlink -f "$build/shim/librocm_smi64.so")" "$tmp/torch/lib/librocm_smi64.so"

# Two workers each keep a device's memory in RAM, so a part of it goes to disk.
extra_env=()
[[ "$tp" == 2 ]] && { mkdir -p "$tmp/spill"; extra_env=(VGPU_MEMORY_RAM_MB=768 VGPU_MEMORY_DIR="$tmp/spill"); }
run() {
  (cd "$tmp" && env -u ROCM_PATH -u ROCM_HOME PYTHONPATH="$tmp" LD_LIBRARY_PATH="$build/shim" \
     VGPU_GPU="$gpu" VGPU_DEVICE_COUNT="$tp" VGPU_VRAM_MB="${VGPU_VRAM_MB:-3072}" VGPU_TELEMETRY_PATH="$tmp/telemetry" \
     ${extra_env[@]+"${extra_env[@]}"} \
     VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CACHE_ROOT="$tmp/vllm-cache" timeout "${VGPU_VLLM_TIMEOUT:-5400}" "$python" "$tmp/run.py" "$model" "$prompts" "$1" "$mode" "$tp" 2>&1)
}
cpu=$(run cpu)
gpu_out=$(run gpu)
status=$?
fail=0
want=$(grep '^TEXT ' <<<"$cpu")
got=$(grep '^TEXT ' <<<"$gpu_out")
texts() { sed 's/^TEXT //' <<<"$1" | paste -sd'|'; }
[[ -n "$want" ]] || { echo "FAIL  the CPU reference ran"; tail -5 <<<"$cpu"; exit 1; }
if grep -q 'VirtualGPU error \[' <<<"$gpu_out"; then
  echo "FAIL  the simulator ran every kernel vLLM gave it"; grep -m5 'VirtualGPU error \[' <<<"$gpu_out" | sed 's/^/      /'; fail=1
fi
if [[ "$got" == "$want" ]]; then
  echo "ok    vLLM ($mode, tensor parallel $tp) on $gpu generates what the CPU does: $(texts "$got")"
else
  echo "FAIL  vLLM ($mode, tensor parallel $tp) on $gpu generates what the CPU does"; echo "      cpu: $(texts "$want")"; echo "      gpu: $(texts "$got")"
  [[ -n "$got" ]] || tail -8 <<<"$gpu_out" | sed 's/^/      /'
  fail=1
fi
if [[ "$mode" == graphs ]]; then
  for what in 'torch.compile took' 'Capturing CUDA graphs (PIECEWISE)' 'Capturing CUDA graphs (FULL)'; do
    if grep -qF "$what" <<<"$gpu_out"; then echo "ok    vLLM logged \"$what\""
    else echo "FAIL  vLLM logged \"$what\""; fail=1; fi
  done
fi
# The process ends as it began: no crash on the way out.
[[ $status == 0 ]] || { echo "FAIL  vLLM exited cleanly (exit $status)"; fail=1; }
exit $fail
