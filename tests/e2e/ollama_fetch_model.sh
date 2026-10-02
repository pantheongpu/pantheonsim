#!/usr/bin/env bash
# Pull an Ollama model into $OLLAMA_MODELS before the ollama tests run, with
# retries: ollama's registry answers 503 now and then, and a test that pulls
# for itself then fails on a registry hiccup rather than on the simulator.
# CI caches the directory (actions/cache), so this pulls only on a miss.
#
#   tests/e2e/ollama_fetch_model.sh [model]      (default smollm2:135m)
#
# Needs `ollama` on PATH and OLLAMA_MODELS set. VGPU_OLLAMA_PULL_TRIES (5)
# and VGPU_OLLAMA_PULL_WAIT (20 s, doubled after each failure) tune the retry.
set -uo pipefail
model="${1:-${VGPU_OLLAMA_MODEL:-smollm2:135m}}"
: "${OLLAMA_MODELS:?set OLLAMA_MODELS to the directory to fill}"
mkdir -p "$OLLAMA_MODELS"
command -v ollama >/dev/null || { echo "no ollama on PATH"; exit 1; }

# A CPU-only server of its own, on a port nothing else uses.
port="${VGPU_OLLAMA_FETCH_PORT:-11499}"
export OLLAMA_HOST="127.0.0.1:$port"
CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ollama serve >"${TMPDIR:-/tmp}/ollama-fetch.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null; wait "$server" 2>/dev/null' EXIT
for _ in $(seq 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done

if ollama list 2>/dev/null | grep -q "^$model "; then
  echo "$model is already in $OLLAMA_MODELS"; exit 0
fi
tries="${VGPU_OLLAMA_PULL_TRIES:-5}"
wait_s="${VGPU_OLLAMA_PULL_WAIT:-20}"
for (( i = 1; i <= tries; i++ )); do
  if ollama pull "$model" 2>&1 | tail -3; then
    if ollama list 2>/dev/null | grep -q "^$model "; then echo "pulled $model (try $i)"; exit 0; fi
  fi
  (( i < tries )) || break
  echo "pull $i of $tries failed; again in $wait_s s"
  sleep "$wait_s"
  wait_s=$(( wait_s * 2 ))
done
echo "could not pull $model after $tries tries"
exit 1
