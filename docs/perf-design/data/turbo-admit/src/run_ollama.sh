#!/bin/bash
# run_ollama.sh <tag> [env...] : one `ollama serve` under the simulator (a10g profile, A10G card), one 3-token greedy generate, then the runner unloaded so that its summary is printed
R=$HOME/par-admit
tag=$1; shift
shim=$R/build/shim
O=$R/ollama
export OLLAMA_MODELS=$O/models
port=$((22000 + RANDOM % 1000))
export OLLAMA_HOST=127.0.0.1:$port
out=$R/ollama_runs/$tag; mkdir -p $out
preload="$shim/libcuda.so.1:$shim/libcudart.so.13:$shim/libcublas.so.13:$shim/libcublasLt.so.13"
env LD_PRELOAD="$preload" LD_LIBRARY_PATH="$shim" VGPU_GPU=nvidia/a10g OLLAMA_DEBUG=1 OLLAMA_LIBRARY_PATH=$O/lib/ollama "$@" $O/bin/ollama serve > $out/serve.log 2>&1 &
pid=$!
for _ in $(seq 100); do curl -s -m 2 http://$OLLAMA_HOST/api/version >/dev/null && break; sleep 0.3; done
s=$(date +%s.%N)
curl -s -m 1500 http://$OLLAMA_HOST/api/generate -d '{"model":"smollm2:135m","prompt":"The capital of France is","stream":false,"keep_alive":0,"options":{"num_predict":3,"temperature":0,"seed":1}}' > $out/reply.json
e=$(date +%s.%N)
echo "wall=$(echo "$e - $s" | bc)" > $out/status
sleep 3
kill $pid 2>/dev/null; wait $pid 2>/dev/null
python3 -c 'import json;d=json.load(open("'$out'/reply.json"));print(repr(d.get("response", d.get("error"))))' >> $out/status
cat $out/status
