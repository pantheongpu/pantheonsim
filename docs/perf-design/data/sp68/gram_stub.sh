#!/bin/bash
# sp68: GRAMSCHM's launch decisions in exact mode with and without the square root idiom, against the stand-in driver as a card that keeps memory and
# computes nothing (STUB_CARD_OK=1 STUB_KEEP_MEMORY=1; server3 has no card). The output is wrong by construction; what is read is where each launch ran and
# how many MiB moved. Both runs are cut at 150 s (the program's CPU reference runs after its device part).
cd ~/.cache/sp68
mkdir -p stub
B=${1:-build}   # build: this branch; build65: this branch merged with sp65-resident-streaks
for sq in 0 1; do
  (cd corp/polybenchGpu/bin && flock /tmp/pantheonsim-builder.lock nice -n 19 env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$HOME/.cache/sp68/$B/shim \
    VGPU_HOST_GPU_LIBCUDA=$HOME/.cache/sp68/libstubcuda.so STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 STUB_SM=86 STUB_MP=80 \
    VGPU_HOST_GPU=exact VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_SQRT=$sq VGPU_HOST_GPU_LOG=1 timeout 150 ./GRAMSCHM > $HOME/.cache/sp68/stub/$B-gram$sq.out 2> $HOME/.cache/sp68/stub/$B-gram$sq.err)
  echo "== SQRT=$sq rc=$?"
  grep -oE "_Z19gramschmidt_kernel[123][^ ]* (on the card|stays in the simulator)" stub/$B-gram$sq.err | sort | uniq -c
  grep -E "MiB went to the card|launches ran on the card|GPU Time" stub/$B-gram$sq.err stub/$B-gram$sq.out | tail -3
done
