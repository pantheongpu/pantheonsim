#!/bin/bash
# on/build: the work build with the knob on by default (the name of the variable it reads replaced by one the bench always sets), otherwise byte for byte the same library
cd /tmp/sp72-amd-tier && mkdir -p on/build/shim && cp work/build/libamdhip64.so.6.4 on/build/ && cp work/build/vgpu on/build/ && python3 - <<"PY"
p="/tmp/sp72-amd-tier/on/build/libamdhip64.so.6.4"
b=open(p,"rb").read()
old=b"VGPU_AMD_BLOCKS\0"; new=b"VGPU_QUIET\0"+b"\0"*5
assert b.count(old)==1, b.count(old)
open(p,"wb").write(b.replace(old,new))
PY
cd on/build/shim && for l in libamdhip64.so libamdhip64.so.6 libamdhip64.so.7 libhsa-runtime64.so libhsa-runtime64.so.1; do ln -sf ../libamdhip64.so.6.4 $l; done
