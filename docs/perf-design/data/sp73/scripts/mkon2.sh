#!/bin/bash
# mkon2.sh <tree> <name>: <name>/build = <tree>/build with the knob on (the variable it reads replaced by one the bench always sets)
src=$1; dst=$2
cd /tmp/sp72-amd-tier && mkdir -p $dst/build/shim && cp $src/build/libamdhip64.so.6.4 $dst/build/ && cp $src/build/vgpu $dst/build/ && python3 - $dst <<"PY"
import sys
p="/tmp/sp72-amd-tier/%s/build/libamdhip64.so.6.4" % sys.argv[1]
b=open(p,"rb").read()
old=b"VGPU_AMD_BLOCKS\0"; new=b"VGPU_QUIET\0"+b"\0"*5
assert b.count(old)==1, b.count(old)
open(p,"wb").write(b.replace(old,new))
PY
cd $dst/build/shim && for l in libamdhip64.so libamdhip64.so.6 libamdhip64.so.7 libhsa-runtime64.so libhsa-runtime64.so.1; do ln -sf ../libamdhip64.so.6.4 $l; done
