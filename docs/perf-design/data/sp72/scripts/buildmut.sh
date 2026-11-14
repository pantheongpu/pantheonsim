#!/bin/bash
# usage: buildmut.sh <log> [targets...]   (the mutation tree: no LTO, so a changed gcn_exec.cpp is one compile and a link)
log=$1; shift
b=/tmp/sp72-amd-tier/mut/build
[ -f $b/build.ninja ] || cmake -S /tmp/sp72-amd-tier/mut -B $b -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DVGPU_LTO=OFF > $log.cmake 2>&1 || { echo CMAKEFAIL >> $log; exit 1; }
flock /tmp/pantheonsim-builder.lock nice -n 19 ninja -C $b -j3 "$@" > $log 2>&1; echo BUILD_EXIT $? >> $log
