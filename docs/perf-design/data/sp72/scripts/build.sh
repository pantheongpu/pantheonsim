#!/bin/bash
# usage: build.sh <srcdir> <log> [targets...]   (build dir is <srcdir>/build)
src=$1; log=$2; shift 2
b=$src/build
[ -f $b/build.ninja ] || cmake -S $src -B $b -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ > $log.cmake 2>&1 || { echo CMAKEFAIL >> $log; exit 1; }
flock /tmp/pantheonsim-builder.lock nice -n 19 ninja -C $b -j3 "$@" > $log 2>&1; echo BUILD_EXIT $? >> $log
