#!/bin/bash
# the whole tree without LTO (for ctest): buildct.sh <log>
log=$1
b=/tmp/sp72-amd-tier/ct/build
[ -f $b/build.ninja ] || cmake -S /tmp/sp72-amd-tier/ct -B $b -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DVGPU_LTO=OFF > $log.cmake 2>&1 || { echo CMAKEFAIL >> $log; exit 1; }
flock /tmp/pantheonsim-builder.lock nice -n 19 ninja -C $b -j3 $(cat /tmp/sp72-amd-tier/ct.targets) > $log 2>&1; echo BUILD_EXIT $? >> $log
