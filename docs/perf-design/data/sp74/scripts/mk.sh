#!/bin/bash
# mk.sh <tree> [targets]: an LTO tree = head.tar plus push/, built (vgpu vgpuhip and the targets)
R=/tmp/sp72-amd-tier; cd $R; t=$1; shift
[ -d $t ] || { mkdir $t; tar -xf head.tar -C $t; }
(cd push && find . -type f | while read f; do cmp -s $f ../$t/$f || cp $f ../$t/$f; done)
./build.sh $R/$t $R/$t.log vgpu vgpuhip "$@"
tail -n1 $t.log
