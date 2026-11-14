#!/bin/bash
# usage: sweep.sh <device> <outfile> [threads]
dev=$1; out=$2; thr=${3:-32}
cd /tmp/ldsprobe-saqib
printf "pmc: SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_INSTS_LDS\n" > in-$dev.txt
: > $out
for wr in 0 1; do
 for w in 4 8 16; do
  for s in 0 4 8 12 16 20 24 28 32 36 48 64 128 256 512 1024 2048 4096; do
   rm -rf o-$dev
   ROCR_VISIBLE_DEVICES=$([ $dev = 0 ] && echo 0 || echo 0,1) HIP_VISIBLE_DEVICES=$dev timeout 60 nice -n 10 /opt/rocm-7.2.0/bin/rocprofv2 -i in-$dev.txt -d o-$dev ${BIN:-./probe} $s $w $wr $thr >/dev/null 2>&1
   f=$(ls o-$dev/pmc_1/results_*.csv 2>/dev/null | head -1)
   if [ -n "$f" ]; then
     v=$(python3 -I -c "import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))[-1]
print(int(float(r[\"SQC_LDS_BANK_CONFLICT\"])),int(float(r[\"SQC_LDS_IDX_ACTIVE\"])),int(float(r[\"SQ_INSTS_LDS\"])))" $f)
   else v="ERR"; fi
   echo "wr=$wr width=$w stride=$s threads=$thr : $v" >> $out
  done
 done
done
echo DONE >> $out
