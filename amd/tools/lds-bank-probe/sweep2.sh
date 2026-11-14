#!/bin/bash
# usage: sweep2.sh <device> <outfile> <threads> [binary]
dev=$1; out=$2; thr=$3; bin=${4:-./probe2}
cd /tmp/ldsprobe-saqib
printf "pmc: SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_INSTS_LDS\n" > in2-$dev.txt
: > $out
for form in 1 2 11 12 13 104 105 106 202 203 204 205 206; do
  for s in 0 4 8 16 24 32 48 64 128 4096; do
    rm -rf o2-$dev
    ROCR_VISIBLE_DEVICES=$([ $dev = 0 ] && echo 0 || echo 0,1) HIP_VISIBLE_DEVICES=$dev timeout 60 nice -n 10 /opt/rocm-7.2.0/bin/rocprofv2 -i in2-$dev.txt -d o2-$dev $bin $s $form $thr >/dev/null 2>&1
    f=$(ls o2-$dev/pmc_1/results_*.csv 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      v=$(python3 -I -c "import csv,sys
r=list(csv.DictReader(open(sys.argv[1])))[-1]
print(int(float(r['SQC_LDS_BANK_CONFLICT'])),int(float(r['SQC_LDS_IDX_ACTIVE'])),int(float(r['SQ_INSTS_LDS'])))" $f)
    else v="ERR"; fi
    echo "form=$form stride=$s threads=$thr : $v" >> $out
  done
done
echo DONE >> $out
