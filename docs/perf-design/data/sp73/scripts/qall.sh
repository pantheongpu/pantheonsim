#!/bin/bash
# usage: qall.sh <tag> <runs> <spec>...   quick.sh over the bench's kernel rows on both targets -> out/q-<tag>.txt
# (device CPU seconds of each spec, the best of <runs> alternating single runs pinned to one core; a spec is <tree>[:ENV=v[,ENV=v]])
cd /tmp/sp72-amd-tier; tag=$1; runs=$2; shift 2
: > out/q-$tag.txt
for w in ${ROWS:-valu_int valu_f32 valu_f64 mem lds narrow smem buf ldsx flat atomics divergent wave p_dpp p_dpp_arch p_dpp_scan p_perm p_lane p_pk16 p_pkf16 p_sdwa p_vopd p_cmp p_i64 p_f64x p_smem_wide p_sbuf}; do
  ./quick.sh $runs $w gfx942 amd/mi300x "$@" >> out/q-$tag.txt
  ./quick.sh $runs $w gfx1030 amd/rx6800 "$@" >> out/q-$tag.txt
done
for w in ${MROWS:-mfma mfma_i8 mfma_fp8 mfma_bf16 mfma_f64 mfma_f64_4b smfmac}; do ./quick.sh $runs $w gfx942 amd/mi300x "$@" >> out/q-$tag.txt; done
./quick.sh $runs p_permlane gfx1030 amd/rx6800 "$@" >> out/q-$tag.txt
echo DONE >> out/q-$tag.txt
