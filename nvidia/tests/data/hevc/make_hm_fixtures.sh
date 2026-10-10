#!/usr/bin/env bash
# Writes the hm_*.h265 fixtures: streams from HM 18.0 (the HEVC reference encoder, https://vcgit.hhi.fraunhofer.de/jvet/HM, built with
# cmake; set HM to its checkout) for the tools the x265 and NVENC streams of make_fixtures.sh do not use -- tiles, dependent slice segments,
# PCM, custom scaling lists, cu_qp_delta, chroma QP offsets, loop filtering across slice and tile boundaries switched off, parallel merge
# level, the range extension tools, weighted prediction, hierarchical B pictures with several temporal layers and an intra period. HM
# writes its own reconstruction next to each stream; it equals FFmpeg's decode of the stream, the RTX 3060's decode (for every stream in
# the compared transcript) and this tree's. The streams are committed: this script says how they were made, it is not run by the tests (the
# noise filter is seeded from the clock, so a rerun gives different, equally good pictures). hm_cuchroma, hm_field and hm_field_ra are
# written too, and checked against HM's reconstruction by test_hevc_decode only (hm_spec_only.txt says why).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$(mktemp -d)"
HM=${HM:-$HOME/hm}
OUT=${OUT:-out}; mkdir -p $OUT cfgs
LD=encoder_lowdelay_P_main.cfg
LDB=encoder_lowdelay_main.cfg
RA=encoder_randomaccess_main.cfg
python3 - <<'PY'
import random
random.seed(5)
out = []
def mat(name, vals, cols, dc=None):
    out.append(name)
    for i in range(0, len(vals), cols):
        out.append('  ' + ','.join(str(v) for v in vals[i:i + cols]) + ',')
    if dc is not None:
        out.append(name + '_DC'); out.append('  %d' % dc)
def ramp(n, lo, hi):
    return [max(1, min(255, lo + (hi - lo) * (x + y) // (2 * n - 2) + random.randint(0, 4))) for y in range(n) for x in range(n)]
for size, n, cols in ((4, 4, 4), (8, 8, 8), (16, 8, 8), (32, 8, 8)):
    for k in ('INTRA', 'INTER'):
        for c in (['LUMA', 'CHROMAU', 'CHROMAV'] if size != 32 else ['LUMA']):
            lo = 6 + random.randint(0, 6); hi = 40 + random.randint(0, 60)
            dc = random.randint(8, 40) if size > 8 else None
            mat('%s%dX%d_%s' % (k, size, size, c), ramp(n, lo, hi), cols, dc)
open('sl_custom.txt', 'w').write('\n'.join(out) + '\n')
PY
grep -v ConformanceMode $HM/cfg/misc/encoder_lowdelay_main_field_coding.cfg > cfgs/field_ld.cfg
grep -v ConformanceMode $HM/cfg/misc/encoder_randomaccess_main_field_coding_simple_GOP.cfg > cfgs/field_ra.cfg
mksrc() { # name size frames pixfmt [vf]
  [ -f $1.yuv ] || ffmpeg -nostdin -v error -y -f lavfi -i "testsrc2=size=$2:rate=30,noise=alls=8:allf=t${5:+,$5},format=$4" -frames:v $3 -f rawvideo $1.yuv
}
mksrc s192_8 192x144 20 yuv420p
mksrc s192_10 192x144 10 yuv420p10le
mksrc s192_12 192x144 8 yuv420p12le
mksrc s192_8_hi 192x144 4 yuv420p "noise=alls=60:allf=t"
mksrc s192_10_hi 192x144 4 yuv420p10le "noise=alls=60:allf=t"
mksrc s192_8_fade 192x144 10 yuv420p "fade=t=in:st=0:d=0.3"
mksrc s192t_8 192x288 10 yuv420p
mksrc s512_8 512x192 5 yuv420p
mksrc s640_8 640x192 5 yuv420p
mksrc s512_10 512x192 5 yuv420p10le
enc() { # name src bitdepth width height frames cfg extra...
  local name=$1 src=$2 bd=$3 w=$4 h=$5 fr=$6 cfg=$7; shift 7
  $HM/bin/TAppEncoderStatic -c $( [ -f cfgs/$cfg ] && echo cfgs/$cfg || echo $HM/cfg/$cfg ) --InputFile=$src.yuv --SourceWidth=$w --SourceHeight=$h --FramesToBeEncoded=$fr --FrameRate=30 \
    --InputBitDepth=$bd --OutputBitDepth=$bd --InternalBitDepth=$bd --BitstreamFile=$OUT/$name.h265 --ReconFile=$OUT/$name.yuv \
    --QP=${QP:-27} "$@" > $OUT/$name.log 2>&1 && echo "ok   $name $(stat -c %s $OUT/$name.h265)" || { echo "FAIL $name"; grep -i -m2 "assert\|error" $OUT/$name.log; }
}
T32="--MaxCUWidth=32 --MaxCUHeight=32 --MaxPartitionDepth=2 --QuadtreeTULog2MaxSize=5"
T16="--MaxCUWidth=16 --MaxCUHeight=16 --MaxPartitionDepth=1 --QuadtreeTULog2MaxSize=4"
enc hm_tiles s512_8 8 512 192 5 $LD $T32 --TileUniformSpacing=1 --NumTileColumnsMinus1=1 --NumTileRowsMinus1=1
enc hm_tiles_nolf s512_8 8 512 192 5 $LD $T32 --TileUniformSpacing=1 --NumTileColumnsMinus1=1 --NumTileRowsMinus1=1 --LFCrossTileBoundaryFlag=0 --SAO=1
enc hm_tiles_nonuni s640_8 8 640 192 5 $LD $T16 --TileUniformSpacing=0 --NumTileColumnsMinus1=1 --NumTileRowsMinus1=1 --TileColumnWidthArray="18" --TileRowHeightArray="6" --LFCrossTileBoundaryFlag=0
enc hm_tiles_ra s512_8 8 512 192 5 $RA $T32 --TileUniformSpacing=1 --NumTileColumnsMinus1=1 --NumTileRowsMinus1=0
enc hm_tiles_slices s512_8 8 512 192 5 $LD $T32 --TileUniformSpacing=1 --NumTileColumnsMinus1=1 --NumTileRowsMinus1=1 --SliceMode=1 --SliceArgument=20 --LFCrossSliceBoundaryFlag=0
enc hm_tiles_main10 s512_10 10 512 192 5 $LD $T32 --TileUniformSpacing=1 --NumTileColumnsMinus1=1 --NumTileRowsMinus1=1 --LFCrossTileBoundaryFlag=0 --Profile=main10 --Profile=main10
enc hm_slices_nolf s192_8 8 192 144 8 $LD --SliceMode=1 --SliceArgument=4 --LFCrossSliceBoundaryFlag=0 --SAO=1
enc hm_slices_lf s192_8 8 192 144 8 $LD --SliceMode=1 --SliceArgument=4 --LFCrossSliceBoundaryFlag=1 --SAO=1
enc hm_depslices s192_8 8 192 144 8 $LD --SliceMode=1 --SliceArgument=7 --SliceSegmentMode=1 --SliceSegmentArgument=2
enc hm_depslices_bytes s192_8 8 192 144 8 $LD --SliceMode=2 --SliceArgument=1500 --SliceSegmentMode=2 --SliceSegmentArgument=400
enc hm_wpp_depslices s192_8 8 192 144 8 $LD --WaveFrontSynchro=1 --SliceSegmentMode=1 --SliceSegmentArgument=3
enc hm_wpp_slices s192_8 8 192 144 8 $LD $T32 --WaveFrontSynchro=1 --SliceMode=1 --SliceArgument=9 
enc hm_pcm s192_8_hi 8 192 144 4 $LD --PCMEnabledFlag=1 --PCMLog2MaxSize=5 --PCMLog2MinSize=3 --PCMInputBitDepthFlag=1 --PCMFilterDisableFlag=1 --QP=4
enc hm_pcm_lf s192_8_hi 8 192 144 4 $LD --PCMEnabledFlag=1 --PCMLog2MaxSize=4 --PCMLog2MinSize=3 --PCMInputBitDepthFlag=1 --PCMFilterDisableFlag=0 --QP=8
enc hm_pcm10 s192_10_hi 10 192 144 4 $LD --PCMEnabledFlag=1 --PCMLog2MaxSize=5 --PCMLog2MinSize=3 --PCMInputBitDepthFlag=1 --PCMFilterDisableFlag=0 --QP=6 --Profile=main10
enc hm_sl_custom s192_8 8 192 144 8 $LD --ScalingList=2 --ScalingListFile=sl_custom.txt
enc hm_sl_custom10 s192_10 10 192 144 8 $LD --ScalingList=2 --ScalingListFile=sl_custom.txt --Profile=main10
enc hm_sl_custom_ra s192_8 8 192 144 8 $RA --ScalingList=2 --ScalingListFile=sl_custom.txt --TransformSkip=1
enc hm_sl_default s192_8 8 192 144 8 $LD --ScalingList=1
enc hm_cuqp s192_8 8 192 144 8 $LD --MaxCuDQPDepth=2 --MaxDeltaQP=6 --AdaptiveQP=1 --MaxQPAdaptationRange=6
enc hm_cuqp_ra s192_8 8 192 144 8 $RA --MaxCuDQPDepth=1 --MaxDeltaQP=4 --AdaptiveQP=1 --RDOQ=1
enc hm_chroma_qp s192_8 8 192 144 8 $LD --CbQpOffset=-4 --CrQpOffset=5
enc hm_chroma_qp_slice s192_8 8 192 144 8 $LD --CbQpOffset=2 --CrQpOffset=-3 --SliceChromaQPOffsetPeriodicity=2 --SliceCbQpOffsetIntraOrPeriodic=2 --SliceCrQpOffsetIntraOrPeriodic=-2
enc hm_deblock s192_8 8 192 144 8 $LD --LoopFilterOffsetInPPS=1 --LoopFilterBetaOffset_div2=2 --LoopFilterTcOffset_div2=-2
enc hm_deblock_slice s192_8 8 192 144 8 $LD --LoopFilterOffsetInPPS=0 --LoopFilterBetaOffset_div2=-3 --LoopFilterTcOffset_div2=3 --SliceMode=1 --SliceArgument=5
enc hm_ra s192_8 8 192 144 10 $RA
enc hm_ra_gop8 s192_8 8 192 144 10 misc/encoder_randomaccess_main_GOP8.cfg
enc hm_ra_main10 s192_10 10 192 144 10 encoder_randomaccess_main10.cfg
enc hm_ld_b s192_8 8 192 144 8 $LDB
enc hm_wp s192_8_fade 8 192 144 10 $LD --WeightedPredP=1 --WeightedPredB=1
enc hm_wp_b s192_8_fade 8 192 144 10 $LDB --WeightedPredP=1 --WeightedPredB=1
enc hm_wp_ra s192_8_fade 8 192 144 10 $RA --WeightedPredP=1 --WeightedPredB=1
enc hm_rext s192_8 8 192 144 8 $LD --Profile=main-RExt --TransformSkip=1 --TransformSkipFast=0 --TransformSkipLog2MaxSize=5 --ImplicitResidualDPCM=1 --ExplicitResidualDPCM=1 --ResidualRotation=1 --SingleSignificanceMapContext=1 --GolombRiceParameterAdaptation=1 --IntraReferenceSmoothing=0
enc hm_rext_ra s192_8 8 192 144 8 $RA --Profile=main-RExt --TransformSkip=1 --TransformSkipLog2MaxSize=4 --ImplicitResidualDPCM=1 --ExplicitResidualDPCM=1 --ResidualRotation=1 --GolombRiceParameterAdaptation=1 --HighPrecisionPredictionWeighting=1 --WeightedPredP=1 --WeightedPredB=1
enc hm_main12 s192_12 12 192 144 8 $LD --Profile=main-RExt --TransformSkip=1 --TransformSkipFast=0 --ImplicitResidualDPCM=1 --GolombRiceParameterAdaptation=1
enc hm_main12_ra s192_12 12 192 144 8 $RA --Profile=main-RExt --SingleSignificanceMapContext=1 --ScalingList=1
enc hm_cuchroma s192_8 8 192 144 8 $LD --Profile=main-RExt --MaxCUChromaQpAdjustmentDepth=1
enc hm_sei_md5 s192_8 8 192 144 6 $LD --SEIDecodedPictureHash=1
enc hm_ctu64 s192_8 8 192 144 6 $LD --MaxCUWidth=64 --MaxCUHeight=64 --MaxPartitionDepth=4
enc hm_tmvp_off s192_8 8 192 144 8 $RA --TMVPMode=0
enc hm_cip s192_8 8 192 144 8 $LD --ConstrainedIntraPred=1
enc hm_merge1 s192_8 8 192 144 8 $RA --MaxNumMergeCand=1
enc hm_pml s192_8 8 192 144 10 $RA --Log2ParallelMergeLevel=4
enc hm_pml_ctu16 s192_8 8 192 144 10 $RA $T16 --Log2ParallelMergeLevel=3
enc hm_2tids s192_8 8 192 144 10 misc/encoder_randomaccess_main_2tids.cfg
enc hm_ra_cra16 s192_8 8 192 144 20 $RA --IntraPeriod=16
enc hm_ra_idr16 s192_8 8 192 144 20 misc/encoder_randomaccess_main_GOP8.cfg --IntraPeriod=16 --DecodingRefreshType=2
enc hm_ra_slices s192_8 8 192 144 10 $RA --SliceMode=1 --SliceArgument=4 --SliceSegmentMode=1 --SliceSegmentArgument=2
enc hm_field s192t_8 8 192 288 10 field_ld.cfg
enc hm_field_ra s192t_8 8 192 288 10 field_ra.cfg
cp $OUT/hm_*.h265 "$HERE"
