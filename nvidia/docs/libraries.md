# Vendor libraries on VirtualGPU

VirtualGPU ships its own build of the NVIDIA libraries that CUDA programs link
against. Each is presented under the real soname, so an unmodified program
finds it by putting the shim directory first on the loader path:

```bash
nvcc -cudart shared app.cu -o app -lcublas -lcudnn -lcufft
LD_LIBRARY_PATH=build/shim ./app
```

Binaries built by any CUDA 12 or 13 toolkit work: the PTX nvcc embeds is LZ4
compressed by CUDA 12 and zstd compressed by CUDA 13, and both are read here.

## nvJPEG: a codec, not a wrapper

There is no JPEG library in this repository to delegate to, so nvJPEG is a
codec written against ITU-T T.81: baseline, extended sequential and
progressive decoding (spectral selection and successive approximation),
restart markers, single-component and interleaved scans, every chroma
subsampling, grey, and Adobe CMYK/YCCK; baseline and progressive encoding with
standard or optimised (Annex K.2) Huffman tables from RGB, BGR or YCbCr planes
of any subsampling.

Every way the API reaches a decode works: `nvjpegDecode`; the batched API
(`nvjpegDecodeBatchedInitialize`, `nvjpegDecodeBatched`), which
`torchvision.io.decode_jpeg` uses on CUDA; and the decoupled three-phase API
(`nvjpegDecodeJpegHost`, `nvjpegDecodeJpegTransferToDevice`,
`nvjpegDecodeJpegDevice`, `nvjpegDecodeJpeg`) with its JPEG streams, decoder
states, pinned and device buffers and decode parameters (output format, region
of interest, CMYK). Every output format NVIDIA's default backend writes is
written: planar, grey, planar and interleaved RGB/BGR, NV12 (from 4:2:0) and
YUY2 (from 4:2:2).

The decoded pixels are NVIDIA's. Against nvJPEG 13.0 on an RTX 3060, across
twenty test images and every output format, 10 of 5.5 million samples
differed, by one or two counts: the inverse DCT here is single precision with
fused multiply-adds, rounded half up after the level shift, as NVIDIA's
rounds; chroma is upsampled by replication; the colour conversion is NVIDIA's
single-precision one with ties to even; CMYK becomes RGB as NVIDIA's makes it
(C*K/255 with an exact half rounded down). The standard leaves the inverse
DCT's internal rounding open, and those few samples are where NVIDIA's is not
this one's. The edges are the card's too, each measured: a file cut short in
its headers is `NVJPEG_STATUS_INCOMPLETE_BITSTREAM` while one cut short in its
entropy-coded data decodes what is there; NV12 only from 4:2:0, YUY2 only from
4:2:2, CMYK to RGB only with CMYK allowed; the decoupled transfer needs a
device buffer attached; the hardware backend is `NVJPEG_STATUS_ARCH_MISMATCH`
(an RTX 3060 has no JPEG engine; NVIDIA's A100 and H100 do, and the simulator
answers the same on every profile); the batched API's argument checks; which
backends make a handle and a decoder; the frame types it parses without
decoding (lossless, and samples of 2, 9, 12 or 16 bits -- encoding and
precision reported, support flag 2, decode `NVJPEG_STATUS_JPEG_NOT_SUPPORTED`)
and those it will not parse (arithmetic and differential frames, and a
hierarchical stream everywhere but the header-only parsers);
`nvjpegEncoderParamsCopyMetadata` (every APPn segment of the parsed stream
ahead of the encoder's JFIF header, which an APP0 replaces; COM left out; empty
markers if the stream was parsed without `save_metadata`);
`nvjpegEncodeGetBufferSize`'s bound. `e2e_nvjpeg_paths` checks all of it --
the decodes against the card's output by checksum -- and passes against
NVIDIA's libnvjpeg 13.0 on the card and against this one. An encoded
bitstream is a correct JPEG of its source, not NVIDIA's bytes: two encoders
make different, equally legal choices.

## cuDSS: a sparse direct solver of its own

As with nvJPEG there is nothing to delegate to, so `nvidia/src/cudss_solver.cpp`
is a multifrontal solver: minimum-degree ordering on the quotient graph,
supernodes from the elimination tree, and dense partial factorization of each
front -- LU with threshold partial pivoting, LDL^T (LDL^H) with 1x1 and 2x2
pivots, Cholesky -- where a column with no stable pivot in its front is delayed
to the parent's. It is deterministic and works in double precision whatever the
caller's type.

What the matrix fixes agrees with NVIDIA's library: solutions, residuals, the
inertia (exactly, for SCS's quasi-definite KKT systems), `INFO` for a matrix
passed as positive definite that is not. What the factorization chooses does
not: the permutation (NVIDIA's default is nested dissection), `LU_NNZ`,
`FLOPS`, `DIAG` (each pivot, reported by original row, depends on the order),
and `NPIVOTS`, which counts pivots replaced by the pivot epsilon -- NVIDIA's
library pivots only on the diagonal of each supernode and perturbs a zero
pivot where this one takes a 2x2 pivot or delays the column. Statuses,
defaults, sizes and phase rules were measured on an RTX 3060 where the
documentation leaves them open, and `nvidia/tests/e2e/cudss_*.cpp` pass against
both libraries. Refused with a message: the Schur complement mode, the nested
dissection tree, double-double values, and a matrix distributed across
processes.

## cuFile: GPUDirect Storage's compatibility mode

GPUDirect Storage moves file data straight between storage and GPU memory by
DMA, through the nvidia-fs kernel driver. A simulated GPU has no nvidia-fs, so
`libcufile.so.0` is what NVIDIA's library becomes without it: compatibility
mode, where a read is a POSIX `pread` staged through host memory into device
memory and a write the reverse. Everything a program sees -- the driver's open
count, staged and running parameters, handle and buffer registration,
`cuFileRead`/`cuFileWrite` into device, pinned, managed or pageable memory, the
batch API, the stream-ordered API, the statistics -- follows NVIDIA's libcufile
from CUDA 13.0 on an RTX 3060 without nvidia-fs, statuses included (a data-path
failure is -1 with the cuFile status in `errno`, as the card answers).
A user-space file system handle (`CU_FILE_HANDLE_TYPE_USERSPACE_FS`, whose
operation table is nvidia-fs's RDMA path's) registers, with any table but none
(`CU_FILE_IO_NOT_SUPPORTED`), and then every `cuFileRead` and `cuFileWrite` on it
returns 5006 -- the number, not -1 -- without calling the table or writing the
buffer, as the card's does; this library says so once on stderr.
`nvidia/tests/e2e/cufile_paths.cpp` passes against both libraries, the
stream-ordered calls excepted: NVIDIA's blocks in stream memory operations
under WSL, so those are checked on the simulator only.

## nvCOMP: standard bitstreams, coded on the host

An nvCOMP chunk in a standard format is that format's bitstream -- an LZ4
block, raw Snappy, raw DEFLATE, a gzip member, a Zstandard frame -- so
`libnvcomp.so.5` reads each chunk out of device memory, codes it on the host
and writes it back, with codecs of its own written from the formats'
specifications (`nvidia/src/nvcomp_codecs.cpp`). The streams are compatible
both ways, which an RTX 3060 checked: NVIDIA's nvCOMP 5.3 decodes what these
codecs write, and they decode what it writes (`nvidia/tests/e2e/nvcomp_vectors.inc`
keeps streams of both, so CI checks one direction and the card the other).
GDeflate is NVIDIA's own layout of DEFLATE for 32-lane decoding, published as
the Internet-Draft draft-uralsky-gdeflate-00; `nvcomp_gdeflate.cpp` follows it,
and where the draft leaves the layout open the reading is the one that decodes
every stream NVIDIA's library wrote word for word -- and NVIDIA's library
decodes this encoder's streams in every block layout, including the
multi-block, stored and fixed-Huffman ones nvCOMP never writes itself.

The high-level interface -- `nvcomp::LZ4Manager` and its siblings, their
configurations, `create_manager` and `get_compression_format` -- is the C++
classes NVIDIA's headers declare, laid out member for member and vtable slot
for vtable slot (`nvidia/include/vgpu_nvcomp.hpp`), so a program compiled
against either header runs on either library. A manager cuts a buffer into
chunks and writes nvCOMP's container (`NVCOMP_NATIVE`), the bare bitstream
(`RAW`) or the bitstream after its uncompressed size (`WITH_UNCOMPRESSED_SIZE`:
4 bytes for LZ4, 8 for the others). The container's layout is not documented;
it is what NVIDIA's library writes, measured on the card -- a 64-byte header,
the format's `formatSpec.hpp` struct, each chunk's offset and size, the chunks
8-byte aligned -- and each library reads the other's
(`nvidia/tests/e2e/nvcomp_manager.cpp`). The container can also carry
checksums whose algorithm is not public (no standard CRC or hash matches
them): a policy that computes them is refused, and one that verifies them if
present decompresses and reports `nvcompErrorCannotVerifyChecksums`.

LZ4's bitshuffle option (`nvcomp/lz4.h` documents that it works in 8 KiB
sub-chunks over whole groups of eight elements, not the layout) is written to
the layout the card wrote: for each bit plane, one byte per group of eight
elements, element 0 in its low bit, the planes from the element's top bit down
(`NVCOMP_BITSHUFFLE_MSB_FIRST`) or its bottom bit up, over 1-, 2- and 4-byte
types (`BITS` as bytes), the elements past the last whole group left as they
are. The card's oddities are kept: any non-zero mode but 1 compresses
LSB-first while decompression unshuffles only modes 1 and 2; the temporary
space is `32768 + chunk` rounded to 32 bytes per chunk with bitshuffle, 32768
without; the decompression output alignment is the element size; and the
8-byte integers are refused (`nvcompErrorNotSupported`) at the compress queries
and at the decompress call. The manager classes still refuse the option.
`nvcompGzipStreamingCompress` (`nvcomp/native/streaming_gzip.hpp`) reads a
stream and writes it as one gzip member (the batched API reads it back), with
the card's gigabyte workspace sizes and statuses; the streaming decompressor
answers `nvcompErrorInvalidValue` for everything, as the RTX 3060 does -- it
needs the hardware decompression engine, which no simulated GPU has either.
`libnvcomp_cpu.so.5` carries `gdeflate::compressCPU`, `decompressCPU` and the
bound `compressCPUGetMaxOutputChunkSize` with the card's limits and exception
messages (`nvidia/tests/e2e/nvcomp_cpu.cpp`: chunks cross between the CPU and
batched GPU interfaces both ways); `nvcomp::LZ4CPUManager` is not here.
`nvcomp_paths`, `nvcomp_streaming` and `nvcomp_cpu` run against NVIDIA's
libraries on the card too. Cascaded, Bitcomp and ANS stay refused, by name, once
on stderr: nvCOMP's headers and documentation describe their options and not their
bitstreams (nor Bitcomp's native API), so there is nothing documented to
implement and a chunk written from guesswork would be unreadable by NVIDIA's
library.

The compressed bytes differ from NVIDIA's (another encoder makes other
choices); the decompressed bytes never do. The queries -- alignments, maximum
output sizes, status strings, which options are refused -- answer what nvCOMP
5.3 answered on the card, except that Deflate's maximum output size follows
NVIDIA's to within 8 bytes and the temporary sizes are the simulator's (it
needs none). A buffer too small and a corrupt chunk are
`nvcompErrorCannotDecompress`, as documented; NVIDIA's LZ4 detects neither,
which on the card is a write past the buffer or a fault.

## NVENC and NVDEC: video, on the host

`libnvidia-encode.so.1` and `libnvcuvid.so.1` are driver components that
applications `dlopen` by their bare sonames, so the simulator supplies them the
way it supplies `libcuda.so.1`. Everything an application can ask of either
API is answered as an RTX 3060 (driver 595, NVENC API 13.0) answered it, from
transcripts the card printed: `nvidia/tests/e2e/nvenc_api.cpp` with
`nvenc_api.rtx3060.txt` (342 lines: every query, capability, preset
configuration, limit and error string; the session's last error is sticky, as
the card's), and `nvidia/tests/e2e/nvcuvid_paths.cpp` with
`nvcuvid_paths.rtx3060.txt` (293 lines). `run_nvenc.sh --card` and
`run_nvcuvid.sh --card` run the same programs against the real libraries.

**NVENC.** The encoders are not NVIDIA's. H.264 and HEVC are compressed by two
encoders written for this tree from ITU-T H.264 and H.265; the lossless tuning stays
the PCM stream described at the end of this paragraph. Which bitstream a picture becomes is a
pure function of the picture, its type, the QP and the pictures before it, so
encoding a frame twice gives the same bytes. What both write is read by ffmpeg, and
the card's NVDEC and the NVDEC of this tree return the same pictures from the H.264
stream.

*H.264* (`nvidia/src/nvenc_h264_enc.cpp`, with `nvenc_h264_xform.inc` and
`nvenc_h264_cabac.inc`): Intra16x16, Intra4x4 and (High profile) Intra8x8 prediction;
the 4x4 and 8x8 integer transforms with quantisation; P pictures with P_Skip and
16x16, 16x8, 8x16 and 8x8 partitions, up to four reference pictures and
quarter-sample motion search; B pictures (Main and High profile) with B_Skip and
B_Direct_16x16 by spatial direct prediction and 16x16, 16x8, 8x16 and 8x8 partitions
predicted from list 0, list 1 or both; CAVLC or CABAC with every context the syntax
uses (`entropyCodingMode`); several slices (`sliceMode` 0, 2 and 3 exactly, 1 by an
estimate of the bytes a macroblock takes); and the loop filter. Every choice is by
squared error plus lambda times the bits the entropy coder would spend, taken from the
CABAC contexts' own states. The reference picture of a P or B picture is the one this
tree's decoder (`h264_decode.cpp`) makes from the bytes just written, so encoder and
decoder cannot disagree about it. The SPS names the profile the configuration asks for
(Baseline 66, Main 77, High 100; Baseline takes no CABAC, no 8x8 transform and no B
pictures, as the card's does not) and the level the frame size and rate need.

*HEVC* (`nvidia/src/nvenc_hevc_enc.cpp`): 32x32 coding tree blocks split into coding
units of 32, 16 and 8 samples; all 35 intra modes with the filtering and boundary rules
of 8.4.4.2, NxN prediction blocks of 4x4; the DST and the 4x4 to 32x32 DCTs with
quantisation and a transform tree; P and B slices with merge and AMVP candidates (no
temporal ones), 2Nx2N, 2NxN and Nx2N units, bi-prediction and several references;
the deblocking filter, in the loop; and CABAC with every context its syntax elements
use, decided by rate-distortion cost with the bits taken from the context states. The
headers are the first VPS, SPS and PPS of the Main profile.

*Behaviour as the card's.* The API follows what an application sets: the GOP
(`gopLength` / `idrPeriod`: an IDR picture every that many pictures; `frameIntervalP = 0`:
an IDR picture and then I pictures), `NV_ENC_PIC_FLAG_FORCEIDR` and `FORCEINTRA`, the
picture type of an `enablePTD = 0` session, constant QP (`qpIntra` / `qpInterP` /
`qpInterB`, bounded by `minQP` / `maxQP`), and `frameIntervalP > 1`, which makes B
pictures. As on the card (probed: `frameIntervalP` 2, 3 and 4, gop lengths, forced IDR
pictures and the end of the stream), a picture that has to wait for the P picture that
follows it in display order is answered `NV_ENC_ERR_NEED_MORE_INPUT`; when that P
picture arrives the group is coded, P picture first, and the output buffers of the
calls that were answered "need more input" are filled, in the order of the calls, with
the group's pictures in coding order, the buffer of the call that made the group
complete last; a forced IDR picture, the end of a GOP (the group is shorter by then) and
`NV_ENC_PIC_FLAG_EOS` code what waits first. `NvEncLockBitstream` reports IDR, I, P or
B and the number of slices. The card's own encoder codes the B pictures of a group in a
pyramid (the middle one first and as a reference); this one codes them in display order
and as non-reference pictures, so the bytes differ and the protocol does not. Baseline
profile sessions take no B pictures (the card, given Baseline and `frameIntervalP` 3,
codes P pictures only), and `entropyCodingMode`, `adaptiveTransformMode`, `sliceMode`,
`sliceModeData`, `numRefL0` and `disableDeblockingFilterIDC` act as set.

*Rate control* (`nvidia/src/nvenc_rc.hpp`): for `averageBitRate` under CBR or VBR the
QP of every picture follows from a size model per picture type, learned from the
pictures coded so far, and the bits the stream may still spend; a picture that misses
its target by more than 6% is encoded again at a corrected QP after the encoder rolled
the first attempt back. Streams of 30 to 45 pictures land within 5% of the
target in the e2e tests (`nvenc_nvdec`, CBR and VBR, H.264 and HEVC): 403, 251, 1546
and 81 kbit/s for targets of 400, 250, 1500 and 80, where the card's own encoder gives
445, 219, 1877 and 91 on the same streams (its rate control is a few percent to a
quarter off at these short lengths). The
QP of an IDR picture is a search on a scratch encoder that sees only that picture, so one frame gives one
bitstream whatever came before. Intra pictures carry a digest of the input as an SEI
message (decoders skip it): a lossy encoder can lose a one-sample change in
quantisation, and encoder corruption checks (pantheon's `media_enc_virus`: one frame, a
forced IDR with the parameter sets every time, the bitstream compared with a golden
one) need the same frame to give the same bytes and a changed frame to give different
ones.

*Quality, measured and unflattering.* Scored by luma PSNR against libx264 and libx265
(`medium`, and `veryslow` as the best the codecs do; tune psnr, no adaptive
quantisation, their SEI removed) on three synthetic 352x288 clips (a panning
natural-image texture `nat352`, a zooming Mandelbrot `mandel352` and the test pattern
`test352`; 1 picture for intra, 20 for P and B), at QP 18 to 42 in steps of 4. The
table is the size our stream needs for the PSNR the reference reaches (interpolated
between the two QPs that bracket it), so `+10%` means ten percent more bytes. These
are PSNR numbers on synthetic content, not a viewing test, and the clips are small:
read them as an order of magnitude.

| clip | mode | H.264 vs x264 medium | vs veryslow | HEVC vs x265 medium | vs veryslow |
|---|---|---|---|---|---|
| nat352 | intra | +3.9% | +9.6% | +1.8% | +6.9% |
| nat352 | P | +8.6% | +14.4% | +0.7% | +16.0% |
| nat352 | B (3 per group) | +17.2% | +30.5% | +4.2% | +12.7% |
| mandel352 | intra | +3.0% | +6.0% | +4.0% | +7.3% |
| mandel352 | P | +27.2% | +42.9% | +13.0% | +62.3% |
| mandel352 | B | +27.5% | +38.1% | +7.2% | +52.4% |
| test352 | intra | +5.0% | +9.8% | +2.4% | +10.9% |
| test352 | P | +30.9% | +53.7% | +5.4% | +44.2% |
| test352 | B | +40.1% | +62.8% | +4.1% | +45.7% |

At the same QP the quantisers agree (the QP means the same step in both encoders) but
the encoders spend it differently: on `nat352` P pictures at QP 26 ours is 12901 bytes
at 40.98 dB against x264's 14425 at 41.11 dB; at QP 34, 5681 bytes at 38.49 dB against
5913 at 39.12 dB. Intra pictures are close to the references (a few percent, about
ten against `veryslow`); inter pictures are where the encoders are weak, most on the
clips with large motion or sharp edges (`mandel352`, `test352`), because the motion
search is a whole-sample diamond from predictor seeds with half and quarter-sample
refinement (a local search, which loses on fast motion) and the partitions stop
at 8x8 (H.264) or 2NxN / Nx2N (HEVC), with no RDOQ, trellis, lookahead or
adaptive quantisation. Our B pictures help less than x264's (the group is coded in display order, as non-reference pictures)
and on H.264 they cost more than P pictures on the harder clips. The H.264 numbers
are for High profile, CABAC, 8x8 transform and two references.

*The lossless tuning* (and `qpPrimeYZeroTransformBypassFlag`) is PCM: macroblocks in
H.264 (CAVLC, `nvidia/src/nvenc_h264.hpp`) and 16x16 coding tree blocks in HEVC
(`nvidia/src/nvenc_hevc.hpp`, with the CABAC encoder HEVC cannot do without: one
context-coded bin, `split_cu_flag`, and the terminate bins). Both are lossless and
conformant, every picture is an IDR picture.

*Tests.* `nvenc_nvdec.cpp` decodes the H.264 streams with ffmpeg and with this tree's
NVDEC and requires the same samples; it covers GOP structures, forced pictures, the
`NEED_MORE_INPUT` protocol and its order of pictures (B pictures, a GOP that ends in the
middle of a group, a forced IDR with pictures waiting), CABAC, the 8x8 transform,
slices, the profiles and the bit rate (CBR and VBR, within 5%); without ffmpeg only
the comparison with ffmpeg is skipped. `nvenc_h264.cpp` encodes NV12, YV12, IYUV, ARGB
and ABGR through both codecs, at sizes that are not multiples of 16 and are odd, with and
without B pictures and at a bit rate, and decodes with ffmpeg. `test_nvenc_h264_enc`
decodes the encoder's streams with this tree's decoder and requires each decoded
picture to equal the encoder's reconstruction (CABAC, 8x8, slices, references, B
pictures, rollback); `test_nvenc_hevc_enc` has ffmpeg's HEVC decoder do the same for
the HEVC encoder, where ffmpeg is installed (`test_nvenc_h264` and `test_nvenc_hevc`
still check the PCM writers with decoders of their own). `run_nvenc.sh --card` runs
`nvenc_api`, `nvenc_h264` and `nvenc_nvdec` against the driver's libraries: the card's
streams pass the same checks, and its picture types, call statuses, order of pictures,
profile bytes and slice counts agree with the simulator's. `nvenc_encode.cu` replays the
encoder SDC flow. RGB input is converted with the BT.601 limited-range matrix the card
applies. 10-bit and 4:4:4 input is refused (`NV_ENC_ERR_UNSUPPORTED_PARAM`) although the
card takes it; AV1 sessions are refused at initialisation as the card refuses them.

**NVDEC.** Motion JPEG is a sequence of independent JPEG pictures, and the
simulator has a JPEG decoder, so `cudaVideoCodec_JPEG` is decoded on the host
(nvJPEG's codec, compiled into `libnvcuvid`) and written to NV12 surfaces
as the card writes them: the pitch (the target width rounded to 512), the
decoded padding past the right edge of the last MCU and the zeros below the
picture, chroma at 4:2:0 -- a 4:4:4 or 4:2:2 picture resampled bilinearly, not
averaged in blocks, as measured -- a picture cut or padded to the decoder's
size, and a target size resampled (bilinear to enlarge; averaging to shrink,
which is the card's exactly only for whole factors). Pixels agree with the
card's to within one level on the baseline and restart-interval fixtures
(`nvidia/tests/data/jpeg`, card output in `nvidia/tests/data/nvdec`). The
subset is sequential 8-bit Huffman JPEG; the card refuses progressive pictures
(`CUDA_ERROR_INVALID_IMAGE`, the picture's status `Error`) and four-component
ones (the parser skips them), and so does this. **H.264 is decoded too**, by a decoder written from ITU-T H.264 (03/2009) alone
(`nvidia/src/h264_decode.cpp`, `h264_dec_*.inc`; the CAVLC, CABAC, scan, QP and
deblocking tables are extracted from the Recommendation's PDF by
`nvidia/tools/gen_h264_tables.py`, which is how a transcription error is kept out)
and a video parser (`h264_parser.cpp`) that produces the sequence, decode and
display callbacks. H.264 output is defined sample for sample, so a conformant
decoder is bit-exact: Baseline, Main and High progressive 4:2:0 8-bit streams with
CAVLC or CABAC, I, P and B slices, 4x4 and 8x8 transforms, scaling matrices,
explicit and implicit weighted prediction, spatial and temporal direct prediction,
multiple slices and references, long-term references, every deblocking setting,
odd sizes with cropping, all decode to the checksums of the card's NVDEC
(`nvidia/tests/data/h264`: 51 streams in 103 parser-driving runs; the card's
transcript is `nvcuvid_h264.rtx3060.txt`, 12 991 lines, and holds the CRC-32 of
every displayed frame, with `run_nvcuvid_h264.sh --card` running the same program
against the driver). The parser reproduces what the card's does and the Recommendation
leaves open: when a picture is complete, the number of decode surfaces it asks for,
the picture indices handed out, the reference slots of every picture, display order,
display delay, and timestamps (given, derived and absent); the rules were fitted to
the card's callbacks, including what does and does not complete a picture (a slice of the next one, an access unit
delimiter or an end of sequence do; SEI messages and parameter sets, repeated or changed, do not -- streams carrying one
before every slice are among the fixtures). Where the fit is approximate it is
listed in the test: with `ulMaxDisplayDelay` above one on a stream with B pictures, and
around IDR pictures, the interleaving of display callbacks with later decode callbacks
is the card's only to within one picture. A rescaled surface (a target size that
is not the display area) is the card's scaler approximated: bilinear to enlarge,
area-averaging to shrink by more than half, compared with the card's pixels to within
one level. The capability answer is the card's: H.264 4:2:0 8-bit, 48x16 to 4096x4096,
and 4:0:0, 4:2:2, 4:4:4 and every deeper bit depth are reported unsupported;
`cuvidCreateDecoder` refuses them as the card does (`CUDA_ERROR_NOT_SUPPORTED`),
as it does the `H264_SVC` codec type. Macroblock-adaptive frame/field frames (MBAFF) are decoded, field and frame pairs mixed,
with the deblocking filter's mixed edges (`mbaff_field*` in `nvidia/tests/data/h264`: interlaced test content coded with half
or more of its macroblocks as field pairs, in CABAC and CAVLC, with P, B (spatial and temporal direct), weighted prediction,
several slices and cropping, all equal to the card's frames and to ffmpeg's). Field pictures (PAFF) are decoded too, field pairs
shown as one frame, two reference fields per picture and the chroma offset between fields of different parity included
(`paff*` fixtures: the only encoder that writes field pictures is this tree's own NVENC H.264 encoder in its test-only field
mode, `make_paff.cpp`; the streams are equal to ffmpeg's decode and to the card's, and the card's callbacks -- the field
pair, its picture order counts, its display -- are reproduced). Not decoded:
flexible macroblock ordering, redundant pictures, SP/SI slices, data partitioning. Pictures that cannot be decoded report
`CUDA_ERROR_INVALID_IMAGE` from `cuvidDecodePicture`; damaged streams never crash
(`test_h264_decode` flips bits in streams under ASan and UBSan). HEVC and MPEG-2 are decoded too
(below). Every other codec
reports `bIsSupported = 0` and `cuvidCreateDecoder` / `cuvidCreateVideoParser`
answer `CUDA_ERROR_NOT_SUPPORTED`: the card has MPEG-1/4, VC-1, VP8, VP9
and AV1 engines, and a software decoder for them is a large separate project --
none is here, so an application falls back to its CPU decoder instead of
receiving a wrong picture. `cuvidCreateVideoSource` (files and URLs) needs a
demuxer and is refused the same way.

**HEVC** is decoded by a decoder written from ITU-T H.265 (v4, 12/2016) alone
(`nvidia/src/hevc_decode.cpp`, `hevc_dec_slice.inc`, `hevc_syntax.cpp`; CABAC, intra prediction in
35 modes, merge and AMVP with temporal candidates, the transforms, transform skip,
scaling lists, PCM, lossless coding, tiles, wavefronts, dependent slice segments,
the deblocking filter and sample adaptive offset), driven by `CUVIDHEVCPICPARAMS` and the
slice NAL units as the hardware is, behind a video parser (`hevc_parser.cpp`) that
reproduces the card's callbacks: the sequence format, the surface count
(`sps_max_dec_pic_buffering + 4`), the picture indices, the reference picture
set as slots of a picture table, display order by output bumping, the two-stage
treatment of a picture (the reference picture set at its first slice, the
decode and the bumping at the next picture's), display delay and timestamps. Main, Main 10 and
the 4:2:0 range extension tools (12 bits, transform-skip extensions, implicit and explicit
residual DPCM, persistent Rice adaptation, 32x32 transform skip) decode to the checksums of the
card's frames: 91 streams (`nvidia/tests/data/hevc`: 44 from x265 and the card's NVENC, 47
from HM, the reference encoder, for tiles, dependent slice segments, PCM, custom scaling lists,
`cu_qp_delta`, chroma QP offsets, filtering across slices and tiles switched off, parallel merge
level, weighted prediction, several temporal layers and intra periods), 128 parser-driving runs and
27 438 lines in the card's transcript (`nvcuvid_hevc.rtx3060.txt`; `run_nvcuvid_hevc.sh --card`
runs the same program against the driver). Every HM stream's frames equal HM's own
reconstruction and FFmpeg's decode as well. Three HM streams are checked against HM's
reconstruction only (`hm_spec_only.txt`): the card does not decode `cu_chroma_qp_offset` as specified
(its P pictures come out wrong), gets some pictures of a hierarchical field-coded stream
wrong, and reports a frame rate for a 27 MHz time scale that this library does not derive the same way. The
card's capabilities are reproduced for 4:2:0 at 8, 10 and 12 bits (129 to 8192 samples); 4:4:4 is
decoded by the card and reported unsupported here, and 4:2:2, monochrome, extended precision and
CABAC bypass alignment are refused as the card refuses them. Where the parser is only approximate: a
`ulMaxDisplayDelay` of 2 or more on a stream with B pictures hands the pictures out in the card's order
but a few of them one decode callback earlier, a VUI time scale above 250001 is scaled down by powers
of two where the card also divides by other common factors, and the sequence callback reports no bit
rate (as the card does).

**MPEG-2** (H.262) is decoded by `nvidia/src/mpeg2_decode.cpp`, written from Rec. ITU-T H.262
(02/2000) clauses 6 and 7 and Annex B: the variable-length code tables are generated from the
Recommendation's tables (`nvidia/tools/gen_mpeg2_tables.py`), with the inverse quantisation and mismatch
control, both scans, both coefficient tables, the four DC precisions, downloaded matrices, both quantiser
scale types, frame and field pictures, frame, field, 16x8 and dual prime motion compensation, concealment
motion vectors, field and frame DCTs, skipped macroblocks and multiple slices per row, and a parser
(`mpeg2_parser.cpp`) that reproduces what the card's does: sequence format, picture parameters
(reference indices, `second_field`, `PicWidthInMbs` as the card reports it, the slice offsets), the surface count,
picture indices, display order (a reference picture is shown when the next is decoded), the repeat counts of pulldown
flags, the pictures a decoder starting in the middle of a stream has to drop, the handling of damaged
and repeated headers, and timestamps. MPEG-2 leaves the inverse DCT to the decoder (Annex A bounds its error only), and the card's is
not bit-reproducible: this decoder's is an exact one with the mismatch control, so its frames differ from the card's by a
level in a few percent of the samples of an intra picture and the differences carry into the pictures predicted from
it. The tests therefore compare each frame with the card's pixels (`nvidia/tests/data/nvdec/mpeg2`, 48 streams)
within 6 levels in at most a quarter of the samples, while every callback
of 100 streams in 155 runs (16 648 lines, `nvcuvid_mpeg2.rtx3060.txt`) is the card's exactly: frame pictures
(I, P and B), field pictures with second fields, dual prime, pulldown, sequence changes, damaged and truncated streams.
Not reproduced: MPEG-1 (the card decodes it and reports it as codec MPEG-1; here it is not decoded and its
capabilities are reported unsupported), 4:2:2 and 4:4:4 (refused as on the card), the scalable extensions,
display delays of 3 or more on streams with B pictures (the same pictures in the same order, a few of them one callback
later on the card), a display delay of 2 on a stream with field pictures (some pictures one callback early or late),
and a picture that is not a multiple of 16 samples wide, where the card's last macroblock column holds garbage
the Recommendation does not define. `test_mpeg2_decode` truncates, flips bits in
and fills with random bytes the streams (it runs in the ASan, UBSan and TSan jobs).

## NVSHMEM: one GPU per process, every heap shared

NVSHMEM runs a job of PEs, each a process with a GPU, and gives each a
symmetric heap the others read and write. In `libnvshmem_host.so.3` each PE's
heap is device memory the simulator backs with a shared file (its CUDA IPC
mechanism), and every PE maps every other PE's heap into its own device
address space: the single-node, all-peer-to-peer case of NVSHMEM, where a
kernel's store to a peer's heap is a store to shared memory. The PEs meet
through a rendezvous file named by the job's unique ID
(`nvshmemx_get_uniqueid` and `NVSHMEMX_INIT_WITH_UNIQUEID`, the bootstrap that
needs no MPI), by an MPI communicator (`NVSHMEMX_INIT_WITH_MPI_COMM`,
`nvshmemx_set_attr_mpi_comm_args`, `NVSHMEM_BOOTSTRAP=MPI`), by OpenSHMEM
(`NVSHMEMX_INIT_WITH_SHMEM`, `NVSHMEM_BOOTSTRAP=SHMEM`) or by PMIx
(`NVSHMEM_BOOTSTRAP=PMI` with `NVSHMEM_BOOTSTRAP_PMI=PMIX`, which needs pmix.h when the
simulator is built and libpmix.so.2 when it runs), or, for scripts,
`VGPU_NVSHMEM_RANK`, `VGPU_NVSHMEM_NPES` and `VGPU_NVSHMEM_ID`; `nvshmem_init()` with none
of these (and PMI, the default, without libpmi.so) is a job of one PE, as with
NVIDIA's library outside a launcher. NVSHMEM's bootstraps are plugin libraries that call MPI,
OpenSHMEM or PMIx; here the job's rank, size and one token come through the application's own MPI
(Open MPI's or MPICH's handles, found by name in the process) or OpenSHMEM, or through
libpmix, and `NVSHMEM_BOOTSTRAP=plugin` accepts NVSHMEM's own plugin file names
(`nvshmem_bootstrap_mpi.so.3`, `_shmem`, `_pmix`, `_pmi`, `_pmi2`) and nothing else. The
settings NVSHMEM refuses are refused here with its words
(`bogus` and `UID` without init flags, a PMI kind it does not know, a plugin that is not named or
cannot be opened). `e2e_nvshmem_bootstrap` runs these as MPI and OpenSHMEM jobs of two PEs and compares
the shim with what NVIDIA's NVSHMEM 3.8 did on an RTX 3060 (`run_nvshmem_bootstrap.sh --card`).
PEs on one GPU make NVSHMEM's multiple-processes-per-GPU mode, status 3
(`NVSHMEM_STATUS_LIMITED_MPG`): the card ran its two PEs only that way, and so does the status
here for PEs with the same device index. The status after `nvshmem_finalize()` stays
bootstrapped (1) for every launcher's bootstrap and is 0 for a unique ID's, as measured.

The host API is implemented here: the symmetric heap, blocking, strided,
typed and stream-ordered puts and gets, signals, barriers, teams (strided and
2-D splits, translation, destruction), and the broadcast, fcollect and alltoall
collectives, and everything the type lists of NVIDIA's public headers add: for each of
the 25 RMA types (the C types, the fixed-width ones, size and ptrdiff, `half` and
`bfloat16`) the nonblocking, strided and stream-ordered puts and gets, put-with-signal on a
stream and `g` on a stream, broadcast, fcollect and alltoall; the reductions and
reduce-scatters (and, or, xor on the unsigned and fixed-width types; max, min, sum and prod
on the standard list, half and bfloat16 computed in float) with their on-stream forms; every
atomic (inc, add, fetch, set, swap, compare-swap, and, or, xor and the fetching forms, and
the half/float/double add extension); waits on a variable in stream order; and teams made
from a unique ID (`nvshmemx_team_get_uniqueid`, `nvshmemx_team_init`). A reduction runs in
team order, so a floating-point sum is the same on every PE; an atomic is a read-modify-write
under the job's lock, atomic against the other PEs' host-side atomics and signals but not
against a kernel's atomics in flight. Streams are synchronous, as everywhere here: a wait
enqueued before the write it waits for holds its PE. The device API is NVIDIA's own: the `nvshmem_*` calls in a kernel
are inline functions in NVIDIA's public headers, linked with NVIDIA's
`libnvshmem_device.a` (`-rdc`). They read one struct, `nvshmemi_device_state_d`,
which the device library's init code asks this library to fill in -- heap
bases, the peers' heaps, the team table, the collectives' synchronization
arrays, laid out as the public headers declare them (`nvidia/src/nvshmem_abi.hpp`,
checked field by field against NVIDIA's headers by
`nvidia/tests/e2e/nvshmem_device.cu`). With every peer reachable by load and
store the inline code never leaves the kernel, so puts, gets, `p`/`g`, atomics,
signal operations and waits, `quiet`, `fence` and the barriers -- at thread and
block scope, in teams, and in kernels started with `nvshmemx_collective_launch`
-- run as NVIDIA compiled them. The device library contains SASS only (PTX is
shipped for sm_120 alone), which the simulator's SASS engine runs. Each PE's
heap is memory shared between the processes, and an atomic on memory the host
maps is made with the CPU's own compare-and-swap in both engines, so PEs
adding into one word at once lose no update (`e2e_ipc` races two processes'
kernels on CUDA-IPC memory; `e2e_host_atomics` races a kernel against host
atomics).

Card ground truth is thin: on two RTX 3060s under WSL NVIDIA's library starts no job of PEs on
two different GPUs (the cards have no peer access, so the topology refuses with
`Peer GPU 1 is not accessible`, `NVSHMEMX_ERROR_NOT_SUPPORTED`, and the program exits with 255), and
two PEs on one GPU run in its multiple-processes-per-GPU mode with a working heap; this simulator's
RTX 3060 profile has no peer access either, so PEs on two simulated 3060s are refused the same way
(`two_gpus` in `run_nvshmem_bootstrap.sh`, `e2e_nvshmem_peerless`), while PEs on two data-centre profiles start. What a multi-PE job
computes here follows NVSHMEM's documentation, and so does all of the typed API above
(none of it has been compared with a card). `e2e_nvshmem_host` runs the host
API in jobs of one and three PEs (the three share a GPU when the machine has fewer, and
are then in the multiple-processes-per-GPU mode); `e2e_nvshmem_device` runs the device API in a
job of three, and skips unless NVIDIA's NVSHMEM is installed (`NVSHMEM_HOME`, or
the `nvidia-nvshmem-cu13` pip package), since its headers and device library
are not the simulator's to ship.

## cuSPARSELt: the card's pruning and compressed layout

`nvidia/src/cusparselt_api.cpp` answers the cuSPARSELt 0.10 API on the host.
Everything an application can observe was measured against NVIDIA's library on
an RTX 3060, and `nvidia/tests/e2e/sparselt_paths.cpp` passes against both:

- the descriptor checks (which refusals are `INVALID_VALUE` and which
  `NOT_SUPPORTED`), attribute defaults and sizes, and the combinations sm_86
  accepts -- fp16, bf16 and tf32 with fp32 compute, int8 with int32 compute
  into int8, int32, fp16 or bf16 when both operands run along K;
- the pruning, value for value: STRIP keeps the two larger magnitudes of each
  group of four (the lower position on a tie); TILE keeps the pattern of
  largest L1 norm in each 4x4 tile, ties broken in an order measured on the
  card (fp32 uses 1:2 groups and 2x2 tiles);
- the compressed matrix: its size and buffer size (formulas fitted to every
  shape of a grid up to 320 x 320), the kept values (fp32 ones carrying the
  tf32 rounding half-unit, as the card stores them) and the 2-bit metadata in
  the card's layout;
- Matmul's rounding: operands rounded to tf32 to nearest (ties away), fp32
  accumulation, round-to-nearest-even into every output type, saturation into
  integers, ReLU's signed zero, GELU (the tanh form), the bias type (D's type,
  float for int8 inputs), alpha and beta vectors, batches and broadcasts.

Where it differs: NVIDIA's metadata layout for 8- and 16-bit values changes at
larger shapes (seen at 256 x 64) and this one keeps the smaller shapes' layout,
so compressed bytes of large matrices differ while products do not; pruning a
group or tile that holds NaN or an infinity is not the card's; 587 pairs of
TILE patterns never tie on their own on the card, so their order here is
unmeasured; `MatmulSearch` runs the product once and keeps the plan's
configuration; NVIDIA's `CompressedSize2` counts one batch until a plan has
used the descriptor, this one always counts them all; the workspace a plan
asks for is the card's for the default split-K and is never used.
## cuTENSOR and cuTensorNet: tensor contractions, and networks of them

NVIDIA's libcutensor and libcutensornet each carry a static CUDA runtime that
cannot reach a simulated driver, so both are written here from the documented
APIs (`nvidia/include/vgpu_cutensor.h`, `vgpu_cutensornet.h`). cuTENSOR
computes on the host like cuBLAS: one loop nest per operation, in double
precision, with each operand rounded first to the precision its compute
descriptor names (half, bfloat16, TF32; single for 3XTF32, 9X16BF and 4X16F;
double for 8XINT8). cuTensorNet is built on that cuTENSOR and on the
simulator's cuSOLVER, as NVIDIA's is on theirs: a network is contracted
pairwise with `cutensorContract` into intermediates carved from the caller's
workspace, one slice at a time; QR and SVD lay the tensor out as a matrix with
`cutensorPermute` and factor it with `cusolverDn?geqrf`/`orgqr` and `gesvd`.

Statuses, attribute sizes and defaults, scalar types, FLOP and byte counts,
the padded layout of a permutation and which type and compute combinations
plan follow NVIDIA's libraries on an RTX 3060 (cuTENSOR 2.8.1, cuTensorNet
2.14, CUDA 13.0) where the documentation leaves them open:
`nvidia/tests/e2e/cutensor_paths.cpp` (440 checks) and `cutensornet_paths.cpp`
(293) pass against both, and the binaries linked against NVIDIA's pass
unchanged on the simulator. Some measured behaviour differs from the
documentation and is followed: `cutensorCreatePlan` requires a plan
preference, a contraction refuses an alignment of 0, CONJ is refused on real
data, and a repeated mode is that operand's diagonal.

### cuTensorNet distributed over MPI

`cutensornetDistributedResetConfiguration` takes a communicator (`MPI_Comm`, by pointer and size) and,
as in NVIDIA's library, the communication goes through the library `$CUTENSORNET_COMM_LIB` names,
which exports the table `cutensornetCommInterface` (`cutensornet/typesDistributed.h`, version 2;
cuQuantum ships `cutensornet_distributed_interface_mpi.c` to build one). Every choice was measured on
NVIDIA's library 2.14 on two RTX 3060s with a communication library that logs each primitive it is asked
for (`nvidia/tests/e2e/cutn_comm_mpi.c`, which also stages the device buffers through the host since the
Open MPI of a distribution is not CUDA-aware):

* A reset asks the old communicator for a barrier, then the new one for its rank, a barrier and the size
  of its shared-memory group; the library opens `$CUTENSORNET_COMM_LIB` at the first reset. A missing library
  is `DISTRIBUTED_FAILURE` only when a communicator is given, a library without the table (or with another version)
  is that either way; a primitive that fails is `DISTRIBUTED_FAILURE` (`INTERNAL_ERROR` for the rank and size
  queries of the Get calls and of a reset). Without a communicator the world is one rank.
* A contraction (`cutensornetNetworkContract`, `cutensornetContractSlices`) deals the slices of the
  group, or all of them, round robin to the ranks (the i-th of a group to rank i modulo the size, in the order a
  hash set of the ids iterates, which `std::unordered_set` with room for one more reproduces), clears the
  output unless it accumulates (an accumulating call adds to each rank's own output, so the sum holds the old
  one once per rank), and sums the outputs in place. The optimizer slices to at least one slice per rank and
  leaves every rank the plan of the lowest estimate. `cutensornetContraction` of one slice does not communicate.
* The state API's amplitudes, marginals, expectation values and norms are summed the same way. An expectation
  value of at least as many terms as ranks gives rank r the terms c with c modulo the size equal to r,
  whole and with that rank's coefficients, and sums once; with fewer, every term is contracted by all
  ranks and summed on its own.
* `cutensornetCreateDistributedTensorDescriptor` needs a configured communicator
  (`NOT_INITIALIZED` without) and gathers a word from every rank; NVIDIA's library then loads NCCL.

What each library chooses for itself is not NVIDIA's. Kernel selection is not
modelled, so every workspace estimate is zero and a plan cache entry records
the problem only (which plans NVIDIA's cache keeps is its own rule). The
contraction path is a greedy pairwise search and slicing cuts whole contracted
modes until the intermediates fit and the minimum slice count is met, where
NVIDIA's hyper-optimizer searches further: paths, slicing, FLOP counts and
workspace sizes differ, the contracted tensor does not. A sliced extent means
what NVIDIA reports -- the extent of the mode within one slice, so a mode of
extent 8 sliced completely shows as 1 and gives 8 slices.

| library | soname | what it covers |
| --- | --- | --- |
| CUDA driver | `libcuda.so.1` | contexts, modules, memory, launches |
| CUDA runtime | `libcudart.so.13` | the nvcc registration ABI, streams, events |
| NVML | `libnvidia-ml.so.1` | discovery and telemetry (`pynvml`, nvitop) |
| cuBLAS | `libcublas.so.13` | GEMM (fp32/fp64/fp16/bf16/int8 and complex) with the Ex forms' type tables and grouped batches, levels 1, 2 and 3 in every type they come in (band and packed storage included; the plane rotations bit for bit), batched GEMV in every type, triangular solves, batched LU (`getrfBatched`/`getrsBatched`/`getriBatched`/`matinvBatched`), QR (`geqrfBatched`) and least squares (`gelsBatched`), the `_64` forms, cuBLASXt over several devices and the legacy (`cublas.h`) API; see [cublas.md](cublas.md) |
| cuBLASLt | `libcublasLt.so.13` | descriptor matmul in fp64/fp32/fp16/bf16/fp8/fp4 and int8 x int8 into int32 (`CUBLAS_COMPUTE_32I`, what `torch._int_mm` calls), strided batches, row-major layouts, bias/ReLU/GELU epilogues with their auxiliary outputs and the backward ones (DRELU, DGELU, bias gradients), FP8 on an Ada profile (sm_89) as an L4's cuBLAS 13.3 answers it, and none below it (see [narrow-precision probes](lowprec.md)), FP8 tensor-wise and row-wise scales with amax, an FP8 auxiliary output with its scale and amax, and the block-scaled FP8/FP4 modes with one scale tensor per batch (Hopper and Blackwell: documentation-derived) |
| cuDNN | `libcudnn.so.9` | training and inference in the classic API: convolution forward, backward-data, backward-filter and backward-bias (every algorithm cuDNN lists, fused bias-activation), activation, pooling, softmax, LRN, batch normalization (with its fused add and activation, and as the cuDNN 8 normalization API), dropout (cuDNN's own generator, mask for mask, also between an RNN's layers), the spatial transformer, CTC loss, im2col, reductions and tensor arithmetic, each in NCHW, NHWC or any strides, in float, double, half (float or half compute) and bfloat16, INT8 convolution in NHWC and, vectorized, in `NCHW_VECT_C` (INT8x4, INT8x32), divisive normalization, tensor transforms and folding, fused-ops plans (the scale-bias-activation weight gradient among them), LSTM projections and the multi-head attention API; the graph API's convolution, matmul, pointwise, reduction, normalization (layer, instance, batch, RMS, group; backward with or without the saved statistics; batch normalization across the GPUs, in one process or several), pooling (with max pooling's index tensor), concatenation, reshape, transpose, slice, RNG, statistics-generation and softmax graphs, and scaled dot-product attention forward and backward -- the single SDPA operation and cudnn-frontend's composite graph alike, with causal, sliding-window and padding masks, bias, grouped-query heads, dropout, paged K/V caches and ragged (packed) sequences -- over ragged and INT8x4/INT8x32-vectorized tensors; RNNs |
| cuFFT | `libcufft.so.12` | C2C/R2C/C2R in 1‑D, 2‑D and 3‑D, batched, in any advanced (strided, padded) layout; the cufftXt plan and exec API, half precision included; multi-GPU plans (`cufftXtSetGPUs`, `cufftXtMalloc`/`cufftXtMemcpy` descriptors, `cufftXtExecDescriptor*`, `cufftXtQueryPlan`) with each GPU's part on its own simulated device, in NVIDIA's natural, shuffled and 1‑D string orders; LTO callbacks (`cufftXtSetJITCallback`) given as PTX, or as LTO-IR where the host has the CUDA toolkit (its libnvJitLink links it into machine code for the simulated device) |
| cuRAND | `libcurand.so.10` | host-side uniform and normal generation; Sobol' direction vectors (Joe and Kuo's, the card's to the bit) and scramble constants |
| cuSPARSE | `libcusparse.so.12` | every entry point NVIDIA's 13.0 exports. CSR/CSC/COO/BSR SpMV, SpMM (strided batches, fp16/bf16/int8), SpGEMM (and SpGEMMreuse), SDDMM, SpSV/SpSM (with updateMatrix), format conversion, CSR to CSC, in real and complex values (A, A^T and A^H); Blocked-ELL SpMM and sliced-ELL SpMV; sparse vectors (SpVV, Axpby, Gather, Scatter, Rot); the tridiagonal and pentadiagonal solvers (gtsv2, gtsv2_nopivot, gtsv2StridedBatch, gtsvInterleavedBatch, gpsvInterleavedBatch); legacy coo2csr, the CSR/CSC/COO sorts, csrgeam2, gemvi, the BSR family (bsrmv, bsrxmv, bsrmm, bsrsv2, bsrsm2, bsric02, bsrilu02, CSR to BSR and back, gebsr2gebsr, gebsr2gebsc), csric02 and csrilu02, pruning, csrcolor, nnz and compression, unsorted CSR. SpMV, SpMM, SDDMM, SpSV/SpSM solves, sparse to dense and CSR to CSC are recorded into a captured CUDA graph and run at each launch |
| cuSOLVER | `libcusolver.so.12` | Cholesky, LU, QR (with `ungqr`/`unmqr` for complex), symmetric and Hermitian eigen, SVD, in real and complex types; the reductions and their back-transforms (`sytrd`/`hetrd`, `orgtr`/`ungtr`, `ormtr`/`unmtr`, `gebrd`, `orgbr`/`ungbr`), `potri`, `lauum`, selected and generalized eigen (`syevdx`/`heevdx`, `sygvd`/`hegvd`, `sygvdx`/`hegvdx`, `sygvj`/`hegvj`); symmetric indefinite (Bunch-Kaufman `sytrf`, `Xsytrs`, `sytri`), `laswp`; the iterative refinement solvers (`<t1><t2>gesv`/`gels`, `IRSXgesv`/`IRSXgels`); the 64-bit X API with `Xgetrf`/`Xgetrs`, `Xtrtri`, `Xsyevdx`, `Xgesvd`, `Xgesvdp`, `Xgesvdr` and `Xlarft`, `Xgeev` (right eigenvectors) on real and complex matrices, Jacobi (gesvdj, syevj, heevj) and batched forms, gesvdaStridedBatched. The sparse module, cusolverSp: `csrlsvlu`/`csrlsvqr`/`csrlsvchol` (host and device), `csrlsqvqr`, `csreigvsi`, `csreigs`, the reorderings (`symrcm`, `symamd` and `symmdq` give NVIDIA's own permutations), `csrperm`, `csrzfd`, batched QR, and the low-level preview API (LU on the host, QR and Cholesky on the host and the device, step by step). The refactorization module, cusolverRf, single and batched |
| cusolverMg | `libcusolverMg.so.12` | getrf/getrs, potrf/potrs/potri and syevd on a matrix, or getrf/getrs and potrf/potrs/potri on a submatrix (IA, JA), spread over several devices in NVIDIA's column-block-cyclic layout |
| NCCL | `libnccl.so.2` | collectives (all-to-all, gather and scatter included, and the `nccl*Config` forms of each) and point-to-point across ranks; ncclCommSplit, ncclCommShrink, ncclCommGetUniqueId + ncclCommGrow, ncclCommRevoke, ncclCommSuspend/Resume/MemStats, ncclCommInitRankScalable, non-blocking communicators, pre-multiplied sums with host or device scalars, the `ncclParam*` registry; the device API's host side answers as the RTX 3060 pair does (unsupported) |
| cuStateVec (cuQuantum) | `libcustatevec.so.1` | dense and diagonal gates with any controls, controlled index-bit swaps, probabilities, projection and Pauli expectation values: what QuEST's cuQuantum backend calls. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuDSS | `libcudss.so.0` | the sparse direct solver, the whole 0.8 API: LU, LDL^T, LDL^H and Cholesky in every index width, view, base and value type, several right-hand sides, the solve sub-phases, iterative refinement, batches, a factorization or solve captured into a CUDA graph -- and SCS's GPU direct backend. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuFile (GPUDirect Storage) | `libcufile.so.0` | compatibility mode: file I/O staged through host memory into device memory, the driver and parameter API, handle and buffer registration (user-space file system handles register but, as on the card, do no I/O), batch and stream-ordered I/O, statistics; NVIDIA's statuses (CUDA 13.0) |
| nvCOMP | `libnvcomp.so.5`, `libnvcomp_cpu.so.5` | the low-level batched API and the C++ manager API for LZ4 (bitshuffle included), Snappy, Deflate, GDeflate, Gzip and Zstd, chunks and containers interoperable with NVIDIA's in both directions, CRC32, streaming gzip compression, and the CPU GDeflate library; Cascaded, Bitcomp and ANS refused (no public bitstream) |
| NVSHMEM | `libnvshmem_host.so.3` | the host API across a job of PEs, one simulated GPU per process, bootstrapped by unique ID, MPI, OpenSHMEM or PMIx: every type of the RMA lists (half and bfloat16 included) with nonblocking, strided and stream-ordered forms, typed collectives and reductions, atomics, waits in stream order, teams (also from a unique ID); the device API of kernels built with NVIDIA's NVSHMEM headers and device library, all PEs peer to peer |
| cuSPARSELt | `libcusparseLt.so.0` | 2:4 structured sparse matrix products, the whole 0.10 API: dense and structured descriptors with batches, fp16, bf16, tf32, int8 (into int8, int32, fp16, bf16) and, on an sm_89 profile, E4M3 and E5M2 (into fp16, bf16, fp32) in either operand, transposes and both orders, STRIP and TILE pruning and the prune check value for value with the card, compression with the card's sizes and layout, bias, ReLU, GELU and alpha/beta vectors, the search, graph capture. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuTENSOR | `libcutensor.so.2` | the 2.x API: contractions and trinary contractions in every type and compute combination an RTX 3060 plans (R16F, R16BF, R32F, C32F, R64F, C64F, R64F x C64F; 16F to 8XINT8), permutations with type conversion and padding, elementwise binary and trinary operations with every unary and binary operator, reductions (ADD, MUL, MAX, MIN), plan preferences, the plan cache and its file, workspace estimation, every execute call captured into a CUDA graph. NVIDIA's own carries a static CUDA runtime that cannot reach a simulated driver; this one is written from the documented API |
| cuTensorNet (cuQuantum) | `libcutensornet.so.2` | what cuQuantum Python's tensor-network contraction calls: networks built tensor by tensor (and the older descriptor and plan API), the contraction optimizer (a greedy path; slicing to a workspace limit and a minimum slice count) with its configuration and information, packed infos, workspace sizing, slice groups, conjugated inputs and hyperedges; QR, SVD (every truncation, normalization and partition; the gesvd, gesvdj, gesvdp and gesvdr algorithms) and gate splitting on cuSOLVER; gradients of a network and of a state's expectation value; distributed execution over an MPI communication library. Built on the simulator's cuTENSOR and cuSOLVER |
| NVRTC | `libnvrtc.so.13` | compiling CUDA C++ at run time with the toolkit's own NVRTC: PTX, the cubin of an sm_ target, LTO-IR, OptiX-IR, precompiled headers, time tables (below) |
| nvJitLink | `libnvJitLink.so.13` | linking PTX, or relocatable SASS, from cubins, fatbins, and host objects' and static libraries' device code into one loadable image, with the device linker's rules; LTO-IR through the toolkit's linker (below) |
| nvFatbin | `libnvfatbin.so.13` | writing fatbins at run time -- PTX, cubins, LTO-IR, a host object's relocatable PTX -- compressed as NVIDIA's are, that the driver loads |
| NPP | `libnppc.so.13` and ten siblings | image and signal primitives: arithmetic, logic and shifts, colour conversion, gamma and Bayer demosaicing, statistics, histograms and integral images, box, rank and morphological filters, gradients and Canny, affine and perspective warps, rotation, remapping, resizing and mirroring, watershed segmentation -- every entry point OpenCV, DALI, FFmpeg, jetson-utils and the CUDA Samples call (below) |
| nvJPEG | `libnvjpeg.so.13` | JPEG decode (baseline, progressive, CMYK; single, batched and decoupled APIs) and encode (baseline, progressive) |
| NVENC | `libnvidia-encode.so.1` | video encode: every status, query, capability, preset configuration and error string of the API as an RTX 3060 answers it; compressing H.264 (CAVLC and CABAC, 8x8 transform, P and B pictures, slices) and HEVC (intra, P and B pictures, in-loop deblocking) encoders, rate control, and the card's NEED_MORE_INPUT protocol for B pictures (below) |
| NVDEC | `libnvcuvid.so.1` | video decode: the cuvid parser and decoder API, Motion JPEG decoded on the host into NV12 surfaces as the card writes them; every other codec reports itself unsupported (below) |

## Why the math runs on the host

A vendor library is not user code. Nothing requires its internals to run on the
simulated device, so these entry points read their operands out of virtual
device memory, do the arithmetic on the host CPU, and write the result back.

That is the boundary a real system draws too: **application kernels are
simulated, vendor library calls are implemented.** It is also the difference
between usable and not. The SIMT interpreter retires ~10⁸ lane-ops/s; the host
CPU does real FLOPs about two orders of magnitude faster. A model whose matmuls
go through cuBLAS spends almost no time in the interpreter — only its own
custom kernels do.

Operands live in, and results return to, the same virtual device memory that
kernels see, so mixing library calls with simulated kernels works normally, and
allocation still fails with `cudaErrorMemoryAllocation` when the device is full.

## Verified against real hardware

`nvidia/tests/conformance/run_conformance.sh` compiles each test once, runs it against
NVIDIA's library on a physical GPU and against VirtualGPU's, and diffs the
output. Anything that differs is a bug in this implementation.

| suite | result on an RTX 3060 |
| --- | --- |
| `cublas_gemm` | every GEMM path bit-identical, mixed precision included; level‑1/2 to ~1e‑7 |
| `lt_and_rand` | cuBLASLt bit-identical; cuRAND matches distribution and reseed semantics |
| `cudnn_ops` | all 50 reported values bit-identical |
| `cudnn_backward` | all 174 lines agree to 1e‑6 relative: the three convolution passes (groups, dilation, both modes, NHWC, strided, 3‑D, double), fused bias-activation, activation, pooling, softmax and LRN backward, reductions with every index identical, op-tensor, transforms, dropout's backward pass, NHWC batch normalization with and without a fused add and activation, the cuDNN 8 normalization API, the spatial transformer's grid and sampler both ways, CTC loss from activations and from probabilities, im2col |
| `cudnn_types` | INT8 convolution identical to the integer; float-to-half and -bfloat16 bits identical; half and bfloat16 convolution, activation, pooling, softmax and batch normalization to 2e‑4 |
| `cufft_transforms` | all 14 bit-identical, across composite, prime, 2‑D, 3‑D and both precisions |
| `cusparse_ops` | all 15 bit-identical; CSR to CSC identical in every index and value, both bases, both value types, structure only and with values |
| `cublas_level1` | single and double `axpy`, `scal`, `dot`, `nrm2`, `i?amax` and `tbmv` agree, every index identical, negative increments included |
| `cusolver_factorizations` | Cholesky, LU (pivots included), QR and every solve bit-identical; one f32 eigenvalue differs by ~1e‑6 relative |
| `nccl_collectives` | all 24 bit-identical at two ranks on two physical GPUs |
| `nccl_comm_ops` (e2e) | all 214 checks pass against NCCL 2.31.2 and the 128 that 2.29.7's header and library have pass against NCCL 2.29.7 (it hangs when one thread suspends the ranks in a group, so that part needs a newer release), at two ranks on two physical GPUs, and all 217 at four ranks (and three) sharing them with `NCCL_MULTI_RANK_GPU_ENABLE=1`, which NCCL 2.31.2 asks for when ranks share a GPU; the simulator runs it at two, three and four ranks on virtual devices: split, shrink, grow, revoke, suspend and resume, memory statistics, non-blocking, pre-multiplied sums, all-to-all, gather, scatter, scalable init, windows, the `nccl*Config` forms, the `ncclParam*` registry, the device API's host side, and their error codes |
| `nvrtc_jit` | identical: compile a kernel at run time, load the PTX, launch it, same numbers |
| `npp_ops` | all 48 bit-identical, across arithmetic, logic, conversion, colour, statistics, morphology and resizing |
| `npp_imgproc` | 289 results: every integer image identical (a dozen near-ties marked approximate, within a count on a pixel or two), floats to 1e‑5 -- warps, rotation, remapping, ResizeSqrPixel, mirroring, logic and shifts, alpha compositing, gamma, demosaicing, lookup, statistics, histograms, integral images, rank and morphological filters, Prewitt gradients and Canny |
| `npp_segment` | watershed segmentation and marker-label compression: every buffer size and status; the segmented image under 8-way connectivity, every boundary type, 8- and 16-bit, a padded pitch; the marker labels of images without equal values; 4-way on images up to 4x4; label compression of any label image -- all identical. Labels of images with equal neighbouring values and 4-way on larger images are not (below) |
| `nvjpeg_codec` | all 24 identical: header parsing exactly, pixels to within the IDCT's own tolerance |
| `multi_gpu` | all 13 identical to two physical GPUs |

Six of those suites, `cublas_level1`, `cusparse_ops`, `cudnn_backward`,
`cudnn_types`, `npp_imgproc` and `npp_segment`, also run in CI on every pull request: `e2e_library_goldens`
compiles them against the simulator alone and compares the output with what
the RTX 3060 printed (`nvidia/tests/conformance/golden/`), so a change that
makes a routine disagree with the hardware fails on a runner with no GPU.

The library majors in that table are the ones this machine has; the build
reads each soname off the installed toolkit, so on a CUDA 12 host the same
shims come out as `libcublas.so.12`, `libcufft.so.11` and so on.

Some of those numbers came out of the hardware rather than the documentation.
cuDNN rejects `CUDNN_ACTIVATION_IDENTITY` from `cudnnActivationForward`, and
cuRAND does *not* rewind its stream when the seed is set again. Both were found
by differential testing and are matched deliberately.

cuDNN's training paths were pinned down the same way, on an RTX 3060 with
cuDNN 9.27. Its gradients read the forward pass's input for ReLU, clipped ReLU
(which passes `0 < x <= coef`) and swish, and its output for sigmoid, tanh and
ELU. Max pooling's gradient goes to the first largest input of a window,
recomputed from x; `MAX_DETERMINISTIC` sends it to the first input equal to y.
An LRN window of even size reaches one channel further up than down.
`cudnnSetTensor` takes a float for a half tensor; an INT8 result is the float
`alpha * sum` rounded to nearest even and saturated. The algorithm lists have
8, 6 and 7 entries (the max counts say 10, 8 and 9) with forward `DIRECT` and
backward-filter `WINOGRAD` never runnable; dropout's reserve space is one bit
per element, least significant first, rounded up to whole words; batch
normalization fuses an add and RELU or SWISH forward, RELU alone backward;
CTC's gradient with respect to probabilities (normalization `NONE`) is
`-posterior / y` where some path passes and `y` where none does, and a label
too long for its input costs 0 with its gradient zeroed up to the batch's
longest input. All of that is matched. Where the arithmetic order is the hardware's own choice it is not:
float and double sums are accumulated exactly here and rounded once, a
`TRUE_HALF` convolution accumulates in half in its own order. Dropout is not
an exception any more: its masks are cuDNN's, see below. And this library runs some configurations
NVIDIA's does not -- every algorithm for every shape, dy with gaps in
backward-data and x with gaps in backward-filter, double-precision NHWC
backward-data -- rather than refusing them.

Dropout is cuDNN's own, mask for mask. Its states buffer, which the
program can read, holds one cuRAND XORWOW state (48 bytes) per thread of the
dropout kernel -- 768 a streaming multiprocessor, 21504 on the RTX 3060,
which is what `cudnnDropoutGetStatesSize` reports; the card's rule for other
GPUs is not known, and 768 per SM is assumed for them -- thread t seeded as
`curand_init(seed, t, 0)` seeds it (Marsaglia's recurrence, whose subsequence
jump is a GF(2) matrix power computed here). `cudnnDropoutForward` gives
element i, in the tensor's logical order, the next output of thread i % T and
keeps it when `curand_uniform` of it exceeds p, writing `x * (1 / (1 - p))`
in float (in double for double data); the reserve space holds the mask one bit
an element. The states buffer after a set and after each pass is word for
word the card's, so `cudnnRestoreDropoutDescriptor` resumes a stream and a
second pass continues it. Edges matched: a states buffer smaller than
`cudnnDropoutGetStatesSize` says is `BAD_PARAM` (a larger one is fine), a null
one is accepted and leaves the descriptor's buffer as it was (a forward pass
through a descriptor that never had one is `BAD_PARAM`), a probability
outside [0, 1] is accepted, bfloat16 is `NOT_SUPPORTED`. An RNN's dropout
between layers is the same kernel: after each layer but the last, each
direction of the next layer gets a mask of its own over the whole of the
lower layer's output, `[T][B][hidden * dirs]`, forward direction first, layer
after layer. With sequences shorter than the longest (padded I/O enabled, in
the sequence-major, batch-major and packed layouts alike) a mask covers the
valid steps only: time step after time step, and at each step the batch
entries from the longest sequence to the shortest (equal lengths in their own
order), whatever order the data is in. The classic multi-head attention API's two dropouts are the same
kernel too: the attention dropout is one application over the probabilities,
`[batch][beam][head][query step][key step]`, the post dropout one over the
output vectors, `[batch][beam][query step][output]`, applied after the output
projection and before the residual is added, both over the dimensions of the
data the call was given, padded steps included.
`e2e_dnn_dropout`, `e2e_dnn_rnn_dropout` and `e2e_dnn_attn_dropout` check all of
it against a host model on the card's library and on this one (the RNN one
with padded batches in all three layouts, unsorted and tied lengths,
bidirectional and three layers). Not measured, and so not claimed: the states
size of GPUs other than the RTX 3060.

Two more things the card does that were first taken for refusals. The classic
API's fused `CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD` (the weight gradient of
a convolution whose input is `activation(x * eqScale + eqBias)`, the affine
result rounded to the data type, as `CONV_BNSTATS` does) runs on an RTX 3060;
it had been written down as unsupported because the configuration first
tried, half data with a float `dw`, is. The plans cuDNN makes: x, dy and dw
all half or all float (a mix, bfloat16 and double are `NOT_SUPPORTED`), float
compute type, RELU, IDENTITY or no activation, scale and bias each optional
and half or float, spatial batch-norm mode, groups, strides, dilation and
padding as the convolution has them, x and dw in either layout and dy NHWC
unless dw is NCHW too; with nothing to fuse at all the three layouts have to
agree. The sum is exact, rounded once to dw's type. `e2e_dnn_classic_paths`
checks it against the card's library and this one. The other fused ops are
unchanged: `CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION` and the two after it
are marked "reserved for future use" in cuDNN's own header and are
`NOT_SUPPORTED` on the card as here. And the graph API's multi-GPU batch
normalization (`CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS`, and the
backward one) runs on the card's two GPUs, which have no peer access, with the
peer tensors in pinned host memory: every execution computes with the batches
of all the GPUs together -- mean, inverse variance, output, running statistics
(variance unbiased over the combined count) and, backward, `dx` -- while the
gradients of the scale and bias come back as the combined sums divided by the
number of GPUs (measured with two; with more it is assumed). The words inside
the peer tensors are cuDNN's own protocol (a pair of a value and a flag for
each statistic, as far as the tensors show), which this library does not use:
its executions meet in memory (threads of one process) or, when the peer
tensors are memory shared between processes -- a file mapped `MAP_SHARED`
(pinned with `cudaHostRegister`, which the card's driver accepts for tmpfs
only, and whose device address is `cudaHostGetDevicePointer`'s, not the
host's) or CUDA IPC memory -- through files in the machine directory, each
process naming the group by the files and offsets behind its tensors; a lone
execution of a two-GPU graph fails after `VGPU_CUDNN_PEER_TIMEOUT_MS`
(60 s) with `CUDNN_STATUS_EXECUTION_FAILED` where NVIDIA's kernel would wait.
`e2e_dnn_multigpu_norm` (two GPUs) checks half and float, forward and
backward, on both libraries, with two threads and with two processes (one
GPU each) on the shared pinned file; the IPC form runs on the simulator only
(the card's two GPUs have no peer access, so it cannot open the other's
memory there).

The graph API's attention was pinned down against cuDNN 9.27 on an RTX 3060
through cudnn-frontend 1.30, the frontend PyTorch and Transformer Engine
build their attention with. The frontend emits it in two forms, and both run
here as the hardware runs them: a single `SDPA_FWD` operation (its
"unified" node, which carries a softmax operation and a subgraph of score
modifiers -- bias, causal and sliding-window masks -- run on the scores), and
a composite graph of matmuls, pointwise operations, a softmax, diagonal-band
masks, an RNG and reshapes, the backward pass always the composite one. For
that to run as a graph the general operations grew what it relies on: matmuls
whose batch dimensions group (K and V heads shared by several query heads),
per-batch M, N and K overrides with a padding value (padded sequences),
grouped reductions, view-only reshapes (a transpose through permuted
strides), scalars of lower rank broadcast from the right, and tensors in
workspace memory. `e2e_dnn_attention` runs nineteen configurations forward
and backward on the RTX 3060 against a double-precision reference -- half,
bfloat16 and float; top-left and bottom-right causal masks; a sliding window;
grouped-query heads; bias and its gradient; padding; paged K and V caches; packed (ragged)
sequences; an interleaved layout; dropout -- and NVIDIA's library agrees in
every one it has an engine for (it has none for float's backward pass), as
VirtualGPU does in all of them. Dropout keeps each probability with chance
1 - p, and the unified operation's mask is the card's, element for element:
each element is a 16-bit number from a Philox4x32-**7** call keyed by the
seed, kept when it is at most floor((1 - p) * 65536); a call covers eight
elements and is picked by the 16x16 tile (numbered down the rows first), the
column mod 8, the batch-and-head plane (eight counters apart), and the offset
(a quarter of it, plus one for every second row of eight). Measured on the RTX
3060 through the rng_dump tensor and by bisecting the probability (nothing
read from cuDNN's code), `e2e_dnn_sdpa_mask` holds the layout on the host and
checks it on both, over lengths from 1 to 1000 (multiples of 16 or not),
batches and heads up to 4 x 8, head sizes 32 to 256, half and bfloat16, seeds
past 32 bits and offsets past 2^32. Two edges: a negative offset is not
matched for every eighth column, and a single query row over 257 to 512 keys
leaves most of the dump unwritten on the card (the same formula is used there).
Seeds and offsets are read through a double, so those past 2^53 lose their low
bits. The composite graph's RNG operation is not the unified node's generator
(it follows the documented Philox4x32-10 stream); the unified node's
backward pass with dropout was not examined. Also measured and matched: max pooling's index tensor
is INT8, the maximum's row-major position within its window with padded taps
counted, and the backward pass may read it in place of x; nearest and
bilinear resampling refuse a window other than 2 when the descriptor is
finalized (cuDNN documents this for bilinear; the hardware also does it for
nearest). Interpolation has one configuration with an engine, which cuDNN
documents for its runtime-fusion engines: bilinear upsampling by 2 (NHWC, float,
window 2, strides 1/2, pre-padding 1/2, post-padding 1, so that the output
is twice the input). It runs on the RTX 3060: output i samples the input at
`s = i * stride - pre + window / 2 - 1/2` per dimension (that is i / 2),
clamped to the input, between the pixels either side of s by their distance
(zero and edge-value padding give the same output); half and bfloat16 data get an engine
whose plan cannot be built (cudnn-frontend's build_plans fails). Everything
else has no engine -- nearest in every configuration (cuDNN documents that),
NCHW, other scales and paddings, backward upsampling (cuDNN documents that
too) -- and is refused here the same way. An earlier version of this
paragraph said the card had no interpolation engine at all; it had been probed
with parameters outside the documented ones. An INT8x4 vectorized convolution (a channel dimension holding vectors
of 4) saturates as the classic API's INT8 convolution does.

The rest of the classic API was measured on the same card and matched.
`NCHW_VECT_C` tensors report strides in vectors (2x8x3x5 INT8x4 is 240
bytes); INT8x4 and UINT8x4 convolve to INT8x4 or FLOAT (NCHW on both
implicit GEMMs, NHWC on the precomputed one), INT8x32 to INT8x32 only, with
no dilation and no backward pass; `cudnnReorderFilterAndBias` permutes an
INT8x32 filter in 32-byte chunks of eight output channels (and its bias in
blocks of 32), which a `CUDNN_NO_REORDER` convolution reads back. INT8
pooling and activation round to nearest even after alpha and beta; FP8,
BOOLEAN and INT64 classic descriptors are BAD_PARAM. Divisive normalization
is `x / (K + alpha / n^d * sum (x_j - m)^2)^beta` over the window, and its
backward pass gathers over each element's own window (which differs from
the exact gradient only for even windows). A tensor transform pads, then
folds channels as `padBefore + offset * C + c`; folding and padding take no
beta. `cudnnGetFoldedConvBackwardDataDescriptors` pads K to a multiple of 8
and C to `floor(roundup(C * sh * sw, 8) / (sh * sw))`. LSTM projections
keep the weight space's matrices before all its biases (as every RNN now
does), and cell clipping limits the state where it is read, with float
bounds, reporting cy unclipped. Fused-ops plans run BNSTATS convolution
(half NHWC, the affine result rounded to half) and both batch-norm
finalizations. Multi-head attention keeps W_Q, W_K, W_V, W_O and then the
biases in the weight buffer, input element slowest, and writes steps past a
sequence as the output bias plus the residual.

## Checked in CI without hardware

The paths PyTorch takes through these libraries are also covered by
self-checking programs in `nvidia/tests/e2e/`, which compare each result with a
host reference (a direct DFT, a dense product, a residual or reconstruction)
and so run on every pull request with no GPU. `run_lib_check.sh` builds and
runs them; each is a ctest of its own.

| test | covers | torch |
| --- | --- | --- |
| `e2e_fft_layouts` | cufftXt plans, strided and padded layouts of every rank, 2‑D/3‑D C2R, half | `torch.fft` |
| `e2e_fft_multigpu` | cuFFT on two devices: what multi-GPU planning refuses, batched, 2‑D/3‑D (x split, then y) and 1‑D (strings, input-shuffled) descriptors, R2C/C2R, device-to-device copies; passes on an RTX 3060 pair too | multi-GPU FFTs |
| `e2e_fft_callbacks` | LTO load and store callbacks in C2C (strided, batched, callerInfo), R2C, Z2Z and C2R; what cufftXtSetJITCallback and planning refuse; libcufft.so's NOT_IMPLEMENTED legacy callbacks | cuFFT callbacks (CUDA 12.6+) |
| `e2e_fft_legacy_callbacks` | `cufftXtSetCallback`: `CUFFT_NOT_IMPLEMENTED` from the shim and from NVIDIA's `libcufft.so` (`--card --dynamic`); with `--card`, the documented route through NVIDIA's static cuFFT on an RTX 3060 (load and store callbacks in C2C, R2C, Z2Z, C2R; callerInfo; clearing one), which cannot run on the simulator | cuFFT callbacks (before CUDA 12.6) |
| `e2e_fft_lto_callbacks` | `cufftXtSetJITCallback` given as LTO-IR (nvcc -dlto): C2C, R2C, Z2Z with separate load and store images; passes against NVIDIA's cuFFT on an RTX 3060 (`--card`), and on the simulator where the host has the CUDA toolkit | cuFFT callbacks (CUDA 12.6+) |
| `e2e_cutensornet_state_gradient` | the gradients of a state's expectation value (`cutensornetExpectationComputeWithGradientsBackward`): PyTorch's convention for complex data, checked against finite differences of the forward value for one-mode and two-mode gates, strided buffers, mixed states and the norm's adjoint; the zeros for adjoint gates and gates outside the light cone; overwrite and accumulate; the arguments and their order; real and single-precision states | quantum tensor-network libraries |
| `e2e_cutensornet_mpi` | cuTensorNet over MPI (Open MPI, two and three ranks, one simulated GPU each): the communicator calls and the primitives behind each, which rank contracts which slice, the sums, failing primitives, the state API's sums and an expectation value's terms, libraries that cannot be loaded; the checks pass against NVIDIA's library on two RTX 3060s (`run_cutensornet_mpi.sh --card`) | multi-GPU quantum simulation |
| `e2e_nvshmem_bootstrap` | NVSHMEM bootstrapped by an MPI communicator, `NVSHMEM_BOOTSTRAP=MPI`, the MPI plugin by name, PMIx, OpenSHMEM and PMI without its library, and the settings it refuses; the shim and NVIDIA's NVSHMEM 3.8 on an RTX 3060 print the same (`run_nvshmem_bootstrap.sh --card`) | multi-GPU jobs |
| `e2e_cutensornet_decomp_paths` | cuTensorNet's SVD with every algorithm (reconstruction, singular values, residual and sweeps, error in sigma, gesvdr's host scratch and rank rule), half precision and capture refusals, workspace statuses, network gradients (real and complex, overwrite and accumulate, argument errors) | quantum tensor-network libraries |
| `e2e_solver_paths` | cuSOLVER X API, gesvdj/syevj and their batched forms, gesvdaStridedBatched, batched potrf/potrs; cuBLAS batched LU | `torch.linalg` |
| `e2e_solver_sparse_paths` | cusolverSp: LU, QR and Cholesky solves in S/D/C/Z with every reorder, singularity, least squares, shift-inverse eigenvalues, reorderings (NVIDIA's permutations for symrcm, symamd, symmdq), permutations, batched QR | `scipy`-style sparse solves |
| `e2e_solver_mg_paths` | cusolverMg on two devices: getrf/getrs, potrf/potrs/potri, syevd, IPIV's layout, submatrices, the grids NVIDIA's refuses | multi-GPU dense solvers |
| `e2e_solver_dense_paths` | the reductions and back-transforms, potri/lauum, syevdx and the generalized eigensolvers, the iterative refinement solvers, Xgetrf/Xgetrs, Xtrtri, Xsyevdx, Xgesvd, Xgesvdp, Xgesvdr, Xlarft, Xgeev's left-eigenvector refusal, the handle modes and Jacobi getters | `torch.linalg.eigh` on generalized problems, `cholesky_inverse`, mixed-precision solves |
| `e2e_solver_rf_paths` | cusolverRf: setup, analyze, refactor, solve, the documented defaults, zero pivots and the boost, the unit-diagonal formats and split factors, batched (on the simulator) | sparse refactorization loops (circuit simulation) |
| `e2e_solver_sparse_ll_paths` | cusolverSp's low-level preview API: threshold LU, QR with a shift and least squares, Cholesky in its elimination tree's postorder, host and device, the call-order refusals | sparse direct solvers built on the preview API |
| `e2e_solver_sytrf_paths` | sytrf + Xsytrs and sytri in S/D/C/Z, both triangles, 2x2 pivots, singular D; laswp; Xgeev on complex matrices | `torch.linalg.ldl_factor`, complex `eig` |
| `e2e_sparse_paths` | coo2csr and the sorts, batched and half SpMM, SpGEMM, csrgeam2, SDDMM, SpSV/SpSM | `torch.sparse` |
| `e2e_sparse_complex_paths` | SpMV, SpMM, SDDMM, SpSV/SpSM, SpGEMM, conversions and csrgeam2 on complex values, every op; the type combinations and conjugate transposes NVIDIA's refuses | complex `torch.sparse` |
| `e2e_sparse_bsr_paths` | generic BSR (SpMV, SpMM, SDDMM) and the legacy BSR family: bsrmv/bsrxmv/bsrmm, bsrsv2/bsrsm2 with their zero pivots, bsric02/bsrilu02 (and csric02/csrilu02) with ILU's boost, CSR to BSR and back | preconditioned iterative solvers |
| `e2e_sparse_tridiag_paths` | gtsv2 (pivoting), gtsv2_nopivot and gtsv2StridedBatch (PCR, and CR past 2048 and 512 unknowns: which unknowns a zero pivot spoils), the interleaved Thomas, LU and QR and the pentadiagonal QR with what each leaves in its inputs, in S, D, C and Z | ADI and spline solvers, PyTorch's `torch.linalg` tridiagonal paths |
| `e2e_sparse_vector_paths` | sparse vectors (SpVV in every compute type, Axpby, Gather, Scatter, Rot), gemvi, Blocked-ELL SpMM and DenseToSparse, sliced-ELL SpMV, and what NVIDIA's refuses for each | sparse optimizers, block-sparse attention |
| `e2e_sparse_helper_paths` | pruning (by threshold and percentage), nnz and compression, unsorted CSR, gebsr2gebsr/gebsr2gebsc, csrcolor, SpGEMMreuse, SpGEMM's product count and memory estimate, SpMMOp's refusal, SpSV/SpSM updateMatrix, the logger, the CSC sort | model pruning, multigrid setup |
| `e2e_solver_metis_paths` | cusolverSpXcsrmetisndHost on 78 graphs (meshes, bands, trees, random, with isolated vertices) against the permutations the card printed (`run_metis_card.sh` re-checks them on a card), the options array, base-one input, and the arguments the card refuses | sparse direct solvers' fill-reducing orderings |
| `e2e_sparse_color_paths` | csrcolor: proper colorings at every fraction, ncolors, reordering, the descriptor's index base, device pointer mode, the refusals | graph coloring for parallel smoothers |
| `e2e_blas_u8gemm_paths` | cublasUint8gemmBias: 400 random calls against the card's formula (10,505 outputs), rounding, saturation, shifts, transposes, the arguments the card refuses | |
| `e2e_blas_emulation_paths` | fixed-point emulation of double precision: bit for bit the card's results at every mantissa bit count from 4 to 64 (`run_blas_emulation_card.sh` re-checks them), the strategy and environment variable, the math mode, dynamic control, batches, Zgemm | |
| `e2e_lt_emulation_paths` | cublasLtEmulationDesc_t and its matmul attribute: defaults, refusals, in-place initialisation, the checks at the matmul | |
| `e2e_complex_paths` | complex cuBLAS (GEMM in every batched form, GEMV, level 1, trsm, batched LU, herk, hemv) and cuSOLVER (LU, Cholesky, QR with ungqr/unmqr, heevd/heevj, gesvd/gesvdj, the X API on complex types) | complex tensors in `torch.linalg`, `@` |
| `e2e_lt_paths` | fp16/bf16 matmul with bias epilogues, strided batches, row-major layouts, FP8 scales and amax | `addmm`, `bmm`, `_scaled_mm` |
| `e2e_lt_epilogue_paths` | RELU_AUX/GELU_AUX's mask and input, DRELU/DGELU and their bias gradients, BGRADA/BGRADB, in fp16/bf16/fp32/fp64, and what the card refuses | a training step's backward pass (cuBLASLt-fused linear layers) |
| `e2e_graph_capture_libs` | cuBLASLt's matmul with a bias epilogue, cuDNN's graph-API convolution and a driver-API `cuLaunchKernel` recorded into a captured CUDA graph, their descriptors, plan and pack destroyed after the capture, then launched with new inputs: the capture runs nothing, every launch reads what the graph's kernels wrote before it (`run_graph_capture_libs.sh --card` runs the same program on NVIDIA's libraries; it passes there) | PyTorch's CUDA graphs: `torch.cuda.graph`, `make_graphed_callables`, `mode="reduce-overhead"`, Triton kernels inside a graph |
| `e2e_graph_capture_dnn`, `e2e_graph_capture_fft`, `e2e_graph_capture_solver`, `e2e_graph_capture_solver_sp`, `e2e_graph_capture_rand`, `e2e_graph_capture_jpeg`, `e2e_graph_capture_npp` | cuDNN's classic API, cuFFT, cuSOLVER's dense API, cuRAND, nvJPEG's decode and NPP's `_Ctx` functions recorded into a captured CUDA graph (descriptors, plans and parameter objects destroyed after the capture), then launched with new inputs, each compared with an eager run; the calls NVIDIA's library cannot capture answer as it does and invalidate the capture. `graph_capture_common.h` is the harness; `run_graph_capture.sh <name> <libs> --card` runs the same program on NVIDIA's libraries (all pass there) | PyTorch's CUDA graphs with BatchNorm, RNNs, FFTs, linear algebra; XLA, Warp |
| `e2e_graph_capture_nccl` | NCCL's all-reduce (sum, max, average, in place, with a pre-multiplied operator destroyed after the capture), broadcast, reduce, all-gather, reduce-scatter and send/recv recorded into captured graphs, a rank per thread on two devices, launched together and compared with eager calls, the order of collectives still in step afterwards; `run_graph_capture_nccl.sh --card` runs it on NVIDIA's NCCL (2.28.9) on two RTX 3060s | PyTorch DDP and `torch.cuda.graph` with NCCL collectives, Megatron |
| `e2e_graph_capture_driver`, `e2e_graph_driver` | the driver API's stream calls inside a capture made with `cuStreamBeginCapture` (copies of every shape, fills, host functions, events across streams, stream memory operations, stream-ordered allocation, and the calls a capture refuses) and the driver's explicit graphs (`cuGraphAdd*Node`, Get/SetParams, executable-graph updates, clones, user objects, capture into a graph); both run on NVIDIA's driver too | XLA, Warp, cuda-python |
| `e2e_lt_blockscaled_paths` | MXFP8 and NVFP4 block scales in the tiled layout, the 128-element and 128x128 FP32 forms, D's block quantization and its output scales (simulator only: documentation-derived) | `_scaled_mm` with block scales |
| `e2e_lowprec_lt` | 1278 cuBLASLt descriptors (FP8 types, layouts, alignment, scales, amax, saturation, beta, batches, epilogues with their auxiliary outputs, the backward ones, row-major and padded layouts, block-scaled modes), each printed as its heuristic and matmul status and a hash of D, the auxiliary output, amax and block scales, compared with what an L4, an L40S, an RTX 3060 and an RTX PRO 6000 printed ([lowprec.md](lowprec.md)) | `_scaled_mm`, Transformer Engine's FP8 linear layers |
| `e2e_lowprec_sparselt` | 1097 cuSPARSELt problems (FP8 and FP4 inputs, every output, compute type, layout, alignment, activation, bias, alpha vector, scale mode), the same way against an L4, an L40S, an RTX 3060 and an RTX PRO 6000 (one case of the last differs, `known-gaps.txt`) | FP8 2:4 sparse inference |
| `e2e_lowprec_cvt` | the packed FP8 conversions (e4m3x2, e5m2x2 from f32 and f16x2, .relu, and back) over 512 values with NaN, infinities, zeros, subnormals and ties, on SASS and PTX, against an L4 | `__nv_fp8` conversions on sm_89 and later |
| `e2e_dnn_int8x32` | an INT8x32 convolution through the graph API with the filter reordered by `cudnnReorderFilterAndBias` (`CUDNN_TENSOR_REORDERING_INT8x32`); the same program passes against NVIDIA's cuDNN on an RTX 3060 | INT8 inference engines built on cudnn-frontend |
| `e2e_blas_packed_paths`, `e2e_blas_batched_paths`, `e2e_blas_64_paths`, `e2e_blas_legacy_paths`, `e2e_blas_xt_paths` | cuBLAS's band and packed level 2, batched GEMV, getri/matinv, syrkx/herkx, the `_64` forms, the handle settings, the legacy API and cuBLASXt on two devices | SciPy-style BLAS callers, multi-GPU GEMM |
| `e2e_dnn_backward` | cuDNN's convolution passes against each other, every backward pass against finite differences, algorithm lists, status codes, dropout, an LSTM's gradients through dropout, LSTMs in half, bfloat16 and double, CTC's gradient | `conv2d`, pooling and activation backward, `nn.LSTM(dropout=)`, `ctc_loss` |
| `e2e_dnn_classic_paths` | cuDNN's classic API beyond the training paths: INT8x4/UINT8x4/INT8x32 convolution and fused bias-ReLU against an integer reference (also through `cudnnReorderFilterAndBias`), transforms to and from `NCHW_VECT_C`, INT8 pooling and activation, divisive normalization against its formula and finite differences, padding/folding/unfolding transforms and the folded backward-data pipeline, LSTM projections and clipping against a host LSTM and finite differences, the RNN getters, fused-ops plans, multi-head attention forward and both gradients | `nn.LSTM(proj_size=)`, `nn.MultiheadAttention`-style models, INT8 inference engines |
| `e2e_dnn_dropout` | cuDNN's classic dropout bit for bit: the states buffer cudnnSetDropoutDescriptor fills (cuRAND XORWOW, one state per kernel thread), the output, the reserve space and the states after a pass in float, half and double, a second pass and a restored descriptor continuing the streams, the backward pass, and the statuses at the edges | `nn.Dropout` through cuDNN in other frameworks, RNN dropout |
| `e2e_dnn_rnn_dropout` | the dropout between an RNN's layers, mask for mask, in two and three layers, unidirectional and bidirectional, against the same host model | `nn.LSTM(dropout=)`, `nn.GRU(dropout=)` |
| `e2e_dnn_multigpu_norm` | multi-GPU batch normalization, forward and backward, half and float, two GPUs and two threads, peer tensors in pinned host memory | apex-style synchronized batch norm through cudnn-frontend |
| `e2e_dnn_graph` | cuDNN graphs: conv + bias + ReLU, dgrad + ReLU backward, matmul + bias + GELU, reductions, pointwise forward and backward, layer/RMS/batch/group norm forward and backward, backward without saved statistics, max and average pooling both ways, max pooling's index tensor, asymmetric padding, concatenation, statistics generation, RNG, reshape, transpose, slice, an INT8x4 vectorized convolution | `cudnn_convolution_add_relu`, cudnn-frontend |
| `e2e_dnn_fp8_attention`, `e2e_dnn_fp8_attention_frontend` | per-tensor FP8 attention forward (descale of Q, K, V and S, scale of S and O, amax of S and O) on a Hopper profile, built from the backend API and by cudnn-frontend's `sdpa_fp8`, against the documented formulas (documentation-derived; no card), and MXFP8's refusal | Transformer Engine's FP8 attention |
| `e2e_dnn_sdpa_mask` | the unified attention node's dropout mask (the rng_dump tensor) cell by cell against a host model of the card's layout (Philox4x32-7, 16x16 tiles, row groups, heads), 21 shapes, offsets, seeds and types, and the output over the kept probabilities; run on the RTX 3060 too (`run_dnn_card.sh dnn_sdpa_mask`) | `scaled_dot_product_attention` with dropout through cuDNN |
| `e2e_dnn_attention` | cuDNN scaled dot-product attention built by cudnn-frontend 1.30 (fetched): the unified and composite forms forward and backward, causal (both alignments) and sliding-window masks, bias, grouped-query heads, padding, paged K/V caches, ragged sequences, dropout, half/bfloat16/float | `scaled_dot_product_attention` with the cuDNN backend, Transformer Engine |

The programs were also run against NVIDIA's own libraries on an RTX 3060, so
what they assert is what the real libraries do, not only what these do. Three
things that run turned up: cuSPARSE 13.0's batched CSR SpMM uses the first
matrix's row offsets for every member (13.2 follows the stride, as documented
and as this does), NVIDIA's SDDMM refuses a NULL buffer even when it asked
for none, and it takes a conjugate transpose of a complex operand, which it
does not document, and computes neither A^H B nor anything else with it (this
refuses one with NOT_SUPPORTED). The FP8 matmuls were compared against an NVIDIA L4 (sm_89) rented for the
purpose, over 1247 descriptors, and against the RTX 3060's refusal of all of them
([narrow-precision probes](lowprec.md)); their output encoding is also checked
against `cuda_fp8.h`'s conversion, bit for bit.

The same suite runs on a rented multi-GPU machine through
`nvidia/tools/verify-multigpu-cloud.sh`, which builds VirtualGPU there and compares
against that machine's own libraries -- so the results above are not one
laptop's. It has been run on two shapes so far, and all thirteen suites match on both:

| machine | what it adds |
| --- | --- |
| 2x H100 SXM5, CUDA 12.8 | NVLink, and everything the older toolkit does differently -- LZ4 fatbins, the `cudaGetDeviceProperties_v2` spelling, different soname majors |
| 4x H100 SXM5 | four-rank NCCL against NVIDIA's libnccl, and four ranks across four processes |
| 8x A100 80GB SXM4, sm_80 | a second architecture, eight-rank NCCL, and eight ranks across eight processes |

Reduced precision is compared with a tolerance, not bit-for-bit, wherever the
two implementations legitimately differ: these libraries compute in double and
round on the way out, so single-precision results are if anything slightly more
accurate than hardware's. A real GPU does not reproduce its own results
bit-for-bit across architectures either.

## More than one GPU

A virtual rack is `VGPU_DEVICE_COUNT` devices built from one profile. Each owns
a disjoint window of the process address space, because CUDA guarantees unified
virtual addressing: a device pointer is unique process-wide and identifies the
device that owns it. Copies, memsets, frees and address-range lookups all
resolve a device pointer against its owner rather than against whichever device
happens to be current, so a cross-device `cudaMemcpyDeviceToDevice` moves the
bytes it should. Without separate windows two devices hand out the same numeric
address for different memory and that copy silently reads the wrong buffer --
which is what it used to do.

`nvidia/tests/conformance/multi_gpu.cu` compares the semantics against a real
multi-GPU machine: enumeration, per-device allocation and kernels,
`cudaSetDevice` stickiness, allocation isolation, peer copies synchronous and
asynchronous, device-to-device through the generic entry point, and an event on
the destination device.

## NCCL: a file-backed transport

There is no NVLink here, so NCCL moves bytes through a directory of
memory-mapped files. Each rank publishes its contribution to its own file, the
ranks rendezvous on a shared metadata segment, and every rank computes its own
output from the peers' files.

That is what makes the ordinary deployment work: one rank per *process*, as
`torchrun` and `mpirun` launch it, where the ranks share no address space to
shortcut through. The single-process forms — `ncclCommInitAll`, and one thread
issuing every rank's call inside `ncclGroupStart`/`ncclGroupEnd` — go through
the same path.

```bash
VGPU_NCCL_DIR=/tmp/my-job     # rendezvous directory (default: $TMPDIR/vgpu-nccl-$UID)
VGPU_NCCL_TIMEOUT=300         # seconds before a stalled collective gives up
```

A collective that never matches up times out with a diagnosis rather than
hanging, because an unexplained hang is the worst way to learn that two ranks
called different collectives. The rank ceiling is 64.

`nvidia/tests/e2e/run_nccl_multiproc.sh` runs 4 genuinely separate processes;
`nvidia/tests/e2e/run_nccl_group.sh` runs the single-process grouped form over 4
virtual devices.

Beyond the collectives:

- **`ncclCommSplit`** is a collective on the parent: every rank publishes its
  color and key, and the members of each color agree on a rendezvous name for
  the child derived from the parent's. Ranks are ordered by key, ties by old
  rank; `NCCL_SPLIT_NOCOLOR` gets a NULL communicator; a NULL config inherits
  the parent's. **`ncclCommShrink`** is called only by the surviving ranks,
  who already agree on who survives, so it needs no exchange.
  **`ncclCommInitRankScalable`** takes the same ids on every rank and joins one
  rendezvous named from all of them.
- **Non-blocking communicators** (`ncclConfig_t.blocking = 0`, or
  `NCCL_COMM_BLOCKING=0`) run their work on a background thread: init,
  collectives, `ncclGroupEnd` and `ncclCommFinalize` return `ncclInProgress`,
  and `ncclCommGetAsyncError` reports `ncclInProgress` until the work is done.
  A split of a non-blocking parent returns `ncclSuccess` and fills in the new
  communicator when the parent settles. As on NCCL, a call on a communicator
  whose previous operation has not finished is `ncclInvalidArgument`.
- **`ncclCommAbort`** also marks the rendezvous, so a peer waiting on the
  aborted rank gives up with `ncclRemoteError` instead of waiting out
  `VGPU_NCCL_TIMEOUT`.
- **Pre-multiplied sums** (`ncclRedOpCreatePreMulSum`): each rank's input is
  scaled by its own scalar, read at creation for `ncclScalarHostImmediate` and
  when the collective runs for `ncclScalarDevice`. fp32 and fp64 accumulate as
  a chain of fused multiply-adds in rank order, which is what NCCL computes on
  every fp64 element measured. NCCL's fp32 order follows the ring's chunks,
  which start the chain at different ranks, so an element can differ by an ulp
  (one of eight in the measured case). fp16 rounds each product first, as NCCL
  does.

- **`ncclCommGetUniqueId` and `ncclCommGrow`.** An id's first 16 bytes name the
  rendezvous the old and the new ranks meet at, derived from the parent's
  rendezvous and the number of grows it has seen; the rest is a nonce, so every
  call returns a different id, as NCCL's does. Because every existing rank
  derives the same name, the non-root form (`uniqueId = NULL`) needs no id
  passed to it. Existing ranks keep their numbers, a new rank brings its own,
  and a rank number or id used twice is `ncclInvalidArgument` rather than a
  hang. Where NCCL fails a new rank that arrives before the existing ranks have
  started (`ncclInternalError` after a fraction of a second), this waits; that
  is the one deliberate difference.
- **`ncclCommRevoke`** is local, and unblocks whatever the revoked
  communicator is waiting for: a rank alone in an all-reduce is released, and
  the call that was waiting returns `ncclSuccess`, as the stream completes on
  NCCL. After it every collective and send/receive is `ncclInvalidUsage`, a
  second revoke and `ncclCommFinalize` are `ncclInvalidArgument`, and split,
  shrink, suspend and destroy still work. A non-blocking communicator answers
  `ncclInProgress` and settles once its queued work has unwound.
- **`ncclCommSuspend` and `ncclCommResume`** are collective barriers (one rank
  waits exactly as long as another is late), valid in a group and on a
  non-blocking communicator. Only bit 0 (`NCCL_SUSPEND_MEM`) suspends; suspending
  twice or resuming what is running is `ncclInvalidUsage`. On the card, work
  issued to a suspended communicator faults on the buffers it released; here it
  is refused with `ncclInvalidUsage`. **`ncclCommMemStats`**: NCCL reports
  GPU memory it holds for the communicator (12 MiB it can release, 4 MiB it
  cannot, for a two-rank one), and suspending frees exactly the releasable part.
  This transport allocates no device memory, so the three sizes are zero and
  only the "suspended" statistic is live.
- **`nccl*Config` collectives** check their `ncclCollConfig_t` as NCCL 2.31.2
  does and then run the plain collective: size at least 64, magic, a
  `forceAlgSelection` of 0 or 1, a `CTAPolicy` unset or 0..3, in that order and
  before the communicator is looked at; a refused config does not spoil the
  group it was issued in. `algSelection` is a comma-separated list of algorithm
  names (`ring`, `tree`, `collnetdirect`, `collnetchain`, `nvls`, `nvlstree`,
  `pat`; any case; `^` in front means every other one). With `forceAlgSelection`
  left at 1, an unknown name or a selection that leaves nothing available is
  `ncclInvalidArgument`: AllReduce has ring and tree, Broadcast, Reduce,
  AllGather and ReduceScatter ring, and AlltoAll, Gather and Scatter none --
  the same sets the PCIe card has; NVLS, CollNet and PAT are not available.
  With 0 it falls back to automatic selection. CTA counts, cluster size,
  profiler tags and extension lists are accepted whatever they hold, as NCCL
  accepts them.
- **`ncclParam*`** reproduces NCCL 2.31.2's parameter registry as measured:
  six public parameters (`NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, `NCCL_DEBUG_FILE`,
  `NCCL_DEBUG_TIMESTAMP_FORMAT`, `NCCL_DEBUG_TIMESTAMP_LEVELS`,
  `NCCL_SET_THREAD_NAME`) and three private ones listed with
  `NCCL_PARAM_DUMP_ALL=true`. Types, defaults, documentation text and the way
  each environment variable is parsed are the card's, so are the getters (a
  getter takes the parameter's own type and is `ncclInvalidArgument` for any
  other), and `ncclParamDumpAll` writes the same text to stdout. The rest of
  NCCL's environment variables are not in the registry on the card either.
- **The device API's host side** answers as the RTX 3060 pair does:
  `ncclCommQueryProperties` reports `deviceApiSupport = false` and
  `hostRmaSupport = false` (no multimem, no GIN, one LSA team), and
  `ncclDevCommCreate` is `ncclInvalidUsage`. The team queries
  (`ncclTeamWorld`, `ncclTeamLsa`, `ncclTeamRail`, `ncclTeamRankToWorld`) and
  the requirement helpers (`ncclLsaBarrierCreateRequirement`,
  `ncclLLA2ACreateRequirement`, `ncclLLA2ACalcSlots`,
  `ncclGinBarrierCreateRequirement`) are plain host arithmetic, reproduced from
  the card's outputs.

Error codes and edge cases where the documentation is silent were measured on
NCCL 2.29.7 and 2.31.2 with two RTX 3060s, and
`nvidia/tests/e2e/nccl_comm_ops.cu` and `nccl_multiproc.cu` pass against both
libraries (sections for a newer API than the library in use say "skipped"). Four
things are deliberately not there:

- **Symmetric memory windows.** `ncclCommWindowRegister` returns `ncclSuccess`
  and a NULL window, which is NCCL's own answer on a machine without the peer
  mappings windows are built on (the RTX 3060 pair gives exactly that), and a
  collective on the buffer works as it always does. A real window promises
  device-side loads and stores into the peers' memory -- the device API's LSA
  pointers -- and another process's simulated device is reachable only
  through a file.
- **The device API.** Kernels that use a device communicator (`ncclDevComm`)
  reach peers through LSA pointers, multimem addresses or GIN, none of which
  can exist between simulated devices that share no address space; and the
  `libnccl_device` bitcode that NCCL ships for linking such kernels has nothing
  to link against. `ncclDevCommCreate` refuses and `ncclCommQueryProperties`
  says why, which is the RTX 3060 pair's answer too, so a program that checks
  support first takes its fallback.
- **One-sided operations.** `ncclPutSignal`, `ncclSignal` and `ncclWaitSignal`
  are `ncclInvalidArgument` ("host RMA is not supported in this
  communicator"), every time, which is what NCCL 2.31.2 does on the card
  with or without `numRmaCtx` and `numRmaSig` configured. NCCL moves a put
  through a GIN transport or (2.32) a socket proxy that needs GDRCopy; this
  transport would need a progress thread in every process holding a window,
  and there is no card here whose signal, context and ordering behaviour that
  could be measured against.
- **The network plugin interface.** A net plugin is a library NCCL loads to
  drive a NIC (`NCCL_NET_PLUGIN`); there is no network transport here for one
  to replace, so those variables are not read.

## Mixed precision

`cublasGemmEx` takes fp16 or bf16 operands with a float accumulator, which is
what a tensor core does under `CUBLAS_COMPUTE_32F`, and int8 operands with an
int32 accumulator. Narrow operands are widened to float on the way in and
rounded once on the way out; the conversions come from the toolkit's own
host-callable intrinsics rather than hand-written bit twiddling, which is the
same rule this repository applies to vendor struct layouts and for the same
reason. `cublasGemmStridedBatchedEx` and `cublasHgemm` go through the same
path, and all of it is bit-identical to hardware on operands that the narrow
formats represent exactly.

## Two NPP entry points that do not match, and why they say so

Everything in `npp_api.cpp`'s part of NPP is bit-identical to hardware except
two, and both are excluded from the conformance comparison rather than quietly
claimed (the image-processing half has its own list, below):

- **`nppiFilter_32f_C1R`** (general convolution). Probing NVIDIA's
  implementation with delta kernels gives a mask-to-source mapping that aliases
  positions -- kernel elements 1 and 2 read the same source pixel, as do 4, 5, 7
  and 8 -- which is neither a convolution nor a correlation and is not what the
  documentation describes. VirtualGPU implements the documented convolution.
- **`nppiResize` with `NPPI_INTER_LINEAR`**. Fitting the hardware output pixel
  by pixel shows the horizontal axis interpolating at pixel centres while the
  vertical axis samples rows exactly, with no blending at all. Nearest-neighbour
  matches and is compared; bilinear here is the standard filter.

Both are usable and both are documented as approximations. Getting an answer
that is *close* to NVIDIA's is not the same as getting NVIDIA's, and this file
is where the difference is written down.

## NPP: what real programs call

NPP has some ten thousand entry points; which of them matter was settled by
reading the programs that use it -- OpenCV's cudaarithm, cudaimgproc,
cudawarping and cudafilters (and its core), DALI, FFmpeg's `scale_npp`,
jetson-utils, torchvision (which calls none) and the CUDA Samples -- and
collecting every `npp*` name they call: 253 functions, all implemented:

| user | calls | here |
| --- | --- | --- |
| OpenCV | 194 | all: warps (affine, perspective, both directions, every depth and channel count), rotation, mirroring in place and not, the logical and shift operators with constants, magnitude, alpha compositing and premultiplication, gamma, channel swaps, masked and float mean/standard deviation, even and ranged histograms with their level and buffer helpers, rectangle standard deviation, windowed sums, box, max and min filters, dilation and erosion with masks, float thresholds, transpose |
| DALI | 12 | all: `nppiRemap` at every depth it uses, `nppiCFAToRGB` 8- and 16-bit |
| FFmpeg | 3 | all: `nppiResizeSqrPixel_8u_C1R` (nearest, linear, cubic), the YCbCr 4:2:0 plane layouts |
| CUDA Samples | 51 | all, `watershedSegmentationNPP` included (below): Canny, Prewitt gradient vectors, `nppiLUT_Linear`, `nppiCompareC`, border-replicating box filter, constant-border copy, every allocator, watershed segmentation and marker-label compression |
| jetson-utils | 1 | `nppiCFAToRGB_8u_C1C3R` |

Watershed segmentation (`nppiSegmentWatershed_8u_C1IR`, `_16u_`), its
buffer-size queries and `nppiCompressMarkerLabelsUF_32u_C1IR` with its own
(`watershedSegmentationNPP`) were the last of the 253. Beyond the list,
nothing else of NPP's is implemented -- `nppiLabelMarkersUF` and the
compressed-marker-label info and contour functions among it, which the same
sample does not call.

NVIDIA publishes no algorithm for the watershed, so it is reproduced from
what NPP 13.0 does on an RTX 3060, probed with random, exhaustive and
photographic inputs (`nvidia/src/npp_core.hpp` has the rules):

- **The segmented image is exact** under 8-way connectivity (`nppiNormInf`):
  a pixel flows to its lowest strictly lower neighbour (the first of equal
  ones in raster order) and takes the value where the flow ends. This held on
  every probe, 300 images full of equal values included, and on the teapot,
  skull and rocks images of the CUDA Samples. Boundary types are drawn on the
  result exactly: a pixel whose upper or left neighbour has another value is
  black, white, black-or-white by half the range (`CONTRAST`), or -- in
  `ONLY` -- black on white.
- **The marker labels are the pixel index above and to the left of the
  region's lowest pixel** (the smallest index in its neighbourhood), exact on
  every image whose values are all different. Where neighbouring values are
  equal NPP's labels follow plateau rules nobody wrote down; the simulator
  fits them (equal roots share a label, tied lowest neighbours link roots),
  and on the three Samples images 96.2% (teapot), 98.5% (skull) and 99.5%
  (rocks) of the labels are NVIDIA's. A label that differs moves every rank
  after it in `nppiCompressMarkerLabelsUF`, so the compressed labels of
  those images differ over most of the image though the regions agree.
- **4-way connectivity** (`nppiNormL1`) matches on images up to 4x4. On larger
  ones NPP leaves pixels near the right and bottom edges unwritten, in a
  pattern that depends on the image width and none of the shapes tried fits
  (it is under 0.5% of a 512x512 image for the skull and teapot, a few tenths
  of a percent for rocks); the simulator writes them.
- **Why the equal-value labels are not exact: NPP merges in 16-pixel tiles and does not finish.** Round 4
  probed it with about 15,000 one-row images of 3 values, flat images of every width and 4-way images of both
  shapes (`tools/probes/npp_watershed_probe.cu` prints what the card writes; compare with the model in
  `npp_core.hpp`). What was found (round 5 solved the 4-way item below and added the findings marked R5); the
  equal-value labels are still not reproduced:
  - the rules of `npp_core.hpp` are exact on every row of up to 16 pixels (every row of 2 to 8 pixels of 3
    values was compared) except those that start with two equal pixels and a different third (`a a b ...`):
    then the region holding pixel 0 is labelled 1, not 0, and a pixel 0 that was a separate root joins its
    neighbour's region. (`a a a b`, `a b ...` and a pair anywhere else, a tile start at x = 16 included, are
    as the model has them);
  - the pull rule (a pixel whose two lowest neighbours are tied makes a root take the other one's pixel index as a
    label) holds inside a tile; across a tile border (the pixel at x = 15 with a non-root at 14 and a root at 16)
    the root keeps the plain label (the pixel index above and to its left), 15 instead of the model's 14;
  - **a plateau wider than a tile is not always merged into one label.** A flat row of width w (any value) comes
    back as one label 0 up to a position and then as strips: for every w up to 600 the strips start at the
    positions x = c + 16 k that are at least x0 and below w, where (c, x0) is (0, 48) for w mod 16 in 0..4,
    (5, 37) for 5..9, (6, 70) for 10 and (11, 59) for 11..15, and the strips are labelled 14, 30, 46, ... in order
    (16 i + 14, wherever they start). Taller flat images split into column and row strips the same way, but at
    other positions (a 256 x 100 one at x = 221, 237, 253: the positions depend on the height too), and on the
    teapot the black background is one label up to x = 495 and another (14) from x = 496 on, in every row of the
    top 69. The labels are merged through some number of passes that leaves a width-dependent remainder;
  - R5, what the strips look like (measured on the card, none reproduced):
    * a row of 2 or more pixels is exact in the model except the "a a b" rule above, which holds for every row up to
      36 pixels and every leading run of exactly two equal pixels (the first region is labelled 1 instead of 0):
      600 random 3-valued rows of 2 to 36 pixels match the model with that one rule;
    * a flat image's labels depend on x alone (every row is the same) when it is tall enough, and on y alone when
      it is narrow; a column (w = 1) behaves like a row. The first strip starts at x0(w % 16, h): 48 for w % 16 in
      0..4 and h <= 16, 37 for 5..9, 70 for 10, 59 for 11..15, and with more rows it grows: for w % 16 = 0, x0 = 60,
      61, ... 64 for h = 17..21, 76 for 22, ... 78 for 27..37, 112 for 38, 124 ... 128 for 39..46, and so on
      (roughly 2 h + 16: the reach in x is about 2 n + 1 tiles for n tiles of 16 rows, and the vertical merge is
      complete from a few tile rows up). The h at which x0 steps are the same for every w % 16 while the size of the
      steps (1 or 12) is not, and rows below row 0 change it (raising the bottom rows to a ramp moves x0 as if h had
      changed), so the labels of row 0 depend on the whole image;
    * a bump (one pixel higher than a flat zero row) at p changes only the pixels x >= p + 34 up to the end of the
      16-pixel tile that holds p + 34, which take the label p + 1: the union across the bump reaches two tiles and
      two pixels to the right and no further;
    * the unwritten pixels of 4-way mode (below) repeat every 112 in the width, the lcm of 14 (a tile of 16 with a
      pixel of halo each side) and 16, while the strips repeat every 16: the merge passes that finish the labels are
      not those that finish the flow;
  - **4-way connectivity** (solved in round 5, for the segmented image and for the labels of images with no equal
    neighbours): NPP 13.0 leaves some pixels unwritten. A pixel whose lowest neighbour is the one to its right (below)
    keeps its own value, as if it were a root, when its column (row) is in a set that depends on the width (height)
    alone: with t = width - 1 - x, the set is a function of the width % 112 (kUnwritten4 in `npp_core.hpp`, one
    bit per t from 1 to 11; it repeats from a width of 12, and below that it is the same set cut at t <= width - 1;
    the same for rows), measured on east-flowing ramps of every width from 2 to 1099 and checked on random images
    of 5 to 300 pixels a side, with and without equal values, 8-bit and 16-bit (no image differs). Such a pixel's
    label is its own index; a root's is the smallest index of its closed 4-neighbourhood. The pixels that flow to an
    unwritten one end there, so the segmented values follow. NPP also keeps state between calls: after a 16-bit run of
    100 x 90 the next large 8-bit image of another type can differ in a pixel or two (found as one line of the
    conformance program, `npp_segment`, that moved when the work buffer was zeroed and still differed; the
    simulator is the state-free behaviour);
  - **Round 6: the equal-neighbour labels were tried again and are NOT reproducible from outside** (exhaustive
    small images, `tools/probes/npp_watershed_ties.py`). The card is deterministic (the same batch twice, and in
    reverse order, gives the same bytes), so this is a rule nobody has, not noise. What the 2-valued images of
    every pattern up to 4 x 3 show:
    * the simulator's labels differ from the card's on 44 of the 512 3x3 images (8-way), 96 of 512 (4-way),
      466 of 4096 4x3 images (8-way) and 927 (4-way); on random 3-valued images the labels differ in most images
      from 12x12 up;
    * the "pull" rule (a pixel whose lowest neighbours are tied merges the later roots among them into the first
      one's region) is right in a row (`0 1 0 1 0` is one region, label 0) and wrong in a plane: in
      `1 1 0 0 / 0 1 0 0` the pixel at x = 1 ties the roots 2, 4 and 6, yet the card keeps root 4 (label 0)
      apart from the plateau of 2, 3, 6, 7 (label 1). Switching the rule off or restricting it to opposite
      neighbours is worse (3x3, 8-way: 44 images of 512 differ with the rule, 69 with it restricted to opposite
      neighbours, 143 without it), so none of these is right;
    * pixel 0 behaves differently from every other pixel and not locally: `1 1 0 / 1 1 0 / 0 0 0` labels the
      whole image 1, including the root at pixel 0 whose own closed neighbourhood starts at 0, while the same
      two columns over two rows (`1 1 0 / 1 1 0`) keep pixel 0 and pixel 3 at label 0; `0 0 1 / 0 0 0 / 1 0 0` is
      all 1 and `0 0 1 / 0 0 0 / 0 0 1` is all 0 -- which pixel of the bottom row is 1 decides whether the
      plateau holding pixel 0 is labelled 0 or 1;
    * flat images split into strips whose start depends on the width mod 16 and, in steps that are not
      periodic (48; 60-64; 76-78; 112; 124-128; 140 ... for w = 320 and h = 16, 17-21, 22-37, 38, 39-46, 58),
      on the height.
    These are the signature of a parallel union that runs a fixed schedule of tile passes and does not converge,
    with a pass count that depends on the whole image. A rule would have to be the schedule itself, which
    cannot be read off black-box output; the simulator keeps the fitted model, and the conformance program keeps
    pinning only what is exact (the segmented image, labels of images without equal neighbours).
  The segmented image is exact for both connectivities regardless; the compressed labels of the Samples' images
  differ because a label that differs moves every rank after it.
- Label compression is exact for any label image: labels below
  `nStartingNumber` take their rank among the labels present plus 0, which is
  always counted, so 0 stays 0, an image with no 0 starts at 1 and
  `*pNewNumber` counts 0; larger labels are left alone.
- Statuses and buffer sizes (72 bytes plus, per row, 24 or 32 bytes a pixel
  rounded up to 128; the compression's 65,556 plus 8 a label) are NVIDIA's.

The conventions NPP leaves unwritten were measured on an RTX 3060 against
NPP 13.0 and are recorded at each function in `npp_core.hpp` and
`npp_imgproc.cpp`; `npp_imgproc` (above) pins them. The few places that are
not exact, and are marked so in that test rather than claimed:

- **Cubic sampling** is four-point Lagrange interpolation in single
  precision (an impulse a quarter pixel away gives -0.0547, 0.8203, 0.2734,
  -0.0391). 8-bit and float results match; a 16-bit image has about one pixel
  in a thousand a count apart, a 32-bit integer one a float ulp apart on about
  one in ten.
- **`nppiResizeSqrPixel`** cubic uses four Lagrange weights rounded the way
  NPP's are, and the order of fused multiply-adds NPP's float output shows:
  bit-identical to NPP 13.0 on the resize cases `npp_imgproc` pins. Lanczos
  (windowed sinc, three lobes, not widened when shrinking) matches the card
  at the factors tried to a float rounding and differs by up to 2e-3
  relative between them, where NPP's sinc is tabulated or approximated;
  super-sampling (area average, both factors below one, else
  `NPP_RESIZE_FACTOR_ERROR`) is compared as approximate. The warps'
  super-sampling is still `NPP_INTERPOLATION_ERROR`.
- **`nppiAlphaComp_8u_AC4R`**: every operator's alpha and every colour is
  exact except the non-premultiplied ATOP and XOR colours, within one count.
- **`nppiFilterCannyBorder`**: on NPP 13.0 the high threshold changes nothing
  (an isolated step of magnitude 40 is an edge at thresholds 30 and 32767
  alike); VirtualGPU does the same. With the Sobel kernel, the CUDA Samples'
  parameters and moderate thresholds match pixel for pixel; a threshold down
  in the noise leaves about ten of 1,500 pixels decided differently, and the
  Scharr kernel some 25 -- its gradients are not the plain 3-10-3 ones.
- **`nppiHistogramEven`**: NVIDIA's writes one or more entries past the
  nLevels - 1 the documentation sizes the histogram for; VirtualGPU writes the
  documented ones.
- **`nppiDilate` and `nppiErode` with an off-centre anchor**: where the mask
  reaches past the ROI on the side away from the anchor, NVIDIA's reads the
  ROI's own edge pixels instead of the image beyond; VirtualGPU reads the
  image, as the documentation describes, so those edge pixels can differ.
  Centred anchors (the usual 3x3 with anchor 1,1) match.
- `nppiRemap_16s` linear, and `nppiCFAToRGB` on a tie in its green
  direction, can be a count apart on a pixel.

## NVRTC: the compiler is the compiler

NVRTC turns a string of CUDA C++ into PTX, a cubin, or LTO-IR at run time.
That is a C++ compiler with a PTX assembler inside, and there is no honest way
to fake either -- so this shim does not try. The toolkit's own libnvrtc is a
host library that needs no GPU, and where it is installed every call goes to
it: what the shim presents is NVIDIA's NVRTC, loaded beside the simulator.
That covers all of its outputs and options -- the PTX, the cubin ptxas makes
for an `sm_` target, LTO-IR (`-dlto`) and OptiX-IR (`--optix-ir`), precompiled
headers (`--pch`, `--create-pch`, `--use-pch`, the heap-size calls and the
creation status), the CSV time table (`--time`), the flow callback, the
supported architectures, the log -- and its results and error codes are the
ones an RTX 3060's NVRTC gives (measured: `e2e_nvrtc_paths` passes unchanged
against NVIDIA's libnvrtc 13.0 there and against this).

Without the toolkit's libnvrtc the program is written out and compiled by
`nvcc --ptx` (and `nvcc --cubin`, for the cubin of an `sm_` target when asked
for), and returns the PTX, the cubin and the compiler's diagnostics. LTO-IR,
OptiX-IR and Tile IR are NVVM bitcode and vendor formats that only NVIDIA's
NVRTC writes: `-dlto` and `--optix-ir` are refused as options, naming the
reason, and the outputs' sizes are 0. So are the precompiled-header and
time-trace options.

**The cubin is the real one.** An `sm_` target's cubin is an ELF image of
SASS, which the SASS engine runs (nvidia/docs/sass.md). Where that engine
cannot yet run an instruction of it, the driver shim runs the PTX of the same
compile instead -- the shim notes it aside when it hands the cubin out,
`vgpu_nvrtc_ptx_for_cubin` -- as it does for a fatbin that has both. The PTX
engine (`VGPU_SASS=0`, or `VGPU_NVRTC_CUBIN=ptx`) has no use for SASS: it
compiles for the matching `compute_` architecture and the "cubin" is the PTX
(NUL included), which its driver loads from wherever a cubin goes; that is
what PyTorch's jiterator ran on before the SASS engine existed.

The toolkit's NVRTC differs from nvcc where it matters: nvcc pre-includes
`cuda_runtime.h`, and with it the host's `<stdint.h>` and `<math.h>`, so a
program that defines `int64_t` or `INFINITY` itself -- PyTorch's jiterator
does both -- compiles under NVRTC and fails under nvcc. `VGPU_NVRTC=nvcc`
forces the nvcc path; `VGPU_NVRTC_LIB` names the library to use.

That closes the loop for the JIT frameworks -- CuPy, Numba, Triton and
PyTorch's inductor all compile through NVRTC and then load the result through
the driver API, which VirtualGPU already runs.

`nvrtcAddNameExpression` / `nvrtcGetLoweredName` are NVRTC's own with the
toolkit's library; on the nvcc path they work by the route the real
implementation uses: a device variable is initialised with the expression's
address, and the mangled symbol is read back out of the generated PTX. That is
what turned up two gaps in the PTX parser -- a forward-declared `.entry`
prototype, and a global initialised with another symbol's address -- both of
which appear in ordinary nvcc output and are now handled.

## nvJitLink and nvFatbin: PTX links to PTX, SASS to SASS

nvJitLink is the device linker as a library. NVIDIA's compiles every input to
SASS and links a cubin. VirtualGPU's links what it is given, in kind:

- **PTX**, when every input has PTX: the inputs' PTX becomes one module,
  handed out as the "cubin" -- the choice NVRTC's shim makes for
  `nvrtcGetCUBIN` under the PTX engine, for the same reason: whatever the
  caller does with a cubin (`cuModuleLoadData`, `cuLibraryLoadData`, write it
  to a file for `cuModuleLoad`, wrap it with nvFatbin) it can do with PTX here.
  `nvJitLinkGetLinkedPtx` returns the same module, without the `-lto -ptx`
  NVIDIA's asks for.
- **SASS**, when every input has SASS for `-arch` and some input has no PTX
  (`-rdc`/`-dc` cubins, or fatbins, objects and libraries built for SASS
  only): a real linked cubin (`nvidia/src/sass_link.cpp`), which the driver
  runs as SASS. `nvJitLinkGetLinkedPtx` then returns
  `NVJITLINK_ERROR_INVALID_INPUT`, as NVIDIA's does.
- **The toolkit's linker**, for what only a compiler can link: an LTO-IR input
  (NVVM bitcode: `nvcc -dlto`, NVRTC's `-dlto`), an index file, and SASS beside
  PTX that has no SASS. Where the toolkit's own libnvJitLink is installed
  (a host library; it needs no GPU) the inputs go to it as given and its
  result comes back -- a cubin, or the PTX of `-lto -ptx`, with its logs and
  its result codes. Without it these are refused by name, as before.
  `VGPU_NVJITLINK=own` keeps the hand-off off; `VGPU_NVJITLINK_LIB` names the
  library. Everything this library links itself stays its own.

The rules are a linker's, measured against NVIDIA's on an RTX 3060: a
symbol with external linkage has one definition, a strong one beating a weak
one; a second strong definition is named in the error log and dropped, the
link still succeeding; an undefined reference fails the link with
`NVJITLINK_ERROR_INTERNAL` and its name (the functions the driver supplies --
`vprintf`, `malloc`, the device runtime's -- excepted); each module's
file-scope names stay its own. A linked PTX module declares everything
before its first use, the way ptxas insists, so NVIDIA's driver JITs it as
readily as VirtualGPU runs it.

The SASS linker was written from `cuobjdump -elf` listings -- section,
symbol and relocation tables and `.nv.info` attributes -- of the relocatable
cubins CUDA 12.0's and 13.0's nvcc write for sm_75 to sm_120 and of what
NVIDIA's libnvJitLink 13.0 links them into, compared field by field at every
relocation; no NVIDIA binary was disassembled. It merges module data and
constant banks, lays out shared memory the way NVIDIA's link does (two
functions' `__shared__` variables share an offset unless some kernel reaches
both; a kernel's own come after; `extern __shared__` starts at the static end
rounded to 16; from sm_90 the driver's reserved 1 KiB is counted in the
kernel's section), applies the relocations the layout fixes (constant-bank
offsets in six instruction encodings, shared offsets), keeps those that need
a load address for the loader, and writes the attributes, call graph,
prototypes and relocation descriptors NVIDIA's driver reads. For sm_75 to
sm_90 its output matches NVIDIA's byte for byte in every code section and
relocation table, and NVIDIA's driver on the card runs it. (One thing NVIDIA's
link does that this one does not, for sm_100 and sm_120: re-finalize code from
the "mercury" sections ptxas leaves beside it; see below.)

Inputs are PTX, a fatbin's PTX or relocatable SASS (the image the driver would
pick for `-arch`), the device code nvcc puts in a host object's `.nv_fatbin`
and `__nv_relfatbin` sections and in a static library's members, relocatable
cubins, and VirtualGPU's own PTX cubins. A linked cubin (`nvcc -cubin`) is
accepted and adds nothing, which is how NVIDIA's treats one; a cubin for an
architecture `-arch` cannot run is refused when it is added. SASS beside PTX
with no SASS is refused by name: NVIDIA's compiles the PTX first, and there
is no compiler here. So is LTO-IR, NVVM bitcode that only NVIDIA's compiler
reads.

A SASS link drops functions nothing reaches, as NVIDIA's does: a function
called by no kernel (a chain of them too) is left out; one whose address a
variable the image keeps holds stays; variables are never dropped. `-g` keeps
every function. `-kernels-used=<pattern>` keeps the kernels whose mangled
names contain the pattern (`*` for any run of characters: `kk` keeps `kk` and
`kk2`, `lib_k*l` keeps `lib_kernel`), and with them what they reach; given
more than once, a kernel is kept if any pattern matches. All measured against
NVIDIA's link (the `-variables-used` and `-optimize-unused-variables`
options are accepted and, as with NVIDIA's outside LTO, drop nothing).

The modules' debug information comes along: every `.debug_*` and
`.nv_debug_*` section (the line tables, the frame table, the register and type
tables, `-G`'s DWARF) is appended to the one of its name, the entries of
dropped functions included. The relocations in them follow the symbols the
link kept: a dropped function's reference goes, its frame entry's length field
is cleared (relocation type 73), a frame entry's pointer to its CIE is applied
to the merged table, a shared variable's field holds its offset and an
`extern __shared__` array's all ones. From sm_90 every relocation is in a
`.rela` section, and the image keeps `.nv.compat`, whose record carries the
`sm_XYa` flag that makes a cubin run on XY alone.

Not linked: the mercury sections sm_100's cubins carry beside the code, which
the driver re-finalizes (NVIDIA's linked code for sm_100 and sm_120 differs in
scheduling, not in what it computes; the code is taken as ptxas wrote it, and
the format is NVIDIA's and undocumented -- nor does this link remove a
dropped function's entry from the frame table there, as NVIDIA's does, but
clears it); legacy texture and surface references (`.texref` symbols in
relocatable SASS, which the driver patches at load and the SASS loader does not
resolve; reported as undefined); and, in a link of two modules whose kernel
and device functions all use shared memory, NVIDIA's exact shared-memory
offsets -- the layout here never overlaps what a kernel reaches, but it is not
always the same one.

nvFatbin needs no GPU at all, so it is the whole library: it writes the
container NVIDIA's writes -- byte for byte with `-compress=false` (measured:
PTX without its comments, the flag word the options and an `a`/`f` arch suffix
set, an LTO-IR entry's version and identity word, a relocatable entry copied on
as the object stored it) -- and NVIDIA's driver loads it as VirtualGPU's
loaders do. Compression follows NVIDIA's rules: by default PTX, LTO-IR and the
relocatable PTX are compressed with zstd, a cubin only with `-compress-all`,
`-compress-mode=size` or `-g`; `-compress-mode=speed` uses LZ4; `-compress=false`
and `-compress-mode=none` store everything as it is. The compressed bytes are a
valid zstd frame or LZ4 block, not NVIDIA's own (another zstd, another level),
so an entry's size is within a few percent of the card's, and the headers
(flags, sizes before compression) are the same. libzstd is opened when first
needed; without it entries are stored as they are. The options are checked as
NVIDIA's are (`-32` with `-64`, `-cuda` with `-opencl`, or a value in the wrong
case are refused), and the result strings are NVIDIA's. It takes a VirtualGPU
cubin (PTX) as the PTX it is. `nvFatbinAddIndex` refuses every input with
`NVFATBIN_ERROR_INVALID_INDEX`: it takes an index file, a format no NVIDIA tool
documents or writes, and NVIDIA's own library answers that to everything that
can be offered it.

`e2e_nvjitlink_paths`, `e2e_nvjitlink_sass`, `e2e_nvfatbin_paths` and
`e2e_nvrtc_paths` check all of this, and pass unchanged against NVIDIA's
libraries (nvJitLink, nvFatbin and NVRTC 13.0) on an RTX 3060 -- and with
VirtualGPU's libraries in their place on the same card, whose driver then runs
the linked PTX and SASS and loads the compressed fatbins. `e2e_arch_specific_cubins`
checks the `sm_XYa` rule on a B200, B300 and Vera Rubin (derived from the
documentation; the local card has no `a` target).

## What is not implemented

Unimplemented entry points return the library's own "not supported" status
rather than a plausible wrong answer, so a caller's fallback path still works.

- **Every function cuda_runtime_api.h and cuda.h declare is exported**
  (`tests/lint/check_header_exports.py`, test `lint_header_exports`, preprocesses
  both headers with and without `CUDA_API_PER_THREAD_DEFAULT_STREAM` and
  compares them with the built shims' dynamic symbols; against the CUDA 13.0
  headers it finds 0 missing, where the sweep found 64 in the runtime and 200 in
  the driver). A program that binds by name (PyTorch is linked `-z now`) no
  longer fails to load for a missing function. What each one does is in
  `runtime_sweep.inc` and `driver_sweep.inc`, and `e2e_exports_sweep_runtime` /
  `e2e_exports_sweep_driver` call them with argument cases that an RTX 3060
  printed (`exports_sweep_*.rtx3060.expected`, regenerated from the card with
  `run_exports_sweep.sh runtime|driver --card --update`). Where the library
  runs a function the sweep implements it (batched copies, the 3D peer copy,
  `cudaGetFuncBySymbol`, the coredump attributes, `cuDeviceGetDevResource`,
  the `_ptsz`/`_ptds` per-thread-stream names, ...). Where it cannot, the name
  answers the library's own "not supported" status, once, with a note on stderr:
  green contexts and their resources, conditional graph handles, external
  semaphore graph nodes, pools in host memory, and file-descriptor external
  memory. One differs from the card in a way the sweep test leaves out: the
  card answers a valid `cuDeviceGetLuid` where the shim refuses, and the card
  segfaults on a garbage handle, a null stream in some calls and a second
  unregister, which the shim answers with the documented status instead.
  Not covered: the functions only the CUDA 13.2 headers declare (26 in the
  runtime, 24 in the driver), because no 13.2 header is part of the build.
  Not changed: `cudaFuncGetAttributes` reports the opt-in shared-memory limit as
  `maxDynamicSharedSizeBytes` of every kernel, where the card reports 48 KiB
  less the kernel's static shared memory (48896 for 256 bytes) until
  `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` raises it.

- **cuBLAS**: the exported names the header does not declare (`cublas?bdmm`,
  `cublasGet/SetBackdoor`, `cublasGet/SetEnvironmentMode`); a program that
  needs one fails to load with the name. cuBLASXt runs GEMM's tiles across the
  selected devices but every other routine whole on the first (the results are
  the same either way; NVIDIA spreads them over the devices, which nothing a
  program can read shows). GEMM's CPU share is real: a routine set with
  `cublasXtSetCpuRoutine` and a ratio from `cublasXtSetCpuRatio` is called, as
  the card calls it, on the tail of the longer dimension of C, and no other
  routine ever calls one (the card takes a routine and ratio for the complex
  Hermitian routines and ignores them). Fixed-point emulation of double
  precision (`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`, the FP64 emulated math
  mode) works under strategy EAGER for GemmEx, its batched forms, Dgemm and
  Zgemm. With FIXED mantissa control it is NVIDIA's slicing, bit for bit the
  card's on 41,538 outputs at every mantissa bit count from 4 to 64. With
  DYNAMIC control the bit count is this library's rule, not NVIDIA's
  automatic dynamic precision (the card chose 54 to 97 bits for the matrices
  tried and its rule is not public; this one is cautious, 54 bits plus the
  widest spread of a dot product's term exponents): the products come out as
  accurate as the card's, and the number the bit count pointer receives
  differs. PERFORMANT never emulates, as on the RTX 3060 (where double
  precision is 1/64 of single); `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` runs as
  plain single precision, which is what the card does below compute
  capability 10 -- the BF16x9 scheme itself is not implemented (no card here
  runs it), nor is the special-values mask. cuBLASLt has the emulation
  descriptor (`cublasLtEmulationDesc*`, the matmul descriptor's attribute 38)
  with the card's defaults and refusals, and its matmul checks one as the card
  does, but does not emulate with it -- NVIDIA's did not on the RTX 3060. The complex form is four real
  emulated products, derived from the real one, not measured.
- **cuBLASLt**: on Hopper and Blackwell profiles (sm_90 and later) the FP8 kernel
  table, the scale modes each GPU takes, and the auxiliary scale and amax
  are documentation-derived, not card-verified, because no H100 or RTX PRO 6000 was
  available (AWS had no capacity when they were tried): the FP8 table
  below sm_89 is an RTX 3060's (every FP8 or FP4 descriptor refused), on sm_89 an
  L4's (TN only, 16-byte leading dimensions, no FP8 outer-vector or block scales,
  no auxiliary scale or amax), and from sm_90 up the descriptor checks measured on
  those two apply and the rest is permissive. Block-scaled FP8 and FP4 modes
  (`VEC16_UE4M3`, `VEC32_UE8M0`, `VEC128_32F`, `BLK128x128_32F`, the packed
  `VEC128/VEC32_MN_K4_UE8M0` of cuBLAS 13.8, `PER_BATCH_SCALAR_32F`) follow
  NVIDIA's documentation, with one scale tensor per batch laid out back to back
  (an assumption). The backward epilogues match the card, except that GELU and
  its derivative are exact where the card's fp32 tanh is approximate (within
  about 5e-5), which is why the probe checks GELU against the function, not a hash.
- **Stream capture**: a library call made while its stream is captured is
  recorded into the graph and runs at each launch, over what the graph's own
  kernels have written by then, with the descriptors, plans and parameter
  objects it was given as they were at the call (the program may destroy them
  as soon as the capture function returns). Recorded: cuBLAS, cuBLASLt,
  cuSPARSE, cuSPARSELt, cuTENSOR, cuDSS, cuDNN's graph API and its **whole
  classic API** (convolution, activation, pooling, softmax, LRN, divisive
  normalization, tensor arithmetic and transforms, batch normalization and the
  normalization API with their Ex forms, the spatial transformer, CTC, dropout,
  fused ops, RNNs and multi-head attention: `VGPU_DEFER`, which probes the call
  with the caller's arguments so every status comes back at the call, then
  copies the descriptors and host scalars and arrays), **cuFFT** (`cufftExec*`
  with a copy of the plan), **cuSOLVER's dense API** (the routines NVIDIA's
  library captures; `gesvd`, `syevd`, `sygvd`, `sytrf` and the 64-bit
  `Xsyevd`, `Xgesvd`, ... cannot be captured on the card -- they wait for the
  stream -- and answer INTERNAL_ERROR here and invalidate the capture, as do
  `syevj` and the iterative-refinement solvers `DSgesv`, `DSgels`, `IRSXgesv`
  and their kin), **cuRAND** (the capture takes its place
  in the generator's stream, the first launch draws what an eager call would
  have, later launches of a pseudorandom generator draw other numbers, a
  quasirandom one repeats: as the card), **NPP**'s `_Ctx` functions, and
  **nvJPEG**'s decode (parsed at the call, the pixels written at each launch;
  the encoders wait for the stream and fail with EXECUTION_FAILED, the capture
  invalidated, as on the card), and **NCCL**'s collectives and
  point-to-point operations (a host node of the graph that does the operation
  at each launch, taking its place in the communicator's order then; the
  ranks' graphs have to be launched concurrently, a thread or process each,
  because a launch runs its graph in the calling thread; inside a group the
  sends are recorded first). The **driver API**: a copy, fill or host
  function on a capturing stream is a node (`cuMemcpy*Async`,
  `cuMemsetD*Async`, `cuLaunchHostFunc`, stream memory operations, `cuLaunchKernel`),
  `cuMemAllocAsync`/`cuMemFreeAsync` are graph allocation nodes, events and
  `cuStreamWaitEvent` fork and join streams, and the calls CUDA refuses while
  capturing (`cuMemAlloc*`, `cuStreamSynchronize`, `cuCtxSynchronize`,
  `cuMemPrefetchAsync`, `cuMemcpyPeerAsync`, ...) answer
  CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED and invalidate the capture. libcuda also
  has the driver's own capture and graph API now (`cuStreamBeginCapture`,
  `cuGraph*`, user objects, `cuStreamWaitValue*`/`WriteValue*`/`BatchMemOp`; it
  forwards to the runtime's graphs). Kernel, copy, fill and batch memory
  operation nodes of the driver are closures over a copy of the driver's
  parameters; a node a capture made, or a graph the runtime made, gives no
  parameters back through `cuGraph*NodeGetParams`.
  Every program that checks this (`e2e_graph_capture_*`, `e2e_graph_driver`)
  also runs against NVIDIA's libraries on the card
  (`run_graph_capture.sh <name> <libs> --card`) and passes there.
  **cuSOLVER's sparse API** (`cusolverSp`): the low-level Cholesky's
  `csrcholFactor`, `csrcholSolve` and `csrcholDiag`, the low-level QR's
  `csrqrSetup`, `csrqrFactor` and `csrqrSolve`, and `csrqrsvBatched` are
  recorded (the info objects they share are not copied: they have to outlive
  the graph); the buffer-size queries and the `...Host` forms do not touch the
  stream and leave the capture alone; what waits for the device -- `csrlsvqr`,
  `csrlsvchol`, `csreigvsi`, the two `csr*ZeroPivot` and the three analyses --
  answers INTERNAL_ERROR and invalidates the capture, and `csrcholFactor`,
  which allocates scratch memory on every call, answers ALLOC_FAILED in a
  capture in the global mode (all measured on an RTX 3060; `csrqrSetup` and
  `csrqrsvBatched` are refused in the global mode too when they are the first
  call on a handle that needs scratch memory, which is not modelled). The classic cuDNN
  `cudnnFindConvolution*Algorithm` calls time kernels: the `Ex` forms answer
  and invalidate the capture, the others are refused with 4004 in the global
  mode as well, as the card does.
  **Not recorded** (they run when called, so a replay misses them -- or, for
  the ones that wait on the stream, fail the capture): multi-GPU cuFFT
  descriptors and cuFFT callbacks,
  cuTensorNet and cuStateVec (no NVIDIA library here to measure against), and
  `cudnnBackendPopulateCudaGraph`, which answers NOT_SUPPORTED as cuDNN does
  for an engine without native CUDA-graph support. cusolverRf and cusolverMg
  take no stream (they work on the default one, which cannot be captured). Not
  copied for a captured call: a cuFFT plan with LTO callbacks, a cuRAND
  generator, an nvJPEG handle or state, a cuSOLVER workspace -- these have to
  outlive the graph, as NVIDIA's do. The driver API does not refuse module
  loading while capturing (NVIDIA's invalidates the capture) so that lazily
  loaded kernels (Triton's) keep working.
- **cuDNN**: in the graph API, interpolating resampling beyond bilinear
  upsampling by 2 (the one configuration cuDNN has an engine for; nearest
  has none, which cuDNN documents), block-scaled (MXFP8) attention (E8M0
  scales in the F8_128x4 layout), FP8 attention's backward pass and the
  block-scale (de)quantize operations (per-tensor FP8 attention forward works
  from a Hopper profile, **documentation-derived**: no Hopper card was
  available, `e2e_dnn_fp8_attention`, and through cudnn-frontend's `sdpa_fp8`,
  `e2e_dnn_fp8_attention_frontend`; FP8 is a graph tensor type now),
  attention's block masks and cumulative sequence lengths, sinks in the
  backward attention operation, F16x16 and FP8-128x4 reordered tensors (the
  INT8x32 filter reordering works: `e2e_dnn_int8x32`), the gradients'
  division by the number of GPUs of multi-GPU normalization with more than
  two GPUs (assumed; the card has two), the MoE backward (cuDNN's
  engine for it wants Hopper or Blackwell and, documented, cuBLASLt 13.5,
  newer than this stack's), band-matrix and standalone RoPE operations (no
  engine on the RTX 3060, the only GPU measured; whether Hopper and Blackwell
  have one was not checked: the AWS H100 launch for it was denied); in the
  classic API, the fused ops cuDNN's header marks "reserved for future use"
  (`CONV_SCALE_BIAS_ADD_ACTIVATION` and the two undocumented ones), which the
  card refuses too, and multi-head attention's one-to-one query mapping with
  beams (refused by the hardware as well).
- **cuFFT**: legacy callbacks (`cufftXtSetCallback` with a device function
  pointer) through this library: NVIDIA ships them only in its static library,
  its `libcufft.so` answers every legacy callback call with
  `CUFFT_NOT_IMPLEMENTED` (RTX 3060, CUDA 13.0), and so does this one, which
  stands in for `libcufft.so`. The documented route is a program linked with
  `libcufft_static.a` and `libculibos.a` (`-rdc=true`, the callbacks' addresses
  read out of `__device__` variables), which carries NVIDIA's own cuFFT: on the
  card `fft_legacy_callbacks.cu` passes that way (load and store callbacks on
  C2C, R2C, Z2Z and C2R, callerInfo, clearing one). It cannot run on the
  simulator: NVIDIA's static cuFFT carries a CUDA runtime of its own, which asks the driver for
  its internal export table (and, before that, resolves the driver's graph API by name, which
  the driver shim lacks); a simulated driver has neither. The same program linked with the
  shim, as with NVIDIA's `libcufft.so`, sees `CUFFT_NOT_IMPLEMENTED`, which is what
  `e2e_fft_legacy_callbacks` checks on the simulator. Routing a device
  function pointer from the user's module into the shim's own kernels is not
  done: a function pointer is an index into the module that defined it (PTX
  engine) or an address in that module's code (SASS engine), and neither
  engine calls across modules. LTO callbacks given as LTO-IR -- the form
  NVIDIA documents for `cufftXtSetJITCallback`, NVVM bitcode that only NVIDIA's
  compiler reads -- run when the host has the CUDA toolkit: the library
  opens the toolkit's `libnvJitLink` (as the NVRTC shim opens the toolkit's
  `libnvrtc`), links the LTO-IR with the kernels that call it into a cubin for
  the simulated device, and runs that SASS. Without the toolkit, or with the
  PTX-only engine (`VGPU_SASS=0`), such a plan fails as a callback that does
  not link fails on the card (`NVJITLINK_FAILURE`, CUDA 13.2's answer; 13.0's
  is `INTERNAL_ERROR`); PTX images, as text or in a fatbin, work either way.
  Both callbacks of one plan from one image fail on NVIDIA's cuFFT
  (`INTERNAL_ERROR`: every symbol is defined twice), and here the real linker
  says so too. LTO callbacks on multi-GPU plans fail as NVIDIA's do. The
  multi-GPU layouts were measured on two GPUs; with more, they follow the
  documentation (batches and planes dealt out in order, 1‑D strings over the
  GPUs in order), and the 1‑D factor choice past 2^27 points keeps the last
  measured one.
- **cuSPARSE**: the legacy `cusparse<t>csrmv` family (and `csrmm`, `csrsv`,
  the HYB routines): NVIDIA's `libcusparse.so.12` exports none of them and
  CUDA 12 and 13's headers declare none (removed in 12.0), so there is nothing
  to match; a program that calls them was built for a `libcusparse.so.11` or
  older, whose soname this library does not answer to. SDDMM with a conjugate
  transpose: NVIDIA's library refuses it for real data
  (`INVALID_VALUE`, "conjugate transpose is not valid for ... data type") and
  accepts it for complex data without computing anything its documentation
  describes (measured: `B^H` is read as `B`, and `A^H` mixes the entries of
  different rows), so this refuses it (`NOT_SUPPORTED`); `cusparseSpMMOp`,
  deprecated and in preview, whose operators are LTO-IR (NVVM bitcode) that
  VirtualGPU cannot compile and whose calling convention only a sample
  documents -- `_createPlan` answers as NVIDIA's does when nvJitLink refuses
  them (INTERNAL_ERROR) and says why on stderr. `csrcolor` is colored by
  Jones-Plassmann-Luby rounds on a fixed hash: neighbours never share a color,
  rounds stop once `fractionToColor` of the nodes are colored and each
  remaining node gets a color of its own, `ncolors` is the largest color plus
  one, `reordering` lists the nodes by color, and all follow the descriptor's
  index base, each as checked on the card (`e2e_sparse_color_paths`); NVIDIA's
  colors are a randomized multi-hash scheme whose numbers are not public (a
  6-node path gets 0, 8, 4, 8, 0, 10 and 11 colors), so the colors themselves
  differ. Where NVIDIA's 13.0 does something no caller can mean, this does what
  the documentation says instead: `csr2csr_compress`
  keeps |a| > tol as `nnz_compress` counts (NVIDIA's drops negative real
  entries and leaves their slots unwritten), a negative pruning threshold keeps
  every entry (NVIDIA's returns column indices past n), `gpsvInterleavedBatch`
  with an algo other than 0 is NOT_SUPPORTED (the documentation: "only support
  algo = 0 (QR)"; NVIDIA's does nothing and reports success). The solvers
  agree with NVIDIA's to rounding, not bit for bit: they compute in double.
- **cuSPARSELt**: FP4 (E2M1) inputs past the descriptor on any GPU but an RTX PRO 6000 (an L4 and an RTX 3060
  accept the descriptor and refuse the algorithm; the RTX PRO 6000's sparse FP4, 4:8 in pairs, is implemented from
  its card transcript, and its compressed *metadata layout* is not the card's, only the sizes are), FP8 outputs and the
  block scale modes `VEC32_UE4M3` and `VEC64_UE8M0` (on C, D and D's output everywhere; on A and B, the GPUs
  other than an RTX PRO 6000 refuse them), FP8 on Hopper and the data-center Blackwell GPUs
  (documentation-derived: the L4's table is what is implemented), fp16 compute
  (no kernel on sm_86 or sm_89 on NVIDIA's library either), and GELU outside int8
  and FP8-into-bf16 output (refused there too; an RTX PRO 6000 takes FP8 into fp16, bf16 and fp32); see the
  section above for where the compressed layout and the search differ.
- **cuSOLVER**: left eigenvectors from `Xgeev` (NVIDIA's CUDA 13.0 and 13.2
  libraries answer jobvl = VECTOR with INTERNAL_ERROR and document right
  eigenvectors only; this does the same), cusolverSp's `csrlsvlu` on the
  device (NVIDIA exports only `csrlsvluHost`), and cusolverMg grids with more
  than one row of devices (NVIDIA's refuses them too, at
  `cusolverMgCreateDeviceGrid`) are not implemented because NVIDIA's own
  library does not implement them. `csrmetisnd` runs METIS 5.1.0's
  `METIS_NodeND` -- NVIDIA documents it as "a wrapper of METIS_NodeND" linking
  the 64-bit metis-5.1.0 -- on the pattern of A + A^T without its diagonal.
  The METIS source (Apache-2.0, like this repository) is vendored in
  `nvidia/third_party/metis` rather than fetched at build time, so the build
  needs no network: only what `METIS_NodeND` reaches, compiled into
  libcusolver with hidden symbols. Three changes from the distribution, each
  marked in the files and in that directory's README: 64-bit indices, a
  thread-local copy of glibc's `rand()` sequence (the original reseeds the
  host process's generator), and the vertex-compression step sorted by key and
  then vertex, which is what the card's compression does (without it the 
  complete graphs differ). 77 of 78 test matrices give the card's permutation
  exactly (`e2e_solver_metis_paths` compares them with what the card printed,
  `run_metis_card.sh` re-checks the data on a card); the one that differs is a
  graph with no edges, where no run of METIS 5.1.0 here gives the card's
  answer and any permutation is a correct one. Arguments are answered as the
  card answers them (n <= 0 or nnz = 0 is INVALID_VALUE, a negative nnz
  ALLOC_FAILED, a null descriptor MATRIX_TYPE_NOT_SUPPORTED, an option METIS
  rejects INTERNAL_ERROR) except where the card crashes (a NULL p, Fortran
  numbering in the options) or reads past its arrays (offsets that disagree
  with nnz, columns out of range): INVALID_VALUE here. The solvers'
  `reorder = 3` uses the same METIS ordering. Measured differences: when several columns of
  a Cholesky factorization are independent of one another NVIDIA's names a
  different one in `singularity`. `Xgeev` on a complex matrix returns its
  eigenvalues in NVIDIA's order up to n = 74 (both are LAPACK's single-shift
  QR there); past that NVIDIA's switches algorithm, as LAPACK's does, and the
  order can differ. NVIDIA's `sytri` (CUDA 13.0, RTX 3060) returns success and
  leaves A as it was; this one computes the inverse the API documents.
  NVIDIA's batched cusolverRf crashed on every input tried (a cudaFree of an
  invalid pointer in `cusolverRfBatchAnalyze`/`BatchRefactor`, CUDA 13.0 and
  13.2), so the batched forms follow the documentation unmeasured. The
  low-level preview QR factors in place on NVIDIA's, so a second `csrqrFactor`
  without a new setup refactors its own output; here each Factor starts from
  the setup's matrix. NVIDIA's cusolverMg getrf on a submatrix that starts
  below its diagonal block (IA > JA) returns nothing recognisable; this
  returns the submatrix's LU. The iterative refinement solvers can take one
  refinement step more or fewer than NVIDIA's (its GMRES variants and some
  n = 200 systems), and workspace sizes (`_bufferSize`) are this library's
  own; Jacobi sweep counts are this implementation's, and singular vectors
  for repeated singular values can differ by a rotation, as LAPACK's
  documentation allows.
- **cuTENSOR**: block-sparse contractions are created and checked but not
  planned (NOT_SUPPORTED; an RTX 3060 cannot plan them either, so there is no
  card to check a kernel against); just-in-time kernels (the JIT mode is accepted
  and changes nothing, as on NVIDIA's library for the plan and its results;
  what differs there is the kernel cache, which holds compiled kernels after
  a batched contraction and is empty here, and plan creation, which took
  394 ms once); the undocumented exports
  (`cutensorCreateComputeDescriptor`, extraction and insertion, ...), which
  answer NOT_SUPPORTED. A permutation whose input has a mode its output lacks
  is planned by NVIDIA's library and writes zeros on an RTX 3060; here its
  plan is refused (NOT_SUPPORTED).
- **cuTensorNet**: the MPS projection, distributed tensor descriptors
  (`cutensornetCreateDistributedTensorDescriptor`, whose distributed QR and SVD need NCCL,
  cuTensorMp and cuSOLVERMp: NVIDIA's library answers NOT_SUPPORTED on a machine without NCCL and
  so does this one on any, after the argument checks that library makes first) and the
  undocumented exports answer NOT_SUPPORTED. Distributed execution is done for contractions and
  the state API's amplitudes, marginals and expectation values (see below): not done are the
  sampler's exchange of samples (every rank samples alone, with the same seed), the hyper-optimizer
  exchanges of the preparation calls (the search here is deterministic, so the optimizer and the
  state API's prepare calls keep what each rank finds; the optimizer does exchange its plan, as
  NVIDIA's does), and a PMI or NCCL communication library (the interface is the MPI one,
  `cutensornetCommInterface` of `$CUTENSORNET_COMM_LIB`). Gradients of an expectation value are done
  (`cutensornetExpectationComputeWithGradientsBackward`, PyTorch's convention, measured on an RTX
  3060 and checked against finite differences; a distributed handle refuses them, as NVIDIA's
  does at prepare). A state's MPS honours its SVD algorithm; what differs from NVIDIA's there: its
  gesvdj with a loose tolerance gives a less accurate MPS and with few sweeps ends StateCompute
  with INTERNAL_ERROR (the cuSOLVER here is exact and does neither), its gesvdp differs from gesvd's in
  the seventh digit of the norm (here it is exact), and its gesvdr always ends StateCompute with
  INTERNAL_ERROR, as here. Where the card's distributed contraction has a defect it is kept: the
  clearing and the sum cover the first `volume` elements of the output buffer whatever strides it
  has, so an output with padding comes out wrong where the padding is (measured). The profile of an
  RTX 3060 here reports peer access between two cards, which the real cards do not have, so the
  NVSHMEM job that NVIDIA's library refuses on two GeForce GPUs starts here. Network gradients
  (`cutensornetNetworkComputeGradientsBackward`) are done and match NVIDIA's on an RTX 3060:
  a real network's gradient is the adjoint contracted with the other tensors, a complex
  one is the adjoint times the conjugate of the other tensors; they are computed as
  a contraction per gradient rather than as the backward sweep of the forward path, so
  the cache workspace the sizes report is required (no cache memory is
  `INSUFFICIENT_WORKSPACE`) and not used, and a network with a tensor that has a repeated mode,
  or a mode summed over inside it alone, has no gradient here (`NOT_SUPPORTED`).
  The SVD algorithms gesvdj, gesvdp and gesvdr run on cuSOLVER's, with the information NVIDIA's
  reports (gesvdj's residual and sweeps, gesvdp's error in sigma); gesvdr is randomized, so its
  vectors are not NVIDIA's, and, as NVIDIA's, it needs host scratch memory and a rank plus
  oversampling that fits in min(m, n). Half-precision tensors are legal descriptors and every
  decomposition of them is `INVALID_VALUE`, as on the card; so is a decomposition under stream
  capture, `CUDA_ERROR` rather (a QR also invalidates the capture, an SVD leaves it empty).
  State API, measured on the card and followed: an accessor that is given a projected
  value out of range refuses every call after it (a NULL list of values, output or
  workspace does not), and an expectation of an operator made for another number of
  modes is accepted and computes; one in another data type than the state's is accepted
  and then refused when prepared (NOT_SUPPORTED), since NVIDIA's reads the tensors in
  the operator's own type and this library reads them in the state's.
  Distributed execution needs MPI (the interface plugin is an MPI library) and an MPI launcher;
  `mpirun` does not start a job on the host used here, and there is no card behaviour to
  follow, so it is not attempted. Autotuning has nothing to tune and returns at once; the cache
  workspace is never used (beyond the check above); `RUNTIME_EST` is 0, there being no timing
  model. cuQuantum Python's own calls are these (traced on an RTX 3060), but
  cuQuantum Python 26.09 does not start on the simulator yet: the runtime
  module of cuda-bindings 13 that it uses through nvmath-python, and CuPy 14,
  carry a static CUDA runtime that asks the driver for its export table.
- **NCCL**: symmetric memory windows (registration returns a NULL window, as
  NCCL does without peer mappings) and the network plugin interface (there is
  no network to plug into); see "NCCL: a file-backed transport". Every entry
  point NCCL 2.31.2 exports is here and answers as the card does, except
  `ncclSetEncryption` (2.32, TLS for the bootstrap sockets, which this transport
  does not have) and the two GIN requirement helpers that no Linux header
  declares. Refused by name, with the card's code: `ncclDevCommCreate` (device
  API), `ncclPutSignal`, `ncclSignal` and `ncclWaitSignal` (host RMA); a window
  query on any window is `ncclInvalidArgument`, there being none. Not
  reproduced: the sizes `ncclCommMemStats` reports (zero, there is no device
  memory behind a communicator), NCCL's failure of a grow whose new rank
  arrives first, and work issued to a suspended communicator (a fault on the
  card, `ncclInvalidUsage` here). Shrink refuses an excluded rank outside the
  communicator, which NCCL 2.29.7 accepts and miscounts. Windows, the device API
  (`ncclDevCommCreate`, `ncclGetLsaDevicePointer`, `ncclGetPeerDevicePointer`,
  `ncclGetLsaMultimemDevicePointer`), the one-sided calls and the network plugin interface were
  looked at again and left refused, for reasons that a card cannot settle: the RTX 3060 pair
  answers every one of them with a refusal (no peer mappings, no multimem, no GIN), so there is
  no behaviour to match, and writing the answer an NVLink machine would give needs the whole
  device API behind it -- `ncclDevComm` is a struct of ~9,500 lines of header-inline device
  code's expectations (LSA barriers in the resource window, flat peer address ranges, GIN
  contexts) and the `libnccl_device` bitcode that links such kernels is LTO-IR -- none of which
  can be checked here. A hardware check of those wants an NVLink machine (an AWS p4d or p5).
- **cuFile**: the nvidia-fs (DMA) path itself and RDMA -- a user-space file
  system handle registers and its I/O returns 5006, as on the card without
  nvidia-fs, the table never called -- and the POSIX bounce-buffer pool's
  configuration (accepted, nothing to configure).
- **nvCOMP**: Cascaded, Bitcomp and ANS, whose bitstreams NVIDIA does not
  publish (its headers describe the options, not the formats) -- every entry
  point answers `nvcompErrorNotSupported` and says why once; LZ4's bitshuffle
  through the manager classes (the batched API has it); the container's
  checksums (their algorithm is not public: computing them is refused,
  verifying them reports `nvcompErrorCannotVerifyChecksums`);
  `nvcomp::LZ4CPUManager`; streaming gzip decompression (the card refuses it
  too: no hardware decompression engine); and the hardware decompression
  engine itself (the backend option is accepted; everything runs on the host).
- **NVENC**: AV1 (refused at initialisation, as on the card), 10-bit and 4:4:4
  encoding (`NV_ENC_ERR_UNSUPPORTED_PARAM`), motion-only encoding, asynchronous
  mode (refused with the card's message), lookahead, adaptive quantisation,
  weighted prediction, long-term references, intra refresh, temporal layers and
  field/interlaced encoding. H.264: partitions smaller than 8x8, temporal direct
  prediction, B pictures as references (the card pyramids), one reference picture
  per list for B pictures, per-macroblock QP changes. HEVC: temporal motion
  vector prediction, asymmetric partitions, sample adaptive offset, transform skip,
  sign data hiding, scaling lists, several slices, tiles and wavefronts. The bytes are
  not NVIDIA's.
- **NVDEC**: every codec but JPEG (reported unsupported), progressive JPEG
  (the card refuses it too), display-area and target-rectangle cropping,
  deinterlacing, output formats other than NV12 (the card refuses those), video
  sources, and the card's decode of truncated or damaged pictures (the card
  returns success with whatever its engine makes of them; a picture the host
  decoder cannot parse is `CUDA_ERROR_INVALID_IMAGE` here).
- **NVSHMEM**: the PMI-1 and PMI-2 bootstraps through libpmi.so and libpmi2.so (the machine this was
  made on has neither, and there NVIDIA's library makes a job of one PE too; with them present NVIDIA's
  would form a job from Slurm's or MPICH's launcher, which was not checked and is not done here), bootstrap
  plugins other than NVSHMEM's own (`NVSHMEM_BOOTSTRAP=plugin`), the UID bootstrap through sockets
  (`NVSHMEM_BOOTSTRAP=UID`, which needs init flags in NVIDIA's too), PEs on more than one node
  and proxy or network transports, NVLink SHARP multicast (`nvshmemx_mc_ptr` is NULL), the
  device API's own proxy and IBGDA paths (never taken: every PE is a peer), regions
  (`nvshmemx_region_start` and `_stop` answer a refusal), queue pairs, externally mapped
  symmetric buffers. The MPI bootstrap reads the handles of Open MPI and of the MPICH family; the
  OpenSHMEM one needs `shmem_getmem` and symmetric allocation of the application's library. NVSHMEM_MAX_TEAMS
  is read (default 32 here, 256 on NVIDIA's: each team costs symmetric heap). PEs on one GPU report the
  multiple-processes-per-GPU status (3) and nothing else of its limits: the features NVIDIA's disables there
  stay on. An OpenSHMEM job under Open MPI crashes now and then in UCX on the machine this was made on, with
  NVIDIA's NVSHMEM and with a plain `shmem_init` program alike; the test runs such a job again.
- **NVRTC**: without the toolkit's libnvrtc on the machine, LTO-IR, OptiX-IR
  and Tile IR output and the precompiled-header and time-trace options (NVVM
  bitcode and vendor formats only NVIDIA's NVRTC writes: refused by name; the
  nvcc path makes PTX and the cubin). With it, nothing: every call is its own.
- **nvJitLink**: without the toolkit's libnvJitLink on the machine, LTO-IR and
  index-file inputs and so link-time optimisation, and SASS beside PTX that
  has no SASS (refused by name; with it they are linked by NVIDIA's). In a
  SASS link, re-finalizing sm_100 and sm_120 code from the mercury sections
  (undocumented format: the code is taken as ptxas wrote it, and a dropped
  function's frame-table entry is cleared there where NVIDIA's removes it), legacy
  texture and surface references, and NVIDIA's own shared-memory offsets where
  a kernel's and device functions' variables in two modules all compete (this
  layout is valid, not identical). Code-generation options (`-O`,
  `-maxrregcount`, `-Xptxas`, ...) are accepted and have nothing to act on.
- **nvFatbin**: `nvFatbinAddIndex` accepts nothing (NVIDIA's accepts an index
  file, which no tool writes), and the compressed bytes are not NVIDIA's own
  (a different zstd or LZ4, the same flags and uncompressed sizes).
- **NPP**: the functions OpenCV, DALI, FFmpeg, jetson-utils and the CUDA
  Samples call (see "NPP: what real programs call") plus the original subset
  -- allocation, per-pixel arithmetic and logic, data exchange, colour
  conversion, thresholding, statistics, filters, morphology, resizing, and
  the signal-processing equivalents, watershed segmentation and marker-label
  compression (labels inexact where neighbouring values are equal; 4-way
  connectivity inexact near the edges of images wider than 4: see "NPP: what
  real programs call"). Not implemented: `nppiLabelMarkersUF` and the
  compressed-marker-label functions built on it, and the rest of NPP's ten
  thousand entry points, which are absent rather than approximated, so a
  program that needs more fails at link time with a name.
- **CUDA runtime and driver**: `cudaImportExternalMemory` and
  `cudaImportExternalSemaphore` (the handles come from Vulkan, Direct3D or NvSciBuf, which this machine does not
  have; NVIDIA's library crashes on the invalid handles a test could pass, so there is nothing to check
  against); cluster occupancy for a part whose GPC count NVIDIA does not publish
  (nvidia/docs/clusters.md); the memory pool and VMM handle types other than the POSIX file descriptor
  (Win32, fabric), which the RTX 3060 refuses too; and exec-affinity types the device does not offer.
- **Device runtime** (cudadevrt, dynamic parallelism), on both engines: all of
  `cuda_device_runtime_api.h` that CUDA 12 and 13 still offer to a kernel --
  device-side launches (`<<<>>>`, `cudaGetParameterBuffer` /
  `cudaLaunchDevice`, tail and fire-and-forget streams, named streams and
  events), the pending-launch limit, `cudaMemcpyAsync`, `cudaMemcpy2DAsync`,
  `cudaMemcpy3DAsync` and the memset family, `cudaMalloc` and `cudaFree`,
  `cudaFuncGetAttributes`, `cudaDeviceGetAttribute`, `cudaDeviceGetLimit`,
  the cache configuration, the occupancy queries (and
  `cudaOccupancyMaxPotentialBlockSize` from them), `cudaGetErrorString`,
  `cudaGetErrorName`, `cudaRuntimeGetVersion`, the last error and
  `cudaGetDevice`/`cudaGetDeviceCount` -- under CDP2's names, and CDP1's
  (`-DCUDA_FORCE_CDP1_IF_SUPPORTED`, which has `cudaDeviceSynchronize`, for
  parts before Hopper). Each call returns what an RTX 3060 returned
  (`nvidia/docs/sass.md`, "Device runtime"). What remains: a parameter buffer
  launched a second time (the card runs it again; here it is
  `cudaErrorInvalidValue`); `cudaMemcpy3DAsync` between `cudaArray`s
  (`cudaErrorInvalidValue`; the card's answer is unmeasured) and copies of
  shared or local memory; `cudaGraphKernelNode*` updates from a kernel and
  cooperative groups' multi-grid and `cudaCG*` calls into the library
  (refused by name, as is any other `__cuda_syscall_*` the library makes);
  and kernels running concurrently with their parent: children run after it,
  so a kernel that waits (spinning on a flag) for a child it launched, or a
  child that waits for its parent, hangs here where it runs on the card. A
  device-side `cudaMemsetAsync` fills with its value; the card's wrote zeros
  whatever it was.
- **nvJPEG**: decoding 12-bit (or any non-8-bit) samples, arithmetic coding,
  lossless and hierarchical JPEG -- the card refuses all of them on its
  default backend (`NVJPEG_STATUS_JPEG_NOT_SUPPORTED` at decode; arithmetic and
  differential frames already at the parse) and so do we. The card has one
  lossless path, the lossless backend through the batched API with 16-bit
  interleaved output, and on the RTX 3060 it is not usable: for
  predictor-1 streams it returns the right samples for the first ~40 bytes of
  entropy-coded data and wrong ones after (checked against ffmpeg's decode of
  the same files, with fixed-length and optimal tables, 8, 12 and 16 bits),
  zeros for any stream with a restart interval, and
  `NVJPEG_STATUS_JPEG_NOT_SUPPORTED` for predictors 2 to 7 and for 3-component
  files; the simulator refuses lossless the same way (`JPEG_NOT_SUPPORTED`)
  rather than invent that output. The hardware backend and what only it does
  (`nvjpegDecodeBatchedEx`, scaled decodes, applying an EXIF orientation,
  `nvjpegDecodeBatchedParseJpegTables`) is the card's `ARCH_MISMATCH`/refusal
  on the default backend too; `nvjpegEncoderParamsCopyHuffmanTables` is a
  no-op (the 13.0 library does not export it).

Add them the way the PTX subset grew: hit one, implement it, prove it against
hardware.
