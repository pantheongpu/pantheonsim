/*
 * vgpu_nvcomp.h -- VirtualGPU's clean-room declarations of nvCOMP's C API.
 *
 * Written from NVIDIA's publicly documented nvCOMP API
 * (docs.nvidia.com/cuda/nvcomp), version 5.3, so that programs built against
 * nvCOMP's low-level batched interface run on VirtualGPU's libnvcomp. It
 * contains no NVIDIA code. Type layouts and numeric values (status codes,
 * option structs and their defaults, chunk-size limits, alignments) follow the
 * documented ABI, so a program compiled with NVIDIA's nvcomp/<format>.h and
 * one compiled with this header call the same library the same way.
 *
 * Every format declares the same ten entry points, named
 * nvcompBatched<Format>{Compress,Decompress}..., differing only in their
 * option structs; VGPU_NVCOMP_BATCHED_API spells them out once.
 */
#ifndef VGPU_NVCOMP_H_
#define VGPU_NVCOMP_H_

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#include <driver_types.h> /* cudaStream_t */

#ifndef NVCOMP_EXPORT
#define NVCOMP_EXPORT __attribute__((visibility("default")))
#endif

#define NVCOMP_VER_MAJOR 5
#define NVCOMP_VER_MINOR 3
#define NVCOMP_VER_PATCH 0
#define MAKE_SEMANTIC_VERSION(major, minor, patch) ((major * 1000) + (minor * 100) + patch)
#define NVCOMP_VER MAKE_SEMANTIC_VERSION(NVCOMP_VER_MAJOR, NVCOMP_VER_MINOR, NVCOMP_VER_PATCH)

#ifdef __cplusplus
extern "C" {
#endif

/* ---- shared types ---- */
typedef enum nvcompStatus_t {
  nvcompSuccess = 0,
  nvcompErrorInvalidValue = 10,
  nvcompErrorNotSupported = 11,
  nvcompErrorCannotDecompress = 12,
  nvcompErrorBadChecksum = 13,
  nvcompErrorCannotVerifyChecksums = 14,
  nvcompErrorOutputBufferTooSmall = 15,
  nvcompErrorWrongHeaderLength = 16,
  nvcompErrorAlignment = 17,
  nvcompErrorChunkSizeTooLarge = 18,
  nvcompErrorCannotCompress = 19,
  nvcompErrorWrongInputLength = 20,
  nvcompErrorBatchSizeTooLarge = 21,
  nvcompErrorSubChunkCountTooLarge = 22,
  nvcompErrorSubChunkCountTooSmall = 23,
  nvcompErrorOutputBufferAlignmentTooSmall = 24,
  nvcompErrorCudaError = 1000,
  nvcompErrorInternal = 10000
} nvcompStatus_t;

typedef enum nvcompType_t {
  NVCOMP_TYPE_CHAR = 0,
  NVCOMP_TYPE_UCHAR = 1,
  NVCOMP_TYPE_SHORT = 2,
  NVCOMP_TYPE_USHORT = 3,
  NVCOMP_TYPE_INT = 4,
  NVCOMP_TYPE_UINT = 5,
  NVCOMP_TYPE_LONGLONG = 6,
  NVCOMP_TYPE_ULONGLONG = 7,
  NVCOMP_TYPE_FLOAT16 = 9,
  NVCOMP_TYPE_FLOAT8_E4M3 = 10,
  NVCOMP_TYPE_BITS = 0xff
} nvcompType_t;

typedef enum nvcompDecompressBackend_t {
  NVCOMP_DECOMPRESS_BACKEND_DEFAULT = 0,
  NVCOMP_DECOMPRESS_BACKEND_HARDWARE = 1,
  NVCOMP_DECOMPRESS_BACKEND_CUDA = 2
} nvcompDecompressBackend_t;

typedef struct {
  uint32_t version;        /* nvCOMP's, as MAKE_SEMANTIC_VERSION */
  uint32_t cudart_version; /* the CUDA runtime it was built with */
} nvcompProperties_t;

typedef struct {
  size_t input;
  size_t output;
  size_t temp;
} nvcompAlignmentRequirements_t;

typedef enum nvcompBitshuffleMode_t {
  NVCOMP_BITSHUFFLE_NONE = 0,
  NVCOMP_BITSHUFFLE_MSB_FIRST = 1,
  NVCOMP_BITSHUFFLE_LSB_FIRST = 2
} nvcompBitshuffleMode_t;

NVCOMP_EXPORT nvcompStatus_t nvcompGetProperties(nvcompProperties_t* properties);
NVCOMP_EXPORT const char* nvcompGetStatusString(nvcompStatus_t status);

/* ---- per-format options ---- */
typedef struct {
  nvcompType_t data_type;
  nvcompBitshuffleMode_t bitshuffle_mode;
  char reserved[56];
} nvcompBatchedLZ4CompressOpts_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  int sort_before_hw_decompress;
  nvcompType_t data_type;
  nvcompBitshuffleMode_t bitshuffle_mode;
  char reserved[48];
} nvcompBatchedLZ4DecompressOpts_t;
static const nvcompBatchedLZ4CompressOpts_t nvcompBatchedLZ4CompressDefaultOpts = {NVCOMP_TYPE_CHAR,
                                                                                   NVCOMP_BITSHUFFLE_NONE, {0}};
static const nvcompBatchedLZ4DecompressOpts_t nvcompBatchedLZ4DecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, 0, NVCOMP_TYPE_CHAR, NVCOMP_BITSHUFFLE_NONE, {0}};
static const size_t nvcompLZ4CompressionMaxAllowedChunkSize = 1 << 24;
static const size_t nvcompLZ4DecompressionMaxAllowedChunkSize = (1ull << 32) - 1;
static const size_t nvcompLZ4RequiredCompressionAlignment = 4;
static const size_t nvcompLZ4RequiredDecompressionAlignment = 1;

typedef struct {
  char reserved[64];
} nvcompBatchedSnappyCompressOpts_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  int sort_before_hw_decompress;
  char reserved[56];
} nvcompBatchedSnappyDecompressOpts_t;
static const nvcompBatchedSnappyCompressOpts_t nvcompBatchedSnappyCompressDefaultOpts = {{0}};
static const nvcompBatchedSnappyDecompressOpts_t nvcompBatchedSnappyDecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, 0, {0}};
static const size_t nvcompSnappyCompressionMaxAllowedChunkSize = 1 << 24;
static const size_t nvcompSnappyDecompressionMaxAllowedChunkSize = (1ull << 31) - 1;
static const size_t nvcompSnappyRequiredCompressionAlignment = 1;
static const size_t nvcompSnappyRequiredDecompressionAlignment = 1;

/* Deflate, Gdeflate and Gzip compression: algorithm 0 is entropy coding
 * only, 1 (the default) fast, up to 5 for the highest ratio. */
typedef struct {
  int algorithm;
  char reserved[60];
} nvcompBatchedDeflateCompressOpts_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  int sort_before_hw_decompress;
  char reserved[56];
} nvcompBatchedDeflateDecompressOpts_t;
static const nvcompBatchedDeflateCompressOpts_t nvcompBatchedDeflateCompressDefaultOpts = {1, {0}};
static const nvcompBatchedDeflateDecompressOpts_t nvcompBatchedDeflateDecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, 0, {0}};
static const size_t nvcompDeflateCompressionMaxAllowedChunkSize = 1u << 31;
static const size_t nvcompDeflateDecompressionMaxAllowedChunkSize = (1ull << 32) - 1;
static const size_t nvcompDeflateRequiredCompressionAlignment = 8;
static const size_t nvcompDeflateRequiredDecompressionAlignment = 4;

typedef struct {
  int algorithm;
  char reserved[60];
} nvcompBatchedGdeflateCompressOpts_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  char reserved[60];
} nvcompBatchedGdeflateDecompressOpts_t;
static const nvcompBatchedGdeflateCompressOpts_t nvcompBatchedGdeflateCompressDefaultOpts = {1, {0}};
static const nvcompBatchedGdeflateDecompressOpts_t nvcompBatchedGdeflateDecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, {0}};
static const size_t nvcompGdeflateCompressionMaxAllowedChunkSize = 1u << 31;
static const size_t nvcompGdeflateDecompressionMaxAllowedChunkSize = (1ull << 32) + 288;
static const size_t nvcompGdeflateRequiredCompressionAlignment = 8;
static const size_t nvcompGdeflateRequiredDecompressionAlignment = 4;

typedef struct {
  int algorithm;
  char reserved[60];
} nvcompBatchedGzipCompressOpts_t;
typedef enum {
  NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE = 0,
  NVCOMP_GZIP_DECOMPRESS_ALGORITHM_LOOKAHEAD = 1
} nvcompBatchedGzipDecompressAlgorithm_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  nvcompBatchedGzipDecompressAlgorithm_t algorithm;
  int sort_before_hw_decompress;
  char reserved[52];
} nvcompBatchedGzipDecompressOpts_t;
static const nvcompBatchedGzipCompressOpts_t nvcompBatchedGzipCompressDefaultOpts = {1, {0}};
static const nvcompBatchedGzipDecompressOpts_t nvcompBatchedGzipDecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, NVCOMP_GZIP_DECOMPRESS_ALGORITHM_NAIVE, 0, {0}};
static const size_t nvcompGzipCompressionMaxAllowedChunkSize = ((size_t)0x7FFFFFFFULL << 15);
static const size_t nvcompGzipNaiveDecompressionMaxAllowedChunkSize = (1ull << 32) - 1;
static const size_t nvcompGzipLookaheadDecompressionMaxAllowedChunkSize = ~0ull;
static const size_t nvcompGzipRequiredDecompressionAlignment = 8;

typedef struct {
  char reserved[64];
} nvcompBatchedZstdCompressOpts_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  char reserved[60];
} nvcompBatchedZstdDecompressOpts_t;
static const nvcompBatchedZstdCompressOpts_t nvcompBatchedZstdCompressDefaultOpts = {{0}};
static const nvcompBatchedZstdDecompressOpts_t nvcompBatchedZstdDecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, {0}};
static const size_t nvcompZstdCompressionMaxAllowedChunkSize = (1UL << 31) - 1;
static const size_t nvcompZstdDecompressionMaxAllowedChunkSize = (1ull << 31) - 1;
static const size_t nvcompZstdRequiredCompressionAlignment = 4;
static const size_t nvcompZstdRequiredDecompressionAlignment = 8;

/* NVIDIA's own formats. Their bitstreams are not publicly specified, so
 * VirtualGPU's library declares them and refuses them (nvcompErrorNotSupported). */
typedef struct {
  size_t internal_chunk_bytes;
  nvcompType_t type;
  int num_RLEs;
  int num_deltas;
  int use_bp;
  char reserved[40];
} nvcompBatchedCascadedCompressOpts_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  char reserved[60];
} nvcompBatchedCascadedDecompressOpts_t;
static const nvcompBatchedCascadedCompressOpts_t nvcompBatchedCascadedCompressDefaultOpts = {
    4096, NVCOMP_TYPE_INT, 2, 1, 1, {0}};
static const nvcompBatchedCascadedDecompressOpts_t nvcompBatchedCascadedDecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, {0}};

typedef struct {
  int algorithm;
  nvcompType_t data_type;
  char reserved[56];
} nvcompBatchedBitcompCompressOpts_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  char reserved[60];
} nvcompBatchedBitcompDecompressOpts_t;
static const nvcompBatchedBitcompCompressOpts_t nvcompBatchedBitcompCompressDefaultOpts = {0, NVCOMP_TYPE_UCHAR,
                                                                                           {0}};
static const nvcompBatchedBitcompDecompressOpts_t nvcompBatchedBitcompDecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, {0}};

typedef enum nvcompANSType_t { nvcomp_rANS } nvcompANSType_t;
typedef struct {
  nvcompANSType_t type;
  nvcompType_t data_type;
  uint8_t max_sub_chunk_count;
  char reserved[55];
} nvcompBatchedANSCompressOpts_t;
typedef struct {
  nvcompDecompressBackend_t backend;
  nvcompType_t data_type;
  uint8_t max_sub_chunk_count;
  char reserved[55];
} nvcompBatchedANSDecompressOpts_t;
static const nvcompBatchedANSCompressOpts_t nvcompBatchedANSCompressDefaultOpts = {nvcomp_rANS, NVCOMP_TYPE_CHAR,
                                                                                   0, {0}};
static const nvcompBatchedANSDecompressOpts_t nvcompBatchedANSDecompressDefaultOpts = {
    NVCOMP_DECOMPRESS_BACKEND_DEFAULT, NVCOMP_TYPE_CHAR, 0, {0}};

/* ---- the ten batched entry points of each format ---- */
#define VGPU_NVCOMP_BATCHED_API(F)                                                                                  \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##CompressGetRequiredAlignments(                                    \
      nvcompBatched##F##CompressOpts_t compress_opts, nvcompAlignmentRequirements_t* alignment_requirements);     \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##CompressGetTempSizeAsync(                                         \
      size_t num_chunks, size_t max_uncompressed_chunk_bytes, nvcompBatched##F##CompressOpts_t compress_opts,      \
      size_t* temp_bytes, size_t max_total_uncompressed_bytes);                                                    \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##CompressGetTempSizeSync(                                          \
      const void* const* const device_uncompressed_chunk_ptrs, const size_t* const device_uncompressed_chunk_bytes, \
      size_t num_chunks, size_t max_uncompressed_chunk_bytes, nvcompBatched##F##CompressOpts_t compress_opts,      \
      size_t* temp_bytes, size_t max_total_uncompressed_bytes, cudaStream_t stream);                               \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##CompressGetMaxOutputChunkSize(                                    \
      size_t max_uncompressed_chunk_bytes, nvcompBatched##F##CompressOpts_t compress_opts,                         \
      size_t* max_compressed_chunk_bytes);                                                                          \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##CompressAsync(                                                    \
      const void* const* device_uncompressed_chunk_ptrs, const size_t* device_uncompressed_chunk_bytes,            \
      size_t max_uncompressed_chunk_bytes, size_t num_chunks, void* device_temp_ptr, size_t temp_bytes,             \
      void* const* device_compressed_chunk_ptrs, size_t* device_compressed_chunk_bytes,                            \
      nvcompBatched##F##CompressOpts_t compress_opts, nvcompStatus_t* device_statuses, cudaStream_t stream);       \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##DecompressGetRequiredAlignments(                                  \
      nvcompBatched##F##DecompressOpts_t decompress_opts, nvcompAlignmentRequirements_t* alignment_requirements); \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##DecompressGetTempSizeAsync(                                       \
      size_t num_chunks, size_t max_uncompressed_chunk_bytes, nvcompBatched##F##DecompressOpts_t decompress_opts,  \
      size_t* temp_bytes, size_t max_total_uncompressed_bytes);                                                    \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##DecompressGetTempSizeSync(                                        \
      const void* const* const device_compressed_chunk_ptrs, const size_t* const device_compressed_chunk_bytes,    \
      size_t num_chunks, size_t max_uncompressed_chunk_bytes, size_t* temp_bytes,                                  \
      size_t max_total_uncompressed_bytes, nvcompBatched##F##DecompressOpts_t decompress_opts,                     \
      nvcompStatus_t* device_statuses, cudaStream_t stream);                                                       \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##GetDecompressSizeAsync(                                           \
      const void* const* device_compressed_chunk_ptrs, const size_t* device_compressed_chunk_bytes,                \
      size_t* device_uncompressed_chunk_bytes, size_t num_chunks, cudaStream_t stream);                            \
  NVCOMP_EXPORT nvcompStatus_t nvcompBatched##F##DecompressAsync(                                                  \
      const void* const* device_compressed_chunk_ptrs, const size_t* device_compressed_chunk_bytes,                \
      const size_t* device_uncompressed_buffer_bytes, size_t* device_uncompressed_chunk_bytes, size_t num_chunks,  \
      void* const device_temp_ptr, size_t temp_bytes, void* const* device_uncompressed_chunk_ptrs,                 \
      nvcompBatched##F##DecompressOpts_t decompress_opts, nvcompStatus_t* device_statuses, cudaStream_t stream);

VGPU_NVCOMP_BATCHED_API(LZ4)
VGPU_NVCOMP_BATCHED_API(Snappy)
VGPU_NVCOMP_BATCHED_API(Deflate)
VGPU_NVCOMP_BATCHED_API(Gdeflate)
VGPU_NVCOMP_BATCHED_API(Gzip)
VGPU_NVCOMP_BATCHED_API(Zstd)
VGPU_NVCOMP_BATCHED_API(Cascaded)
VGPU_NVCOMP_BATCHED_API(Bitcomp)
VGPU_NVCOMP_BATCHED_API(ANS)

/* ---- CRC32 ---- */
typedef struct {
  uint32_t poly;
  uint32_t init;
  bool ref_in;
  bool ref_out;
  uint32_t xorout;
  char reserved[16];
} nvcompCRC32Spec_t;
static const nvcompCRC32Spec_t nvcompCRC32 = {0x04C11DB7, 0xFFFFFFFF, true, true, 0xFFFFFFFF, {0}};
static const nvcompCRC32Spec_t nvcompCRC32_C = {0x1EDC6F41, 0xFFFFFFFF, true, true, 0xFFFFFFFF, {0}};
static const nvcompCRC32Spec_t nvcompCRC32_BZIP2 = {0x04C11DB7, 0xFFFFFFFF, false, false, 0xFFFFFFFF, {0}};
static const nvcompCRC32Spec_t nvcompCRC32_POSIX = {0x04C11DB7, 0x00000000, false, false, 0xFFFFFFFF, {0}};
static const nvcompCRC32Spec_t nvcompCRC32_MPEG_2 = {0x04C11DB7, 0xFFFFFFFF, false, false, 0x00000000, {0}};

typedef enum nvcompCRC32KernelKind_t { nvcompCRC32WarpKernel = 0, nvcompCRC32BlockKernel = 1 } nvcompCRC32KernelKind_t;
typedef struct {
  nvcompCRC32KernelKind_t kernel_kind;
  int32_t bytes_per_read;
  int32_t blocks_per_msg;
  char reserved[20];
} nvcompCRC32KernelConf_t;
typedef struct {
  nvcompCRC32Spec_t spec;
  nvcompCRC32KernelConf_t kernel_conf;
  char reserved[64];
} nvcompBatchedCRC32Opts_t;
typedef enum nvcompCRC32SegmentKind_t {
  nvcompCRC32OnlySegment = 0,
  nvcompCRC32FirstSegment,
  nvcompCRC32MidSegment,
  nvcompCRC32LastSegment
} nvcompCRC32SegmentKind_t;
static const size_t* const nvcompCRC32IgnoredInputChunkBytes = NULL;
static const size_t nvcompCRC32DeducedMaxInputChunkBytes = 0;

NVCOMP_EXPORT nvcompStatus_t nvcompBatchedCRC32Async(const void* const* device_input_chunk_ptrs,
                                                     const size_t* device_input_chunk_bytes, size_t num_chunks,
                                                     uint32_t* device_crc32_ptr, nvcompBatchedCRC32Opts_t opts,
                                                     nvcompCRC32SegmentKind_t segment_kind,
                                                     nvcompStatus_t* device_statuses, cudaStream_t stream);
NVCOMP_EXPORT nvcompStatus_t nvcompBatchedCRC32GetHeuristicConf(const size_t* device_input_chunk_bytes,
                                                                size_t num_chunks,
                                                                nvcompCRC32KernelConf_t* kernel_conf,
                                                                size_t max_input_chunk_bytes, cudaStream_t stream);
NVCOMP_EXPORT nvcompStatus_t nvcompBatchedCRC32SearchConf(const void* const* device_input_chunk_ptrs,
                                                          const size_t* device_input_chunk_bytes, size_t num_chunks,
                                                          uint32_t* device_crc32_ptr, nvcompCRC32Spec_t spec,
                                                          nvcompCRC32KernelConf_t* kernel_conf, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_NVCOMP_H_ */
