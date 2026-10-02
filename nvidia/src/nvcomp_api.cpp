// libvgpunvcomp -- VirtualGPU's nvCOMP, presented as libnvcomp.so.5.
//
// nvCOMP compresses and decompresses batches of chunks with GPU kernels. Its
// standard formats are standard bitstreams -- an LZ4 chunk is an LZ4 block, a
// Deflate chunk raw DEFLATE, a Gzip chunk a gzip member, a Zstd chunk a
// Zstandard frame, a Snappy chunk raw Snappy -- so this library reads each
// chunk out of simulated device memory, runs the host codec for its format
// (nvcomp_codecs.cpp, written from the formats' specifications) and writes the
// result back. What it writes NVIDIA's library reads, and the other way round:
// both directions were checked on an RTX 3060 against nvCOMP 5.3.
//
// GDeflate is NVIDIA's own rearrangement of DEFLATE for SIMD decoding,
// published as an Internet-Draft (draft-uralsky-gdeflate-00); see
// nvcomp_gdeflate.cpp.
//
// Cascaded, Bitcomp and ANS are NVIDIA's own formats too, but their bitstreams
// are not publicly specified. Decoding NVIDIA's output would mean reverse
// engineering it, and an encoder of an invented layout would produce chunks
// NVIDIA's library cannot read, so every entry point of those three formats
// answers nvcompErrorNotSupported, and says why once.
//
// Written from NVIDIA's documented API with the simulator's own declarations
// (nvidia/include/vgpu_nvcomp.h). Where the documentation leaves an answer
// open -- a status, a size, an alignment -- it is what nvCOMP 5.3 answered on
// an RTX 3060, and the code says so where it is applied.
#include "../include/vgpu_nvcomp.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <vector>

#include "nvcomp_codecs.hpp"
#include "nvcomp_gdeflate.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/runtime/capture.hpp"
#include "vgpu/runtime/shim_memory.hpp"

namespace {

using vgpu::codec::Bytes;
using vgpu::codec::Result;

enum class Fmt { LZ4, Snappy, Deflate, Gdeflate, Gzip, Zstd };

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

void say_once(const char* key, const char* msg) {
  static std::mutex mu;
  static std::vector<const char*> said;
  std::lock_guard<std::mutex> lock(mu);
  for (const char* k : said)
    if (k == key) return;
  said.push_back(key);
  if (!quiet()) std::fprintf(stderr, "[vgpu] %s\n", msg);
}

// ---- moving bytes --------------------------------------------------------

// Whether n bytes at p can be read or written: device memory must lie in
// one allocation; anything else is host memory a kernel could reach (pinned,
// managed, mapped), which the simulator copies directly.
bool accessible(const void* p, size_t n) {
  if (n == 0) return true;
  if (!p) return false;
  if (!vgpu::is_device_va(reinterpret_cast<uint64_t>(p))) return true;
  void* base = nullptr;
  size_t size = 0;
  if (!vgpu_device_allocation(p, &base, &size)) return false;
  const size_t off = static_cast<size_t>(static_cast<const char*>(p) - static_cast<const char*>(base));
  return off <= size && n <= size - off;
}

bool get(void* host, const void* src, size_t n) {
  if (n == 0) return true;
  if (!accessible(src, n)) return false;
  if (cudaMemcpy(host, src, n, cudaMemcpyDefault) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

bool put(void* dst, const void* host, size_t n) {
  if (n == 0) return true;
  if (!accessible(dst, n)) return false;
  if (cudaMemcpy(dst, host, n, cudaMemcpyDefault) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

template <class T>
bool get_array(std::vector<T>& v, const T* src, size_t n) {
  v.resize(n);
  return get(v.data(), src, n * sizeof(T));
}

// Runs `work` in stream order: recorded into the graph while the stream is
// being captured, otherwise once the work queued before it has finished.
nvcompStatus_t in_stream_order(cudaStream_t stream, std::function<void()> work) {
  if (vgpu_record_host_op_if_capturing(stream, work)) return nvcompSuccess;
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGetLastError();
    return nvcompErrorCudaError;
  }
  work();
  return nvcompSuccess;
}

// ---- per-format facts -------------------------------------------------------

struct Facts {
  const char* name;
  size_t max_compress_chunk;    // the format's <Name>CompressionMaxAllowedChunkSize
  size_t max_decompress_chunk;  // ...DecompressionMaxAllowedChunkSize
  nvcompAlignmentRequirements_t comp_align, decomp_align;
};

// Alignments (input, output, temporary) as nvcompBatched*GetRequiredAlignments
// answered on an RTX 3060 with default options (nvCOMP 5.3). LZ4's input
// alignment is its data type's size; see lz4_input_alignment.
const Facts& facts(Fmt f) {
  static const Facts t[] = {
      {"LZ4", size_t(1) << 24, (size_t(1) << 32) - 1, {1, 1, 2}, {1, 1, 1}},
      {"Snappy", size_t(1) << 24, (size_t(1) << 31) - 1, {1, 1, 1}, {1, 1, 1}},
      {"Deflate", size_t(1) << 31, (size_t(1) << 32) - 1, {1, 8, 8}, {4, 1, 1}},
      {"Gdeflate", size_t(1) << 31, (size_t(1) << 32) + 288, {1, 4, 8}, {4, 1, 1}},
      {"Gzip", size_t(0x7FFFFFFF) << 15, (size_t(1) << 32) - 1, {1, 8, 8}, {1, 1, 1}},
      {"Zstd", (size_t(1) << 31) - 1, (size_t(1) << 31) - 1, {4, 1, 1}, {1, 1, 8}},
  };
  return t[static_cast<int>(f)];
}

// The bound nvcompBatched*CompressGetMaxOutputChunkSize answers, as measured
// on an RTX 3060 for sizes from 0 to 16 MiB. Each is at least what this
// library's encoder can write (the codec unit test checks it), so a buffer
// sized for either library fits both.
size_t max_output(Fmt f, size_t n) {
  switch (f) {
    case Fmt::LZ4: return vgpu::codec::lz4_bound(n);    // (n + n / 255 + 9) & ~7, as NVIDIA's
    case Fmt::Snappy: return vgpu::codec::snappy_bound(n);  // 32 + n + n / 6, as NVIDIA's
    case Fmt::Deflate:
      // NVIDIA's is about 2.26 n; this follows it to within 8 bytes, and is
      // never below this encoder's own bound.
      return std::max(std::max<size_t>(64, 8 * (n * 579 / 2048) + 32), vgpu::codec::deflate_bound(n));
    case Fmt::Gdeflate: return vgpu::codec::gdeflate_bound(n);  // (2 n + 288) & ~3, exactly
    case Fmt::Gzip: return vgpu::codec::gzip_bound(n);  // as NVIDIA's
    case Fmt::Zstd: return vgpu::codec::zstd_bound(n);  // as NVIDIA's
  }
  return n;
}

Bytes compress(Fmt f, const uint8_t* in, size_t n, int level) {
  switch (f) {
    case Fmt::LZ4: return vgpu::codec::lz4_compress(in, n);
    case Fmt::Snappy: return vgpu::codec::snappy_compress(in, n);
    case Fmt::Deflate: return vgpu::codec::deflate_compress(in, n, level);
    case Fmt::Gdeflate: return vgpu::codec::gdeflate_compress(in, n, level);
    case Fmt::Gzip: return vgpu::codec::gzip_compress(in, n, level);
    case Fmt::Zstd: return vgpu::codec::zstd_compress(in, n);
  }
  return Bytes();
}

Result decompress(Fmt f, const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced) {
  switch (f) {
    case Fmt::LZ4: return vgpu::codec::lz4_decompress(in, n, out, cap, produced);
    case Fmt::Snappy: return vgpu::codec::snappy_decompress(in, n, out, cap, produced);
    case Fmt::Deflate: return vgpu::codec::inflate(in, n, out, cap, produced);
    case Fmt::Gdeflate: return vgpu::codec::gdeflate_decompress(in, n, out, cap, produced);
    case Fmt::Gzip: return vgpu::codec::gunzip(in, n, out, cap, produced);
    case Fmt::Zstd: return vgpu::codec::zstd_decompress(in, n, out, cap, produced);
  }
  return Result::Corrupt;
}

// ---- the batched operations ----------------------------------------------------

struct CompressArgs {
  Fmt fmt;
  int level;
  const void* const* ptrs;
  const size_t* sizes;
  size_t num;
  void* const* out_ptrs;
  size_t* out_sizes;
  nvcompStatus_t* statuses;
};

void run_compress(const CompressArgs& a) {
  std::vector<const void*> in;
  std::vector<size_t> sizes;
  std::vector<void*> out;
  std::vector<size_t> out_sizes(a.num, 0);
  std::vector<nvcompStatus_t> st(a.num, nvcompSuccess);
  if (!get_array(in, const_cast<const void**>(reinterpret_cast<const void* const*>(a.ptrs)), a.num) ||
      !get_array(sizes, a.sizes, a.num) || !get_array(out, const_cast<void**>(a.out_ptrs), a.num)) {
    say_once("nvcomp-args", "nvCOMP: a chunk pointer or size array is not readable memory");
    return;
  }
  const Facts& fx = facts(a.fmt);
  for (size_t i = 0; i < a.num; ++i) {
    if (sizes[i] > fx.max_compress_chunk) {
      st[i] = nvcompErrorChunkSizeTooLarge;
      continue;
    }
    std::vector<uint8_t> buf(sizes[i]);
    if (!get(buf.data(), in[i], sizes[i])) {
      st[i] = nvcompErrorCudaError;
      continue;
    }
    const Bytes c = compress(a.fmt, buf.data(), buf.size(), a.level);
    if (!put(out[i], c.data(), c.size())) {
      st[i] = nvcompErrorCudaError;
      continue;
    }
    out_sizes[i] = c.size();
  }
  put(a.out_sizes, out_sizes.data(), a.num * sizeof(size_t));
  if (a.statuses) put(a.statuses, st.data(), a.num * sizeof(nvcompStatus_t));
}

struct DecompressArgs {
  Fmt fmt;
  const void* const* ptrs;
  const size_t* sizes;
  const size_t* buffer_sizes;
  size_t* actual_sizes;
  size_t num;
  void* const* out_ptrs;
  nvcompStatus_t* statuses;
};

nvcompStatus_t chunk_status(Result r) {
  switch (r) {
    case Result::Ok: return nvcompSuccess;
    case Result::BadChecksum: return nvcompErrorBadChecksum;
    // A chunk that does not fit its buffer, like a chunk that is not a valid
    // stream, is nvcompErrorCannotDecompress (documented for the former).
    default: return nvcompErrorCannotDecompress;
  }
}

void run_decompress(const DecompressArgs& a) {
  std::vector<const void*> in;
  std::vector<size_t> sizes, caps;
  std::vector<void*> out;
  std::vector<size_t> actual(a.num, 0);
  std::vector<nvcompStatus_t> st(a.num, nvcompSuccess);
  if (!get_array(in, const_cast<const void**>(reinterpret_cast<const void* const*>(a.ptrs)), a.num) ||
      !get_array(sizes, a.sizes, a.num) || !get_array(caps, a.buffer_sizes, a.num) ||
      !get_array(out, const_cast<void**>(a.out_ptrs), a.num)) {
    say_once("nvcomp-args", "nvCOMP: a chunk pointer or size array is not readable memory");
    return;
  }
  for (size_t i = 0; i < a.num; ++i) {
    std::vector<uint8_t> src(sizes[i]);
    if (!get(src.data(), in[i], sizes[i])) {
      st[i] = nvcompErrorCudaError;
      continue;
    }
    std::vector<uint8_t> dst(caps[i]);
    size_t produced = 0;
    const Result r = decompress(a.fmt, src.data(), src.size(), dst.data(), dst.size(), &produced);
    st[i] = chunk_status(r);
    if (r != Result::Ok) continue;
    if (!put(out[i], dst.data(), produced)) {
      st[i] = nvcompErrorCudaError;
      continue;
    }
    actual[i] = produced;
  }
  if (a.actual_sizes) put(a.actual_sizes, actual.data(), a.num * sizeof(size_t));
  if (a.statuses) put(a.statuses, st.data(), a.num * sizeof(nvcompStatus_t));
}

void run_sizes(Fmt fmt, const void* const* ptrs, const size_t* sizes_p, size_t* out_sizes, size_t num) {
  std::vector<const void*> in;
  std::vector<size_t> sizes;
  if (!get_array(in, const_cast<const void**>(reinterpret_cast<const void* const*>(ptrs)), num) ||
      !get_array(sizes, sizes_p, num)) {
    say_once("nvcomp-args", "nvCOMP: a chunk pointer or size array is not readable memory");
    return;
  }
  std::vector<size_t> out(num, 0);
  for (size_t i = 0; i < num; ++i) {
    std::vector<uint8_t> src(sizes[i]);
    if (!get(src.data(), in[i], sizes[i])) continue;
    size_t n = 0;
    // Measured by decoding without writing: every format's own length fields
    // are checked against its data, and a stream that does not parse is 0.
    if (fmt == Fmt::Snappy) {
      if (vgpu::codec::snappy_length(src.data(), src.size(), &n) != Result::Ok) n = 0;
    } else if (decompress(fmt, src.data(), src.size(), nullptr, 0, &n) != Result::Ok) {
      n = 0;
    }
    out[i] = n;
  }
  put(out_sizes, out.data(), num * sizeof(size_t));
}

// ---- argument checks shared by every format ---------------------------------------


// Each format's options, read into what the codecs need.
struct Opts {
  bool ok = true;
  int level = 1;
};

// The reserved bytes are not checked (NVIDIA's library accepts nonzero ones).
// An option outside its documented values is nvcompErrorNotSupported on the
// card: an LZ4 data type other than the 1-, 2- and 4-byte integers and bits,
// a Deflate algorithm outside 0..5.
Opts check(const nvcompBatchedLZ4CompressOpts_t& o) {
  Opts r;
  const int t = o.data_type;
  r.ok = t == NVCOMP_TYPE_CHAR || t == NVCOMP_TYPE_UCHAR || t == NVCOMP_TYPE_SHORT || t == NVCOMP_TYPE_USHORT ||
         t == NVCOMP_TYPE_INT || t == NVCOMP_TYPE_UINT || t == NVCOMP_TYPE_BITS;
  return r;
}
Opts check(const nvcompBatchedSnappyCompressOpts_t&) { return {true, 1}; }
Opts check_level(int algorithm, const char*, size_t) {
  Opts r;
  r.ok = algorithm >= 0 && algorithm <= 5;
  r.level = algorithm;
  return r;
}
Opts check(const nvcompBatchedDeflateCompressOpts_t& o) {
  return check_level(o.algorithm, o.reserved, sizeof o.reserved);
}
Opts check(const nvcompBatchedGdeflateCompressOpts_t& o) {
  return check_level(o.algorithm, o.reserved, sizeof o.reserved);
}
Opts check(const nvcompBatchedGzipCompressOpts_t& o) {
  return check_level(o.algorithm, o.reserved, sizeof o.reserved);
}
Opts check(const nvcompBatchedZstdCompressOpts_t&) { return {true, 1}; }

size_t lz4_input_alignment(const nvcompBatchedLZ4CompressOpts_t& o) {
  switch (o.data_type) {
    case NVCOMP_TYPE_SHORT:
    case NVCOMP_TYPE_USHORT: return 2;
    case NVCOMP_TYPE_INT:
    case NVCOMP_TYPE_UINT: return 4;
    default: return 1;
  }
}
template <class O>
size_t input_alignment(const O&, size_t dflt) { return dflt; }
size_t input_alignment(const nvcompBatchedLZ4CompressOpts_t& o, size_t) { return lz4_input_alignment(o); }

bool backend_ok(nvcompDecompressBackend_t b) {
  return b == NVCOMP_DECOMPRESS_BACKEND_DEFAULT || b == NVCOMP_DECOMPRESS_BACKEND_CUDA ||
         b == NVCOMP_DECOMPRESS_BACKEND_HARDWARE;
}
bool check(const nvcompBatchedLZ4DecompressOpts_t& o) { return backend_ok(o.backend); }
bool check(const nvcompBatchedSnappyDecompressOpts_t& o) { return backend_ok(o.backend); }
bool check(const nvcompBatchedDeflateDecompressOpts_t& o) { return backend_ok(o.backend); }
bool check(const nvcompBatchedGdeflateDecompressOpts_t& o) { return backend_ok(o.backend); }
bool check(const nvcompBatchedGzipDecompressOpts_t& o) { return backend_ok(o.backend); }
bool check(const nvcompBatchedZstdDecompressOpts_t& o) { return backend_ok(o.backend); }

// LZ4's bitshuffle pre-pass is not implemented: refused by name rather than
// compressed without it, which NVIDIA's library would then un-shuffle wrongly.
bool lz4_bitshuffle(nvcompBitshuffleMode_t m, const char* api) {
  if (m == NVCOMP_BITSHUFFLE_NONE) return false;
  (void)api;
  say_once("nvcomp-bitshuffle", "nvCOMP LZ4: the bitshuffle option is refused (nvcompErrorNotSupported); "
                                "VirtualGPU's LZ4 has no bitshuffle pass");
  return true;
}
template <class O>
bool unsupported_option(const O&) { return false; }
bool unsupported_option(const nvcompBatchedLZ4CompressOpts_t& o) { return lz4_bitshuffle(o.bitshuffle_mode, ""); }
bool unsupported_option(const nvcompBatchedLZ4DecompressOpts_t& o) {
  return lz4_bitshuffle(o.bitshuffle_mode, "");
}

// Temporary storage: the simulator needs none, but a program allocates what
// it is told and may check what it gets, so these follow NVIDIA's answers
// where they were measured, and are otherwise one alignment unit.
size_t temp_compress(Fmt f, size_t num, size_t max_chunk) {
  (void)max_chunk;
  switch (f) {
    case Fmt::LZ4: return num * 32768;  // 32 KiB per chunk on the card
    default: return 0;
  }
}

template <Fmt F, class CO, class DO>
struct Api {
  static nvcompStatus_t comp_align(CO o, nvcompAlignmentRequirements_t* a) {
    if (!a) return nvcompErrorInvalidValue;
    if (!check(o).ok || unsupported_option(o)) return nvcompErrorNotSupported;
    *a = facts(F).comp_align;
    a->input = input_alignment(o, a->input);
    return nvcompSuccess;
  }
  static nvcompStatus_t decomp_align(DO o, nvcompAlignmentRequirements_t* a) {
    if (!a || !check(o)) return nvcompErrorInvalidValue;
    if (unsupported_option(o)) return nvcompErrorNotSupported;
    *a = facts(F).decomp_align;
    return nvcompSuccess;
  }
  static nvcompStatus_t comp_temp(size_t num, size_t max_chunk, CO o, size_t* temp, size_t) {
    if (!temp) return nvcompErrorInvalidValue;
    if (!check(o).ok || unsupported_option(o)) return nvcompErrorNotSupported;
    if (max_chunk > facts(F).max_compress_chunk) return nvcompErrorChunkSizeTooLarge;
    *temp = temp_compress(F, num, max_chunk);
    return nvcompSuccess;
  }
  static nvcompStatus_t decomp_temp(size_t num, size_t max_chunk, DO o, size_t* temp, size_t) {
    if (!temp || !check(o)) return nvcompErrorInvalidValue;
    if (unsupported_option(o)) return nvcompErrorNotSupported;
    (void)num;
    (void)max_chunk;
    *temp = 0;  // the card's answer for every format but Zstd's
    return nvcompSuccess;
  }
  static nvcompStatus_t max_out(size_t max_chunk, CO o, size_t* out) {
    if (!out) return nvcompErrorInvalidValue;
    if (!check(o).ok || unsupported_option(o)) return nvcompErrorNotSupported;
    if (max_chunk > facts(F).max_compress_chunk) return nvcompErrorChunkSizeTooLarge;
    *out = max_output(F, max_chunk);
    return nvcompSuccess;
  }
  static nvcompStatus_t compress(const void* const* ptrs, const size_t* sizes, size_t max_chunk, size_t num,
                                 void* temp, size_t temp_bytes, void* const* out_ptrs, size_t* out_sizes, CO o,
                                 nvcompStatus_t* statuses, cudaStream_t stream) {
    (void)temp;
    (void)temp_bytes;
    const Opts op = check(o);
    if (!op.ok || unsupported_option(o)) return nvcompErrorNotSupported;
    if (num == 0) return nvcompSuccess;
    if (!ptrs || !sizes || !out_ptrs || !out_sizes) return nvcompErrorInvalidValue;
    if (max_chunk > facts(F).max_compress_chunk) return nvcompErrorChunkSizeTooLarge;
    CompressArgs a{F, op.level, ptrs, sizes, num, out_ptrs, out_sizes, statuses};
    return in_stream_order(stream, [a] { run_compress(a); });
  }
  static nvcompStatus_t sizes(const void* const* ptrs, const size_t* sizes_p, size_t* out, size_t num,
                              cudaStream_t stream) {
    if (num == 0) return nvcompSuccess;
    if (!ptrs || !sizes_p || !out) return nvcompErrorInvalidValue;
    return in_stream_order(stream, [=] { run_sizes(F, ptrs, sizes_p, out, num); });
  }
  static nvcompStatus_t decompress(const void* const* ptrs, const size_t* sizes_p, const size_t* buffer_sizes,
                                   size_t* actual, size_t num, void* const temp, size_t temp_bytes,
                                   void* const* out_ptrs, DO o, nvcompStatus_t* statuses, cudaStream_t stream) {
    (void)temp;
    (void)temp_bytes;
    if (!check(o)) return nvcompErrorInvalidValue;
    if (unsupported_option(o)) return nvcompErrorNotSupported;
    if (num == 0) return nvcompSuccess;
    if (!ptrs || !sizes_p || !buffer_sizes || !out_ptrs) return nvcompErrorInvalidValue;
    DecompressArgs a{F, ptrs, sizes_p, buffer_sizes, actual, num, out_ptrs, statuses};
    return in_stream_order(stream, [a] { run_decompress(a); });
  }
  static nvcompStatus_t decomp_temp_sync(const void* const* ptrs, const size_t* sizes_p, size_t num,
                                         size_t max_chunk, size_t* temp, size_t max_total, DO o,
                                         nvcompStatus_t* statuses, cudaStream_t stream) {
    (void)ptrs;
    (void)sizes_p;
    const nvcompStatus_t r = decomp_temp(num, max_chunk, o, temp, max_total);
    if (r != nvcompSuccess) return r;
    if (statuses && num) {
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaGetLastError();
        return nvcompErrorCudaError;
      }
      std::vector<nvcompStatus_t> ok(num, nvcompSuccess);
      put(statuses, ok.data(), num * sizeof(nvcompStatus_t));
    }
    return nvcompSuccess;
  }
};

// The three formats without a public bitstream specification.
nvcompStatus_t proprietary(const char* fmt) {
  static std::mutex mu;
  static std::vector<std::string> said;
  std::lock_guard<std::mutex> lock(mu);
  if (std::find(said.begin(), said.end(), fmt) == said.end()) {
    said.push_back(fmt);
    if (!quiet())
      std::fprintf(stderr,
                   "[vgpu] nvCOMP %s is refused (nvcompErrorNotSupported): NVIDIA does not publish the %s "
                   "bitstream, so its chunks can be neither read nor written compatibly\n",
                   fmt, fmt);
  }
  return nvcompErrorNotSupported;
}

}  // namespace

// ---- exports -------------------------------------------------------------------

#define VGPU_NVCOMP_DEFINE(F, FMT)                                                                                  \
  using Api##F = Api<FMT, nvcompBatched##F##CompressOpts_t, nvcompBatched##F##DecompressOpts_t>;                   \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressGetRequiredAlignments(nvcompBatched##F##CompressOpts_t o,    \
                                                                            nvcompAlignmentRequirements_t* a) {    \
    return Api##F::comp_align(o, a);                                                                                \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressGetTempSizeAsync(                                            \
      size_t num, size_t max_chunk, nvcompBatched##F##CompressOpts_t o, size_t* temp, size_t total) {              \
    return Api##F::comp_temp(num, max_chunk, o, temp, total);                                                      \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressGetTempSizeSync(                                             \
      const void* const* const, const size_t* const, size_t num, size_t max_chunk,                                 \
      nvcompBatched##F##CompressOpts_t o, size_t* temp, size_t total, cudaStream_t) {                              \
    return Api##F::comp_temp(num, max_chunk, o, temp, total);                                                      \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressGetMaxOutputChunkSize(                                       \
      size_t max_chunk, nvcompBatched##F##CompressOpts_t o, size_t* out) {                                         \
    return Api##F::max_out(max_chunk, o, out);                                                                     \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressAsync(                                                       \
      const void* const* p, const size_t* s, size_t max_chunk, size_t num, void* temp, size_t temp_bytes,         \
      void* const* op, size_t* os, nvcompBatched##F##CompressOpts_t o, nvcompStatus_t* st, cudaStream_t stream) {  \
    return Api##F::compress(p, s, max_chunk, num, temp, temp_bytes, op, os, o, st, stream);                       \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##DecompressGetRequiredAlignments(                                     \
      nvcompBatched##F##DecompressOpts_t o, nvcompAlignmentRequirements_t* a) {                                    \
    return Api##F::decomp_align(o, a);                                                                             \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##DecompressGetTempSizeAsync(                                          \
      size_t num, size_t max_chunk, nvcompBatched##F##DecompressOpts_t o, size_t* temp, size_t total) {            \
    return Api##F::decomp_temp(num, max_chunk, o, temp, total);                                                    \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##DecompressGetTempSizeSync(                                           \
      const void* const* const p, const size_t* const s, size_t num, size_t max_chunk, size_t* temp,               \
      size_t total, nvcompBatched##F##DecompressOpts_t o, nvcompStatus_t* st, cudaStream_t stream) {               \
    return Api##F::decomp_temp_sync(p, s, num, max_chunk, temp, total, o, st, stream);                             \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##GetDecompressSizeAsync(                                              \
      const void* const* p, const size_t* s, size_t* out, size_t num, cudaStream_t stream) {                       \
    return Api##F::sizes(p, s, out, num, stream);                                                                  \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##DecompressAsync(                                                     \
      const void* const* p, const size_t* s, const size_t* bs, size_t* as, size_t num, void* const temp,          \
      size_t temp_bytes, void* const* op, nvcompBatched##F##DecompressOpts_t o, nvcompStatus_t* st,                \
      cudaStream_t stream) {                                                                                        \
    return Api##F::decompress(p, s, bs, as, num, temp, temp_bytes, op, o, st, stream);                             \
  }

VGPU_NVCOMP_DEFINE(LZ4, Fmt::LZ4)
VGPU_NVCOMP_DEFINE(Snappy, Fmt::Snappy)
VGPU_NVCOMP_DEFINE(Deflate, Fmt::Deflate)
VGPU_NVCOMP_DEFINE(Gdeflate, Fmt::Gdeflate)
VGPU_NVCOMP_DEFINE(Gzip, Fmt::Gzip)
VGPU_NVCOMP_DEFINE(Zstd, Fmt::Zstd)

#define VGPU_NVCOMP_REFUSE(F)                                                                                       \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressGetRequiredAlignments(nvcompBatched##F##CompressOpts_t,      \
                                                                            nvcompAlignmentRequirements_t*) {      \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressGetTempSizeAsync(size_t, size_t,                             \
                                                                       nvcompBatched##F##CompressOpts_t, size_t*, \
                                                                       size_t) {                                    \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressGetTempSizeSync(const void* const* const, const size_t* const, \
                                                                      size_t, size_t,                               \
                                                                      nvcompBatched##F##CompressOpts_t, size_t*,   \
                                                                      size_t, cudaStream_t) {                       \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressGetMaxOutputChunkSize(size_t,                                \
                                                                            nvcompBatched##F##CompressOpts_t,      \
                                                                            size_t*) {                              \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##CompressAsync(const void* const*, const size_t*, size_t, size_t,     \
                                                            void*, size_t, void* const*, size_t*,                   \
                                                            nvcompBatched##F##CompressOpts_t, nvcompStatus_t*,     \
                                                            cudaStream_t) {                                         \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##DecompressGetRequiredAlignments(nvcompBatched##F##DecompressOpts_t,  \
                                                                              nvcompAlignmentRequirements_t*) {    \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##DecompressGetTempSizeAsync(                                          \
      size_t, size_t, nvcompBatched##F##DecompressOpts_t, size_t*, size_t) {                                       \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##DecompressGetTempSizeSync(                                           \
      const void* const* const, const size_t* const, size_t, size_t, size_t*, size_t,                              \
      nvcompBatched##F##DecompressOpts_t, nvcompStatus_t*, cudaStream_t) {                                         \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##GetDecompressSizeAsync(const void* const*, const size_t*, size_t*,   \
                                                                     size_t, cudaStream_t) {                        \
    return proprietary(#F);                                                                                         \
  }                                                                                                                 \
  extern "C" nvcompStatus_t nvcompBatched##F##DecompressAsync(                                                     \
      const void* const*, const size_t*, const size_t*, size_t*, size_t, void* const, size_t, void* const*,       \
      nvcompBatched##F##DecompressOpts_t, nvcompStatus_t*, cudaStream_t) {                                         \
    return proprietary(#F);                                                                                         \
  }

VGPU_NVCOMP_REFUSE(Cascaded)
VGPU_NVCOMP_REFUSE(Bitcomp)
VGPU_NVCOMP_REFUSE(ANS)

// ---- library-wide ---------------------------------------------------------------

extern "C" nvcompStatus_t nvcompGetProperties(nvcompProperties_t* properties) {
  if (!properties) return nvcompErrorInvalidValue;
  properties->version = NVCOMP_VER;
  int rt = 0;
  if (cudaRuntimeGetVersion(&rt) != cudaSuccess) {
    cudaGetLastError();
    rt = CUDART_VERSION;
  }
  properties->cudart_version = (uint32_t)CUDART_VERSION;
  (void)rt;
  return nvcompSuccess;
}

// The descriptions NVIDIA's library returns (nvCOMP 5.3, measured).
extern "C" const char* nvcompGetStatusString(nvcompStatus_t status) {
  switch (status) {
    case nvcompSuccess: return "The operation completed successfully";
    case nvcompErrorInvalidValue: return "An invalid value was provided to the function";
    case nvcompErrorNotSupported: return "The requested operation or configuration is not supported";
    case nvcompErrorCannotDecompress:
      return "The data cannot be decompressed, possibly due to corruption or invalid format";
    case nvcompErrorBadChecksum: return "The checksum verification failed, indicating data corruption";
    case nvcompErrorCannotVerifyChecksums: return "Unable to verify checksums for the data";
    case nvcompErrorOutputBufferTooSmall: return "The provided output buffer size is insufficient for the operation";
    case nvcompErrorWrongHeaderLength: return "The header length does not match the expected value";
    case nvcompErrorAlignment: return "The buffer alignment does not meet the required alignment constraints";
    case nvcompErrorChunkSizeTooLarge: return "The chunk size exceeds the maximum allowed size";
    case nvcompErrorCannotCompress: return "The data cannot be compressed with the specified configuration";
    case nvcompErrorWrongInputLength: return "The input length is invalid or inconsistent with the operation";
    case nvcompErrorBatchSizeTooLarge: return "The batch size exceeds the maximum supported batch size";
    case nvcompErrorSubChunkCountTooLarge: return "The sub-chunk count exceeds the maximum supported per chunk";
    case nvcompErrorSubChunkCountTooSmall: return "The sub-chunk count is below the minimum required per chunk";
    case nvcompErrorOutputBufferAlignmentTooSmall: return "The output buffer alignment is too small for the data type";
    case nvcompErrorCudaError: return "A CUDA runtime or device error occurred during the operation";
    case nvcompErrorInternal: return "An internal library error occurred";
  }
  return "Unrecognized nvcompStatus_t error code";
}

// ---- CRC32 -------------------------------------------------------------------------
//
// Any CRC-32 model (polynomial, initial value, reflections, final XOR), over
// chunks that may arrive as segments of one message: a segment's state is the
// running register value, kept in the output slot between calls.

namespace {

uint32_t reflect32(uint32_t v) {
  uint32_t r = 0;
  for (int i = 0; i < 32; ++i) r |= ((v >> i) & 1u) << (31 - i);
  return r;
}

// Advances the CRC register over data, in the model's bit order. The register
// is kept unreflected for ref_in = false and reflected for ref_in = true.
uint32_t crc_update(const nvcompCRC32Spec_t& s, uint32_t reg, const uint8_t* p, size_t n) {
  if (s.ref_in) {
    const uint32_t poly = reflect32(s.poly);
    for (size_t i = 0; i < n; ++i) {
      reg ^= p[i];
      for (int k = 0; k < 8; ++k) reg = (reg & 1) ? (reg >> 1) ^ poly : reg >> 1;
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      reg ^= uint32_t(p[i]) << 24;
      for (int k = 0; k < 8; ++k) reg = (reg & 0x80000000u) ? (reg << 1) ^ s.poly : reg << 1;
    }
  }
  return reg;
}

uint32_t crc_init(const nvcompCRC32Spec_t& s) { return s.ref_in ? reflect32(s.init) : s.init; }

uint32_t crc_final(const nvcompCRC32Spec_t& s, uint32_t reg) {
  // reg is in input bit order; the result is reflected iff ref_out.
  const uint32_t out = (s.ref_in == s.ref_out) ? reg : reflect32(reg);
  return out ^ s.xorout;
}

}  // namespace

extern "C" nvcompStatus_t nvcompBatchedCRC32Async(const void* const* ptrs, const size_t* sizes_p, size_t num,
                                                  uint32_t* crc_out, nvcompBatchedCRC32Opts_t opts,
                                                  nvcompCRC32SegmentKind_t kind, nvcompStatus_t* statuses,
                                                  cudaStream_t stream) {
  if (kind < nvcompCRC32OnlySegment || kind > nvcompCRC32LastSegment) return nvcompErrorInvalidValue;
  if (num == 0) return nvcompSuccess;
  if (!crc_out || (!ptrs && kind != nvcompCRC32LastSegment) || (ptrs && !sizes_p)) return nvcompErrorInvalidValue;
  const nvcompCRC32Spec_t spec = opts.spec;
  return in_stream_order(stream, [=] {
    std::vector<uint32_t> state(num, 0);
    std::vector<nvcompStatus_t> st(num, nvcompSuccess);
    const bool first = kind == nvcompCRC32OnlySegment || kind == nvcompCRC32FirstSegment;
    const bool last = kind == nvcompCRC32OnlySegment || kind == nvcompCRC32LastSegment;
    // A segment after the first continues from the register the previous one
    // left in the output slot (unfinalized: see below).
    if (!first && !get(state.data(), crc_out, num * sizeof(uint32_t))) return;
    if (ptrs) {
      std::vector<const void*> in;
      std::vector<size_t> sizes;
      if (!get_array(in, const_cast<const void**>(reinterpret_cast<const void* const*>(ptrs)), num) ||
          !get_array(sizes, sizes_p, num))
        return;
      for (size_t i = 0; i < num; ++i) {
        uint32_t reg = first ? crc_init(spec) : state[i];
        std::vector<uint8_t> buf(sizes[i]);
        if (!get(buf.data(), in[i], sizes[i])) {
          st[i] = nvcompErrorCudaError;
          continue;
        }
        state[i] = crc_update(spec, reg, buf.data(), buf.size());
      }
    } else {
      // Retroactively the last segment: nothing more to read.
    }
    if (last)
      for (size_t i = 0; i < num; ++i) state[i] = crc_final(spec, state[i]);
    put(crc_out, state.data(), num * sizeof(uint32_t));
    if (statuses) put(statuses, st.data(), num * sizeof(nvcompStatus_t));
  });
}

// The kernel configuration only tunes NVIDIA's kernels; any answer is valid.
extern "C" nvcompStatus_t nvcompBatchedCRC32GetHeuristicConf(const size_t*, size_t,
                                                             nvcompCRC32KernelConf_t* kernel_conf, size_t,
                                                             cudaStream_t) {
  if (!kernel_conf) return nvcompErrorInvalidValue;
  std::memset(kernel_conf, 0, sizeof *kernel_conf);
  kernel_conf->kernel_kind = nvcompCRC32WarpKernel;
  kernel_conf->bytes_per_read = 4;
  kernel_conf->blocks_per_msg = 1;
  return nvcompSuccess;
}

extern "C" nvcompStatus_t nvcompBatchedCRC32SearchConf(const void* const* ptrs, const size_t* sizes_p, size_t num,
                                                       uint32_t* crc_out, nvcompCRC32Spec_t spec,
                                                       nvcompCRC32KernelConf_t* kernel_conf, cudaStream_t stream) {
  if (!kernel_conf) return nvcompErrorInvalidValue;
  nvcompBatchedCRC32GetHeuristicConf(sizes_p, num, kernel_conf, 0, stream);
  nvcompBatchedCRC32Opts_t o{};
  o.spec = spec;
  o.kernel_conf = *kernel_conf;
  const nvcompStatus_t r = nvcompBatchedCRC32Async(ptrs, sizes_p, num, crc_out, o, nvcompCRC32OnlySegment,
                                                   nullptr, stream);
  if (r != nvcompSuccess) return r;
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGetLastError();
    return nvcompErrorCudaError;
  }
  return nvcompSuccess;
}
