// nvCOMP's high-level interface: the managers (nvcomp::LZ4Manager and its
// siblings), the configurations they hand out, and the factory that makes a
// manager for a compressed buffer.
//
// A manager compresses a whole buffer by cutting it into chunks of a fixed
// size and running the batched codecs (nvcomp_api.cpp's). Its output is one
// of three layouts (BitstreamKind):
//   NVCOMP_NATIVE           nvCOMP's container: a header, each chunk's offset
//                           and size, the chunks, 8-byte aligned;
//   RAW                     the format's bitstream of the whole buffer, one chunk;
//   WITH_UNCOMPRESSED_SIZE  the same after the uncompressed size (4 bytes for
//                           LZ4, 8 for the others).
// The container's layout is not documented; it was measured from what
// NVIDIA's nvCOMP 5.3 writes on an RTX 3060, and NVIDIA's library reads what
// this one writes (e2e_nvcomp_manager). Its format-specific part is the
// public formatSpec.hpp's structs.
//
// Checksums: the container can carry per-chunk checksums whose algorithm is
// not public (it is no standard CRC or hash of the chunks). A policy that
// computes them is refused at construction (nvcompErrorNotSupported); a
// policy that verifies them if present decompresses a buffer that carries
// them and reports nvcompErrorCannotVerifyChecksums in its status.
//
// Cascaded, Bitcomp and ANS managers are refused like their batched APIs.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "../include/vgpu_nvcomp.hpp"
#include "nvcomp_internal.hpp"

namespace impl = vgpu::nvcomp_impl;
using impl::Fmt;
using vgpu::codec::Bytes;
using vgpu::codec::Result;

namespace {

constexpr uint32_t kMagic = 0x52911326u;
constexpr uint8_t kMajor = 5, kMinor = 3;
constexpr size_t kCommonSize = 0x40;
constexpr size_t kChunkAlign = 8;

[[noreturn]] void raise(nvcompStatus_t s, const std::string& msg) { throw nvcomp::NVCompException(s, msg); }

size_t align8(size_t x) { return (x + kChunkAlign - 1) / kChunkAlign * kChunkAlign; }

// The container's fixed part, as nvCOMP 5.3 writes it (measured).
struct CommonHeader {
  uint32_t magic;
  uint8_t major, minor;
  uint8_t format;  // nvcompFormatType_t
  uint8_t pad0;
  uint64_t comp_data_size;    // the chunks' region: the last chunk's offset plus its size
  uint64_t uncomp_size;
  uint64_t num_chunks;
  uint8_t one;                // 1 in every buffer NVIDIA's library wrote
  uint8_t pad1[3];
  uint32_t zero;
  uint32_t checksum;          // with per-chunk checksums: one over them (algorithm not public)
  uint8_t checksums_present;  // both 1 when the buffer carries checksums
  uint8_t checksums_present2;
  uint8_t pad2[2];
  uint64_t uncomp_chunk_size;
  uint32_t data_offset;       // where the first chunk starts
  uint32_t pad3;
};
static_assert(sizeof(CommonHeader) == kCommonSize, "the container's fixed part is 64 bytes");

}  // namespace

// ============================================================================
// Configurations
// ============================================================================

namespace nvcomp {

// A status per configuration, in device memory, as a kernel would write it.
struct CompressionConfig::CompressionConfigImpl {
  nvcompStatus_t* status = nullptr;
  CompressionConfigImpl() {
    if (cudaMalloc(reinterpret_cast<void**>(&status), sizeof(nvcompStatus_t)) != cudaSuccess) {
      cudaGetLastError();
      status = nullptr;
    } else {
      const nvcompStatus_t ok = nvcompSuccess;
      cudaMemcpy(status, &ok, sizeof ok, cudaMemcpyHostToDevice);
    }
  }
  ~CompressionConfigImpl() {
    if (status) cudaFree(status);
  }
};
struct DecompressionConfig::DecompressionConfigImpl {
  nvcompStatus_t* status = nullptr;
  // What a manager learned about the buffer: where it came from, its layout.
  size_t comp_size = 0;
  std::vector<uint64_t> offsets, sizes;
  size_t chunk_size = 0, data_offset = 0;
  BitstreamKind kind = BitstreamKind::NVCOMP_NATIVE;
  DecompressionConfigImpl() {
    if (cudaMalloc(reinterpret_cast<void**>(&status), sizeof(nvcompStatus_t)) != cudaSuccess) {
      cudaGetLastError();
      status = nullptr;
    } else {
      const nvcompStatus_t ok = nvcompSuccess;
      cudaMemcpy(status, &ok, sizeof ok, cudaMemcpyHostToDevice);
    }
  }
  ~DecompressionConfigImpl() {
    if (status) cudaFree(status);
  }
};

CompressionConfig::CompressionConfig()
    : impl(std::make_shared<CompressionConfigImpl>()),
      uncompressed_buffer_size(0),
      max_compressed_buffer_size(0),
      num_chunks(0),
      compute_checksums(false) {}
CompressionConfig::CompressionConfig(size_t n) : CompressionConfig() { uncompressed_buffer_size = n; }
nvcompStatus_t* CompressionConfig::get_status() const { return impl ? impl->status : nullptr; }
CompressionConfig::CompressionConfig(CompressionConfig&& o) = default;
CompressionConfig::CompressionConfig(const CompressionConfig& o) = default;
CompressionConfig& CompressionConfig::operator=(CompressionConfig&& o) = default;
CompressionConfig& CompressionConfig::operator=(const CompressionConfig& o) = default;
CompressionConfig::~CompressionConfig() noexcept = default;

DecompressionConfig::DecompressionConfig()
    : impl(std::make_shared<DecompressionConfigImpl>()), decomp_data_size(0), num_chunks(0), checksums_present(false) {}
nvcompStatus_t* DecompressionConfig::get_status() const { return impl ? impl->status : nullptr; }
DecompressionConfig::DecompressionConfig(DecompressionConfig&& o) = default;
DecompressionConfig::DecompressionConfig(const DecompressionConfig& o) = default;
DecompressionConfig& DecompressionConfig::operator=(DecompressionConfig&& o) = default;
DecompressionConfig& DecompressionConfig::operator=(const DecompressionConfig& o) = default;
DecompressionConfig::~DecompressionConfig() noexcept = default;

// ============================================================================
// The manager
// ============================================================================

namespace detail {

struct nvcompManagerInternalBase {
  Fmt fmt;
  nvcompFormatType_t format;
  int level = 1;
  std::vector<uint8_t> spec;  // the format's formatSpec.hpp header, as stored
  size_t chunk = 65536;
  cudaStream_t stream = nullptr;
  ChecksumPolicy policy = NoComputeNoVerify;
  BitstreamKind kind = BitstreamKind::NVCOMP_NATIVE;

  size_t spec_end() const { return align8(kCommonSize + std::max<size_t>(spec.size(), 1)); }
  size_t prefix() const { return kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE ? (fmt == Fmt::LZ4 ? 4 : 8) : 0; }
  size_t chunks_for(size_t n) const {
    if (kind != BitstreamKind::NVCOMP_NATIVE) return 1;
    return chunk ? (n + chunk - 1) / chunk : 0;
  }

  void set_status(nvcompStatus_t* where, nvcompStatus_t s) const {
    if (where) impl::put(where, &s, sizeof s);
  }

  CompressionConfig configure_compression(size_t n) const {
    CompressionConfig c(n);
    c.num_chunks = chunks_for(n);
    c.compute_checksums = false;
    if (kind != BitstreamKind::NVCOMP_NATIVE) {
      c.max_compressed_buffer_size = prefix() + impl::max_output(fmt, n);
    } else {
      // NVIDIA's bound: the header, an offset and a size per chunk, and each
      // chunk's worst case rounded up to the 8-byte chunk alignment.
      const size_t per = align8(impl::max_output(fmt, std::min(n, chunk)));
      c.max_compressed_buffer_size = spec_end() + std::max<size_t>(c.num_chunks, 1) * 16 + c.num_chunks * per;
      if (c.num_chunks == 0) c.max_compressed_buffer_size = spec_end() + 8;
    }
    return c;
  }

  // Compresses one buffer; returns the compressed size.
  size_t compress_one(const uint8_t* in, uint8_t* out, size_t n) const {
    std::vector<uint8_t> src(n);
    if (!impl::get(src.data(), in, n)) raise(nvcompErrorCudaError, "the uncompressed buffer is not readable");
    if (kind != BitstreamKind::NVCOMP_NATIVE) {
      Bytes c = impl::compress(fmt, src.data(), n, level);
      std::vector<uint8_t> o(prefix());
      for (size_t i = 0; i < prefix(); ++i) o[i] = uint8_t(uint64_t(n) >> (8 * i));
      o.insert(o.end(), c.begin(), c.end());
      if (!impl::put(out, o.data(), o.size())) raise(nvcompErrorCudaError, "the compressed buffer is not writable");
      return o.size();
    }
    const size_t nc = chunks_for(n);
    std::vector<Bytes> pieces(nc);
    std::vector<uint64_t> offsets(nc), sizes(nc);
    size_t at = 0;
    for (size_t i = 0; i < nc; ++i) {
      const size_t begin = i * chunk, len = std::min(chunk, n - begin);
      pieces[i] = impl::compress(fmt, src.data() + begin, len, level);
      at = align8(at);
      offsets[i] = at;
      sizes[i] = pieces[i].size();
      at += pieces[i].size();
    }
    const size_t data_offset = spec_end() + nc * 16;
    std::vector<uint8_t> o(data_offset + at, 0);
    CommonHeader h{};
    h.magic = kMagic;
    h.major = kMajor;
    h.minor = kMinor;
    h.format = static_cast<uint8_t>(format);
    h.comp_data_size = at;
    h.uncomp_size = n;
    h.num_chunks = nc;
    h.one = 1;
    h.uncomp_chunk_size = chunk;
    h.data_offset = uint32_t(data_offset);
    std::memcpy(o.data(), &h, sizeof h);
    // An empty vector's data() may be null, which memcpy may not be given
    // even for no bytes: a format with no spec, no chunks, an empty chunk.
    auto copy = [&](size_t at, const void* from, size_t len) {
      if (len) std::memcpy(o.data() + at, from, len);
    };
    copy(kCommonSize, spec.data(), spec.size());
    copy(spec_end(), offsets.data(), nc * 8);
    copy(spec_end() + nc * 8, sizes.data(), nc * 8);
    for (size_t i = 0; i < nc; ++i) copy(data_offset + offsets[i], pieces[i].data(), sizes[i]);
    if (!impl::put(out, o.data(), o.size())) raise(nvcompErrorCudaError, "the compressed buffer is not writable");
    return o.size();
  }

  CommonHeader read_header(const uint8_t* comp) const {
    CommonHeader h{};
    if (!impl::get(&h, comp, sizeof h)) raise(nvcompErrorCudaError, "the compressed buffer is not readable");
    if (h.magic != kMagic) raise(nvcompErrorInvalidValue, "the buffer is not an nvCOMP buffer (no nvCOMP header)");
    return h;
  }

  DecompressionConfig configure_decompression(const uint8_t* comp, const size_t* comp_size) const {
    DecompressionConfig d;
    auto& di = *d.impl;
    di.kind = kind;
    size_t csize = 0;
    if (comp_size && !impl::get(&csize, comp_size, sizeof csize)) raise(nvcompErrorCudaError, "comp_size is not readable");
    di.comp_size = csize;
    if (kind != BitstreamKind::NVCOMP_NATIVE) {
      if (!comp_size) raise(nvcompErrorInvalidValue, "a buffer that is not NVCOMP_NATIVE needs its compressed size");
      d.num_chunks = 1;
      if (kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE) {
        uint64_t n = 0;
        if (csize < prefix() || !impl::get(&n, comp, prefix())) raise(nvcompErrorCannotDecompress, "no uncompressed size");
        d.decomp_data_size = size_t(n);
      } else {
        std::vector<uint8_t> src(csize);
        if (!impl::get(src.data(), comp, csize)) raise(nvcompErrorCudaError, "the compressed buffer is not readable");
        size_t n = 0;
        if (impl::decompress(fmt, src.data(), csize, nullptr, 0, &n) != Result::Ok) {
          set_status(d.get_status(), nvcompErrorCannotDecompress);
          n = 0;
        }
        d.decomp_data_size = n;
      }
      return d;
    }
    const CommonHeader h = read_header(comp);
    if (h.format != static_cast<uint8_t>(format))
      raise(nvcompErrorInvalidValue, "the buffer was compressed with another format than this manager's");
    d.decomp_data_size = h.uncomp_size;
    d.num_chunks = h.num_chunks;
    d.checksums_present = h.checksums_present != 0;
    di.chunk_size = h.uncomp_chunk_size;
    di.data_offset = h.data_offset;
    const size_t nc = h.num_chunks;
    if (nc > (size_t(1) << 32) || (nc && !h.uncomp_chunk_size)) raise(nvcompErrorCannotDecompress, "a corrupt header");
    di.offsets.resize(nc);
    di.sizes.resize(nc);
    if (nc && (!impl::get(di.offsets.data(), comp + spec_end(), nc * 8) ||
               !impl::get(di.sizes.data(), comp + spec_end() + nc * 8, nc * 8)))
      raise(nvcompErrorCudaError, "the compressed buffer is not readable");
    return d;
  }

  nvcompStatus_t decompress_one(uint8_t* out, const uint8_t* comp, const DecompressionConfig& d,
                                const size_t* comp_size) const {
    const auto& di = *d.impl;
    nvcompStatus_t st = nvcompSuccess;
    if (di.kind != BitstreamKind::NVCOMP_NATIVE) {
      size_t csize = di.comp_size;
      if (comp_size && !impl::get(&csize, comp_size, sizeof csize)) raise(nvcompErrorCudaError, "comp_size is not readable");
      const size_t pre = di.kind == BitstreamKind::WITH_UNCOMPRESSED_SIZE ? prefix() : 0;
      if (csize < pre) raise(nvcompErrorCannotDecompress, "the buffer is shorter than its size prefix");
      std::vector<uint8_t> src(csize - pre), dst(d.decomp_data_size);
      if (!impl::get(src.data(), comp + pre, src.size())) raise(nvcompErrorCudaError, "the buffer is not readable");
      size_t got = 0;
      if (impl::decompress(fmt, src.data(), src.size(), dst.data(), dst.size(), &got) != Result::Ok ||
          got != dst.size())
        st = nvcompErrorCannotDecompress;
      else if (!impl::put(out, dst.data(), got))
        st = nvcompErrorCudaError;
      return st;
    }
    const size_t nc = d.num_chunks;
    std::vector<uint8_t> dst(d.decomp_data_size);
    for (size_t i = 0; i < nc && st == nvcompSuccess; ++i) {
      const size_t begin = i * di.chunk_size;
      if (begin > dst.size()) {
        st = nvcompErrorCannotDecompress;
        break;
      }
      const size_t want = std::min(di.chunk_size, dst.size() - begin);
      std::vector<uint8_t> src(di.sizes[i]);
      if (!impl::get(src.data(), comp + di.data_offset + di.offsets[i], src.size())) {
        st = nvcompErrorCudaError;
        break;
      }
      size_t got = 0;
      if (impl::decompress(fmt, src.data(), src.size(), dst.data() + begin, want, &got) != Result::Ok || got != want)
        st = nvcompErrorCannotDecompress;
    }
    if (st == nvcompSuccess && !impl::put(out, dst.data(), dst.size())) st = nvcompErrorCudaError;
    // Checksums this library cannot check: the data is there, the status says so.
    if (st == nvcompSuccess && d.checksums_present &&
        (policy == NoComputeAndVerifyIfPresent || policy == ComputeAndVerifyIfPresent || policy == ComputeAndVerify))
      st = nvcompErrorCannotVerifyChecksums;
    return st;
  }

  size_t compressed_size(const uint8_t* comp) const {
    if (kind != BitstreamKind::NVCOMP_NATIVE)
      raise(nvcompErrorNotSupported, "get_compressed_output_size can only be called if bitstream kind is NVCOMP_NATIVE");
    const CommonHeader h = read_header(comp);
    return h.data_offset + h.comp_data_size;
  }

  size_t decompressed_size(const uint8_t* comp) const {
    if (kind != BitstreamKind::NVCOMP_NATIVE)
      raise(nvcompErrorNotSupported,
            "get_decompressed_output_size can only be called if bitstream kind is NVCOMP_NATIVE");
    return read_header(comp).uncomp_size;
  }

  // Every operation runs in the manager's stream's order.
  void ordered(const std::function<void()>& f) const {
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      cudaGetLastError();
      raise(nvcompErrorCudaError, "the manager's stream failed");
    }
    f();
  }
};

PimplManager::PimplManager(std::unique_ptr<nvcompManagerInternalBase> p) noexcept : impl(std::move(p)) {}

CompressionConfig PimplManager::configure_compression(const size_t n) { return impl->configure_compression(n); }
std::vector<CompressionConfig> PimplManager::configure_compression(const std::vector<size_t>& sizes) {
  std::vector<CompressionConfig> v;
  for (size_t n : sizes) v.push_back(impl->configure_compression(n));
  return v;
}
void PimplManager::compress(const uint8_t* in, uint8_t* out, const CompressionConfig& c, size_t* comp_size) {
  impl->ordered([&] {
    const size_t n = impl->compress_one(in, out, c.uncompressed_buffer_size);
    if (comp_size) impl::put(comp_size, &n, sizeof n);
    impl->set_status(c.get_status(), nvcompSuccess);
  });
}
void PimplManager::compress(const uint8_t* const* in, uint8_t* const* out, const std::vector<CompressionConfig>& cs,
                            size_t* comp_sizes) {
  for (size_t i = 0; i < cs.size(); ++i) {
    const uint8_t* src = nullptr;
    uint8_t* dst = nullptr;
    impl::get(&src, in + i, sizeof src);
    impl::get(&dst, out + i, sizeof dst);
    compress(src, dst, cs[i], comp_sizes ? comp_sizes + i : nullptr);
  }
}
DecompressionConfig PimplManager::configure_decompression(const uint8_t* comp, const size_t* comp_size) {
  DecompressionConfig d;
  impl->ordered([&] { d = impl->configure_decompression(comp, comp_size); });
  return d;
}
std::vector<DecompressionConfig> PimplManager::configure_decompression(const uint8_t* const* comps, size_t n,
                                                                       const size_t* comp_sizes) {
  std::vector<DecompressionConfig> v;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* c = nullptr;
    impl::get(&c, comps + i, sizeof c);
    v.push_back(configure_decompression(c, comp_sizes ? comp_sizes + i : nullptr));
  }
  return v;
}
DecompressionConfig PimplManager::configure_decompression(const CompressionConfig& c) {
  DecompressionConfig d;
  d.decomp_data_size = c.uncompressed_buffer_size;
  d.num_chunks = c.num_chunks;
  d.impl->kind = impl->kind;
  d.impl->chunk_size = impl->chunk;
  return d;
}
std::vector<DecompressionConfig> PimplManager::configure_decompression(const std::vector<CompressionConfig>& cs) {
  std::vector<DecompressionConfig> v;
  for (const CompressionConfig& c : cs) v.push_back(configure_decompression(c));
  return v;
}
void PimplManager::decompress(uint8_t* out, const uint8_t* comp, const DecompressionConfig& d, size_t* comp_size) {
  impl->ordered([&] {
    // A configuration made from a CompressionConfig has not seen the buffer:
    // its layout is read now. The status goes to the caller's configuration.
    const DecompressionConfig* use = &d;
    DecompressionConfig fresh;
    if (impl->kind == BitstreamKind::NVCOMP_NATIVE && d.impl->offsets.empty() && d.num_chunks) {
      fresh = impl->configure_decompression(comp, comp_size);
      use = &fresh;
    } else if (impl->kind != BitstreamKind::NVCOMP_NATIVE && comp_size && d.impl->comp_size == 0) {
      size_t cs = 0;
      impl::get(&cs, comp_size, sizeof cs);
      d.impl->comp_size = cs;
    }
    impl->set_status(d.get_status(), impl->decompress_one(out, comp, *use, comp_size));
  });
}
void PimplManager::decompress(uint8_t* const* outs, const uint8_t* const* comps,
                              const std::vector<DecompressionConfig>& ds, const size_t* comp_sizes) {
  for (size_t i = 0; i < ds.size(); ++i) {
    uint8_t* o = nullptr;
    const uint8_t* c = nullptr;
    impl::get(&o, outs + i, sizeof o);
    impl::get(&c, comps + i, sizeof c);
    decompress(o, c, ds[i], comp_sizes ? const_cast<size_t*>(comp_sizes + i) : nullptr);
  }
}
void PimplManager::decompress(uint8_t* const* outs, const uint8_t* const* comps,
                              const std::vector<DecompressionConfig>& ds, const size_t* comp_sizes,
                              const size_t batch_count, const uint8_t* const*) {
  std::vector<DecompressionConfig> v(ds.begin(), ds.begin() + std::min(batch_count, ds.size()));
  decompress(outs, comps, v, comp_sizes);
}
// The simulator needs no scratch memory to allocate or give back.
void PimplManager::set_scratch_allocators(const AllocFn_t&, const DeAllocFn_t&) {}
void PimplManager::deallocate_gpu_mem() {}
size_t PimplManager::get_compressed_output_size(const uint8_t* comp) {
  size_t n = 0;
  impl->ordered([&] { n = impl->compressed_size(comp); });
  return n;
}
std::vector<size_t> PimplManager::get_compressed_output_size(const uint8_t* const* comps, size_t n) {
  std::vector<size_t> v;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* c = nullptr;
    impl::get(&c, comps + i, sizeof c);
    v.push_back(get_compressed_output_size(c));
  }
  return v;
}
size_t PimplManager::get_decompressed_output_size(const uint8_t* comp) {
  size_t n = 0;
  impl->ordered([&] { n = impl->decompressed_size(comp); });
  return n;
}
std::vector<size_t> PimplManager::get_decompressed_output_size(const uint8_t* const* comps, size_t n) {
  std::vector<size_t> v;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* c = nullptr;
    impl::get(&c, comps + i, sizeof c);
    v.push_back(get_decompressed_output_size(c));
  }
  return v;
}

}  // namespace detail

// ============================================================================
// The managers
// ============================================================================

namespace {

std::unique_ptr<detail::nvcompManagerInternalBase> make(Fmt fmt, nvcompFormatType_t format, int level,
                                                        std::vector<uint8_t> spec, size_t chunk,
                                                        cudaStream_t stream, ChecksumPolicy policy,
                                                        BitstreamKind kind) {
  if (kind != BitstreamKind::NVCOMP_NATIVE && policy != NoComputeNoVerify)
    raise(nvcompErrorNotSupported,
          "Only ChecksumPolicy::NoComputeNoVerify is allowed when using bitstream kind different than NVCOMP_NATIVE");
  if (policy == ComputeAndNoVerify || policy == ComputeAndVerifyIfPresent || policy == ComputeAndVerify)
    raise(nvcompErrorNotSupported,
          "VirtualGPU's nvCOMP cannot compute nvCOMP's container checksums: their algorithm is not public");
  // A chunk the format cannot bound: NVIDIA's message (measured).
  size_t max_chunk = 0;
  switch (fmt) {
    case Fmt::LZ4: case Fmt::Snappy: max_chunk = size_t(1) << 24; break;
    case Fmt::Deflate: case Fmt::Gdeflate: max_chunk = size_t(1) << 31; break;
    case Fmt::Gzip: max_chunk = size_t(0x7FFFFFFF) << 15; break;
    case Fmt::Zstd: max_chunk = (size_t(1) << 31) - 1; break;
  }
  if (chunk > max_chunk)
    raise(nvcompErrorChunkSizeTooLarge, "Could not determine the maximum compressed chunk size.");
  auto m = std::make_unique<detail::nvcompManagerInternalBase>();
  m->fmt = fmt;
  m->format = format;
  m->level = level;
  m->spec = std::move(spec);
  m->chunk = chunk;
  m->stream = stream;
  m->policy = policy;
  m->kind = kind;
  return m;
}

template <class T>
std::vector<uint8_t> bytes_of(const T& v) {
  std::vector<uint8_t> b(sizeof v);
  std::memcpy(b.data(), &v, sizeof v);
  return b;
}

[[noreturn]] void proprietary(const char* name) {
  raise(nvcompErrorNotSupported, std::string("VirtualGPU's nvCOMP does not implement ") + name +
                                     ": NVIDIA does not publish its bitstream");
}

}  // namespace

LZ4Manager::LZ4Manager(size_t chunk, const nvcompBatchedLZ4CompressOpts_t& co, const nvcompBatchedLZ4DecompressOpts_t&,
                       cudaStream_t stream, ChecksumPolicy policy, BitstreamKind kind)
    : PimplManager(nullptr) {
  if (co.bitshuffle_mode != NVCOMP_BITSHUFFLE_NONE) raise(nvcompErrorNotSupported, "LZ4's bitshuffle option");
  impl = make(Fmt::LZ4, nvcompFormatType_t::LZ4, 1, bytes_of(int32_t(co.data_type)), chunk, stream, policy, kind);
}
LZ4Manager::~LZ4Manager() noexcept = default;

SnappyManager::SnappyManager(size_t chunk, const nvcompBatchedSnappyCompressOpts_t&,
                             const nvcompBatchedSnappyDecompressOpts_t&, cudaStream_t stream, ChecksumPolicy policy,
                             BitstreamKind kind)
    : PimplManager(make(Fmt::Snappy, nvcompFormatType_t::Snappy, 1, {}, chunk, stream, policy, kind)) {}
SnappyManager::~SnappyManager() noexcept = default;

DeflateManager::DeflateManager(size_t chunk, const nvcompBatchedDeflateCompressOpts_t& co,
                               const nvcompBatchedDeflateDecompressOpts_t&, cudaStream_t stream,
                               ChecksumPolicy policy, BitstreamKind kind)
    : PimplManager(nullptr) {
  if (co.algorithm < 0 || co.algorithm > 5) raise(nvcompErrorNotSupported, "a Deflate algorithm outside 0..5");
  impl = make(Fmt::Deflate, nvcompFormatType_t::Deflate, co.algorithm, bytes_of(int32_t(co.algorithm)), chunk, stream,
              policy, kind);
}
DeflateManager::~DeflateManager() noexcept = default;

GdeflateManager::GdeflateManager(size_t chunk, const nvcompBatchedGdeflateCompressOpts_t& co,
                                 const nvcompBatchedGdeflateDecompressOpts_t&, cudaStream_t stream,
                                 ChecksumPolicy policy, BitstreamKind kind)
    : PimplManager(nullptr) {
  if (co.algorithm < 0 || co.algorithm > 5) raise(nvcompErrorNotSupported, "a GDeflate algorithm outside 0..5");
  impl = make(Fmt::Gdeflate, nvcompFormatType_t::GDeflate, co.algorithm, bytes_of(int32_t(co.algorithm)), chunk,
              stream, policy, kind);
}
GdeflateManager::~GdeflateManager() noexcept = default;

GzipManager::GzipManager(size_t chunk, const nvcompBatchedGzipCompressOpts_t& co,
                         const nvcompBatchedGzipDecompressOpts_t&, cudaStream_t stream, ChecksumPolicy policy,
                         BitstreamKind kind)
    : PimplManager(nullptr) {
  if (co.algorithm < 0 || co.algorithm > 5) raise(nvcompErrorNotSupported, "a Gzip algorithm outside 0..5");
  impl = make(Fmt::Gzip, nvcompFormatType_t::Gzip, co.algorithm, bytes_of(uint8_t(co.algorithm)), chunk, stream,
              policy, kind);
}
GzipManager::~GzipManager() noexcept = default;

ZstdManager::ZstdManager(size_t chunk, const nvcompBatchedZstdCompressOpts_t&, const nvcompBatchedZstdDecompressOpts_t&,
                         cudaStream_t stream, ChecksumPolicy policy, BitstreamKind kind)
    : PimplManager(make(Fmt::Zstd, nvcompFormatType_t::Zstd, 1, {}, chunk, stream, policy, kind)) {}
ZstdManager::~ZstdManager() noexcept = default;

CascadedManager::CascadedManager(size_t, const nvcompBatchedCascadedCompressOpts_t&,
                                 const nvcompBatchedCascadedDecompressOpts_t&, cudaStream_t, ChecksumPolicy,
                                 BitstreamKind)
    : PimplManager(nullptr) {
  proprietary("Cascaded");
}
CascadedManager::~CascadedManager() noexcept = default;
BitcompManager::BitcompManager(size_t, const nvcompBatchedBitcompCompressOpts_t&,
                               const nvcompBatchedBitcompDecompressOpts_t&, cudaStream_t, ChecksumPolicy, BitstreamKind)
    : PimplManager(nullptr) {
  proprietary("Bitcomp");
}
BitcompManager::~BitcompManager() noexcept = default;
ANSManager::ANSManager(size_t, const nvcompBatchedANSCompressOpts_t&, const nvcompBatchedANSDecompressOpts_t&,
                       cudaStream_t, ChecksumPolicy, BitstreamKind)
    : PimplManager(nullptr) {
  proprietary("ANS");
}
ANSManager::~ANSManager() noexcept = default;

// ============================================================================
// The factory
// ============================================================================

nvcompFormatType_t get_compression_format(const uint8_t* comp_buffer, cudaStream_t stream) {
  if (cudaStreamSynchronize(stream) != cudaSuccess) cudaGetLastError();
  uint8_t b[8] = {};
  if (!impl::get(b, comp_buffer, sizeof b)) raise(nvcompErrorCudaError, "the compressed buffer is not readable");
  // The format byte of the header, read whatever the buffer is: NVIDIA's
  // library answers the same way for a RAW buffer, which has no header.
  return static_cast<nvcompFormatType_t>(b[6]);
}

std::shared_ptr<nvcompManagerBase> create_manager(const uint8_t* comp_buffer, cudaStream_t stream,
                                                  ChecksumPolicy policy, nvcompDecompressBackend_t backend, bool) {
  if (cudaStreamSynchronize(stream) != cudaSuccess) cudaGetLastError();
  uint8_t h[kCommonSize + 8] = {};
  if (!impl::get(h, comp_buffer, sizeof h)) raise(nvcompErrorCudaError, "the compressed buffer is not readable");
  CommonHeader ch{};
  std::memcpy(&ch, h, sizeof ch);
  if (ch.magic != kMagic) raise(nvcompErrorInvalidValue, "the buffer is not an nvCOMP buffer (no nvCOMP header)");
  int32_t spec_i = 0;
  std::memcpy(&spec_i, h + kCommonSize, sizeof spec_i);
  const size_t chunk = ch.uncomp_chunk_size;
  switch (static_cast<nvcompFormatType_t>(ch.format)) {
    case nvcompFormatType_t::LZ4: {
      nvcompBatchedLZ4CompressOpts_t o = nvcompBatchedLZ4CompressDefaultOpts;
      o.data_type = static_cast<nvcompType_t>(spec_i);
      nvcompBatchedLZ4DecompressOpts_t d = nvcompBatchedLZ4DecompressDefaultOpts;
      d.backend = backend;
      return std::make_shared<LZ4Manager>(chunk, o, d, stream, policy);
    }
    case nvcompFormatType_t::Snappy: {
      nvcompBatchedSnappyDecompressOpts_t d = nvcompBatchedSnappyDecompressDefaultOpts;
      d.backend = backend;
      return std::make_shared<SnappyManager>(chunk, nvcompBatchedSnappyCompressDefaultOpts, d, stream, policy);
    }
    case nvcompFormatType_t::Deflate: {
      nvcompBatchedDeflateCompressOpts_t o = nvcompBatchedDeflateCompressDefaultOpts;
      o.algorithm = spec_i;
      return std::make_shared<DeflateManager>(chunk, o, nvcompBatchedDeflateDecompressDefaultOpts, stream, policy);
    }
    case nvcompFormatType_t::GDeflate: {
      nvcompBatchedGdeflateCompressOpts_t o = nvcompBatchedGdeflateCompressDefaultOpts;
      o.algorithm = spec_i;
      return std::make_shared<GdeflateManager>(chunk, o, nvcompBatchedGdeflateDecompressDefaultOpts, stream, policy);
    }
    case nvcompFormatType_t::Gzip: {
      nvcompBatchedGzipCompressOpts_t o = nvcompBatchedGzipCompressDefaultOpts;
      o.algorithm = h[kCommonSize];
      return std::make_shared<GzipManager>(chunk, o, nvcompBatchedGzipDecompressDefaultOpts, stream, policy);
    }
    case nvcompFormatType_t::Zstd:
      return std::make_shared<ZstdManager>(chunk, nvcompBatchedZstdCompressDefaultOpts,
                                           nvcompBatchedZstdDecompressDefaultOpts, stream, policy);
    case nvcompFormatType_t::Cascaded: proprietary("Cascaded");
    case nvcompFormatType_t::Bitcomp: proprietary("Bitcomp");
    case nvcompFormatType_t::ANS: proprietary("ANS");
    default: break;
  }
  raise(nvcompErrorNotSupported, "the buffer's format is not one nvCOMP knows");
}

}  // namespace nvcomp
