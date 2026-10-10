// nvCOMP's low-level batched API: for LZ4, Snappy, Deflate, GDeflate, Gzip and
// Zstd, the alignment and size queries, a batch of chunks compressed,
// measured and decompressed, a buffer too small and a corrupt chunk, a
// compression captured into a CUDA graph; streams NVIDIA's library wrote and
// streams the simulator's wrote, decoded by whichever library this is; CRC32
// in one piece and in segments; and NVIDIA's three unpublished formats.
//
// The expectations are NVIDIA's: this program also runs against NVIDIA's
// libnvcomp 5.3 on an RTX 3060, and every status and size below is what that
// library answered. The stored streams (nvcomp_vectors.inc) make the check
// cross: on the card NVIDIA's library decodes the simulator's streams, on the
// simulator its codecs decode NVIDIA's.
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../include/vgpu_nvcomp.h"

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
static void check(bool ok, const std::string& what) { check(ok, what.c_str()); }
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

#include "nvcomp_vectors.inc"

using Bytes = std::vector<unsigned char>;

// VirtualGPU's runtime exports this; NVIDIA's does not.
static bool on_simulator() { return dlsym(RTLD_DEFAULT, "_Z22vgpu_device_allocationPKvPPvPm") != nullptr; }

template <class T>
static T* dev_alloc(size_t n) {
  T* p = nullptr;
  cudaMalloc((void**)&p, n * sizeof(T) + 8);
  return p;
}
template <class T>
static T* upload(const std::vector<T>& v) {
  T* p = dev_alloc<T>(v.size());
  if (!v.empty()) cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return p;
}
template <class T>
static std::vector<T> download(const T* p, size_t n) {
  std::vector<T> v(n);
  cudaDeviceSynchronize();
  if (n) cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

// Chunks of different kinds and sizes, the last empty.
static std::vector<Bytes> sample_chunks() {
  std::vector<Bytes> c;
  const Bytes text = nvcomp_vector_input();
  c.push_back(text);
  Bytes noise(5000);
  uint32_t x = 7;
  for (auto& b : noise) b = (unsigned char)((x = x * 1664525u + 1013904223u) >> 24);
  c.push_back(noise);
  c.push_back(Bytes(70000, 'z'));
  Bytes mixed;
  for (int i = 0; i < 20000; ++i) mixed.push_back((unsigned char)(i % 251 < 200 ? text[i % text.size()] : i));
  c.push_back(mixed);
  c.push_back(Bytes(1, 'q'));
  return c;
}

// One format's ten entry points, and what the test needs to know about it.
template <class CO, class DO>
struct Format {
  const char* name;
  CO copts;
  DO dopts;
  nvcompStatus_t (*comp_align)(CO, nvcompAlignmentRequirements_t*);
  nvcompStatus_t (*decomp_align)(DO, nvcompAlignmentRequirements_t*);
  nvcompStatus_t (*comp_temp)(size_t, size_t, CO, size_t*, size_t);
  nvcompStatus_t (*comp_temp_sync)(const void* const* const, const size_t* const, size_t, size_t, CO, size_t*,
                                   size_t, cudaStream_t);
  nvcompStatus_t (*max_out)(size_t, CO, size_t*);
  nvcompStatus_t (*compress)(const void* const*, const size_t*, size_t, size_t, void*, size_t, void* const*,
                             size_t*, CO, nvcompStatus_t*, cudaStream_t);
  nvcompStatus_t (*decomp_temp)(size_t, size_t, DO, size_t*, size_t);
  nvcompStatus_t (*decomp_temp_sync)(const void* const* const, const size_t* const, size_t, size_t, size_t*,
                                     size_t, DO, nvcompStatus_t*, cudaStream_t);
  nvcompStatus_t (*sizes)(const void* const*, const size_t*, size_t*, size_t, cudaStream_t);
  nvcompStatus_t (*decompress)(const void* const*, const size_t*, const size_t*, size_t*, size_t, void* const,
                               size_t, void* const*, DO, nvcompStatus_t*, cudaStream_t);
};

#define FORMAT(F, name, copts, dopts)                                                                            \
  Format<nvcompBatched##F##CompressOpts_t, nvcompBatched##F##DecompressOpts_t> {                                  \
    name, copts, dopts, nvcompBatched##F##CompressGetRequiredAlignments,                                         \
        nvcompBatched##F##DecompressGetRequiredAlignments, nvcompBatched##F##CompressGetTempSizeAsync,          \
        nvcompBatched##F##CompressGetTempSizeSync, nvcompBatched##F##CompressGetMaxOutputChunkSize,             \
        nvcompBatched##F##CompressAsync, nvcompBatched##F##DecompressGetTempSizeAsync,                           \
        nvcompBatched##F##DecompressGetTempSizeSync, nvcompBatched##F##GetDecompressSizeAsync,                  \
        nvcompBatched##F##DecompressAsync                                                                         \
  }

// A batch on the device: chunk pointers and sizes in device memory.
struct DevBatch {
  std::vector<void*> ptrs;
  void** d_ptrs = nullptr;
  size_t* d_sizes = nullptr;
  std::vector<size_t> sizes;
};

static DevBatch upload_batch(const std::vector<Bytes>& chunks) {
  DevBatch b;
  for (const Bytes& c : chunks) {
    b.ptrs.push_back(upload(c));
    b.sizes.push_back(c.size());
  }
  b.d_ptrs = upload(b.ptrs);
  b.d_sizes = upload(b.sizes);
  return b;
}

static DevBatch empty_batch(const std::vector<size_t>& caps) {
  DevBatch b;
  for (size_t c : caps) b.ptrs.push_back(dev_alloc<unsigned char>(c));
  b.sizes = caps;
  b.d_ptrs = upload(b.ptrs);
  b.d_sizes = upload(b.sizes);
  return b;
}

static std::vector<Bytes> download_batch(const DevBatch& b, const std::vector<size_t>& sizes) {
  std::vector<Bytes> out;
  for (size_t i = 0; i < sizes.size(); ++i) out.push_back(download((unsigned char*)b.ptrs[i], sizes[i]));
  return out;
}

template <class CO, class DO>
static std::vector<Bytes> compress_all(const Format<CO, DO>& f, const std::vector<Bytes>& chunks,
                                       std::vector<nvcompStatus_t>* st_out = nullptr) {
  const size_t n = chunks.size();
  size_t max_chunk = 0;
  for (const Bytes& c : chunks) max_chunk = std::max(max_chunk, c.size());
  size_t max_out = 0, temp = 0;
  f.max_out(max_chunk, f.copts, &max_out);
  f.comp_temp(n, max_chunk, f.copts, &temp, max_chunk * n);
  DevBatch in = upload_batch(chunks);
  DevBatch out = empty_batch(std::vector<size_t>(n, max_out));
  void* d_temp = dev_alloc<unsigned char>(temp);
  size_t* d_out_sizes = dev_alloc<size_t>(n);
  nvcompStatus_t* d_st = dev_alloc<nvcompStatus_t>(n);
  const nvcompStatus_t r = f.compress((const void* const*)in.d_ptrs, in.d_sizes, max_chunk, n, d_temp, temp,
                                      out.d_ptrs, d_out_sizes, f.copts, d_st, 0);
  check(r == nvcompSuccess, std::string(f.name) + ": compression launches");
  const std::vector<size_t> sizes = download(d_out_sizes, n);
  if (st_out) *st_out = download(d_st, n);
  return download_batch(out, sizes);
}

// Decompresses with the library; per chunk the status, the actual size, and
// the bytes.
template <class CO, class DO>
static std::vector<Bytes> decompress_all(const Format<CO, DO>& f, const std::vector<Bytes>& comp,
                                         const std::vector<size_t>& caps, std::vector<nvcompStatus_t>* st,
                                         std::vector<size_t>* actual) {
  const size_t n = comp.size();
  size_t max_cap = 0, temp = 0;
  for (size_t c : caps) max_cap = std::max(max_cap, c);
  f.decomp_temp(n, max_cap, f.dopts, &temp, max_cap * n);
  DevBatch in = upload_batch(comp);
  DevBatch out = empty_batch(caps);
  size_t* d_caps = upload(caps);
  size_t* d_actual = dev_alloc<size_t>(n);
  nvcompStatus_t* d_st = dev_alloc<nvcompStatus_t>(n);
  void* d_temp = dev_alloc<unsigned char>(temp);
  const nvcompStatus_t r = f.decompress((const void* const*)in.d_ptrs, in.d_sizes, d_caps, d_actual, n, d_temp,
                                        temp, out.d_ptrs, f.dopts, d_st, 0);
  check(r == nvcompSuccess, std::string(f.name) + ": decompression launches");
  *st = download(d_st, n);
  *actual = download(d_actual, n);
  std::vector<size_t> got(n);
  for (size_t i = 0; i < n; ++i) got[i] = std::min((*actual)[i], caps[i]);
  return download_batch(out, got);
}

template <class CO, class DO>
static std::vector<size_t> measure_all(const Format<CO, DO>& f, const std::vector<Bytes>& comp) {
  DevBatch in = upload_batch(comp);
  size_t* d_out = dev_alloc<size_t>(comp.size());
  check(f.sizes((const void* const*)in.d_ptrs, in.d_sizes, d_out, comp.size(), 0) == nvcompSuccess,
        std::string(f.name) + ": GetDecompressSizeAsync launches");
  return download(d_out, comp.size());
}

// What NVIDIA's library answers for the queries, per format: compression and
// decompression alignments (input, output, temp), and the maximum output of
// chunks of 0, 1000 and 65536 bytes.
struct Expect {
  const char* name;
  size_t ca[3], da[3];
  size_t max0, max1000, max65536;
};
static const Expect kExpect[] = {
    {"LZ4", {1, 1, 2}, {1, 1, 1}, 8, 1008, 65800},
    {"Snappy", {1, 1, 1}, {1, 1, 1}, 32, 1198, 76490},
    {"Deflate", {1, 8, 8}, {4, 1, 1}, 64, 2288, 148256},
    {"Gdeflate", {1, 4, 8}, {4, 1, 1}, 288, 2288, 131360},
    {"Gzip", {1, 8, 8}, {1, 1, 1}, 26, 74176, 148328},
    {"Zstd", {4, 1, 1}, {1, 1, 8}, 9, 1011, 65547},
};

template <class CO, class DO>
static void run_format(const Format<CO, DO>& f, const unsigned char* nv_stream, size_t nv_len,
                       const unsigned char* our_stream, size_t our_len) {
  const std::string F = f.name;
  // Queries.
  const Expect* e = nullptr;
  for (const Expect& x : kExpect)
    if (F.compare(0, std::strlen(x.name), x.name) == 0 && (F.size() == std::strlen(x.name) || F[std::strlen(x.name)] == ' '))
      e = &x;
  nvcompAlignmentRequirements_t a{};
  IS(f.comp_align(f.copts, &a), nvcompSuccess);
  check(e && a.input == e->ca[0] && a.output == e->ca[1] && a.temp == e->ca[2], F + ": compression alignments");
  IS(f.decomp_align(f.dopts, &a), nvcompSuccess);
  check(e && a.input == e->da[0] && a.output == e->da[1] && a.temp == e->da[2], F + ": decompression alignments");
  size_t m0 = 0, m1 = 0, m2 = 0;
  IS(f.max_out(0, f.copts, &m0), nvcompSuccess);
  IS(f.max_out(65536, f.copts, &m1), nvcompSuccess);
  IS(f.max_out(1000, f.copts, &m2), nvcompSuccess);
  check(e && m0 == e->max0 && m2 == e->max1000 && m1 == e->max65536, F + ": maximum output sizes");
  IS(f.max_out(size_t(1) << 47, f.copts, &m1), nvcompErrorChunkSizeTooLarge);
  size_t t1 = 0, t2 = 0;
  IS(f.comp_temp(4, 65536, f.copts, &t1, 4 * 65536), nvcompSuccess);
  IS(f.decomp_temp(4, 65536, f.dopts, &t2, 4 * 65536), nvcompSuccess);

  // A batch, round trip.
  const std::vector<Bytes> chunks = sample_chunks();
  std::vector<nvcompStatus_t> st;
  const std::vector<Bytes> comp = compress_all(f, chunks, &st);
  bool all_ok = true;
  for (nvcompStatus_t s : st) all_ok = all_ok && s == nvcompSuccess;
  check(all_ok, F + ": every chunk compresses");
  if (F.find("entropy") == std::string::npos)  // Huffman alone spends a bit a byte
    check(comp[2].size() < 5000, F + ": a run of one byte compresses");
  const std::vector<size_t> measured = measure_all(f, comp);
  bool sizes_ok = true;
  for (size_t i = 0; i < chunks.size(); ++i) sizes_ok = sizes_ok && measured[i] == chunks[i].size();
  check(sizes_ok, F + ": GetDecompressSizeAsync gives each chunk's size");
  std::vector<size_t> caps, actual;
  for (const Bytes& c : chunks) caps.push_back(c.size());
  std::vector<Bytes> back = decompress_all(f, comp, caps, &st, &actual);
  bool same = true;
  for (size_t i = 0; i < chunks.size(); ++i)
    same = same && st[i] == nvcompSuccess && actual[i] == chunks[i].size() && back[i] == chunks[i];
  check(same, F + ": every chunk decompresses to what was compressed");

  // A buffer one byte short, and a corrupt chunk. The documented answer for
  // the first is nvcompErrorCannotDecompress; on NVIDIA's library both are
  // undefined behaviour in practice (its LZ4 reports success for a short
  // buffer, writing past it, and a corrupt chunk faults the context, which
  // every later check would inherit), so only the simulator is asked.
  if (on_simulator()) {
    caps[0] -= 1;
    std::vector<Bytes> bad = comp;
    for (size_t k = 0; k < bad[3].size(); k += 7) bad[3][k] ^= 0x5a;
    back = decompress_all(f, bad, caps, &st, &actual);
    check(st[0] == nvcompErrorCannotDecompress, F + ": a buffer too small is nvcompErrorCannotDecompress");
    check(st[3] == nvcompErrorCannotDecompress, F + ": a corrupt chunk is nvcompErrorCannotDecompress");
    check(st[1] == nvcompSuccess && back[1] == chunks[1], F + ": its neighbours are unaffected");
  }


  // A compression captured into a graph runs when the graph is launched, on
  // what the kernels before it produced.
  {
    const std::vector<Bytes> one = {chunks[0]};
    size_t max_out = 0, temp = 0;
    f.max_out(one[0].size(), f.copts, &max_out);
    f.comp_temp(1, one[0].size(), f.copts, &temp, one[0].size());
    DevBatch in = upload_batch({Bytes(one[0].size(), 0)});
    DevBatch out = empty_batch({max_out});
    void* d_temp = dev_alloc<unsigned char>(temp);
    size_t* d_os = dev_alloc<size_t>(1);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(in.ptrs[0], one[0].data(), one[0].size(), cudaMemcpyHostToDevice, s);
    const nvcompStatus_t r = f.compress((const void* const*)in.d_ptrs, in.d_sizes, one[0].size(), 1, d_temp, temp,
                                        out.d_ptrs, d_os, f.copts, nullptr, s);
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, 0);
    cudaGraphLaunch(ge, s);
    cudaStreamSynchronize(s);
    check(r == nvcompSuccess, F + ": compression is captured");
    const size_t os = download(d_os, 1)[0];
    const std::vector<Bytes> c = {download((unsigned char*)out.ptrs[0], os)};
    std::vector<size_t> cp = {one[0].size()};
    const std::vector<Bytes> d = decompress_all(f, c, cp, &st, &actual);
    check(st[0] == nvcompSuccess && d[0] == one[0], F + ": the graph compressed the bytes copied before it");
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
  }

  // The stored streams: NVIDIA's and the simulator's.
  const Bytes input = nvcomp_vector_input();
  for (int k = 0; k < 2; ++k) {
    const unsigned char* p = k ? our_stream : nv_stream;
    const size_t len = k ? our_len : nv_len;
    if (!p) continue;
    const std::vector<Bytes> c = {Bytes(p, p + len)};
    std::vector<size_t> cp = {input.size()};
    const std::vector<Bytes> d = decompress_all(f, c, cp, &st, &actual);
    check(st[0] == nvcompSuccess && d[0] == input && measure_all(f, c)[0] == input.size(),
          F + (k ? ": decodes the simulator's stream" : ": decodes NVIDIA's stream"));
  }
}

// GDeflate's other block layouts, written by the simulator: NVIDIA's library
// decodes them on the card, which pins down the layout the draft leaves open.
static void gdeflate_layouts() {
  auto f = FORMAT(Gdeflate, "Gdeflate", nvcompBatchedGdeflateCompressDefaultOpts,
                  nvcompBatchedGdeflateDecompressDefaultOpts);
  const Bytes input = nvcomp_vector_input();
  for (const auto& v : kGdeflateLayouts) {
    const std::vector<Bytes> c = {Bytes(v.data, v.data + v.size)};
    std::vector<size_t> cp = {input.size()}, actual;
    std::vector<nvcompStatus_t> st;
    const std::vector<Bytes> d = decompress_all(f, c, cp, &st, &actual);
    check(st[0] == nvcompSuccess && d[0] == input, std::string("Gdeflate decodes ") + v.what);
  }
}

// Cascaded, Bitcomp and ANS: NVIDIA's library round-trips them; the
// simulator's refuses every entry point by name, as nothing public specifies
// their bitstreams.
template <class CO, class DO>
static void proprietary(const Format<CO, DO>& f) {
  const std::string F = f.name;
  nvcompAlignmentRequirements_t a{};
  const nvcompStatus_t r = f.comp_align(f.copts, &a);
  if (r == nvcompErrorNotSupported) {
    size_t x = 0;
    bool all = f.decomp_align(f.dopts, &a) == nvcompErrorNotSupported &&
               f.max_out(4096, f.copts, &x) == nvcompErrorNotSupported &&
               f.comp_temp(1, 4096, f.copts, &x, 4096) == nvcompErrorNotSupported &&
               f.decomp_temp(1, 4096, f.dopts, &x, 4096) == nvcompErrorNotSupported &&
               f.compress(nullptr, nullptr, 0, 1, nullptr, 0, nullptr, nullptr, f.copts, nullptr, 0) ==
                   nvcompErrorNotSupported &&
               f.sizes(nullptr, nullptr, nullptr, 1, 0) == nvcompErrorNotSupported &&
               f.decompress(nullptr, nullptr, nullptr, nullptr, 1, nullptr, 0, nullptr, f.dopts, nullptr, 0) ==
                   nvcompErrorNotSupported;
    check(all, F + ": refused by every entry point (no public bitstream)");
    return;
  }
  check(r == nvcompSuccess, F + ": supported by this library");
  std::vector<Bytes> chunks = {nvcomp_vector_input()};
  chunks[0].resize(2048);
  std::vector<nvcompStatus_t> st;
  const std::vector<Bytes> comp = compress_all(f, chunks, &st);
  std::vector<size_t> caps = {chunks[0].size()}, actual;
  const std::vector<Bytes> back = decompress_all(f, comp, caps, &st, &actual);
  check(st[0] == nvcompSuccess && back[0] == chunks[0], F + ": round trip");
}

static uint32_t crc_of(const nvcompCRC32Spec_t& spec, const std::vector<Bytes>& segments, size_t n_chunks) {
  // segments: the message split in pieces; each call carries n_chunks copies.
  uint32_t* d_crc = dev_alloc<uint32_t>(n_chunks);
  nvcompBatchedCRC32Opts_t o{};
  o.spec = spec;
  nvcompBatchedCRC32GetHeuristicConf(nullptr, n_chunks, &o.kernel_conf, 4096, 0);
  for (size_t k = 0; k < segments.size(); ++k) {
    DevBatch b = upload_batch(std::vector<Bytes>(n_chunks, segments[k]));
    const nvcompCRC32SegmentKind_t kind = segments.size() == 1 ? nvcompCRC32OnlySegment
                                          : k == 0             ? nvcompCRC32FirstSegment
                                          : k + 1 == segments.size() ? nvcompCRC32LastSegment
                                                                     : nvcompCRC32MidSegment;
    nvcompBatchedCRC32Async((const void* const*)b.d_ptrs, b.d_sizes, n_chunks, d_crc, o, kind, nullptr, 0);
  }
  const std::vector<uint32_t> v = download(d_crc, n_chunks);
  for (uint32_t c : v)
    if (c != v[0]) return 0xdeadbeef;
  return v[0];
}

static void crc32_checks() {
  const Bytes msg = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  const Bytes a(msg.begin(), msg.begin() + 2), b(msg.begin() + 2, msg.begin() + 5), c(msg.begin() + 5, msg.end());
  // The catalogue's check values for "123456789".
  check(crc_of(nvcompCRC32, {msg}, 3) == 0xCBF43926u, "CRC-32 of 123456789");
  check(crc_of(nvcompCRC32_C, {msg}, 1) == 0xE3069283u, "CRC-32C of 123456789");
  check(crc_of(nvcompCRC32_BZIP2, {msg}, 2) == 0xFC891918u, "CRC-32/BZIP2 of 123456789");
  check(crc_of(nvcompCRC32_POSIX, {msg}, 1) == 0x765E7680u, "CRC-32/POSIX of 123456789");
  check(crc_of(nvcompCRC32_MPEG_2, {msg}, 1) == 0x0376E6E7u, "CRC-32/MPEG-2 of 123456789");
  check(crc_of(nvcompCRC32, {a, b, c}, 2) == 0xCBF43926u, "CRC-32 over three segments");
  check(crc_of(nvcompCRC32_BZIP2, {a, c.empty() ? b : Bytes(msg.begin() + 2, msg.end())}, 1) == 0xFC891918u,
        "CRC-32/BZIP2 over two segments");
}

// Options outside their documented values are nvcompErrorNotSupported; an LZ4
// data type sets the input alignment.
static void option_checks() {
  nvcompBatchedLZ4CompressOpts_t l = nvcompBatchedLZ4CompressDefaultOpts;
  nvcompAlignmentRequirements_t a{};
  size_t m = 0;
  l.data_type = NVCOMP_TYPE_INT;
  IS(nvcompBatchedLZ4CompressGetRequiredAlignments(l, &a), nvcompSuccess);
  check(a.input == 4, "LZ4 on 4-byte integers wants 4-byte-aligned input");
  l.data_type = NVCOMP_TYPE_USHORT;
  IS(nvcompBatchedLZ4CompressGetRequiredAlignments(l, &a), nvcompSuccess);
  check(a.input == 2, "LZ4 on 2-byte integers wants 2-byte-aligned input");
  l.data_type = NVCOMP_TYPE_LONGLONG;
  IS(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(4096, l, &m), nvcompErrorNotSupported);
  l.data_type = NVCOMP_TYPE_FLOAT16;
  IS(nvcompBatchedLZ4CompressGetRequiredAlignments(l, &a), nvcompErrorNotSupported);
  nvcompBatchedDeflateCompressOpts_t d = nvcompBatchedDeflateCompressDefaultOpts;
  d.algorithm = 6;
  IS(nvcompBatchedDeflateCompressGetMaxOutputChunkSize(4096, d, &m), nvcompErrorNotSupported);
  d.algorithm = -1;
  IS(nvcompBatchedDeflateCompressGetTempSizeAsync(1, 4096, d, &m, 4096), nvcompErrorNotSupported);
  nvcompBatchedZstdCompressOpts_t z = nvcompBatchedZstdCompressDefaultOpts;
  z.reserved[3] = 1;  // not checked
  IS(nvcompBatchedZstdCompressGetMaxOutputChunkSize(1000, z, &m), nvcompSuccess);
  check(std::string(nvcompGetStatusString(nvcompErrorChunkSizeTooLarge)) ==
            "The chunk size exceeds the maximum allowed size",
        "nvcompGetStatusString describes a status");
}


// ---- LZ4 bitshuffle ----------------------------------------------------------------
//
// nvcomp/lz4.h documents the option (sub-chunks of 8 KiB; whole groups of eight
// elements only) but not the layout, which is what the card wrote: for each bit
// plane one byte per group of eight elements, element 0 in the byte's low bit; planes
// from the element's top bit down (MSB-first, mode 1) or from bit 0 up (LSB-first). This
// reference is written bit by bit from that description, apart from the code under test.

static Bytes ref_bitshuffle(const Bytes& in, int es, bool msb_first) {
  Bytes out;
  for (size_t off = 0; off < in.size(); off += 8192) {
    const size_t len = std::min<size_t>(8192, in.size() - off);
    const size_t groups = len / es / 8;
    for (int plane = 0; plane < 8 * es; ++plane)
      for (size_t g = 0; g < groups; ++g) {
        unsigned char v = 0;
        for (int e = 0; e < 8; ++e) {
          const int bit_of_element = msb_first ? 8 * es - 1 - plane : plane;
          const unsigned char byte = in[off + (g * 8 + e) * es + bit_of_element / 8];
          v |= (unsigned char)(((byte >> (bit_of_element % 8)) & 1) << e);
        }
        out.push_back(v);
      }
    out.insert(out.end(), in.begin() + off + groups * 8 * es, in.begin() + off + len);
  }
  return out;
}

static Bytes lz4_block_decode(const Bytes& src) {
  Bytes out;
  size_t i = 0;
  while (i < src.size()) {
    const unsigned tok = src[i++];
    size_t lit = tok >> 4;
    if (lit == 15)
      for (;;) {
        const unsigned b = src[i++];
        lit += b;
        if (b != 255) break;
      }
    out.insert(out.end(), src.begin() + i, src.begin() + i + lit);
    i += lit;
    if (i >= src.size()) break;
    const size_t off = src[i] | (size_t)src[i + 1] << 8;
    i += 2;
    size_t ml = tok & 15;
    if (ml == 15)
      for (;;) {
        const unsigned b = src[i++];
        ml += b;
        if (b != 255) break;
      }
    ml += 4;
    for (size_t k = 0; k < ml; ++k) out.push_back(out[out.size() - off]);
  }
  return out;
}

// One chunk through the batched LZ4 API.
static nvcompStatus_t lz4_compress_one(const nvcompBatchedLZ4CompressOpts_t& o, const Bytes& in, Bytes* out) {
  size_t max_out = 0, temp = 0;
  nvcompStatus_t st = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(in.size(), o, &max_out);
  if (st != nvcompSuccess) return st;
  st = nvcompBatchedLZ4CompressGetTempSizeAsync(1, in.size(), o, &temp, in.size());
  if (st != nvcompSuccess) return st;
  unsigned char* din = upload(in);
  unsigned char* dout = dev_alloc<unsigned char>(max_out);
  void* dtemp = dev_alloc<unsigned char>(temp);
  const void* hin[1] = {din};
  void* hout[1] = {dout};
  size_t hsz[1] = {in.size()};
  const void** pin = upload(std::vector<const void*>{hin[0]});
  void** pout = upload(std::vector<void*>{hout[0]});
  size_t* psz = upload(std::vector<size_t>{hsz[0]});
  size_t* pcomp = dev_alloc<size_t>(1);
  nvcompStatus_t* pst = dev_alloc<nvcompStatus_t>(1);
  st = nvcompBatchedLZ4CompressAsync(pin, psz, in.size(), 1, dtemp, temp, pout, pcomp, o, pst, 0);
  const size_t n = download(pcomp, 1)[0];
  const nvcompStatus_t chunk = download(pst, 1)[0];
  if (st == nvcompSuccess && chunk == nvcompSuccess) *out = download(dout, n);
  cudaFree(din); cudaFree(dout); cudaFree(dtemp); cudaFree(pin); cudaFree(pout); cudaFree(psz); cudaFree(pcomp); cudaFree(pst);
  return st != nvcompSuccess ? st : chunk;
}

static nvcompStatus_t lz4_decompress_one(const nvcompBatchedLZ4DecompressOpts_t& o, const Bytes& comp, size_t cap, Bytes* out) {
  size_t temp = 0;
  nvcompStatus_t st = nvcompBatchedLZ4DecompressGetTempSizeAsync(1, cap, o, &temp, cap);
  if (st != nvcompSuccess) return st;
  unsigned char* din = upload(comp);
  unsigned char* dout = dev_alloc<unsigned char>(cap);
  void* dtemp = dev_alloc<unsigned char>(temp);
  const void** pin = upload(std::vector<const void*>{din});
  void** pout = upload(std::vector<void*>{dout});
  size_t* psz = upload(std::vector<size_t>{comp.size()});
  size_t* pcap = upload(std::vector<size_t>{cap});
  size_t* pact = dev_alloc<size_t>(1);
  nvcompStatus_t* pst = dev_alloc<nvcompStatus_t>(1);
  st = nvcompBatchedLZ4DecompressAsync(pin, psz, pcap, pact, 1, dtemp, temp, pout, o, pst, 0);
  const size_t n = download(pact, 1)[0];
  const nvcompStatus_t chunk = download(pst, 1)[0];
  if (st == nvcompSuccess && chunk == nvcompSuccess) *out = download(dout, n);
  cudaFree(din); cudaFree(dout); cudaFree(dtemp); cudaFree(pin); cudaFree(pout); cudaFree(psz); cudaFree(pcap); cudaFree(pact); cudaFree(pst);
  return st != nvcompSuccess ? st : chunk;
}

static void bitshuffle_checks() {
  struct T {
    nvcompType_t t;
    int es;
    const char* name;
  } types[] = {{NVCOMP_TYPE_CHAR, 1, "char"},   {NVCOMP_TYPE_UCHAR, 1, "uchar"}, {NVCOMP_TYPE_SHORT, 2, "short"}, {NVCOMP_TYPE_USHORT, 2, "ushort"},
               {NVCOMP_TYPE_INT, 4, "int"},     {NVCOMP_TYPE_UINT, 4, "uint"},   {NVCOMP_TYPE_BITS, 1, "bits"}};
  const size_t sizes[] = {8, 61, 64, 100, 8216, 16384};
  bool all_layouts = true, all_round = true;
  for (const T& ty : types)
    for (int mode : {1, 2, 3}) {  // 3: a value with no name, which the card treats as LSB-first
      for (size_t n : sizes) {
        Bytes in(n);
        uint32_t x = (uint32_t)(n * 977 + ty.es);
        for (auto& b : in) b = (unsigned char)((x = x * 1664525u + 1013904223u) >> 24);
        nvcompBatchedLZ4CompressOpts_t co = nvcompBatchedLZ4CompressDefaultOpts;
        co.data_type = ty.t;
        std::memcpy(&co.bitshuffle_mode, &mode, sizeof mode);
        Bytes comp;
        if (lz4_compress_one(co, in, &comp) != nvcompSuccess) {
          all_layouts = false;
          continue;
        }
        if (lz4_block_decode(comp) != ref_bitshuffle(in, ty.es, mode == 1)) all_layouts = false;
        nvcompBatchedLZ4DecompressOpts_t dopt = nvcompBatchedLZ4DecompressDefaultOpts;
        dopt.data_type = ty.t;
        dopt.bitshuffle_mode = mode == 1 ? NVCOMP_BITSHUFFLE_MSB_FIRST : NVCOMP_BITSHUFFLE_LSB_FIRST;
        Bytes back;
        if (lz4_decompress_one(dopt, comp, n, &back) != nvcompSuccess || back != in) all_round = false;
      }
    }
  check(all_layouts, "LZ4 bitshuffle chunks decode to the documented bit-plane layout (every type, mode and size)");
  check(all_round, "LZ4 bitshuffle chunks decompress back to their input");

  // Options that do not match the compression: the data comes back as the options say.
  Bytes in(100);
  uint32_t x = 99;
  for (auto& b : in) b = (unsigned char)((x = x * 1664525u + 1013904223u) >> 24);
  nvcompBatchedLZ4CompressOpts_t co = nvcompBatchedLZ4CompressDefaultOpts;
  co.data_type = NVCOMP_TYPE_SHORT;
  co.bitshuffle_mode = NVCOMP_BITSHUFFLE_MSB_FIRST;
  Bytes comp, got;
  IS(lz4_compress_one(co, in, &comp), nvcompSuccess);
  const Bytes shuffled = ref_bitshuffle(in, 2, true);
  const struct {
    nvcompType_t t;
    int mode;
    const char* what;
  } mismatches[] = {{NVCOMP_TYPE_INT, 1, "a wider type"}, {NVCOMP_TYPE_SHORT, 2, "the other order"}, {NVCOMP_TYPE_SHORT, 0, "no bitshuffle"}, {NVCOMP_TYPE_SHORT, 3, "a mode with no name"}};
  for (const auto& m : mismatches) {
    nvcompBatchedLZ4DecompressOpts_t d = nvcompBatchedLZ4DecompressDefaultOpts;
    d.data_type = m.t;
    std::memcpy(&d.bitshuffle_mode, &m.mode, sizeof m.mode);
    got.clear();
    const nvcompStatus_t st = lz4_decompress_one(d, comp, 100, &got);
    // Mode 0 and 3 leave the shuffled bytes as they are; 1 and 2 unshuffle by the type.
    Bytes want = (m.mode == 1 || m.mode == 2) ? Bytes() : shuffled;
    if (m.mode == 1 || m.mode == 2) {
      // The inverse of ref_bitshuffle, by search over planes: shuffling `got` must give back `shuffled`.
      want = got;
    }
    const bool ok = st == nvcompSuccess && got.size() == 100 &&
                    ((m.mode == 1 || m.mode == 2) ? ref_bitshuffle(got, m.t == NVCOMP_TYPE_INT ? 4 : 2, m.mode == 1) == shuffled
                                                  : got == shuffled);
    check(ok, std::string("decompressing with ") + m.what + " gives what the options say");
  }

  // Queries: the temporary space grows with the chunk by 32-byte units, the alignments by the type.
  nvcompBatchedLZ4CompressOpts_t q = nvcompBatchedLZ4CompressDefaultOpts;
  size_t t0 = 0, t1 = 0, t2 = 0;
  IS(nvcompBatchedLZ4CompressGetTempSizeAsync(3, 100, q, &t0, 300), nvcompSuccess);
  q.bitshuffle_mode = NVCOMP_BITSHUFFLE_LSB_FIRST;
  IS(nvcompBatchedLZ4CompressGetTempSizeAsync(3, 100, q, &t1, 300), nvcompSuccess);
  q.bitshuffle_mode = NVCOMP_BITSHUFFLE_MSB_FIRST;
  IS(nvcompBatchedLZ4CompressGetTempSizeAsync(2, 8216, q, &t2, 16432), nvcompSuccess);
  check(t0 == 3 * 32768 && t1 == 3 * (32768 + 128) && t2 == 2 * (32768 + 8224), "bitshuffle adds the chunk, to 32 bytes, to the temporary space");
  size_t mo = 0;
  IS(nvcompBatchedLZ4CompressGetMaxOutputChunkSize(64, q, &mo), nvcompSuccess);
  check(mo == 72, "bitshuffle leaves the maximum output size alone");
  nvcompBatchedLZ4DecompressOpts_t dq = nvcompBatchedLZ4DecompressDefaultOpts;
  nvcompAlignmentRequirements_t al{};
  dq.data_type = NVCOMP_TYPE_INT;
  dq.bitshuffle_mode = NVCOMP_BITSHUFFLE_MSB_FIRST;
  IS(nvcompBatchedLZ4DecompressGetRequiredAlignments(dq, &al), nvcompSuccess);
  check(al.output == 4, "decompressing with bitshuffle wants the output aligned to the type");
  dq.bitshuffle_mode = NVCOMP_BITSHUFFLE_NONE;
  IS(nvcompBatchedLZ4DecompressGetRequiredAlignments(dq, &al), nvcompSuccess);
  check(al.output == 1, "... and without it, to a byte");
  // Eight-byte integers: no bitshuffle for them, at the compress queries and the decompress call.
  q.data_type = NVCOMP_TYPE_LONGLONG;
  IS(nvcompBatchedLZ4CompressGetTempSizeAsync(1, 100, q, &t0, 100), nvcompErrorNotSupported);
  dq.data_type = NVCOMP_TYPE_LONGLONG;
  dq.bitshuffle_mode = NVCOMP_BITSHUFFLE_MSB_FIRST;
  IS(nvcompBatchedLZ4DecompressGetTempSizeAsync(1, 100, dq, &t0, 100), nvcompSuccess);
  got.clear();
  IS(lz4_decompress_one(dq, comp, 100, &got), nvcompErrorNotSupported);
}

int main() {
  nvcompProperties_t p{};
  IS(nvcompGetProperties(&p), nvcompSuccess);
  check(p.version == 5300, "nvCOMP 5.3");
  IS(nvcompGetProperties(nullptr), nvcompErrorInvalidValue);
  check(nvcompGetStatusString(nvcompErrorCannotDecompress) != nullptr, "status strings");

  run_format(FORMAT(LZ4, "LZ4", nvcompBatchedLZ4CompressDefaultOpts, nvcompBatchedLZ4DecompressDefaultOpts),
             kNvLz4, sizeof kNvLz4, kOurLz4, sizeof kOurLz4);
  run_format(FORMAT(Snappy, "Snappy", nvcompBatchedSnappyCompressDefaultOpts,
                    nvcompBatchedSnappyDecompressDefaultOpts),
             kNvSnappy, sizeof kNvSnappy, kOurSnappy, sizeof kOurSnappy);
  run_format(FORMAT(Deflate, "Deflate", nvcompBatchedDeflateCompressDefaultOpts,
                    nvcompBatchedDeflateDecompressDefaultOpts),
             kNvDeflate, sizeof kNvDeflate, kOurDeflate, sizeof kOurDeflate);
  nvcompBatchedDeflateCompressOpts_t entropy = nvcompBatchedDeflateCompressDefaultOpts;
  entropy.algorithm = 0;
  run_format(FORMAT(Deflate, "Deflate (entropy only)", entropy, nvcompBatchedDeflateDecompressDefaultOpts),
             nullptr, 0, kOurDeflate0, sizeof kOurDeflate0);
  run_format(FORMAT(Gdeflate, "Gdeflate", nvcompBatchedGdeflateCompressDefaultOpts,
                    nvcompBatchedGdeflateDecompressDefaultOpts),
             kNvGdeflate, sizeof kNvGdeflate, kOurGdeflate, sizeof kOurGdeflate);
  nvcompBatchedGdeflateCompressOpts_t g5 = nvcompBatchedGdeflateCompressDefaultOpts;
  g5.algorithm = 5;
  run_format(FORMAT(Gdeflate, "Gdeflate (algorithm 5)", g5, nvcompBatchedGdeflateDecompressDefaultOpts),
             kNvGdeflate5, sizeof kNvGdeflate5, kOurGdeflate0, sizeof kOurGdeflate0);
  run_format(FORMAT(Gzip, "Gzip", nvcompBatchedGzipCompressDefaultOpts, nvcompBatchedGzipDecompressDefaultOpts),
             kNvGzip, sizeof kNvGzip, kOurGzip, sizeof kOurGzip);
  run_format(FORMAT(Zstd, "Zstd", nvcompBatchedZstdCompressDefaultOpts, nvcompBatchedZstdDecompressDefaultOpts),
             kNvZstd, sizeof kNvZstd, kOurZstd, sizeof kOurZstd);
  gdeflate_layouts();

  proprietary(FORMAT(Cascaded, "Cascaded", nvcompBatchedCascadedCompressDefaultOpts,
                     nvcompBatchedCascadedDecompressDefaultOpts));
  proprietary(FORMAT(Bitcomp, "Bitcomp", nvcompBatchedBitcompCompressDefaultOpts,
                     nvcompBatchedBitcompDecompressDefaultOpts));
  proprietary(FORMAT(ANS, "ANS", nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts));

  crc32_checks();
  option_checks();
  bitshuffle_checks();

  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
