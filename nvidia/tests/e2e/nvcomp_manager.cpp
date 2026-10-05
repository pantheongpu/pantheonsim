// nvCOMP's high-level interface: a manager per standard format compressing a
// buffer into nvCOMP's container (NVCOMP_NATIVE), a bare bitstream (RAW) and
// one after its size (WITH_UNCOMPRESSED_SIZE); the configurations, the sizes,
// the factory; and containers written by NVIDIA's library and by the
// simulator's, each read by whichever library this is.
//
// The expectations are NVIDIA's: this program also runs against NVIDIA's
// libnvcomp 5.3 on an RTX 3060, where it reads the simulator's containers.
// The few answers that differ by design -- checksums, whose algorithm is not
// public, and the formats with no public bitstream -- are checked on the
// simulator only.
//
// NVCOMP_DUMP_DIR=dir writes this library's containers there, for
// nvcomp_manager_vectors.inc.
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../../include/vgpu_nvcomp.hpp"

using namespace nvcomp;
using Bytes = std::vector<unsigned char>;

static int failures = 0;
static void check(bool ok, const std::string& what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

#include "nvcomp_vectors.inc"
#include "nvcomp_manager_vectors.inc"

// VirtualGPU's runtime exports this; NVIDIA's does not.
static bool on_simulator() { return dlsym(RTLD_DEFAULT, "_Z22vgpu_device_allocationPKvPPvPm") != nullptr; }


static uint8_t* upload(const Bytes& b) {
  uint8_t* p = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&p), b.size() + 8);
  cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice);
  return p;
}
static Bytes download(const uint8_t* p, size_t n) {
  Bytes b(n);
  cudaDeviceSynchronize();
  cudaMemcpy(b.data(), p, n, cudaMemcpyDeviceToHost);
  return b;
}
// A configuration's status. NVIDIA's library leaves it unwritten after a
// success (an RTX 3060 reads back whatever was in the memory), so a success is
// only checked on the simulator, which writes it.
static nvcompStatus_t status_of(nvcompStatus_t* p) {
  if (!on_simulator()) return nvcompSuccess;
  nvcompStatus_t s = nvcompErrorInternal;
  cudaDeviceSynchronize();
  if (p) cudaMemcpy(&s, p, sizeof s, cudaMemcpyDeviceToHost);
  return s;
}
static size_t read_size(const size_t* p) {
  size_t n = 0;
  cudaDeviceSynchronize();
  cudaMemcpy(&n, p, sizeof n, cudaMemcpyDeviceToHost);
  return n;
}

// Compresses `in` with `m`; returns the compressed bytes.
static Bytes compress(nvcompManagerBase& m, const Bytes& in, CompressionConfig* cfg = nullptr) {
  uint8_t* d_in = upload(in);
  CompressionConfig c = m.configure_compression(in.size());
  uint8_t* d_out = nullptr;
  size_t* d_size = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_out), c.max_compressed_buffer_size + 8);
  cudaMalloc(reinterpret_cast<void**>(&d_size), sizeof(size_t));
  m.compress(d_in, d_out, c, d_size);
  const size_t n = read_size(d_size);
  Bytes out = download(d_out, n);
  if (cfg) *cfg = c;
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_size);
  return out;
}

// Decompresses `comp` with `m`: the bytes, and the status.
static Bytes decompress(nvcompManagerBase& m, const Bytes& comp, nvcompStatus_t* st = nullptr,
                        DecompressionConfig* cfg = nullptr) {
  uint8_t* d_comp = upload(comp);
  size_t* d_size = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_size), sizeof(size_t));
  const size_t n = comp.size();
  cudaMemcpy(d_size, &n, sizeof n, cudaMemcpyHostToDevice);
  DecompressionConfig d = m.configure_decompression(d_comp, d_size);
  uint8_t* d_out = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_out), d.decomp_data_size + 8);
  m.decompress(d_out, d_comp, d, d_size);
  Bytes out = download(d_out, d.decomp_data_size);
  if (st) *st = status_of(d.get_status());
  if (cfg) *cfg = d;
  cudaFree(d_comp);
  cudaFree(d_out);
  cudaFree(d_size);
  return out;
}

struct Case {
  const char* name;
  std::shared_ptr<nvcompManagerBase> native, raw, sized;
  size_t max_native;  // NVIDIA's max_compressed_buffer_size for 3000 bytes in 1024-byte chunks
  size_t prefix;      // WITH_UNCOMPRESSED_SIZE's size field
  const unsigned char* nv_container;
  size_t nv_len;
  const unsigned char* our_container;
  size_t our_len;
};

static void run(const Case& c, const char* dump_dir) {
  const std::string F = c.name;
  const Bytes in = nvcomp_vector_input();
  // ---- the native container ----
  CompressionConfig cc;
  const Bytes comp = compress(*c.native, in, &cc);
  check(cc.uncompressed_buffer_size == in.size() && cc.num_chunks == 3, F + ": three chunks of 1024 bytes");
  check(cc.max_compressed_buffer_size == c.max_native,
        F + ": the bound is NVIDIA's (" + std::to_string(cc.max_compressed_buffer_size) + ")");
  check(status_of(cc.get_status()) == nvcompSuccess, F + ": compression succeeds");
  check(comp.size() > 64 && comp.size() <= cc.max_compressed_buffer_size, F + ": within the bound");
  uint32_t magic = 0;
  std::memcpy(&magic, comp.data(), 4);
  check(magic == 0x52911326u, F + ": the container's magic");
  {
    uint8_t* d = upload(comp);
    check(c.native->get_compressed_output_size(d) == comp.size(), F + ": get_compressed_output_size");
    check(c.native->get_decompressed_output_size(d) == in.size(), F + ": get_decompressed_output_size");
    auto m = create_manager(d);
    DecompressionConfig dc = m->configure_decompression(d);
    check(dc.decomp_data_size == in.size() && dc.num_chunks == 3 && !dc.checksums_present,
          F + ": create_manager reads the container");
    cudaFree(d);
  }
  nvcompStatus_t st = nvcompErrorInternal;
  DecompressionConfig dcfg;
  check(decompress(*c.native, comp, &st, &dcfg) == in && st == nvcompSuccess, F + ": native round trip");
  if (dump_dir) {
    const std::string path = std::string(dump_dir) + "/our_" + c.name + ".bin";
    if (FILE* fp = std::fopen(path.c_str(), "wb")) {
      std::fwrite(comp.data(), 1, comp.size(), fp);
      std::fclose(fp);
    }
  }
  // ---- the bare bitstream, and with its size ----
  const Bytes raw = compress(*c.raw, in, &cc);
  check(cc.num_chunks == 1, F + ": RAW is one chunk");
  check(decompress(*c.raw, raw, &st) == in && st == nvcompSuccess, F + ": RAW round trip");
  {
    uint8_t* d = upload(raw);
    bool threw = false;
    try {
      c.raw->get_compressed_output_size(d);
    } catch (const NVCompException& e) {
      threw = e.get_error() == nvcompErrorNotSupported;
    }
    check(threw, F + ": get_compressed_output_size refuses a RAW buffer");
    cudaFree(d);
  }
  const Bytes sized = compress(*c.sized, in, &cc);
  uint64_t n = 0;
  std::memcpy(&n, sized.data(), c.prefix);
  check(n == in.size() && sized.size() == raw.size() + c.prefix &&
            std::memcmp(sized.data() + c.prefix, raw.data(), raw.size()) == 0,
        F + ": WITH_UNCOMPRESSED_SIZE is the size, then the RAW bitstream");
  check(decompress(*c.sized, sized, &st) == in && st == nvcompSuccess, F + ": WITH_UNCOMPRESSED_SIZE round trip");
  // ---- the other library's containers ----
  if (c.nv_len > 1) {
    const Bytes b(c.nv_container, c.nv_container + c.nv_len);
    check(decompress(*c.native, b, &st) == in && st == nvcompSuccess, F + ": reads NVIDIA's container");
  }
  if (c.our_len > 1) {
    const Bytes b(c.our_container, c.our_container + c.our_len);
    check(decompress(*c.native, b, &st) == in && st == nvcompSuccess, F + ": reads the simulator's container");
    uint8_t* d = upload(b);
    auto m = create_manager(d);
    check(m && get_compression_format(d) == static_cast<nvcompFormatType_t>(c.our_container[6]),
          F + ": the factory takes the simulator's container");
    cudaFree(d);
  }
}

template <class M, class CO, class DO>
static std::shared_ptr<M> mk(size_t chunk, CO co, DO dop, BitstreamKind k) {
  return std::make_shared<M>(chunk, co, dop, (cudaStream_t)0, NoComputeNoVerify, k);
}
#define CASE(NAME, M, F, MAXN, PREFIX, NV, OUR)                                                                    \
  Case {                                                                                                            \
    NAME, mk<M>(1024, nvcompBatched##F##CompressDefaultOpts, nvcompBatched##F##DecompressDefaultOpts,             \
                BitstreamKind::NVCOMP_NATIVE),                                                                      \
        mk<M>(65536, nvcompBatched##F##CompressDefaultOpts, nvcompBatched##F##DecompressDefaultOpts,               \
              BitstreamKind::RAW),                                                                                  \
        mk<M>(65536, nvcompBatched##F##CompressDefaultOpts, nvcompBatched##F##DecompressDefaultOpts,               \
              BitstreamKind::WITH_UNCOMPRESSED_SIZE),                                                               \
        MAXN, PREFIX, NV, sizeof NV, OUR, sizeof OUR                                                                \
  }

int main() {
  const char* dump = std::getenv("NVCOMP_DUMP_DIR");
  run(CASE("lz4", LZ4Manager, LZ4, 3216, 4, kNvManagerLz4, kOurManagerLz4), dump);
  run(CASE("snappy", SnappyManager, Snappy, 3816, 8, kNvManagerSnappy, kOurManagerSnappy), dump);
  run(CASE("deflate", DeflateManager, Deflate, 7152, 8, kNvManagerDeflate, kOurManagerDeflate), dump);
  run(CASE("gdeflate", GdeflateManager, Gdeflate, 7128, 8, kNvManagerGdeflate, kOurManagerGdeflate), dump);
  run(CASE("gzip", GzipManager, Gzip, 222648, 8, kNvManagerGzip, kOurManagerGzip), dump);
  run(CASE("zstd", ZstdManager, Zstd, 3240, 8, kNvManagerZstd, kOurManagerZstd), dump);

  const Bytes in = nvcomp_vector_input();
  if (dump && !on_simulator()) {  // a container with checksums, which only NVIDIA's library writes
    LZ4Manager m(1024, nvcompBatchedLZ4CompressDefaultOpts, nvcompBatchedLZ4DecompressDefaultOpts, 0,
                 ComputeAndVerify);
    const Bytes b = compress(m, in);
    if (FILE* fp = std::fopen((std::string(dump) + "/our_lz4_checksums.bin").c_str(), "wb")) {
      std::fwrite(b.data(), 1, b.size(), fp);
      std::fclose(fp);
    }
  }
  // A container NVIDIA's library wrote with checksums: read without checking
  // them by either library; asked to check them, NVIDIA's does and the
  // simulator's says it cannot (their algorithm is not public).
  {
    const Bytes b(kNvManagerLz4Checksums, kNvManagerLz4Checksums + sizeof kNvManagerLz4Checksums);
    if (b.size() > 1) {
    LZ4Manager plain(1024);
    nvcompStatus_t st = nvcompErrorInternal;
    DecompressionConfig d;
    check(decompress(plain, b, &st, &d) == in && st == nvcompSuccess && d.checksums_present,
          "a container with checksums, not checked");
    LZ4Manager verify(1024, nvcompBatchedLZ4CompressDefaultOpts, nvcompBatchedLZ4DecompressDefaultOpts, 0,
                      NoComputeAndVerifyIfPresent);
    const Bytes out = decompress(verify, b, &st);
    check(out == in, "a container with checksums, verified if present: the data");
    if (on_simulator())
      check(st == nvcompErrorCannotVerifyChecksums, "and the status says the checksums could not be verified");
    }
  }
  // Refusals both libraries make.
  {
    bool threw = false;
    try {
      LZ4Manager m(4096, nvcompBatchedLZ4CompressDefaultOpts, nvcompBatchedLZ4DecompressDefaultOpts, 0,
                   ComputeAndVerify, BitstreamKind::RAW);
    } catch (const NVCompException& e) {
      threw = e.get_error() == nvcompErrorNotSupported;
    }
    check(threw, "checksums with a RAW bitstream are refused");
    threw = false;
    try {
      LZ4Manager m(size_t(1) << 25);
    } catch (const NVCompException& e) {
      threw = e.get_error() == nvcompErrorChunkSizeTooLarge;
    }
    check(threw, "a chunk LZ4 cannot bound is refused");
  }
  // What only the simulator refuses: computing checksums, and the formats
  // with no public bitstream.
  if (on_simulator()) {
    bool threw = false;
    try {
      ZstdManager m(1024, nvcompBatchedZstdCompressDefaultOpts, nvcompBatchedZstdDecompressDefaultOpts, 0,
                    ComputeAndVerify);
    } catch (const NVCompException& e) {
      threw = e.get_error() == nvcompErrorNotSupported;
    }
    check(threw, "computing checksums is refused (their algorithm is not public)");
    threw = false;
    try {
      CascadedManager m(1024);
    } catch (const NVCompException& e) {
      threw = e.get_error() == nvcompErrorNotSupported;
    }
    check(threw, "a Cascaded manager is refused (no public bitstream)");
  }
  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
