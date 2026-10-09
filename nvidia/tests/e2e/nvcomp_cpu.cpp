// nvCOMP's CPU library (libnvcomp_cpu.so.5): the GDeflate compressor and decompressor
// that run on the host (nvcomp/native/gdeflate_cpu.h). The chunk-size bound it reports,
// the limits it enforces and the exceptions it throws, round trips at every level, and
// the interoperation that matters: chunks the CPU compressor writes are read by the
// batched GPU GDeflate decompressor, and chunks that one writes are read by the CPU
// decompressor.
//
// The expectations are NVIDIA's: this program also runs against NVIDIA's libnvcomp 5.3 and
// libnvcomp_cpu 5.3 on an RTX 3060, and every size, status and message below is what they
// answered.
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "../../include/vgpu_nvcomp.h"

namespace gdeflate {
void compressCPUGetMaxOutputChunkSize(size_t max_uncompressed_chunk_bytes, size_t* max_compressed_chunk_bytes);
void compressCPU(const void* const* in_ptr, const size_t* in_bytes, const size_t max_uncompressed_chunk_bytes, size_t batch_size,
                 void* const* out_ptr, size_t* out_bytes, int level = 12);
void decompressCPU(const void* const* in_ptr, const size_t* in_bytes, size_t batch_size, void* const* out_ptr,
                   size_t* out_buffer_bytes, size_t* out_bytes);
}  // namespace gdeflate

static int failures = 0;
static void check(bool ok, const std::string& what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

using Bytes = std::vector<unsigned char>;

// What a call threw, or "" when it did not.
template <class F>
static std::string thrown(F f) {
  try {
    f();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

static Bytes sample(size_t n, int kind) {
  Bytes v(n);
  uint32_t x = (uint32_t)(n * 31 + kind);
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = kind == 0 ? (unsigned char)(x >> 24) : (kind == 1 ? (unsigned char)("abcabcabd"[i % 9]) : (unsigned char)((i / 7) % 5));
  }
  return v;
}

static Bytes cpu_compress(const Bytes& in, int level) {
  size_t bound = 0;
  gdeflate::compressCPUGetMaxOutputChunkSize(in.size(), &bound);
  Bytes out(bound ? bound : 1);
  const void* ip[1] = {in.data()};
  size_t isz[1] = {in.size()};
  void* op[1] = {out.data()};
  size_t osz[1] = {0};
  gdeflate::compressCPU(ip, isz, in.size() ? in.size() : 1, 1, op, osz, level);
  out.resize(osz[0]);
  return out;
}

static Bytes cpu_decompress(const Bytes& comp, size_t cap, std::string* error) {
  Bytes out(cap + 1);
  const void* ip[1] = {comp.data()};
  size_t isz[1] = {comp.size()};
  void* op[1] = {out.data()};
  size_t cap_in[1] = {cap}, produced[1] = {0};
  *error = thrown([&] { gdeflate::decompressCPU(ip, isz, 1, op, cap_in, produced); });
  out.resize(*error == "" ? produced[0] : 0);
  return out;
}

// The batched GPU GDeflate API, one chunk each way.
static bool gpu_decompress(const Bytes& comp, size_t cap, Bytes* out) {
  nvcompBatchedGdeflateDecompressOpts_t o = nvcompBatchedGdeflateDecompressDefaultOpts;
  size_t temp = 0;
  if (nvcompBatchedGdeflateDecompressGetTempSizeAsync(1, cap, o, &temp, cap) != nvcompSuccess) return false;
  void *din, *dout, *dtemp;
  cudaMalloc(&din, comp.size() + 8);
  cudaMalloc(&dout, cap + 8);
  cudaMalloc(&dtemp, temp + 8);
  cudaMemcpy(din, comp.data(), comp.size(), cudaMemcpyHostToDevice);
  const void* hin[1] = {din};
  void* hout[1] = {dout};
  size_t hsz[1] = {comp.size()}, hcap[1] = {cap}, hact[1] = {0};
  nvcompStatus_t hst[1] = {nvcompErrorInternal};
  const void** pin;
  void** pout;
  size_t *psz, *pcap, *pact;
  nvcompStatus_t* pst;
  cudaMalloc((void**)&pin, 8); cudaMalloc((void**)&pout, 8); cudaMalloc((void**)&psz, 8); cudaMalloc((void**)&pcap, 8);
  cudaMalloc((void**)&pact, 8); cudaMalloc((void**)&pst, sizeof(nvcompStatus_t));
  cudaMemcpy(pin, hin, 8, cudaMemcpyHostToDevice); cudaMemcpy(pout, hout, 8, cudaMemcpyHostToDevice);
  cudaMemcpy(psz, hsz, 8, cudaMemcpyHostToDevice); cudaMemcpy(pcap, hcap, 8, cudaMemcpyHostToDevice);
  const nvcompStatus_t st = nvcompBatchedGdeflateDecompressAsync(pin, psz, pcap, pact, 1, dtemp, temp, pout, o, pst, 0);
  cudaDeviceSynchronize();
  cudaMemcpy(hact, pact, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hst, pst, sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost);
  out->resize(hact[0]);
  if (hact[0]) cudaMemcpy(out->data(), dout, hact[0], cudaMemcpyDeviceToHost);
  cudaFree(din); cudaFree(dout); cudaFree(dtemp); cudaFree(pin); cudaFree(pout); cudaFree(psz); cudaFree(pcap); cudaFree(pact); cudaFree(pst);
  return st == nvcompSuccess && hst[0] == nvcompSuccess;
}

static bool gpu_compress(const Bytes& in, Bytes* out) {
  nvcompBatchedGdeflateCompressOpts_t o = nvcompBatchedGdeflateCompressDefaultOpts;
  size_t bound = 0, temp = 0;
  if (nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(in.size(), o, &bound) != nvcompSuccess) return false;
  if (nvcompBatchedGdeflateCompressGetTempSizeAsync(1, in.size(), o, &temp, in.size()) != nvcompSuccess) return false;
  void *din, *dout, *dtemp;
  cudaMalloc(&din, in.size() + 8);
  cudaMalloc(&dout, bound + 8);
  cudaMalloc(&dtemp, temp + 8);
  cudaMemcpy(din, in.data(), in.size(), cudaMemcpyHostToDevice);
  const void* hin[1] = {din};
  void* hout[1] = {dout};
  size_t hsz[1] = {in.size()}, hact[1] = {0};
  nvcompStatus_t hst[1] = {nvcompErrorInternal};
  const void** pin;
  void** pout;
  size_t *psz, *pact;
  nvcompStatus_t* pst;
  cudaMalloc((void**)&pin, 8); cudaMalloc((void**)&pout, 8); cudaMalloc((void**)&psz, 8); cudaMalloc((void**)&pact, 8);
  cudaMalloc((void**)&pst, sizeof(nvcompStatus_t));
  cudaMemcpy(pin, hin, 8, cudaMemcpyHostToDevice); cudaMemcpy(pout, hout, 8, cudaMemcpyHostToDevice);
  cudaMemcpy(psz, hsz, 8, cudaMemcpyHostToDevice);
  const nvcompStatus_t st = nvcompBatchedGdeflateCompressAsync(pin, psz, in.size(), 1, dtemp, temp, pout, pact, o, pst, 0);
  cudaDeviceSynchronize();
  cudaMemcpy(hact, pact, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hst, pst, sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost);
  out->resize(hact[0]);
  if (hact[0]) cudaMemcpy(out->data(), dout, hact[0], cudaMemcpyDeviceToHost);
  cudaFree(din); cudaFree(dout); cudaFree(dtemp); cudaFree(pin); cudaFree(pout); cudaFree(psz); cudaFree(pact); cudaFree(pst);
  return st == nvcompSuccess && hst[0] == nvcompSuccess;
}

int main() {
  // ---- the bound ----
  const size_t sizes[] = {0, 1, 100, 9999, 10000, 10001, 20001, 65535, 65536};
  bool bounds = true;
  for (size_t n : sizes) {
    size_t b = 12345;
    gdeflate::compressCPUGetMaxOutputChunkSize(n, &b);
    const size_t want = n == 0 ? 0 : 65678 + 5 * ((n - 1) / 10000);
    if (b != want) {
      bounds = false;
      std::printf("     size %zu -> %zu, wanted %zu\n", n, b, want);
    }
  }
  check(bounds, "the maximum compressed chunk size is 65678 plus 5 per started 10000 bytes, 0 for an empty chunk");
  size_t b = 0;
  const std::string too_big = thrown([&] { gdeflate::compressCPUGetMaxOutputChunkSize(65537, &b); });
  check(too_big.find("Maximum allowed chunk size for Gdeflate CPU is 64kB") != std::string::npos, "a chunk over 64 KiB throws: " + too_big);

  // ---- the limits ----
  {
    Bytes in = sample(1000, 1), out(70000);
    const void* ip[1] = {in.data()};
    size_t isz[1] = {in.size()};
    void* op[1] = {out.data()};
    size_t osz[1] = {0};
    const std::string low = thrown([&] { gdeflate::compressCPU(ip, isz, 1000, 1, op, osz, -1); });
    check(low.find("Compression level must be between 0 and 12, both inclusive") != std::string::npos, "level -1 throws: " + low);
    const std::string high = thrown([&] { gdeflate::compressCPU(ip, isz, 1000, 1, op, osz, 13); });
    check(high.find("Compression level must be between 0 and 12, both inclusive") != std::string::npos, "level 13 throws");
    const std::string big = thrown([&] { gdeflate::compressCPU(ip, isz, 65537, 1, op, osz, 6); });
    check(big.find("Maximum allowed chunk size for Gdeflate CPU is 64kB") != std::string::npos, "a maximum chunk size over 64 KiB throws");
    const std::string small = thrown([&] { gdeflate::compressCPU(ip, isz, 999, 1, op, osz, 6); });
    check(small.find("max_uncompressed_chunk_bytes cannot be lower than any single chunk size") != std::string::npos,
          "a chunk larger than the stated maximum throws: " + small);
  }

  // ---- round trips ----
  bool round = true;
  for (int level : {0, 1, 5, 9, 12})
    for (int kind = 0; kind < 3; ++kind)
      for (size_t n : {size_t(1), size_t(100), size_t(4096), size_t(65536)}) {
        const Bytes in = sample(n, kind);
        const Bytes c = cpu_compress(in, level);
        std::string err;
        const Bytes back = cpu_decompress(c, n, &err);
        if (!err.empty() || back != in) round = false;
      }
  check(round, "chunks round trip through the CPU compressor and decompressor at levels 0, 1, 5, 9 and 12");

  // ---- the CPU's chunks on the GPU, and the GPU's on the CPU ----
  bool cpu_to_gpu = true, gpu_to_cpu = true;
  for (int kind = 0; kind < 3; ++kind)
    for (size_t n : {size_t(100), size_t(5000), size_t(65536)}) {
      const Bytes in = sample(n, kind);
      for (int level : {0, 6, 12}) {
        Bytes back;
        if (!gpu_decompress(cpu_compress(in, level), n, &back) || back != in) cpu_to_gpu = false;
      }
      Bytes comp;
      std::string err;
      if (!gpu_compress(in, &comp) || cpu_decompress(comp, n, &err) != in || !err.empty()) gpu_to_cpu = false;
    }
  check(cpu_to_gpu, "chunks the CPU compressor writes decompress on the GPU API");
  check(gpu_to_cpu, "chunks the GPU API writes decompress on the CPU");

  // ---- corruption ----
  {
    std::string err;
    Bytes junk(300, 0x5a);
    cpu_decompress(junk, 2000, &err);
    check(err.find("Failed to decompress chunk") != std::string::npos, "a chunk that is not GDeflate throws: " + err);
  }
  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
