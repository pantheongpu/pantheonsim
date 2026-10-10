// nvCOMP's streaming gzip (nvcomp/native/streaming_gzip.hpp): gzip from one C++ stream
// to another. The workspace queries, a stream compressed to one gzip member that the
// batched Gzip API reads back, the statuses for a missing input and an unwritable output
// and a workspace too small, and decompression, which on an RTX 3060 (no hardware
// decompression engine) is nvcompErrorInvalidValue for everything.
//
// The expectations are NVIDIA's: this program also runs against NVIDIA's libnvcomp 5.3 on
// an RTX 3060, and every status and size below is what that library answered.
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../include/vgpu_nvcomp.h"
#include "../../include/vgpu_nvcomp.hpp"

static int failures = 0;
static void check(bool ok, const std::string& what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

using Bytes = std::vector<unsigned char>;

// VirtualGPU's runtime exports this; NVIDIA's does not.
static bool on_simulator() { return dlsym(RTLD_DEFAULT, "_Z22vgpu_device_allocationPKvPPvPm") != nullptr; }

// A stream compressed, with the workspace the library asks for.
static nvcompStatus_t compress(const std::string& in, int algorithm, std::string* out, size_t temp_override = 0, bool use_override = false) {
  nvcompBatchedGzipCompressOpts_t o = nvcompBatchedGzipCompressDefaultOpts;
  o.algorithm = algorithm;
  size_t need = 0;
  nvcompStatus_t st = nvcompGzipStreamingCompressGetTempSize(o, &need);
  if (st != nvcompSuccess) return st;
  const size_t temp = use_override ? temp_override : need;
  // NVIDIA's library works in this memory (gigabytes of it); the simulator needs none.
  void* d = nullptr;
  if (cudaMalloc(&d, on_simulator() || temp < need ? 16 : temp) != cudaSuccess) return nvcompErrorCudaError;
  std::istringstream is(in);
  std::ostringstream os;
  st = nvcompGzipStreamingCompress(is, os, temp, d, o, 0);
  cudaDeviceSynchronize();
  cudaFree(d);
  *out = os.str();
  return st;
}

// A gzip stream read by the batched Gzip decompressor as one chunk.
static bool batched_gunzip(const std::string& gz, size_t expect, std::string* back) {
  Bytes comp(gz.begin(), gz.end());
  void *din = nullptr, *dout = nullptr, *dtemp = nullptr;
  cudaMalloc(&din, comp.size() + 8);
  cudaMalloc(&dout, expect + 8);
  cudaMemcpy(din, comp.data(), comp.size(), cudaMemcpyHostToDevice);
  nvcompBatchedGzipDecompressOpts_t o = nvcompBatchedGzipDecompressDefaultOpts;
  size_t temp = 0;
  nvcompBatchedGzipDecompressGetTempSizeAsync(1, expect, o, &temp, expect);
  cudaMalloc(&dtemp, temp + 8);
  const void* hin[1] = {din};
  void* hout[1] = {dout};
  size_t hsz[1] = {comp.size()}, hcap[1] = {expect + 8}, hact[1] = {0};
  nvcompStatus_t hst[1] = {nvcompErrorInternal};
  const void** pin;
  void** pout;
  size_t *psz, *pcap, *pact;
  nvcompStatus_t* pst;
  cudaMalloc((void**)&pin, 8); cudaMalloc((void**)&pout, 8); cudaMalloc((void**)&psz, 8); cudaMalloc((void**)&pcap, 8);
  cudaMalloc((void**)&pact, 8); cudaMalloc((void**)&pst, sizeof(nvcompStatus_t));
  cudaMemcpy(pin, hin, 8, cudaMemcpyHostToDevice); cudaMemcpy(pout, hout, 8, cudaMemcpyHostToDevice);
  cudaMemcpy(psz, hsz, 8, cudaMemcpyHostToDevice); cudaMemcpy(pcap, hcap, 8, cudaMemcpyHostToDevice);
  const nvcompStatus_t st = nvcompBatchedGzipDecompressAsync(pin, psz, pcap, pact, 1, dtemp, temp, pout, o, pst, 0);
  cudaDeviceSynchronize();
  cudaMemcpy(hact, pact, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(hst, pst, sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost);
  back->assign(hact[0], '\0');
  if (hact[0]) cudaMemcpy(&(*back)[0], dout, hact[0], cudaMemcpyDeviceToHost);
  cudaFree(din); cudaFree(dout); cudaFree(dtemp); cudaFree(pin); cudaFree(pout); cudaFree(psz); cudaFree(pcap); cudaFree(pact); cudaFree(pst);
  return st == nvcompSuccess && hst[0] == nvcompSuccess;
}

int main() {
  cudaFree(nullptr);
  // ---- workspace sizes ----
  const size_t expected[6] = {876299944u, 2487174824u, 4903093928ull, 4903093928ull, 5440293544ull, 5708729000ull};
  for (int a = 0; a < 6; ++a) {
    nvcompBatchedGzipCompressOpts_t o = nvcompBatchedGzipCompressDefaultOpts;
    o.algorithm = a;
    size_t t = 0;
    IS(nvcompGzipStreamingCompressGetTempSize(o, &t), nvcompSuccess);
    check(t == expected[a], "compress workspace of algorithm " + std::to_string(a) + " is " + std::to_string(expected[a]));
  }
  {
    nvcompBatchedGzipCompressOpts_t o = nvcompBatchedGzipCompressDefaultOpts;
    size_t t = 77;
    o.algorithm = 6;
    IS(nvcompGzipStreamingCompressGetTempSize(o, &t), nvcompErrorInternal);
    o.algorithm = -1;
    IS(nvcompGzipStreamingCompressGetTempSize(o, &t), nvcompErrorInternal);
    o.algorithm = 0;
    IS(nvcompGzipStreamingCompressGetTempSize(o, nullptr), nvcompErrorInvalidValue);
    size_t d = 77;
    IS(nvcompGzipStreamingDecompressGetTempSize(&d), nvcompErrorInvalidValue);
    check(d == 77, "the failed decompress workspace query leaves the size alone");
    IS(nvcompGzipStreamingDecompressGetTempSize(nullptr), nvcompErrorInvalidValue);
  }

  // ---- compression ----
  std::string empty, one = "q", hundred, k64, k65, repetitive, noise;
  for (int i = 0; i < 100; ++i) hundred.push_back((char)("the quick brown fox jumps over the lazy dog"[i % 43]));
  for (size_t i = 0; i < 65537; ++i) k65.push_back((char)((i * 7 + i / 13) & 0xff));
  k64 = k65.substr(0, 65536);
  for (size_t i = 0; i < (1u << 20); ++i) repetitive.push_back("abcdefgh0123456789"[(i / 3) % 18]);
  uint32_t x = 5;
  for (size_t i = 0; i < (1u << 20); ++i) noise.push_back((char)((x = x * 1664525u + 1013904223u) >> 24));
  const struct {
    const char* name;
    const std::string* data;
  } inputs[] = {{"empty", &empty}, {"one byte", &one}, {"100 bytes", &hundred}, {"64 KiB", &k64}, {"64 KiB + 1", &k65}, {"1 MiB of text", &repetitive}, {"1 MiB of noise", &noise}};
  const unsigned char header[10] = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff};
  for (int a : {0, 1, 5})
    for (const auto& in : inputs) {
      std::string gz, back;
      const nvcompStatus_t st = compress(*in.data, a, &gz);
      const std::string what = std::string(in.name) + ", algorithm " + std::to_string(a);
      check(st == nvcompSuccess, "compressing " + what);
      check(gz.size() >= 20 && std::memcmp(gz.data(), header, 10) == 0, "... writes a gzip header with no name, time or flags and OS 255");
      if (in.data->empty()) {
        check(gz.size() == 20, "... an empty input is a 20-byte member");
        continue;
      }
      check(batched_gunzip(gz, in.data->size(), &back) && back == *in.data, "... reads back through the batched Gzip API as one member");
    }

  // ---- the statuses for a bad stream or a small workspace ----
  {
    nvcompBatchedGzipCompressOpts_t o = nvcompBatchedGzipCompressDefaultOpts;
    o.algorithm = 0;
    std::string gz;
    IS(compress(hundred, 0, &gz, expected[0] - 1, true), nvcompErrorInvalidValue);
    IS(compress(hundred, 0, &gz, 0, true), nvcompErrorInvalidValue);
    check(gz.empty(), "a workspace too small writes nothing");
    IS(compress(hundred, 6, &gz), nvcompErrorInternal);
    // An input that cannot be read is an empty input; an output that cannot be written is an error.
    void* d = nullptr;
    cudaMalloc(&d, on_simulator() ? 16 : expected[0]);
    {
      std::ifstream missing("/nonexistent/vgpu-streaming-gzip");
      std::ostringstream os;
      IS(nvcompGzipStreamingCompress(missing, os, expected[0], d, o, 0), nvcompSuccess);
      check(os.str().size() == 20, "an unreadable input compresses as an empty one");
    }
    {
      std::istringstream is(hundred);
      std::ofstream unopened;
      IS(nvcompGzipStreamingCompress(is, unopened, expected[0], d, o, 0), nvcompErrorInternal);
    }
    cudaDeviceSynchronize();
    cudaFree(d);
  }

  // ---- decompression: not on this part ----
  {
    std::string gz;
    compress(hundred, 0, &gz);
    void* d = nullptr;
    cudaMalloc(&d, 1 << 20);
    for (size_t temp : {size_t(0), size_t(1) << 20}) {
      std::istringstream is(gz);
      std::ostringstream os;
      IS(nvcompGzipStreamingDecompress(is, os, temp, d, 0), nvcompErrorInvalidValue);
      check(os.str().empty(), "decompression writes nothing");
    }
    cudaFree(d);
  }
  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
