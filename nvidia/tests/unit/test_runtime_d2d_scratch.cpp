// Two paths of the runtime that every API-heavy program pays for, made cheaper without changing what they answer:
//  * a device-to-device cudaMemcpy goes through a buffer (so an overlapping copy reads its whole source first); the
//    buffer is the call's frame for a small copy and an uninitialised allocation for a large one, never zero-filled.
//  * the context's fault (sticky_error in runtime_api.cpp) is read without the API lock once the machine exists.
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

extern "C" const char* __asan_default_options() { return "detect_leaks=1:detect_odr_violation=0"; }
extern "C" vgpu::runtime::Runtime* vgpu_shared_runtime_v1();

namespace {
std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + i * 31u + (i >> 8));
  return v;
}
}  // namespace

// Sizes around the frame buffer's limit (16 KiB) and well past it: every byte arrives.
VTEST(a_device_to_device_copy_moves_every_byte_at_every_size) {
  for (size_t n : {size_t{1}, size_t{7}, size_t{4096}, size_t{16383}, size_t{16384}, size_t{16385}, size_t{100000}, size_t{300000}}) {
    uint8_t *a = nullptr, *b = nullptr;
    VCHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&a), n), cudaSuccess);
    VCHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&b), n), cudaSuccess);
    const auto in = pattern(n, 5);
    VCHECK_EQ(cudaMemcpy(a, in.data(), n, cudaMemcpyHostToDevice), cudaSuccess);
    VCHECK_EQ(cudaMemset(b, 0xEE, n), cudaSuccess);
    VCHECK_EQ(cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice), cudaSuccess);
    std::vector<uint8_t> out(n, 0);
    VCHECK_EQ(cudaMemcpy(out.data(), b, n, cudaMemcpyDeviceToHost), cudaSuccess);
    VCHECK(std::memcmp(in.data(), out.data(), n) == 0);
    VCHECK_EQ(cudaFree(a), cudaSuccess);
    VCHECK_EQ(cudaFree(b), cudaSuccess);
  }
}

// Memory never written reads as zero, and copying it over written memory clears that (the buffer's old contents are never what arrives).
VTEST(a_copy_from_untouched_memory_writes_zeros_over_stale_scratch) {
  uint8_t *a = nullptr, *b = nullptr, *dirty = nullptr;
  const size_t n = 8192;
  VCHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&a), n), cudaSuccess);
  VCHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&b), n), cudaSuccess);
  VCHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&dirty), n), cudaSuccess);
  const auto junk = pattern(n, 99);
  // a first copy leaves junk in whatever buffer the call used
  VCHECK_EQ(cudaMemcpy(dirty, junk.data(), n, cudaMemcpyHostToDevice), cudaSuccess);
  VCHECK_EQ(cudaMemcpy(b, dirty, n, cudaMemcpyDeviceToDevice), cudaSuccess);
  VCHECK_EQ(cudaMemset(a, 0, n), cudaSuccess);   // (a is untouched too: zero)
  VCHECK_EQ(cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice), cudaSuccess);
  std::vector<uint8_t> out(n, 0x55);
  VCHECK_EQ(cudaMemcpy(out.data(), b, n, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < n; ++i) VCHECK_EQ(out[i], 0);
  VCHECK_EQ(cudaFree(a), cudaSuccess);
  VCHECK_EQ(cudaFree(b), cudaSuccess);
  VCHECK_EQ(cudaFree(dirty), cudaSuccess);
}

// An overlapping copy reads all of its source before it writes, in both directions, small and large.
VTEST(an_overlapping_device_to_device_copy_reads_the_whole_source_first) {
  for (size_t n : {size_t{4000}, size_t{40000}}) {
    const size_t shift = 1000;
    uint8_t* d = nullptr;
    VCHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&d), n + shift), cudaSuccess);
    const auto in = pattern(n + shift, 17);
    VCHECK_EQ(cudaMemcpy(d, in.data(), n + shift, cudaMemcpyHostToDevice), cudaSuccess);
    VCHECK_EQ(cudaMemcpy(d + shift, d, n, cudaMemcpyDeviceToDevice), cudaSuccess);   // forwards
    std::vector<uint8_t> want = in;
    std::memmove(want.data() + shift, in.data(), n);
    std::vector<uint8_t> out(n + shift);
    VCHECK_EQ(cudaMemcpy(out.data(), d, n + shift, cudaMemcpyDeviceToHost), cudaSuccess);
    VCHECK(out == want);
    VCHECK_EQ(cudaMemcpy(d, d + shift, n, cudaMemcpyDeviceToDevice), cudaSuccess);   // backwards
    std::vector<uint8_t> want2 = want;
    std::memmove(want2.data(), want.data() + shift, n);
    VCHECK_EQ(cudaMemcpy(out.data(), d, n + shift, cudaMemcpyDeviceToHost), cudaSuccess);
    VCHECK(out == want2);
    VCHECK_EQ(cudaFree(d), cudaSuccess);
  }
}

// A copy that cannot be made fails as before and leaves the destination alone.
VTEST(a_device_to_device_copy_past_the_end_is_refused_and_changes_nothing) {
  uint8_t *a = nullptr, *b = nullptr;
  VCHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&a), 1024), cudaSuccess);
  VCHECK_EQ(cudaMalloc(reinterpret_cast<void**>(&b), 1024), cudaSuccess);
  VCHECK_EQ(cudaMemset(b, 0x42, 1024), cudaSuccess);
  VCHECK(cudaMemcpy(b, a, 5000, cudaMemcpyDeviceToDevice) != cudaSuccess);
  (void)cudaGetLastError();
  std::vector<uint8_t> out(1024);
  VCHECK_EQ(cudaMemcpy(out.data(), b, 1024, cudaMemcpyDeviceToHost), cudaSuccess);
  for (uint8_t v : out) VCHECK_EQ(v, 0x42);
  VCHECK_EQ(cudaFree(a), cudaSuccess);
  VCHECK_EQ(cudaFree(b), cudaSuccess);
}

// A dead context answers every call that needs one, from any thread, without the API lock; clearing it brings the calls back.
VTEST(a_context_fault_is_seen_by_calls_on_every_thread_and_cleared_again) {
  void* p = nullptr;
  VCHECK_EQ(cudaMalloc(&p, 64), cudaSuccess);   // (makes the machine)
  vgpu::runtime::Runtime* rt = vgpu_shared_runtime_v1();
  VCHECK(rt != nullptr);
  rt->set_context_fault(static_cast<int>(cudaErrorAssert));
  cudaStream_t s = nullptr;
  VCHECK_EQ(cudaStreamCreate(&s), cudaErrorAssert);
  cudaEvent_t e = nullptr;
  VCHECK_EQ(cudaEventCreate(&e), cudaErrorAssert);
  cudaError_t other = cudaSuccess;
  std::thread([&] {
    cudaStream_t t = nullptr;
    other = cudaStreamCreate(&t);
  }).join();
  VCHECK_EQ(other, cudaErrorAssert);
  (void)cudaGetLastError();
  rt->set_context_fault(0);
  VCHECK_EQ(cudaStreamCreate(&s), cudaSuccess);
  VCHECK_EQ(cudaStreamQuery(s), cudaSuccess);
  VCHECK_EQ(cudaStreamDestroy(s), cudaSuccess);
  VCHECK_EQ(cudaFree(p), cudaSuccess);
}

// Many threads asking while another sets and clears the fault: no crash, and every answer is one of the two.
VTEST(the_fault_can_be_read_while_another_thread_sets_it) {
  VCHECK_EQ(cudaFree(nullptr), cudaSuccess);
  vgpu::runtime::Runtime* rt = vgpu_shared_runtime_v1();
  std::atomic<bool> stop{false};
  std::atomic<int> bad{0};
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i)
    readers.emplace_back([&] {
      while (!stop.load()) {
        cudaStream_t t = nullptr;
        const cudaError_t r = cudaStreamCreate(&t);
        if (r == cudaSuccess) {
          const cudaError_t d = cudaStreamDestroy(t);
          if (d != cudaSuccess && d != cudaErrorAssert) ++bad;
        } else if (r != cudaErrorAssert) {
          ++bad;
        }
      }
    });
  for (int i = 0; i < 200; ++i) {
    rt->set_context_fault(static_cast<int>(cudaErrorAssert));
    std::this_thread::yield();
    rt->set_context_fault(0);
    std::this_thread::yield();
  }
  stop = true;
  for (auto& t : readers) t.join();
  rt->set_context_fault(0);
  (void)cudaGetLastError();
  VCHECK_EQ(bad.load(), 0);
}

VTEST_MAIN
