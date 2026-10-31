// A large hipMemcpy between device buffers goes chunk to chunk (MemoryManager::copy_to), with no buffer for the row. Every byte
// arrives at every alignment of either end to the 64 KiB chunks, the bytes around the range stay, an overlapping copy still reads
// its whole source first, a 2D copy of large rows lands row by row, and a copy past the end of a buffer is refused.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vgpu_hip.h"

extern "C" hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                                  hipMemcpyKind kind);

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}
std::vector<uint8_t> pattern(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + i * 31u + (i >> 8));
  return v;
}
}  // namespace

int main() {
  const size_t total = 700000;
  uint8_t *a = nullptr, *b = nullptr;
  check(hipMalloc(reinterpret_cast<void**>(&a), total) == hipSuccess && hipMalloc(reinterpret_cast<void**>(&b), total) == hipSuccess, "malloc");
  const auto in = pattern(total, 3), bg = pattern(total, 200);
  std::vector<uint8_t> out(total);
  struct Case { size_t so, dO, n; } cases[] = {{0, 0, 70000}, {65537, 130000, 400000}, {100, 65536, 200000}, {5, 17, 20000}};
  for (const Case& c : cases) {
    check(hipMemcpy(a, in.data(), total, hipMemcpyHostToDevice) == hipSuccess, "h2d a");
    check(hipMemcpy(b, bg.data(), total, hipMemcpyHostToDevice) == hipSuccess, "h2d b");
    check(hipMemcpy(b + c.dO, a + c.so, c.n, hipMemcpyDeviceToDevice) == hipSuccess, "d2d");
    std::vector<uint8_t> want = bg;
    std::memcpy(want.data() + c.dO, in.data() + c.so, c.n);
    check(hipMemcpy(out.data(), b, total, hipMemcpyDeviceToHost) == hipSuccess, "d2h");
    check(out == want, "a large copy moves exactly its range");
  }
  // Overlap inside one buffer, both directions: the whole source is read before the first byte is written.
  {
    const size_t n = 300000, shift = 1000;
    check(hipMemcpy(a, in.data(), total, hipMemcpyHostToDevice) == hipSuccess, "h2d");
    check(hipMemcpy(a + shift, a, n, hipMemcpyDeviceToDevice) == hipSuccess, "overlap forwards");
    std::vector<uint8_t> want = in;
    std::memmove(want.data() + shift, in.data(), n);
    check(hipMemcpy(out.data(), a, total, hipMemcpyDeviceToHost) == hipSuccess && out == want, "forwards overlap is a memmove");
    check(hipMemcpy(a, a + shift, n, hipMemcpyDeviceToDevice) == hipSuccess, "overlap backwards");
    std::vector<uint8_t> want2 = want;
    std::memmove(want2.data(), want.data() + shift, n);
    check(hipMemcpy(out.data(), a, total, hipMemcpyDeviceToHost) == hipSuccess && out == want2, "backwards overlap is a memmove");
  }
  // A 2D copy of rows of 100000 bytes (each row is a large copy), pitches apart.
  {
    const size_t w = 100000, rows = 5, spitch = 120000, dpitch = 130000;
    check(hipMemcpy(a, in.data(), total, hipMemcpyHostToDevice) == hipSuccess, "h2d");
    check(hipMemcpy(b, bg.data(), total, hipMemcpyHostToDevice) == hipSuccess, "h2d");
    check(hipMemcpy2D(b, dpitch, a, spitch, w, rows, hipMemcpyDeviceToDevice) == hipSuccess, "2d");
    std::vector<uint8_t> want = bg;
    for (size_t r = 0; r < rows; ++r) std::memcpy(want.data() + r * dpitch, in.data() + r * spitch, w);
    check(hipMemcpy(out.data(), b, total, hipMemcpyDeviceToHost) == hipSuccess && out == want, "2D rows land where they belong");
  }
  // A copy past the end is refused and changes nothing.
  {
    check(hipMemcpy(b, bg.data(), total, hipMemcpyHostToDevice) == hipSuccess, "h2d");
    check(hipMemcpy(b, a, total + 100000, hipMemcpyDeviceToDevice) != hipSuccess, "a copy past the end is refused");
    (void)hipGetLastError();
    check(hipMemcpy(out.data(), b, total, hipMemcpyDeviceToHost) == hipSuccess && out == bg, "and leaves the destination alone");
  }
  hipFree(a);
  hipFree(b);
  if (failures) return 1;
  std::printf("ok large device-to-device copies go chunk to chunk and land exactly\n");
  return 0;
}
