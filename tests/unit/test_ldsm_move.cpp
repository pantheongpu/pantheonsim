// The data movement of LDSM (include/vgpu/exec/ldsm.hpp) against the element-at-a-time form it replaced (kept below as it stood in the SASS tier's
// handler): every count of matrices, plain and transposed, rows anywhere in the array (aligned or not, shared between lanes, the last 16 bytes of the array,
// the first), and with the array ending at the end of a mapped page so that a read past a row's 16 bytes faults. Only the low 24 bits of a lane's
// address are an offset: the bits above them must not matter.
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "vgpu/exec/ldsm.hpp"
#include "vtest.hpp"

namespace {

// (the handler's loop before the change)
void reference(const uint8_t* sh, const uint32_t* rows, unsigned nmat, bool trans, uint32_t out[4][32]) {
  for (unsigned i = 0; i < nmat; ++i)
    for (unsigned t = 0; t < 32; ++t) {
      if (!trans) {
        std::memcpy(&out[i][t], sh + (rows[8 * i + t / 4] & 0xffffff) + 4 * (t % 4), 4);
      } else {
        uint16_t lo, hi;
        std::memcpy(&lo, sh + (rows[8 * i + 2 * (t % 4)] & 0xffffff) + 2 * (t / 4), 2);
        std::memcpy(&hi, sh + (rows[8 * i + 2 * (t % 4) + 1] & 0xffffff) + 2 * (t / 4), 2);
        out[i][t] = lo | (static_cast<uint32_t>(hi) << 16);
      }
    }
}

struct EdgeArray {   // `size` bytes ending at the last byte of a mapped page, the page after it not mapped
  char* base = nullptr;
  size_t map_len = 0;
  uint8_t* data = nullptr;
  size_t size = 0;
  explicit EdgeArray(size_t n) : size(n) {
    const size_t pg = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    map_len = (n + pg - 1) / pg * pg;
    void* m = mmap(nullptr, map_len + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) throw vtest::Failure("mmap failed");
    base = static_cast<char*>(m);
    if (mprotect(base + map_len, pg, PROT_NONE) != 0) throw vtest::Failure("mprotect failed");
    data = reinterpret_cast<uint8_t*>(base + map_len) - n;
  }
  EdgeArray(const EdgeArray&) = delete;
  EdgeArray& operator=(const EdgeArray&) = delete;
  ~EdgeArray() { munmap(base, map_len + static_cast<size_t>(sysconf(_SC_PAGESIZE))); }
};

}  // namespace

VTEST(ldsm_move_equals_the_element_form) {
  std::mt19937_64 rng(0x1d5);
  const size_t sizes[] = {16, 48, 256, 4099, 65536};
  long calls = 0;
  for (size_t size : sizes) {
    EdgeArray arr(size);
    for (size_t i = 0; i < size; ++i) arr.data[i] = static_cast<uint8_t>(rng());
    for (int round = 0; round < 3000; ++round) {
      uint32_t rows[32];
      const int mode = round % 6;
      for (unsigned l = 0; l < 32; ++l) {
        int64_t off;
        switch (mode) {
          case 0: off = static_cast<uint32_t>(rng() % (size - 15)); break;                                    // anywhere
          case 1: off = static_cast<uint32_t>((rng() % (size / 16)) * 16); break;                              // 16-byte aligned (what hardware requires)
          case 2: off = static_cast<uint32_t>(size - 16 - (rng() % 3 ? 0 : rng() % 4)); break;                 // at the end of the array
          case 3: off = static_cast<uint32_t>(rng() % 2 ? 0 : rng() % 5); break;                               // at the start, shared between lanes
          case 4: off = static_cast<uint32_t>((l % 8) * 16 % (size - 15)); break;                              // eight consecutive rows, repeated
          default: off = static_cast<uint32_t>(rng() % (size - 15)); break;
        }
        off = off < 0 ? 0 : off > static_cast<int64_t>(size) - 16 ? static_cast<int64_t>(size) - 16 : off;   // (the window check's guarantee: 16 bytes inside)
        rows[l] = static_cast<uint32_t>(off) | (static_cast<uint32_t>(rng()) & 0xff000000u);   // (the bits above the offset are some other block's rank, not part of the offset)
      }
      if (size < 128 && mode == 4) continue;
      for (unsigned nmat : {1u, 2u, 4u})
        for (int trans = 0; trans < 2; ++trans) {
          uint32_t want[4][32], got[4][32];
          std::memset(want, 0xa5, sizeof want);
          std::memset(got, 0xa5, sizeof got);
          reference(arr.data, rows, nmat, trans != 0, want);
          vgpu::exec::ldsm_move(arr.data, rows, nmat, trans != 0, &got[0][0]);
          ++calls;
          if (std::memcmp(want, got, sizeof want) != 0)   // (also: registers past the last matrix are untouched, as the reference leaves them)
            throw vtest::Failure("size " + std::to_string(size) + " round " + std::to_string(round) + " nmat " + std::to_string(nmat) + (trans ? " transposed" : " plain") +
                                 ": the moved words differ from the element form");
        }
    }
  }
  std::printf("  %ld LDSM moves compared word for word\n", calls);
  VCHECK(calls > 40000);
}

VTEST_MAIN
