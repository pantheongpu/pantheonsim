// Throwaway prototype (r7-design). Shared setup for the harness: loading a cubin
// onto a simulated RTX 3060, allocating and filling buffers, launching through
// the real SASS engine, and comparing results.
#pragma once

#include <time.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"

namespace r7 {

inline double cpu_seconds() {
  timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}
inline double wall_seconds() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

inline std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string s = ss.str();
  return std::vector<uint8_t>(s.begin(), s.end());
}

template <class T>
std::vector<uint8_t> arg(T v) {
  std::vector<uint8_t> b(sizeof(T));
  std::memcpy(b.data(), &v, sizeof(T));
  return b;
}

// Values that break float arithmetic, mixed with ordinary ones, deterministically.
inline uint32_t tricky_f32_bits(uint64_t i) {
  static const uint32_t special[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x00800000u,
                                     0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7fc00123u,
                                     0xffc00000u, 0x7f800001u, 0x3f800000u, 0xbf800000u};
  uint64_t x = i * 0x9e3779b97f4a7c15ull + 0x1234567ull;
  x ^= x >> 29;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 32;
  if ((x & 15) == 0) return special[(x >> 8) & 15];
  // an ordinary float in [-8, 8)
  const float f = (static_cast<float>(static_cast<int32_t>(x >> 8) & 0xffffff) / 8388608.0f - 1.0f) * 8.0f;
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

struct Env {
  std::unique_ptr<vgpu::MemoryManager> mem;
  vgpu::DeviceProfile profile;
  std::shared_ptr<vgpu::sass::Module> mod;
  std::vector<uint8_t> image;
  Env(const std::string& cubin_path) {
    mem = std::make_unique<vgpu::MemoryManager>(1ull << 32);
    profile = vgpu::load_gpu("nvidia/rtx3060");
    image = read_file(cubin_path);
    mod = vgpu::sass::load(image.data(), image.size(), *mem, profile);
  }
};

}  // namespace r7
