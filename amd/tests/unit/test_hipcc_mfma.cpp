// The matrix instructions, checked against a layout nothing here decided.
//
// v_mfma_f32_16x16x16_f16 multiplies two 16x16 matrices spread across the
// wave's 64 lanes, and which element sits in which lane's register is the
// whole question: a model that had it wrong would still produce plausible
// matrices. The float and double forms (v_mfma_f32_16x16x4_f32,
// v_mfma_f64_16x16x4_f64) are the same question again, with answers of their
// own: a double's accumulator does not hold its rows as a float's does. So the kernel here is a GEMM written with rocWMMA
// (amd/tests/hipcc/wmma.cpp) -- AMD's library, whose loads and stores put each
// element where the hardware expects it -- built by hipcc 7.1, and its answer
// is compared with the same product worked out in C. The inputs are small
// integers, so every product and sum is exact and any order of summing gives
// the same float.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/hipcc/wmma.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

template <typename T>
uint64_t upload(MemoryManager& mem, const std::vector<T>& v) {
  const uint64_t p = mem.alloc(v.size() * sizeof(T));
  mem.write(p, v.data(), v.size() * sizeof(T));
  return p;
}

template <typename T>
std::vector<T> download(MemoryManager& mem, uint64_t p, size_t n) {
  std::vector<T> out(n);
  mem.read(p, out.data(), n * sizeof(T));
  return out;
}

uint64_t kernargs(MemoryManager& mem, const amd::Kernel& k, const std::vector<uint64_t>& values) {
  std::vector<uint8_t> buf(k.kernarg_size, 0);
  for (size_t i = 0; i < values.size() && i < k.args.size(); ++i) {
    const amd::KernelArg& a = k.args[i];
    for (uint32_t b = 0; b < a.size && b < 8; ++b) buf[a.offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  }
  const uint64_t p = mem.alloc(buf.empty() ? 1 : buf.size());
  if (!buf.empty()) mem.write(p, buf.data(), buf.size());
  return p;
}

void run(const amd::CodeObject& o, const char* name, MemoryManager& mem, const std::vector<uint64_t>& args) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, args);
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);
}

const int kN = 64;

uint64_t bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

uint16_t half_bits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  uint16_t u = 0;
  std::memcpy(&u, &h, 2);
  return u;
}

}  // namespace

namespace {

uint16_t h16(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  uint16_t u = 0;
  std::memcpy(&u, &h, 2);
  return u;
}

}  // namespace

VTEST(a_rocwmma_gemm_gives_the_product_worked_out_in_c) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  float a[16][16], b[16][16], c[16][16];
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      a[i][j] = static_cast<float>((i * 5 + j * 3) % 17 - 8);
      b[i][j] = static_cast<float>((i * 7 + j * 11) % 19 - 9);
      c[i][j] = static_cast<float>(i * 16 + j) - 100.0f;
    }
  // A row-major, B column-major, C row-major: the layouts the kernel asks
  // rocWMMA to load.
  std::vector<uint16_t> ha(256), hb(256);
  std::vector<float> hc(256);
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      ha[i * 16 + j] = h16(a[i][j]);
      hb[j * 16 + i] = h16(b[i][j]);   // element (k = i, n = j) of B, stored by column
      hc[i * 16 + j] = c[i][j];
    }
  const uint64_t pa = upload(mem, ha), pb = upload(mem, hb), pc = upload(mem, hc), pd = mem.alloc(256 * 4);
  run(o, "_Z4gemmPKDF16_S0_PKfPf", mem, {pa, pb, pc, pd});
  const std::vector<float> d = download<float>(mem, pd, 256);
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      float want = c[i][j];
      for (int k = 0; k < 16; ++k) want += a[i][k] * b[k][j];
      if (d[i * 16 + j] != want)
        throw vtest::Failure("D[" + std::to_string(i) + "][" + std::to_string(j) + "] is " +
                             std::to_string(d[i * 16 + j]) + ", not " + std::to_string(want));
    }
}

// A rocWMMA GEMM of floats or doubles, M square and K deep, against the
// product worked out in C.
template <typename T>
void check_gemm(const char* kernel, int m, int kk) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<T> a(m * kk), b(kk * m), c(m * m);
  for (int i = 0; i < m; ++i)
    for (int k = 0; k < kk; ++k) {
      a[i * kk + k] = static_cast<T>((i * 5 + k * 3) % 17 - 8);        // row-major
      b[i * kk + k] = static_cast<T>((k * 7 + i * 11) % 19 - 9);       // column i of B, by column
    }
  for (int i = 0; i < m * m; ++i) c[i] = static_cast<T>(i % 97) - 40;
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pc = upload(mem, c), pd = mem.alloc(m * m * sizeof(T));
  run(o, kernel, mem, {pa, pb, pc, pd});
  const std::vector<T> d = download<T>(mem, pd, m * m);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) {
      T want = c[i * m + j];
      for (int k = 0; k < kk; ++k) want += a[i * kk + k] * b[j * kk + k];
      if (d[i * m + j] != want)
        throw vtest::Failure(std::string(kernel) + ": D[" + std::to_string(i) + "][" + std::to_string(j) + "] is " +
                             std::to_string(d[i * m + j]) + ", not " + std::to_string(want));
    }
}

VTEST(a_rocwmma_gemm_of_floats_gives_the_product_worked_out_in_c) {
  check_gemm<float>("_Z17gemm_f32_16x16x16PKfS0_S0_Pf", 16, 16);
  check_gemm<float>("_Z16gemm_f32_32x32x8PKfS0_S0_Pf", 32, 8);
}

VTEST(a_rocwmma_gemm_of_doubles_gives_the_product_worked_out_in_c) {
  check_gemm<double>("_Z17gemm_f64_16x16x16PKdS0_S0_Pd", 16, 16);
}

VTEST_MAIN
