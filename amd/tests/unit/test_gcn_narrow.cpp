// Values packed into parts of a register, and a private array of them, run
// and checked against what the C means.
//
// The kernels are amd/tests/data/narrow.c, compiled for gfx942. They are
// what finishes the sub-dword form: an instruction here writes its result
// into a named half of the destination and zeroes the rest, reads a source
// that is a scalar register rather than a vector one, or is a move with only
// one source. And the last of them keeps a private array of bytes, which the
// work-item's own memory holds a byte at a time.
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/narrow.gfx942.o";
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

}  // namespace

VTEST(two_floats_narrowed_to_halves_and_put_in_one_register) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<float> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(i) * 0.25f - 8.0f;
    b[i] = static_cast<float>(i) * -0.5f + 3.0f;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
  run(o, "pack_halves", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const _Float16 lo = static_cast<_Float16>(a[i]), hi = static_cast<_Float16>(b[i]);
    uint16_t l = 0, h = 0;
    std::memcpy(&l, &lo, 2);
    std::memcpy(&h, &hi, 2);
    const uint32_t want = static_cast<uint32_t>(l) | static_cast<uint32_t>(h) << 16;
    if (out[i] != want)
      throw vtest::Failure("pack_halves[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want) + " (the high half is written on its own)");
  }
}

VTEST(a_half_of_the_result_written_into_the_high_half_of_a_register) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint16_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = static_cast<uint16_t>(i * 1013 + 60000);   // it wraps
  const uint32_t s = 40001;
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "pair_of_shorts", mem, {pa, s, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint16_t v = static_cast<uint16_t>(a[i] + s);
    const uint32_t want =
        v | static_cast<uint32_t>(static_cast<uint16_t>(static_cast<uint32_t>(v) * static_cast<uint16_t>(s)))
                << 16;
    if (out[i] != want)
      throw vtest::Failure("pair_of_shorts[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_shift_whose_amount_every_work_item_shares) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<uint16_t> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = static_cast<uint16_t>(i * 577 + 11);
  const uint32_t s = 33333;
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 4);
  run(o, "shifted", mem, {pa, s, pout, static_cast<uint64_t>(kN)});

  const std::vector<uint32_t> out = download<uint32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const uint32_t lo = static_cast<uint16_t>(s - a[i]);
    const uint32_t hi = static_cast<uint16_t>(static_cast<uint16_t>(s) << (a[i] & 7));
    const uint32_t want = lo | hi << 16;
    if (out[i] != want)
      throw vtest::Failure("shifted[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(a_private_array_of_bytes_is_kept_a_byte_at_a_time) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const int n = 64;
  std::vector<int32_t> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i * 37 - 500;
  const uint64_t pidx = upload(mem, idx), pout = mem.alloc(n * 4);
  run(o, "spill_bytes", mem, {pidx, pout, static_cast<uint64_t>(n)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, n);
  for (int t = 0; t < n; ++t) {
    // The same array of bytes, and the same reads of it. They are signed
    // bytes, so the values above 127 come back negative.
    signed char v[64];
    for (int i = 0; i < 64; ++i) v[i] = static_cast<signed char>(idx[i % n] + i);
    int32_t want = 0;
    for (int i = 0; i < 24; ++i) want += v[idx[(t + i) % n] & 63];
    if (out[t] != want)
      throw vtest::Failure("spill_bytes[" + std::to_string(t) + "] is " + std::to_string(out[t]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
