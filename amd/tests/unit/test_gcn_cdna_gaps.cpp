// Instructions of the CDNA parts (gfx90a, gfx942, gfx950) that the executor used to refuse and the CDNA ISA guides define
// (CDNA2: "AMD Instinct MI200 ISA", CDNA3: "AMD Instinct MI300 ISA", CDNA4: "CDNA4 ISA"), in kernels written in assembly
// (amd/tests/data/asm_cdna_*.s, assembled once per target by amd/tests/data/build.sh). The expected values are written from the
// guides' pseudocode, not taken from the simulator: each test does the arithmetic again on the host.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

amd::CodeObject object(const char* name, const char* target) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name + "." + target + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

// Runs one wave of `name` with `args` (device addresses, in the order the kernel takes them) and `group` work-items.
void launch(const amd::CodeObject& o, const char* name, MemoryManager& mem, const std::vector<uint64_t>& args,
            uint32_t group = 64) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  std::vector<uint8_t> buf(std::max<size_t>(k->kernarg_size, 8 * args.size()), 0);
  for (size_t i = 0; i < args.size(); ++i) std::memcpy(&buf[8 * i], &args[i], 8);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.group_size[0] = group;
  amd::execute(d, mem);
}

template <class T>
uint64_t put(MemoryManager& mem, const std::vector<T>& v) {
  const uint64_t at = mem.alloc(std::max<size_t>(v.size() * sizeof(T), 16));
  mem.write(at, v.data(), v.size() * sizeof(T));
  return at;
}
template <class T>
std::vector<T> get(MemoryManager& mem, uint64_t at, size_t n) {
  std::vector<T> v(n);
  mem.read(at, v.data(), n * sizeof(T));
  return v;
}

double d_of(uint64_t b) { return std::bit_cast<double>(b); }
uint64_t b_of(double d) { return std::bit_cast<uint64_t>(d); }

const char* const kTargets[] = {"gfx90a", "gfx942", "gfx950"};

}  // namespace

// BUFFER_ATOMIC_{SWAP,ADD,SUB,SMIN,UMIN,SMAX,UMAX,AND,OR,XOR,INC,DEC,CMPSWAP}_X2 and BUFFER_ATOMIC_{ADD,MIN,MAX}_F64 (the guides' pseudocode,
// e.g. "tmp = MEM[addr].u64; MEM[addr].u64 += DATA.u64; RETURN_DATA.u64 = tmp"): the old value comes back, the word changes as the
// operation says, and an access past the resource's end changes nothing and returns zero.
VTEST(buffer_atomics_of_64_bits_and_doubles_do_what_the_guides_pseudocode_says) {
  for (const char* target : kTargets) {
    const amd::CodeObject o = object("asm_cdna_mem64", target);
    MemoryManager mem(16ull << 20);
    std::vector<uint64_t> buf(32), vals(18);
    for (uint32_t i = 0; i < 32; ++i) buf[i] = 0x0123456789abcdefull * (i + 3) + i;
    for (uint32_t i = 0; i < 16; ++i) vals[i] = 0xfedcba9876543210ull / (i + 2) + 17 * i;
    buf[3] = 0x8000000000000001ull;   // a negative number: smin of it and 5 is itself, umin the 5
    vals[3] = 5;
    buf[4] = 100;
    vals[4] = 0x8000000000000000ull;   // umin(100, 2^63) is 100
    buf[5] = 0x8000000000000001ull;
    vals[5] = 5;
    buf[6] = 7;
    vals[6] = 0xffffffffffffffffull;
    buf[10] = 5;   // inc: 5 >= 5 wraps to 0
    vals[10] = 5;
    buf[11] = 3;   // dec: 3 counts down to 2
    vals[11] = 100;
    buf[12] = 0x1111222233334444ull;   // cmpswap finds what it compares with
    vals[16] = buf[12];
    buf[13] = b_of(1.5), vals[13] = b_of(2.25);
    buf[14] = b_of(3.0), vals[14] = b_of(-7.5);
    buf[15] = b_of(-2.0), vals[15] = b_of(10.0);
    const uint64_t b = put(mem, buf), v = put(mem, vals), out = mem.alloc(18 * 8);
    launch(o, "mem64", mem, {b, v, out});
    const std::vector<uint64_t> got = get<uint64_t>(mem, b, 32), old = get<uint64_t>(mem, out, 18);
    std::vector<uint64_t> want = buf;
    const auto s = [](uint64_t x) { return static_cast<int64_t>(x); };
    want[0] = vals[0];
    want[1] = buf[1] + vals[1];
    want[2] = buf[2] - vals[2];
    want[3] = s(vals[3]) < s(buf[3]) ? vals[3] : buf[3];
    want[4] = vals[4] < buf[4] ? vals[4] : buf[4];
    want[5] = s(vals[5]) > s(buf[5]) ? vals[5] : buf[5];
    want[6] = vals[6] > buf[6] ? vals[6] : buf[6];
    want[7] = buf[7] & vals[7];
    want[8] = buf[8] | vals[8];
    want[9] = buf[9] ^ vals[9];
    want[10] = buf[10] >= vals[10] ? 0 : buf[10] + 1;
    want[11] = (buf[11] == 0 || buf[11] > vals[11]) ? vals[11] : buf[11] - 1;
    want[12] = buf[12] == vals[16] ? vals[12] : buf[12];
    want[13] = b_of(d_of(buf[13]) + d_of(vals[13]));
    want[14] = b_of(d_of(vals[14]) < d_of(buf[14]) ? d_of(vals[14]) : d_of(buf[14]));
    want[15] = b_of(d_of(vals[15]) > d_of(buf[15]) ? d_of(vals[15]) : d_of(buf[15]));
    want[31] = buf[31] + 5;   // the in-range access to the last word, and no second add from the one past the end
    int wrong = 0;
    for (uint32_t i = 0; i < 32; ++i) wrong += got[i] != want[i];
    for (uint32_t i = 0; i < 16; ++i) wrong += old[i] != buf[i];
    wrong += old[16] != buf[31];
    wrong += old[17] != 0;
    if (wrong) std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
    VCHECK_EQ(wrong, 0);
  }
}

namespace {
// 512 bytes of source for the LDS loads: any value in the first three bytes of a lane's dword, and in the fourth either 0 or 255 (which
// an 8-bit unorm format converts to 0.0 and 1.0 exactly).
std::vector<uint8_t> lds_source() {
  std::vector<uint8_t> src(512);
  for (uint32_t i = 0; i < 512; ++i) src[i] = static_cast<uint8_t>(i * 37 + 11);
  for (uint32_t l = 0; l < 64; ++l) src[4 * l + 3] = l % 3 == 1 ? 0x00 : 0xFF;
  src[0] = 0x80;   // so that the first lane's sign extensions show
  src[1] = 0x81;
  return src;
}
uint32_t u16_at(const std::vector<uint8_t>& s, size_t at) { return s[at] | uint32_t{s[at + 1]} << 8; }
uint32_t u32_at(const std::vector<uint8_t>& s, size_t at) { return u16_at(s, at) | u16_at(s, at + 2) << 16; }
}  // namespace

// MUBUF loads with the LDS bit, the narrow ones and format_x (CDNA2, CDNA3 and CDNA4 ISA 9.1.9: "BUFFER_LOAD_{ubyte, sbyte, ushort, sshort,
// dword, format_x}", LDS_ADDR = LDSbase + M0[15:0] + inst_offset + TIDinWave * 4): lane L's byte or half lands in its dword of LDS, zero or
// sign extended, and format_x puts the converted channel there. On gfx90a and gfx942 only M0[15:0] is the offset.
VTEST(buffer_loads_of_bytes_and_halves_into_lds_land_as_dwords) {
  for (const char* target : kTargets) {
    const amd::CodeObject o = object("asm_cdna_ldsdma", target);
    MemoryManager mem(16ull << 20);
    const std::vector<uint8_t> src = lds_source();
    const uint64_t in = put(mem, src), out = mem.alloc(7 * 64 * 4);
    launch(o, "buf_lds_narrow", mem, {in, out});
    const std::vector<uint32_t> got = get<uint32_t>(mem, out, 7 * 64);
    std::vector<uint32_t> want(7 * 64, 0);
    for (uint32_t l = 0; l < 64; ++l) {
      want[0 * 64 + l] = src[4 * l];
      want[1 * 64 + l] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(src[4 * l])));
      want[2 * 64 + l] = u16_at(src, 4 * l);
      want[3 * 64 + l] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(u16_at(src, 4 * l))));
      want[4 * 64 + l] = src[4 * l + 3] == 0xFF ? 0x3f800000u : 0u;   // an 8-bit unorm 255 is 1.0f
      want[5 * 64 + l] = u16_at(src, 4 * l + 2);
      if (std::string(target) != "gfx950") want[6 * 64 + l] = src[4 * l + 1];   // M0 = 0x10600: LDS offset 0x600
    }
    int wrong = 0;
    for (size_t i = 0; i < want.size(); ++i) wrong += got[i] != want[i];
    if (wrong) {
      std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
      for (uint32_t r = 0; r < 7; ++r)
        std::fprintf(stderr, "  row %u: got %08x %08x %08x %08x want %08x %08x %08x %08x\n", r, got[r * 64], got[r * 64 + 1], got[r * 64 + 2],
                     got[r * 64 + 3], want[r * 64], want[r * 64 + 1], want[r * 64 + 2], want[r * 64 + 3]);
    }
    VCHECK_EQ(wrong, 0);
  }
}

// global_load_lds_{ubyte,sbyte,ushort,sshort,dword} (CDNA3 and CDNA4: "Load 8 bits of untyped data from the global aperture, zero extend to
// 32 bits and store the result into a data share"; LDS_ADDR = LDSbase + LDSoffset(M0[17:2] * 4) + INST.OFFSET + ThreadID*4): the same
// dwords, at M0 with its two low bits dropped, and on gfx950 (160 KiB of LDS) with the offset's top bits.
VTEST(global_loads_of_bytes_and_halves_into_lds_land_as_dwords_at_m0_dwords) {
  for (const char* target : {"gfx942", "gfx950"}) {
    const amd::CodeObject o = object("asm_cdna_ldsdma", target);
    MemoryManager mem(16ull << 20);
    const std::vector<uint8_t> src = lds_source();
    const uint64_t in = put(mem, src), out = mem.alloc(7 * 64 * 4);
    launch(o, "glob_lds_narrow", mem, {in, out});
    const std::vector<uint32_t> got = get<uint32_t>(mem, out, 7 * 64);
    std::vector<uint32_t> want(7 * 64, 0);
    for (uint32_t l = 0; l < 64; ++l) {
      want[0 * 64 + l] = src[4 * l];
      want[1 * 64 + l] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(src[4 * l])));
      want[2 * 64 + l] = u16_at(src, 4 * l);
      want[3 * 64 + l] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(u16_at(src, 4 * l))));
      want[4 * 64 + l] = u32_at(src, 4 * l);   // M0 = 0x403: LDS offset 0x400
      want[5 * 64 + l] = u16_at(src, 4 * l + 2);
      if (std::string(target) == "gfx950") want[6 * 64 + l] = u32_at(src, 4 * l);   // M0 = 0x10601: LDS offset 0x10600
    }
    int wrong = 0;
    for (size_t i = 0; i < want.size(); ++i) wrong += got[i] != want[i];
    if (wrong) std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
    VCHECK_EQ(wrong, 0);
  }
}

VTEST_MAIN
