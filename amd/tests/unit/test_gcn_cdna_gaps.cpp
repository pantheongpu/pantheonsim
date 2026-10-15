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
    const uint64_t in = put(mem, src), out = mem.alloc(8 * 64 * 4);
    launch(o, "buf_lds_narrow", mem, {in, out});
    const std::vector<uint32_t> got = get<uint32_t>(mem, out, 8 * 64);
    std::vector<uint32_t> want(8 * 64, 0);
    for (uint32_t l = 0; l < 64; ++l) {
      want[0 * 64 + l] = src[4 * l];
      want[1 * 64 + l] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(src[4 * l])));
      want[2 * 64 + l] = u16_at(src, 4 * l);
      want[3 * 64 + l] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(u16_at(src, 4 * l))));
      want[4 * 64 + l] = src[4 * l + 3] == 0xFF ? 0x3f800000u : 0u;   // an 8-bit unorm 255 is 1.0f
      want[5 * 64 + l] = u16_at(src, 4 * l + 2);
      if (std::string(target) != "gfx950") want[6 * 64 + l] = src[4 * l + 1];   // M0 = 0x10600: LDS offset 0x600
      want[7 * 64 + l] = u32_at(src, 4 * l);   // the dword form, which gfx90a's guide lists too
    }
    int wrong = 0;
    for (size_t i = 0; i < want.size(); ++i) wrong += got[i] != want[i];
    if (wrong) {
      std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
      for (uint32_t r = 0; r < 8; ++r)
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

// The rest of LDS's atomics and exchanges (CDNA2, CDNA3 and CDNA4 ISA, DS_MSKOR_B32 .. DS_CONDXCHG32_RTN_B64, DS_APPEND, DS_CONSUME and the
// ADDTID pair), each lane on words of its own so that the host can do the arithmetic lane by lane.
VTEST(lds_masked_ors_exchanges_compare_stores_wraps_append_and_addtid_do_what_the_guides_say) {
  for (const char* target : kTargets) {
    const amd::CodeObject o = object("asm_cdna_ds", target);
    MemoryManager mem(16ull << 20);
    uint64_t seed = 0x9e3779b97f4a7c15ull;
    const auto rnd = [&] {
      seed = seed * 6364136223846793005ull + 1442695040888963407ull;
      return static_cast<uint32_t>(seed >> 32);
    };
    std::vector<uint32_t> init(2432), data(64 * 16);
    for (uint32_t& x : init) x = rnd();
    for (uint32_t& x : data) x = rnd();
    const auto q = [](const std::vector<uint32_t>& v, size_t at) { return uint64_t{v[at]} | uint64_t{v[at + 1]} << 32; };
    for (uint32_t l = 0; l < 64; ++l) {
      uint32_t* d = &data[l * 16];
      uint32_t* m = &init[l * 32];
      if (l % 2 == 0) m[12] = d[8], m[13] = d[9];   // DS_CMPST_B64 finds what it compares with in the even lanes
      // The float compare-and-stores: equal bits, unequal, +0.0 against -0.0 (equal as floats), the same NaN (never equal).
      const uint64_t special[4][2] = {{0x4008000000000000ull, 0x4008000000000000ull}, {0x4008000000000000ull, 0x4010000000000000ull},
                                      {0x0000000000000000ull, 0x8000000000000000ull}, {0x7ff8000000000001ull, 0x7ff8000000000001ull}};
      const uint64_t* sp = special[l % 4];
      m[14] = static_cast<uint32_t>(sp[0]), m[15] = static_cast<uint32_t>(sp[0] >> 32);   // the memory, the data register's the other
      d[12] = static_cast<uint32_t>(sp[1]), d[13] = static_cast<uint32_t>(sp[1] >> 32);
      m[16] = m[14], m[17] = m[15];
    }
    const uint64_t in = put(mem, init), dat = put(mem, data), out = mem.alloc(16900 + 8);
    launch(o, "ds_gaps", mem, {in, dat, out});
    const std::vector<uint32_t> got = get<uint32_t>(mem, out, 16900 / 4 + 1);
    int wrong = 0;
    std::string first;
    const auto check = [&](uint32_t l, const char* what, size_t at, uint32_t want) {
      if (got[at] == want) return;
      if (++wrong == 1) {
        char b[160];
        std::snprintf(b, sizeof b, "%s lane %u: %s (out word %zu) is %08x, wanted %08x", target, l, what, at, got[at], want);
        first = b;
      }
    };
    for (uint32_t l = 0; l < 64; ++l) {
      const uint32_t* d = &data[l * 16];
      std::vector<uint32_t> m(init.begin() + l * 32, init.begin() + l * 32 + 32);
      uint32_t r[24] = {};
      const auto m64 = [&](int at) { return uint64_t{m[at]} | uint64_t{m[at + 1]} << 32; };
      const auto set64 = [&](int at, uint64_t v) { m[at] = static_cast<uint32_t>(v), m[at + 1] = static_cast<uint32_t>(v >> 32); };
      const auto d64 = [&](int at) { return uint64_t{d[at]} | uint64_t{d[at + 1]} << 32; };
      // DS_MSKOR_B32: MEM = (tmp & ~DATA) | DATA2, and the old value back.
      r[0] = m[0];
      m[0] = (m[0] & ~d[0]) | d[1];
      m[1] = (m[1] & ~d[2]) | d[3];
      {
        const uint64_t old = m64(2);
        r[1] = static_cast<uint32_t>(old), r[2] = static_cast<uint32_t>(old >> 32);
        set64(2, (old & ~d64(4)) | d64(6));
        set64(4, (m64(4) & ~d64(8)) | d64(10));
      }
      // DS_WRXCHG2_RTN_B32: addr1 = base + OFFSET0 * 4, addr2 = base + OFFSET1 * 4, the old values back in order.
      r[3] = m[6], r[4] = m[7];
      m[6] = d[0], m[7] = d[1];
      // DS_WRXCHG2_RTN_B64 with offsets 4 and 5: 8-byte elements, bytes 32 and 40 (dwords 8 and 10).
      r[5] = m[8], r[6] = m[9], r[7] = m[10], r[8] = m[11];
      set64(8, d64(4));
      set64(10, d64(6));
      // DS_CMPST_B64: "MEM = tmp == cmp ? src : tmp" with cmp = DATA, src = DATA2.
      if (m64(12) == d64(8)) set64(12, d64(10));
      // The float ones compare as doubles.
      const auto dbl = [](uint64_t b) { return std::bit_cast<double>(b); };
      if (dbl(m64(14)) == dbl(d64(12))) set64(14, d64(14));
      {
        const uint64_t old = m64(16);
        r[10] = static_cast<uint32_t>(old), r[11] = static_cast<uint32_t>(old >> 32);
        if (dbl(old) == dbl(d64(12))) set64(16, d64(14));
      }
      // DS_WRAP_RTN_B32: "MEM = tmp >= DATA ? tmp - DATA : tmp + DATA2".
      r[9] = m[20];
      m[20] = m[20] >= d[2] ? m[20] - d[2] : m[20] + d[3];
      // DS_CONDXCHG32_RTN_B64: each dword is written, without its top bit, where DATA's top bit says.
      r[12] = m[22], r[13] = m[23];
      if (d[4] >> 31) m[22] = d[4] & 0x7fffffffu;
      if (d[5] >> 31) m[23] = d[5] & 0x7fffffffu;
      // The ST64 exchanges, in B and C: a stride of 64 elements.
      const uint32_t b0 = init[2048 + l], b1 = init[2112 + l];
      const uint64_t c0 = q(init, 2176 + 2 * l), c1 = q(init, 2304 + 2 * l);
      r[14] = b0, r[15] = b1;
      r[16] = static_cast<uint32_t>(c0), r[17] = static_cast<uint32_t>(c0 >> 32);
      r[18] = static_cast<uint32_t>(c1), r[19] = static_cast<uint32_t>(c1 >> 32);
      for (int k = 0; k < 20; ++k) check(l, "a returned value", l * 24 + k, r[k]);
      for (int k = 0; k < 32; ++k) check(l, "a word of A", 1536 + l * 32 + k, m[k]);
      check(l, "B's first word", 3584 + l, d[0]);
      check(l, "B's second word", 3584 + 64 + l, d[1]);
      check(l, "C's first quadword (low)", 3712 + 2 * l, d[4]);
      check(l, "C's first quadword (high)", 3712 + 2 * l + 1, d[5]);
      check(l, "C's second quadword (low)", 3712 + 128 + 2 * l, d[6]);
      check(l, "C's second quadword (high)", 3712 + 128 + 2 * l + 1, d[7]);
      // DS_APPEND adds the number of active lanes (48) to the counter (1000) and gives every active lane what it found; DS_CONSUME takes it away.
      if (l < 48) {
        check(l, "ds_append's result", 3968 + l * 3, 1000);
        check(l, "ds_consume's result", 3968 + l * 3 + 1, 1048);
        check(l, "the second ds_consume's result", 3968 + l * 3 + 2, 1000);
      } else {
        check(l, "an off lane's ds_append result", 3968 + l * 3, 0);
      }
      check(l, "the ADDTID readback", 4160 + l, d[2]);
    }
    check(0, "the counter", 4224, 1000 + 48 - 48 - 48);
    if (wrong) std::fprintf(stderr, "%s: %d wrong; first: %s\n", target, wrong, first.c_str());
    VCHECK_EQ(wrong, 0);
  }
}

// Scalar stores and atomics through a buffer resource, the cache instructions, vector GPR indexing's index and mode, VSKIP, S_SETREG_B32, the
// debug branches, the performance and trace instructions, and S_SETKILL (CDNA2, CDNA3 and CDNA4 ISA, 12.5 and 12.6).
VTEST(scalar_buffer_stores_and_atomics_indexing_vskip_setreg_and_setkill_do_what_the_guides_say) {
  // The data the kernel (asm_cdna_scalar.s) puts in registers.
  const uint32_t d32[13] = {0x13572468, 0x00001000, 0x00000234, 0x80000005, 0x00000007, 0xfffffff0, 0x00000009,
                            0xff00ff00, 0x0f0f0000, 0xaaaa5555, 5,          100,        0xcafebabe};
  const uint32_t cmp = 0x24681357;
  const uint64_t d64[4] = {0x1122334455667788ull, 0x0000000100000001ull, 0xff00ff00ff00ff00ull, 0xdeadbeefdeadbeefull};
  const uint64_t cmp64 = 0x0102030405060708ull;
  for (const char* target : kTargets) {
    const amd::CodeObject o = object("asm_cdna_scalar", target);
    MemoryManager mem(16ull << 20);
    std::vector<uint32_t> buf(128);
    uint32_t seed = 12345;
    for (uint32_t& x : buf) x = (seed = seed * 1664525u + 1013904223u);
    buf[16 + 3] = 0x80000001u;   // smin of it and 0x80000005: both negative; the smaller is the 1
    buf[16 + 4] = 0x00000005u;   // umin(5, 7) = 5
    buf[16 + 5] = 0x00000003u;   // smax(3, -16) = 3
    buf[16 + 10] = 5;            // inc: 5 >= 5 gives 0
    buf[16 + 11] = 0;            // dec: 0 gives the data, 100
    buf[16 + 12] = cmp;          // cmpswap finds what it compares with
    const auto q = [&](int at) { return uint64_t{buf[at]} | uint64_t{buf[at + 1]} << 32; };
    const auto setq = [&](int at, uint64_t v) { buf[at] = static_cast<uint32_t>(v), buf[at + 1] = static_cast<uint32_t>(v >> 32); };
    setq(32 + 6, cmp64);
    const std::vector<uint32_t> before = buf;
    const uint64_t b = put(mem, buf), out = mem.alloc(36 * 4);
    launch(o, "scalar_gaps", mem, {b, out});
    const std::vector<uint32_t> got = get<uint32_t>(mem, b, 128), res = get<uint32_t>(mem, out, 36);
    std::vector<uint32_t> want = before, ret(36, 0);
    want[0] = 0x11111111;
    want[2] = 0x22222222, want[3] = 0x33333333;
    want[4] = 0x44444444, want[5] = 0x55555555, want[6] = 0x66666666, want[7] = 0x77777777;
    const auto s32 = [](uint32_t x) { return static_cast<int32_t>(x); };
    for (int i = 0; i < 13; ++i) {
      const uint32_t m = before[16 + i], v = d32[i];
      uint32_t now = m;
      switch (i) {
        case 0: now = v; break;
        case 1: now = m + v; break;
        case 2: now = m - v; break;
        case 3: now = s32(v) < s32(m) ? v : m; break;
        case 4: now = v < m ? v : m; break;
        case 5: now = s32(v) > s32(m) ? v : m; break;
        case 6: now = v > m ? v : m; break;
        case 7: now = m & v; break;
        case 8: now = m | v; break;
        case 9: now = m ^ v; break;
        case 10: now = m >= v ? 0 : m + 1; break;
        case 11: now = (m == 0 || m > v) ? v : m - 1; break;
        case 12: now = m == cmp ? v : m; break;
      }
      want[16 + i] = now;
      ret[i] = m;
    }
    for (int j = 0; j < 4; ++j) {
      const uint64_t m = uint64_t{before[32 + 2 * j]} | uint64_t{before[33 + 2 * j]} << 32;
      uint64_t now = m;
      if (j == 0) now = d64[0];
      if (j == 1) now = m + d64[1];
      if (j == 2) now = m & d64[2];
      if (j == 3) now = m == cmp64 ? d64[3] : m;
      want[32 + 2 * j] = static_cast<uint32_t>(now), want[33 + 2 * j] = static_cast<uint32_t>(now >> 32);
      ret[13 + 2 * j] = static_cast<uint32_t>(m), ret[14 + 2 * j] = static_cast<uint32_t>(m >> 32);
    }
    (void)q;
    // Vector GPR indexing: s_set_gpr_idx_on (index 1, source 0) reads v5, s_set_gpr_idx_idx 2 then reads v6, s_set_gpr_idx_mode DST writes v12 + 2.
    ret[21] = 222, ret[22] = 333, ret[23] = 0xaaaa, ret[24] = 111, ret[25] = 111;
    // VSKIP: the store and the move in between are not issued; the scalar move is; afterwards vector instructions run again.
    ret[26] = 0, ret[27] = 3, ret[28] = 9, ret[29] = 5;
    ret[30] = 0xd;   // S_SETREG_B32 then S_GETREG_B32 of MODE[3:0]
    ret[31] = 4;     // the four debug branches fall through
    ret[32] = 1;     // the counters, the trace, HALT cleared, KILL cleared and V_CLREXCP change nothing
    ret[33] = 0x11;  // S_SETKILL 1 ends the wave: the second value is never stored
    ret[34] = 0;
    int wrong = 0;
    for (size_t i = 0; i < 128; ++i) wrong += got[i] != want[i];
    for (size_t i = 0; i < 35; ++i) wrong += res[i] != ret[i];
    if (wrong) {
      std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
      for (size_t i = 0; i < 128; ++i)
        if (got[i] != want[i]) std::fprintf(stderr, "  buf word %zu is %08x, wanted %08x\n", i, got[i], want[i]);
      for (size_t i = 0; i < 35; ++i)
        if (res[i] != ret[i]) std::fprintf(stderr, "  out word %zu is %08x, wanted %08x\n", i, res[i], ret[i]);
    }
    VCHECK_EQ(wrong, 0);
  }
}

namespace {
uint16_t half_bits(double v) {
  const _Float16 h = static_cast<_Float16>(v);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
double half_value(uint16_t b) {
  _Float16 h;
  std::memcpy(&h, &b, 2);
  return static_cast<double>(h);
}
uint32_t f32_bits(float f) { return std::bit_cast<uint32_t>(f); }
}  // namespace

// The vector instructions of the CDNA2 guide that gfx90a's executor refused (and gfx942's and gfx950's guides list with the same meaning): sine and
// cosine of a half in turns with the guide's own examples, the cube-map instructions from its pseudocode, V_CVT_RPI_I32_F32, V_CVT_OFF_F32_I4's
// table, frexp and ldexp of halves, the fused half multiply-add, the 16-bit multiply-add whose high half is zeroed, the packed FMAC and the saturating pack.
VTEST(gfx90a_sine_cube_map_frexp_ldexp_fused_half_and_saturating_pack_instructions_follow_the_guide) {
  for (const char* target : kTargets) {
    const amd::CodeObject o = object("asm_cdna_valu", target);
    MemoryManager mem(16ull << 20);
    // The guide's examples for V_SIN_F16 and V_COS_F16 (CDNA2 ISA 12.8): input, sin, cos.
    const uint16_t ex[6][3] = {{0xfc00, 0xfe00, 0xfe00}, {0xfbff, 0x0000, 0x3c00}, {0x8000, 0x8000, 0x3c00},
                               {0x3400, 0x3c00, 0x0000}, {0x7bff, 0x0000, 0x3c00}, {0x7c00, 0xfe00, 0xfe00}};
    const uint16_t more[4] = {0x3c00, 0x4200, 0xc500, 0x0000};   // 1.0, 3.0, -5.0, 0 (for frexp, which has no examples here)
    const float coords[9] = {-4.0f, -2.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.5f, 4.0f};
    const int ints[11] = {-20, -7, -3, -1, 0, 1, 2, 5, 9, 13, 20};
    const uint16_t sat_hi[7] = {0xff9c, 0x0000, 0x0064, 0x00ff, 0x012c, 0x7fff, 0x8000};   // -100, 0, 100, 255, 300, 32767, -32768
    uint32_t seed = 777;
    const auto rnd = [&](uint32_t n) { return (seed = seed * 1664525u + 1013904223u) >> 8, (seed >> 8) % n; };
    std::vector<uint32_t> in(64 * 8);
    for (uint32_t l = 0; l < 64; ++l) {
      uint32_t* k = &in[l * 8];
      k[0] = l % 10 < 6 ? ex[l % 10][0] : more[l % 10 - 6];
      for (int i = 1; i <= 3; ++i) k[i] = f32_bits(coords[rnd(9)]);
      for (int i = 4; i <= 6; ++i) k[i] = half_bits(ints[rnd(11)]) | uint32_t{half_bits(ints[rnd(11)])} << 16;
      k[7] = static_cast<uint32_t>(rnd(7)) - 3 & 0xFFFF | uint32_t{sat_hi[rnd(7)]} << 16;
    }
    const uint64_t i0 = put(mem, in), out = mem.alloc(64 * 16 * 4);
    launch(o, "valu_gaps", mem, {i0, out});
    const std::vector<uint32_t> got = get<uint32_t>(mem, out, 64 * 16);
    int wrong = 0;
    std::string first;
    const auto check = [&](uint32_t l, const char* what, int slot, uint32_t have, uint32_t want, uint32_t mask = 0xFFFFFFFFu) {
      if ((have & mask) == (want & mask)) return;
      if (++wrong == 1) {
        char b[160];
        std::snprintf(b, sizeof b, "%s lane %u: %s is %08x, wanted %08x (slot %d)", target, l, what, have, want, slot);
        first = b;
      }
    };
    for (uint32_t l = 0; l < 64; ++l) {
      const uint32_t* k = &in[l * 8];
      const uint32_t* r = &got[l * 16];
      if (l % 10 < 6) {
        check(l, "v_sin_f16", 0, r[0], ex[l % 10][1], 0xFFFF);
        check(l, "v_cos_f16", 1, r[1], ex[l % 10][2], 0xFFFF);
      }
      // The cube-map instructions: the guide's pseudocode on (x, y, z).
      const float x = std::bit_cast<float>(k[1]), y = std::bit_cast<float>(k[2]), z = std::bit_cast<float>(k[3]);
      float id, sc, tc, ma;
      if (std::fabs(z) >= std::fabs(x) && std::fabs(z) >= std::fabs(y)) {
        id = z < 0 ? 5.0f : 4.0f, sc = z < 0 ? -x : x, tc = -y, ma = 2.0f * z;
      } else if (std::fabs(y) >= std::fabs(x)) {
        id = y < 0 ? 3.0f : 2.0f, sc = x, tc = y < 0 ? -z : z, ma = 2.0f * y;
      } else {
        id = x < 0 ? 1.0f : 0.0f, sc = x < 0 ? z : -z, tc = -y, ma = 2.0f * x;
      }
      check(l, "v_cubeid_f32", 2, r[2], f32_bits(id));
      check(l, "v_cubesc_f32", 3, r[3], f32_bits(sc));
      check(l, "v_cubetc_f32", 4, r[4], f32_bits(tc));
      check(l, "v_cubema_f32", 5, r[5], f32_bits(ma));
      check(l, "v_cvt_rpi_i32_f32", 6, r[6], static_cast<uint32_t>(static_cast<int32_t>(std::floor(x + 0.5f))));
      check(l, "v_cvt_off_f32_i4", 7, r[7], f32_bits(static_cast<float>((k[0] & 8) ? static_cast<int>(k[0] & 15) - 16 : static_cast<int>(k[0] & 15)) / 16.0f));
      // frexp: the significand in [0.5, 1) and the exponent (an infinity or a NaN gives itself, and 0).
      const uint16_t h = static_cast<uint16_t>(k[0]);
      const double hv = half_value(h);
      if (std::isfinite(hv)) {
        int e = 0;
        const double m = std::frexp(hv, &e);
        check(l, "v_frexp_mant_f16", 8, r[8], half_bits(m), 0xFFFF);
        if (hv != 0) check(l, "v_frexp_exp_i16_f16", 9, r[9], static_cast<uint32_t>(e), 0xFFFF);
      } else {
        check(l, "v_frexp_mant_f16 of a non-finite", 8, r[8], h, std::isnan(hv) ? 0x7E00u : 0xFFFFu);
        check(l, "v_frexp_exp_i16_f16 of a non-finite", 9, r[9], 0, 0xFFFF);
      }
      const double a = half_value(static_cast<uint16_t>(k[4])), bb = half_value(static_cast<uint16_t>(k[5])), c = half_value(static_cast<uint16_t>(k[6]));
      check(l, "v_ldexp_f16", 10, r[10], half_bits(std::ldexp(a, static_cast<int16_t>(k[7]))), 0xFFFF);
      check(l, "v_fma_legacy_f16", 11, r[11], half_bits(a * bb + c), 0xFFFF);
      // V_MAD_LEGACY_I16: "D.i16 = S0.i16 * S1.i16 + S2.i16", the high half zero (the registers hold the halves' bit patterns).
      check(l, "v_mad_legacy_i16", 12, r[12], static_cast<uint16_t>(static_cast<int16_t>(k[4]) * static_cast<int16_t>(k[5]) + static_cast<int16_t>(k[6])));
      uint32_t pk = 0;
      for (int half = 0; half < 2; ++half)
        pk |= uint32_t{half_bits(half_value(static_cast<uint16_t>(k[4] >> (16 * half))) * half_value(static_cast<uint16_t>(k[5] >> (16 * half))) +
                                 half_value(static_cast<uint16_t>(k[6] >> (16 * half))))}
              << (16 * half);
      check(l, "v_pk_fmac_f16", 13, r[13], pk);
      const auto sat8 = [](int16_t n) -> uint32_t { return n <= 0 ? 0u : n >= 255 ? 255u : static_cast<uint32_t>(n); };
      check(l, "v_sat_pk_u8_i16", 14, r[14], sat8(static_cast<int16_t>(k[7])) | sat8(static_cast<int16_t>(k[7] >> 16)) << 8);
    }
    if (wrong) std::fprintf(stderr, "%s: %d wrong; first: %s\n", target, wrong, first.c_str());
    VCHECK_EQ(wrong, 0);
  }
}

// Formatted buffer accesses (CDNA2, CDNA3 and CDNA4 ISA): MUBUF's BUFFER_LOAD_FORMAT_* and BUFFER_STORE_FORMAT_* take their format from the resource,
// MTBUF's TBUFFER_* from the instruction; each channel is converted to (or from) a 32-bit integer or float in a register, or to a half with d16.
// gfx942 and gfx950 had no decoder rows for them before.
VTEST(formatted_buffer_loads_and_stores_convert_by_the_resources_or_the_instructions_format) {
  for (const char* target : kTargets) {
    const amd::CodeObject o = object("asm_cdna_formats", target);
    MemoryManager mem(16ull << 20);
    std::vector<uint32_t> src(64 * 4);
    for (uint32_t l = 0; l < 64; ++l) {
      src[4 * l] = f32_bits(static_cast<float>(l) + 0.5f);
      src[4 * l + 1] = f32_bits(static_cast<float>(l) + 1.5f);
      src[4 * l + 2] = f32_bits(static_cast<float>(l) + 2.5f);
      src[4 * l + 3] = l % 2 ? 0x3f8000ffu : 0x3f800000u;   // the low byte is 255 or 0: an 8-bit unorm's 1.0 or 0.0
    }
    const uint64_t in = put(mem, src), out = mem.alloc(64 * 32), out2 = mem.alloc(1024 + 64 * 8);
    launch(o, "fmt", mem, {in, out, out2});
    const std::vector<uint32_t> got = get<uint32_t>(mem, out, 64 * 8), st = get<uint32_t>(mem, out2, 256 + 128);
    int wrong = 0;
    for (uint32_t l = 0; l < 64; ++l) {
      const uint32_t* w = &src[4 * l];
      const uint32_t want[8] = {w[0], w[1], w[2], w[3], w[0] & 0xFFFF, w[0] >> 16, l % 2 ? 0x3f800000u : 0u, half_bits(static_cast<float>(l) + 0.5f)};
      for (int k = 0; k < 8; ++k) wrong += (got[l * 8 + k] & (k == 7 ? 0xFFFFu : 0xFFFFFFFFu)) != want[k];
      for (int k = 0; k < 4; ++k) wrong += st[l * 4 + k] != w[k];
      wrong += st[256 + l * 2] != (w[0] & 0xFFFF);
      wrong += st[256 + l * 2 + 1] != (w[0] >> 16);
    }
    if (wrong) std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
    VCHECK_EQ(wrong, 0);
  }
}

namespace {
uint16_t bf16_bits(float f) { return static_cast<uint16_t>(std::bit_cast<uint32_t>(f) >> 16); }
float bf16_value(uint16_t b) { return std::bit_cast<float>(uint32_t{b} << 16); }
}  // namespace

// gfx940's packed atomics (CDNA3 and CDNA4 ISA, DS_PK_ADD_F16 and its bfloat16 and _RTN forms, BUFFER_ATOMIC_PK_ADD_F16, gfx950's
// BUFFER_ATOMIC_PK_ADD_BF16: "dst[31:16].f16 = tmp[31:16].f16 + src[31:16].f16; dst[15:0].f16 = tmp[15:0].f16 + src[15:0].f16"). The values are
// multiples of 1/8 below 16, whose sums every one of the formats holds exactly.
VTEST(packed_half_and_bfloat16_atomics_add_each_half_and_hand_back_the_old_pair) {
  for (const char* target : {"gfx942", "gfx950"}) {
    const bool bf_buffer = std::string(target) == "gfx950";
    const amd::CodeObject o = object("asm_cdna_pk", target);
    MemoryManager mem(16ull << 20);
    uint32_t seed = 4242;
    const auto eighth = [&] { return (static_cast<int>((seed = seed * 1664525u + 1013904223u) >> 16) % 129 - 64) / 8.0; };
    const auto pk_f16 = [&] { const uint32_t lo = half_bits(eighth()); return lo | uint32_t{half_bits(eighth())} << 16; };
    const auto pk_bf16 = [&] { const uint32_t lo = bf16_bits(static_cast<float>(eighth())); return lo | uint32_t{bf16_bits(static_cast<float>(eighth()))} << 16; };
    std::vector<uint32_t> buf(384), vals(128);
    for (uint32_t l = 0; l < 64; ++l) {
      buf[4 * l] = pk_f16(), buf[4 * l + 1] = pk_f16(), buf[4 * l + 2] = pk_bf16(), buf[4 * l + 3] = pk_bf16();
      buf[256 + l] = pk_f16();
      buf[320 + l] = pk_bf16();
      vals[2 * l] = pk_f16(), vals[2 * l + 1] = pk_bf16();
    }
    const uint64_t b = put(mem, buf), v = put(mem, vals), out = mem.alloc(4096 + 512);
    launch(o, "pk_atomics", mem, {b, v, out});
    const std::vector<uint32_t> got = get<uint32_t>(mem, out, (4096 + 512) / 4), after = get<uint32_t>(mem, b, 384);
    const auto add_f16 = [&](uint32_t a, uint32_t d) {
      uint32_t r = 0;
      for (int h = 0; h < 2; ++h)
        r |= uint32_t{half_bits(half_value(static_cast<uint16_t>(a >> (16 * h))) + half_value(static_cast<uint16_t>(d >> (16 * h))))} << (16 * h);
      return r;
    };
    const auto add_bf16 = [&](uint32_t a, uint32_t d) {
      uint32_t r = 0;
      for (int h = 0; h < 2; ++h)
        r |= uint32_t{bf16_bits(bf16_value(static_cast<uint16_t>(a >> (16 * h))) + bf16_value(static_cast<uint16_t>(d >> (16 * h))))} << (16 * h);
      return r;
    };
    int wrong = 0;
    for (uint32_t l = 0; l < 64; ++l) {
      const uint32_t* m = &buf[4 * l];
      const uint32_t d0 = vals[2 * l], d1 = vals[2 * l + 1];
      wrong += got[4 * l + 0] != add_f16(m[0], d0);
      wrong += got[4 * l + 1] != add_f16(m[1], d0);
      wrong += got[4 * l + 2] != add_bf16(m[2], d1);
      wrong += got[4 * l + 3] != add_bf16(m[3], d1);
      wrong += got[512 + 2 * l] != m[1];   // the _RTN forms' old pairs
      wrong += got[512 + 2 * l + 1] != m[3];
      wrong += got[1024 + l] != buf[256 + l];   // the buffer atomic's old pair
      wrong += after[256 + l] != add_f16(buf[256 + l], d0);
      wrong += got[1024 + 64 + l] != (bf_buffer ? buf[320 + l] : 0u);
      wrong += after[320 + l] != (bf_buffer ? add_bf16(buf[320 + l], d1) : buf[320 + l]);
    }
    if (wrong) std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
    VCHECK_EQ(wrong, 0);
  }
}

// scratch_load_lds_* (CDNA3 and CDNA4 ISA 10.3: the data goes to LDS at M0[17:2] * 4 + INST.OFFSET + ThreadID * 4, a byte or a half extended to a
// dword as the global forms' are): this segment used to be refused by a crash, from reading a destination register the instruction has not got.
VTEST(scratch_loads_into_lds_land_as_dwords) {
  for (const char* target : {"gfx942", "gfx950"}) {
    const amd::CodeObject o = object("asm_cdna_pk", target);
    MemoryManager mem(16ull << 20);
    const uint64_t out = mem.alloc(5 * 256);
    launch(o, "scratch_lds", mem, {out});
    const std::vector<uint32_t> got = get<uint32_t>(mem, out, 5 * 64);
    int wrong = 0;
    for (uint32_t l = 0; l < 64; ++l) {
      const uint32_t byte = (l + 0x80) & 0xFF, half = (0x8000 + l) & 0xFFFF;
      wrong += got[l] != 7 * l + 1;
      wrong += got[64 + l] != byte;
      wrong += got[128 + l] != static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(byte)));
      wrong += got[192 + l] != half;
      wrong += got[256 + l] != static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(half)));
    }
    if (wrong) std::fprintf(stderr, "%s: %d wrong\n", target, wrong);
    VCHECK_EQ(wrong, 0);
  }
}

VTEST_MAIN
