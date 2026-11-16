// The approximate-math instructions (MUFU) against an RTX 3060's results (docs/numerics/math-functions.md).
//
// The model (include/vgpu/exec/mufu_model.hpp) was fitted to the card and checked against it on all 2^32 inputs of each of the
// ten sub-operations by the card probe (tools/numerics/mufu_probe.cu --check). What a machine without a card can hold:
//
//  * the model, on the card's results for 310 operands an operation (nvidia/tests/data/mufu/rtx3060.vectors.txt), and on the
//    card's hashes of 2^26 consecutive operands (tools/numerics/mufu/rtx3060.hashes.txt): one chunk an operation by default,
//    all 64 of each -- every one of the 2^32 inputs -- with VGPU_EXHAUSTIVE=1;
//  * the SASS executor, on the same vectors: it runs the card probe's own kernel (the cubin nvcc 13.0 made of
//    tools/numerics/mufu_kernel.cu for sm_86, with the MUFU sub-operation rewritten as the probe rewrites it), through the
//    32-lane fast path, the general path (VGPU_FASTPATH=0) and with the |x| and -x operand modifiers;
//  * the PTX interpreter, on the vectors (the forms ptxas makes one instruction of) and on the card's results for every PTX
//    approximate form where ptxas puts more around the instruction (nvidia/tests/data/mufu/rtx3060.forms.txt, from
//    tools/numerics/mufu_forms.cu: subnormal operands, |x| above 2^126, x below -126, the four divisions);
//  * VGPU_MUFU_EXACT=0: both engines give the function at the host's precision again;
//  * the half-precision forms (MUFU.<op>.F16 and the PTX ex2.approx.f16/.f16x2, tanh.approx.f16/.f16x2): the model, the SASS executor
//    and the PTX interpreter against the card's hashes over every one of the 65,536 half patterns of each form
//    (tools/numerics/mufu/rtx3060.f16.hashes.txt).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/mufu_model.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

const char* const kOps[] = {"COS", "SIN", "EX2", "LG2", "RCP", "RSQ", "RCP64H", "RSQ64H", "SQRT", "TANH"};
int op_of(const std::string& name) {
  for (int i = 0; i < 10; ++i)
    if (name == kOps[i]) return i;
  throw vtest::Failure("unknown operation " + name);
}
std::string hex(uint64_t v, int digits = 8) {
  char b[32];
  std::snprintf(b, sizeof b, "%0*llx", digits, static_cast<unsigned long long>(v));
  return b;
}
bool exhaustive() {
  const char* e = std::getenv("VGPU_EXHAUSTIVE");
  return e && *e && *e != '0';
}
std::vector<uint8_t> bytes_of(const void* p, size_t n) {
  std::vector<uint8_t> b(n);
  std::memcpy(b.data(), p, n);
  return b;
}

// A file of sections: "= <name>" starts one, "#" a comment, every other line a row of hexadecimal words.
std::map<std::string, std::vector<std::vector<uint64_t>>> sections(const std::string& path) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/" + path);
  if (!in) throw vtest::Failure("cannot read " + path);
  std::map<std::string, std::vector<std::vector<uint64_t>>> out;
  std::vector<std::vector<uint64_t>>* cur = nullptr;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    if (line[0] == '=') {
      cur = &out[line.substr(2)];
      continue;
    }
    if (!cur) throw vtest::Failure(path + ": a row before any section");
    std::istringstream ss(line);
    std::vector<uint64_t> row;
    for (std::string w; ss >> w;) row.push_back(std::stoull(w, nullptr, 16));
    cur->push_back(row);
  }
  return out;
}
const std::map<std::string, std::vector<std::vector<uint64_t>>>& vectors() {
  static const auto v = sections("nvidia/tests/data/mufu/rtx3060.vectors.txt");
  return v;
}

// ---- the SASS executor on the probe's kernel -----------------------------------------------------------------------------

// The cubin of tools/numerics/mufu_kernel.cu with every MUFU's sub-operation (bits 74-77) and operand modifiers (|x|: bit 62,
// -x: bit 63) rewritten, as tools/numerics/mufu_probe.cu rewrites it for the card.
// sm 90 and later: the cubin's architecture is patched to `sm` (e_flags) and the half bit is the one of that architecture: .F16 is
// instruction bit 73 up to sm_89 and bit 72 from sm_90, where bit 73 is .BF16 (`bf16`).
std::vector<uint8_t> probe_cubin(unsigned op, bool src_abs, bool src_neg, bool f16 = false, unsigned part = 0, int sm = 86, bool bf16 = false) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/mufu/mufu_kernel.sm_86.cubin", std::ios::binary);
  if (!in) throw vtest::Failure("cannot read mufu_kernel.sm_86.cubin");
  std::vector<uint8_t> elf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (sm != 86) {   // the architecture byte of e_flags (ABI version 8: bits 8-15; version 7: bits 0-7)
    VCHECK(elf[7] != 0x33);
    elf[0x31] = static_cast<uint8_t>(sm);
  }
  const auto rd = [&](size_t off, int bytes) { uint64_t v = 0; std::memcpy(&v, elf.data() + off, bytes); return v; };
  const uint64_t shoff = rd(0x28, 8);
  const unsigned shentsize = rd(0x3a, 2), shnum = rd(0x3c, 2), shstrndx = rd(0x3e, 2);
  const uint64_t stroff = rd(shoff + static_cast<uint64_t>(shstrndx) * shentsize + 0x18, 8);
  int count = 0;
  for (unsigned i = 0; i < shnum; ++i) {
    const uint64_t sh = shoff + static_cast<uint64_t>(i) * shentsize;
    if (std::strncmp(reinterpret_cast<const char*>(elf.data() + stroff + rd(sh, 4)), ".text.", 6) != 0) continue;
    const uint64_t off = rd(sh + 0x18, 8), size = rd(sh + 0x20, 8);
    for (uint64_t p = off; p + 16 <= off + size; p += 16) {
      uint64_t lo = rd(p, 8), hi = rd(p + 8, 8);
      if ((lo & 0x1ff) != 0x108) continue;
      hi = (hi & ~(0xfull << 10)) | (static_cast<uint64_t>(op) << 10);
      hi = (hi & ~(3ull << 8)) | (static_cast<uint64_t>(sm >= 90 ? (f16 ? 1 : 0) | (bf16 ? 2 : 0) : (f16 ? 2 : 0)) << 8);   // bit 73: .F16 (up to sm_89); bit 72 .F16, bit 73 .BF16 (sm_90 and later)
      lo = (lo & ~(3ull << 60)) | (static_cast<uint64_t>(part) << 60);   // bits 60-61: the half read (.H1)
      lo = (lo & ~(3ull << 62)) | (static_cast<uint64_t>(src_abs) << 62) | (static_cast<uint64_t>(src_neg) << 63);
      std::memcpy(elf.data() + p, &lo, 8);
      std::memcpy(elf.data() + p + 8, &hi, 8);
      ++count;
    }
  }
  VCHECK_EQ(count, 6);   // five in mufu_hash's unrolled loop, one in mufu_dump
  return elf;
}

struct Sass {
  MemoryManager mem{1 << 26};
  DeviceProfile prof;
  std::shared_ptr<sass::Module> m;
  uint64_t out;
  Sass(unsigned op, bool src_abs = false, bool src_neg = false, bool f16 = false, unsigned part = 0, int sm = 86, bool bf16 = false) {
    const std::vector<uint8_t> c = probe_cubin(op, src_abs, src_neg, f16, part, sm, bf16);
    prof = load_gpu(sm >= 90 ? "nvidia/h100-pcie" : "nvidia/rtx3060");
    m = sass::load(c.data(), c.size(), mem, prof);
    out = mem.alloc(1 << 20);
  }
  // mufu_dump: out[i] = MUFU(base + i), i < n.
  std::vector<uint32_t> dump(uint32_t base, uint32_t n) {
    exec::LaunchConfig cfg;
    cfg.grid = {(n + 255) / 256, 1, 1};
    cfg.block = {256, 1, 1};
    sass::launch(*m, "mufu_dump", cfg, {bytes_of(&out, 8), bytes_of(&base, 4), bytes_of(&n, 4)}, mem, prof);
    std::vector<uint32_t> r(n);
    mem.read(out, r.data(), n * 4);
    return r;
  }
  // mufu_hash: one hash a thread, of 2^log2_block consecutive inputs.
  std::vector<uint64_t> hash(uint32_t base, uint32_t log2_block, uint32_t n) {
    exec::LaunchConfig cfg;
    cfg.grid = {(n + 255) / 256, 1, 1};
    cfg.block = {256, 1, 1};
    sass::launch(*m, "mufu_hash", cfg, {bytes_of(&out, 8), bytes_of(&base, 4), bytes_of(&log2_block, 4), bytes_of(&n, 4)}, mem, prof);
    std::vector<uint64_t> r(n);
    mem.read(out, r.data(), n * 8);
    return r;
  }
};

void sass_on_the_vectors() {
  for (int op = 0; op < 10; ++op) {
    Sass s(static_cast<unsigned>(op));
    for (const auto& row : vectors().at(kOps[op])) {
      const uint32_t got = s.dump(static_cast<uint32_t>(row[0]), 1)[0];
      if (got != row[1])
        throw vtest::Failure(std::string("SASS MUFU.") + kOps[op] + "(" + hex(row[0]) + "): " + hex(got) + ", an RTX 3060 gives " + hex(row[1]));
    }
    // Whole warps (the 32-lane path), across a binade boundary: the model, which the vectors and the hashes hold to the card.
    for (uint32_t base : {0x3f7fff00u, 0xbf7fff80u, 0x007fff80u, 0x7f7fff80u}) {
      const std::vector<uint32_t> r = s.dump(base, 512);
      for (uint32_t i = 0; i < 512; ++i)
        if (r[i] != mufu::eval(static_cast<unsigned>(op), base + i))
          throw vtest::Failure(std::string("SASS MUFU.") + kOps[op] + "(" + hex(base + i) + ") in a full warp: " + hex(r[i]) + ", the model gives " +
                               hex(mufu::eval(static_cast<unsigned>(op), base + i)));
    }
  }
}

// ---- the PTX interpreter -------------------------------------------------------------------------------------------------

// out[i] = <form>(in[i]) (or <form>(a[i], b[i])) for i < n, on registers of `width` bits.
std::vector<uint64_t> run_ptx(const std::string& form, int width, const std::vector<uint64_t>& a, const std::vector<uint64_t>* b = nullptr) {
  const std::string t = width == 64 ? "f64" : "f32", w = std::to_string(width / 8);
  std::string ptx = ".version 8.0\n.target sm_86\n.address_size 64\n.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 po, .param .u32 pn)\n{\n"
                    " .reg .b32 %r<8>; .reg .b64 %rd<10>; .reg ." + t + " %a, %b, %c; .reg .pred %p;\n"
                    " ld.param.u64 %rd1,[pa]; ld.param.u64 %rd2,[pb]; ld.param.u64 %rd3,[po]; ld.param.u32 %r6,[pn];\n"
                    " mov.u32 %r1,%tid.x; mov.u32 %r2,%ctaid.x; mov.u32 %r3,%ntid.x; mad.lo.u32 %r4,%r2,%r3,%r1;\n"
                    " setp.ge.u32 %p,%r4,%r6; @%p bra done;\n"
                    " mul.wide.u32 %rd4,%r4," + w + "; add.u64 %rd5,%rd1,%rd4; add.u64 %rd6,%rd2,%rd4; add.u64 %rd7,%rd3,%rd4;\n"
                    " ld.global." + t + " %a,[%rd5];\n";
  if (b) ptx += " ld.global." + t + " %b,[%rd6];\n " + form + " %c, %a, %b;\n";
  else ptx += " " + form + " %c, %a;\n";
  ptx += " st.global." + t + " [%rd7],%c;\ndone:\n ret;\n}\n";
  MemoryManager mem{1 << 24};
  const DeviceProfile prof = load_gpu("nvidia/rtx3060");
  auto m = ptx::parse(ptx);
  const uint32_t n = static_cast<uint32_t>(a.size());
  const size_t bytes = static_cast<size_t>(n) * (width / 8);
  const uint64_t pa = mem.alloc(bytes), pb = mem.alloc(bytes), po = mem.alloc(bytes);
  for (uint32_t i = 0; i < n; ++i) {
    mem.store_scalar(pa + i * (width / 8), width / 8, a[i]);
    mem.store_scalar(pb + i * (width / 8), width / 8, b ? (*b)[i] : 0);
  }
  exec::LaunchConfig cfg;
  cfg.grid = {(n + 63) / 64, 1, 1};
  cfg.block = {64, 1, 1};
  exec::launch(m.entries[0], cfg, {bytes_of(&pa, 8), bytes_of(&pb, 8), bytes_of(&po, 8), bytes_of(&n, 4)}, mem, prof);
  std::vector<uint64_t> r(n);
  for (uint32_t i = 0; i < n; ++i) r[i] = mem.load_scalar(po + i * (width / 8), width / 8);
  return r;
}

// The PTX form ptxas makes the one instruction of, or nothing (MUFU.SIN and MUFU.COS have none: sin.approx multiplies first).
const char* lone_form(int op) {
  switch (op) {
    case 2: return "ex2.approx.ftz.f32";
    case 3: return "lg2.approx.ftz.f32";
    case 4: return "rcp.approx.ftz.f32";
    case 5: return "rsqrt.approx.ftz.f32";
    case 6: return "rcp.approx.ftz.f64";
    case 7: return "rsqrt.approx.ftz.f64";
    case 8: return "sqrt.approx.ftz.f32";
    case 9: return "tanh.approx.f32";
    default: return nullptr;
  }
}

void ptx_on_the_vectors() {
  for (int op = 0; op < 10; ++op) {
    const char* form = lone_form(op);
    if (!form) continue;
    const bool wide = op == 6 || op == 7;
    std::vector<uint64_t> in;
    for (const auto& row : vectors().at(kOps[op])) in.push_back(wide ? row[0] << 32 | 0x89abcdefu : row[0]);   // (the lower word is not read)
    const std::vector<uint64_t> r = run_ptx(form, wide ? 64 : 32, in);
    size_t i = 0;
    for (const auto& row : vectors().at(kOps[op])) {
      const uint64_t want = wide ? row[1] << 32 : row[1];
      if (r[i] != want)
        throw vtest::Failure(std::string("PTX ") + form + "(" + hex(in[i], wide ? 16 : 8) + "): " + hex(r[i], wide ? 16 : 8) + ", an RTX 3060 gives " +
                             hex(want, wide ? 16 : 8));
      ++i;
    }
  }
}

struct Env {   // a variable set for a scope
  const char* name;
  Env(const char* n, const char* v) : name(n) { setenv(n, v, 1); }
  ~Env() { unsetenv(name); }
};

// A chunk's hash as tools/numerics/mufu_probe.cu defines it, from the model.
uint64_t chunk_hash(unsigned op, uint64_t start, uint64_t* total = nullptr) {
  const uint64_t basis = 0xcbf29ce484222325ull, prime = 0x100000001b3ull;
  uint64_t h = basis;
  for (uint64_t b = start; b < start + (1ull << 26); b += 4096) {
    uint64_t bh = basis;
    for (uint32_t j = 0; j < 4096; ++j) bh = (bh ^ mufu::eval(op, static_cast<uint32_t>(b) + j)) * prime;
    h = (h ^ bh) * prime;
    if (total) *total = (*total ^ bh) * prime;
  }
  return h;
}

// ---- the half-precision forms --------------------------------------------------------------------------------------------

const uint64_t kFnvBasis = 0xcbf29ce484222325ull, kFnvPrime = 0x100000001b3ull;
uint64_t fnv(const std::vector<uint32_t>& v) {
  uint64_t h = kFnvBasis;
  for (uint32_t x : v) h = (h ^ x) * kFnvPrime;
  return h;
}
// The card's half-precision hashes by name ("EX2.F16 all", "ex2.approx.f16x2").
std::map<std::string, uint64_t> hash_file(const std::string& name) {
  {
    std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/tools/numerics/mufu/" + name);
    if (!in) throw vtest::Failure("cannot read " + name);
    std::map<std::string, uint64_t> r;
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      std::string name, a, b;
      ss >> name >> a >> b;
      r[name] = std::stoull(b.empty() ? a : b, nullptr, 16);
    }
    return r;
  }
}
std::map<std::string, uint64_t> f16_hashes() {
  static const auto m = hash_file("rtx3060.f16.hashes.txt");
  return m;
}
// The H100's: the bfloat16 forms ("EX2.BF16 all", "ex2.approx.bf16x2") and the f16 forms ptxas emits for sm_90 (bit 72).
std::map<std::string, uint64_t> bf16_hashes() {
  static const auto m = hash_file("h100.bf16.hashes.txt");
  return m;
}
std::map<std::string, uint64_t> h100_f16_ptx_hashes() {
  static const auto m = hash_file("h100.f16-ptx.hashes.txt");
  return m;
}
const char* const kHalfOps[] = {"COS", "SIN", "EX2", "LG2", "RCP", "RSQ", "SQRT", "TANH"};
unsigned half_op(const std::string& name) { return static_cast<unsigned>(op_of(name)); }
std::vector<uint32_t> all_halves() {
  std::vector<uint32_t> a(65536);
  for (uint32_t i = 0; i < 65536; ++i) a[i] = i;
  return a;
}
// The registers of the .f16x2 forms: x in the low half, (x * 40503 + 12345) mod 65536 in the high half.
std::vector<uint32_t> half_pairs() {
  std::vector<uint32_t> a(65536);
  for (uint32_t i = 0; i < 65536; ++i) a[i] = i | (((i * 40503u + 12345u) & 0xffffu) << 16);
  return a;
}
std::vector<uint32_t> model_halves(unsigned op, bool packed) {
  std::vector<uint32_t> r(65536);
  const std::vector<uint32_t> in = packed ? half_pairs() : all_halves();
  for (uint32_t i = 0; i < 65536; ++i)
    r[i] = packed ? mufu::eval_f16(op, in[i] & 0xffffu) | (mufu::eval_f16(op, in[i] >> 16) << 16) : mufu::eval_f16(op, in[i]);
  return r;
}
std::vector<uint32_t> model_bf16(unsigned op, bool packed) {
  std::vector<uint32_t> r(65536);
  const std::vector<uint32_t> in = packed ? half_pairs() : all_halves();
  for (uint32_t i = 0; i < 65536; ++i)
    r[i] = packed ? mufu::eval_bf16(op, in[i] & 0xffffu) | (mufu::eval_bf16(op, in[i] >> 16) << 16) : mufu::eval_bf16(op, in[i]);
  return r;
}

// out[i] = <form>(in[i]) of a half form on b16 (ex2.approx.f16) or b32 (.f16x2) registers, in the order of `in`.
std::vector<uint32_t> run_ptx_half(const std::string& form, bool packed, const std::vector<uint32_t>& in, const char* target = "sm_86",
                                   const char* gpu = "nvidia/rtx3060") {
  const std::string rt = packed ? "b32" : "b16", w = packed ? "4" : "2";
  std::string ptx = ".version 8.0\n.target " + std::string(target) + "\n.address_size 64\n.visible .entry k(.param .u64 pa, .param .u64 po, .param .u32 pn)\n{\n"
                    " .reg .b32 %r<8>; .reg .b64 %rd<10>; .reg ." + rt + " %a, %c; .reg .pred %p;\n"
                    " ld.param.u64 %rd1,[pa]; ld.param.u64 %rd3,[po]; ld.param.u32 %r6,[pn];\n"
                    " mov.u32 %r1,%tid.x; mov.u32 %r2,%ctaid.x; mov.u32 %r3,%ntid.x; mad.lo.u32 %r4,%r2,%r3,%r1;\n"
                    " setp.ge.u32 %p,%r4,%r6; @%p bra done;\n"
                    " mul.wide.u32 %rd4," ;
  ptx += "%r4," + w + "; add.u64 %rd5,%rd1,%rd4; add.u64 %rd7,%rd3,%rd4;\n"
         " ld.global." + rt + " %a,[%rd5];\n " + form + " %c, %a;\n st.global." + rt + " [%rd7],%c;\ndone:\n ret;\n}\n";
  MemoryManager mem{1 << 24};
  const DeviceProfile prof = load_gpu(gpu);
  auto m = ptx::parse(ptx);
  const uint32_t n = static_cast<uint32_t>(in.size()), bytes = n * (packed ? 4u : 2u);
  const uint64_t pa = mem.alloc(bytes), po = mem.alloc(bytes);
  for (uint32_t i = 0; i < n; ++i) mem.store_scalar(pa + i * (packed ? 4u : 2u), packed ? 4 : 2, in[i]);
  exec::LaunchConfig cfg;
  cfg.grid = {(n + 255) / 256, 1, 1};
  cfg.block = {256, 1, 1};
  exec::launch(m.entries[0], cfg, {bytes_of(&pa, 8), bytes_of(&po, 8), bytes_of(&n, 4)}, mem, prof);
  std::vector<uint32_t> r(n);
  for (uint32_t i = 0; i < n; ++i) r[i] = static_cast<uint32_t>(mem.load_scalar(po + i * (packed ? 4u : 2u), packed ? 4 : 2));
  return r;
}

}  // namespace

// The half-precision forms. The model: the single-precision unit on the exact float, the result cut toward zero to a half.
VTEST(the_model_hashes_as_the_card_over_every_half_pattern) {
  const auto want = f16_hashes();
  VCHECK_EQ(want.size(), size_t{12});
  for (const char* name : kHalfOps) {
    const unsigned op = half_op(name);
    VCHECK(mufu::has_f16(op));
    const uint64_t got = fnv(model_halves(op, false));
    if (got != want.at(std::string(name) + ".F16")) throw vtest::Failure(std::string("MUFU.") + name + ".F16: model " + hex(got, 16) + ", an RTX 3060 gives " + hex(want.at(std::string(name) + ".F16"), 16));
  }
  VCHECK(!mufu::has_f16(mufu::OpRcp64h) && !mufu::has_f16(mufu::OpRsq64h));
  for (const char* name : {"ex2", "tanh"}) {
    const unsigned op = name[0] == 'e' ? mufu::OpEx2 : mufu::OpTanh;
    VCHECK_EQ(fnv(model_halves(op, true)), want.at(std::string(name) + ".approx.f16x2"));
  }
}

// MUFU.<op>.F16 on the SASS executor, every half pattern of every operation: the registers 0 .. 65535 through the 32-lane path and
// the general path, then the high half (.H1), |x| and -x on a sample of patterns (the modifiers act on the half's sign bit).
static void sass_half_forms() {
  const auto want = f16_hashes();
  for (const char* name : kHalfOps) {
    const unsigned op = half_op(name);
    Sass s(op, false, false, true, 0);
    const std::vector<uint32_t> r = s.dump(0, 65536);
    if (fnv(r) != want.at(std::string(name) + ".F16")) {
      for (uint32_t i = 0; i < 65536; ++i)
        if (r[i] != mufu::eval_f16(op, i)) throw vtest::Failure(std::string("SASS MUFU.") + name + ".F16(" + hex(i, 4) + "): " + hex(r[i]) + ", an RTX 3060 gives " + hex(mufu::eval_f16(op, i)));
      throw vtest::Failure(std::string("SASS MUFU.") + name + ".F16: the hash differs from the card's");
    }
  }
  // The high half, and the modifiers: 1/251 of the patterns, the upper half of the operand register being the pattern. A
  // consecutive run of registers shares its upper half, so a launch of one thread a pattern is used.
  for (const char* name : {"EX2", "TANH", "RCP", "SIN"}) {
    const unsigned op = half_op(name);
    for (int mods = 0; mods < 4; ++mods) {
      const bool ab = mods & 1, ng = mods & 2;
      for (unsigned part : {0u, 1u}) {
        Sass s(op, ab, ng, true, part);
        for (uint32_t h = 3; h < 65536; h += 2503) {
          const uint32_t reg = part ? (h << 16) | 0x3c00u : (0xbc00u << 16) | h;   // the half read, and the other half something else
          uint32_t x = h;
          if (ab) x &= 0x7fffu;
          if (ng) x ^= 0x8000u;
          const uint32_t got = s.dump(reg, 1)[0], w = mufu::eval_f16(op, x);
          if (got != w) throw vtest::Failure(std::string("SASS MUFU.") + name + ".F16" + (part ? ".H1" : "") + (ng ? " -" : " ") + (ab ? "|x|" : "x") + " of " + hex(reg) + ": " + hex(got) + ", want " + hex(w));
        }
      }
    }
  }
}
VTEST(the_sass_executor_gives_the_cards_half_precision_results) { sass_half_forms(); }
VTEST(the_sass_executor_general_path_gives_the_cards_half_precision_results) {
  Env e("VGPU_FASTPATH", "0");
  sass_half_forms();
}

// ex2.approx.f16 / .f16x2 and tanh.approx.f16 / .f16x2 on the PTX interpreter: the card's hashes.
VTEST(the_ptx_interpreter_gives_the_cards_half_precision_results) {
  const auto want = f16_hashes();
  for (const char* base : {"ex2.approx", "tanh.approx"})
    for (const char* ty : {".f16", ".f16x2"}) {
      const bool packed = ty[4] == 'x';
      const std::string form = std::string(base) + ty;
      const std::vector<uint32_t> r = run_ptx_half(form, packed, packed ? half_pairs() : all_halves());
      if (fnv(r) != want.at(form)) {
        const unsigned op = base[0] == 'e' ? mufu::OpEx2 : mufu::OpTanh;
        const std::vector<uint32_t> m = model_halves(op, packed);
        for (uint32_t i = 0; i < 65536; ++i)
          if (r[i] != m[i]) throw vtest::Failure("PTX " + form + "(" + hex(packed ? half_pairs()[i] : i) + "): " + hex(r[i]) + ", an RTX 3060 gives " + hex(m[i]));
        throw vtest::Failure("PTX " + form + ": the hash differs from the card's");
      }
    }
}

// VGPU_MUFU_EXACT=0: the half forms are what they were: the PTX interpreter's the function in double rounded once to the nearest
// half, the SASS executor's the register read as a float. The numbers are how many of the 65,536 patterns of a form differ from the card.
VTEST(the_switch_restores_the_host_precision_half_results) {
  const auto want = f16_hashes();
  Env e("VGPU_MUFU_EXACT", "0");
  for (const char* base : {"ex2.approx", "tanh.approx"}) {
    const std::string form = std::string(base) + ".f16";
    const std::vector<uint32_t> r = run_ptx_half(form, false, all_halves()), card = model_halves(base[0] == 'e' ? mufu::OpEx2 : mufu::OpTanh, false);
    size_t differ = 0;
    for (uint32_t i = 0; i < 65536; ++i) differ += r[i] != card[i];
    std::printf("EXACT=0 PTX %s: %zu of 65536 differ from the card\n", form.c_str(), differ);
    VCHECK(fnv(r) != want.at(form));
    VCHECK_EQ(differ, base[0] == 'e' ? size_t{19417} : size_t{14736});
  }
  for (const char* name : kHalfOps) {
    Sass s(half_op(name), false, false, true, 0);
    const std::vector<uint32_t> r = s.dump(0, 65536), card = model_halves(half_op(name), false);
    size_t differ = 0;
    for (uint32_t i = 0; i < 65536; ++i) differ += r[i] != card[i];
    std::printf("EXACT=0 SASS MUFU.%s.F16: %zu of 65536 differ from the card\n", name, differ);
    VCHECK(fnv(r) != want.at(std::string(name) + ".F16"));
  }
}

// ---- the bfloat16 forms and the sm_90 encoding (measured on an H100 PCIe, 2026-11-16) -------------------------------------

// The model against the H100's hashes: all 65,536 bfloat16 patterns of each of the eight sub-operations, and the packed PTX forms.
VTEST(the_model_hashes_as_the_h100_over_every_bfloat16_pattern) {
  const auto want = bf16_hashes();
  VCHECK_EQ(want.size(), size_t{12});
  for (const char* name : kHalfOps) {
    const uint64_t got = fnv(model_bf16(half_op(name), false));
    if (got != want.at(std::string(name) + ".BF16 all"))
      throw vtest::Failure(std::string("MUFU.") + name + ".BF16: model " + hex(got, 16) + ", an H100 gives " + hex(want.at(std::string(name) + ".BF16 all"), 16));
  }
  VCHECK_EQ(fnv(model_bf16(mufu::OpEx2, false)), want.at("ex2.approx.bf16"));
  VCHECK_EQ(fnv(model_bf16(mufu::OpTanh, false)), want.at("tanh.approx.bf16"));
  VCHECK_EQ(fnv(model_bf16(mufu::OpEx2, true)), want.at("ex2.approx.bf16x2"));
  VCHECK_EQ(fnv(model_bf16(mufu::OpTanh, true)), want.at("tanh.approx.bf16x2"));
  // The F16 forms ptxas emits for sm_90 give the RTX 3060's hashes.
  const auto h = h100_f16_ptx_hashes(), r = f16_hashes();
  for (const char* f : {"ex2.approx.f16", "tanh.approx.f16", "ex2.approx.f16x2", "tanh.approx.f16x2"}) VCHECK_EQ(h.at(f), r.at(f));
}

// A few values of the model, by hand: the bfloat16 widened by a shift, the float result's upper 16 bits.
VTEST(eval_bf16_is_the_unit_on_the_widened_value_cut_to_the_upper_half) {
  for (const char* name : kHalfOps) {
    const unsigned op = half_op(name);
    for (uint32_t x : {0x0000u, 0x8000u, 0x0001u, 0x007fu, 0x0080u, 0x3f80u, 0xbf80u, 0x4000u, 0x4049u, 0x7f7fu, 0x7f80u, 0xff80u, 0x7fc0u, 0xffc1u, 0x1234u, 0xabcdu}) {
      VCHECK_EQ(mufu::eval_bf16(op, x), mufu::eval(op, x << 16) >> 16);
      VCHECK_EQ(mufu::eval_bf16(op, x | 0xdead0000u), mufu::eval_bf16(op, x));   // only the low 16 bits are the operand
      VCHECK(mufu::eval_bf16(op, x) <= 0xffffu);
    }
  }
  // The values the card gave (an H100, and the model they equal): the exponential and the hyperbolic tangent of one, the square root
  // of four and the reciprocal of two.
  VCHECK_EQ(mufu::eval_bf16(mufu::OpEx2, 0x3f80u), SPOT_EX2_1);
  VCHECK_EQ(mufu::eval_bf16(mufu::OpTanh, 0x3f80u), SPOT_TANH_1);
  VCHECK_EQ(mufu::eval_bf16(mufu::OpSqrt, 0x4080u), SPOT_SQRT_4);
  VCHECK_EQ(mufu::eval_bf16(mufu::OpRcp, 0x4000u), SPOT_RCP_2);
  VCHECK_EQ(mufu::eval_bf16(mufu::OpEx2, 0x7fc0u), SPOT_EX2_NAN);
  VCHECK_EQ(mufu::eval_bf16(mufu::OpLg2, 0x0000u), SPOT_LG2_0);
}

// The decoder: bit 73 is .F16 up to sm_89, and from sm_90 bit 72 is .F16 and bit 73 is .BF16.
sass::Instr mufu_word(int sm, bool b72, bool b73, unsigned op, unsigned part) {
  sass::Word w;
  w.lo = 0x0000000200097308ull | (static_cast<uint64_t>(part) << 60);   // MUFU R9, R2 (bits 60-61: the half)
  w.hi = (static_cast<uint64_t>(op) << 10) | (static_cast<uint64_t>(b72) << 8) | (static_cast<uint64_t>(b73) << 9);
  return sass::decode(w, 0, sm);
}
bool has_mod(const sass::Instr& i, const char* m) { return std::find(i.mods.begin(), i.mods.end(), std::string(m)) != i.mods.end(); }

VTEST(the_decoder_reads_the_f16_and_bf16_bits_by_architecture) {
  for (int sm : {75, 80, 86, 89}) {   // bit 73 is .F16, as before; bit 72 is nothing
    const sass::Instr f = mufu_word(sm, false, true, 2, 0);
    VCHECK_EQ(f.f[0], 2u);
    VCHECK(f.f[1] == 1 && f.f[2] == 0);
    VCHECK(has_mod(f, "EX2") && has_mod(f, "F16") && !has_mod(f, "BF16"));
    const sass::Instr n = mufu_word(sm, true, false, 9, 0);   // bit 72 alone is a plain 32-bit MUFU
    VCHECK(n.f[1] == 0 && n.f[2] == 0 && !has_mod(n, "F16") && !has_mod(n, "BF16"));
    const sass::Instr h = mufu_word(sm, false, true, 9, 1);
    VCHECK(h.f[1] == 1 && h.f[5] == 1 && h.src[0].suffix == ".H1");
  }
  for (int sm : {90, 100, 103, 120}) {   // bit 72 is .F16, bit 73 is .BF16
    const sass::Instr f = mufu_word(sm, true, false, 2, 0);   // EX2.F16
    VCHECK(f.f[1] == 1 && f.f[2] == 0 && has_mod(f, "F16") && !has_mod(f, "BF16"));
    const sass::Instr t = mufu_word(sm, false, true, 9, 0);   // TANH.BF16
    VCHECK_EQ(t.f[0], 9u);
    VCHECK(t.f[1] == 0 && t.f[2] == 1 && has_mod(t, "TANH") && has_mod(t, "BF16") && !has_mod(t, "F16"));
    const sass::Instr h = mufu_word(sm, false, true, 2, 1);   // EX2.BF16 R9, R2.H1
    VCHECK(h.f[2] == 1 && h.f[5] == 1 && h.src[0].suffix == ".H1");
    const sass::Instr hf = mufu_word(sm, true, false, 2, 1);
    VCHECK(hf.f[1] == 1 && hf.f[5] == 1 && hf.src[0].suffix == ".H1");
    const sass::Instr n = mufu_word(sm, false, false, 2, 0);
    VCHECK(n.f[1] == 0 && n.f[2] == 0 && n.src[0].suffix.empty());
  }
  const std::string text = sass::to_text(mufu_word(90, false, true, 2, 0));
  VCHECK(text.find("MUFU.EX2.BF16") != std::string::npos);
}

// MUFU.<op>.BF16 and .F16 as an sm_90 cubin has them (the probe's kernel with the bits of that architecture, run on the h100-pcie
// profile): every bfloat16 pattern against the H100's hashes, the f16 form against the RTX 3060's, the high half, |x| and -x.
static void sass_sm90_forms() {
  const auto wbf = bf16_hashes(), wf = f16_hashes();
  for (const char* name : kHalfOps) {
    const unsigned op = half_op(name);
    Sass sb(op, false, false, false, 0, 90, true);
    const std::vector<uint32_t> rb = sb.dump(0, 65536);
    if (fnv(rb) != wbf.at(std::string(name) + ".BF16 all")) {
      for (uint32_t i = 0; i < 65536; ++i)
        if (rb[i] != mufu::eval_bf16(op, i)) throw vtest::Failure(std::string("SASS sm_90 MUFU.") + name + ".BF16(" + hex(i, 4) + "): " + hex(rb[i]) + ", an H100 gives " + hex(mufu::eval_bf16(op, i)));
      throw vtest::Failure(std::string("SASS sm_90 MUFU.") + name + ".BF16: the hash differs from the card's");
    }
    Sass sf(op, false, false, true, 0, 90, false);
    if (fnv(sf.dump(0, 65536)) != wf.at(std::string(name) + ".F16")) throw vtest::Failure(std::string("SASS sm_90 MUFU.") + name + ".F16 (bit 72): the hash differs");
  }
  for (const char* name : {"EX2", "TANH", "RCP", "SIN"}) {
    const unsigned op = half_op(name);
    for (int mods = 0; mods < 4; ++mods) {
      const bool ab = mods & 1, ng = mods & 2;
      for (unsigned part : {0u, 1u})
        for (bool bf : {false, true}) {
          Sass s(op, ab, ng, !bf, part, 90, bf);
          for (uint32_t h = 3; h < 65536; h += 2503) {
            const uint32_t reg = part ? (h << 16) | 0x3c00u : (0xbc00u << 16) | h;
            uint32_t x = h;
            if (ab) x &= 0x7fffu;
            if (ng) x ^= 0x8000u;
            const uint32_t got = s.dump(reg, 1)[0], w = bf ? mufu::eval_bf16(op, x) : mufu::eval_f16(op, x);
            if (got != w) throw vtest::Failure(std::string("SASS sm_90 MUFU.") + name + (bf ? ".BF16" : ".F16") + (part ? ".H1" : "") + " of " + hex(reg) + ": " + hex(got) + ", want " + hex(w));
          }
        }
    }
  }
}
VTEST(the_sass_executor_gives_the_h100s_bf16_results_on_sm_90) { sass_sm90_forms(); }
VTEST(the_sass_executor_general_path_gives_the_h100s_bf16_results_on_sm_90) {
  Env e("VGPU_FASTPATH", "0");
  sass_sm90_forms();
}

// ex2.approx.ftz.bf16 / .bf16x2 and tanh.approx.bf16 / .bf16x2 on the PTX interpreter (sm_90): the H100's hashes.
VTEST(the_ptx_interpreter_gives_the_h100s_bf16_results) {
  const auto want = bf16_hashes();
  for (const char* base : {"ex2.approx.ftz", "tanh.approx"})
    for (const char* ty : {".bf16", ".bf16x2"}) {
      const bool packed = ty[5] == 'x';
      const std::string form = std::string(base) + ty;
      const std::string key = std::string(base[0] == 'e' ? "ex2.approx" : "tanh.approx") + ty;
      const std::vector<uint32_t> r = run_ptx_half(form, packed, packed ? half_pairs() : all_halves(), "sm_90", "nvidia/h100-pcie");
      if (fnv(r) != want.at(key)) {
        const std::vector<uint32_t> m = model_bf16(base[0] == 'e' ? mufu::OpEx2 : mufu::OpTanh, packed);
        for (uint32_t i = 0; i < 65536; ++i)
          if (r[i] != m[i]) throw vtest::Failure("PTX " + form + "(" + hex(packed ? half_pairs()[i] : i) + "): " + hex(r[i]) + ", an H100 gives " + hex(m[i]));
        throw vtest::Failure("PTX " + form + ": the hash differs from the card's");
      }
    }
}

// VGPU_MUFU_EXACT=0: the PTX bf16 forms are the function at host precision, rounded to nearest, as before (they differ from the card).
VTEST(the_switch_restores_the_host_precision_bf16_results) {
  const auto want = bf16_hashes();
  Env e("VGPU_MUFU_EXACT", "0");
  const std::vector<uint32_t> r = run_ptx_half("ex2.approx.ftz.bf16", false, all_halves(), "sm_90", "nvidia/h100-pcie");
  VCHECK(fnv(r) != want.at("ex2.approx.bf16"));
  const std::vector<uint32_t> card = model_bf16(mufu::OpEx2, false);
  size_t differ = 0;
  for (uint32_t i = 0; i < 65536; ++i) differ += r[i] != card[i];
  std::printf("EXACT=0 PTX ex2.approx.ftz.bf16: %zu of 65536 differ from the card\n", differ);
  VCHECK(differ > 0);
}

VTEST(the_model_gives_the_cards_result_on_every_measured_vector) {
  size_t rows = 0;
  for (int op = 0; op < 10; ++op) {
    VCHECK(mufu::has(static_cast<unsigned>(op)));
    for (const auto& row : vectors().at(kOps[op])) {
      const uint32_t got = mufu::eval(static_cast<unsigned>(op), static_cast<uint32_t>(row[0]));
      if (got != row[1]) throw vtest::Failure(std::string("MUFU.") + kOps[op] + "(" + hex(row[0]) + "): model " + hex(got) + ", an RTX 3060 gives " + hex(row[1]));
      ++rows;
    }
  }
  VCHECK_EQ(rows, size_t{3100});
}

// sq19 is the folded partial-product array of x * x without the products whose weight is below 2^19, shifted right by 19.
VTEST(sq19_is_the_square_without_its_low_partial_products) {
  for (uint32_t x = 0; x < (1u << 17); ++x) {
    uint64_t sum = 0;
    for (unsigned i = 0; i < 17; ++i) {
      if (!((x >> i) & 1)) continue;
      if (2 * i >= 19) sum += 1ull << (2 * i);
      for (unsigned j = i + 1; j < 17; ++j)
        if (((x >> j) & 1) && i + j + 1 >= 19) sum += 1ull << (i + j + 1);
    }
    if (mufu::sq19(x) != (sum >> 19)) throw vtest::Failure("sq19(" + hex(x) + ") = " + hex(mufu::sq19(x)) + ", the array gives " + hex(sum >> 19));
  }
  // It is not the square cut to its upper bits: the dropped products carry into them.
  VCHECK(mufu::sq19(0x1ffff) != static_cast<uint32_t>((0x1ffffull * 0x1ffffull) >> 19));
}

// The card's hashes of 2^26 consecutive operands. By default the chunk from 0x3c000000 (0.0078 to 2, or the doubles from
// 2^-63 to 2), of every operation; VGPU_EXHAUSTIVE=1: all 64 chunks of every operation, which is every input there is.
VTEST(the_model_hashes_as_the_card_over_whole_chunks) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/tools/numerics/mufu/rtx3060.hashes.txt");
  VCHECK(static_cast<bool>(in));
  struct Want { int op; bool all; uint64_t start, hash; };
  std::vector<Want> want;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string name, start, hash;
    ss >> name >> start >> hash;
    want.push_back({op_of(name), start == "all", start == "all" ? 0 : std::stoull(start, nullptr, 16), std::stoull(hash, nullptr, 16)});
  }
  VCHECK_EQ(want.size(), size_t{650});   // ten operations: 64 chunks and the whole
  const bool all = exhaustive();
  std::vector<std::string> bad(10);
  const auto one = [&](int op) {
    uint64_t total = 0xcbf29ce484222325ull;
    for (const Want& w : want) {
      if (w.op != op) continue;
      if (w.all) {
        if (all && total != w.hash) bad[op] = std::string("MUFU.") + kOps[op] + " over all 2^32 inputs: model " + hex(total, 16) + ", card " + hex(w.hash, 16);
        continue;
      }
      if (!all && w.start != 0x3c000000u) continue;
      const uint64_t h = chunk_hash(static_cast<unsigned>(op), w.start, &total);
      if (h != w.hash && bad[op].empty())
        bad[op] = std::string("MUFU.") + kOps[op] + " chunk " + hex(w.start) + ": model " + hex(h, 16) + ", card " + hex(w.hash, 16);
    }
  };
  if (all) {   // (x at a time: a minute or two an operation)
    for (int first = 0; first < 10; first += 3) {
      std::vector<std::thread> t;
      for (int op = first; op < std::min(10, first + 3); ++op) t.emplace_back(one, op);
      for (std::thread& th : t) th.join();
    }
  } else {
    for (int op = 0; op < 10; ++op) one(op);
  }
  for (const std::string& b : bad)
    if (!b.empty()) throw vtest::Failure(b);
}

VTEST(the_sass_executor_gives_the_cards_result) { sass_on_the_vectors(); }

VTEST(the_sass_executor_general_path_gives_the_cards_result) {
  Env e("VGPU_FASTPATH", "0");
  sass_on_the_vectors();
}

// MUFU.<op> Rd, |Rs| and -Rs: the modifier is applied to the operand's sign bit before the unit (the 64H forms too), as the
// card probe measured with --abs and --neg.
VTEST(the_sass_operand_modifiers_are_applied_before_the_unit) {
  for (const char* fast : {"1", "0"}) {
    Env e("VGPU_FASTPATH", fast);
    for (int op : {4, 1, 6, 7, 9, 2}) {
      for (int mods = 1; mods < 4; ++mods) {
        const bool ab = mods & 1, ng = mods & 2;
        Sass s(static_cast<unsigned>(op), ab, ng);
        for (uint32_t base : {0x3f7fffc0u, 0xbf7fffc0u, 0x3ff7ffc0u, 0xbfe7ffc0u}) {
          const std::vector<uint32_t> r = s.dump(base, 128);
          for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t x = base + i, want = mufu::eval(static_cast<unsigned>(op), (ab ? x & 0x7fffffffu : x) ^ (ng ? 0x80000000u : 0u));
            if (r[i] != want)
              throw vtest::Failure(std::string("SASS MUFU.") + kOps[op] + (ng ? " -" : " ") + (ab ? "|x|" : "x") + " of " + hex(x) + ": " + hex(r[i]) + ", want " + hex(want));
          }
        }
      }
    }
  }
}

// The probe's hash kernel (a loop of MUFU and 64-bit arithmetic) on the SASS executor: the hashes the model gives.
VTEST(the_sass_executor_runs_the_probes_hash_kernel) {
  for (int op : {4, 5, 2, 0}) {
    Sass s(static_cast<unsigned>(op));
    const uint32_t base = 0x3f7f0000u, log2_block = 5, n = 512;
    const std::vector<uint64_t> r = s.hash(base, log2_block, n);
    for (uint32_t t = 0; t < n; ++t) {
      uint64_t h = 0xcbf29ce484222325ull;
      for (uint32_t j = 0; j < (1u << log2_block); ++j) h = (h ^ mufu::eval(static_cast<unsigned>(op), base + (t << log2_block) + j)) * 0x100000001b3ull;
      if (r[t] != h) throw vtest::Failure(std::string("mufu_hash, MUFU.") + kOps[op] + ", thread " + std::to_string(t) + ": " + hex(r[t], 16) + ", want " + hex(h, 16));
    }
  }
}

VTEST(the_ptx_interpreter_gives_the_cards_result_for_the_lone_forms) { ptx_on_the_vectors(); }

// Every PTX approximate form on the operands where ptxas puts more than the instruction around it, and the divisions.
VTEST(the_ptx_forms_are_what_ptxas_makes_of_them_on_the_card) {
  const auto forms = sections("nvidia/tests/data/mufu/rtx3060.forms.txt");
  VCHECK_EQ(forms.size(), size_t{22});
  size_t rows = 0;
  for (const auto& [form, lines] : forms) {
    const bool div = form.rfind("div.", 0) == 0, wide = form.find("f64") != std::string::npos;
    std::vector<uint64_t> a, b;
    for (const auto& row : lines) {
      a.push_back(row[0]);
      if (div) b.push_back(row[1]);
    }
    const std::vector<uint64_t> r = run_ptx(form, wide ? 64 : 32, a, div ? &b : nullptr);
    for (size_t i = 0; i < lines.size(); ++i) {
      const uint64_t want = lines[i].back();
      if (r[i] != want)
        throw vtest::Failure("PTX " + form + "(" + hex(a[i], wide ? 16 : 8) + (div ? ", " + hex(b[i]) : "") + "): " + hex(r[i], wide ? 16 : 8) +
                             ", an RTX 3060 gives " + hex(want, wide ? 16 : 8));
      ++rows;
    }
  }
  VCHECK_EQ(rows, size_t{52 * 15 + 32 * 3 + 144 * 4});
}

// VGPU_MUFU_EXACT=0: the function at the host's precision, rounded once, in both engines; unset, the card's unit again.
VTEST(the_switch_restores_the_host_precision_results) {
  const uint32_t x = 0x3f8525e1u;   // the card's unit gives 0x3f761a27 (a line of the vectors), the correctly rounded reciprocal is 0x3f761a26
  const std::vector<uint64_t> divisor{x};
  VCHECK_EQ(mufu::eval(mufu::OpRcp, x), 0x3f761a27u);
  VCHECK_EQ(Sass(mufu::OpRcp).dump(x, 1)[0], 0x3f761a27u);
  VCHECK_EQ(run_ptx("rcp.approx.ftz.f32", 32, {x})[0], uint64_t{0x3f761a27u});
  VCHECK_EQ(run_ptx("div.full.f32", 32, {0x3f800000u}, &divisor)[0], uint64_t{0x3f761a27u});
  {
    Env e("VGPU_MUFU_EXACT", "0");
    VCHECK_EQ(Sass(mufu::OpRcp).dump(x, 1)[0], 0x3f761a26u);
    VCHECK_EQ(run_ptx("rcp.approx.ftz.f32", 32, {x})[0], uint64_t{0x3f761a26u});
    VCHECK_EQ(run_ptx("div.full.f32", 32, {0x3f800000u}, &divisor)[0], uint64_t{0x3f761a26u});
    // Every operation, on the vectors' operands: the function in double, rounded once (the 64H forms: the upper word of it).
    for (int op : {0, 1, 2, 3, 4, 5, 8, 9}) {
      Sass s(static_cast<unsigned>(op));
      for (uint32_t in : {0x3f9d70a4u, 0x3e99999au, 0x40490fdbu}) {
        float f;
        std::memcpy(&f, &in, 4);
        const double d = f;
        const double v = op == 0 ? std::cos(d * 6.28318530717958647692) : op == 1 ? std::sin(d * 6.28318530717958647692) : op == 2 ? std::exp2(d)
                         : op == 3 ? std::log2(d) : op == 4 ? 1.0 / d : op == 5 ? 1.0 / std::sqrt(d) : op == 8 ? std::sqrt(d) : std::tanh(d);
        const float want = static_cast<float>(v);
        uint32_t wb;
        std::memcpy(&wb, &want, 4);
        VCHECK_EQ(s.dump(in, 1)[0], wb);
      }
    }
    VCHECK_EQ(Sass(mufu::OpRcp64h).dump(0x40080000u, 1)[0], 0x3fd55555u);
  }
  VCHECK_EQ(Sass(mufu::OpRcp).dump(x, 1)[0], 0x3f761a27u);   // the switch is read at every launch
  VCHECK_EQ(run_ptx("rcp.approx.ftz.f32", 32, {x})[0], uint64_t{0x3f761a27u});
}

VTEST_MAIN
