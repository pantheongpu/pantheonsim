// The SASS executor's tile route for the MMA instructions its SIMD HMMA kernel does not take (Runner::exec_mma: sparse and 8-bit float sums, the
// integer products, double precision and single bits, through simd::Kernels::tile_f32 / tile_i32 / tile_f64 and exec/tile_mma.hpp) against its
// general form, bit for bit: the kernels are switched off (VGPU_SIMD=off) in a second run of this program, and every instruction form NVIDIA's
// disassembler corpora hold for HMMA, IMMA, BMMA, DMMA and QMMA (nvidia/tests/data/sass, sm_80, sm_86 and sm_89) is run in both on the same random
// operands -- zeros of both signs, subnormals, infinities, NaNs, the largest values, integers at their limits, accumulators that saturate --
// with the warp's registers filled by loads in a cubin of hand-built words (a corpus word for the MMA itself, its registers as the corpus has them).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/wgmma.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

constexpr unsigned kBlocks = 24, kLaneWords = 64, kOutWords = 16;   // blocks of one warp; 32-bit words a lane takes in, and gives out
constexpr unsigned kScratch = 240;                                   // the registers from here up are the program's own

using sass::Word;

Word with(Word w, unsigned pos, unsigned len, uint64_t v) {
  for (unsigned i = 0; i < len; ++i) {
    const unsigned p = pos + i;
    uint64_t& half = p >= 64 ? w.hi : w.lo;
    const uint64_t bit = uint64_t{1} << (p % 64);
    half = (v >> i & 1) ? (half | bit) : (half & ~bit);
  }
  return w;
}

struct Line {
  Word w;
  std::string text;
};

std::vector<Line> read_corpus(const std::string& name) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/sass/sm_" + name + ".txt");
  std::vector<Line> out;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
    if (t1 == std::string::npos || t2 == std::string::npos) continue;
    Line l;
    l.w.hi = std::stoull(line.substr(0, 16), nullptr, 16);   // (the corpora print the high half first)
    l.w.lo = std::stoull(line.substr(16, 16), nullptr, 16);
    l.text = line.substr(t2 + 1);
    out.push_back(std::move(l));
  }
  return out;
}

const Line& find_line(const std::vector<Line>& c, const char* pattern) {
  const std::regex re(pattern);
  for (const Line& l : c)
    if (std::regex_match(l.text, re)) return l;
  throw vtest::Failure(std::string("the corpus has no instruction like ") + pattern);
}
Word find_word(const std::vector<Line>& c, const char* pattern) { return find_line(c, pattern).w; }
// The "desc[URn]" a Hopper memory operand carries ("" before sm_90), as the template spells it.
std::string desc_of(const Line& l) {
  std::smatch m;
  return std::regex_search(l.text, m, std::regex("desc\\[UR[0-9]+\\]")) ? m.str() : std::string();
}

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm `sm` with one kernel "k" taking a pointer, 256 registers (the module-wide REGCOUNT record needs a symbol table).
std::string make_cubin(int sm, const std::vector<Word>& code, uint64_t shared_bytes = 0) {
  const std::string strtab("\0k\0", 3);
  std::string shstr(1, '\0');
  for (const char* n : {".shstrtab", ".strtab", ".symtab", ".nv.info", ".text.k", ".nv.info.k", ".nv.shared.k"}) {
    shstr += n;
    shstr.push_back('\0');
  }
  const auto name_of = [&](const char* n) { return static_cast<uint32_t>(shstr.find(n, 1)); };
  std::string info_mod;   // REGCOUNT of symbol 1
  put<uint8_t>(info_mod, 4), put<uint8_t>(info_mod, 0x2f), put<uint16_t>(info_mod, 8), put<uint32_t>(info_mod, 1), put<uint32_t>(info_mod, 250);
  std::string info_k;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info_k, 4), put<uint8_t>(info_k, 0x17), put<uint16_t>(info_k, 12);
  put<uint32_t>(info_k, 0), put<uint32_t>(info_k, 0), put<uint32_t>(info_k, 8u << 18);
  std::string symtab(24, '\0');   // the null symbol, then k: global function in section 5
  put<uint32_t>(symtab, 1), put<uint8_t>(symtab, 0x12), put<uint8_t>(symtab, 0), put<uint16_t>(symtab, 5);
  put<uint64_t>(symtab, 0), put<uint64_t>(symtab, 16 * code.size());
  std::string out(64, '\0');
  const auto place = [&](const std::string& bytes) {
    while (out.size() % 16) out.push_back('\0');
    const uint64_t at = out.size();
    out += bytes;
    return at;
  };
  const uint64_t shstr_off = place(shstr), str_off = place(strtab), sym_off = place(symtab), imod_off = place(info_mod);
  std::string text;
  for (const Word& w : code) put(text, w.lo), put(text, w.hi);
  const uint64_t text_off = place(text), ik_off = place(info_k);
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size, uint32_t link, uint64_t entsize) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, link), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, entsize);
  };
  shdr(0, 0, 0, 0, 0, 0);                                                                     // 0
  shdr(name_of(".shstrtab"), 3, shstr_off, shstr.size(), 0, 0);                              // 1
  shdr(name_of(".strtab"), 3, str_off, strtab.size(), 0, 0);                                 // 2
  shdr(name_of(".symtab"), 2, sym_off, symtab.size(), 2, 24);                                // 3
  shdr(name_of(".nv.info"), 0x70000000, imod_off, info_mod.size(), 0, 0);                    // 4
  shdr(name_of(".text.k"), 1, text_off, text.size(), 0, 0);                                  // 5
  shdr(name_of(".nv.info.k"), 0x70000000, ik_off, info_k.size(), 0, 0);                      // 6
  if (shared_bytes) shdr(name_of(".nv.shared.k"), 8, 0, shared_bytes, 0, 0);                  // 7: SHT_NOBITS
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = shared_bytes ? 8 : 7, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (static_cast<uint32_t>(sm) << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

uint32_t f32_class(std::mt19937& g, bool wide) {
  const uint32_t h = g();
  switch (h % 16) {
    case 0: return 0;
    case 1: return 0x80000000u;
    case 2: return h & 0x807fffffu;
    case 3: return 0x7f800000u | (h & 0x80000000u);
    case 4: return 0x7fc00000u | (h & 0x803fffffu);
    case 5: return 0x7f800001u | (h & 0x80000000u);
    case 6: return (h & 0x80000000u) | 0x7f7fffffu;
    default: return (h & 0x80000000u) | ((wide ? 1 + (h >> 8) % 254 : 118 + (h >> 8) % 20) << 23) | (g() & 0x7fffffu);
  }
}
uint16_t half_class(std::mt19937& g, bool wide) {
  const uint32_t h = g();
  switch (h % 14) {
    case 0: return 0;
    case 1: return 0x8000;
    case 2: return static_cast<uint16_t>((h & 0x8000u) | ((h >> 8) & 0x3ffu));
    case 3: return static_cast<uint16_t>(0x7c00u | (h & 0x8000u));
    case 4: return static_cast<uint16_t>(0x7e00u | (h & 0x83ffu));
    case 5: return static_cast<uint16_t>((h & 0x8000u) | 0x7bffu);
    default: return static_cast<uint16_t>((h & 0x8000u) | ((wide ? 1 + (h >> 8) % 30 : 10 + (h >> 8) % 10) << 10) | (g() & 0x3ffu));
  }
}
uint64_t f64_class(std::mt19937& g, bool wide) {
  const uint64_t h = (uint64_t{g()} << 32) | g();
  switch (h % 12) {
    case 0: return 0;
    case 1: return 0x8000000000000000ull;
    case 2: return h & 0x800fffffffffffffull;
    case 3: return 0x7ff0000000000000ull | (h & 0x8000000000000000ull);
    case 4: return 0x7ff8000000000000ull | (h >> 12);
    case 5: return (h & 0x8000000000000000ull) | 0x7fefffffffffffffull;
    default: return (h & 0x8000000000000000ull) | ((wide ? 1 + (h >> 20) % 2046 : 1013 + (h >> 20) % 20) << 52) | (h & 0xfffffffffffffull);
  }
}

// What a form needs: registers (in 32-bit words) of A, B, C, D and the metadata, and how its operands are drawn.
struct Shape {
  bool ok = false;
  unsigned a = 0, b = 0, c = 0, ra = 0, rb = 0, rc = 0, rd = 0, rm = 0;
  bool sparse = false;
  enum { Half, Bf16, Tf32, Fp8, Int, F64, Bits } kind = Int;
  bool c_f16 = false;
};

Shape shape_of(const sass::Instr& ins) {
  Shape s;
  if (ins.dst.empty() || ins.src.size() < 3) return s;
  s.rd = ins.dst[0].reg, s.ra = ins.src[0].reg, s.rb = ins.src[1].reg, s.rc = ins.src[2].reg;
  unsigned M = 16, K = 8, p = 1;
  switch (ins.op) {
    case sass::Op::HMMA: {
      static const unsigned ks[] = {8, 16, 4, 0}, sp_ks[] = {8, 16, 0, 32};
      const unsigned type = ins.f[2];
      s.sparse = ins.f[3];
      K = s.sparse ? sp_ks[ins.f[0]] : ks[ins.f[0]];
      if (type == 2 && ins.f[0] == 0) K = 8;
      if (!K || type > 2) return s;
      p = type == 2 ? 1 : 2;
      s.kind = type == 0 ? Shape::Half : type == 1 ? Shape::Bf16 : Shape::Tf32;
      s.c_f16 = !ins.f[1];
      s.c = s.c_f16 ? 2 : 4;
      break;
    }
    case sass::Op::IMMA: {
      static const unsigned ks[] = {16, 0, 32, 0, 16, 32, 64, 128};
      M = ins.f[0] < 4 ? 8 : 16;
      K = ks[ins.f[0]];
      if (!K || ins.f[4] != ins.f[5]) return s;
      p = ins.f[4] ? 8 : 4;
      s.sparse = ins.f[3];
      s.kind = Shape::Int;
      s.c = M * 8 / 32;
      break;
    }
    case sass::Op::BMMA: {
      static const unsigned ms[] = {8, 16, 16, 0}, ks[] = {128, 128, 256, 0};
      M = ms[ins.f[0]];
      K = ks[ins.f[0]];
      if (!M) return s;
      p = 32;
      s.kind = Shape::Bits;
      s.c = M * 8 / 32;
      break;
    }
    case sass::Op::DMMA: {
      s.kind = Shape::F64;
      s.a = 2, s.b = 2, s.c = 4;
      s.ok = true;
      return s;
    }
    case sass::Op::QMMA: {
      if (ins.sm >= 120 || ins.f[1] > 1 || ins.f[2] > 1 || (!ins.mods.empty() && ins.mods[0] == "SF")) return s;
      K = 32;
      p = 4;
      s.kind = Shape::Fp8;
      s.c_f16 = !ins.f[0];
      s.c = s.c_f16 ? 2 : 4;
      break;
    }
    default: return s;
  }
  s.a = M * K / (32 * p) / (s.sparse ? 2 : 1);
  s.b = K * 8 / (32 * p);
  if (s.sparse) {
    if (ins.src.size() < 4) return s;
    s.rm = ins.src[3].reg;
  }
  s.ok = true;
  return s;
}

uint64_t fnv(const std::vector<uint32_t>& v) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t w : v) h = (h ^ w) * 1099511628211ull;
  return h;
}

struct Form {
  int sm;
  std::string key;
  Word w;
};

std::string normalised(const std::string& text) {
  std::string t = std::regex_replace(text, std::regex("\\.reuse"), "");
  return std::regex_replace(t, std::regex("R[0-9]+"), "R#");
}

// The forms of the corpora worth running, one of each (by mnemonic and modifiers, registers aside).
std::vector<Form> forms() {
  std::vector<Form> out;
  for (int sm : {80, 86, 89}) {
    std::map<std::string, bool> seen;
    for (const Line& l : read_corpus(std::to_string(sm))) {
      const std::string& t = l.text;
      if (t.empty() || t[0] == '@') continue;
      const std::string mn = t.substr(0, t.find('.') < t.find(' ') ? t.find('.') : t.find(' '));
      if (mn != "HMMA" && mn != "IMMA" && mn != "BMMA" && mn != "DMMA" && mn != "QMMA") continue;
      const std::string key = normalised(t);
      if (seen.count(key)) continue;
      seen[key] = true;
      out.push_back({sm, key, l.w});
    }
  }
  return out;
}

const char* profile_of(int sm) { return sm == 80 ? "nvidia/a100" : sm == 86 ? "nvidia/a10" : sm == 89 ? "nvidia/rtx4090" : "nvidia/h100"; }

// Runs one form on kBlocks warps of random operands; the output words of every lane (empty: a form the engine or this test cannot run).
std::vector<uint32_t> run_form(const Form& f, const std::vector<Line>& corpus, std::mt19937& g, std::string* why) {
  sass::Instr ins = sass::decode(f.w, 0, f.sm);
  const Shape s = shape_of(ins);
  const auto in_range = [&](unsigned base, unsigned n) { return base == sass::kRZ || (n == 0 || base + n <= kScratch); };
  if (!s.ok || !in_range(s.ra, s.a) || !in_range(s.rb, s.b) || !in_range(s.rc, s.c) || !in_range(s.rd, s.c) || (s.sparse && !in_range(s.rm, 1))) {
    *why = "skipped";
    return {};
  }
  // The scaffolding, from words of the corpus (a template each) with their fields patched.
  const auto hex = [](uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
    return std::string(b);
  };
  // Each patched word is decoded again and must read as intended: a mistaken field fails here and not as a strange result.
  std::vector<Word> code;
  const auto emit = [&](const Word& w, const std::string& text) {
    const sass::Instr d = sass::decode(w, 16 * code.size(), f.sm);
    if (sass::to_text(d) != text) throw vtest::Failure("a scaffolding word reads \"" + sass::to_text(d) + "\", not \"" + text + "\"");
    code.push_back(w);
  };
  const Word mov = find_word(corpus, "MOV R[0-9]+, c\\[0x0\\]\\[0x[0-9a-f]+\\]");
  const Word imad = find_word(corpus, "IMAD\\.WIDE\\.U32 R[0-9]+, R[0-9]+, 0x[0-9a-f]+, R[0-9]+");
  const Word ldg = find_word(corpus, "LDG\\.E R[0-9]+, \\[R[0-9]+\\.64(\\+0x[0-9a-f]+)?\\]");
  const Word stg = find_word(corpus, "STG\\.E \\[R[0-9]+\\.64(\\+0x[0-9a-f]+)?\\], R[0-9]+");
  const auto wide_imad = [&](unsigned dst, unsigned a, uint32_t imm, unsigned c) {
    emit(with(with(with(with(imad, 16, 8, dst), 24, 8, a), 32, 32, imm), 64, 8, c),
         "IMAD.WIDE.U32 R" + std::to_string(dst) + ", R" + std::to_string(a) + ", " + hex(imm) + ", R" + std::to_string(c));
  };
  emit(with(find_word(corpus, "S2R R[0-9]+, SR_TID\\.X"), 16, 8, 250), "S2R R250, SR_TID.X");
  emit(with(find_word(corpus, "S2R R[0-9]+, SR_CTAID\\.X"), 16, 8, 251), "S2R R251, SR_CTAID.X");
  emit(with(with(mov, 16, 8, 248), 40, 16, 0x58), "MOV R248, c[0x0][0x160]");   // (the constant offset, in words, at bit 40)
  emit(with(with(mov, 16, 8, 249), 40, 16, 0x59), "MOV R249, c[0x0][0x164]");
  wide_imad(246, 251, 32 * kLaneWords * 4, 248);   // R246:247 = base + ctaid * 32 lanes * 256 B
  wide_imad(246, 250, kLaneWords * 4, 246);        // ... + tid * 256 B: this lane's operands
  wide_imad(244, 251, 32 * kOutWords * 4, 248);    // R244:245 = base + ctaid * 32 * 64 B, and tid * 64 B (the outputs sit above the inputs)
  wide_imad(244, 250, kOutWords * 4, 244);
  const unsigned out_base = kBlocks * 32 * kLaneWords * 4;
  unsigned word = 0;
  const auto off_text = [&](uint64_t o) { return o ? "+" + hex(o) : std::string(); };
  const auto load = [&](unsigned base, unsigned n) {
    if (base == sass::kRZ) {
      word += n;
      return;
    }
    for (unsigned i = 0; i < n; ++i, ++word)
      emit(with(with(with(ldg, 16, 8, base + i), 24, 8, 246), 40, 24, word * 4),
           "LDG.E R" + std::to_string(base + i) + ", [R246.64" + off_text(word * 4) + "]");
  };
  load(s.ra, s.a);
  load(s.rb, s.b);
  load(s.rc, s.c);
  if (s.sparse) load(s.rm, 1);
  code.push_back(f.w);
  for (unsigned i = 0; i < s.c; ++i)
    if (s.rd != sass::kRZ)
      emit(with(with(with(stg, 24, 8, 244), 32, 8, s.rd + i), 40, 24, out_base + i * 4),
           "STG.E [R244.64" + off_text(out_base + i * 4) + "], R" + std::to_string(s.rd + i));
  emit(find_word(corpus, "EXIT"), "EXIT");
  const std::string cubin = make_cubin(f.sm, code);
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu(profile_of(f.sm));
  auto m = sass::load(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size(), mem, prof);
  // Operands: kBlocks x 32 lanes x kLaneWords words, then the outputs.
  std::vector<uint32_t> in(size_t{kBlocks} * 32 * kLaneWords);
  for (unsigned b = 0; b < kBlocks; ++b) {
    const bool wide = b % 2 == 1;
    for (unsigned l = 0; l < 32; ++l) {
      uint32_t* wd = &in[(size_t{b} * 32 + l) * kLaneWords];
      unsigned at = 0;
      const auto fill = [&](unsigned n, bool is_c) {
        for (unsigned i = 0; i < n; ++i, ++at) {
          switch (s.kind) {
            case Shape::Half: wd[at] = is_c && !s.c_f16 ? f32_class(g, wide) : half_class(g, wide) | uint32_t{half_class(g, wide)} << 16; break;
            case Shape::Bf16: wd[at] = is_c ? f32_class(g, wide) : (f32_class(g, wide) >> 16) | (f32_class(g, wide) & 0xffff0000u); break;
            case Shape::Tf32: wd[at] = f32_class(g, wide); break;
            case Shape::Fp8: wd[at] = is_c ? (s.c_f16 ? half_class(g, wide) | uint32_t{half_class(g, wide)} << 16 : f32_class(g, wide)) : g(); break;
            case Shape::F64: {
              const uint64_t v = f64_class(g, wide);
              wd[at] = static_cast<uint32_t>(v);
              wd[++at] = static_cast<uint32_t>(v >> 32);
              ++i;
              break;
            }
            default: {
              const uint32_t h = g();
              wd[at] = is_c ? (h % 4 == 0 ? 0x7fffffffu - (h >> 8) % 100000 : h % 4 == 1 ? 0x80000000u + (h >> 8) % 100000 : g()) : g();
            }
          }
        }
      };
      fill(s.a, false);
      fill(s.b, false);
      fill(s.c, true);
      if (s.sparse) wd[at++] = g();
    }
  }
  const uint64_t buf = mem.alloc(out_base + size_t{kBlocks} * 32 * kOutWords * 4);
  mem.write(buf, in.data(), in.size() * 4);
  exec::LaunchConfig cfg;
  cfg.grid = {kBlocks, 1, 1};
  cfg.block = {32, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &buf, 8);
  sass::launch(*m, "k", cfg, {arg}, mem, prof);
  std::vector<uint32_t> out(size_t{kBlocks} * 32 * kOutWords);
  mem.read(buf + out_base, out.data(), out.size() * 4);
  return out;
}

}  // namespace

VTEST(sass_mma_tile_route_matches_the_general_form) {
  if (const char* child = std::getenv("VGPU_TEST_MMA_CHILD")) {
    // A child run: every form, one line each.
    if (std::strcmp(child, "mma") != 0) return;
    std::map<int, std::vector<Line>> corpora;
    std::mt19937 g(0x5a5a);
    for (const Form& f : forms()) {
      if (!corpora.count(f.sm)) corpora[f.sm] = read_corpus(std::to_string(f.sm));
      std::string why;
      const std::vector<uint32_t> out = run_form(f, corpora[f.sm], g, &why);
      std::printf("OUT sm_%d %s : %s\n", f.sm, f.key.c_str(), out.empty() ? why.c_str() : [&] {
        static char buf[32];
        std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(fnv(out)));
        return buf;
      }());
    }
    std::fflush(stdout);
    return;
  }
  char path[4096];
  const ssize_t n = ::readlink("/proc/self/exe", path, sizeof path - 1);
  if (n <= 0) throw vtest::Failure("cannot find this program to run it again");
  path[n] = 0;
  const auto run_child = [&](const char* simd) {
    std::map<std::string, std::string> res;
    const std::string cmd = std::string("VGPU_TEST_MMA_CHILD=mma VGPU_SIMD=") + simd + " '" + path + "' 2>&1";
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) throw vtest::Failure("cannot run " + cmd);
    char line[8192];
    while (std::fgets(line, sizeof line, p)) {
      const std::string s(line);
      if (s.rfind("OUT ", 0) != 0) {
        if (s.find("FAIL") != std::string::npos || s.find("rror") != std::string::npos) res["!" + s] = s;
        continue;
      }
      const size_t colon = s.rfind(" : ");
      res[s.substr(4, colon - 4)] = s.substr(colon + 3);
    }
    ::pclose(p);
    return res;
  };
  const auto on = run_child("avx2"), off = run_child("off");
  size_t compared = 0, ran = 0;
  for (const auto& [key, v] : on) {
    if (key[0] == '!') throw vtest::Failure("the run with the kernels on reports: " + v);
    const auto it = off.find(key);
    if (it == off.end()) throw vtest::Failure(key + " is missing from the run with the kernels off");
    ++compared;
    if (v.rfind("skipped", 0) != 0) ++ran;
    if (v != it->second) throw vtest::Failure(key + ": kernels on " + v.substr(0, v.size() - 1) + ", off " + it->second.substr(0, it->second.size() - 1));
  }
  for (const auto& [key, v] : off)
    if (key[0] == '!') throw vtest::Failure("the run with the kernels off reports: " + v);
  VCHECK(compared > 100);
  VCHECK(ran > 100);
}

// ---- the warpgroup MMA (HGMMA, IGMMA, QGMMA) ---------------------------------------------------------------------------------
//
// Four warps, the shared-memory operands copied in from the input by the threads (LDG, STS, BAR.SYNC), the descriptors in uniform registers
// (UMOV), the accumulators and any register A loaded, then WARPGROUP.ARRIVE, the corpus's HGMMA word, WARPGROUP.DEPBAR and the stores.

namespace {

constexpr unsigned kSmemBytes = 16384, kGLaneWords = 136, kGOutWords = 128;   // the shared image copied in; words a lane takes in and gives out

uint64_t gdesc_bits(uint32_t start, uint32_t lbo, uint32_t sbo, uint32_t code) {
  return uint64_t{(start & 0x3FFFF) >> 4} | (uint64_t{(lbo & 0x3FFFF) >> 4} << 16) | (uint64_t{(sbo & 0x3FFFF) >> 4} << 32) | (uint64_t{code} << 62);
}

// A descriptor whose rows x K operand lies inside the image.
uint64_t pick_gdesc(std::mt19937& g, bool k_major, uint32_t eb, uint32_t rows, uint32_t K) {
  for (;;) {
    const uint32_t code = g() % 4;
    exec::WgmmaDesc d;
    d.swizzle = (uint32_t[]){0, 128, 64, 32}[code];
    d.start = 16 * (g() % 128);
    d.lbo = 16 * (g() % 96);
    d.sbo = 16 * (g() % 96);
    uint64_t hi = 0;
    for (uint32_t mn = 0; mn < rows; ++mn)
      for (uint32_t k = 0; k < K; ++k) hi = std::max(hi, exec::wgmma_smem_offset(d, k_major, eb, mn, k));
    if (hi + eb <= kSmemBytes) return gdesc_bits(d.start, d.lbo, d.sbo, code);
  }
}

std::vector<Form> gforms() {
  std::vector<Form> out;
  std::map<std::string, bool> seen;
  for (const Line& l : read_corpus("90a")) {
    const std::string& t = l.text;
    if (t.empty() || t[0] == '@') continue;
    const std::string mn = t.substr(0, t.find('.'));
    if (mn != "HGMMA" && mn != "IGMMA" && mn != "QGMMA") continue;
    const std::string key = normalised(std::regex_replace(t, std::regex("UR[0-9]+"), "UR#"));
    if (seen.count(key)) continue;
    seen[key] = true;
    out.push_back({90, key, l.w});
  }
  return out;
}

std::vector<uint32_t> run_gform(const Form& f, const std::vector<Line>& corpus, std::mt19937& g, std::string* why) {
  sass::Instr ins = sass::decode(f.w, 0, f.sm);
  const unsigned N = ins.f[0], K = ins.f[1], tb = ins.f[2], flags = ins.f[5];
  const bool reg_a = ins.f[3], sparse = ins.f[4];
  const bool is_int = ins.op == sass::Op::IGMMA;
  const bool acc32 = is_int || (tb & 0x08);
  const bool tf32 = ins.op == sass::Op::HGMMA && (tb & 0x20), bf16 = ins.op == sass::Op::HGMMA && (tb & 0x10);
  const unsigned eb = ins.op == sass::Op::HGMMA ? (tf32 ? 4 : 2) : 1;
  const unsigned nregs = acc32 ? N / 2 : N / 4;
  const unsigned rd = ins.dst[0].reg, rc = ins.src[reg_a ? 2 : 1].reg, ra = reg_a ? ins.src[0].reg : 0;
  const unsigned desc_ur = ins.src[reg_a ? 1 : 0].reg, rmeta = sparse ? (ins.f[7] & 0xff) : 0;
  if (N == 0 || N % 8 || N > 256 || rd == sass::kRZ || rd + nregs > kScratch || (rc != sass::kRZ && rc + nregs > kScratch) || (reg_a && ra + 4 > kScratch) ||
      desc_ur + 4 > 56 || (sparse && rmeta >= kScratch)) {
    *why = "skipped";
    return {};
  }
  const auto hex = [](uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
    return std::string(b);
  };
  std::vector<Word> code;
  const auto emit = [&](const Word& w, const std::string& text) {
    const sass::Instr d = sass::decode(w, 16 * code.size(), f.sm);
    if (sass::to_text(d) != text) throw vtest::Failure("a scaffolding word reads \"" + sass::to_text(d) + "\", not \"" + text + "\"");
    code.push_back(w);
  };
  const Word ldc = find_word(corpus, "LDC\\.64 R[0-9]+, c\\[0x0\\]\\[0x210\\]");   // (the parameter base is the cubin's default, 0x160)
  const Word imad = find_word(corpus, "IMAD\\.WIDE\\.U32 R[0-9]+, R[0-9]+, 0x[0-9a-f]+, R[0-9]+");
  const Line& ldg_line = find_line(corpus, "LDG\\.E R[0-9]+, (desc\\[UR[0-9]+\\])?\\[R[0-9]+\\.64(\\+0x[0-9a-f]+)?\\]");
  const Line& stg_line = find_line(corpus, "STG\\.E (desc\\[UR[0-9]+\\])?\\[R[0-9]+\\.64(\\+0x[0-9a-f]+)?\\], R[0-9]+");
  const Word ldg = ldg_line.w, stg = stg_line.w;
  const std::string ldg_desc = desc_of(ldg_line), stg_desc = desc_of(stg_line);
  const Word sts = find_word(corpus, "STS \\[R[0-9]+\\], R[0-9]+");
  const Word umov = find_word(corpus, "UMOV UR[0-9]+, 0x[0-9a-f]+");
  const auto wide_imad = [&](unsigned dst, unsigned a, uint32_t imm, unsigned c) {
    emit(with(with(with(with(imad, 16, 8, dst), 24, 8, a), 32, 32, imm), 64, 8, c),
         "IMAD.WIDE.U32 R" + std::to_string(dst) + ", R" + std::to_string(a) + ", " + hex(imm) + ", " + (c == sass::kRZ ? std::string("RZ") : "R" + std::to_string(c)));
  };
  const unsigned lane_in = kGLaneWords * 4, block_in = kSmemBytes + 128 * lane_in, lane_out = kGOutWords * 4;
  emit(with(find_word(corpus, "S2R R[0-9]+, SR_TID\\.X"), 16, 8, 250), "S2R R250, SR_TID.X");
  emit(with(find_word(corpus, "S2R R[0-9]+, SR_CTAID\\.X"), 16, 8, 251), "S2R R251, SR_CTAID.X");
  emit(with(with(ldc, 16, 8, 248), 38, 16, 0x160), "LDC.64 R248, c[0x0][0x160]");
  wide_imad(242, 251, block_in, 248);   // R242:243 = this block's image + this thread's word
  wide_imad(242, 250, 4, 242);
  wide_imad(246, 251, block_in, 248);   // R246:247 = this block's input, this lane's operands (above the image)
  wide_imad(246, 250, lane_in, 246);
  wide_imad(244, 251, 128 * lane_out, 248);   // R244:245 = the block's outputs, this lane's (above all the inputs)
  wide_imad(244, 250, lane_out, 244);
  wide_imad(238, 250, 4, sass::kRZ);    // R238 = this thread's shared address
  const unsigned out_base = kBlocks * block_in;
  const auto off_text = [&](uint64_t o) { return o ? "+" + hex(o) : std::string(); };
  // The image: 4096 words, thread t taking words t + 128 i.
  for (unsigned i = 0; i < kSmemBytes / 4 / 128; ++i) {
    emit(with(with(with(ldg, 16, 8, 240), 24, 8, 242), 40, 24, i * 512), "LDG.E R240, " + ldg_desc + "[R242.64" + off_text(i * 512) + "]");
    emit(with(with(with(sts, 24, 8, 238), 32, 8, 240), 40, 24, i * 512), "STS [R238" + off_text(i * 512) + "], R240");
  }
  emit(find_word(corpus, "BAR\\.SYNC\\.DEFER_BLOCKING 0x0"), "BAR.SYNC.DEFER_BLOCKING 0x0");
  // The descriptors: A's (when it is in shared memory) and B's in the four uniform registers from desc_ur.
  const unsigned Kp = sparse ? K / 2 : K;
  const bool a_kmajor = !(flags & 1), b_kmajor = !(flags & 2);
  const uint64_t da = reg_a ? 0 : pick_gdesc(g, a_kmajor, eb, 64, Kp), db = pick_gdesc(g, b_kmajor, eb, N, K);
  const uint64_t descs[2] = {da, db};
  for (unsigned i = 0; i < 4; ++i) {
    const uint32_t v = static_cast<uint32_t>(descs[i / 2] >> (32 * (i % 2)));
    emit(with(with(umov, 16, 8, desc_ur + i), 32, 32, v), "UMOV UR" + std::to_string(desc_ur + i) + ", " + hex(v));
  }
  unsigned word = 0;
  const auto load = [&](unsigned base, unsigned n) {
    if (base == sass::kRZ) {
      word += n;
      return;
    }
    for (unsigned i = 0; i < n; ++i, ++word)
      emit(with(with(with(ldg, 16, 8, base + i), 24, 8, 246), 40, 24, word * 4),
           "LDG.E R" + std::to_string(base + i) + ", " + ldg_desc + "[R246.64" + off_text(word * 4) + "]");
  };
  if (reg_a) load(ra, 4);
  load(rc, nregs);
  if (sparse) load(rmeta, 1);
  emit(find_word(read_corpus("90a"), "WARPGROUP\\.ARRIVE"), "WARPGROUP.ARRIVE");
  code.push_back(f.w);
  emit(find_word(read_corpus("90a"), "WARPGROUP\\.DEPBAR\\.LE gsb0, 0x0"), "WARPGROUP.DEPBAR.LE gsb0, 0x0");
  for (unsigned i = 0; i < nregs; ++i)
    emit(with(with(with(stg, 24, 8, 244), 32, 8, rd + i), 40, 24, out_base + i * 4), "STG.E " + stg_desc + "[R244.64" + off_text(out_base + i * 4) + "], R" + std::to_string(rd + i));
  emit(find_word(corpus, "EXIT"), "EXIT");
  const std::string cubin = make_cubin(f.sm, code, kSmemBytes);
  MemoryManager mem{1 << 26};
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto m = sass::load(reinterpret_cast<const uint8_t*>(cubin.data()), cubin.size(), mem, prof);
  // Per block: the shared image, then 128 lanes of operands.
  std::vector<uint32_t> in(size_t{kBlocks} * block_in / 4);
  for (unsigned b = 0; b < kBlocks; ++b) {
    const bool wide = b % 2 == 1;
    uint32_t* blk = &in[size_t{b} * block_in / 4];
    for (unsigned i = 0; i < kSmemBytes / 4; ++i) {
      if (is_int || eb == 1) blk[i] = g();
      else if (tf32) blk[i] = f32_class(g, wide);
      else blk[i] = bf16 ? (f32_class(g, wide) >> 16) | (f32_class(g, wide) & 0xffff0000u) : half_class(g, wide) | uint32_t{half_class(g, wide)} << 16;
    }
    for (unsigned l = 0; l < 128; ++l) {
      uint32_t* wd = blk + kSmemBytes / 4 + size_t{l} * kGLaneWords;
      unsigned at = 0;
      if (reg_a)
        for (unsigned i = 0; i < 4; ++i, ++at)
          wd[at] = is_int || eb == 1 ? g() : tf32 ? f32_class(g, wide) : bf16 ? (f32_class(g, wide) >> 16) | (f32_class(g, wide) & 0xffff0000u) : half_class(g, wide) | uint32_t{half_class(g, wide)} << 16;
      for (unsigned i = 0; i < nregs; ++i, ++at) {
        const uint32_t h = g();
        wd[at] = is_int ? (h % 4 == 0 ? 0x7fffffffu - (h >> 8) % 100000 : h % 4 == 1 ? 0x80000000u + (h >> 8) % 100000 : g())
                        : acc32 ? f32_class(g, wide) : half_class(g, wide) | uint32_t{half_class(g, wide)} << 16;
      }
      if (sparse) wd[at++] = g();
    }
  }
  const uint64_t buf = mem.alloc(out_base + size_t{kBlocks} * 128 * lane_out);
  mem.write(buf, in.data(), in.size() * 4);
  exec::LaunchConfig cfg;
  cfg.grid = {kBlocks, 1, 1};
  cfg.block = {128, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &buf, 8);
  sass::launch(*m, "k", cfg, {arg}, mem, prof);
  std::vector<uint32_t> out(size_t{kBlocks} * 128 * kGOutWords);
  mem.read(buf + out_base, out.data(), out.size() * 4);
  return out;
}

}  // namespace

VTEST(sass_gmma_tile_route_matches_the_general_form) {
  if (const char* child = std::getenv("VGPU_TEST_MMA_CHILD")) {
    if (std::strcmp(child, "gmma") != 0) return;
    const std::vector<Line> corpus = read_corpus("90a");
    std::mt19937 g(0x6a6a);
    for (const Form& f : gforms()) {
      std::string why;
      const std::vector<uint32_t> out = run_gform(f, corpus, g, &why);
      if (out.empty()) std::printf("OUT sm_%d %s : %s\n", f.sm, f.key.c_str(), why.c_str());
      else std::printf("OUT sm_%d %s : %016llx\n", f.sm, f.key.c_str(), static_cast<unsigned long long>(fnv(out)));
    }
    std::fflush(stdout);
    return;
  }
  char path[4096];
  const ssize_t n = ::readlink("/proc/self/exe", path, sizeof path - 1);
  if (n <= 0) throw vtest::Failure("cannot find this program to run it again");
  path[n] = 0;
  const auto run_child = [&](const char* simd) {
    std::map<std::string, std::string> res;
    const std::string cmd = std::string("VGPU_TEST_MMA_CHILD=gmma VGPU_SIMD=") + simd + " '" + path + "' 2>&1";
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) throw vtest::Failure("cannot run " + cmd);
    char line[8192];
    while (std::fgets(line, sizeof line, p)) {
      const std::string s(line);
      if (s.rfind("OUT ", 0) != 0) {
        if (s.find("FAIL") != std::string::npos || s.find("rror") != std::string::npos) res["!" + s] = s;
        continue;
      }
      const size_t colon = s.rfind(" : ");
      res[s.substr(4, colon - 4)] = s.substr(colon + 3);
    }
    ::pclose(p);
    return res;
  };
  const auto on = run_child("avx2"), off = run_child("off");
  size_t compared = 0, ran = 0;
  for (const auto& [key, v] : on) {
    if (key[0] == '!') throw vtest::Failure("the run with the kernels on reports: " + v);
    const auto it = off.find(key);
    if (it == off.end()) throw vtest::Failure(key + " is missing from the run with the kernels off");
    ++compared;
    if (v.rfind("skipped", 0) != 0) ++ran;
    if (v != it->second) throw vtest::Failure(key + ": kernels on " + v.substr(0, v.size() - 1) + ", off " + it->second.substr(0, it->second.size() - 1));
  }
  for (const auto& [key, v] : off)
    if (key[0] == '!') throw vtest::Failure("the run with the kernels off reports: " + v);
  VCHECK(compared > 50);
  VCHECK(ran > 40);
}

VTEST_MAIN
