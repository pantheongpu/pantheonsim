// The SASS decoder's output must not change when it is made faster (round 10, "load"): every Instr field and every
// text a user can see -- to_text, the mnemonic, the error a refused word throws -- for every word of the
// disassembler corpora (nvidia/tests/data/sass, sm_75 to sm_120), each decoded for every architecture, and for
// a few hundred thousand words made by flipping bits of the corpus words (so refusals, odd forms and the
// architecture-specific paths are covered too). Each pass is folded into a 128-bit digest and compared with the
// digest the unmodified decoder gave (the table below). (Round 10b regenerated the table for FFMA2/FMUL2/FADD2: with the corpora grown by the
// packed fp32 lines, the unmodified decoder's dump and the new one's differ in exactly the words with low opcode bits 0x49-0x4b on sm_100 and sm_103
// that have an operand form -- docs/perf-design/round10-ffma2.md, "Same bits".)
// (The table was regenerated again for VIADD.16x2 on sm_90 (add.u16x2 / add.s16x2) and the packed VIADDMNMX / VIMNMX forms (S16x2 and U16x2, with
// the per-half choice predicates of the sm_90 VIMNMX): with the corpora grown by those rows, the unmodified decoder's dump and the new one's differ in
// exactly the words with bit 73 set that decode as VIADD (before sm_120), VIADDMNMX (every architecture) or VIMNMX (sm_90 only).)
// (The flipped and random digests were regenerated for HMNMX2 and FMNMX.XORSIGN (pantheonsim #547: main's dump and the new one's differ in
// exactly the FMNMX and HMNMX2 words) and for the one-conversion f16/bf16 F2FP.MERGE_C (bit 78 set, family clear, type 4 or 5, which was
// read as a TF32 or unnamed pack) and the two-operand f16/bf16 F2FP.SATFINITE (bit 77, a type of 2 or 3, which was an unnamed pack): they differ
// in exactly the F2FP words with those types; the corpora digest is unchanged).)
//
// *** 2026-11-16: the flipped and random digests were RE-RECORDED on purpose for commit d9e19aaa (half-precision MUFU). That commit added
// one line to dec_mufu in src/sass/decode.cpp, `if (w.bit(73)) f16_part(ins, w, 1);`, so a MUFU word with bit 73 set (the .F16 form) now reads
// the .H1 / low-half select of its source (bits 60-61): the source operand gets its part (f[5], the selector text ".H1") where it had none.
// Verified by dumping both passes (VGPU_DECODE_DUMP) from a build with and without that line and diffing them: 97 dump lines differ (5 in
// "flipped", 92 in "random", 24 distinct words over the architectures they decode on), every one a MUFU word with bit 73 set, and the
// only differences are that part (the f[5] field, the source selector, the ".H1" in the text); the corpora digest is UNCHANGED (no word of the
// real-code corpora differs), and without the line the old digests came out again. ***
//
// *** 2026-11-16 (second): the flipped and random digests were RE-RECORDED on purpose again, for the sm_90+ MUFU half-precision encoding
// (r8-bf16): dec_mufu reads .F16 from instruction bit 73 up to sm_89 and, from sm_90, .F16 from bit 72 and .BF16 from bit 73 (nvcc output
// for sm_90, sm_100, sm_120; an H100 ran the BF16 form). Verified by dumping both passes (VGPU_DECODE_DUMP) from a build with the
// architecture test and from one without it (the old rule, which gave the previous digests again) and diffing them: 108 dump lines differ
// in "flipped" and 100 in "random" (86 distinct words), all of them MUFU words decoded for sm_90, sm_100, sm_103 or sm_120 (none for an
// older architecture, none that is not a MUFU), and the differences are only the F16 / BF16 mod, f[1] (.F16) and f[2] (.BF16) and the
// text they print; the corpora digest is UNCHANGED (no word of the real-code corpora differs). ***
//
// *** 2026-11-17: the flipped and random digests were re-recorded for #694 (a RET word with a condition predicate in bits 87-90 now keeps it
// as a source). The corpora digest is unchanged. ***
//
// *** 2026-11-18 (round 12, ci124): the flipped and random digests were RE-RECORDED on purpose for the register-mask form of ELECT that ptxas 12.4 writes
// (dec_elect: bit 72 set makes the third operand a uniform register, with bit 91 as its bitwise not; before, the word was read as a predicate source and
// `elect.sync` found no leader). Verified by dumping both passes (VGPU_DECODE_DUMP) from a build with the `if (w.bit(72))` branch and one with it switched off
// (which gave the previous digests again) and diffing them: 5 dump lines differ in "flipped" and 28 in "random" (7 distinct words, each on 4 architectures),
// every one an ELECT word with bit 72 set, and the differences are only that source operand (a UR with its not, where a predicate was), f[0] = 1 and the text;
// the corpora digest is UNCHANGED (no word of the real-code corpora has bit 72 set in an ELECT). The digests are the same on a CUDA 12.4 and a 13.0 build. ***
//
// *** 2026-11-23 (xarch, #749): all three digests were RE-RECORDED on purpose for the sm_100+ `IADD` word (opcode 0x35): dec_iadd now reads the
// carry-out predicate (bits 81-83, when not PT) as a second destination and, with bit 74 (.X), the carry-in predicate (bits 87-90) as a third source, with
// a negated source of a carry-in form a one's complement, as IADD3's. `add.cc.u64` is `IADD.64 RZ, P0, R2, UR8` and `addc` the `IADD.X` after it. Verified by dumping
// both passes (VGPU_DECODE_DUMP) from a build with the old dec_iadd and one with the new and diffing them: 45 dump lines differ in "corpora", 148 in "flipped"
// and 134 in "random", every one an IADD word decoded for sm_100, sm_103 or sm_120 (the corpora's other-architecture words that decode as IADD there), and the
// differences are only the carry-out destination, the carry-in source and the X modifier. ***
//
// VGPU_DECODE_PRINT=1 prints the digests (to regenerate the table after a decoder change that is meant to change
// them); VGPU_DECODE_DUMP=<dir> writes the full dump of every pass, so two builds can be diffed to see which
// instruction differs; VGPU_DECODE_BENCH=1 times decode and to_text over the corpora.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace vgpu::sass {
void annotate_instr(Instr& ins);   // src/sass/exec.cpp
bool runs_instr(const Instr& ins);
}

namespace {

constexpr int kArchs[] = {75, 80, 86, 89, 90, 100, 103, 120};
constexpr const char* kCorpora[] = {"sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a",
                                    "sm_100", "sm_100a", "sm_103a", "sm_120", "sm_120a"};

struct Digest {
  uint64_t a = 0xcbf29ce484222325ull, b = 0x84222325cbf29ce4ull;
  size_t bytes = 0;
  void add(const std::string& s) {
    for (unsigned char c : s) {
      a = (a ^ c) * 0x100000001b3ull;
      b = (b + c + 1) * 0x9e3779b97f4a7c15ull;
      b ^= b >> 29;
    }
    bytes += s.size();
  }
  std::string text() const {
    char t[64];
    std::snprintf(t, sizeof t, "%016llx%016llx", static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
    return t;
  }
};

void put(std::string& s, const sass::Operand& o) {
  s += "{" + std::to_string(static_cast<int>(o.kind)) + "," + std::to_string(o.reg) + "," + std::to_string(o.imm) + "," +
       std::to_string(o.width) + "," + (o.neg ? "n" : "-") + (o.abs ? "a" : "-") + (o.bnot ? "b" : "-") +
       (o.reuse ? "r" : "-") + ",s=" + o.suffix + ",t=" + o.text + "," + std::to_string(o.fbits) + "," +
       std::to_string(o.fwidth) + "," + std::to_string(o.slot) + "}";
}

// Everything a decoded word shows: all of Instr, its text, and whether the executor takes it.
std::string dump(const sass::Word& w, uint64_t pc, int sm) {
  std::string s;
  try {
    sass::Instr ins = sass::decode(w, pc, sm);
    sass::annotate_instr(ins);
    s += std::to_string(static_cast<int>(ins.op)) + " " + sass::op_name(ins.op) + " m=" + ins.mnemonic + " mods=";
    for (const std::string& m : ins.mods) s += m + "|";
    s += " g=" + std::to_string(ins.guard) + (ins.guard_not ? "!" : "") + (ins.guard_uniform ? "u" : "") +
         " pc=" + std::to_string(ins.pc) + " sm=" + std::to_string(ins.sm) + " dst=";
    for (const sass::Operand& o : ins.dst) put(s, o);
    s += " src=";
    for (const sass::Operand& o : ins.src) put(s, o);
    s += " f=";
    for (uint32_t f : ins.f) s += std::to_string(f) + ",";
    s += " cls=" + std::to_string(ins.cls) + (ins.tensor ? "T" : "-") + (ins.straight ? "S" : "-") +
         " runs=" + (sass::runs_instr(ins) ? "1" : "0") + " text=" + sass::to_text(ins);
  } catch (const Error& e) {
    s += "error " + std::to_string(static_cast<int>(e.code())) + " " + e.message();
  } catch (const std::exception& e) {
    s += std::string("exception ") + e.what();
  }
  return s;
}

std::vector<sass::Word> corpus_words() {
  std::vector<sass::Word> out;
  for (const char* name : kCorpora) {
    std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/sass/" + name + ".txt");
    if (!in) throw vtest::Failure(std::string("no corpus ") + name);
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line[0] == '#') continue;
      const size_t t1 = line.find('\t');
      sass::Word w;
      w.hi = std::stoull(line.substr(0, 16), nullptr, 16);
      w.lo = std::stoull(line.substr(16, 16), nullptr, 16);
      (void)t1;
      out.push_back(w);
    }
  }
  return out;
}

struct Rng {
  uint64_t s = 0x2545F4914F6CDD1Dull;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
};

Digest run_pass(const char* name, const std::vector<std::pair<sass::Word, uint64_t>>& words, const std::vector<int>& archs) {
  Digest d;
  const char* dir = std::getenv("VGPU_DECODE_DUMP");
  std::ofstream out;
  if (dir && *dir) out.open(std::string(dir) + "/" + name + ".txt");
  for (const auto& [w, pc] : words)
    for (int sm : archs) {
      const std::string line = dump(w, pc, sm) + "\n";
      d.add(line);
      if (out) out << (static_cast<unsigned long long>(w.hi)) << ' ' << w.lo << ' ' << pc << ' ' << sm << ' ' << line;
    }
  return d;
}

void check(const char* name, const Digest& d, const char* want) {
  if (std::getenv("VGPU_DECODE_PRINT")) std::fprintf(stderr, "digest %s %s (%zu bytes)\n", name, d.text().c_str(), d.bytes);
  VCHECK_EQ(d.text(), std::string(want));
}

}  // namespace

VTEST(the_corpora_decode_to_the_same_bits_on_every_architecture) {
  std::vector<std::pair<sass::Word, uint64_t>> words;
  uint64_t pc = 0;
  for (const sass::Word& w : corpus_words()) words.push_back({w, (pc += 16) & 0xfffff0});
  const std::vector<int> archs(std::begin(kArchs), std::end(kArchs));
  check("corpora", run_pass("corpora", words, archs), "58343ad8c0f1fb6e73bc5b95b1d259d7");
}

VTEST(words_with_flipped_bits_decode_to_the_same_bits) {
  const std::vector<sass::Word> base = corpus_words();
  Rng rng;
  std::vector<std::pair<sass::Word, uint64_t>> words;
  for (size_t i = 0; i < 120000; ++i) {
    sass::Word w = base[rng.next() % base.size()];
    const unsigned flips = 1 + rng.next() % 3;
    for (unsigned k = 0; k < flips; ++k) {
      const unsigned bit = rng.next() % 128;
      (bit < 64 ? w.lo : w.hi) ^= uint64_t{1} << (bit % 64);
    }
    words.push_back({w, 16 * (rng.next() % 4096)});
  }
  const std::vector<int> archs = {75, 86, 90, 100, 120};
  check("flipped", run_pass("flipped", words, archs), "47b07a0f78bf31698c9609744e52b639");
}

VTEST(random_opcodes_decode_to_the_same_bits) {
  Rng rng;
  std::vector<std::pair<sass::Word, uint64_t>> words;
  for (size_t i = 0; i < 40000; ++i) {
    sass::Word w;
    w.lo = rng.next();
    w.hi = rng.next() & 0x1fffffffffffull;
    words.push_back({w, 16 * (rng.next() % 4096)});
  }
  const std::vector<int> archs = {80, 89, 103, 120};
  check("random", run_pass("random", words, archs), "912ce8a6c57091999e2693f3bdfa83b5");
}

VTEST(bench_decode) {
  if (!std::getenv("VGPU_DECODE_BENCH")) return;
  std::vector<sass::Word> words = corpus_words();
  const int reps = 5;
  for (const char* what : {"decode", "decode+annotate", "decode+to_text"}) {
    size_t n = 0, sink = 0;
    double best = 1e9;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      n = 0;
      uint64_t pc = 0;
      for (const sass::Word& w : words) {
        try {
          sass::Instr ins = sass::decode(w, pc += 16, 86);
          if (what[6] == '+' && what[7] == 'a') sass::annotate_instr(ins);
          if (what[6] == '+' && what[7] == 't') sink += sass::to_text(ins).size();
          sink += ins.src.size();
        } catch (const Error&) {
        }
        ++n;
      }
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (s < best) best = s;
    }
    std::fprintf(stderr, "bench %-16s %zu words, best of %d: %.1f ms, %.0f ns/word (%zu)\n", what, n, reps, best * 1e3,
                 best * 1e9 / static_cast<double>(n), sink);
  }
}

// VGPU_UNSUPPORTED_BENCH=<dir of *.cubin>: the time sass::unsupported takes on each cubin in turn, in this one
// process (the first-launch cost of a process that loads them in this order).
VTEST(bench_unsupported) {
  const char* dir = std::getenv("VGPU_UNSUPPORTED_BENCH");
  if (!dir || !*dir) return;
  std::vector<std::string> files;
  if (DIR* d = opendir(dir)) {
    while (const dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 6 && n.substr(n.size() - 6) == ".cubin") files.push_back(std::string(dir) + "/" + n);
    }
    closedir(d);
  }
  std::sort(files.begin(), files.end());
  double total = 0;
  for (const std::string& f : files) {
    std::ifstream in(f, std::ios::binary);
    in.seekg(0, std::ios::end);
    std::string image(static_cast<size_t>(in.tellg()), '\0');
    in.seekg(0);
    in.read(image.data(), static_cast<std::streamsize>(image.size()));
    const auto t0 = std::chrono::steady_clock::now();
    const std::string why = sass::unsupported(reinterpret_cast<const uint8_t*>(image.data()), image.size());
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    total += s;
    std::fprintf(stderr, "unsupported %s: %.1f ms [%s]\n", f.substr(f.rfind('/') + 1).c_str(), s * 1e3, why.c_str());
  }
  std::fprintf(stderr, "unsupported total: %.1f ms over %zu cubins\n", total * 1e3, files.size());
  // The same cubins again in this process: what is left when every word has been seen.
  double again = 0;
  for (const std::string& f : files) {
    std::ifstream in(f, std::ios::binary);
    in.seekg(0, std::ios::end);
    std::string image(static_cast<size_t>(in.tellg()), '\0');
    in.seekg(0);
    in.read(image.data(), static_cast<std::streamsize>(image.size()));
    const auto t0 = std::chrono::steady_clock::now();
    sass::unsupported(reinterpret_cast<const uint8_t*>(image.data()), image.size());
    again += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  std::fprintf(stderr, "unsupported again: %.1f ms\n", again * 1e3);
}

VTEST_MAIN
