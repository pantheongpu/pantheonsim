// The SASS tier's analysis (src/sass/tier.cpp) over every instruction of NVIDIA's disassembler corpora
// (nvidia/tests/data/sass): what it accepts must be well formed, whatever the architecture. The handlers' arithmetic is
// held to the interpreter's by nvidia/tests/e2e/run_sass_tier.sh; this is the part that needs no program to run --
// operand resolution, constant rows, register bounds, the guard -- and it runs on every instruction shape the
// corpora hold (41,000 lines over 11 targets), which no kernel of ours reaches.
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>

#include "sass/tier.hpp"
#include "vgpu/error.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace vgpu::sass {
void annotate_instr(Instr& ins);   // src/sass/exec.cpp
}

namespace {

struct Tally {
  size_t total = 0, straight = 0, accepted = 0;
};

// The rows a source can name, and what a handler would read from them, must lie inside the program.
void check_src(const sass::tier::Src& s, const sass::tier::Program& p, size_t pool_extra, const char* what) {
  using sass::tier::SrcKind;
  if (s.kind == SrcKind::Pool && s.off + pool_extra >= p.pool.size()) throw vtest::Failure(std::string(what) + ": a pool row past the pool");
  if (s.kind == SrcKind::Row && s.off % 32) throw vtest::Failure(std::string(what) + ": a register row off a row boundary");
  if (s.kind == SrcKind::Row && s.off / 32 > 255) throw vtest::Failure(std::string(what) + ": a register past R255");
}

void check_corpus(const std::string& file, int sm) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/sass/" + file;
  std::ifstream in(path);
  if (!in) throw vtest::Failure("no corpus at " + path);
  std::vector<sass::Instr> instrs;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
    const std::string hexw = line.substr(0, t1);
    sass::Word w;
    w.hi = std::stoull(hexw.substr(0, 16), nullptr, 16);
    w.lo = std::stoull(hexw.substr(16, 16), nullptr, 16);
    const uint64_t pc = std::stoull(line.substr(t1 + 1, t2 - t1 - 1), nullptr, 16);
    try {
      sass::Instr ins = sass::decode(w, pc, sm);
      sass::annotate_instr(ins);
      instrs.push_back(std::move(ins));
    } catch (const std::exception&) {
    }
  }
  VCHECK(instrs.size() > 500);
  sass::tier::Program prog;
  prog.ops.resize(instrs.size());
  std::map<std::string, Tally> by_op;
  size_t accepted = 0;
  for (size_t i = 0; i < instrs.size(); ++i) {
    const sass::Instr& ins = instrs[i];
    sass::tier::MOp& op = prog.ops[i];
    sass::tier::analyze(ins, sm, prog, op);
    Tally& t = by_op[sass::op_name(ins.op)];
    ++t.total;
    t.straight += ins.straight;
    VCHECK(op.ins == &ins);
    VCHECK(op.straight == ins.straight);
    VCHECK(op.cls == ins.cls);
    VCHECK(op.never == (ins.guard == sass::kPT && ins.guard_not));
    VCHECK(op.unguarded == (ins.guard == sass::kPT && !ins.guard_not));
    if (op.form == sass::tier::Form::None) {
      VCHECK(op.maxreg == 0);
      continue;
    }
    ++t.accepted;
    ++accepted;
    // Only instructions step_warp runs as straight-line ones can have a handler (a plain BRA, BSSY or NOP ends a run, and tier_run runs it).
    const bool ends = op.form == sass::tier::Form::Bra || op.form == sass::tier::Form::Bssy || op.form == sass::tier::Form::Nop;
    const bool terminal = op.form == sass::tier::Form::Plop3 || op.form == sass::tier::Form::S2r || op.form == sass::tier::Form::Sel;   // end a run, and have handlers
    VCHECK(ins.straight != (ends || terminal));
    VCHECK(op.maxreg <= 256);   // 0 is possible: constants into RZ
    const std::string what = std::string(sass::tier::form_name(op.form)) + " " + sass::to_text(ins);
    const bool mem = op.form >= sass::tier::Form::Ldg && op.form <= sass::tier::Form::Stl;   // s[] holds raw register numbers there
    // A double operand, and IMAD.WIDE's and IMAD.HI's addend, are pairs of rows.
    for (int i = 0; i < 3 && !mem; ++i) {
      const bool pair = op.form == sass::tier::Form::Dop || (i == 2 && (op.form == sass::tier::Form::ImadWide ||
                                                                       (op.form == sass::tier::Form::ImadHi && (op.aux & 4))));
      check_src(op.s[i], prog, pair ? 1 : 0, what.c_str());
    }
    // A destination register the handler writes lies inside the file the op claims to need.
    if (op.dst != sass::tier::kNoReg && op.form != sass::tier::Form::Isetp && op.form != sass::tier::Form::Fsetp &&
        op.form != sass::tier::Form::IsetpEx && op.form != sass::tier::Form::Plop3 && !ends)
      VCHECK(op.dst < op.maxreg);
    if (ends) VCHECK(op.fn == nullptr);
  }
  // pool[0] is the zero row and every constant row is the same value in all 32 lanes except a double's pair halves.
  VCHECK(!prog.pool.empty());
  for (unsigned l = 0; l < 32; ++l) VCHECK(prog.pool[0].v[l] == 0);
  for (const auto& r : prog.pool)
    for (unsigned l = 1; l < 32; ++l) VCHECK(r.v[l] == r.v[0]);
  // Not vacuous: a corpus of an architecture with this many integer and float instructions gives the tier many.
  VCHECK(accepted > 200);
  const bool report = std::getenv("VGPU_SASS_REPORT") != nullptr;
  if (report) {
    for (const auto& [op, t] : by_op)
      if (t.straight) std::fprintf(stderr, "  %-10s %5zu of %5zu straight-line forms taken\n", op.c_str(), t.accepted, t.straight);
    std::fprintf(stderr, "%s: %zu of %zu instructions have a handler\n", file.c_str(), accepted, instrs.size());
  }
  // The forms the tier exists for must be taken somewhere in every corpus that has them.
  for (const char* must : {"IMAD", "IADD3", "LOP3", "ISETP", "SHF", "FFMA", "LDG", "STG"})
    VCHECK(by_op[must].accepted > 0);
}

}  // namespace

VTEST(tier_analysis_sm_80) { check_corpus("sm_80.txt", 80); }
VTEST(tier_analysis_sm_86) { check_corpus("sm_86.txt", 86); }
VTEST(tier_analysis_sm_89) { check_corpus("sm_89.txt", 89); }
VTEST(tier_analysis_sm_90) { check_corpus("sm_90.txt", 90); }
VTEST(tier_analysis_sm_120) { check_corpus("sm_120.txt", 120); }

// Turing's global loads take an address form the handlers do not know: the interpreter keeps them.
VTEST(tier_leaves_turing_global_memory_to_the_interpreter) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/sass/sm_75.txt";
  std::ifstream in(path);
  VCHECK(static_cast<bool>(in));
  sass::tier::Program prog;
  size_t seen = 0;
  std::vector<sass::Instr> keep;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
    sass::Word w;
    w.hi = std::stoull(line.substr(0, 16), nullptr, 16);
    w.lo = std::stoull(line.substr(16, 16), nullptr, 16);
    try {
      sass::Instr ins = sass::decode(w, std::stoull(line.substr(t1 + 1, t2 - t1 - 1), nullptr, 16), 75);
      sass::annotate_instr(ins);
      if (ins.op != sass::Op::LDG && ins.op != sass::Op::STG) continue;
      keep.push_back(std::move(ins));
    } catch (const std::exception&) {
    }
  }
  prog.ops.resize(keep.size());
  for (size_t i = 0; i < keep.size(); ++i) {
    sass::tier::analyze(keep[i], 75, prog, prog.ops[i]);
    VCHECK(prog.ops[i].form == sass::tier::Form::None);
    ++seen;
  }
  VCHECK(seen > 0);
}

VTEST_MAIN
