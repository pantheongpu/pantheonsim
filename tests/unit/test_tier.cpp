// The SASS tier's analysis (src/sass/tier.cpp) over every instruction of NVIDIA's disassembler corpora
// (nvidia/tests/data/sass): what it accepts must be well formed, whatever the architecture. The handlers' arithmetic is
// held to the interpreter's by nvidia/tests/e2e/run_sass_tier.sh; this is the part that needs no program to run --
// operand resolution, constant rows, register bounds, the guard -- and it runs on every instruction shape the
// corpora hold (41,000 lines over 11 targets), which no kernel of ours reaches.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
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
  {   // (the cost of the analysis, for VGPU_SASS_REPORT: the whole corpus built again into a scratch program)
    const auto t0 = std::chrono::steady_clock::now();
    sass::tier::Program scratch;
    scratch.ops.resize(instrs.size());
    for (size_t i = 0; i < instrs.size(); ++i) sass::tier::analyze(instrs[i], sm, scratch, scratch.ops[i]);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    if (std::getenv("VGPU_SASS_REPORT"))
      std::fprintf(stderr, "%s: analysis %.0f ns per instruction (%zu instructions, %.0f us)\n", file.c_str(), 1000.0 * us / double(instrs.size()), instrs.size(), us);
  }
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
    const bool ends = op.form == sass::tier::Form::Bra || op.form == sass::tier::Form::Bssy || op.form == sass::tier::Form::Nop ||
                      op.form == sass::tier::Form::Bsync || op.form == sass::tier::Form::Yield || op.form == sass::tier::Form::Bar || op.form == sass::tier::Form::Exit;
    const bool terminal = op.form == sass::tier::Form::Plop3 || op.form == sass::tier::Form::S2r || op.form == sass::tier::Form::Sel ||
                          op.form == sass::tier::Form::Shfl || op.form == sass::tier::Form::Vote || op.form == sass::tier::Form::Hmma ||
                          op.form == sass::tier::Form::Umov || op.form == sass::tier::Form::Uldc || op.form == sass::tier::Form::Usel ||
                          op.form == sass::tier::Form::Cs2r || op.form == sass::tier::Form::Match || op.form == sass::tier::Form::Imma || op.form == sass::tier::Form::Ldsm;   // end a run, and have handlers
    VCHECK(ins.straight != (ends || terminal));
    VCHECK(op.maxreg <= 256);   // 0 is possible: constants into RZ
    const std::string what = std::string(sass::tier::form_name(op.form)) + " " + sass::to_text(ins);
    const bool uniform = op.form >= sass::tier::Form::Umov && op.form <= sass::tier::Form::Usel;   // dst is a UR, not a row
    const bool mem = (op.form >= sass::tier::Form::Ldg && op.form <= sass::tier::Form::Stl) || op.form == sass::tier::Form::Hmma || op.form == sass::tier::Form::Imma || op.form == sass::tier::Form::Ldsm;   // s[] holds raw register numbers there
    // A double operand, and IMAD.WIDE's and IMAD.HI's addend, are pairs of rows.
    for (int i = 0; i < 3 && !mem; ++i) {
      const bool pair = op.form == sass::tier::Form::Dop || op.form == sass::tier::Form::Dsetp || (i == 2 && (op.form == sass::tier::Form::ImadWide ||
                                                                       (op.form == sass::tier::Form::ImadHi && (op.aux & 4))));
      check_src(op.s[i], prog, pair ? 1 : 0, what.c_str());
    }
    // A destination register the handler writes lies inside the file the op claims to need.
    if (op.dst != sass::tier::kNoReg && op.form != sass::tier::Form::Isetp && op.form != sass::tier::Form::Fsetp &&
        op.form != sass::tier::Form::IsetpEx && op.form != sass::tier::Form::Plop3 && op.form != sass::tier::Form::Dsetp && op.form != sass::tier::Form::Vote && !uniform && !ends)
      VCHECK(op.dst < op.maxreg);
    if (ends) VCHECK(op.fn == nullptr);
  }
  // pool[0] is the zero row and every constant row is the same value in all 32 lanes except a double's pair halves.
  VCHECK(!prog.pool.empty());
  for (unsigned l = 0; l < 32; ++l) VCHECK(prog.pool[0].v[l] == 0);
  // (the exception is a shuffle's table of source lanes, one lane per lane: each entry names a lane of the warp)
  std::vector<bool> table(prog.pool.size(), false);
  for (const auto& op : prog.ops)
    if (op.form == sass::tier::Form::Shfl && (op.aux & 1)) {
      VCHECK(op.s[1].kind == sass::tier::SrcKind::Pool && op.s[1].off < prog.pool.size());
      table[op.s[1].off] = true;
      for (unsigned l = 0; l < 32; ++l) VCHECK(prog.pool[op.s[1].off].v[l] < 32);
    }
  for (size_t i = 0; i < prog.pool.size(); ++i)
    if (!table[i])
      for (unsigned l = 1; l < 32; ++l) VCHECK(prog.pool[i].v[l] == prog.pool[i].v[0]);
  // Not vacuous: a corpus of an architecture with this many integer and float instructions gives the tier many.
  VCHECK(accepted > 200);
  const bool report = std::getenv("VGPU_SASS_REPORT") != nullptr;
  if (report) {
    for (const auto& [op, t] : by_op)
      if (t.straight) std::fprintf(stderr, "  %-10s %5zu of %5zu straight-line forms taken\n", op.c_str(), t.accepted, t.straight);
    std::fprintf(stderr, "%s: %zu of %zu instructions have a handler\n", file.c_str(), accepted, instrs.size());
    std::fprintf(stderr, "%s: sizeof(MOp) %zu, sizeof(Instr) %zu; the tier of these %zu instructions: %zu bytes of MOps, %zu pool rows of %zu bytes\n",
                 file.c_str(), sizeof(sass::tier::MOp), sizeof(sass::Instr), instrs.size(), instrs.size() * sizeof(sass::tier::MOp),
                 prog.pool.size(), sizeof(sass::tier::Row));
  }
  // The forms the tier exists for must be taken somewhere in every corpus that has them.
  for (const char* must : {"IMAD", "IADD3", "LOP3", "ISETP", "SHF", "FFMA", "LDG", "STG", "MATCH"})
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


// ---- the structural guard: no form's result rules live in the tier alone ------------------------------------------------------------
//
// Round 9 changed the interpreter's result rules (NaNs, -0 under .SAT, MUFU, packed halves, conversions, the sm_86 tensor-core sum) and the
// tier's handlers, which had copied the old rules, drifted from it. The tier must call what the interpreter calls. The table
// sass::tier::form_impl says, per form, whether its handler (K) calls the SIMD kernel or shared function the interpreter's fast path calls,
// (I) runs the interpreter's own code, or (O) computes it alone, which is for work with no number-format rules only. This test holds that table
// to the list checked in here (so that a new own implementation, or a form that stops sharing, is a deliberate change of this file too) and
// reads tier_ops.inc: a handler of a K form must call a kernel, one of an I form the interpreter, and no handler may contain the scalar
// float arithmetic a rule would be written in (the interpreter's functions for it are called, not copied).
const char* const kExpectedImpl =
    "none:O mov:O iadd3:K iadd3x:O imad:K imad32:O imadwide:O imadhi:O isetpex:O popc:K i2fp:K ldg:O stg:O lds:O sts:O ldl:O stl:O lop3:K lea:O shf:O isetp:K fsetp:K fsel:K ffma:K fadd:K fmul:K dop:K bra:O bssy:O nop:O plop3:O s2r:O cs2r:O sel:O prmt:K imnmx:K fmnmx:K mufu:K f2i:K i2f:K f2fp:I dsetp:K hop2:K unary:K iabs:O ldc:O bsync:O yield:O bar:O exit:O shfl:O vote:O umov:O uldc:O uiadd3:O uimad:O uisetp:O ulea:O ulop3:O ushf:O uunary:O usel:O hmma:K match:O ldsm:O imma:O";

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw vtest::Failure("cannot read " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The text with // comments and string literals removed.
std::string strip_comments(const std::string& t) {
  std::string out;
  for (size_t i = 0; i < t.size();) {
    if (t.compare(i, 2, "//") == 0) {
      while (i < t.size() && t[i] != '\n') ++i;
    } else if (t.compare(i, 2, "/*") == 0) {
      const size_t e = t.find("*/", i + 2);
      i = e == std::string::npos ? t.size() : e + 2;
    } else {
      out += t[i++];
    }
  }
  return out;
}

VTEST(tier_form_table_is_the_checked_in_one) {
  using sass::tier::Form;
  std::string got;
  for (unsigned f = 0; f < static_cast<unsigned>(Form::Count_); ++f) {
    const auto& fi = sass::tier::form_impl(static_cast<Form>(f));
    VCHECK(fi.note != nullptr && std::strlen(fi.note) > 3);
    if (!got.empty()) got += ' ';
    got += std::string(sass::tier::form_name(static_cast<Form>(f))) + ":" +
           (fi.impl == sass::tier::Impl::Kernel ? "K" : fi.impl == sass::tier::Impl::Interp ? "I" : "O");
  }
  if (got != kExpectedImpl)
    throw vtest::Failure(std::string("the tier's form table differs from the checked-in list (tests/unit/test_tier.cpp): a form that computes its "
                                     "result itself (O) must be justified there.\n  table:    ") + got + "\n  expected: " + kExpectedImpl);
}

VTEST(tier_handlers_call_what_the_interpreter_calls) {
  using sass::tier::Form;
  const std::string src = strip_comments(read_file(std::string(VGPU_SOURCE_DIR) + "/src/sass/tier_ops.inc"));
  // The handler of each form, from tier_handler_for's switch.
  std::map<std::string, std::string> handler_of;   // form name -> function name
  for (size_t at = 0; (at = src.find("case F::", at)) != std::string::npos;) {
    const size_t name_end = src.find(':', at + 8);
    const std::string en = src.substr(at + 8, name_end - at - 8);
    const size_t r = src.find("return &h_", name_end);
    const size_t next_case = src.find("case F::", name_end);
    if (r != std::string::npos && (next_case == std::string::npos || r < next_case) && src.compare(name_end, 2, ": ") == 0) {
      const size_t e = src.find(';', r);
      std::string low = en;
      for (char& ch : low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      handler_of[low] = src.substr(r + 8, e - r - 8);
    }
    at = name_end;
  }
  VCHECK(handler_of.size() > 30);
  // The text of each handler.
  const auto body_of = [&](std::string fn) {
    fn = fn.substr(0, fn.find('<'));   // (a template's arguments)
    size_t b = src.find("TIER_HANDLER(" + fn + ")");
    if (b == std::string::npos) b = src.find("void " + fn + "(");
    if (b == std::string::npos) b = src.find("void " + fn + "(");
    if (b == std::string::npos) throw vtest::Failure("no handler " + fn);
    // (a handler ends at the first closing brace in the first column)
    const size_t close = src.find("\n}", b);
    const size_t stop = close == std::string::npos ? src.size() : close + 2;
    return src.substr(b, stop - b);
  };
  static const char* const kScalarFloat[] = {"as_f32(",   "f32_bits(", "ftz(",       "std::fabs",   "std::sin",      "std::cos",       "std::exp2",
                                             "std::log2", "std::tanh", "std::sqrt",  "std::nearbyint", "std::floor", "std::ceil",     "std::trunc",
                                             "std::fma",  "__builtin_fma", "static_cast<float>", "static_cast<double>", "f32_to_bf16(", "f32_to_half(",
                                             "half_to_f32(", "bf16_to_f32(", "f64_to_half(", "float_cmp(", "narrow_core(", "narrow_from_",
                                             "fma_round_odd(", "dop_nan("};
  for (unsigned f = 1; f < static_cast<unsigned>(Form::Count_); ++f) {
    const std::string name = sass::tier::form_name(static_cast<Form>(f));
    const auto& fi = sass::tier::form_impl(static_cast<Form>(f));
    const auto it = handler_of.find(name);
    if (it == handler_of.end()) {   // run by tier_run itself: control flow only
      VCHECK(fi.impl == sass::tier::Impl::Own);
      continue;
    }
    const std::string body = body_of(it->second);
    VCHECK(body.size() > 20);
    for (const char* bad : kScalarFloat)
      if (body.find(bad) != std::string::npos)
        throw vtest::Failure("handler " + it->second + " (form " + name + ") contains `" + bad + "`: a result rule written in the tier; call the "
                             "interpreter's kernel or code instead (see tests/unit/test_tier.cpp)");
    if (fi.impl == sass::tier::Impl::Kernel && body.find("sk->") == std::string::npos && body.find("prmt(") == std::string::npos)
      throw vtest::Failure("handler " + it->second + " is declared a kernel form but calls no SIMD kernel");
    if (fi.impl == sass::tier::Impl::Interp && body.find("tier_float_op(") == std::string::npos)
      throw vtest::Failure("handler " + it->second + " is declared an interpreter form but does not call it");
    // (HMMA: which sum a cubin gets is decided in one place, Runner::hmma_kernel_kind, which exec_mma and this handler both call.)
    if (name == "hmma" && body.find("hmma_kernel_kind(") == std::string::npos)
      throw vtest::Failure("the HMMA handler does not choose its sum with Runner::hmma_kernel_kind, as exec_mma does");
  }
}

VTEST_MAIN
