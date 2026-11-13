// The IEEE square root idiom of host-GPU exact mode (include/vgpu/host_gpu_idiom.hpp), on the real SASS nvcc gives for sm_86: PolyBench GRAMSCHM's
// kernel1 (sqrtf), CORR's std_kernel (a division, then sqrtf, the slow path predicated instead of branched around) and the exhaustive probe's kernel
// (tools/probes/hostgpu_sfu_exhaustive.cu: __fsqrt_rn next to a bare rsqrt.approx and the other SFU forms). The words are from cuobjdump of own builds
// (host_gpu_idiom_sass.inc; nvcc 13.0 and 13.2 give the same words). Each negative case changes one thing of a recognised kernel and must be refused.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vgpu/host_gpu_idiom.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace vgpu::hostgpu;
using namespace vgpu::hostgpu::idiom;

namespace {
struct SassWord { uint64_t pc, lo, hi; };
#include "host_gpu_idiom_sass.inc"

template <size_t N>
std::vector<sass::Instr> decode_all(const SassWord (&w)[N]) {
  std::vector<sass::Instr> v;
  for (const SassWord& x : w) v.push_back(sass::decode(sass::Word{x.lo, x.hi}, x.pc, 86));
  return v;
}
size_t at(const std::vector<sass::Instr>& c, uint64_t pc) {
  for (size_t i = 0; i < c.size(); ++i)
    if (c[i].pc == pc) return i;
  std::fprintf(stderr, "no instruction at 0x%llx\n", static_cast<unsigned long long>(pc));
  std::abort();
}
bool verbose() {
  const char* e = std::getenv("VGPU_IDIOM_VERBOSE");
  return e && *e && *e != '0';
}
// The kernel's verdict with the switches given; says why when VGPU_IDIOM_VERBOSE is set.
bool admitted(const std::vector<sass::Instr>& c, bool div, bool sqrt, std::string* why = nullptr) {
  const KernelVerdict v = idioms_only(c, div, sqrt);
  if (why) *why = v.why;
  if (verbose()) std::fprintf(stderr, "  verdict %d sites %u: %s\n", v.ok, v.sqrt_sites, v.why);
  return v.ok;
}
std::string site_why(const std::vector<sass::Instr>& c, uint64_t mufu_pc) {
  const Code code(c);
  const Site s = sqrt_site(code, at(c, mufu_pc));
  return s.ok ? "" : s.why;
}
sass::Instr nop_at(uint64_t pc) {
  sass::Instr n;
  n.pc = pc;
  n.sm = 86;
  n.op = sass::Op::NOP;
  n.mnemonic = "NOP";
  return n;
}
}  // namespace

VTEST(the_reference_has_the_six_paths_of_sqrt_rn) {
  const auto& ref = reference_paths();
  if (verbose())
    for (const auto& [k, v] : ref) std::fprintf(stderr, "  path [%s]\n    -> %s\n", k.c_str(), v.c_str());
  VCHECK_EQ(ref.size(), size_t{6});   // the fast path; zero, negative, NaN, infinity and the small positive floats of the slow path
  unsigned with_mufu = 0;
  for (const auto& [k, v] : ref) with_mufu += v.find("MUFU.RSQ") != std::string::npos;
  VCHECK_EQ(with_mufu, 2u);           // the fast path and the scaled slow path refine a MUFU; the special values do not use one
}

VTEST(gramschmidt_kernel1_is_the_sqrt_idiom) {
  const auto c = decode_all(kGramschmidtKernel1);
  VCHECK_EQ(site_why(c, 0x3a0), std::string());
  VCHECK(admitted(c, false, true));
  VCHECK(admitted(c, true, true));
  VCHECK(!admitted(c, false, false));   // the switch off: refused as before
  VCHECK(!admitted(c, true, false));    // the division idiom alone does not take it (sp67: "a CALL with no FCHK before it")
  const KernelVerdict v = idioms_only(c, false, true);
  VCHECK_EQ(v.sqrt_sites, 1u);
}

VTEST(corr_std_kernel_needs_both_idioms) {
  const auto c = decode_all(kCorrStdKernel);
  VCHECK_EQ(site_why(c, 0x560), std::string());   // the slow path predicated, not branched: the same paths
  VCHECK(admitted(c, true, true));
  VCHECK(!admitted(c, false, true));    // its division stays outside
  VCHECK(!admitted(c, true, false));
}

VTEST(gramschmidt_kernel2_is_the_division_idiom_only) {
  const auto c = decode_all(kGramschmidtKernel2);
  VCHECK(admitted(c, true, false));
  VCHECK(admitted(c, true, true));
  VCHECK(!admitted(c, false, true));
}

VTEST(the_probe_kernel_has_one_site_and_a_bare_rsqrt) {
  const auto c = decode_all(kProbeRun);
  VCHECK_EQ(site_why(c, 0x540), std::string());                                   // __fsqrt_rn: the reference itself, in its kernel
  VCHECK_EQ(site_why(c, 0x190), std::string("no guard branch after the MUFU.RSQ (a bare rsqrt)"));   // rsqrt.approx.ftz.f32
  VCHECK(!admitted(c, true, true));     // EX2, LG2, SIN, COS, TANH, the bare forms
}

// ---- one change at a time to GRAMSCHM's kernel1 ----
//   0x390 IADD3 R3, R0, -0xd000000, RZ     0x3a0 MUFU.RSQ R7, R0     0x3c0 ISETP.GT.U32.AND P0, PT, R3, 0x727fffff, PT    0x400 @!P0 BRA 0x450
//   0x410 MOV R6, 0x430   0x420 CALL.REL.NOINC 0x4b0   0x430 IMAD.MOV.U32 R5, RZ, RZ, R0   0x440 BRA 0x490
//   0x450 FMUL.FTZ R5, R7, R0   0x460 FMUL.FTZ R4, R7, 0.5   0x470 FFMA R0, -R5, R5, R0   0x480 FFMA R5, R0, R4, R5   0x490 STG.E [R2.64], R5   0x4a0 EXIT
//   0x4b0.. the slow path; 0x570 FFMA R0, R0, 2^64, RZ   0x580 MUFU.RSQ R5, R0 ... 0x5e0 FMUL.FTZ R4, R4, 2^-32 ... 0x620 RET.REL.NODEC R4
namespace {
std::vector<sass::Instr> gk1() { return decode_all(kGramschmidtKernel1); }
bool refused(const std::vector<sass::Instr>& c, const char* what) {
  std::string why;
  const bool ok = admitted(c, true, true, &why);
  if (verbose() || ok) std::fprintf(stderr, "  %s: %s\n", what, ok ? "ADMITTED" : why.c_str());
  return !ok;
}
}  // namespace

VTEST(a_sqrt_without_the_refinement_is_refused) {
  auto c = gk1();
  c[at(c, 0x480)] = nop_at(0x480);                       // the last FFMA gone: R5 is the unrefined y x
  VCHECK(refused(c, "no final FFMA"));
  c = gk1();
  c[at(c, 0x470)] = nop_at(0x470);                       // the residual gone
  VCHECK(refused(c, "no residual"));
  c = gk1();
  c[at(c, 0x460)].src[1].fbits = 0x3e800000u;            // y / 4 instead of y / 2
  VCHECK(refused(c, "half is a quarter"));
  c = gk1();
  c[at(c, 0x470)].src[0].neg = false;                    // x + s s instead of x - s s
  VCHECK(refused(c, "residual sign"));
}

VTEST(wrong_registers_are_refused) {
  auto c = gk1();
  c[at(c, 0x470)].src[2].reg = 2;                        // the residual of another register, not x
  VCHECK(refused(c, "residual of R2"));
  c = gk1();
  c[at(c, 0x450)].src[1].reg = 3;                        // y times the range check's value, not x
  VCHECK(refused(c, "s of R3"));
  c = gk1();
  c[at(c, 0x3a0)].src[0].reg = 3;                        // the MUFU of another register than the range check's
  VCHECK(refused(c, "MUFU of R3"));
  c = gk1();
  c[at(c, 0x430)].src[2].reg = 4;                        // the slow side hands over another register than the subroutine's result
  VCHECK(refused(c, "slow result R4"));
  c = gk1();
  c[at(c, 0x490)].src[1].reg = 7;                        // the kernel stores the raw MUFU (R7) instead of the root
  VCHECK(refused(c, "stores R7"));
  c = gk1();
  c[at(c, 0x490)].src[1].reg = 4;                        // or y / 2 (R4: the fast path's, and the slow path's intermediate)
  VCHECK(refused(c, "stores R4"));
}

VTEST(a_changed_range_check_or_scale_is_refused) {
  auto c = gk1();
  c[at(c, 0x3c0)].src[1].imm = 0x7f7fffff;               // subnormals sent to the fast path
  VCHECK(refused(c, "range check bound"));
  c = gk1();
  c[at(c, 0x390)].src[1].imm = -0x0c000000;              // another lower bound
  VCHECK(refused(c, "range check base"));
  c = gk1();
  c[at(c, 0x400)].guard_not = false;                     // the branch the other way: the fast path for the extremes
  VCHECK(refused(c, "guard inverted"));
  c = gk1();
  c[at(c, 0x570)].src[1].fbits = 0x5f000000u;            // the slow path scales by 2^63
  VCHECK(refused(c, "slow scale"));
  c = gk1();
  c[at(c, 0x5e0)].src[1].fbits = 0x2f000000u;            // and back by 2^-33
  VCHECK(refused(c, "slow unscale"));
  c = gk1();
  c[at(c, 0x4f0)].src[2].imm = 0x7fc00000;               // another NaN for a negative argument
  VCHECK(refused(c, "negative NaN"));
}

VTEST(a_slow_path_with_memory_or_a_barrier_is_refused) {
  auto c = gk1();
  sass::Instr st = c[at(c, 0x490)];                      // STG.E [R2.64], R5 placed in the slow path
  st.pc = 0x5f0;
  c[at(c, 0x5f0)] = st;
  VCHECK(refused(c, "STG in slow path"));
  c = gk1();
  sass::Instr bar = nop_at(0x5f0);
  bar.mnemonic = "BAR";
  bar.op = sass::Op::Unknown;
  bar.mods = {"SYNC", "DEFER_BLOCKING"};
  c[at(c, 0x5f0)] = bar;
  VCHECK(refused(c, "BAR in slow path"));
  c = gk1();
  sass::Instr ex = c[at(c, 0x4a0)];                      // an EXIT in the fast path
  ex.pc = 0x460;
  c[at(c, 0x460)] = ex;
  VCHECK(refused(c, "EXIT in fast path"));
}

VTEST(control_flow_into_or_out_of_the_idiom_is_refused) {
  auto c = gk1();
  c[at(c, 0x0c0)].src[0].imm = 0x450;                    // a branch from the loop straight to the fast path (y of no MUFU)
  VCHECK(refused(c, "branch into the fast path"));
  c = gk1();
  c[at(c, 0x0c0)].src[0].imm = 0x500;                    // a branch into the middle of the slow path
  VCHECK(refused(c, "branch into the slow path"));
  c = gk1();
  c[at(c, 0x410)].src[0].imm = 0x440;                    // the return address is not the CALL's next instruction
  VCHECK(refused(c, "return address"));
  c = gk1();
  c[at(c, 0x4a0)].guard = 0;                             // @P0 EXIT: the code before the slow path may fall into it
  VCHECK(refused(c, "fall into the slow path"));
  c = gk1();
  c[at(c, 0x3a0)].guard = 1;                             // a MUFU under a predicate
  VCHECK(refused(c, "predicated MUFU"));
}

VTEST(the_switches_keep_the_old_verdicts) {
  // With the square root off, every kernel gets the division recogniser's verdict as before (host_gpu.hpp division_idiom_only on the same summary).
  for (const auto& c : {decode_all(kGramschmidtKernel1), decode_all(kGramschmidtKernel2), decode_all(kCorrStdKernel), decode_all(kProbeRun)}) {
    std::vector<char> none(c.size(), 0);
    VCHECK_EQ(admitted(c, true, false), division_idiom_only(division_summary(c, none)));
  }
  // and with both off nothing outside the exact set passes
  VCHECK(!admitted(decode_all(kGramschmidtKernel2), false, false));
}

VTEST(may_be_read_follows_branches_and_calls) {
  const auto c = gk1();
  const Code code(c);
  VCHECK(!may_be_read(code, at(c, 0x490), false, 7));    // STG R5; EXIT: R7 is dead at the join
  VCHECK(may_be_read(code, at(c, 0x490), false, 5));     // R5 is stored
  VCHECK(may_be_read(code, at(c, 0x410), false, 0));     // into the subroutine: it reads its argument R0
  VCHECK(!may_be_read(code, at(c, 0x410), false, 8));    // R8 is read nowhere after
  VCHECK(may_be_read(code, at(c, 0x0c0), true, 0));      // @!P0 BRA reads P0
}


// ---- the liveness check on small hand-made sequences (the forms the corpus scan met: predicated slow paths, IMAD.WIDE, 32-bit shared addresses) ----
namespace {
sass::Operand R(unsigned n) {
  sass::Operand o;
  o.kind = sass::Kind::Reg;
  o.reg = n;
  return o;
}
sass::Operand P(unsigned n) {
  sass::Operand o;
  o.kind = sass::Kind::Pred;
  o.reg = n;
  return o;
}
sass::Operand L(uint64_t pc) {
  sass::Operand o;
  o.kind = sass::Kind::Label;
  o.imm = static_cast<int64_t>(pc);
  return o;
}
sass::Operand M(unsigned base, const char* text) {
  sass::Operand o;
  o.kind = sass::Kind::Mem;
  o.reg = base;
  o.text = text;
  return o;
}
// one instruction: mnemonic, modifiers, written operands, read operands, guard (7 = none) and its negation
sass::Instr I(const char* m, std::vector<std::string> mods, std::vector<sass::Operand> dst, std::vector<sass::Operand> src, unsigned guard = sass::kPT,
              bool guard_not = false) {
  sass::Instr i;
  i.mnemonic = m;
  i.mods = std::move(mods);
  i.dst = std::move(dst);
  i.src = std::move(src);
  i.guard = static_cast<uint8_t>(guard);
  i.guard_not = guard_not;
  return i;
}
std::vector<sass::Instr> placed(std::vector<sass::Instr> v) {
  for (size_t i = 0; i < v.size(); ++i) v[i].pc = 16 * i;
  return v;
}
bool read_after(const std::vector<sass::Instr>& v, unsigned reg) {
  const Code c(v);
  return may_be_read(c, 0, false, reg);
}
}  // namespace

VTEST(a_write_under_a_predicate_covers_reads_under_it) {
  // @P2 FMUL R28, R5, R4 ; @P2 FADD R25, -R28, -RZ ; EXIT        (adam's shared slow path): the old R28 is not read
  VCHECK(!read_after(placed({I("FMUL", {"FTZ"}, {R(28)}, {R(5), R(4)}, 2), I("FADD", {"FTZ"}, {R(25)}, {R(28), R(255)}, 2), I("EXIT", {}, {}, {})}), 28));
  // under the other polarity it is
  VCHECK(read_after(placed({I("FMUL", {"FTZ"}, {R(28)}, {R(5), R(4)}, 2), I("FADD", {"FTZ"}, {R(25)}, {R(28), R(255)}, 2, true), I("EXIT", {}, {}, {})}), 28));
  // and after P2 is written again it is
  VCHECK(read_after(placed({I("FMUL", {"FTZ"}, {R(28)}, {R(5), R(4)}, 2), I("ISETP", {"GE", "AND"}, {P(2)}, {R(1), R(255)}),
                            I("FADD", {"FTZ"}, {R(25)}, {R(28), R(255)}, 2), I("EXIT", {}, {}, {})}),
                    28));
  // writes under P and under !P are a write: @P2 MOV R28, RZ ; @!P2 MOV R28, -0x40 ; @!P3 IADD3 R28, R28, 0x40, RZ   (a division slow path)
  VCHECK(!read_after(placed({I("MOV", {}, {R(28)}, {R(255)}, 2), I("IMAD", {"MOV", "U32"}, {R(28)}, {R(255), R(255), R(9)}, 2, true),
                             I("IADD3", {}, {R(28)}, {R(28), R(255), R(255)}, 3, true), I("EXIT", {}, {}, {})}),
                     28));
  // a branch under P after a write under P: the taken edge has the new value; the fall-through has the old one
  VCHECK(!read_after(placed({I("MOV", {}, {R(5)}, {R(255)}, 0, true), I("BRA", {}, {}, {L(48)}, 0, true), I("EXIT", {}, {}, {}), I("IADD3", {}, {R(6)}, {R(5), R(255), R(255)}),
                             I("EXIT", {}, {}, {})}),
                     5));
  VCHECK(read_after(placed({I("MOV", {}, {R(5)}, {R(255)}, 0, true), I("BRA", {}, {}, {L(48)}, 0, true), I("IADD3", {}, {R(6)}, {R(5), R(255), R(255)}),
                            I("EXIT", {}, {}, {})}),
                    5));
}

VTEST(pairs_and_addresses_are_counted_as_the_code_uses_them) {
  // IMAD.WIDE.U32 R22, R20, R27, RZ writes R22 and R23; IADD3.X then reads the new R23
  VCHECK(!read_after(placed({I("IMAD", {"WIDE", "U32"}, {R(22)}, {R(20), R(27), R(255)}), I("IADD3", {"X"}, {R(14)}, {R(23), R(255), R(255)}), I("EXIT", {}, {}, {})}), 23));
  // its 64-bit addend is a pair: IMAD.WIDE R2, R4, R3, R6 reads R7
  VCHECK(read_after(placed({I("IMAD", {"WIDE"}, {R(2)}, {R(4), R(3), R(6)}), I("EXIT", {}, {}, {})}), 7));
  // a 32-bit shared address reads one register, a 64-bit global one two
  VCHECK(!read_after(placed({I("LDS", {}, {R(13)}, {M(12, "[R12]")}), I("EXIT", {}, {}, {})}), 13));
  VCHECK(read_after(placed({I("LDG", {"E"}, {R(9)}, {M(2, "[R2.64]")}), I("EXIT", {}, {}, {})}), 3));
  // the data of a 128-bit store is four registers; a double two; F2F.F64.F32 reads one
  VCHECK(read_after(placed({I("STG", {"E", "128"}, {}, {M(2, "[R2.64]"), R(8)}), I("EXIT", {}, {}, {})}), 11));
  VCHECK(read_after(placed({I("DADD", {}, {R(2)}, {R(8), R(10)}), I("EXIT", {}, {}, {})}), 9));
  VCHECK(!read_after(placed({I("F2F", {"F64", "F32"}, {R(20)}, {R(20)}), I("EXIT", {}, {}, {})}), 21));
  VCHECK(!read_after(placed({I("SHF", {"L", "U64", "HI"}, {R(33)}, {R(33), R(255), R(23)}), I("EXIT", {}, {}, {})}), 34));
  // a RET reads its return address pair; a computed branch counts as a read of everything
  VCHECK(read_after(placed({I("BRX", {}, {}, {R(4)}), I("EXIT", {}, {}, {})}), 40));
}

VTEST(each_structural_check_on_its_own) {
  // (cases where the other checks would pass: the mutants of docs/perf-design/data/sp68/mutants.py showed these checks unreached without them)
  // the code before the slow path falls into it: the probe's division RET at 0xeb0, just before the square root's slow path at 0xec0, made conditional
  auto p = decode_all(kProbeRun);
  p[at(p, 0xeb0)].guard = 0;
  VCHECK_EQ(site_why(p, 0x540), std::string("code before the slow path falls into it"));
  // a store inside the idiom, of a value that is not the MUFU's: CORR's BSSY B0 at 0x570 (between the MUFU and the guard branch) made STG [R2.64], R9
  auto c = decode_all(kCorrStdKernel);
  sass::Instr st = c[at(c, 0x640)];   // STG.E [R2.64], R5
  st.pc = 0x570;
  st.src[1].reg = 9;
  c[at(c, 0x570)] = st;
  VCHECK_EQ(site_why(c, 0x560), std::string("memory, a barrier, another thread or control flow inside the idiom"));
  // a branch from outside to the slow path's entry (only another site's CALL may go there)
  auto g = gk1();
  g[at(g, 0x0c0)].src[0].imm = 0x4b0;
  VCHECK(refused(g, "branch to the slow path's entry"));
  // the division's CALL (0x520) sent into the square root's slow path (0x690): the division idiom would take it
  c = decode_all(kCorrStdKernel);
  c[at(c, 0x520)].src[0].imm = 0x690;
  VCHECK(!admitted(c, true, true));
}

VTEST_MAIN
