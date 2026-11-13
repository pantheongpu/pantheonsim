// Host-GPU mode, the logic that needs no card (docs/perf-design/09-host-gpu-spike.md).
//
// With `VGPU_HOST_GPU=exact` a launch may run on the real GPU of the machine instead of in the simulator, when its result is
// the same bits either way. Three decisions are made here, with no CUDA and no simulator state, so they are unit tested on any
// machine (tests/unit/test_host_gpu_logic.cpp):
//   * exact_opcode_allowed: may a kernel that holds this SASS instruction run on a card with the simulator's bits?
//   * Residency: which 64 KiB chunks of an allocation the simulator holds, the card holds, or both hold (the design for a
//     persistent card copy; the first prototype moves everything on every launch and does not need it, see the doc).
//   * should_offload / SimTimeTable: does the card path (launch, copy in, copy out) beat the simulator's estimate?
// The part that talks to the driver is src/runtime/host_gpu.cpp.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <utility>
#include <string>
#include <vector>

#include "vgpu/exec/block_progress.hpp"

namespace vgpu::hostgpu {

enum class Mode { Off, Exact, Turbo };

// "exact" and "turbo" turn the mode on; anything else, empty or null is off. (`approx` is not a mode: a typo must not turn one on.)
// Turbo (docs/perf-design/12-host-gpu-turbo.md) also sends kernels to the card whose results are the card's own and not the simulator's bits.
inline Mode mode_from_text(const char* text) {
  if (!text) return Mode::Off;
  const std::string t(text);
  return t == "exact" ? Mode::Exact : t == "turbo" ? Mode::Turbo : Mode::Off;
}

// A mode value that is none of off, exact and turbo (a typo, or `approx`, which is not a mode) must be said, never read as off in silence.
inline bool mode_text_is_known(const char* text) {
  if (!text || !*text) return true;
  const std::string t(text);
  return t == "off" || t == "exact" || t == "turbo" || t == "0";
}

// The switches of diagnostics that only the simulator can give: with any of them on the mode refuses to start (08 section 3), never quietly drops them.
inline constexpr const char* kIncompatibleVars[] = {
    "VGPU_COUNTERS",   "VGPU_COUNTERS_FILE", "VGPU_SASS_TRACE", "VGPU_KERNEL_DIGEST", "VGPU_KERNEL_DIGEST_DUMP", "VGPU_HOSTGPU_CAPTURE", "VGPU_TRACE",
    "VGPU_MAX_STEPS",  "VGPU_FAIL_ALLOC",    "VGPU_FAIL_LAUNCH", "VGPU_FAIL_MEMCPY", "VGPU_RACE", "VGPU_STRICT", "VGPU_SCHEDULER", "VGPU_LOCKSTEP",
};

// Whether the value of one of kIncompatibleVars switches its diagnostic on. Unset, empty and 0 are off; the scheduler's default name is not a diagnostic
// (a config file that says `scheduler: deterministic` writes it).
inline bool diagnostic_switched_on(const std::string& name, const char* value) {
  if (!value || !*value) return false;
  const std::string v(value);
  if (v == "0") return false;
  if (name == "VGPU_SCHEDULER") return v != "deterministic";
  return true;
}

// ---- eligibility ------------------------------------------------------------------------------------------------------------

// The instructions whose result is defined bit for bit by the architecture and that the simulator computes as the card does:
// integer and IEEE float arithmetic and conversions, predicates, moves, loads and stores, barriers and plain control flow.
// Measured in the spike (09 doc, M2): every kernel of the Rodinia, PolyBench and HeCBench launches sampled whose code stayed
// inside this set agreed with the card to the bit, and every kernel that differed held one instruction outside it (MUFU, a
// compare-and-swap, a floating-point atomic). A mnemonic that is not named here is refused: an unknown instruction is not
// known to be exact.
inline bool exact_mnemonic_allowed(const std::string& m) {
  static const char* const kAllowed[] = {
      // integer
      "IMAD", "IADD3", "IADD", "LOP3", "LOP", "LEA", "SHF", "SHL", "SHR", "ISETP", "IMNMX", "IABS", "POPC", "FLO", "BREV", "PRMT",
      "SEL", "MOV", "R2P", "P2R", "PLOP3", "UPLOP3", "PSETP", "ISET", "FSET", "DSET", "I2I", "I2IP", "SGXT", "BMSK", "VIADD", "VIMNMX",
      // IEEE float and double (no SFU)
      "FADD", "FMUL", "FFMA", "FSETP", "FSEL", "FMNMX", "DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "I2F", "I2FP", "F2I", "F2F", "F2FP",
      "FRND", "FCHK",
      // memory (plain loads and stores; integer reductions below)
      "LDG", "STG", "LDS", "STS", "LDL", "STL", "LD", "ST", "LDC", "ULDC",
      // special registers, uniform datapath
      "S2R", "S2UR", "CS2R", "UMOV", "UIADD3", "UIMAD", "ULOP3", "USHF", "ULEA", "UISETP", "USEL", "UPRMT", "UFLO", "UPOPC", "UP2UR",
      "R2UR", "VOTEU",
      // control, barriers, warp
      "BRA", "BRX", "BSSY", "BSYNC", "BREAK", "BMOV", "EXIT", "NOP", "YIELD", "WARPSYNC", "BAR", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL",
      "SHFL", "VOTE", "MATCH",
      // integer reductions with no returned value (order of the adds does not matter)
      "RED",
  };
  for (const char* a : kAllowed)
    if (m == a) return true;
  return false;
}

// Modifiers that turn an allowed mnemonic into an order- or implementation-dependent one: a floating-point reduction adds in
// whatever order the blocks arrive, and a reduction on a float is not the integer one.
// The decoder prints a float type and its rounding as ONE modifier ("F32.FTZ.RN", "F16x2.RN", "F32x2.FTZ.RN"), so the dot-separated parts are
// looked at, and a type by how it starts. (Until turbo mode was added this compared whole modifiers with "F32", "F64" and "F16", which the decoder
// never prints alone: a float reduction passed as an integer one on sm_75 to sm_89. On sm_90 and later the mnemonic is REDG, which is not in the set.)
inline bool exact_mods_allowed(const std::string& mnemonic, const std::vector<std::string>& mods) {
  if (mnemonic == "RED") {
    for (const std::string& m : mods)
      for (size_t at = 0; at <= m.size();) {
        const size_t dot = std::min(m.find('.', at), m.size());
        const std::string p = m.substr(at, dot - at);
        at = dot + 1;
        if (p.rfind("F16", 0) == 0 || p.rfind("F32", 0) == 0 || p.rfind("F64", 0) == 0 || p.rfind("BF16", 0) == 0 || p == "FTZ" || p == "CAS" || p == "EXCH" ||
            p == "INC" || p == "DEC" || p == "SPIN" || p == "CAST" || p == "POPC")
          return false;
      }
  }
  return true;
}

inline bool exact_opcode_allowed(const std::string& mnemonic, const std::vector<std::string>& mods) {
  return exact_mnemonic_allowed(mnemonic) && exact_mods_allowed(mnemonic, mods);
}

// ---- the IEEE division idiom (MUFU.RCP + FCHK + a call to the slow path) --------------------------------------------------------------
// MUFU.RCP alone is NOT in the exact set: the card's reciprocal differs from the simulator's correctly rounded one by one ulp on about 13% of the mantissas
// (tools/probes/mufu_rcp_probe.cu, round 5), and 09 doc section 2 saw 1 to 171 ulp results downstream of an SFU instruction. But ptxas builds the IEEE
// quotient a / b (div.rn.f32, also `/` in CUDA C++) from MUFU.RCP, a few FFMAs that refine it, an FCHK that sends the rare operands to a slow path
// (CALL.REL.NOINC to a subroutine that also uses MUFU.RCP, ending in RET), and the result is the correctly rounded quotient whatever the last bit of the
// approximation: the PTX ISA requires div.rn.f32 to be IEEE-compliant, so the card and the simulator both give the one correctly rounded answer.
// Tested here without a card: the refinement over every mantissa of the divisor with the simulator's reciprocal and with it one ulp off either way
// (tests/unit/test_host_gpu_ref.cpp); on a card, by tools/probes/hostgpu_sfu_exhaustive.cu div: 12 dividends against all 2^32 divisors, 0 different on an A10G
// (docs/perf-design/rounds/11-rounds-sp36-sp41.md "Round sp44", B; other cards were not run). The recognizer is still behind VGPU_HOST_GPU_DIV=1 and the
// default exact set is unchanged.
//
// What the recognizer accepts (division_idiom_only): every MUFU is a plain RCP; each one outside a subroutine has an FCHK of the same divisor register within
// kIdiomMufuToFchk instructions after it; every CALL goes to a subroutine (from its target to the first RET at or after it) and has an FCHK within
// kIdiomFchkToCall instructions before it; a subroutine holds arithmetic only (no memory, barrier or warp operation, no call); every RET is in a subroutine.
// The slow path makes the NaN of 0/0 and inf/inf with `MUFU.RSQ R, -QNAN` (the reciprocal square root of an immediate NaN: the canonical NaN 0x7fffffff, which
// the simulator gives and the card run checks with dividends 0, inf and NaN): that one form (NanGen) is accepted inside a subroutine and nowhere else.
// A kernel with a bare rcp.approx (no FCHK), a device function that is not the slow path, or any other MUFU fails the check and stays refused.
struct CodeInsn {
  enum class Kind : uint8_t { Other, MufuRcp, MufuOther, Fchk, Call, Ret, Memory, NanGen };
  Kind kind = Kind::Other;
  uint64_t pc = 0;
  int64_t target = -1;    // CALL: the address of the subroutine
  int src0 = -1;          // MUFU: the register read; FCHK: the dividend's
  int src1 = -1;          // FCHK: the divisor's register
};
inline constexpr unsigned kIdiomMufuToFchk = 32;   // (the scheduler interleaves other work: 9 instructions apart in a kernel that also has an expf)
inline constexpr unsigned kIdiomFchkToCall = 12;

inline bool division_idiom_only(const std::vector<CodeInsn>& code, const char** why = nullptr) {
  const char* sink = "";
  if (!why) why = &sink;
  using K = CodeInsn::Kind;
  struct Region { uint64_t lo, hi; };
  std::vector<Region> regions;
  bool any = false;
  for (size_t i = 0; i < code.size(); ++i) {
    if (code[i].kind != K::Call) continue;
    any = true;
    // the subroutine: from the target to the first RET at or after it
    size_t t = code.size();
    for (size_t j = 0; j < code.size(); ++j)
      if (static_cast<int64_t>(code[j].pc) == code[i].target) { t = j; break; }
    if (t == code.size() || t <= i) { *why = "a call to an address that is not later in the code"; return false; }   // no such address, or a call backwards (a loop of calls is not the idiom)
    size_t r = t;
    while (r < code.size() && code[r].kind != K::Ret) {
      if (code[r].kind == K::Call || code[r].kind == K::Memory) { *why = "the subroutine calls or touches memory or other threads"; return false; }
      ++r;
    }
    if (r == code.size()) { *why = "a subroutine with no RET"; return false; }
    bool fchk_before = false;
    for (size_t j = i > kIdiomFchkToCall ? i - kIdiomFchkToCall : 0; j < i; ++j) fchk_before = fchk_before || code[j].kind == K::Fchk;
    if (!fchk_before) { *why = "a CALL with no FCHK before it"; return false; }
    regions.push_back({code[t].pc, code[r].pc});
  }
  const auto in_region = [&](uint64_t pc) {
    for (const Region& g : regions)
      if (pc >= g.lo && pc <= g.hi) return true;
    return false;
  };
  for (size_t i = 0; i < code.size(); ++i) {
    switch (code[i].kind) {
      case K::MufuOther: *why = "an SFU instruction other than RCP"; return false;
      case K::NanGen:
        if (!in_region(code[i].pc)) { *why = "a NaN generator outside a division subroutine"; return false; }
        break;
      case K::Ret:
        if (!in_region(code[i].pc)) { *why = "a RET outside a division subroutine"; return false; }
        break;
      case K::MufuRcp: {
        if (in_region(code[i].pc)) break;
        bool checked = false;
        for (size_t j = i + 1; j < code.size() && j <= i + kIdiomMufuToFchk && !checked; ++j)
          checked = code[j].kind == K::Fchk && code[j].src1 >= 0 && code[j].src1 == code[i].src0;
        if (!checked) { *why = "a MUFU.RCP with no FCHK of its divisor right after"; return false; }
        break;
      }
      default: break;
    }
  }
  (void)any;
  return true;
}

// The kind of an instruction for the check above, from its mnemonic and modifiers. Anything that touches memory, a barrier or other threads is Memory
// (not allowed inside the slow path); an instruction that is not in the exact set at all is Memory too, so a subroutine holding one fails.
inline CodeInsn::Kind classify_for_idiom(const std::string& mnemonic, const std::vector<std::string>& mods, bool source_is_nan_constant = false) {
  using K = CodeInsn::Kind;
  if (mnemonic == "MUFU") {
    bool rcp = false, rsq = false, other = false;
    for (const std::string& m : mods) (m == "RCP" ? rcp : m == "RSQ" ? rsq : other) = true;
    if (rsq && !rcp && !other && source_is_nan_constant) return K::NanGen;
    return rcp && !rsq && !other ? K::MufuRcp : K::MufuOther;
  }
  if (mnemonic == "FCHK") return K::Fchk;
  if (mnemonic == "CALL") return K::Call;
  if (mnemonic == "RET") return K::Ret;
  static const char* const kTouchesOthers[] = {"LDG", "STG", "LDS", "STS", "LDL", "STL", "LD", "ST", "BAR", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL", "SHFL",
                                               "VOTE", "MATCH", "RED", "WARPSYNC", "VOTEU"};
  for (const char* t : kTouchesOthers)
    if (mnemonic == t) return K::Memory;
  return exact_mnemonic_allowed(mnemonic) ? K::Other : K::Memory;
}

// ---- turbo mode: what a kernel's code is made of (docs/perf-design/12-host-gpu-turbo.md) ------------------------------------------------
// VGPU_HOST_GPU=turbo also sends kernels to the card whose result is the card's and not the simulator's bits. Every instruction of a kernel is put in
// one class, once, and the mode admits a kernel by the classes it holds. It stays a whitelist: a mnemonic that is not named below is Never.
enum class InsnClass : uint8_t {
  Exact,       // today's exact set
  Mufu,        // every MUFU form (the card's approximation is 1 to 2 ulp from the simulator's, sp44)
  Half,        // packed half arithmetic: HADD2, HMUL2, HFMA2, HSETP2, HSET2, HMNMX2
  FloatRed,    // a float reduction with no returned value: the sum depends on the order the threads arrive in
  Call,        // CALL and RET (the runtime also checks that every call stays inside the kernel's own code section)
  Extra,       // plain compute and memory instructions the exact set never measured: they read and write registers, shared memory and the same global
               // addresses as LDG and STG, so the card needs nothing that only the simulator holds
  Tensor,      // HMMA, IMMA, DMMA
  AtomInt,     // integer atomics that return the old value (ADD, MIN, MAX, AND, OR, XOR, INC, DEC): memory ends the same, the returned values depend on the order
  AtomFloat,   // float atomics that return a value: admitted unless VGPU_HOST_GPU_ATOMICS=0 (an opt-in until Round sp66)
  AtomSwap,    // CAS, CAST.SPIN, EXCH: as AtomFloat; the card does not agree with itself from run to run on programs built on these (09 doc, M2)
  Never,       // textures, surfaces, anything not named
};
inline constexpr unsigned kInsnClasses = 11;
inline const char* insn_class_name(InsnClass c) {
  static const char* const n[kInsnClasses] = {"exact", "mufu", "half", "float-red", "call", "extra", "tensor", "atomic-int", "atomic-float", "atomic-swap", "never"};
  return n[static_cast<unsigned>(c)];
}

// The decoder keeps some modifiers joined as nvdisasm prints them ("F32.FTZ.RN", "F16x2.RN"): these look at the dot-separated parts.
template <class Pred>
inline bool any_mod_part(const std::vector<std::string>& mods, Pred pred) {
  for (const std::string& m : mods)
    for (size_t at = 0; at <= m.size();) {
      const size_t dot = std::min(m.find('.', at), m.size());
      if (dot > at && pred(m.substr(at, dot - at))) return true;
      at = dot + 1;
    }
  return false;
}
inline bool mods_have(const std::vector<std::string>& mods, const char* part) {
  return any_mod_part(mods, [&](const std::string& p) { return p == part; });
}
// A float operand type: F16x2, F32, F32x4, F64, BF16x2 ...
inline bool mods_float_typed(const std::vector<std::string>& mods) {
  return any_mod_part(mods, [](const std::string& p) { return p.rfind("F16", 0) == 0 || p.rfind("F32", 0) == 0 || p.rfind("F64", 0) == 0 || p.rfind("BF16", 0) == 0; });
}
// A field the decoder could not name prints as "(9)": an instruction that carries one is not known.
inline bool mods_undecoded(const std::vector<std::string>& mods) {
  return any_mod_part(mods, [](const std::string& p) { return p[0] == '('; });
}

inline InsnClass turbo_class(const std::string& mnemonic, const std::vector<std::string>& mods) {
  using C = InsnClass;
  const auto is_one_of = [&](std::initializer_list<const char*> names) {
    for (const char* n : names)
      if (mnemonic == n) return true;
    return false;
  };
  if (mnemonic.empty() || mods_undecoded(mods)) return C::Never;
  // Reductions and atomics first, by what they do.
  if (is_one_of({"RED", "REDG"})) {
    if (mods_have(mods, "CAS") || mods_have(mods, "CAST") || mods_have(mods, "SPIN") || mods_have(mods, "EXCH")) return C::AtomSwap;
    if (mods_float_typed(mods)) return C::FloatRed;
    if (mods_have(mods, "INC") || mods_have(mods, "DEC") || mods_have(mods, "POPC")) return C::AtomInt;
    return mnemonic == "RED" ? C::Exact : C::Extra;   // (REDG is sm_90's spelling: the exact set never named it)
  }
  if (is_one_of({"ATOM", "ATOMG", "ATOMS"})) {
    if (mods_have(mods, "ARRIVE")) return C::Never;   // sm_80's mbarrier.arrive
    if (mods_have(mods, "CAS") || mods_have(mods, "CAST") || mods_have(mods, "SPIN") || mods_have(mods, "EXCH")) return C::AtomSwap;
    return mods_float_typed(mods) ? C::AtomFloat : C::AtomInt;
  }
  if (exact_opcode_allowed(mnemonic, mods)) return C::Exact;
  if (mnemonic == "MUFU") return C::Mufu;
  if (is_one_of({"HADD2", "HMUL2", "HFMA2", "HSETP2", "HSET2", "HMNMX2"})) return C::Half;
  // (LEPC: the return address of an absolute call, CUDA 13's vprintf call; the card's own program counter, used by the call and its RET alone. Round sp66.)
  if (is_one_of({"CALL", "RET", "LEPC"})) return C::Call;
  if (is_one_of({"HMMA", "IMMA", "DMMA"})) return C::Tensor;
  // (Round sp66: B2R reads the result of a block's BAR.RED, the same function on both sides; NANOSLEEP is a hint that a spin-wait backs off, as cub's
  // decoupled look-back does: neither touches anything the card lacks)
  if (is_one_of({"IDP", "VABSDIFF", "VABSDIFF4", "VIADDMNMX", "VIMNMX3", "FMNMX3", "F2IP", "FSWZADD", "REDUX", "LDGSTS", "LDGDEPBAR", "LDSM", "MOVM", "FFMA2",
                 "FMUL2", "FADD2", "B2R", "NANOSLEEP"}))
    return C::Extra;
  return C::Never;
}

// A call stays inside the kernel when it is relative, to a label, and the label is an address of the kernel's own code section [lo, hi]. A call through a
// register or to an absolute address (another section, a function pointer) leaves the code this scan has seen.
inline bool call_stays_inside(bool relative, bool to_label, int64_t target, uint64_t lo, uint64_t hi) {
  return relative && to_label && target >= 0 && static_cast<uint64_t>(target) >= lo && static_cast<uint64_t>(target) <= hi;
}

// ---- what a kernel reaches besides its code (Round sp66) ----------------------------------------------------------------------------
// A function the module calls but does not define (Cubin::externs): the driver links it when it loads the module on the card, the simulator runs a
// builtin in its place. What each kind needs on the card, and so whether a kernel that reaches one can go there:
//   * vprintf: the card writes the arguments to its printf buffer and the driver prints them when the host synchronises with the kernel, which the
//     card path does at once (cuCtxSynchronize), after flushing the process's stdout; the text is the card's formatting of the card's values. Turbo
//     mode admits it (VGPU_HOST_GPU_TURBO_OFF=printf keeps it out); exact mode does not (its promise is about memory, not about a second printer);
//   * __assertfail: a failed assert on the card ends the kernel with a sticky error, the context is lost, and data only the card holds with it;
//   * malloc and free: the card's heap is the card's own, outside the simulator's window: a pointer from it, kept in memory, means nothing to the
//     simulator, and the simulator's heap is not on the card;
//   * the device runtime (device-side launches, cudaDeviceSynchronize in a kernel): the child grids would be the card's, with the card's addresses;
//   * anything else: not known.
enum class ExternKind : uint8_t { Printf, Assert, Heap, DeviceLaunch, Other };
inline ExternKind extern_kind(const std::string& name, bool device_runtime) {
  if (device_runtime) return ExternKind::DeviceLaunch;
  if (name == "vprintf") return ExternKind::Printf;
  if (name == "__assertfail" || name == "__assert_fail") return ExternKind::Assert;
  if (name == "malloc" || name == "free") return ExternKind::Heap;
  return ExternKind::Other;
}
inline const char* extern_kind_name(ExternKind k) {
  static const char* const n[] = {"vprintf", "__assertfail", "malloc/free", "device-side launch", "an unknown external function"};
  return n[static_cast<unsigned>(k)];
}
inline bool extern_admitted(ExternKind k, Mode mode, bool printf_off) { return k == ExternKind::Printf && mode == Mode::Turbo && !printf_off; }

// The texture and surface instructions: they read the simulator's own tables (the handle in the instruction or a register is an index into
// them), so a kernel that holds one cannot run on the card. Neither the exact set nor any turbo class names them; a kernel without one does
// not read a texture object, whatever objects exist (until Round sp66 one live texture object kept every launch of the device off the card).
inline bool texture_mnemonic(const std::string& m) {
  static const char* const kTex[] = {"TEX", "TEXS", "TLD", "TLDS", "TLD4", "TLD4S", "TXD", "TXQ", "TMML", "SULD", "SUST", "SUATOM", "SURED", "SUQUERY"};
  for (const char* t : kTex)
    if (m == t) return true;
  return false;
}

// An absolute call to the first instruction of a code section of the module (Round sp66): a device function the compiler kept out of line, in a
// section of its own (a -rdc build). The driver links the call to its own copy of the same code, so the call is followed: the callee's section is
// classed with the kernel's. `base`, `count`: the section the target lies in (its first address, its instructions; 0 is no code).
inline bool call_to_entry(bool relative, bool to_label, int64_t target, uint64_t base, uint64_t count) {
  return !relative && to_label && target >= 0 && count > 0 && static_cast<uint64_t>(target) == base;
}

// The classes a kernel holds, as bits, and what the mode's switches make of them.
using ClassMask = uint32_t;
inline constexpr ClassMask class_bit(InsnClass c) { return ClassMask{1} << static_cast<unsigned>(c); }

struct TurboSwitches {
  // Float atomics with a returned value, CAS, CAST.SPIN and EXCH: admitted unless VGPU_HOST_GPU_ATOMICS=0 (Round sp66; an opt-in before, see
  // docs/perf-design/12-host-gpu-turbo.md section 2 for why the default changed).
  bool atomics = true;
  ClassMask off = 0;        // VGPU_HOST_GPU_TURBO_OFF: classes switched off for an experiment
  bool printf_off = false;  // VGPU_HOST_GPU_TURBO_OFF=printf: a kernel that calls vprintf stays in the simulator (Round sp66)
};

// VGPU_HOST_GPU_TURBO_OFF=mufu,tensor,half,call,atomics,red,extra,printf (a comma list). False, with the bad name in *bad, for anything else.
// `printf` is not a class of instructions (a kernel's call to vprintf is, once admitted, of the call class): it sets *printf_off, and is a bad
// name when printf_off is null.
inline bool parse_turbo_off(const char* text, ClassMask* off, std::string* bad = nullptr, bool* printf_off = nullptr) {
  *off = 0;
  if (printf_off) *printf_off = false;
  if (!text) return true;
  const std::string t(text);
  for (size_t at = 0; at <= t.size();) {
    const size_t comma = std::min(t.find(',', at), t.size());
    const std::string name = t.substr(at, comma - at);
    at = comma + 1;
    if (name.empty()) continue;
    if (name == "mufu") *off |= class_bit(InsnClass::Mufu);
    else if (name == "tensor") *off |= class_bit(InsnClass::Tensor);
    else if (name == "half") *off |= class_bit(InsnClass::Half);
    else if (name == "call") *off |= class_bit(InsnClass::Call);
    else if (name == "atomics") *off |= class_bit(InsnClass::AtomInt) | class_bit(InsnClass::AtomFloat) | class_bit(InsnClass::AtomSwap);
    else if (name == "red") *off |= class_bit(InsnClass::FloatRed);
    else if (name == "extra") *off |= class_bit(InsnClass::Extra);
    else if (name == "printf" && printf_off) *printf_off = true;
    else {
      if (bad) *bad = name;
      return false;
    }
  }
  return true;
}

// Why turbo mode keeps a kernel in the simulator, by the classes of its code. The first that applies, in this order.
enum class TurboRefusal : uint8_t {
  None,      // admitted
  Never,     // an instruction outside every class
  Off,       // a class switched off with VGPU_HOST_GPU_TURBO_OFF
  Atomics,   // a float atomic with a returned value, a CAS or an EXCH, with VGPU_HOST_GPU_ATOMICS=0
};
inline TurboRefusal turbo_refusal(ClassMask classes, const TurboSwitches& sw) {
  if (classes & class_bit(InsnClass::Never)) return TurboRefusal::Never;
  if (classes & sw.off) return TurboRefusal::Off;
  if (!sw.atomics && (classes & (class_bit(InsnClass::AtomFloat) | class_bit(InsnClass::AtomSwap)))) return TurboRefusal::Atomics;
  return TurboRefusal::None;
}

// ---- residency (the design for a card copy that outlives a launch) ---------------------------------------------------------

enum class Where : uint8_t {
  Host,   // the simulator's bytes are the truth; a card copy, if any, is stale
  Card,   // the card's bytes are the truth; the simulator's are stale (never read them)
  Both,   // the two are equal
};

// The residency of the chunks of one allocation. A new allocation is Host (zeros live in the simulator's sparse table), a
// simulator write makes a chunk Host, a launch on the card makes every chunk of every allocation it may reach Card (it may
// write any of them: pointers inside memory cannot be followed), a push makes Host chunks Both, and a pull makes Card chunks
// Both. The protocol's whole content: a reader or writer on one side first calls the "need" function for its range and
// copies what it returns; a copy in the other direction is never needed for correctness, only for speed.
class Residency {
 public:
  explicit Residency(uint64_t chunks = 0) : state_(chunks, Where::Host) { n_[0] = chunks; }
  uint64_t chunks() const { return state_.size(); }
  Where at(uint64_t c) const { return state_[c]; }

  // The simulator is about to write chunks [first, last]: anything the card holds of them goes stale. The chunks must be
  // current in the simulator first, unless the whole chunk is overwritten (the caller says so with `whole`).
  // Returns the chunks that have to be pulled from the card before the write (partial writes only).
  std::vector<uint64_t> sim_write(uint64_t first, uint64_t last, bool whole) {
    std::vector<uint64_t> pull;
    for (uint64_t c = first; c <= last && c < state_.size(); ++c) {
      if (state_[c] == Where::Card && !whole) pull.push_back(c);
      put(c, Where::Host);
    }
    return pull;
  }
  // The simulator is about to read chunks [first, last]: the Card ones have to come back first (they become Both).
  std::vector<uint64_t> sim_read(uint64_t first, uint64_t last) {
    std::vector<uint64_t> pull;
    for (uint64_t c = first; c <= last && c < state_.size(); ++c)
      if (state_[c] == Where::Card) {
        pull.push_back(c);
        put(c, Where::Both);
      }
    return pull;
  }
  // A launch on the card is about to run: the Host chunks have to be pushed (they become Both). Every chunk is then Card,
  // for the launch may write any of them; call `card_launched` after it.
  std::vector<uint64_t> card_prepare() {
    std::vector<uint64_t> push;
    for (uint64_t c = 0; c < state_.size(); ++c)
      if (state_[c] == Where::Host) {
        push.push_back(c);
        put(c, Where::Both);
      }
    return push;
  }
  void card_launched() {
    std::fill(state_.begin(), state_.end(), Where::Card);
    n_[0] = n_[2] = 0;
    n_[1] = state_.size();
  }
  // After a launch whose written chunks are known (a host-side diff, or a kernel whose stores are bounded).
  void card_launched_writing(uint64_t first, uint64_t last) {
    for (uint64_t c = first; c <= last && c < state_.size(); ++c) put(c, Where::Card);
  }
  // O(1): the counts are kept by every change of a chunk (the gate asked for them after every access to a card-valid allocation, in a loop over the chunks, which
  // made PolyBench 2DCONV with VGPU_HOST_GPU_VERIFY=1 and 128 MiB live run for more than 8 minutes, Round sp42 finding 6).
  uint64_t count(Where w) const { return n_[static_cast<unsigned>(w)]; }
  // Chunks whose card copy is valid (Card or Both): an allocation with any is kept out of the memory manager's lookup memories.
  uint64_t card_valid() const { return n_[1] + n_[2]; }
  // Whether an access of [first, last] needs the gate to change anything: a read, when a chunk of it is the card's alone (it has to be pulled); a write, when a chunk
  // of it is not the simulator's alone (a card copy goes stale, or has to be pulled first). Reads of Host and Both chunks, and writes to Host chunks, change nothing.
  bool access_changes_state(uint64_t first, uint64_t last, bool write) const {
    for (uint64_t c = first; c <= last && c < state_.size(); ++c)
      if (write ? state_[c] != Where::Host : state_[c] == Where::Card) return true;
    return false;
  }
  // The chunks a launch on the card has to push (the simulator's bytes are the truth for them).
  std::vector<uint64_t> host_chunks() const {
    std::vector<uint64_t> out;
    if (n_[0] == 0) return out;
    for (uint64_t c = 0; c < state_.size(); ++c)
      if (state_[c] == Where::Host) out.push_back(c);
    return out;
  }
  // A launch the simulator runs: it may read or write anything, so the Card chunks come back first and every chunk is then Host.
  std::vector<uint64_t> sim_launch() {
    std::vector<uint64_t> pull;
    for (uint64_t c = 0; c < state_.size(); ++c)
      if (state_[c] == Where::Card) pull.push_back(c);
    std::fill(state_.begin(), state_.end(), Where::Host);
    n_[1] = n_[2] = 0;
    n_[0] = state_.size();
    return pull;
  }
  // The same, for the chunks a simulator launch is known not to write (not used by the runtime: pointers inside memory cannot be followed).
  void set(uint64_t c, Where w) { put(c, w); }

 private:
  void put(uint64_t c, Where w) {
    --n_[static_cast<unsigned>(state_[c])];
    ++n_[static_cast<unsigned>(w)];
    state_[c] = w;
  }
  std::vector<Where> state_;
  uint64_t n_[3] = {0, 0, 0};   // chunks per Where (Host, Card, Both)
};

// Chunk indices to [first, count) runs, for one copy call per run of adjacent chunks.
inline std::vector<std::pair<uint64_t, uint64_t>> chunk_runs(const std::vector<uint64_t>& idx) {
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (uint64_t c : idx) {
    if (!out.empty() && out.back().first + out.back().second == c) ++out.back().second;
    else out.emplace_back(c, 1);
  }
  return out;
}

// The chunks a host access of [addr, addr + len) touches in the allocation [base, base + size), clamped to the allocation, and
// whether the access covers its first and its last chunk completely (a write that does not needs the chunk's old bytes first).
struct ChunkSpan {
  bool any = false;
  uint64_t first = 0, last = 0;
  bool first_whole = false, last_whole = false;
};
inline ChunkSpan chunk_span(uint64_t base, uint64_t size, uint64_t addr, uint64_t len, uint64_t chunk_bytes = 64 * 1024) {
  ChunkSpan s;
  if (!len || !size || addr >= base + size || addr + len <= base) return s;
  const uint64_t lo = addr < base ? 0 : addr - base;
  const uint64_t hi = std::min<uint64_t>(addr + len - base, size);   // exclusive
  s.any = true;
  s.first = lo / chunk_bytes;
  s.last = (hi - 1) / chunk_bytes;
  const auto chunk_end = [&](uint64_t c) { return std::min<uint64_t>((c + 1) * chunk_bytes, size); };
  s.first_whole = lo <= s.first * chunk_bytes && hi >= chunk_end(s.first);
  s.last_whole = lo <= s.last * chunk_bytes && hi >= chunk_end(s.last);
  return s;
}

// ---- the size rule ---------------------------------------------------------------------------------------------------------

// Measured on a g5 rig (A10G, PCIe 4.0): 09 doc, M3. One launch and its synchronisation 7 us; contiguous pageable copies 12 GB/s
// in and 10 GB/s out (1 MiB and up; 64 KiB chunk-at-a-time copies 4.4 and 4.0 GB/s, gathered through pinned staging 5.8).
struct Costs {
  double launch_s = 7e-6;
  double push_bytes_per_s = 12e9;
  double pull_bytes_per_s = 10e9;
  double kernel_s_default = 100e-6;   // before the card has run this kernel once
  double margin = 2.0;                // offload only when the simulator is expected to take this many times the card path
  double sim_warp_instr_per_s = 100e6;   // a lower bound of the simulator's speed in warp instructions per second (tens to ~150 M per thread in the benchmarks)
  double min_sim_s = 200e-6;          // never below this, whatever the arithmetic says (timer noise, the estimate's error)
  // The one-off cost of the first card launch: loading the driver, cuInit, the primary context and the window reservation. Measured on an A10G at 0.23 s of
  // process wall time (docs/performance.md "Round sp42": the native process of a trivial program also takes 0.24 s). 0.25 rounds it up. VGPU_HOST_GPU_START_COST
  // sets it in seconds; 0 charges nothing (the rule before sp43: the driver starts at the first launch the per-launch rule accepts).
  double start_cost_s = 0.25;
  // While the driver is not up, a launch that only the account sends (should_offload_with_start: the program has spent start_cost_s already, but the
  // launch does not repay the start by itself) must save at least start_cost_s / start_launches by itself: the start is then repaid within this many
  // launches like it. 0 asks nothing more of it (the account alone, the rule from sp43 to the integration of sp64-sp68; VGPU_HOST_GPU_START_LAUNCHES).
  double start_launches = 50;
  // ... and the first-launch rule's launch, whose estimate is static (0.2 to 80 times off), a larger share: start_cost_s / start_launches_static.
  double start_launches_static = 20;
};

struct Estimate {
  double sim_s = 0;                   // what the simulator is expected to need
  std::optional<double> card_kernel_s;
  uint64_t push_bytes = 0;
  uint64_t pull_bytes = 0;
  double start_s = 0;                 // the driver start this launch would pay (the first card launch of the process); 0 once the driver is up
};

inline double card_path_seconds(const Estimate& e, const Costs& c) {
  return e.start_s + c.launch_s + e.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(e.push_bytes) / c.push_bytes_per_s +
         static_cast<double>(e.pull_bytes) / c.pull_bytes_per_s;
}

inline bool should_offload(const Estimate& e, const Costs& c) {
  return e.sim_s >= c.min_sim_s && e.sim_s > c.margin * card_path_seconds(e, c);
}

// The rule for a launch, charging the driver start while it has not been paid (sp43). Before it, the rule judged a launch against the card path ALONE and a
// program whose launches each looked worth a card paid the 0.23 s start for a saving of milliseconds (all 8 Rodinia programs: 0.04x to 0.29x), and some paid
// it with every launch kept in the simulator (the start was kicked off at the first big-enough launch and awaited at exit). The driver now starts only when
// this function says a launch goes, and it says so when EITHER
//   * the launch alone repays the start: the simulator is expected to take more than `margin` times the card path INCLUDING the start (a single large launch:
//     PolyBench SYRK, SYR2K, 2MM, FDTD-2D), or
//   * the launch is worth a card with the driver up (should_offload without the start) AND the program has already spent at least `start_cost_s` of simulator
//     time in launches that could have gone to the card (`eligible_sim_spent_s`: the sum over every launch of an exact-set kernel that stayed in the simulator).
//     This is the ski-rental rule: the start is bought only after as much has been spent renting as buying costs, so a start that turns out useless costs at most
//     the time the program had already spent, never more than doubling it, and a program that never spends that much never starts the driver.
// With the driver up the start is paid and only should_offload applies. start_cost_s <= 0 is the old rule.
//
// The account alone bought starts that short programs could not repay: it says the program has run long enough, not that the launches still to
// come will save the start. On one A10G (Round sp66) four CUDA Samples whose newly admitted kernels made the account buy it lost 0.13 to 0.24 s:
// LargeKernelParameter (launches of under 0.5 ms), threadFenceReduction (about 3.5 ms), dct8x8 (3.8 ms), and MC_EstimatePiInlineP, whose last
// launch (a static estimate of 3.0 ms, by the first-launch rule) was sent after a 3.3 s launch had filled the account. So a launch the account
// sends must also save a share of the start by itself (start_share_saved, Costs::start_launches): 5 ms with the defaults, the start repaid within
// 50 launches like it. The launches that bought the start in the programs where it paid were longer: logged on server3 (integration of sp64-sp68),
// whose simulator ran these programs about 4 times as fast as the A10G rig's, 18 to 28 ms in PolyBench FDTD-2D, LU and GRAMSCHM, 19 ms (static) in
// 3MM, 5.4 ms in Rodinia gaussian (about 20 ms on the rig) and 1.9 ms in HeCBench asyncAllocation's turbo run (about 8 ms on the rig: 800 launches
// in 6.6 s; 6.6x faster in turbo mode, Round sp66). The constant is fitted to that corpus on the rig: the closest are dct8x8 below it (1.3x) and
// asyncAllocation above it (1.6x), and a host whose simulator is much faster than the rig's sees shorter launches and buys fewer starts.
// A first launch sent by its static estimate (first_launch_offload) must save start_cost_s / start_launches_static, 12.5 ms: the estimate does not
// depend on the host, and on server3 it was 8.5 ms for the dct8x8 kernel whose first launch would otherwise buy the start in turbo mode, against 15.5 ms
// in HeCBench atomicPerf (1.43x in turbo mode on the rig, Round sp66) and 19 ms in 3MM (1.25x).
// Launches that repay the start alone (the sampled ones of SYRK, SYR2K, 2MM, COVAR, CORR and black-scholes among them) are not affected, and
// neither is any launch once the driver is up.
inline bool start_share_saved(const Estimate& e, const Costs& c) {
  if (!(c.start_launches > 0)) return true;
  return e.sim_s - card_path_seconds(e, c) >= c.start_cost_s / c.start_launches;
}
inline bool should_offload_with_start(const Estimate& e, const Costs& c, bool driver_started, double eligible_sim_spent_s) {
  if (!should_offload(e, c)) return false;                      // not worth a card even with the driver up
  if (driver_started || c.start_cost_s <= 0) return true;
  Estimate cold = e;
  cold.start_s = c.start_cost_s;
  if (should_offload(cold, c)) return true;                     // the launch repays the start by itself
  return eligible_sim_spent_s >= c.start_cost_s && start_share_saved(e, c);   // (`e`: the card path with the driver up)
}

// ---- a launch while the card holds the live memory (Round sp65) ---------------------------------------------------------------------
// With the residency of the live memory (Residency), a launch on the card leaves every live chunk the card's alone (Where::Card), and a launch the
// simulator runs first pulls every such chunk back and leaves every chunk the simulator's (Residency::sim_launch), so the next card launch pushes them
// all again. should_offload weighs the launch's simulator time alone against the card path: a short launch between two card launches (PolyBench
// GRAMSCHM's two small kernels between its long one, Rodinia gaussian's Fan1 between its Fan2) is then kept in the simulator, which pulls the whole
// live memory back for it, and the card launch after it pushes it all again: Round sp65 measured X MiB moved that way over the corpus.
// So, once the card holds chunks the simulator does not, a launch of a kernel with an estimate is priced with what each side really moves:
//   keeping it costs its simulator time AND the pull of every chunk only the card holds (`card_only_bytes`), at the size rule's pull rate;
//   sending it costs the launch, the card's time for the kernel (its last time there; kernel_s_default before it has run there) and the push of the
//   chunks the simulator holds alone (`push_bytes`): nothing else moves, the card copy stays.
// It goes when keeping it costs `margin` times sending it, and at least min_sim_s (the same margin and floor as should_offload, now on both sides'
// real costs). The pushes the next card launch would make after a simulator launch are not counted: they are not paid if no card launch follows,
// so the rule errs towards the simulator. Only with the driver up (a launch is never what starts it: the start-cost rules are untouched) and only
// when the card holds data (`card_only_bytes` > 0, which needs the last launch that touched the memory to have run on the card).
struct StreakLaunch {
  double sim_s = 0;                       // the history's estimate (SimTimeTable::predict)
  uint64_t card_only_bytes = 0;           // live bytes in Where::Card chunks: what the simulator pulls before it runs the launch
  uint64_t push_bytes = 0;                // live bytes in Where::Host chunks: what the card needs first
  std::optional<double> card_kernel_s;    // this kernel's last time on the card
};
// What keeping the launch in the simulator costs, and what sending it does.
inline double streak_keep_seconds(const StreakLaunch& s, const Costs& c) {
  return s.sim_s + static_cast<double>(s.card_only_bytes) / c.pull_bytes_per_s;
}
inline double streak_card_seconds(const StreakLaunch& s, const Costs& c) {
  return c.launch_s + s.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(s.push_bytes) / c.push_bytes_per_s;
}
inline bool streak_offload(const StreakLaunch& s, const Costs& c, bool driver_started) {
  if (!driver_started || s.card_only_bytes == 0 || !(s.sim_s > 0)) return false;
  const double keep = streak_keep_seconds(s, c);
  return keep >= c.min_sim_s && keep > c.margin * streak_card_seconds(s, c);
}

// What the bans (offload_paid_off, TrafficTable) weigh a card launch's path against: the simulator's estimate, and while the card holds chunks the
// simulator does not (`card_only_bytes` > 0, the streak rule on), the pull that keeping the launch would have cost as well, whichever rule sent it.
// Against the estimate alone, one slow launch of GRAMSCHM's 1-block kernel1 (0.15 ms on the card, 0.29 ms estimated, sent by the size rule)
// ended its offloading and broke the streak for the rest of the run (Round sp65, D).
inline double ban_reference_seconds(const StreakLaunch& s, const Costs& c, bool streak_rule) {
  return streak_rule && s.card_only_bytes > 0 ? streak_keep_seconds(s, c) : s.sim_s;
}

// A grid or block with a zero dimension is the simulator's to refuse with its own error (cudaErrorInvalidConfiguration); on the card the launch
// fails, and with memory resident there that ended the mode with an internal error (PolyBench LU's last lu_kernel1, Round sp65).
inline bool launch_shape_empty(const uint32_t grid[3], const uint32_t block[3]) {
  return !grid[0] || !grid[1] || !grid[2] || !block[0] || !block[1] || !block[2];
}

// ---- start-up of the real driver ---------------------------------------------------------------------------------------------------
// Loading the driver and making its context costs 0.15 to 0.35 s (09 doc, section 5; 0.23 s on the A10G of docs/performance.md "Round sp42"). Nothing touches
// the driver, and no thread is started for it, until should_offload_with_start has decided that a launch goes to the card: the start is then made at that launch
// and the launch waits for it (Costs::start_cost_s is part of that decision). A program that never has such a launch never loads the driver and does not wait for it
// at exit. (Until sp42 the start was kicked off, on a thread, at the first launch of an exact-set kernel on a grid of 32 warps or more and awaited at exit, so
// every program with one such launch paid it, offloaded or not.)

// An estimate for a launch the simulator has not run yet: every warp of the grid executes the kernel's static instruction count, at the
// simulator's rate. It is NOT a lower bound: a thread that leaves at a bounds check runs a few instructions (PolyBench 3DCONV, a stencil
// whose grid covers the array and whose threads mostly exit, simulated in 3 ms and estimated at 60 ms), and a loop adds to the count.
// The runtime uses it only when asked (VGPU_HOST_GPU_STATIC=1); by default a kernel's first launch is the simulator's.
inline double static_sim_seconds(uint64_t blocks, uint32_t threads_per_block, uint64_t static_instrs, const Costs& c) {
  const uint64_t warps_per_block = (threads_per_block + 31) / 32;
  return static_cast<double>(blocks * warps_per_block * static_instrs) / c.sim_warp_instr_per_s;
}

// What the simulator took for a kernel before, by kernel, per block of the grid. The first two launches of a kernel run in the simulator: the first one
// pays decoding and set-up that the later ones do not (PolyBench 3DCONV's first launch took far longer than the 252 after it, and an estimate from it sent
// 253 launches of 3 ms to a card path of 100 ms: 38x slower), so the estimate is the smallest per-block time seen, which errs towards the simulator, as the
// rule should. It is available after two samples.
//
// One exception, for the launches that cost the most (sp15): a first sample that took kOneSampleMinSeconds or more is a launch whose simulator time is far above any
// decode or set-up cost, and running the kernel a second time in the simulator just to learn what it already showed would cost as much again. Such a kernel
// has an estimate after ONE sample, for at most kOneSampleOffloads offloaded launches (the estimate may still hold set-up time, which would blind the
// offload_paid_off check below): after that it needs a second, simulated sample like any other kernel, and the ban rules judge the card path against that.
inline constexpr double kOneSampleMinSeconds = 0.25;
inline constexpr unsigned kOneSampleOffloads = 2;

class SimTimeTable {
 public:
  void record(const std::string& kernel, uint64_t blocks, double seconds) {
    Entry& e = table_[kernel];
    const double per_block = blocks ? seconds / static_cast<double>(blocks) : seconds;
    if (!e.samples) e.first_total_s = seconds;
    e.per_block_s = e.samples ? std::min(e.per_block_s, per_block) : per_block;
    ++e.samples;
  }
  std::optional<double> predict(const std::string& kernel, uint64_t blocks) const {
    const auto it = table_.find(kernel);
    if (it == table_.end()) return std::nullopt;
    const Entry& e = it->second;
    const bool one_sample_ok = e.samples == 1 && e.first_total_s >= kOneSampleMinSeconds && e.one_sample_offloads < kOneSampleOffloads;
    if (e.samples < kSamplesNeeded && !one_sample_ok) return std::nullopt;
    return e.per_block_s * static_cast<double>(std::max<uint64_t>(blocks, 1));
  }
  // An offloaded launch used an estimate from `kernel`'s samples: counts against the one-sample allowance.
  void note_offload(const std::string& kernel) {
    const auto it = table_.find(kernel);
    if (it != table_.end() && it->second.samples < kSamplesNeeded) ++it->second.one_sample_offloads;
  }
  unsigned samples(const std::string& kernel) const {
    const auto it = table_.find(kernel);
    return it == table_.end() ? 0 : it->second.samples;
  }
  // What the samples so far say for a launch of `blocks`, however few they are (predict() answers only with enough of them). For the decision to
  // time the first blocks of a launch (worth_sampling below), which is not made again for a kernel whose one whole launch was short.
  std::optional<double> peek(const std::string& kernel, uint64_t blocks) const {
    const auto it = table_.find(kernel);
    if (it == table_.end() || !it->second.samples) return std::nullopt;
    return it->second.per_block_s * static_cast<double>(std::max<uint64_t>(blocks, 1));
  }
  // A launch that went to the card on the time of its first blocks (sample_offload below): `seconds` is what the simulator would have taken for
  // the whole grid. It is an estimate made to err low (sampled_launch_seconds) and without the first block's decode and set-up (the reason a
  // first whole sample is otherwise not trusted alone), so it counts as a full history; a later launch the simulator runs whole can only lower it.
  void record_extrapolated(const std::string& kernel, uint64_t blocks, double seconds) {
    record(kernel, blocks, seconds);
    Entry& e = table_[kernel];
    e.samples = std::max(e.samples, kSamplesNeeded);
  }
  static constexpr unsigned kSamplesNeeded = 2;

 private:
  struct Entry {
    double per_block_s = 0;
    double first_total_s = 0;
    unsigned samples = 0;
    unsigned one_sample_offloads = 0;
  };
  std::map<std::string, Entry> table_;
};

// ---- the first launch of a kernel (docs/perf-design/09-host-gpu-spike.md 6, step 2) --------------------------------------------------
// The history rule needs two simulator launches of a kernel, so a program that launches each kernel once (PolyBench GEMM, SYRK, SYR2K, 3MM) was never
// offloaded. A static count of the code cannot say how long threads run (3DCONV, a stencil with no loop whose threads mostly leave at a bounds check, was
// 38x slower with VGPU_HOST_GPU_STATIC=1), so the first launch of a kernel goes to the card only when ALL of these hold:
//  * the kernel has a loop (a branch back to an earlier instruction): threads cannot all leave at a bounds check, they have work that grows with the code's loop;
//  * the grid covers at least kFirstLaunchBlocksPerSm blocks per multiprocessor: a grid that does not fill the machine is small however long a thread runs;
//  * the live bytes (the most the launch can move each way) are at most kFirstLaunchMaxBytes: a mistaken first launch is bounded;
//  * the static estimate beats the card path by the usual margin, with the pushes and pulls of the whole live memory.
// Only the kernel's first launch is decided this way: the next ones are the simulator's until its history exists, as before.
inline constexpr uint64_t kFirstLaunchBlocksPerSm = 8;
inline constexpr uint64_t kFirstLaunchMaxBytes = 256ull << 20;

// Whether a branch goes back to its own instruction or an earlier one. `branches` holds (pc, target) of every branch of the kernel's code.
inline bool has_back_edge(const std::vector<std::pair<uint64_t, uint64_t>>& branches) {
  for (const auto& [pc, target] : branches)
    if (target <= pc) return true;
  return false;
}

// Whether the code can run an instruction twice: a cycle in its flow that the entry reaches. has_back_edge says yes for any branch to itself or a
// lower address, and compilers end every kernel with a branch to itself after its EXIT (PolyBench 2DCONV, 3DCONV and GEMVER have a back edge and no loop). Each
// instruction is given by where it may go next: the one after it (`falls_through`), a known `target` (an index, or -1), or anywhere (`indirect`:
// a jump through a register, or a call, is taken to close a loop).
struct FlowInsn {
  int64_t target = -1;
  bool falls_through = true;
  bool indirect = false;
};
inline bool code_has_loop(const std::vector<FlowInsn>& code) {
  const size_t n = code.size();
  std::vector<uint8_t> state(n, 0);   // 0 not seen, 1 on the path being walked, 2 done
  std::vector<std::pair<size_t, int>> stack;   // (instruction, which of its edges comes next)
  for (const FlowInsn& f : code)
    if (f.indirect) return true;
  // From the entry only: what cannot be reached is not run (compilers end a kernel with a branch to itself, after its EXIT).
  for (size_t root = 0; root < std::min<size_t>(n, 1); ++root) {
    stack.emplace_back(root, 0);
    state[root] = 1;
    while (!stack.empty()) {
      auto& [i, edge] = stack.back();
      size_t next = n;
      for (; edge < 2 && next == n; ++edge) {
        if (edge == 0 && code[i].falls_through && i + 1 < n) next = i + 1;
        if (edge == 1 && code[i].target >= 0 && static_cast<size_t>(code[i].target) < n) next = static_cast<size_t>(code[i].target);
      }
      if (next == n) {
        state[i] = 2;
        stack.pop_back();
        continue;
      }
      if (state[next] == 1) return true;
      if (state[next] == 0) {
        state[next] = 1;
        stack.emplace_back(next, 0);
      }
    }
  }
  return false;
}

struct FirstLaunch {
  uint64_t blocks = 0;
  uint32_t threads_per_block = 0;
  uint64_t static_instrs = 0;
  bool back_edge = false;
  uint32_t multiprocessors = 0;
  uint64_t live_bytes = 0;
};

inline bool first_launch_offload(const FirstLaunch& f, const Costs& c, bool driver_started = true, double eligible_sim_spent_s = 0) {
  if (!f.back_edge || f.multiprocessors == 0 || f.blocks < kFirstLaunchBlocksPerSm * f.multiprocessors || f.live_bytes > kFirstLaunchMaxBytes) return false;
  Estimate e;
  e.sim_s = static_sim_seconds(f.blocks, f.threads_per_block, f.static_instrs, c);
  e.push_bytes = f.live_bytes;
  e.pull_bytes = f.live_bytes;
  Costs fc = c;   // (the static estimate's share of the start, see should_offload_with_start; none when start_launches asks none)
  if (c.start_launches > 0) fc.start_launches = c.start_launches_static;
  return should_offload_with_start(e, fc, driver_started, eligible_sim_spent_s);
}

// ---- the first blocks of a launch, timed (docs/perf-design/12-host-gpu-turbo.md "The first blocks of a launch, timed"; Round sp60b) ----------
// The static estimate of first_launch_offload counts every instruction of the code once a warp, so it is low by the trip count of the kernel's loops,
// and the history rule needs launches that some programs never make again (one launch a kernel). Nothing static says how long a thread runs; the
// launch itself does. So a launch the rules above leave to the simulator for want of an estimate is SAMPLED before the simulator runs it: the process
// is forked, the copy runs a few of the launch's blocks on one thread, each twice with the second run timed, reports the times and ends. Nothing the copy did exists in the process
// that goes on (its memory is its own, copy-on-write), so there is nothing to undo: if the times say that the launch repays the card path, the
// driver start included, the card runs the launch; if not the simulator does, all of it, as it would have.
// The decision is from measured time, so it cannot be low by a loop's trip count, and it charges the driver start in full against this one launch:
// the account of should_offload_with_start (simulator time already spent) is not consulted, which is what let short programs pay for a start they
// could not use (sp59: backprop, bitonic-sort, atomicPerf).

// How many blocks the copy runs: none for a grid of fewer than kSampleMinBlocks; else up to kSampleMaxBlocks, spread evenly over the grid (block 0,
// then every sample_stride-th: the blocks of a grid need not be alike, Rodinia gaussian's first blocks do nine times the work of its average one),
// and no further block once the timed runs have taken kSampleBudgetSeconds. Each block is run twice and the second run is the timed one: the
// first pays what a launch pays once (decode, set-up, the chunks of memory the launch is the first to write), which made PolyBench GEMVER's
// blocks look seven times as long as they are.
inline constexpr uint64_t kSampleMinBlocks = 256;
inline constexpr uint64_t kSampleMaxBlocks = 32;
inline constexpr double kSampleBudgetSeconds = 0.002;
inline uint64_t sample_blocks(uint64_t blocks) {
  if (blocks < kSampleMinBlocks) return 0;
  return std::min<uint64_t>(blocks / 2, kSampleMaxBlocks);
}
inline uint64_t sample_stride(uint64_t blocks) {
  const uint64_t n = sample_blocks(blocks);
  return n ? blocks / n : 1;
}
// A sample costs the fork, the copy's decode and set-up and the sampled blocks, whatever the launch turns out to be, and the process waits for it:
// 1 to 12 ms in Round sp60b's corpus. It also costs afterwards: the fork write-protects the process's pages, and each page the process writes later
// takes a fault (Rodinia backprop, 36 MiB live: 6 ms waited and 19 ms more over the run, on the A10G rig). The copy is given kSampleTimeoutSeconds
// at most (sample_timeout_seconds: more with much live memory); one that has not answered by then is ended and the launch is the simulator's. What keeps a short program from paying for samples it
// cannot use:
//  * only a kernel with a loop is sampled (code_has_loop). Without one a warp runs each instruction once at most, the static count bounds its work,
//    and the launches of such kernels took 0.4 to 1.2 times their static estimate: none is long without a long estimate, which the rules above see;
//  * and EITHER its static estimate (static_sim_seconds) is at least sample_static_floor: 30 ms, ten times the cost of a small sample, and at
//    least three times what the faults afterwards are expected to cost (kSampleSecondsPerLiveByte);
//  * OR the process can afford it: what its samples cost so far, plus what this one is expected to cost, is within kSampleShareOfAge of the time
//    the process has run (`expected_s`: the dearest sample of the process so far, or kSampleCostGuessSeconds and the faults before the first).
// The floor is a rule of thumb fitted to this corpus, not a bound: the static estimate of a kernel with a loop says little (0.2 to 80 times off).
inline constexpr double kSampleTimeoutSeconds = 0.050;
// ... and longer with much live memory, whose page tables the fork copies: HeCBench black-scholes (1.9 GiB live) had not answered after 50 ms on
// the A10G rig and was not sent. 0.2 ms more per MiB: 0.43 s for it, a ninth of the static estimate the floor asks of such a launch.
inline constexpr double kSampleTimeoutSecondsPerLiveByte = 2e-10;
inline double sample_timeout_seconds(uint64_t live_bytes) {
  return kSampleTimeoutSeconds + kSampleTimeoutSecondsPerLiveByte * static_cast<double>(live_bytes);
}
inline constexpr double kSampleStaticFloorSeconds = 0.030;
inline constexpr double kSampleShareOfAge = 0.02;
inline constexpr double kSampleCostGuessSeconds = 0.003;
inline constexpr double kSampleSecondsPerLiveByte = 6e-10;
inline double sample_static_floor(uint64_t live_bytes) {
  return std::max(kSampleStaticFloorSeconds, 3.0 * kSampleSecondsPerLiveByte * static_cast<double>(live_bytes));
}
inline bool sample_affordable(double age_s, double spent_s, double expected_s) {
  return spent_s + std::max(expected_s, 0.0) <= kSampleShareOfAge * age_s;
}
// The card's kernel time is not known before its first launch there: assumed to be 1/kSampleCardSpeedup of what the simulator would take. 8 is the
// least the card was faster than the simulator on the device phase of any workload of docs/perf-design/07-host-gpu.md (8x to 105x), so the
// unknown counts against the card.
inline constexpr double kSampleCardSpeedup = 8.0;
// The first-launch rule's cap on the live memory (kFirstLaunchMaxBytes) is not applied: that rule cannot know what a mistaken move costs, and
// this one charges the copies each way against a measured time (HeCBench black-scholes has 1 GiB live and 6 s in one launch).

struct Sample {
  double seconds = 0;       // the timed (second) runs of the sampled blocks: their sum
  uint64_t timed = 0;       // how many blocks were sampled
  uint64_t total = 0;       // blocks of the grid
  unsigned threads = 1;     // host threads the whole launch would run on
};

// The seconds the simulator needs for the launch, from the sample: the per-block time over the whole grid, shared among the host threads as if
// they scaled perfectly. They do not, so the estimate mostly errs low, towards the simulator: the eight launches of Round sp60b's corpus that it
// would send took 0.9 to 1.6 times it; over all 68 sampled launches the estimate was between 0.1 and 6 times the launch, the high end on launches
// of a few milliseconds (docs/perf-design/data/sp60b/). None when no block was timed.
inline std::optional<double> sampled_launch_seconds(const Sample& s) {
  if (s.timed == 0 || s.total == 0 || !(s.seconds > 0)) return std::nullopt;
  const double per_block = s.seconds / static_cast<double>(s.timed);
  return per_block * static_cast<double>(s.total) / static_cast<double>(std::max<unsigned>(s.threads, 1));
}

// The card path of a sampled launch: the live memory the card does not hold goes, and all of it is charged for coming back, as the first-launch
// rule charges it.
inline Estimate sampled_estimate(double sim_s, uint64_t push_bytes, uint64_t live_bytes, const Costs& c, bool driver_started,
                                 std::optional<double> card_kernel_s = std::nullopt) {
  Estimate e;
  e.sim_s = sim_s;
  e.push_bytes = push_bytes;
  e.pull_bytes = live_bytes;
  e.card_kernel_s = card_kernel_s ? *card_kernel_s : std::max(c.kernel_s_default, e.sim_s / kSampleCardSpeedup);
  e.start_s = driver_started ? 0 : std::max(c.start_cost_s, 0.0);
  return e;
}

inline Estimate sample_estimate(const Sample& s, uint64_t push_bytes, uint64_t live_bytes, const Costs& c, bool driver_started,
                                std::optional<double> card_kernel_s = std::nullopt) {
  return sampled_estimate(sampled_launch_seconds(s).value_or(0), push_bytes, live_bytes, c, driver_started, card_kernel_s);
}

// Whether a sampled launch goes to the card: the simulator's time must repay the card path, with the driver start if it is not up, by the usual margin.
// The same rule for a launch sampled inside its blocks (inblock_launch_seconds gives its simulator time).
inline bool sampled_offload(double sim_s, uint64_t push_bytes, uint64_t live_bytes, const Costs& c, bool driver_started,
                            std::optional<double> card_kernel_s = std::nullopt) {
  // (a sample with no estimate has a simulator time of 0, which no card path is under)
  return should_offload(sampled_estimate(sim_s, push_bytes, live_bytes, c, driver_started, card_kernel_s), c);
}
inline bool sample_offload(const Sample& s, uint64_t push_bytes, uint64_t live_bytes, const Costs& c, bool driver_started,
                           std::optional<double> card_kernel_s = std::nullopt) {
  return sampled_offload(sampled_launch_seconds(s).value_or(0), push_bytes, live_bytes, c, driver_started, card_kernel_s);
}

// ---- progress inside a block (docs/perf-design/12-host-gpu-turbo.md section 8, "Progress inside a block"; Round sp61) ---------------------
// A grid of fewer than kSampleMinBlocks blocks gives no sample above: its blocks are few and each may run for seconds (PolyBench COVAR's
// covar_kernel: 8 blocks of 256 threads, every thread looping over the matrix). Such a launch is sampled INSIDE its blocks: the copy runs each
// block it takes until its warps stand in the kernel's top-level loop and reads, per lane, how far the loop's induction register is from the
// bound its compare reads (vgpu/exec/block_progress.hpp); in the first block it then runs the warp with the most turns left alone for a few
// milliseconds, which gives the steps of one turn and the engine's time per step. The launch is the steps taken plus the turns left at that
// cost, over the blocks and the host threads. Nothing about the loop is guessed from its shape: the bound is the value the compare will read.
//
// The loops are found in the code (find_probe_loops). A loop is followed when ALL of these hold, and a kernel with no such loop is not sampled:
//  * it is a top-level loop: a back edge reachable from the entry whose range [target, branch] lies in no other loop's range;
//  * a conditional branch in it (the back edge, or a branch out of the range) is guarded by a predicate that the last instruction writing it
//    before the branch, inside the loop, sets by an integer compare (ISETP / UISETP, not the .EX half of a 64-bit one);
//  * one side of the compare is the induction register, or a register the loop computes once from it by adding a constant: a register the loop
//    writes exactly once, by adding a constant to itself (IADD3 / VIADD / IADD / UIADD3 with one register source);
//  * the other side is an immediate, a constant-bank word (a kernel parameter) or a register the loop never writes.

// What the loop finder needs of an instruction: its flow (as code_has_loop takes it), and the few kinds of instruction a counted loop is made of.
// Registers are numbered R0..R254 as 0..254 and UR0..UR254 as 256..510; predicates P0..P6 as 0..6 and UP0..UP6 as 8..14 (PT and UPT are not
// numbered: an instruction guarded by one is unconditional).
struct LoopCodeInsn {
  FlowInsn flow;
  int branch_pred = -1;          // a conditional BRA: the predicate that guards it
  enum class Kind : uint8_t { Other, AddImm, Compare } kind = Kind::Other;
  int add_dst = -1, add_src = -1;   // AddImm: add_dst = add_src + add_imm
  int64_t add_imm = 0;
  int cmp_pred = -1, cmp_pred2 = -1;   // Compare: the predicates it writes
  exec::ProbeOperand cmp_a, cmp_b;     // Compare: its two values (Reg and UReg operands hold the register's number, unshifted)
  bool cmp_unsigned = false;
  std::vector<int> writes;       // every register it writes (an AddImm's add_dst included)
  std::vector<int> pred_writes;  // every predicate it writes (a Compare's included)
};

// A followed loop, by instruction index: what the runtime turns into an exec::ProbeLoop (code addresses) for the copy.
struct FoundLoop {
  size_t lo = 0, hi = 0;
  exec::ProbeLoop loop;   // lo and hi are left 0 here
};

inline int probe_reg_number(const exec::ProbeOperand& o) {
  if (o.kind == exec::ProbeOperand::Reg) return static_cast<int>(o.reg);
  if (o.kind == exec::ProbeOperand::UReg) return 256 + static_cast<int>(o.reg);
  return -1;
}
inline exec::ProbeOperand probe_reg_operand(int n) {
  exec::ProbeOperand o;
  o.kind = n >= 256 ? exec::ProbeOperand::UReg : exec::ProbeOperand::Reg;
  o.reg = static_cast<uint32_t>(n >= 256 ? n - 256 : n);
  return o;
}

inline std::vector<FoundLoop> find_probe_loops(const std::vector<LoopCodeInsn>& code) {
  std::vector<FoundLoop> out;
  const size_t n = code.size();
  for (const LoopCodeInsn& c : code)
    if (c.flow.indirect) return out;   // a jump through a register or a call: the flow is not all in view
  // What the entry reaches.
  std::vector<uint8_t> reach(n, 0);
  std::vector<size_t> todo;
  if (n) {
    reach[0] = 1;
    todo.push_back(0);
  }
  while (!todo.empty()) {
    const size_t i = todo.back();
    todo.pop_back();
    const auto go = [&](int64_t j) {
      if (j >= 0 && static_cast<size_t>(j) < n && !reach[static_cast<size_t>(j)]) {
        reach[static_cast<size_t>(j)] = 1;
        todo.push_back(static_cast<size_t>(j));
      }
    };
    if (code[i].flow.falls_through) go(static_cast<int64_t>(i) + 1);
    go(code[i].flow.target);
  }
  // Back edges, one range per loop head.
  std::map<size_t, size_t> range;   // head -> last back edge to it
  for (size_t i = 0; i < n; ++i) {
    const int64_t t = code[i].flow.target;
    if (!reach[i] || t < 0 || static_cast<size_t>(t) > i || !reach[static_cast<size_t>(t)]) continue;
    size_t& hi = range[static_cast<size_t>(t)];
    hi = std::max(hi, i);
  }
  // The top-level ones: in no other range. Two ranges that overlap without one holding the other are both left out.
  std::vector<std::pair<size_t, size_t>> top;
  for (const auto& [lo, hi] : range) {
    bool inner = false, crossed = false;
    for (const auto& [lo2, hi2] : range) {
      if (lo2 == lo && hi2 == hi) continue;
      if (lo2 <= lo && hi <= hi2) inner = true;
      else if ((lo2 < lo && lo <= hi2 && hi2 < hi) || (lo < lo2 && lo2 <= hi && hi < hi2)) crossed = true;
    }
    if (!inner && !crossed) top.emplace_back(lo, hi);
  }
  for (const auto& [lo, hi] : top) {
    if (out.size() >= exec::kProbeLoops) break;
    const auto writes_in_loop = [&](int reg, size_t* where) {
      unsigned k = 0;
      for (size_t j = lo; j <= hi; ++j)
        for (int w : code[j].writes)
          if (w == reg) {
            ++k;
            if (where) *where = j;
          }
      return k;
    };
    bool found = false;
    // The branches that decide: the back edges first (from the last), then the branches out of the range.
    std::vector<size_t> branches;
    for (size_t i = hi + 1; i-- > lo;)
      if (code[i].branch_pred >= 0 && code[i].flow.target >= static_cast<int64_t>(lo) && static_cast<size_t>(code[i].flow.target) <= i) branches.push_back(i);
    for (size_t i = hi + 1; i-- > lo;)
      if (code[i].branch_pred >= 0 && code[i].flow.target >= 0 &&
          (static_cast<size_t>(code[i].flow.target) < lo || static_cast<size_t>(code[i].flow.target) > hi))
        branches.push_back(i);
    for (size_t b : branches) {
      if (found) break;
      const int p = code[b].branch_pred;
      // The last write of the predicate before the branch, inside the loop.
      size_t j = b;
      bool have = false;
      while (j-- > lo) {
        bool writes = false;
        for (int w : code[j].pred_writes) writes = writes || w == p;
        if (writes) {
          have = true;
          break;
        }
      }
      if (!have || code[j].kind != LoopCodeInsn::Kind::Compare) continue;
      const LoopCodeInsn& cmp = code[j];
      if (cmp.cmp_pred != p && cmp.cmp_pred2 != p) continue;
      for (int side = 0; side < 2 && !found; ++side) {
        const exec::ProbeOperand& x = side ? cmp.cmp_b : cmp.cmp_a;
        const exec::ProbeOperand& y = side ? cmp.cmp_a : cmp.cmp_b;
        const int xr = probe_reg_number(x);
        if (xr < 0) continue;
        size_t at = 0;
        if (writes_in_loop(xr, &at) != 1 || code[at].kind != LoopCodeInsn::Kind::AddImm || code[at].add_dst != xr) continue;
        int iv = xr;
        int64_t offset = 0, step = code[at].add_imm;
        if (code[at].add_src != xr) {   // x = iv + offset, iv = iv + step
          iv = code[at].add_src;
          offset = code[at].add_imm;
          size_t at2 = 0;
          if (iv < 0 || writes_in_loop(iv, &at2) != 1 || code[at2].kind != LoopCodeInsn::Kind::AddImm || code[at2].add_dst != iv || code[at2].add_src != iv) continue;
          step = code[at2].add_imm;
        }
        if (step == 0) continue;
        // The bound: an immediate, a constant-bank word, or a register the loop does not write.
        if (y.kind == exec::ProbeOperand::None) continue;
        const int yr = probe_reg_number(y);
        if (yr >= 0 && writes_in_loop(yr, nullptr) != 0) continue;
        FoundLoop f;
        f.lo = lo;
        f.hi = hi;
        f.loop.iv = probe_reg_operand(iv);
        f.loop.iv_offset = offset;
        f.loop.step = step;
        f.loop.bound = y;
        f.loop.is_unsigned = cmp.cmp_unsigned;
        out.push_back(f);
        found = true;
      }
    }
  }
  return out;
}

// Which blocks the copy reads: up to kSampleMaxBlocks, spread over the grid, for a grid too small for the sample of whole blocks (and at least two blocks: a
// launch of one block has its own rules nowhere, and its one block is the whole launch).
inline uint64_t inblock_blocks(uint64_t blocks) {
  if (blocks >= kSampleMinBlocks || blocks < 1) return 0;
  return std::min<uint64_t>(blocks, kSampleMaxBlocks);
}
inline uint64_t inblock_stride(uint64_t blocks) {
  const uint64_t n = inblock_blocks(blocks);
  return n ? blocks / n : 1;
}
// Per block, the most its warps are run to reach the loop (their prologue is a few dozen instructions; this bounds a kernel whose warps never get there),
// and how long one warp of the first block is run alone to measure a turn of the loop. The copy takes no further block once the blocks read have taken
// kInBlockBudgetSeconds.
inline constexpr double kInBlockEnterSeconds = 0.001;
inline constexpr double kInBlockMeasureSeconds = 0.004;
inline constexpr double kInBlockBudgetSeconds = 0.012;

// The seconds the simulator needs for the launch, from the blocks read: the mean block's steps over the grid, shared among the host threads as if they
// scaled perfectly, and never less than the longest block read (a block is not shared among threads), at the engine's measured time per step. None when no
// block was read or the measure took no step.
inline std::optional<double> inblock_launch_seconds(const std::vector<exec::BlockReading>& blocks, const exec::TurnMeasure& m, uint64_t total, unsigned threads) {
  const double per_step = exec::seconds_per_step(m);
  const double per_turn = exec::steps_per_turn(m);
  if (blocks.empty() || total == 0 || !(per_step > 0) || !(per_turn > 0)) return std::nullopt;
  double sum = 0, most = 0;
  for (const exec::BlockReading& b : blocks) {
    const double s = exec::block_steps(b, per_turn);
    sum += s;
    most = std::max(most, s);
  }
  const double lanes = static_cast<double>(std::max<unsigned>(threads, 1));   // (more threads than blocks: the longest block decides)
  const double steps = std::max(sum / static_cast<double>(blocks.size()) * static_cast<double>(total) / lanes, most);
  return steps * per_step;
}

// Whether a launch with no estimate is sampled at all:
//  * the kernel has a loop and the grid has blocks to sample;
//  * what is known of the kernel does not already say it is short: `whole_s` is what its whole launches so far give for this grid (SimTimeTable::peek);
//    a kernel whose earlier launch was short is not sampled again (PolyBench 3DCONV: 254 launches of 3 ms);
//  * its static estimate reaches the floor, or the process can afford it (sample_affordable).
// `card_path_s` is the card path of this launch with the whole live memory moved each way and the driver start if it is not up.
struct SampleCandidate {
  uint64_t blocks = 0;
  bool loop = false;        // code_has_loop
  uint64_t live_bytes = 0;
  std::optional<double> whole_s;
  double static_s = 0;      // the launch's static estimate (static_sim_seconds)
  double age_s = 0;         // how long the process has run
  double spent_s = 0;       // what its samples have cost so far
  double expected_s = kSampleCostGuessSeconds;
  bool followed = false;    // the kernel has a loop a sample inside its blocks can follow (find_probe_loops; Round sp61)
  double bound_turns = 1;   // ... and the most turns one of them has if it runs from 0 to its bound (bound_turns), when the launch gives the bound
};
// Whether the launch is sampled inside its blocks (Round sp61) rather than by whole blocks: a grid too small for the latter, of a kernel with a followed loop.
inline bool sample_inside(const SampleCandidate& w) { return sample_blocks(w.blocks) == 0 && w.followed && inblock_blocks(w.blocks) != 0; }
// The static estimate the floor is asked about. For a launch sampled inside its blocks it is the static count times the turns of the followed loop as
// the launch's own arguments bound it (bound_turns): the plain count sees a loop's body once, and the kernels this sample is for are the ones whose
// threads loop for seconds (COVAR's covar_kernel: 87 us counted, 9.5 s taken on server3; its loop is bounded by the parameter m = 2048, which makes 0.18 s).
// It assumes the loop starts at 0, so it can be high (the prologue is counted every turn, a loop may start near its bound) or low (the loops inside it
// are not counted): it decides whether a sample is worth its cost, never whether the launch goes.
inline double sample_static_estimate(const SampleCandidate& w) {
  return sample_inside(w) ? w.static_s * std::max(w.bound_turns, 1.0) : w.static_s;
}
// The turns a followed loop has at most when it counts from 0 to `bound` by `step` (0 for no step); the floor's use of it is above.
inline double bound_turns(int64_t bound, int64_t step) {
  if (step == 0) return 0;
  const double b = static_cast<double>(bound < 0 ? -bound : bound), s = static_cast<double>(step < 0 ? -step : step);
  return std::floor(b / s);
}
inline bool worth_sampling(const SampleCandidate& w, const Costs& c, double card_path_s) {
  if (!w.loop || (sample_blocks(w.blocks) == 0 && !sample_inside(w))) return false;
  if (w.whole_s && *w.whole_s <= c.margin * card_path_s) return false;
  return sample_static_estimate(w) >= sample_static_floor(w.live_bytes) || sample_affordable(w.age_s, w.spent_s, w.expected_s);
}

// A safety net for any estimate that turns out wrong: an offloaded launch that took longer on the card path (copies, launch, copy back)
// than the simulator was expected to need ends the offloading of that kernel. One bad launch is the most it can cost.
inline bool offload_paid_off(double card_total_s, double sim_estimate_s) { return card_total_s <= sim_estimate_s; }

// A second safety net, by traffic (docs/perf-design/09-host-gpu-spike.md 7.3; docs/perf-design/rounds/08-rounds-sp11-sp15.md "Round sp14"). With a persistent card copy, a kernel that
// alternates with simulator launches on the same buffers gives the copy up and gets it pushed again every time. A kernel is banned when kTrafficLaunches
// offloaded launches IN A ROW each pushed more than kTrafficSharePercent percent of the live bytes AND more than kTrafficMinBytes, and each one's card path
// took more than kTrafficSlowShare of the simulator's estimate. All three conditions are needed: replayed over the 64 saved traces (residency_model.py), the share
// and the size alone also catch GRAMSCHM, whose offloaded launches are worth 3x their traffic (banning it loses 29%), and the share keeps base64e (8.4% of live bytes
// per push, 8.06 MiB) out. The kernel stays in the simulator for the rest of the process; a ban only moves a launch back, so results do not change.
// Measured on the traces the ban fires on LU only, and late: it is a net, not the fix for LU's 18% against write-through.
inline constexpr unsigned kTrafficLaunches = 2;
inline constexpr unsigned kTrafficSharePercent = 25;
inline constexpr uint64_t kTrafficMinBytes = 12ull << 20;
inline constexpr double kTrafficSlowShare = 0.5;

class TrafficTable {
 public:
  // One offloaded launch of `kernel`: it pushed `push_bytes` of `live_bytes`, its card path took `card_total_s`, the simulator was expected to need
  // `sim_estimate_s`. `timing_valid` is false for a kernel's first card launch, whose path includes the module load: then only the traffic counts.
  // Returns true when the kernel must go back to the simulator for good.
  bool record(const std::string& kernel, uint64_t push_bytes, uint64_t live_bytes, double card_total_s, double sim_estimate_s, bool timing_valid = true) {
    unsigned& heavy = heavy_[kernel];
    const bool big = push_bytes > kTrafficMinBytes && live_bytes > 0 && push_bytes * 100 > live_bytes * kTrafficSharePercent;
    const bool slow = !timing_valid || card_total_s > kTrafficSlowShare * sim_estimate_s;
    heavy = (big && slow) ? heavy + 1 : 0;
    return heavy >= kTrafficLaunches;
  }

 private:
  std::map<std::string, unsigned> heavy_;   // per kernel: offloaded launches in a row that were heavy
};

}  // namespace vgpu::hostgpu
