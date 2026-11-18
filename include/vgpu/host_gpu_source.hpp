// Host-GPU turbo mode: the images of a module that the card could run, and whether a launch fits the card
// (docs/perf-design/12-host-gpu-turbo.md "Turbo fit"). No simulator state and no CUDA: unit tested on any machine
// (tests/unit/test_host_gpu_logic.cpp).
//
// The simulator runs a module from the image its own profile picks (pick_cubin / best_ptx for the simulated device's compute capability).
// Turbo mode is not promised to return the simulator's bits, so the card is given the best image the CARD can run, which for a simulated
// profile of another architecture is another image of the same fatbin: a simulated H100 (sm_90) on an sm_86 card runs the fatbin's sm_86
// (or sm_80) cubin, or its PTX through the driver's own JIT, where the simulator runs the sm_90 one.
//
// The order is the driver's (NVIDIA's cuModuleLoadData): a cubin built for exactly the card's architecture, then the newest cubin of the card's
// major whose minor is not newer, then the newest PTX whose target is not newer than the card. An `a` cubin (sm_90a) runs on exactly its own
// architecture; PTX `.target sm_XYa` the same, `sm_XYf` within the family. These are the rules of sass::runs_on and nvidia/src/fatbin.cpp
// (pick_cubin, pick_ptx, ptx_runs): the shim extracts the candidates there, this header only chooses between them.
#pragma once

#include <cctype>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "vgpu/host_gpu.hpp"

namespace vgpu::hostgpu {

// One image of a module: a cubin (kind Cubin: `arch` is its sm_XX, `suffix` 'a' when it is an architecture-specific image) or PTX text (kind Ptx:
// `arch` and `suffix` are those of its `.target`).
struct CardCandidate {
  enum Kind : uint8_t { Cubin, Ptx };
  Kind kind = Cubin;
  int arch = 0;
  char suffix = 0;
  std::string bytes;
};

// The images the card may be given for a module. Shared by every kernel of the module (ptx::EntryFn::card).
struct CardSource {
  std::vector<CardCandidate> candidates;
};

// sass::runs_on: the same major and an architecture no newer than the card's; an `a` image only on exactly its architecture.
inline bool cubin_runs_on(int cubin_sm, bool arch_specific, int host_sm) {
  if (arch_specific) return cubin_sm == host_sm;
  return cubin_sm / 10 == host_sm / 10 && cubin_sm <= host_sm;
}

// fatbin.cpp ptx_runs: `a` exactly, `f` within the major, plain: any older or equal architecture.
inline bool ptx_target_runs_on(int target, char suffix, int host_sm) {
  return suffix == 'a' ? target == host_sm : suffix == 'f' ? (target / 10 == host_sm / 10 && target <= host_sm) : target <= host_sm;
}

// The `.target sm_XY[a|f]` of PTX text: the number, and the suffix (0 when plain). `ok` false when the text has no target line; the caller then
// reads the image as the fatbin header's architecture.
inline bool ptx_target_of(const std::string& text, int* arch, char* suffix) {
  *arch = 0;
  *suffix = 0;
  const size_t at = text.find(".target");
  if (at == std::string::npos) return false;
  const size_t sm = text.find("sm_", at);
  const size_t eol = text.find('\n', at);
  if (sm == std::string::npos || (eol != std::string::npos && sm > eol)) return false;
  size_t i = sm + 3;
  int n = 0;
  bool digits = false;
  while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
    n = n * 10 + (text[i] - '0');
    ++i;
    digits = true;
  }
  if (!digits) return false;
  *arch = n;
  if (i < text.size() && (text[i] == 'a' || text[i] == 'f')) *suffix = text[i];
  return true;
}

// The image the card runs: the index into `c`, or -1 when none of them runs on `host_sm`. Cubins first (the newest the card can run; of two for one
// architecture the `a` one), then PTX (the newest target; `a` before `f` before plain).
inline int pick_card_image(const std::vector<CardCandidate>& c, int host_sm) {
  int best = -1;
  for (size_t i = 0; i < c.size(); ++i) {
    if (c[i].kind != CardCandidate::Cubin || !cubin_runs_on(c[i].arch, c[i].suffix == 'a', host_sm)) continue;
    if (best < 0 || c[i].arch > c[best].arch || (c[i].arch == c[best].arch && c[i].suffix == 'a' && c[best].suffix != 'a')) best = static_cast<int>(i);
  }
  if (best >= 0) return best;
  const auto specificity = [](char s) { return s == 'a' ? 2 : s == 'f' ? 1 : 0; };
  for (size_t i = 0; i < c.size(); ++i) {
    if (c[i].kind != CardCandidate::Ptx || !ptx_target_runs_on(c[i].arch, c[i].suffix, host_sm)) continue;
    if (best < 0 || c[i].arch > c[best].arch || (c[i].arch == c[best].arch && specificity(c[i].suffix) > specificity(c[best].suffix))) best = static_cast<int>(i);
  }
  return best;
}

// Retargeting (turbo fit, docs/perf-design/12-host-gpu-turbo.md section 15): a module that holds PTX only for `.target sm_90` (a build with -arch=sm_90, or compute_90 alone) runs on
// an sm_80, sm_86 or sm_87 card when its `.target` line is rewritten to the card's and the driver compiles it, provided it uses no form that architecture lacks (scan_ptx_for_card
// with retarget). Measured with ptxas 13.0: 125 of 130 PTX files of the corpus assemble for sm_86 after the rewrite alone. The `.version` is never lowered.
inline bool retarget_host(int host_sm) { return host_sm == 80 || host_sm == 86 || host_sm == 87; }
inline int pick_retarget_image(const std::vector<CardCandidate>& c, int host_sm) {
  if (!retarget_host(host_sm)) return -1;
  for (size_t i = 0; i < c.size(); ++i)
    if (c[i].kind == CardCandidate::Ptx && c[i].arch == 90 && c[i].suffix == 0) return static_cast<int>(i);
  return -1;
}
// The text with its `.target sm_90` line rewritten to `sm_<host_sm>` (empty when it has none: not plain sm_90 PTX).
inline std::string retarget_ptx_text(const std::string& text, int host_sm) {
  for (size_t at = text.find(".target"); at != std::string::npos; at = text.find(".target", at + 7)) {
    const size_t eol = text.find('\n', at);
    const size_t sm = text.find("sm_90", at);
    if (sm == std::string::npos || (eol != std::string::npos && sm > eol)) continue;
    const size_t after = sm + 5;
    if (after < text.size() && (std::isalnum(static_cast<unsigned char>(text[after])) || text[after] == '_')) return {};   // sm_90a, sm_90f, sm_900: not plain sm_90
    return text.substr(0, sm) + "sm_" + std::to_string(host_sm) + text.substr(after);
  }
  return {};
}

// What a module holds, for the reason a module no image of which runs on the card stays in the simulator: "SASS sm_75, sm_90, PTX sm_90".
inline std::string describe_candidates(const std::vector<CardCandidate>& c) {
  std::string sass, ptx;
  for (const CardCandidate& x : c) {
    std::string& to = x.kind == CardCandidate::Cubin ? sass : ptx;
    to += (to.empty() ? "" : ", ") + std::string("sm_") + std::to_string(x.arch) + (x.suffix ? std::string(1, x.suffix) : std::string());
  }
  std::string out;
  if (!sass.empty()) out = "SASS " + sass;
  if (!ptx.empty()) out += (out.empty() ? "" : ", ") + std::string("PTX ") + ptx;
  return out.empty() ? "no image" : out;
}

// ---- the PTX the card is given ------------------------------------------------------------------------------------------------------------
// A module the card runs from PTX is compiled by the driver into a module of its own: its own copies of the module's variables (the simulator's are
// elsewhere, and two copies of one variable cannot both be the truth), its own handles for textures and surfaces, its own heap and printf buffer.
// So the card is given PTX only for a module that declares no variable, texture or surface, executes no texture or surface instruction, trap or
// breakpoint, makes no indirect call (a function pointer from memory is the simulator's index, not an address), and calls no external function
// but vprintf. A text scan, statement by statement (a statement ends at `;`, `{`, `}` or a line end), conservative: what it cannot place it refuses.
struct PtxScan {
  std::string refuse;      // why the card cannot be given this PTX for the simulator; empty: it can
  bool never = false;      // ... and the reason is class Never (an instruction that needs the simulator's tables, a trap, an indirect call)
  bool printf = false;     // it calls vprintf (admitted in turbo mode unless switched off)
  ClassMask classes = 0;   // the classes of its instructions (vgpu/host_gpu.hpp InsnClass bits), so that the turbo switches (TURBO_OFF, ATOMICS) apply to PTX as to SASS
};
inline PtxScan scan_ptx_for_card(const std::string& text, bool printf_off, bool retarget = false) {
  PtxScan out;
  const auto refuse = [&](std::string why, bool never) {
    if (out.refuse.empty()) {
      out.refuse = std::move(why);
      out.never = never;
    }
  };
  // The text without comments, cut into statements.
  std::vector<std::string> stmts;
  std::string cur;
  const auto push = [&] {
    size_t b = 0;
    while (b < cur.size() && std::isspace(static_cast<unsigned char>(cur[b]))) ++b;
    size_t e = cur.size();
    while (e > b && std::isspace(static_cast<unsigned char>(cur[e - 1]))) --e;
    if (e > b) stmts.push_back(cur.substr(b, e - b));
    cur.clear();
  };
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      while (i < text.size() && text[i] != '\n') ++i;
      push();
    } else if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      i += 2;
      while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/')) ++i;
      ++i;
      cur += ' ';
    } else if (c == ';' || c == '{' || c == '}' || c == '\n' || c == '\0') {
      push();
    } else {
      cur += c;
    }
  }
  push();
  const auto words = [](const std::string& s) {
    std::vector<std::string> w;
    std::string t;
    for (char c : s) {
      if (std::isspace(static_cast<unsigned char>(c)) || c == ',') {
        if (!t.empty()) w.push_back(t), t.clear();
      } else {
        t += c;
      }
    }
    if (!t.empty()) w.push_back(t);
    return w;
  };
  std::set<std::string> externs;
  for (const std::string& st : stmts) {   // the external functions the module declares
    const std::vector<std::string> w = words(st);
    if (w.empty()) continue;
    bool ext = false, fn = false;
    for (const std::string& t : w) {
      ext = ext || t == ".extern";
      fn = fn || t == ".func";
    }
    if (!ext || !fn) continue;
    size_t at = st.find(".func") + 5;
    while (at < st.size() && std::isspace(static_cast<unsigned char>(st[at]))) ++at;
    if (at < st.size() && st[at] == '(') {   // the return parameter list
      int depth = 0;
      for (; at < st.size(); ++at) {
        depth += st[at] == '(';
        depth -= st[at] == ')';
        if (depth == 0) { ++at; break; }
      }
    }
    while (at < st.size() && std::isspace(static_cast<unsigned char>(st[at]))) ++at;
    size_t end = at;
    while (end < st.size() && (std::isalnum(static_cast<unsigned char>(st[end])) || st[end] == '_' || st[end] == '$' || st[end] == '.')) ++end;
    if (end > at) externs.insert(st.substr(at, end - at));
  }
  for (const std::string& st : stmts) {
    std::vector<std::string> w = words(st);
    if (w.empty()) continue;
    size_t i = 0;
    if (w[i].back() == ':') ++i;                           // a label
    if (i < w.size() && w[i][0] == '@') ++i;               // a guard
    if (i >= w.size()) continue;
    const std::string op = w[i];
    if (retarget && op[0] == '.' && (op == ".reqnctapercluster" || op == ".maxclusterrank" || op == ".explicitcluster"))
      refuse("the PTX has the cluster directive " + op + ", which the card's architecture lacks", true);
    if (op[0] == '.') {   // a directive: a declaration of something the card's own module would keep a second copy of
      size_t k = i;
      while (k < w.size() && (w[k] == ".visible" || w[k] == ".extern" || w[k] == ".weak" || w[k] == ".common" || w[k] == ".hidden")) ++k;
      bool tex = false;   // (`.global .texref t;` is how a texture is declared: the texture is what it is, not a variable)
      for (size_t t = k; t < w.size(); ++t) tex = tex || w[t] == ".tex" || w[t] == ".texref" || w[t] == ".surf" || w[t] == ".surfref" || w[t] == ".samplerref";
      if (tex) refuse("the PTX declares a texture or surface", true);
      else if (k < w.size() && (w[k] == ".global" || w[k] == ".const")) refuse("the PTX declares a module variable (" + w[k] + ")", false);
      continue;
    }
    const std::string base = op.substr(0, op.find('.'));
    {   // the classes (the PTX spelling of vgpu/host_gpu.hpp turbo_class)
      const auto has = [&](const char* part) { return op.find(part) != std::string::npos; };
      const bool is_float = has(".f16") || has(".f32") || has(".f64") || has(".bf16");
      if (base == "atom" || base == "red") {
        if (has(".cas") || has(".exch")) out.classes |= class_bit(InsnClass::AtomSwap);
        else if (is_float) out.classes |= class_bit(base == "red" ? InsnClass::FloatRed : InsnClass::AtomFloat);
        else if (base == "atom") out.classes |= class_bit(InsnClass::AtomInt);
      } else if (base == "mma" || base == "wmma") {
        out.classes |= class_bit(InsnClass::Tensor);
      } else if (has(".approx") && (base == "sin" || base == "cos" || base == "ex2" || base == "lg2" || base == "rcp" || base == "rsqrt" || base == "sqrt" || base == "tanh" || base == "div")) {
        out.classes |= class_bit(InsnClass::Mufu);
      } else if ((has(".f16x2") || has(".bf16x2")) && (base == "add" || base == "sub" || base == "mul" || base == "fma" || base == "min" || base == "max" || base == "set" || base == "setp")) {
        out.classes |= class_bit(InsnClass::Half);
      }
      if (base == "call") out.classes |= class_bit(InsnClass::Call);
      // sm_90 and later forms (vgpu/host_gpu.hpp unmeasured_private_mnemonic): the tensor-map descriptor is the simulator's own, the rest has no card evidence
      if (op.rfind("cp.async.bulk", 0) == 0 || op.rfind("cp.reduce.async.bulk", 0) == 0 || op.rfind("prefetch.tensormap", 0) == 0 || op.rfind("tensormap.", 0) == 0 ||
          op.rfind("wgmma", 0) == 0 || base == "setmaxnreg" || base == "elect" || base == "stmatrix" || base == "griddepcontrol" || base == "multimem" || base == "mapa" ||
          base == "getctarank" || op.rfind("barrier.cluster", 0) == 0 || op.rfind("st.async", 0) == 0 || op.rfind("red.async", 0) == 0 || has(".cluster") || st.find("%cluster") != std::string::npos ||
          st.find("%is_explicit_cluster") != std::string::npos)
        refuse("the PTX uses an sm_90 or later form (" + op + ") that turbo does not send to the card", true);
    }
    if (retarget) {   // PTX for sm_90 compiled for an older card: each form ptxas rejects there (micro-tests of docs/perf-design/12-host-gpu-turbo.md section 15: "requires .target sm_90 or higher")
      const auto has = [&](const char* part) { return op.find(part) != std::string::npos; };
      const char* why = nullptr;
      if (op.rfind("mbarrier.try_wait", 0) == 0) why = "mbarrier.try_wait";
      else if (has("expect_tx") || has("complete_tx")) why = "mbarrier expect_tx/complete_tx";
      else if (op.rfind("mbarrier.arrive", 0) == 0 && w.size() - i - 1 >= 3) why = "mbarrier.arrive with a count";
      else if ((base == "atom" || base == "red") && (has(".v2") || has(".v4") || has(".v8") || has(".b128"))) why = "a vector or 128-bit atomic";
      else if (op.rfind("fence.proxy.async", 0) == 0) why = "fence.proxy.async";
      else if (st.find("%aggr_smem_size") != std::string::npos) why = "%aggr_smem_size";
      else if ((base == "max" || base == "min") && has(".relu")) why = "min/max with .relu";
      else if ((base == "add" || base == "sub" || base == "min" || base == "max") && (has(".s16x2") || has(".u16x2"))) why = "packed 16-bit integer arithmetic";
      else if (base == "cvt" && (has(".e4m3") || has(".e5m2") || has(".e2m1") || has(".ue8m0"))) why = "an fp8/fp4 convert";
      else if (base == "mma" && (has(".e4m3") || has(".e5m2"))) why = "an fp8 mma";
      if (why) refuse(std::string("the PTX uses ") + why + ", which the card's architecture lacks", true);
    }
    if (base == "tex" || base == "tld4" || base == "txq" || base == "suld" || base == "sust" || base == "sured" || base == "suq") refuse("the PTX has a texture or surface instruction (" + base + ")", true);
    else if (base == "trap" || base == "brkpt") refuse("the PTX has " + base + ", which ends the card's context", true);
    else if (base == "call") {
      size_t at = st.find("call") + 4;
      while (at < st.size() && (st[at] == '.' || std::isalpha(static_cast<unsigned char>(st[at])))) ++at;   // (.uni)
      while (at < st.size() && std::isspace(static_cast<unsigned char>(st[at]))) ++at;
      if (at < st.size() && st[at] == '(') {   // the return value
        int depth = 0;
        for (; at < st.size(); ++at) {
          depth += st[at] == '(';
          depth -= st[at] == ')';
          if (depth == 0) { ++at; break; }
        }
        while (at < st.size() && (std::isspace(static_cast<unsigned char>(st[at])) || st[at] == ',')) ++at;
      }
      size_t end = at;
      while (end < st.size() && !std::isspace(static_cast<unsigned char>(st[end])) && st[end] != ',' && st[end] != '(') ++end;
      const std::string target = st.substr(at, end - at);
      if (target.empty() || target[0] == '%') refuse("the PTX calls through a pointer", true);
      else if (externs.count(target)) {
        if (target == "vprintf" && !printf_off) out.printf = true;
        else refuse("the PTX calls " + target + (target == "vprintf" ? " (VGPU_HOST_GPU_TURBO_OFF=printf)" : ""), false);
      }
    }
  }
  return out;
}

// ---- does a launch fit the card? ----------------------------------------------------------------------------------------------------------
// What the driver says of the card (cuDeviceGetAttribute) and of a function (cuFuncGetAttribute). A launch the card cannot take fails there, and a
// failed launch ends the mode for the rest of the run, so the limits are checked first and the launch that exceeds one stays in the simulator.
struct CardLimits {
  uint32_t max_threads_per_block = 0;    // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
  uint32_t max_block[3] = {0, 0, 0};     // ..._MAX_BLOCK_DIM_X, Y, Z
  uint64_t max_grid[3] = {0, 0, 0};      // ..._MAX_GRID_DIM_X, Y, Z
  uint32_t shared_optin = 0;             // ..._MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: static + dynamic shared memory of one block
  uint32_t regs_per_block = 0;           // ..._MAX_REGISTERS_PER_BLOCK
  uint64_t memory_total = 0;             // cuMemGetInfo's total
  bool cooperative = false;              // ..._COOPERATIVE_LAUNCH
  bool known() const { return max_threads_per_block != 0 && shared_optin != 0 && regs_per_block != 0 && max_grid[0] != 0 && max_block[0] != 0; }
};
struct FuncAttrs {
  bool known = false;
  uint32_t max_threads = 0;      // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK (the registers and shared memory of the function folded in)
  uint32_t static_shared = 0;    // ..._SHARED_SIZE_BYTES
  uint32_t regs = 0;             // ..._NUM_REGS
};
enum class FitRefusal : uint8_t { None, Unknown, BlockDim, Threads, KernelThreads, Registers, Shared, GridDim, Memory, SharedAttribute, Cooperative, HostMemory };
inline const char* fit_refusal_text(FitRefusal r) {
  static const char* const n[] = {"", "the card's limits or the function's attributes are not known", "a block dimension over the card's limit",
                                  "more threads in a block than the card allows", "more threads in a block than the kernel can run (its registers or __launch_bounds__)",
                                  "more registers for the block than the card has", "more shared memory than the card allows a block", "a grid dimension over the card's limit",
                                  "the simulator's memory is larger than the card's", "the card refused the dynamic shared memory size",
                                  "a cooperative grid that the card cannot hold at once", "the card could not be given the mapped host memory at its own address"};
  return n[static_cast<unsigned>(r)];
}
// Whether a launch of a function with these attributes, this shape and `dynamic_shared` bytes of dynamic shared memory runs on a card with these
// limits, with `window_bytes` of simulated memory mirrored there. The local memory a thread's frame needs is not here: the driver allocates it when
// the kernel launches and only then can say that it does not fit (documented: cuFuncGetAttribute reports LOCAL_SIZE_BYTES, not whether the card has room).
inline FitRefusal fits_card(const CardLimits& card, const FuncAttrs& f, const uint32_t grid[3], const uint32_t block[3], uint64_t dynamic_shared, uint64_t window_bytes) {
  if (!card.known() || !f.known) return FitRefusal::Unknown;
  for (int i = 0; i < 3; ++i)
    if (block[i] > card.max_block[i]) return FitRefusal::BlockDim;
  const uint64_t threads = uint64_t{block[0]} * block[1] * block[2];
  if (threads > card.max_threads_per_block) return FitRefusal::Threads;
  if (threads > f.max_threads) return FitRefusal::KernelThreads;
  if (threads * f.regs > card.regs_per_block) return FitRefusal::Registers;
  if (uint64_t{f.static_shared} + dynamic_shared > card.shared_optin) return FitRefusal::Shared;
  for (int i = 0; i < 3; ++i)
    if (grid[i] > card.max_grid[i]) return FitRefusal::GridDim;
  if (card.memory_total && window_bytes > card.memory_total) return FitRefusal::Memory;
  return FitRefusal::None;
}

// A cooperative launch needs every block resident together: `per_sm` is the driver's occupancy of the kernel at this block size and shared memory.
inline bool coop_fits(uint64_t blocks, uint32_t per_sm, uint32_t sms) { return per_sm != 0 && blocks <= uint64_t{per_sm} * sms; }

// The room the mode leaves on the card when it backs the simulator's window (src/runtime/host_gpu.cpp ensure_mapped): the window must fit in the free
// memory with this much to spare for the driver's own allocations (the context, the module images, the printf buffer).
inline constexpr uint64_t kWindowReserve = 256ull << 20;
inline bool window_fits(uint64_t bytes, uint64_t free_bytes) { return bytes + kWindowReserve <= free_bytes; }

// ---- the switch of turbo mode's performance rules ----------------------------------------------------------------------------------------------
// VGPU_HOST_GPU_FIT=1 (the default in turbo mode): every kernel that passes the safety checks and fits the card goes to the card, whatever the
// simulator would have taken; =0 keeps the size rule, the sample of the first blocks, the first-launch rule, the start cost and the bans (the
// rules turbo had before). Unset or empty: the default. False for anything else (the caller says so and does not guess).
inline bool parse_fit(const char* text, bool* out) {
  if (!text || !*text) return true;
  const std::string t(text);
  if (t == "1") { *out = true; return true; }
  if (t == "0") { *out = false; return true; }
  return false;
}

// WSL2 cannot serve the mode's window: there `cuMemAddressReserve` at the simulator's address never returns (09-host-gpu-spike.md), so the driver start would wait for its whole
// 10 s deadline in every process. Turbo with FIT=1 starts the driver at the first launch, so it checks the host first: /proc/version of a WSL kernel names Microsoft or WSL.
inline bool host_is_wsl(const std::string& proc_version, bool dxg_present) {
  std::string v;
  for (char c : proc_version) v += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return dxg_present || v.find("microsoft") != std::string::npos || v.find("wsl") != std::string::npos;
}

// What turbo mode says once, when it is on (it is not the simulator's bits: docs/perf-design/12-host-gpu-turbo.md).
inline const char* turbo_notice() {
  return "host_gpu: VGPU_HOST_GPU=turbo runs the kernels it can on the host GPU, in the card's own arithmetic and order of operations: results may differ from "
         "the simulator's (floating-point approximations, the order of atomics and reductions); VGPU_HOST_GPU=exact keeps the simulator's bits";
}

}  // namespace vgpu::hostgpu
