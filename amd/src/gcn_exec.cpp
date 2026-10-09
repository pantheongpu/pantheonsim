#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_image.hpp"

#include <cfenv>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstring>
#include <algorithm>
#include <array>
#include <list>
#include <map>
#include <cstdlib>
#include <exception>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "vgpu/amd_debug.hpp"
#include "vgpu/amd_decode_cache.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vgpu/amd_hostcall.hpp"
#include "vgpu/error.hpp"
#include "vgpu/host_cpus.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace vgpu::amd {
namespace {

using gcn::Inst;
using gcn::Operand;
using gcn::OperandKind;

constexpr uint32_t kSgprs = 106;      // s0 through s101 and FLAT_SCRATCH (102, 103) on gfx9; s0 through s105 on RDNA
constexpr uint32_t kVgprs = 256;
constexpr uint32_t kLanes = 64;

// The GPU's real-time clock: a counter at a constant 100 MHz, which is what
// the runtime reports as the wall clock rate. Taken from the host's steady
// clock, so a kernel that waits on it for a stretch of time waits that long.
uint64_t realtime_ticks() {
  return static_cast<uint64_t>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                 .count()) /
         10;
}

// Where LDS sits in the one address space a flat access uses. The hardware
// puts it in an aperture the wave reads from src_shared_base; this model puts
// it below every device allocation (vgpu/memory.hpp starts those at
// 0x2000'0000'0000), so an address says for itself which memory it means.
constexpr uint64_t kSharedBase = 0x1000'0000'0000ull;
constexpr uint64_t kSharedSize = 1ull << 20;
// What a CDNA compute unit's LDS holds: no work-group has more.
constexpr uint64_t kLdsPerComputeUnit = 64 * 1024;
// The LDS a work-group may have: 64 KB on gfx942, 160 KB on gfx950 (CDNA4).
// Where the private segment buffer resource a gfx90a kernel is handed says
// its scratch is: not a device address, but the mark buffer accesses through
// it are recognised by, and sent to the work-items' private memory.
constexpr uint64_t kScratchResourceBase = 0xFFFF00000000ull;
// Where a work-group runs (Dispatch::Layout): its die, the engine on that
// die, the array in that engine, and the unit in that array.
struct Place {
  uint32_t die, engine, array, unit;
};
Place place_of(const Dispatch& d, uint64_t group) {
  const Dispatch::Layout& l = d.layout;
  const uint32_t dies = std::max(l.dies, 1u), engines = std::max(l.engines, 1u), arrays = std::max(l.arrays, 1u);
  const uint32_t u = static_cast<uint32_t>(group / dies % std::max(l.units, 1u));
  return {static_cast<uint32_t>(group % dies), u % engines, u / engines % arrays, u / (engines * arrays)};
}

uint64_t lds_limit(const Dispatch& d) { return d.object && d.object->gfx950() ? 160 * 1024 : kLdsPerComputeUnit; }
// And a work-item's private memory, which the wave reads the aperture of from
// src_private_base: an address in it is an offset into the work-item's own.
constexpr uint64_t kPrivateBase = 0x1800'0000'0000ull;
constexpr uint64_t kPrivateSize = 1ull << 32;

// An instruction's name, as the interpreter tells instructions apart: its
// text and a 64-bit FNV-1a hash of it, taken once per instruction. A literal
// written "v_add_f32_e32"_op has its hash taken by the compiler, so telling
// one instruction from another in a lane loop is an integer comparison rather
// than a string one -- which is where the interpreter used to spend most of
// its time. The text is compared as well whenever the hashes agree, so a
// collision could never make one instruction run as another.
constexpr uint64_t name_hash(std::string_view s) {
  uint64_t h = 1469598103934665603ull;
  for (char c : s) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ull;
  return h;
}
struct OpLit {
  uint64_t hash;
  std::string_view text;
};
consteval OpLit operator""_op(const char* s, size_t n) { return {name_hash({s, n}), {s, n}}; }
class OpName {
 public:
  explicit OpName(const std::string& s) : s_(s), hash_(name_hash(s)) {}
  bool operator==(const OpLit& l) const { return hash_ == l.hash && std::string_view(s_) == l.text; }
  bool operator!=(const OpLit& l) const { return !(*this == l); }
  operator const std::string&() const { return s_; }
  size_t size() const { return s_.size(); }
  size_t find(const char* t) const { return s_.find(t); }
  size_t rfind(const char* t, size_t at = std::string::npos) const { return s_.rfind(t, at); }
  size_t rfind(char c) const { return s_.rfind(c); }
  int compare(size_t at, size_t n, const char* t) const { return s_.compare(at, n, t); }
  std::string substr(size_t at, size_t n = std::string::npos) const { return s_.substr(at, n); }
  friend std::ostream& operator<<(std::ostream& o, const OpName& n) { return o << n.s_; }

 private:
  const std::string& s_;
  uint64_t hash_;
};

float as_float(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
uint32_t as_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}
double as_double(uint64_t bits) {
  double d;
  std::memcpy(&d, &bits, 8);
  return d;
}
uint64_t as_bits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, 8);
  return b;
}
_Float16 as_half(uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, 2);
  return h;
}
uint16_t as_bits(_Float16 h) {
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}

// The kinds of float v_cmp_class asks about, one bit each, in the ISA's order.
template <typename T>
bool matches_class(T f, uint32_t mask) {
  const bool negative = std::signbit(f);
  uint32_t bit = 0;
  if (std::isnan(f)) bit = 1u << 1;                       // a quiet NaN; nothing here signals
  else if (std::isinf(f)) bit = negative ? 1u << 2 : 1u << 9;
  else if (f == 0) bit = negative ? 1u << 5 : 1u << 6;
  else if (std::fpclassify(f) == FP_SUBNORMAL) bit = negative ? 1u << 4 : 1u << 7;
  else bit = negative ? 1u << 3 : 1u << 8;
  return (mask & bit) != 0;
}

// The same for a half, whose subnormals a float would hold as normal numbers.
bool matches_class_half(_Float16 h, uint32_t mask) {
  uint16_t bits;
  std::memcpy(&bits, &h, 2);
  const bool negative = bits >> 15, subnormal = (bits & 0x7C00) == 0 && (bits & 0x3FF) != 0;
  if (!subnormal) return matches_class(static_cast<float>(h), mask);
  return (mask & (negative ? 1u << 4 : 1u << 7)) != 0;
}

// One wavefront: its own scalar registers, VCC, EXEC and SCC, and 64 lanes of
// vector registers.
struct Wave {
  uint32_t sgpr[kSgprs] = {};
  // The trap handler's registers. RDNA4 gives a kernel its work-group's id in
  // two of them (TTMP9 x; TTMP7 y and z, a half each) rather than in scalar
  // registers after the user ones.
  uint32_t ttmp[16] = {};
  uint32_t vgpr[kVgprs][kLanes] = {};
  // The accumulation registers: a second bank a kernel keeps values in when
  // it has more of them than the vector registers hold.
  std::vector<std::array<uint32_t, kLanes>> agpr;
  uint64_t vcc = 0, exec = 0;
  uint32_t m0 = 0;   // a lane number, where an instruction takes one from it
  bool scc = false;
  uint64_t pc = 0;
  bool done = false;
  bool at_barrier = false;
  // How many global loads the wave has issued that no s_waitcnt has waited for yet (capped), and
  // whether it gave up its turn at such a wait (run_round). A load takes a card hundreds of cycles
  // and a barrier, an LDS read or a wave next to this one a few: the waves of a group that wait for
  // a load let the others on, so that a wave that read LDS after a barrier sees what it did before
  // the wave that went on to a load overwrote it.
  uint32_t loads_in_flight = 0;
  bool soft_yield = false;
  bool round_stopped = false;
  uint64_t round_steps = 0;
  // At a GWS barrier (Machine::gws_barrier): counted there, and waiting for
  // the barrier's generation to move past this one.
  bool gws_waiting = false;
  uint64_t gws_generation = 0;
  uint32_t first_lane = 0;   // this wave's first work-item in the group
  uint64_t group = 0;        // its work-group, numbered x fastest, then y, then z
  // How many lanes: 64 on CDNA, 32 for an RDNA kernel built wave32. A
  // wave32 wave is a wave64 whose upper half is never switched on, so an
  // instruction over "every lane" needs only its EXEC.
  uint32_t lanes = kLanes;
  // The MODE hardware register, as a kernel reads and sets it: what its
  // descriptor asks for (Kernel::mode) to begin with. FloatMode applies it.
  uint32_t mode = 0xF0 | 1u << 8 | 1u << 9;
  // gfx12's SCHED_MODE (hardware register 26): whether the hardware checks
  // instruction dependencies or leaves them to the compiler ("expert" mode,
  // which hipBLASLt's kernels set). Instructions here run one at a time in
  // order either way, so it is kept only to be read back.
  uint32_t sched_mode = 0;
  // gfx10's FLAT_SCRATCH (hardware registers 20 and 21, low and high): where
  // the wave's private memory is, which a kernel's prologue sets before it
  // reaches the stack through flat addresses. Each lane's private memory is
  // found here without it, so it is kept only to be read back.
  uint32_t flat_scratch[2] = {0, 0};
  // VGPR indexing (s_set_gpr_idx_on): which operands are offset -- source
  // 0, 1, 2 and the destination, a bit each -- by M0's low byte. 0 is off.
  uint8_t gpr_idx = 0;
};

// The MODE register's round and denormal modes, applied to the host's own
// floating point for the one instruction that runs under them: the round
// mode through <cfenv>, and the denormal mode through the host's flush
// controls -- on x86 MXCSR's DAZ (inputs read as zero) and FTZ (results
// flushed), which match MODE's two bits exactly; on AArch64 FPCR.FZ, which
// does both at once, for the mode that flushes both. A float instruction
// takes MODE's single-precision fields, a double one the 16/64-bit fields.
// A half's denormals are a float's normal numbers, so the host cannot flush
// them this way: halves keep their denormals whatever MODE says.
//
// Every kernel PyTorch's and ROCm's libraries ship asks for round to nearest
// even with denormals kept, which the host already does; this costs nothing
// then, and only code built to flush (-fgpu-flush-denormals-to-zero), or that
// sets MODE itself, pays for it.
class FloatMode {
 public:
  static constexpr uint32_t kDefault = 0xF0;   // round to nearest even, denormals kept
  static bool is_default(uint32_t mode) { return (mode & 0xFF) == kDefault; }

  FloatMode(uint32_t mode, bool wide) {
    const uint32_t round = wide ? (mode >> 2) & 3 : mode & 3, denorm = wide ? (mode >> 6) & 3 : (mode >> 4) & 3;
    // MODE's round modes: to nearest even, toward +infinity, toward
    // -infinity, toward zero.
    static constexpr int kRound[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
    saved_round_ = std::fegetround();
    if (round) std::fesetround(kRound[round]);
    // Denormal mode bit 0 keeps input denormals, bit 1 output ones.
#if defined(__x86_64__) || defined(__i386__)
    saved_csr_ = _mm_getcsr();
    _mm_setcsr((saved_csr_ & ~(1u << 6 | 1u << 15)) | ((denorm & 1) ? 0u : 1u << 6) | ((denorm & 2) ? 0u : 1u << 15));
#elif defined(__aarch64__)
    __asm__ volatile("mrs %0, fpcr" : "=r"(saved_fpcr_));
    const uint64_t fz = uint64_t{1} << 24;
    __asm__ volatile("msr fpcr, %0" ::"r"(denorm == 0 ? saved_fpcr_ | fz : saved_fpcr_ & ~fz));
#endif
  }
  ~FloatMode() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(saved_csr_);
#elif defined(__aarch64__)
    __asm__ volatile("msr fpcr, %0" ::"r"(saved_fpcr_));
#endif
    std::fesetround(saved_round_);
  }
  FloatMode(const FloatMode&) = delete;
  FloatMode& operator=(const FloatMode&) = delete;

 private:
  int saved_round_ = FE_TONEAREST;
#if defined(__x86_64__) || defined(__i386__)
  unsigned saved_csr_ = 0;
#elif defined(__aarch64__)
  uint64_t saved_fpcr_ = 0;
#endif
};

// The work-group the waves share: its LDS, and how many waves are still to
// reach the barrier.
struct Group {
  std::vector<uint8_t> lds;
  // Each work-item's private memory, one block per lane of the group: what a
  // kernel spills into when it runs out of registers.
  std::vector<uint8_t> scratch;
  uint32_t scratch_per_lane = 0;
  std::vector<Wave> waves;
  uint32_t id[3] = {};   // the work-group's id, for the debugger to say
};

struct Machine {
  Machine(const Dispatch& dispatch, MemoryManager& memory, DecodeCache& cache)
      : d(dispatch), mem(memory), decoded(&cache) {}
  const Dispatch& d;
  MemoryManager& mem;
  DispatchStats stats;
  // Work-groups run on several host threads at once, and device memory is
  // shared between them (vgpu/memory.hpp makes each word's load and store
  // atomic). A read-modify-write is not, so across threads each takes one of
  // a set of locks striped by address. On one thread there is no need: a
  // wave's lanes take their turns in order, which is atomic already.
  bool concurrent = false;
  // A vector load or store of device memory. ROCm runs CDNA with unaligned
  // access enabled (SH_MEM_CONFIG's unaligned mode), so a word may start at
  // any byte, and the compiler counts on it -- printf packs its string eight
  // bytes to a word from wherever the string starts. One that is aligned is a
  // single access; one that is not is its bytes. An atomic still has to be
  // aligned, as the hardware requires, and goes to memory directly.
  // Where an aligned word or pair starts inside an allocation and runs past
  // its end, how many of its bytes are the allocation's; 0 otherwise.
  // Compilers make an atomic on one or two bytes (a short, a half) as a
  // word-wide compare-and-swap on the aligned word that holds them, after a
  // word-wide load of it. A card, whose allocations are whole pages, takes
  // that; here the rest of the word reads as zero and is not written. Any
  // other access past the end is still refused.
  static uint32_t word_tail(const MemoryManager& m, uint64_t addr, uint32_t size) {
    if ((size != 4 && size != 8) || addr % size) return 0;
    uint64_t base = 0, bytes = 0;
    if (!m.find_allocation(addr, &base, &bytes)) return 0;
    const uint64_t left = base + bytes - addr;
    return left < size ? static_cast<uint32_t>(left) : 0;
  }
  uint64_t load(uint64_t addr, uint32_t size) const {
    MemoryManager& m = at(addr);
    if (addr % size == 0) {
      try {
        return m.load_scalar(addr, size);
      } catch (...) {
        const uint32_t valid = word_tail(m, addr, size);
        if (!valid) throw;
        uint64_t v = 0;   // byte by byte, each atomically, as every device access is
        for (uint32_t b = 0; b < valid; ++b) v |= m.load_scalar(addr + b, 1) << (8 * b);
        return v;
      }
    }
    uint64_t v = 0;
    for (uint32_t b = 0; b < size; ++b) v |= m.load_scalar(addr + b, 1) << (8 * b);
    return v;
  }
  void store(uint64_t addr, uint32_t size, uint64_t v) {
    MemoryManager& m = at(addr);
    if (addr % size == 0) return m.store_scalar(addr, size, v);
    for (uint32_t b = 0; b < size; ++b) m.store_scalar(addr + b, 1, (v >> (8 * b)) & 0xFF);
  }
  // Two to four words, as a wide load or store moves them: each aligned four
  // as one 16-byte access and each aligned pair as one 8-byte access, the way
  // the hardware moves them. Another kernel, on another host thread, reading
  // while this one writes sees each old or new, never half of each. RCCL's LL
  // protocol counts on it for eight bytes, a word of data and the flag that
  // says it has arrived; rocPRIM's decoupled look-back for sixteen, a tile's
  // flag and its 64-bit prefix. Moved in pieces, a reader could take the old
  // data with the new flag.
  static constexpr uint32_t kMaxWords = 16;
  // Four words at a 16-byte boundary as one access, where they are in one
  // device allocation; false, having done nothing, where they are not (the
  // pieces then say what is wrong).
  bool load_quad(uint64_t addr, uint32_t* out) const {
    uint64_t v[2];
    try {
      at(addr).load_quad(addr, v);
    } catch (const Error&) {
      return false;
    }
    std::memcpy(out, v, 16);
    return true;
  }
  bool store_quad(uint64_t addr, const uint32_t* words) {
    uint64_t v[2];
    std::memcpy(v, words, 16);
    try {
      at(addr).store_quad(addr, v);
    } catch (const Error&) {
      return false;
    }
    return true;
  }
  void load_words(uint64_t addr, uint32_t n, uint32_t* out) const {
    for (uint32_t k = 0; k < n;) {
      if (k + 3 < n && (addr + 4 * k) % 16 == 0 && load_quad(addr + 4 * k, out + k)) {
        k += 4;
      } else if (k + 1 < n && (addr + 4 * k) % 8 == 0) {
        const uint64_t v = load(addr + 4 * k, 8);
        out[k] = static_cast<uint32_t>(v), out[k + 1] = static_cast<uint32_t>(v >> 32);
        k += 2;
      } else {
        out[k] = static_cast<uint32_t>(load(addr + 4 * k, 4));
        k += 1;
      }
    }
  }
  void store_words(uint64_t addr, uint32_t n, const uint32_t* words) {
    for (uint32_t k = 0; k < n;) {
      if (k + 3 < n && (addr + 4 * k) % 16 == 0 && store_quad(addr + 4 * k, words + k)) {
        k += 4;
      } else if (k + 1 < n && (addr + 4 * k) % 8 == 0) {
        store(addr + 4 * k, 8, words[k] | uint64_t{words[k + 1]} << 32);
        k += 2;
      } else {
        store(addr + 4 * k, 4, words[k]);
        k += 1;
      }
    }
  }
  // The memory an address is in: the launching device's, or a peer's the
  // kernel was given access to -- each device's window is where its ordinal
  // puts it (vgpu/memory.hpp). An address in neither goes to the device's
  // own memory, which says what is wrong with it.
  MemoryManager& at(uint64_t addr) const {
    if (addr - mem.va_base() < kDeviceVaStride || d.peers.empty() || !is_device_va(addr)) return mem;
    const uint64_t window = (addr - kDeviceVaBase) / kDeviceVaStride;
    return window < d.peers.size() && d.peers[window] ? *d.peers[window] : mem;
  }
  std::unique_lock<std::mutex> atomic_guard(uint64_t addr) {
    return concurrent ? std::unique_lock<std::mutex>(memory_atomic_lock(addr)) : std::unique_lock<std::mutex>();
  }
  // Each instruction decoded once, the first time any wave reaches it: every
  // wave of every launch of a module runs the same code, and decoding it again
  // each time cost more than running it. The cache is the module's, or the
  // dispatch's where the runtime keeps none.
  DecodeCache* decoded = nullptr;

  gcn::Target target() const { return gcn::target_of_mach(d.object->mach); }
  // An RDNA kernel built for 64-lane waves: its lane masks are register pairs.
  bool wave64() const { return gcn::is_rdna(target()) && d.kernel && !d.kernel->wave32; }
  const Inst& fetch(uint64_t pc) {
    const CodeObject& o = *d.object;
    const uint64_t at = pc - d.code_base - o.text_addr;
    const Inst* in = at % 4 == 0 && at < o.text.size()
                         ? decoded->get(at / 4, [&] { return gcn::decode(o.text, at, pc, target(), wave64()); })
                         : nullptr;
    if (!in) return *(scratch_inst = std::make_unique<const Inst>(gcn::decode(o.text, at, pc, target(), wave64())));
    return *in;
  }
  std::unique_ptr<const Inst> scratch_inst;   // one that is not where an instruction starts
  // The mask a VOP3b instruction writes to its scalar pair (a carry out, or
  // which lanes v_div_scale scaled), gathered lane by lane and written once
  // every lane has run: the pair may be one the instruction reads, and every
  // lane reads it as it was. A lane switched off leaves its bit zero.
  uint64_t sdst_bits = 0;

  // An error from the instruction at pc, saying which one: the kernel, how
  // far into it, and the instruction as the assembler writes it.
  Error at_instruction(const Error& e, uint64_t pc) {
    std::string name = d.kernel ? d.kernel->name : std::string("the kernel");
    if (name.size() > 60) name = name.substr(0, 57) + "...";
    std::string text;
    try {
      text = ": " + gcn::to_text(fetch(pc));
    } catch (const std::exception&) {
    }
    const uint64_t entry = d.code_base + (d.kernel ? d.kernel->entry : 0);
    // Code before the kernel's entry is a function it called (unoptimized code
    // calls the device library's helpers rather than inlining them), and a
    // distance back from the entry would wrap into nonsense.
    if (pc < entry)
      return Error::make(e.code(), e.message(), " (in a function ", name, " called, at +0x", std::hex,
                         pc - d.code_base, std::dec, " in the code object", text, ")");
    return Error::make(e.code(), e.message(), " (", name, " +0x", std::hex, pc - entry, std::dec, text, ")");
  }

  // ---- Register access ----------------------------------------------------

  uint32_t sgpr(const Wave& w, uint32_t i) const {
    if (i >= kSgprs) throw Error::make(Err::Internal, "scalar register ", i, " is past the file");
    return w.sgpr[i];
  }
  void set_sgpr(Wave& w, uint32_t i, uint32_t v) {
    if (i >= kSgprs) throw Error::make(Err::Internal, "scalar register ", i, " is past the file");
    w.sgpr[i] = v;
  }
  uint64_t sgpr64(const Wave& w, uint32_t i) const { return sgpr(w, i) | static_cast<uint64_t>(sgpr(w, i + 1)) << 32; }
  // A memory instruction's scalar base or offset: a scalar register, or one
  // of the special ones the same field can name (VCC most often, which a
  // kernel short of scalar registers keeps an address in).
  uint64_t scalar_field(const Wave& w, uint32_t field, bool wide) const {
    if (field == 106) return wide ? w.vcc : static_cast<uint32_t>(w.vcc);
    if (field == 107) return static_cast<uint32_t>(w.vcc >> 32);
    if (field == 124) return w.m0;
    if (field == 126) return wide ? w.exec : static_cast<uint32_t>(w.exec);
    if (field == 127) return static_cast<uint32_t>(w.exec >> 32);
    return wide ? sgpr64(w, field) : sgpr(w, field);
  }
  void set_sgpr64(Wave& w, uint32_t i, uint64_t v) {
    set_sgpr(w, i, static_cast<uint32_t>(v));
    set_sgpr(w, i + 1, static_cast<uint32_t>(v >> 32));
  }

  // A scalar operand's value: a register, a special register, or a constant.
  uint64_t scalar(const Wave& w, const Operand& o) const {
    switch (o.kind) {
      case OperandKind::Sgpr: return o.width >= 2 ? sgpr64(w, o.index) : sgpr(w, o.index);
      case OperandKind::Vcc: return o.width >= 2 ? w.vcc : static_cast<uint32_t>(w.vcc);
      case OperandKind::Exec: return w.exec;
      case OperandKind::ExecLo: return static_cast<uint32_t>(w.exec);
      case OperandKind::ExecHi: return static_cast<uint32_t>(w.exec >> 32);
      case OperandKind::VccHi: return static_cast<uint32_t>(w.vcc >> 32);
      // An inline float constant is the number itself: a float's bits in a
      // 32-bit operand, a double's in a 64-bit one (s_mov_b64 s[6:7], 1.0 is
      // the double 1.0, not a float's bits with zeroes above them).
      case OperandKind::InlineFloat:
        return o.width >= 2 ? as_bits(o.fvalue) : as_bits(static_cast<float>(o.fvalue));
      // An aperture's base as a pair is the address; as one register, the
      // high half of it, which is what a kernel puts above an offset.
      case OperandKind::SharedBase: return o.width >= 2 ? kSharedBase : kSharedBase >> 32;
      case OperandKind::PrivateBase: return o.width >= 2 ? kPrivateBase : kPrivateBase >> 32;
      case OperandKind::Inline:
      case OperandKind::Literal: return static_cast<uint64_t>(o.value);
      case OperandKind::M0: return w.m0;
      // RDNA's null register reads zero, and SCC reads as 0 or 1.
      case OperandKind::Null: return 0;
      case OperandKind::Scc: return w.scc ? 1 : 0;
      case OperandKind::Ttmp:
        if (o.index + o.width > 16) break;
        return o.width >= 2 ? w.ttmp[o.index] | static_cast<uint64_t>(w.ttmp[o.index + 1]) << 32 : w.ttmp[o.index];
      case OperandKind::SharedLimit:
      case OperandKind::PrivateLimit:
        throw Error::make(Err::Unsupported, "reading ", operand_text(o), ", which this does not model");
      case OperandKind::Vgpr:
      case OperandKind::Agpr:
      case OperandKind::None: break;
    }
    throw Error::make(Err::Internal, "a scalar operand this does not read");
  }
  void write_scalar(Wave& w, const Operand& o, uint64_t v) {
    switch (o.kind) {
      case OperandKind::Sgpr:
        if (o.width >= 2) set_sgpr64(w, o.index, v);
        else set_sgpr(w, o.index, static_cast<uint32_t>(v));
        return;
      // A 32-bit write to VCC is to its low half (vcc_lo), and leaves the
      // high half as it was.
      case OperandKind::Vcc: w.vcc = o.width >= 2 ? v : (w.vcc & ~0xFFFFFFFFull) | static_cast<uint32_t>(v); return;
      case OperandKind::VccHi: w.vcc = (w.vcc & 0xFFFFFFFFull) | (v << 32); return;
      case OperandKind::Exec: w.exec = v; return;
      case OperandKind::ExecLo: w.exec = (w.exec & ~0xFFFFFFFFull) | static_cast<uint32_t>(v); return;
      case OperandKind::ExecHi: w.exec = (w.exec & 0xFFFFFFFFull) | (v << 32); return;
      case OperandKind::M0: w.m0 = static_cast<uint32_t>(v); return;
      case OperandKind::Null: return;   // RDNA's null register: the write goes nowhere
      case OperandKind::Ttmp:
        if (o.index + o.width > 16) break;
        w.ttmp[o.index] = static_cast<uint32_t>(v);
        if (o.width >= 2) w.ttmp[o.index + 1] = static_cast<uint32_t>(v >> 32);
        return;
      default: break;
    }
    throw Error::make(Err::Internal, "a scalar destination this does not write");
  }

  // ---- Reading a lane of another lane's register ---------------------------
  //
  // Set while a cross-lane instruction runs: the operand whose lanes were
  // shuffled, and what each lane's copy of it came to.
  const Operand* dpp_operand = nullptr;
  const std::array<uint32_t, kLanes>* dpp_values = nullptr;

  // A source as one lane sees it: a vector register's lane, or the same
  // scalar value for every lane.
  uint32_t lane_src(const Wave& w, const Operand& o, uint32_t lane) const {
    if (&o == dpp_operand) return (*dpp_values)[lane];
    if (o.kind == OperandKind::Agpr)
      return o.index < w.agpr.size() ? w.agpr[o.index][lane] : 0;   // one never written holds nothing
    const uint32_t v = o.kind == OperandKind::Vgpr ? w.vgpr[o.index][lane] : static_cast<uint32_t>(scalar(w, o));
    return o.sel == 6 ? v : selected(v, o.sel, o.sext);
  }
  // The part of a register a sub-dword instruction reads: a byte or a half of
  // it, taken into 32 bits with its sign or without.
  static uint32_t selected(uint32_t v, uint8_t sel, bool sext) {
    if (sel <= 3) {
      const uint8_t byte = static_cast<uint8_t>(v >> (8 * sel));
      return sext ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(byte))) : byte;
    }
    if (sel <= 5) {
      const uint16_t half = static_cast<uint16_t>(sel == 4 ? v : v >> 16);
      return sext ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(half))) : half;
    }
    return v;
  }
  uint64_t lane_src64(const Wave& w, const Operand& o, uint32_t lane) const {
    if (o.kind == OperandKind::Vgpr)
      return w.vgpr[o.index][lane] | static_cast<uint64_t>(w.vgpr[o.index + 1][lane]) << 32;
    // An inline constant is the number itself, so in a 64-bit instruction it
    // is that number as a double, not a float's bits with something above them.
    if (o.kind == OperandKind::InlineFloat) return as_bits(o.fvalue);
    if (o.kind == OperandKind::Literal && o.literal_high) return static_cast<uint64_t>(o.value) << 32;
    return scalar(w, o);
  }
  // A source read as a half: the low half of the register, or an inline
  // constant, which is the number itself rather than a float's bits.
  _Float16 lane_half(const Wave& w, const Operand& o, uint32_t lane) const {
    _Float16 h;
    if (o.kind == OperandKind::InlineFloat) {
      h = static_cast<_Float16>(o.fvalue);
    } else {
      const uint16_t bits = static_cast<uint16_t>(lane_src(w, o, lane));
      std::memcpy(&h, &bits, 2);
    }
    if (o.abs) h = h < static_cast<_Float16>(0) ? -h : h;
    return o.neg ? -h : h;
  }
  // A float narrowed to a half, rounded toward zero rather than to nearest:
  // the nearest half, stepped one toward zero where it landed further out
  // than the float was. A float past the largest half becomes the largest
  // half, not an infinity, since rounding toward zero never grows.
  static uint16_t half_toward_zero(float x) {
    const _Float16 nearest = static_cast<_Float16>(x);
    uint16_t bits = 0;
    std::memcpy(&bits, &nearest, 2);
    if (std::isnan(x)) return bits;
    if (std::fabs(static_cast<double>(nearest)) > std::fabs(static_cast<double>(x))) --bits;   // sign and magnitude: one step toward zero
    return bits;
  }

  // The small floats. gfx942's 8-bit ones are the forms without infinities
  // or a negative zero ("FNUZ"): fp8 with four exponent bits (bias 8) and
  // three of mantissa, bf8 with five (bias 16) and two; 0x80 is the one NaN,
  // and 0x7f the largest value. gfx950's are the OCP formats every other
  // vendor has: E4M3 (bias 7, largest 448; S.1111.111 its NaN, no infinity)
  // and E5M2 (bias 15, IEEE-like: an all-ones exponent is infinity or NaN).
  // gfx950's matrix instructions also take 6- and 4-bit floats, all finite:
  // FP6 E2M3, BF6 E3M2 and FP4 E2M1. The rules are those of HIP's software
  // conversion (amd_hip_fp8.h), which is how AMD says the hardware rounds.
  struct F8 {
    int mant, bias;
    double max;
    enum Style { Fnuz, OcpE4M3, OcpE5M2, Finite } style;
    int bits;
  };
  static constexpr F8 kFp8{3, 8, 240.0, F8::Fnuz, 8}, kBf8{2, 16, 57344.0, F8::Fnuz, 8},
      kOcpFp8{3, 7, 448.0, F8::OcpE4M3, 8}, kOcpBf8{2, 15, 57344.0, F8::OcpE5M2, 8}, kFp6{3, 1, 7.5, F8::Finite, 6},
      kBf6{2, 3, 28.0, F8::Finite, 6}, kFp4{1, 1, 6.0, F8::Finite, 4};
  // The 8-bit formats the code's processor has.
  // gfx950 and RDNA4 use the OCP formats; gfx942's are FNUZ.
  bool ocp8() const { return d.object->gfx950() || gcn::is_gfx12(target()); }
  const F8& fp8() const { return ocp8() ? kOcpFp8 : kFp8; }
  const F8& bf8() const { return ocp8() ? kOcpBf8 : kBf8; }
  static float f8_to_float(uint32_t v, const F8& t) {
    const uint32_t sign_bit = 1u << (t.bits - 1);
    v &= (sign_bit << 1) - 1;
    const uint32_t mag = v & (sign_bit - 1);
    if (t.style == F8::Fnuz && v == sign_bit) return std::numeric_limits<float>::quiet_NaN();
    if (t.style == F8::OcpE4M3 && mag == 0x7F) return std::numeric_limits<float>::quiet_NaN();
    if (t.style == F8::OcpE5M2 && (mag >> 2) == 0x1F) {
      if (mag & 3) return std::numeric_limits<float>::quiet_NaN();
      return v & sign_bit ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }
    const int e = static_cast<int>(mag) >> t.mant, m = static_cast<int>(mag) & ((1 << t.mant) - 1);
    const float x = e == 0 ? std::ldexp(static_cast<float>(m), 1 - t.bias - t.mant)
                           : std::ldexp(static_cast<float>((1 << t.mant) | m), e - t.bias - t.mant);
    return v & sign_bit ? -x : x;
  }
  // A float narrowed to one, rounded to nearest with ties to even -- or,
  // given random bits, stochastically: the bits the narrowing drops (the
  // float's 23 - mantissa lowest, counted where the small float's last
  // mantissa bit falls) have the random ones added before they are cut
  // off. A float past the largest value, before any rounding (so 241 is
  // past FNUZ fp8's 240), becomes the NaN (FNUZ, E4M3) or infinity (E5M2),
  // or with saturate the largest value. An infinity becomes the NaN, but
  // stays one in E5M2.
  static uint32_t float_to_f8(float x, const F8& t, bool saturate, const uint32_t* random) {
    const uint32_t sign_bit = 1u << (t.bits - 1), sign = std::signbit(x) ? sign_bit : 0;
    const uint32_t top = sign_bit - 1;                                // the largest magnitude's code
    const uint32_t nan = t.style == F8::Fnuz ? sign_bit : sign | 0x7F;
    const uint32_t largest = t.style == F8::OcpE4M3 ? 0x7E : t.style == F8::OcpE5M2 ? 0x7B : top;
    if (t.style == F8::Finite && (std::isnan(x) || std::isinf(x))) return sign | largest;
    if (std::isnan(x)) return nan;
    if (std::isinf(x)) return t.style == F8::OcpE5M2 ? sign | 0x7C : nan;
    const double a = std::fabs(static_cast<double>(x));
    if (a > t.max) {
      if (saturate || t.style == F8::Finite) return sign | largest;
      return t.style == F8::OcpE5M2 ? sign | 0x7C : nan;
    }
    const uint32_t zero = t.style == F8::Fnuz ? 0 : sign;   // OCP keeps a negative zero
    if (a == 0) return zero;
    const int emin = 1 - t.bias;
    const int e = std::max(std::ilogb(a), emin);
    const double step = std::ldexp(1.0, e - t.mant);   // between neighbouring values near a
    double n = a / step;                                 // exact: a whole number of steps and a fraction
    if (random) {
      const int drop = 23 - t.mant;
      const double scale = std::ldexp(1.0, drop);
      n = std::floor((std::floor(n * scale) + static_cast<double>(*random & ((1u << drop) - 1))) / scale);
    } else {
      n = std::nearbyint(n);
    }
    const double v = n * step;
    if (v == 0) return zero;
    if (v < std::ldexp(1.0, emin)) return sign | static_cast<uint32_t>(n);   // below the least normal
    const int ve = std::ilogb(v);
    const uint32_t m = static_cast<uint32_t>(std::ldexp(v, t.mant - ve)) - (1u << t.mant);
    return sign | static_cast<uint32_t>(ve + t.bias) << t.mant | m;
  }

  // A half result, which fills the low half of the register and zeroes the
  // high half, as every 16-bit instruction here does.
  void write_half(Wave& w, const Inst& in, uint32_t lane, _Float16 v) {
    uint16_t bits = 0;
    std::memcpy(&bits, &v, 2);
    write_lane(w, in.dst[0], lane, bits);
  }

  // A packed instruction's source k, for its low result (half 0) or its high
  // one: which part of the source is op_sel's bit or op_sel_hi's, and whether
  // it is negated neg_lo's or neg_hi's. A constant has no second part to
  // give: the compiler asks for its first one for both results, and a
  // program that asked for the other would get what this cannot say.
  uint32_t packed_word(const Wave& w, const Inst& in, uint32_t k, uint32_t top, uint32_t lane) const {
    const Operand& o = in.src[k];
    switch (o.kind) {
      case OperandKind::Vgpr:
      case OperandKind::Agpr: return word(const_cast<Wave&>(w), o, top, lane);
      case OperandKind::Inline:
      case OperandKind::InlineFloat:
      case OperandKind::Literal:
        if (top)
          throw Error::make(Err::Unsupported, in.name, " takes the second part of a constant, which this does not model");
        // Each half of a packed float is a float: an inline float constant
        // is a float's bits here, though the operand is a register pair.
        if (o.kind == OperandKind::InlineFloat) return as_bits(static_cast<float>(o.fvalue));
        return static_cast<uint32_t>(scalar(w, o));
      default: return static_cast<uint32_t>(scalar(w, o) >> (32 * top));
    }
  }
  float packed_float(const Wave& w, const Inst& in, uint32_t k, uint32_t half, uint32_t lane) const {
    const uint32_t top = ((half ? in.op_sel_hi : in.op_sel) >> k) & 1;
    const float f = as_float(packed_word(w, in, k, top, lane));
    return ((half ? in.neg_hi : in.neg_lo) >> k) & 1 ? -f : f;
  }
  // Half `top` of a constant a packed 16-bit instruction reads. As LLVM
  // records the hardware doing it (AMDGPUBaseInfo, getInlineEncodingV216):
  // an integer constant is its sign-extended 32 bits, so its high half is
  // their top; a float constant is the half in the low 16 bits with zero
  // above. The compiler asks for a constant in both halves by pointing the
  // high result at the low half. A literal (RDNA's; gfx9's packed
  // instructions take none) is the whole pair: its top 16 bits are the high
  // half, as LLVM encodes a packed constant (3.0 in both halves is
  // 0x42004200).
  static uint16_t constant_half(const Inst&, const Operand& o, uint32_t top) {
    if (o.kind == OperandKind::InlineFloat) return top ? 0 : as_bits(static_cast<_Float16>(o.fvalue));
    return static_cast<uint16_t>(static_cast<uint32_t>(o.value) >> (16 * top));
  }
  // Source k's 16 bits for the low result (half 0) or the high one, as the
  // packed integer instructions read them.
  uint16_t packed_bits(const Wave& w, const Inst& in, uint32_t k, uint32_t half, uint32_t lane) const {
    const Operand& o = in.src[k];
    const uint32_t top = ((half ? in.op_sel_hi : in.op_sel) >> k) & 1;
    if (o.kind == OperandKind::Inline || o.kind == OperandKind::InlineFloat || o.kind == OperandKind::Literal)
      return constant_half(in, o, top);
    return static_cast<uint16_t>(lane_src(w, o, lane) >> (16 * top));
  }
  float packed_half(const Wave& w, const Inst& in, uint32_t k, uint32_t half, uint32_t lane) const {
    const Operand& o = in.src[k];
    const uint32_t top = ((half ? in.op_sel_hi : in.op_sel) >> k) & 1;
    uint16_t bits;
    if (o.kind == OperandKind::Inline || o.kind == OperandKind::InlineFloat || o.kind == OperandKind::Literal) {
      bits = constant_half(in, o, top);
    } else {
      bits = static_cast<uint16_t>(lane_src(w, o, lane) >> (16 * top));
    }
    const float f = static_cast<float>(as_half(bits));
    return ((half ? in.neg_hi : in.neg_lo) >> k) & 1 ? -f : f;
  }

  // A source read as a float, with the modifiers a VOP3 source carries: the
  // absolute value first, then the negation, as the ISA applies them.
  // A source's 32 bits with its modifiers applied as a float's are: the
  // absolute value clears the sign bit, the negation flips it. What an
  // instruction that moves bits rather than doing arithmetic (v_cndmask)
  // makes of them.
  uint32_t lane_bits(const Wave& w, const Operand& o, uint32_t lane) const {
    uint32_t v = lane_src(w, o, lane);
    if (o.abs) v &= 0x7FFFFFFFu;
    if (o.neg) v ^= 0x80000000u;
    return v;
  }
  float lane_float(const Wave& w, const Operand& o, uint32_t lane) const {
    float f = as_float(lane_src(w, o, lane));
    if (o.abs) f = std::fabs(f);
    return o.neg ? -f : f;
  }
  double lane_double(const Wave& w, const Operand& o, uint32_t lane) const {
    double d = as_double(lane_src64(w, o, lane));
    if (o.abs) d = std::fabs(d);
    return o.neg ? -d : d;
  }
  // A float result, held to [0, 1] where the instruction asked for it.
  void write_float(Wave& w, const Inst& in, uint32_t lane, float v) {
    // The output multiplier, then the clamp, as the ISA orders them.
    if (in.omod) v *= in.omod == 1 ? 2.0f : in.omod == 2 ? 4.0f : 0.5f;
    if (in.clamp) v = std::isnan(v) ? 0.0f : std::fmin(1.0f, std::fmax(0.0f, v));
    write_lane(w, in.dst[0], lane, as_bits(v));
  }
  void write_double(Wave& w, const Inst& in, uint32_t lane, double v) {
    if (in.omod) v *= in.omod == 1 ? 2.0 : in.omod == 2 ? 4.0 : 0.5;
    if (in.clamp) v = std::isnan(v) ? 0.0 : std::fmin(1.0, std::fmax(0.0, v));
    write_lane64(w, in.dst[0], lane, as_bits(v));
  }
  // An accumulation register is written the first time a kernel uses one, so
  // the bank grows to what the kernel actually asked for rather than always
  // being as large as it could be.
  static std::array<uint32_t, kLanes>& acc(Wave& w, uint32_t index) {
    if (index >= kVgprs)
      throw Error::make(Err::InvalidValue, "an accumulation register numbered ", index, " is past the ", kVgprs,
                        " a wave has");
    if (w.agpr.size() <= index) w.agpr.resize(index + 1);
    return w.agpr[index];
  }
  // Set while a sub-dword instruction runs, when it writes part of its
  // destination rather than all of it.
  const Operand* narrow_dst = nullptr;
  uint8_t narrow_dst_sel = 6;
  bool narrow_dst_preserve = false;
  bool narrow_dst_sext = false;

  void write_lane(Wave& w, const Operand& o, uint32_t lane, uint32_t v) {
    // Where the instruction named part of the destination, the result's low
    // bits go there and the rest of the register is zeroed, which is what
    // padding the unused part means.
    if (&o == narrow_dst) {
      const uint8_t sel = narrow_dst_sel;
      const uint32_t mask = sel <= 3 ? 0xFFu << (8 * sel) : sel == 4 ? 0xFFFFu : sel == 5 ? 0xFFFF0000u : ~0u;
      v = sel <= 3   ? (v & 0xFFu) << (8 * sel)
          : sel == 4 ? v & 0xFFFFu
          : sel == 5 ? (v & 0xFFFFu) << 16
                     : v;
      // Or the bits above the part take its sign (those below it stay zero),
      // or the rest of the register is kept, where the instruction says so.
      if (narrow_dst_sext && sel < 6) {
        const uint32_t top = mask & ~(mask >> 1);   // the part's highest bit
        if (v & top) v |= ~(mask | (top - 1));
      }
      if (narrow_dst_preserve) {
        const uint32_t was = o.kind == OperandKind::Agpr ? acc(w, o.index)[lane] : w.vgpr[o.index][lane];
        v = (was & ~mask) | (v & mask);
      }
    }
    if (o.kind == OperandKind::Agpr) acc(w, o.index)[lane] = v;
    else w.vgpr[o.index][lane] = v;
  }
  void write_lane64(Wave& w, const Operand& o, uint32_t lane, uint64_t v) {
    set_word(w, o, 0, lane, static_cast<uint32_t>(v));
    set_word(w, o, 1, lane, static_cast<uint32_t>(v >> 32));
  }
  // Register k of an operand that covers several, in whichever bank it names:
  // memory instructions move vector or accumulation registers alike.
  static uint32_t word(Wave& w, const Operand& o, uint32_t k, uint32_t lane) {
    return o.kind == OperandKind::Agpr ? acc(w, o.index + k)[lane] : w.vgpr[o.index + k][lane];
  }
  static void set_word(Wave& w, const Operand& o, uint32_t k, uint32_t lane, uint32_t v) {
    if (o.kind == OperandKind::Agpr) acc(w, o.index + k)[lane] = v;
    else w.vgpr[o.index + k][lane] = v;
  }

  // ---- The instructions ---------------------------------------------------

  // A half operand of a scalar float instruction: a register's low 16 bits,
  // or a constant's. An inline float constant is the half's own encoding
  // (2.0 is 0x4000), as the vector unit reads one for a 16-bit instruction,
  // not the float's bits, whose low half is zero.
  _Float16 scalar_half(Wave& w, const Operand& o) {
    if (o.kind == OperandKind::InlineFloat) return static_cast<_Float16>(o.fvalue);
    const uint16_t bits = static_cast<uint16_t>(scalar(w, o));
    _Float16 r;
    std::memcpy(&r, &bits, 2);
    return r;
  }

  // RDNA's scalar float instructions (gfx11.5 and gfx12): a float or a half
  // in a scalar register, IEEE arithmetic as the vector unit does it.
  // Returns whether `op` was one.
  bool scalar_float(Wave& w, const Inst& in, const OpName& op, uint64_t a, uint64_t b) {
    const float x = as_float(static_cast<uint32_t>(a)), y = as_float(static_cast<uint32_t>(b));
    const auto put = [&](float v) { write_scalar(w, in.dst[0], as_bits(v)); };
    const auto h = [](uint64_t v) {
      _Float16 r;
      const uint16_t bits = static_cast<uint16_t>(v);
      std::memcpy(&r, &bits, 2);
      return r;
    };
    const auto put_h = [&](_Float16 v) {
      uint16_t bits;
      std::memcpy(&bits, &v, 2);
      write_scalar(w, in.dst[0], bits);
    };
    // Source k of a half instruction (scalar_half: an inline float constant
    // is the half's own encoding).
    const auto hs = [&](size_t k) { return scalar_half(w, in.src[k]); };
    if (op == "s_add_f32"_op) put(x + y);
    else if (op == "s_sub_f32"_op) put(x - y);
    else if (op == "s_mul_f32"_op) put(x * y);
    else if (op == "s_min_num_f32"_op || op == "s_min_f32"_op) put(std::fmin(x, y));
    else if (op == "s_max_num_f32"_op || op == "s_max_f32"_op) put(std::fmax(x, y));
    else if (op == "s_fmac_f32"_op) put(std::fma(x, y, as_float(static_cast<uint32_t>(scalar(w, in.dst[0])))));
    else if (op == "s_fmaak_f32"_op) put(std::fma(x, y, as_float(static_cast<uint32_t>(scalar(w, in.src[2])))));
    else if (op == "s_fmamk_f32"_op) put(std::fma(x, y, as_float(static_cast<uint32_t>(scalar(w, in.src[2])))));
    else if (op == "s_cvt_f32_i32"_op) put(static_cast<float>(static_cast<int32_t>(a)));
    else if (op == "s_cvt_f32_u32"_op) put(static_cast<float>(static_cast<uint32_t>(a)));
    else if (op == "s_cvt_i32_f32"_op)
      write_scalar(w, in.dst[0], static_cast<uint32_t>(std::isnan(x) ? 0 : x <= -2147483648.0f ? INT32_MIN
                                                     : x >= 2147483648.0f ? INT32_MAX : static_cast<int32_t>(x)));
    else if (op == "s_cvt_u32_f32"_op)
      write_scalar(w, in.dst[0], std::isnan(x) || x <= 0 ? 0u : x >= 4294967296.0f ? 0xFFFFFFFFu : static_cast<uint32_t>(x));
    else if (op == "s_cvt_f16_f32"_op) put_h(static_cast<_Float16>(x));
    else if (op == "s_cvt_f32_f16"_op) put(static_cast<float>(hs(0)));
    else if (op == "s_cvt_hi_f32_f16"_op) put(static_cast<float>(h(a >> 16)));
    else if (op == "s_ceil_f32"_op) put(std::ceil(x));
    else if (op == "s_floor_f32"_op) put(std::floor(x));
    else if (op == "s_trunc_f32"_op) put(std::trunc(x));
    else if (op == "s_rndne_f32"_op) put(std::nearbyint(x));
    else if (op == "s_add_f16"_op) put_h(hs(0) + hs(1));
    else if (op == "s_sub_f16"_op) put_h(hs(0) - hs(1));
    else if (op == "s_mul_f16"_op) put_h(hs(0) * hs(1));
    else if (op == "s_fmac_f16"_op)
      put_h(static_cast<_Float16>(std::fma(static_cast<float>(hs(0)), static_cast<float>(hs(1)),
                                           static_cast<float>(h(scalar(w, in.dst[0]))))));
    else if (op == "s_min_num_f16"_op) put_h(hs(0) < hs(1) || hs(1) != hs(1) ? hs(0) : hs(1));
    else if (op == "s_max_num_f16"_op) put_h(hs(0) > hs(1) || hs(1) != hs(1) ? hs(0) : hs(1));
    else return false;
    return true;
  }

  void scalar_alu(Wave& w, const Inst& in) {
    // A comparison against the instruction's own constant -- signed, or for
    // the unsigned forms without its sign: the register field names what is
    // compared, and only SCC is written. The SOPK opcode says which test.
    if (in.enc == gcn::Enc::Sopk && in.opcode >= 0x02 && in.opcode <= 0x0d) {
      // gfx9's decoder keeps the compared register as the destination field
      // it is encoded in; RDNA's lists it as the source it is.
      const uint32_t x = static_cast<uint32_t>(scalar(w, in.dst.empty() ? in.src.at(0) : in.dst[0]));
      const bool is_unsigned = in.opcode >= 0x08;
      const uint32_t k = is_unsigned ? static_cast<uint16_t>(in.simm) : static_cast<uint32_t>(static_cast<int32_t>(in.simm));
      const auto test = [&](auto a, auto b) {
        switch ((in.opcode - 0x02) % 6) {
          case 0: return a == b;
          case 1: return a != b;
          case 2: return a > b;
          case 3: return a >= b;
          case 4: return a < b;
          default: return a <= b;
        }
      };
      w.scc = is_unsigned ? test(x, k) : test(static_cast<int32_t>(x), static_cast<int32_t>(k));
      return;
    }
    const OpName op(in.name);
    const uint64_t a = in.src.empty() ? 0 : scalar(w, in.src[0]);
    const uint64_t b = in.src.size() > 1 ? scalar(w, in.src[1]) : 0;
    if (op == "s_getpc_b64"_op) {
      // The address of the instruction after this one, which is what the
      // hardware gives: the program counter has already moved on.
      write_scalar(w, in.dst[0], w.pc);
    } else if (op == "s_mov_b32"_op || op == "s_mov_b64"_op) {
      write_scalar(w, in.dst[0], a);
    } else if (op == "s_movk_i32"_op) {
      write_scalar(w, in.dst[0], static_cast<uint64_t>(static_cast<int64_t>(in.simm)));
    } else if (op == "s_addk_i32"_op) {
      // The destination is also a source, the constant is signed, and SCC
      // says whether the signed sum overflowed.
      const int32_t x = static_cast<int32_t>(scalar(w, in.dst[0])), k = static_cast<int32_t>(in.simm);
      const int32_t sum = static_cast<int32_t>(static_cast<uint32_t>(x) + static_cast<uint32_t>(k));
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = ((x >= 0) == (k >= 0)) && ((sum >= 0) != (x >= 0));
    } else if (op == "s_mulk_i32"_op) {
      // The destination is also a source: it is multiplied in place.
      write_scalar(w, in.dst[0], static_cast<uint32_t>(scalar(w, in.dst[0]) * static_cast<uint32_t>(in.simm)));
    } else if (op == "s_add_u32"_op) {
      const uint64_t sum = static_cast<uint32_t>(a) + static_cast<uint64_t>(static_cast<uint32_t>(b));
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = sum >> 32;                           // the carry out
    } else if (op == "s_sub_u32"_op || op == "s_subb_u32"_op) {
      // A borrow is SCC, both out of the subtraction and, for the second,
      // into it: the low half of a 64-bit subtract, then the high half.
      const uint64_t take = static_cast<uint64_t>(static_cast<uint32_t>(b)) + (op == "s_subb_u32"_op ? w.scc : 0);
      write_scalar(w, in.dst[0], static_cast<uint32_t>(static_cast<uint32_t>(a) - take));
      w.scc = static_cast<uint32_t>(a) < take;
    } else if (op == "s_lshl_b32"_op) {
      const uint32_t r = static_cast<uint32_t>(a) << (b & 31);
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_lshr_b64"_op) {
      const uint64_t r = a >> (b & 63);
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_orn2_b64"_op) {
      const uint64_t r = a | ~b;
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_orn2_b32"_op || op == "s_nand_b32"_op || op == "s_nor_b32"_op || op == "s_xnor_b32"_op) {
      // The rest of the bitwise family: or with the second inverted, and the
      // inverses of and, or and xor. SCC says whether any bit is set.
      const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
      const uint32_t r = op == "s_orn2_b32"_op  ? x | ~y
                         : op == "s_nand_b32"_op ? ~(x & y)
                         : op == "s_nor_b32"_op  ? ~(x | y)
                                                 : ~(x ^ y);
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_nand_b64"_op || op == "s_nor_b64"_op || op == "s_xnor_b64"_op) {
      const uint64_t r = op == "s_nand_b64"_op ? ~(a & b) : op == "s_nor_b64"_op ? ~(a | b) : ~(a ^ b);
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_bfm_b64"_op) {
      // A mask of the first source's low six bits' worth of ones, shifted up
      // by the second's. SCC is left alone.
      const uint32_t width = static_cast<uint32_t>(a) & 63, shift = static_cast<uint32_t>(b) & 63;
      write_scalar(w, in.dst[0], ((uint64_t{1} << width) - 1) << shift);
    } else if (op == "s_bfe_u64"_op) {
      // A field of the 64-bit first source: the second source's low six bits
      // say where it starts, bits 16 to 22 how wide it is.
      const uint32_t start = static_cast<uint32_t>(b) & 63, width = (static_cast<uint32_t>(b) >> 16) & 0x7F;
      const uint64_t shifted = a >> start;
      const uint64_t r = width == 0 ? 0 : width >= 64 ? shifted : shifted & ((uint64_t{1} << width) - 1);
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_absdiff_i32"_op) {
      const int64_t d = int64_t{static_cast<int32_t>(a)} - static_cast<int32_t>(b);
      const uint32_t r = static_cast<uint32_t>(d < 0 ? -d : d);
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_bcnt1_i32_b64"_op) {
      // The set bits of a 64-bit mask: how many lanes are active, which is
      // how a wave that adds one value for all its lanes knows how many.
      const uint32_t r = static_cast<uint32_t>(__builtin_popcountll(a));
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_or_saveexec_b64"_op) {
      const uint64_t saved = w.exec;
      w.exec = a | saved;
      write_scalar(w, in.dst[0], saved);
      w.scc = w.exec != 0;
    } else if (op == "s_addc_u32"_op) {
      const uint64_t sum = static_cast<uint32_t>(a) + static_cast<uint64_t>(static_cast<uint32_t>(b)) + w.scc;
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = sum >> 32;
    } else if (op == "s_add_i32"_op) {
      const int64_t sum = static_cast<int32_t>(a) + static_cast<int64_t>(static_cast<int32_t>(b));
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = sum != static_cast<int32_t>(sum);      // signed overflow
    } else if (op == "s_sub_i32"_op) {
      const int64_t diff = static_cast<int32_t>(a) - static_cast<int64_t>(static_cast<int32_t>(b));
      write_scalar(w, in.dst[0], static_cast<uint32_t>(diff));
      w.scc = diff != static_cast<int32_t>(diff);
    } else if (op == "s_or_b32"_op) {
      const uint32_t v = static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_xor_b32"_op) {
      const uint32_t v = static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_mul_i32"_op) {
      write_scalar(w, in.dst[0], static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
    } else if (op == "s_ashr_i32"_op) {
      const uint32_t v = static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31));
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_and_b32"_op) {
      const uint32_t v = static_cast<uint32_t>(a) & static_cast<uint32_t>(b);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_xor_b64"_op) {
      const uint64_t v = a ^ b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_andn2_b64"_op) {
      const uint64_t v = a & ~b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_ff1_i32_b64"_op) {
      // The first set bit, counting from bit 0, or -1 when there is none.
      write_scalar(w, in.dst[0], a ? static_cast<uint32_t>(__builtin_ctzll(a)) : 0xFFFFFFFFu);
    } else if (op == "s_getreg_b32"_op || op == "s_setreg_imm32_b32"_op || op == "s_setreg_b32"_op) {
      // A field of a hardware register: the immediate's low six bits say
      // which register, the next five where the field starts, the top five
      // how wide it is less one. MODE is kept per wave, and gfx12's SCHED_MODE;
      // HW_ID, HW_ID1 and XCC_ID say where the wave runs, gfx12's STATE_PRIV
      // holds SCC, and SHADER_CYCLES is the shader clock; the rest are
      // refused by name.
      const uint32_t id = static_cast<uint32_t>(in.simm) & 0x3F, at = (static_cast<uint32_t>(in.simm) >> 6) & 0x1F,
                     width = ((static_cast<uint32_t>(in.simm) >> 11) & 0x1F) + 1;
      const uint32_t mask = (width >= 32 ? ~0u : (1u << width) - 1) << at;
      const bool sched = id == 26 && gcn::is_gfx12(in.arch);
      const bool flat_scr = (id == 20 || id == 21) && in.arch == gcn::Target::Gfx1030;
      if (op != "s_getreg_b32"_op) {
        if (id != 1 && !sched && !flat_scr)
          throw Error::make(Err::Unsupported, op, " of hardware register ", id, ", which this does not model");
        // The immediate form's value is its literal; the register form's is
        // the scalar register the instruction names.
        const uint32_t value = op == "s_setreg_imm32_b32"_op ? static_cast<uint32_t>(in.src[0].value)
                               : !in.src.empty()             ? static_cast<uint32_t>(scalar(w, in.src[0]))
                                                             : static_cast<uint32_t>(scalar(w, in.dst[0]));
        uint32_t& reg = sched ? w.sched_mode : flat_scr ? w.flat_scratch[id - 20] : w.mode;
        reg = (reg & ~mask) | ((value << at) & mask);
      } else {
        uint32_t reg;
        if (id == 1) reg = w.mode;
        else if (sched) reg = w.sched_mode;
        else if (flat_scr) reg = w.flat_scratch[id - 20];
        else if (id == 4 && gcn::is_gfx12(in.arch)) {
          // gfx12's STATE_PRIV, where register 4 was HW_ID before: of its
          // fields only SCC (bit 9) has a value here. The barrier, priority,
          // halt, debug and trace states read as a running wave's are, 0.
          // (HIP's __smid still reads HW_ID's fields from it on gfx12.)
          reg = static_cast<uint32_t>(w.scc) << 9;
        } else if (id == 4) {
          // HW_ID: the wave's slot (WAVE_ID, 3:0), and on CDNA its compute
          // unit (11:8), shader array (12) and engine (14:13, or 15:13 on
          // gfx90a).
          reg = static_cast<uint32_t>(w.first_lane / w.lanes) & 0xF;
          if (!is_rdna(in.arch)) {
            const Place p = place_of(d, w.group);
            reg |= (p.unit & 0xF) << 8 | (p.array & 1) << 12 | (p.engine & 7) << 13;
          }
        } else if (id == 20 && (in.arch == gcn::Target::Gfx942 || in.arch == gcn::Target::Gfx950)) {
          reg = place_of(d, w.group).die & 0xF;   // XCC_ID: the compute die (3:0)
        } else if (id == 23 && is_rdna(in.arch)) {
          // HW_ID1: the wave's slot (4:0), its workgroup processor (13:10),
          // shader array (16) and engine (20:18). A work-group's waves are
          // all on one workgroup processor, as in HIP's default WGP mode.
          const Place p = place_of(d, w.group);
          reg = (static_cast<uint32_t>(w.first_lane / w.lanes) & 0x1F) | (p.unit & 0xF) << 10 | (p.array & 1) << 16 |
                (p.engine & 7) << 18;
        }
        // SHADER_CYCLES, which clock() reads on RDNA: 20 bits of the cycle
        // count on gfx10.3 and gfx11; on gfx12 its low word (29) and high
        // word (30). The count is s_memtime's, the instructions retired.
        else if (id == 29 && is_rdna(in.arch))
          reg = gcn::is_gfx12(in.arch) ? static_cast<uint32_t>(stats.instructions)
                                                : static_cast<uint32_t>(stats.instructions) & 0xFFFFF;
        else if (id == 30 && gcn::is_gfx12(in.arch)) reg = static_cast<uint32_t>(stats.instructions >> 32);
        // gfx1250's IB_STS2, whose bits 9:6 are the wave's number in its work-group (llvm.amdgcn.wave.id reads
        // them, where gfx12 read TTMP8).
        else if (id == 28 && in.arch == gcn::Target::Gfx1250) reg = (static_cast<uint32_t>(w.first_lane / w.lanes) & 0xF) << 6;
        else throw Error::make(Err::Unsupported, "s_getreg_b32 of hardware register ", id, ", which this does not model");
        write_scalar(w, in.dst[0], (reg & mask) >> at);
      }
    } else if (op == "s_call_b64"_op) {
      // Where to come back to is the next instruction, which the program
      // counter already holds.
      write_scalar(w, in.dst[0], w.pc);
      w.pc = in.target;
    } else if (op == "s_bfm_b32"_op) {
      write_scalar(w, in.dst[0], ((1u << (a & 31)) - 1) << (b & 31));
    } else if (op == "s_ff1_i32_b32"_op) {
      const uint32_t x = static_cast<uint32_t>(a);
      write_scalar(w, in.dst[0], x ? static_cast<uint32_t>(__builtin_ctz(x)) : 0xFFFFFFFFu);
    } else if (op == "s_flbit_i32_b32"_op || op == "s_flbit_i32_b64"_op) {
      // The first set bit counting from the top, or -1 when there is none.
      const bool wide = op == "s_flbit_i32_b64"_op;
      const uint64_t x = wide ? a : static_cast<uint32_t>(a);
      write_scalar(w, in.dst[0], !x ? 0xFFFFFFFFu
                                    : static_cast<uint32_t>(wide ? __builtin_clzll(x) : __builtin_clz(static_cast<uint32_t>(x))));
    } else if (op == "s_flbit_i32"_op) {
      // The first bit from the top that differs from the sign bit, or -1
      // when every bit is the sign.
      const uint32_t x = static_cast<uint32_t>(a), y = x >> 31 ? ~x : x;
      write_scalar(w, in.dst[0], y ? static_cast<uint32_t>(__builtin_clz(y)) : 0xFFFFFFFFu);
    } else if (op == "s_bcnt1_i32_b32"_op) {
      const uint32_t r = static_cast<uint32_t>(__builtin_popcount(static_cast<uint32_t>(a)));
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_sext_i32_i8"_op) {
      write_scalar(w, in.dst[0], static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(a))));
    } else if (op == "s_sext_i32_i16"_op) {
      write_scalar(w, in.dst[0], static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(a))));
    } else if (op == "s_max_u32"_op) {
      const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
      w.scc = x > y;
      write_scalar(w, in.dst[0], w.scc ? x : y);
    } else if (op == "s_lshl1_add_u32"_op || op == "s_lshl2_add_u32"_op || op == "s_lshl3_add_u32"_op ||
               op == "s_lshl4_add_u32"_op) {
      // SCC is the carry out of the whole: of the shift and of the add.
      const uint32_t by = static_cast<uint32_t>(op.substr(6, 1)[0] - '0');
      const uint64_t v = (static_cast<uint64_t>(static_cast<uint32_t>(a)) << by) + static_cast<uint32_t>(b);
      write_scalar(w, in.dst[0], static_cast<uint32_t>(v));
      w.scc = (v >> 32) != 0;
    } else if (op == "s_pack_lh_b32_b16"_op) {
      write_scalar(w, in.dst[0], (static_cast<uint32_t>(a) & 0xFFFFu) | (static_cast<uint32_t>(b) & 0xFFFF0000u));
    } else if (op == "s_pack_hh_b32_b16"_op) {
      write_scalar(w, in.dst[0], static_cast<uint32_t>(a) >> 16 | (static_cast<uint32_t>(b) & 0xFFFF0000u));
    } else if (op == "s_and_b64"_op) {
      const uint64_t v = a & b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_or_b64"_op) {
      const uint64_t v = a | b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_lshl_b64"_op) {
      const uint64_t v = a << (b & 63);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_cselect_b64"_op) {
      // What the last comparison decided picks a source.
      write_scalar(w, in.dst[0], w.scc ? a : b);
    } else if (op == "s_and_saveexec_b64"_op) {
      // Divergence, as the compiler writes it: keep EXEC, narrow it to the
      // lanes the condition took.
      const uint64_t saved = w.exec;
      w.exec = a & saved;
      write_scalar(w, in.dst[0], saved);
      w.scc = w.exec != 0;
    } else if (op == "s_swappc_b64"_op) {
      // A call: where to come back to is what the program counter already
      // holds, since it was moved past this instruction before it ran.
      const uint64_t to = scalar(w, in.src[0]);
      write_scalar(w, in.dst[0], w.pc);
      w.pc = to;
    } else if (op == "s_setpc_b64"_op || op == "s_set_pc_i64"_op) {
      w.pc = scalar(w, in.src[0]);   // the return
    } else if (op == "s_get_pc_i64"_op) {
      write_scalar(w, in.dst[0], w.pc);   // (gfx1250's s_getpc_b64: the next instruction's address)
    } else if (op == "s_swap_pc_i64"_op) {
      const uint64_t to = scalar(w, in.src[0]);
      write_scalar(w, in.dst[0], w.pc);
      w.pc = to;
    } else if (op == "s_add_pc_i64"_op) {
      // The program counter (already past this instruction) moved by a signed offset; a 32-bit literal is sign-extended.
      const int64_t off = in.src[0].kind == OperandKind::Literal && !in.src[0].lit64
                              ? static_cast<int64_t>(static_cast<int32_t>(a))
                              : static_cast<int64_t>(a);
      w.pc = static_cast<uint64_t>(static_cast<int64_t>(w.pc) + off);
    } else if (op == "s_call_i64"_op) {
      write_scalar(w, in.dst[0], w.pc);
      w.pc = in.target;
    } else if (op == "s_get_shader_cycles_u64"_op) {
      write_scalar(w, in.dst[0], stats.instructions);
    } else if (op == "s_lshr_b32"_op) {
      write_scalar(w, in.dst[0], static_cast<uint32_t>(a) >> (b & 31));
      w.scc = static_cast<uint32_t>(a) >> (b & 31);
    } else if (op == "s_mul_hi_u32"_op) {
      write_scalar(w, in.dst[0],
                   static_cast<uint32_t>((static_cast<uint64_t>(static_cast<uint32_t>(a)) *
                                          static_cast<uint32_t>(b)) >> 32));
    } else if (op == "s_cselect_b32"_op) {
      write_scalar(w, in.dst[0], w.scc ? a : b);
    } else if (op == "s_brev_b32"_op) {
      uint32_t v = static_cast<uint32_t>(a), r = 0;
      for (uint32_t k = 0; k < 32; ++k) r |= ((v >> k) & 1) << (31 - k);
      write_scalar(w, in.dst[0], r);
      // A bit reversal sets no condition code, as the move it is a form of
      // does not.
    } else if (op == "s_brev_b64"_op) {
      uint64_t r = 0;
      for (uint32_t k = 0; k < 64; ++k) r |= ((a >> k) & 1) << (63 - k);
      write_scalar(w, in.dst[0], r);
    } else if (op == "s_cmov_b64"_op) {
      if (w.scc) write_scalar(w, in.dst[0], a);
    } else if (op == "s_bcnt0_i32_b32"_op || op == "s_bcnt0_i32_b64"_op) {
      const uint32_t r = op == "s_bcnt0_i32_b32"_op ? 32 - __builtin_popcount(static_cast<uint32_t>(a))
                                                    : 64 - __builtin_popcountll(a);
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_ff0_i32_b32"_op || op == "s_ff0_i32_b64"_op) {
      // The first clear bit, counting from bit 0, or -1 when there is none.
      const uint64_t clear = op == "s_ff0_i32_b32"_op ? ~static_cast<uint32_t>(a) & 0xFFFFFFFFull : ~a;
      write_scalar(w, in.dst[0], clear ? static_cast<uint32_t>(__builtin_ctzll(clear)) : 0xFFFFFFFFu);
    } else if (op == "s_wqm_b32"_op || op == "s_wqm_b64"_op) {
      uint64_t r = 0;
      for (uint32_t q = 0; q < (op == "s_wqm_b32"_op ? 8u : 16u); ++q)
        if ((a >> (4 * q)) & 0xF) r |= uint64_t{0xF} << (4 * q);
      write_scalar(w, in.dst[0], r);
      w.scc = r != 0;
    } else if (op == "s_cmov_b32"_op) {
      if (w.scc) write_scalar(w, in.dst[0], a);
    } else if (op == "s_abs_i32"_op) {
      const int32_t x = static_cast<int32_t>(a);
      const uint32_t v = x < 0 ? 0u - static_cast<uint32_t>(x) : static_cast<uint32_t>(x);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_not_b32"_op || op == "s_not_b64"_op) {
      const uint64_t v = op == "s_not_b32"_op ? static_cast<uint32_t>(~a) : ~a;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_bitset0_b32"_op || op == "s_bitset1_b32"_op) {
      // One bit of the destination, the source's low five bits say which;
      // the rest of it, and SCC, are kept.
      const uint32_t bit = 1u << (a & 31), was = static_cast<uint32_t>(scalar(w, in.dst[0]));
      write_scalar(w, in.dst[0], op == "s_bitset1_b32"_op ? was | bit : was & ~bit);
    } else if (op == "s_min_i32"_op || op == "s_max_i32"_op) {
      // SCC says whether the first source was the one kept.
      const int32_t x = static_cast<int32_t>(a), y = static_cast<int32_t>(b);
      w.scc = op == "s_min_i32"_op ? x < y : x > y;
      write_scalar(w, in.dst[0], static_cast<uint32_t>(w.scc ? x : y));
    } else if (op == "s_min_u32"_op) {
      const uint32_t x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
      w.scc = x < y;
      write_scalar(w, in.dst[0], w.scc ? x : y);
    } else if (op == "s_andn2_b32"_op) {
      const uint32_t v = static_cast<uint32_t>(a & ~b);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_ashr_i64"_op) {
      const uint64_t v = static_cast<uint64_t>(static_cast<int64_t>(a) >> (b & 63));
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_bfe_i32"_op) {
      const uint32_t start = static_cast<uint32_t>(b) & 31, width = (static_cast<uint32_t>(b) >> 16) & 0x7F;
      int32_t v = 0;
      if (width) {
        const uint32_t field = static_cast<uint32_t>(a) >> start;
        v = width >= 32 ? static_cast<int32_t>(field)
                        : static_cast<int32_t>(field << (32 - width)) >> (32 - width);
      }
      write_scalar(w, in.dst[0], static_cast<uint32_t>(v));
      w.scc = v != 0;
    } else if (op == "s_bfe_u32"_op || op == "s_bfe_i64"_op) {
      // The field starts at the second source's low bits and is as wide as
      // its bits 16 to 22 say; the signed form carries the field's top bit up.
      const bool wide = op == "s_bfe_i64"_op;
      const uint32_t start = static_cast<uint32_t>(b) & (wide ? 63 : 31), width = (static_cast<uint32_t>(b) >> 16) & 0x7F;
      const uint32_t bits = wide ? 64 : 32;
      uint64_t v = (wide ? a : static_cast<uint32_t>(a)) >> start;
      if (width == 0) v = 0;
      else if (width < bits) {
        v &= (uint64_t{1} << width) - 1;
        if (wide && (v >> (width - 1)) & 1) v |= ~uint64_t{0} << width;
      }
      if (!wide) v = static_cast<uint32_t>(v);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_mul_hi_i32"_op) {
      write_scalar(w, in.dst[0], static_cast<uint32_t>(static_cast<uint64_t>(
                                     static_cast<int64_t>(static_cast<int32_t>(a)) * static_cast<int32_t>(b)) >> 32));
    } else if (op == "s_pack_ll_b32_b16"_op) {
      write_scalar(w, in.dst[0], (static_cast<uint32_t>(a) & 0xFFFFu) | (static_cast<uint32_t>(b) & 0xFFFFu) << 16);
    } else if (op == "s_xor_saveexec_b64"_op) {
      // With -1, the lanes that are off: how a function saves a register all
      // 64 lanes share without disturbing the ones its caller left running.
      const uint64_t saved = w.exec;
      w.exec = a ^ saved;
      write_scalar(w, in.dst[0], saved);
      w.scc = w.exec != 0;
    } else if (op == "s_andn2_saveexec_b64"_op) {
      // The other half of a divergence: keep EXEC, and take the lanes the
      // condition did not.
      const uint64_t saved = w.exec;
      w.exec = a & ~saved;
      write_scalar(w, in.dst[0], saved);
      w.scc = w.exec != 0;
    } else if (op == "s_and_saveexec_b32"_op || op == "s_or_saveexec_b32"_op || op == "s_xor_saveexec_b32"_op ||
               op == "s_and_not1_saveexec_b32"_op || op == "s_or_not1_saveexec_b32"_op ||
               op == "s_and_not0_saveexec_b32"_op || op == "s_or_not0_saveexec_b32"_op) {
      // RDNA's wave32 forms of the same: EXEC is its low half (exec_lo).
      const uint32_t saved = static_cast<uint32_t>(w.exec), x = static_cast<uint32_t>(a);
      const uint32_t v = op == "s_and_saveexec_b32"_op        ? x & saved
                         : op == "s_or_saveexec_b32"_op       ? x | saved
                         : op == "s_xor_saveexec_b32"_op      ? x ^ saved
                         : op == "s_and_not1_saveexec_b32"_op ? x & ~saved
                         : op == "s_or_not1_saveexec_b32"_op  ? x | ~saved
                         : op == "s_and_not0_saveexec_b32"_op ? ~x & saved
                                                              : ~x | saved;
      w.exec = (w.exec & ~0xFFFFFFFFull) | v;
      write_scalar(w, in.dst[0], saved);
      w.scc = v != 0;
    } else if (op == "s_or_not1_b32"_op) {
      const uint32_t v = static_cast<uint32_t>(a) | ~static_cast<uint32_t>(b);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_pack_hl_b32_b16"_op) {
      write_scalar(w, in.dst[0], static_cast<uint32_t>(a) >> 16 | static_cast<uint32_t>(b) << 16);
    } else if (op == "s_sendmsg_rtn_b32"_op || op == "s_sendmsg_rtn_b64"_op) {
      // A message that answers. The one compute code asks is the time
      // (realtime_ticks).
      if (in.simm != 131)
        throw Error::make(Err::Unsupported, "s_sendmsg_rtn of message ", in.simm, ", which this does not answer");
      write_scalar(w, in.dst[0], realtime_ticks());
    } else if (scalar_float(w, in, op, a, b)) {
    } else if (op == "s_add_nc_u64"_op || op == "s_sub_nc_u64"_op || op == "s_mul_u64"_op) {
      // RDNA4's 64-bit scalar arithmetic, SCC untouched.
      write_scalar(w, in.dst[0], op == "s_add_nc_u64"_op ? a + b : op == "s_sub_nc_u64"_op ? a - b : a * b);
    } else if (op == "s_barrier_signal"_op || op == "s_barrier_signal_isfirst"_op) {
      // RDNA4's split barrier: signalling arrives; s_barrier_wait is where the
      // wave waits for the rest (and is the barrier here).
      if (op == "s_barrier_signal_isfirst"_op) w.scc = false;
    } else if (op == "s_waitcnt_vscnt"_op || op == "s_waitcnt_vmcnt"_op || op == "s_waitcnt_expcnt"_op ||
               op == "s_waitcnt_lgkmcnt"_op || op == "s_version"_op) {
      // RDNA's separate counters: nothing is outstanding here to wait for.
      // (s_version only marks which ISA the code was written for.)
    } else {
      throw Error::make(Err::Unsupported, "scalar instruction ", op, " is decoded but not implemented");
    }
  }

  // A scalar comparison: SCC is the whole result.
  void scalar_compare(Wave& w, const Inst& in) {
    if (in.name == "s_set_gpr_idx_on") {
      // M0's low byte becomes the index, its bits 12 to 15 which operands
      // it applies to.
      w.m0 = (w.m0 & ~0xF0FFu) | (static_cast<uint32_t>(scalar(w, in.src[0])) & 0xFF) |
             (static_cast<uint32_t>(in.simm) & 0xF) << 12;
      w.gpr_idx = static_cast<uint8_t>(in.simm & 0xF);
      return;
    }
    const uint64_t a = scalar(w, in.src[0]), b = scalar(w, in.src[1]);
    const OpName op(in.name);
    // RDNA's scalar float comparisons: s_cmp_<test>_f32 and _f16, the vector
    // comparisons' tests.
    if (in.name.size() > 10 && in.name.rfind("s_cmp_", 0) == 0 &&
        (in.name.compare(in.name.size() - 4, 4, "_f32") == 0 || in.name.compare(in.name.size() - 4, 4, "_f16") == 0)) {
      const bool half = in.name.compare(in.name.size() - 4, 4, "_f16") == 0;
      const auto value = [&](size_t k, uint64_t v) -> double {
        return half ? static_cast<double>(scalar_half(w, in.src[k])) : as_float(static_cast<uint32_t>(v));
      };
      const double x = value(0, a), y = value(1, b);
      const std::string t = in.name.substr(6, in.name.size() - 10);
      const bool unordered = x != x || y != y;
      w.scc = t == "lt" ? x < y : t == "eq" ? x == y : t == "le" ? x <= y : t == "gt" ? x > y
              : t == "lg" ? (x < y || x > y) : t == "ge" ? x >= y : t == "o" ? !unordered : t == "u" ? unordered
              : t == "nge" ? !(x >= y) : t == "nlg" ? !(x < y || x > y) : t == "ngt" ? !(x > y)
              : t == "nle" ? !(x <= y) : t == "neq" ? !(x == y) : t == "nlt" ? !(x < y)
              : throw Error::make(Err::Unsupported, in.name, " is decoded but not implemented");
      return;
    }
    if (op == "s_cmp_lt_i32"_op) w.scc = static_cast<int32_t>(a) < static_cast<int32_t>(b);
    else if (op == "s_cmp_eq_u32"_op) w.scc = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
    else if (op == "s_cmp_ge_u32"_op) w.scc = static_cast<uint32_t>(a) >= static_cast<uint32_t>(b);
    else if (op == "s_cmp_gt_u32"_op) w.scc = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
    else if (op == "s_cmp_lg_u32"_op) w.scc = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
    else if (op == "s_cmp_gt_i32"_op) w.scc = static_cast<int32_t>(a) > static_cast<int32_t>(b);
    else if (op == "s_cmp_ge_i32"_op) w.scc = static_cast<int32_t>(a) >= static_cast<int32_t>(b);
    else if (op == "s_cmp_eq_u64"_op) w.scc = a == b;
    else if (op == "s_cmp_lt_u32"_op) w.scc = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
    else if (op == "s_cmp_lg_u64"_op) w.scc = a != b;
    else if (op == "s_cmp_le_i32"_op) w.scc = static_cast<int32_t>(a) <= static_cast<int32_t>(b);
    else if (op == "s_cmp_eq_i32"_op) w.scc = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
    else if (op == "s_cmp_lg_i32"_op) w.scc = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
    else if (op == "s_cmp_le_u32"_op) w.scc = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
    else if (op == "s_bitcmp0_b32"_op) w.scc = ((a >> (b & 31)) & 1) == 0;
    else if (op == "s_bitcmp1_b32"_op) w.scc = ((a >> (b & 31)) & 1) == 1;
    else throw Error::make(Err::Unsupported, "scalar comparison ", op, " is decoded but not implemented");
  }

  void scalar_load(Wave& w, const Inst& in) {
    // Counters, rather than loads. s_memtime is the shader clock, which on a
    // card runs with the core clock; here it is the instructions retired so
    // far by the host thread running the wave, which is this model's cycle
    // count. A wave stays on one thread, so for it the count only ever goes
    // up, which is what a program timing a stretch of its own code depends
    // on. s_memrealtime is the real-time clock (realtime_ticks), at the rate
    // the runtime reports: a kernel waiting on it for some milliseconds, as
    // wall_clock64() loops do, waits that long.
    if (OpName(in.name) == "s_memtime"_op || OpName(in.name) == "s_memrealtime"_op) {
      const uint64_t t = OpName(in.name) == "s_memtime"_op ? stats.instructions : realtime_ticks();
      set_sgpr(w, in.dst[0].index, static_cast<uint32_t>(t));
      set_sgpr(w, in.dst[0].index + 1, static_cast<uint32_t>(t >> 32));
      return;
    }
    // The scalar cache holds nothing here to write back or drop.
    if (OpName(in.name) == "s_dcache_wb"_op || OpName(in.name) == "s_dcache_inv"_op) return;
    // The base, the instruction's own offset, and a scalar register's where
    // it names one.
    // (RDNA's decoder keeps the register as the second source: null, reading
    // zero, where there is none.)
    const uint64_t base = scalar(w, in.src[0]) + static_cast<uint64_t>(in.offset) +
                          (in.has_saddr ? scalar_field(w, in.saddr, false) : 0) +
                          (gcn::is_rdna(in.arch) && in.src.size() > 1
                               ? scaled_offset(in, static_cast<uint32_t>(scalar(w, in.src[1])),
                                               in.dst.empty() ? 4 : 4 * in.dst[0].width)
                               : 0);
    if (in.name.rfind("s_store_dword", 0) == 0) {
      for (uint32_t i = 0; i < in.dst[0].width; ++i)
        at(base + 4 * i).store_scalar(base + 4 * i, 4, sgpr(w, in.dst[0].index + i));
      return;
    }
    if (in.name.rfind("s_atomic_", 0) == 0) {
      // One value for the whole wave, read and replaced at once; with glc
      // the data register gets back what memory held.
      const bool wide = in.name.size() > 3 && in.name.compare(in.name.size() - 3, 3, "_x2") == 0;
      const uint32_t bytes = wide ? 8 : 4, r = in.dst[0].index;
      const std::string what = in.name.substr(9, in.name.size() - 9 - (wide ? 3 : 0));
      const uint64_t mask = wide ? ~uint64_t{0} : 0xFFFFFFFFull;
      const uint64_t data = wide ? sgpr64(w, r) : sgpr(w, r);
      const auto guard = atomic_guard(base);
      const uint64_t old = at(base).load_scalar(base, bytes);
      const auto sx = [&](uint64_t v) { return wide ? static_cast<int64_t>(v) : static_cast<int64_t>(static_cast<int32_t>(v)); };
      uint64_t now;
      if (what == "swap") now = data;
      else if (what == "cmpswap") now = old == (wide ? sgpr64(w, r + 2) : sgpr(w, r + 1)) ? data : old;
      else if (what == "add") now = old + data;
      else if (what == "sub") now = old - data;
      else if (what == "smin") now = sx(old) < sx(data) ? old : data;
      else if (what == "umin") now = std::min(old, data);
      else if (what == "smax") now = sx(old) > sx(data) ? old : data;
      else if (what == "umax") now = std::max(old, data);
      else if (what == "and") now = old & data;
      else if (what == "or") now = old | data;
      else if (what == "xor") now = old ^ data;
      else if (what == "inc") now = old >= data ? 0 : old + 1;
      else if (what == "dec") now = old == 0 || old > data ? data : old - 1;
      else throw Error::make(Err::Unsupported, in.name, " is decoded but not implemented");
      at(base).store_scalar(base, bytes, now & mask);
      if (in.cache & 1) {
        if (wide) set_sgpr64(w, r, old);
        else set_sgpr(w, r, static_cast<uint32_t>(old));
      }
      return;
    }
    // RDNA4's sub-dword scalar loads: a byte or a short, widened with its sign
    // or without.
    if (OpName(in.name) == "s_load_u8"_op || OpName(in.name) == "s_load_i8"_op ||
        OpName(in.name) == "s_load_u16"_op || OpName(in.name) == "s_load_i16"_op) {
      const uint32_t bytes = in.name.back() == '8' ? 1 : 2;
      const uint64_t raw = at(base).load_scalar(base, bytes);
      const bool sign = in.name[7] == 'i';
      const uint32_t v = bytes == 1 ? (sign ? static_cast<uint32_t>(static_cast<int8_t>(raw)) : static_cast<uint32_t>(raw & 0xFF))
                                    : (sign ? static_cast<uint32_t>(static_cast<int16_t>(raw)) : static_cast<uint32_t>(raw & 0xFFFF));
      set_sgpr(w, in.dst[0].index, v);
      return;
    }
    const uint32_t words = in.dst[0].width;
    if (in.name.rfind("s_buffer_load_", 0) == 0) {
      // Through a buffer resource: its base (48 bits) and its size in bytes
      // (num_records); each dword past the end reads 0.
      const uint32_t r = in.src[0].index;
      const uint64_t buffer = sgpr(w, r) | uint64_t{sgpr(w, r + 1) & 0xFFFF} << 32;
      const uint64_t records = sgpr(w, r + 2);
      const uint64_t offset = static_cast<uint64_t>(in.offset) + (in.has_saddr ? scalar_field(w, in.saddr, false) : 0);
      for (uint32_t i = 0; i < words; ++i) {
        const uint64_t off = offset + 4 * i, addr = buffer + off;
        set_sgpr(w, in.dst[0].index + i,
                 off + 4 <= records ? static_cast<uint32_t>(at(addr).load_scalar(addr, 4)) : 0);
      }
      return;
    }
    for (uint32_t i = 0; i < words; ++i)
      set_sgpr(w, in.dst[0].index + i, static_cast<uint32_t>(at(base + 4 * i).load_scalar(base + 4 * i, 4)));
  }

  static uint32_t first_active(const Wave& w) {
    return w.exec ? static_cast<uint32_t>(__builtin_ctzll(w.exec)) : 0;
  }

  // The three instructions a float division is built from. The compiler
  // emits v_div_scale to bring extreme operands into range, refines a
  // reciprocal, then v_div_fmas and v_div_fixup put the result back.
  //
  // The scaling exists because the hardware's reciprocal is approximate, and
  // this reciprocal is exact, so the sequence lands on the correctly rounded
  // quotient either way.
  void divide_step(Wave& w, const Inst& in, uint32_t lane) {
    const OpName op(in.name);
    // The same three steps, in double precision: the ISA gives each a form of
    // its own, and the scaling is by 2^128 rather than 2^64.
    if (op.size() > 4 && op.compare(op.size() - 4, 4, "_f64") == 0) {
      if (op == "v_div_scale_f64"_op) {
        const double value = lane_double(w, in.src[0], lane), den = lane_double(w, in.src[1], lane),
                     num = lane_double(w, in.src[2], lane);
        bool scaled = false;
        double out = value;
        // Both have to be ordinary numbers to compare their exponents:
        // ilogb of a zero is INT_MIN, and subtracting that overflows.
        if (std::isfinite(den) && std::isfinite(num) && den != 0 && num != 0) {
          const int64_t de = std::ilogb(den), ne = std::ilogb(num);
          if (ne - de >= 1022 || ne - de <= -1022) {
            out = std::ldexp(value, 128);
            scaled = true;
          }
        }
        write_lane64(w, in.dst[0], lane, as_bits(out));
        if (scaled) sdst_bits |= uint64_t{1} << lane;
      } else if (op == "v_div_fmas_f64"_op) {
        const double r = std::fma(lane_double(w, in.src[0], lane), lane_double(w, in.src[1], lane),
                                  lane_double(w, in.src[2], lane));
        write_lane64(w, in.dst[0], lane, as_bits((w.vcc >> lane) & 1 ? std::ldexp(r, 128) : r));
      } else {   // v_div_fixup_f64
        const double q = lane_double(w, in.src[0], lane), den = lane_double(w, in.src[1], lane),
                     num = lane_double(w, in.src[2], lane);
        double out = q;
        if (std::isnan(num) || std::isnan(den)) out = std::numeric_limits<double>::quiet_NaN();
        else if (den == 0) out = num == 0 ? std::numeric_limits<double>::quiet_NaN()
                                          : std::copysign(std::numeric_limits<double>::infinity(), num) *
                                                std::copysign(1.0, den);
        else if (std::isinf(den)) out = std::isinf(num) ? std::numeric_limits<double>::quiet_NaN()
                                                        : std::copysign(0.0, num) * std::copysign(1.0, den);
        else if (std::isinf(num)) out = std::copysign(std::numeric_limits<double>::infinity(), num) *
                                        std::copysign(1.0, den);
        write_lane64(w, in.dst[0], lane, as_bits(out));
      }
      return;
    }
    if (op == "v_div_scale_f32"_op) {
      // src0 is the value to scale, src1 the denominator, src2 the numerator.
      const float value = lane_float(w, in.src[0], lane), den = lane_float(w, in.src[1], lane),
                  num = lane_float(w, in.src[2], lane);
      // Scale only where the quotient would otherwise overflow or flush to
      // zero: the ISA scales by 2^64, and says so in the condition register.
      bool scaled = false;
      float out = value;
      // As above: a zero has no exponent to compare.
      if (std::isfinite(den) && std::isfinite(num) && den != 0 && num != 0) {
        const int64_t de = std::ilogb(den), ne = std::ilogb(num);
        if (ne - de >= 126 || ne - de <= -126) {
          out = std::ldexp(value, 64);
          scaled = true;
        }
      }
      write_lane(w, in.dst[0], lane, as_bits(out));
      if (scaled) sdst_bits |= uint64_t{1} << lane;
    } else if (op == "v_div_fmas_f32"_op) {
      // A fused multiply-add, times 2^64 where the scaling said so.
      const float r = std::fma(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane),
                               lane_float(w, in.src[2], lane));
      write_lane(w, in.dst[0], lane, as_bits((w.vcc >> lane) & 1 ? std::ldexp(r, 64) : r));
    } else {   // v_div_fixup_f32: the quotient, with the cases the sequence cannot do
      const float q = lane_float(w, in.src[0], lane), den = lane_float(w, in.src[1], lane),
                  num = lane_float(w, in.src[2], lane);
      float out = q;
      if (std::isnan(num) || std::isnan(den)) out = std::numeric_limits<float>::quiet_NaN();
      else if (den == 0) out = num == 0 ? std::numeric_limits<float>::quiet_NaN()
                                        : std::copysign(std::numeric_limits<float>::infinity(), num) *
                                              std::copysign(1.0f, den);
      else if (std::isinf(den)) out = std::isinf(num) ? std::numeric_limits<float>::quiet_NaN()
                                                     : std::copysign(0.0f, num) * std::copysign(1.0f, den);
      else if (std::isinf(num)) out = std::copysign(std::numeric_limits<float>::infinity(), num) *
                                      std::copysign(1.0f, den);
      write_lane(w, in.dst[0], lane, as_bits(out));
    }
  }

  // The arithmetic a 64-bit value needs, which a register pair holds: an add
  // or a subtract over one half at a time, reporting what carried out of it
  // as a mask of lanes, and the forms that read that mask back for the other
  // half. The mask goes to VCC in the short form, and to the pair the long
  // form names.
  bool carry_alu(Wave& w, const Inst& in) {
    const OpName op(in.name);
    if (op.find("_co_u32") == std::string::npos) return false;
    const bool add = op.rfind("v_add", 0) == 0;
    const bool takes_carry = op.find("_addc_") != std::string::npos || op.find("_subb") != std::string::npos;
    // The "rev" forms subtract the first source from the second.
    const bool reversed = op.find("subbrev") != std::string::npos || op.find("subrev") != std::string::npos;
    const uint64_t carried_in = takes_carry ? scalar(w, in.src[2]) : 0;
    uint64_t carried_out = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      uint64_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
      if (reversed) std::swap(a, b);
      const uint64_t c = takes_carry ? (carried_in >> lane) & 1 : 0;
      const uint64_t r = add ? a + b + c : a - b - c;
      if (add ? (r >> 32) & 1 : a < b + c) carried_out |= uint64_t{1} << lane;
      write_lane(w, in.dst[0], lane, static_cast<uint32_t>(r));
    }
    write_scalar(w, in.dst[1], carried_out);
    return true;
  }

  // The same arithmetic, with the first source taken from another lane and
  // only some of the lanes written. Narrowing EXEC to those lanes is what
  // keeps the rest of them as they were, since every write here asks EXEC
  // first.
  void cross_lane_alu(Wave& w, const Inst& in) {
    std::array<uint32_t, kLanes> values{};
    const uint64_t writes = dpp_shuffle(w, in, values);
    const uint64_t saved = w.exec;
    dpp_operand = &in.src[0];
    dpp_values = &values;
    w.exec = writes;
    try {
      vector_alu(w, in);
    } catch (...) {
      w.exec = saved;
      dpp_operand = nullptr;
      dpp_values = nullptr;
      throw;
    }
    w.exec = saved;
    dpp_operand = nullptr;
    dpp_values = nullptr;
  }

  // Holds the destination's part for as long as the instruction runs.
  struct Narrowed {
    Machine& m;
    explicit Narrowed(Machine& machine, const Inst& in) : m(machine) {
      if (in.dst.empty()) return;
      if (in.dst[0].half) {
        // RDNA's 16-bit register halves (v1.l, v1.h): the result goes to its
        // half, and the other half is kept.
        m.narrow_dst = &in.dst[0];
        m.narrow_dst_sel = in.dst[0].hi ? 5 : 4;
        m.narrow_dst_preserve = true;
      } else if (in.sdwa && in.dst_sel != 6) {
        m.narrow_dst = &in.dst[0];
        m.narrow_dst_sel = in.dst_sel;
        m.narrow_dst_preserve = in.dst_unused == 2;
        m.narrow_dst_sext = in.dst_unused == 1;
      } else if (in.enc == gcn::Enc::Vop3 && (in.op_sel & 8) && in.name.find("fp8") == std::string::npos &&
                 in.name.find("bf8") == std::string::npos) {
        // A 16-bit instruction told by op_sel to write the high half of its
        // destination, the low half kept. (The 8-bit float conversions name
        // their part of the destination with these bits themselves.)
        m.narrow_dst = &in.dst[0];
        m.narrow_dst_sel = 5;
        m.narrow_dst_preserve = true;
      }
    }
    ~Narrowed() {
      m.narrow_dst = nullptr;
      m.narrow_dst_sel = 6;
      m.narrow_dst_preserve = false;
      m.narrow_dst_sext = false;
    }
  };

  // 2/pi's first 1216 bits, most significant first: what v_trig_preop
  // takes 53 at a time from. Computed exactly (Machin's formula, in integers).
  static constexpr uint32_t kTwoOverPi[38] = {
      0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0, 0xdb629599, 0x3c439041, 0xfe5163ab, 0xdebbc561,
      0xb7246e3a, 0x424dd2e0, 0x06492eea, 0x09d1921c, 0xfe1deb1c, 0xb129a73e, 0xe88235f5, 0x2ebb4484,
      0xe99c7026, 0xb45f7e41, 0x3991d639, 0x835339f4, 0x9c845f8b, 0xbdf9283b, 0x1ff897ff, 0xde05980f,
      0xef2f118b, 0x5a0a6d1f, 0x6d367ecf, 0x27cb09b7, 0x4f463f66, 0x9e5fea2d, 0x7527bac7, 0xebe5f17b,
      0x3d0739f7, 0x8a5292ea, 0x6bfb5fb1, 0x1f8d5d08, 0x56033046, 0xfc7b6bab};
  static uint64_t two_over_pi_bits(uint32_t from) {   // bits [from, from + 53), as a whole number
    uint64_t v = 0;
    for (uint32_t k = 0; k < 53; ++k) {
      const uint32_t bit = from + k;
      v = v << 1 | (bit / 32 < 38 ? (kTwoOverPi[bit / 32] >> (31 - bit % 32)) & 1 : 0);
    }
    return v;
  }

  // RDNA's own vector instructions, which gfx9 does not have: 16-bit logic,
  // selects and multiply-adds on register halves, the three-input min/max
  // and xor, and relative register addressing. A 16-bit source reads its
  // half of the register (sel) and the result goes to its half of the
  // destination (Narrowed). Returns whether `op` was one.
  // gfx1250's vector instructions that gfx12 does not have: 64-bit integer arithmetic, the three-operand
  // add-then-clamp forms, packed saturating conversions, tanh and the bfloat16 transcendentals, and the
  // 8-bit float conversions to halves. False for an instruction that is not one of them.
  bool cdna5_alu(Wave& w, const Inst& in, const OpName& op_in) {
    std::string name = op_in;
    if (name.size() > 4 && (name.compare(name.size() - 4, 4, "_e32") == 0 || name.compare(name.size() - 4, 4, "_e64") == 0))
      name.resize(name.size() - 4);
    const OpName op(name);
    const auto each = [&](auto&& body) {
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (w.exec >> lane & 1) body(lane);
    };
    const auto u = [&](uint32_t k, uint32_t lane) { return lane_src(w, in.src[k], lane); };
    const auto u64 = [&](uint32_t k, uint32_t lane) { return lane_src64(w, in.src[k], lane); };
    const auto bf = [&](uint32_t k, uint32_t lane) { return as_float((lane_src(w, in.src[k], lane) & 0xFFFFu) << 16); };
    const auto sat8 = [](int64_t v, bool is_signed) -> uint32_t {
      return static_cast<uint32_t>(is_signed ? std::clamp<int64_t>(v, -128, 127) : std::clamp<int64_t>(v, 0, 255)) & 0xFF;
    };
    if (op == "v_add_nc_u64"_op) {
      each([&](uint32_t lane) { write_lane64(w, in.dst[0], lane, u64(0, lane) + u64(1, lane)); });
    } else if (op == "v_sub_nc_u64"_op) {
      each([&](uint32_t lane) { write_lane64(w, in.dst[0], lane, u64(0, lane) - u64(1, lane)); });
    } else if (op == "v_mul_u64"_op) {
      each([&](uint32_t lane) { write_lane64(w, in.dst[0], lane, u64(0, lane) * u64(1, lane)); });
    } else if (op == "v_min_u64"_op || op == "v_max_u64"_op) {
      each([&](uint32_t lane) {
        const uint64_t a = u64(0, lane), b = u64(1, lane);
        write_lane64(w, in.dst[0], lane, op == "v_min_u64"_op ? std::min(a, b) : std::max(a, b));
      });
    } else if (op == "v_min_i64"_op || op == "v_max_i64"_op) {
      each([&](uint32_t lane) {
        const int64_t a = static_cast<int64_t>(u64(0, lane)), b = static_cast<int64_t>(u64(1, lane));
        write_lane64(w, in.dst[0], lane, static_cast<uint64_t>(op == "v_min_i64"_op ? std::min(a, b) : std::max(a, b)));
      });
    } else if (op == "v_mad_u32"_op) {
      each([&](uint32_t lane) { write_lane(w, in.dst[0], lane, u(0, lane) * u(1, lane) + u(2, lane)); });
    } else if (op == "v_mad_nc_u64_u32"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane, uint64_t{u(0, lane)} * u(1, lane) + u64(2, lane));
      });
    } else if (op == "v_mad_nc_i64_i32"_op) {
      each([&](uint32_t lane) {
        const int64_t p = int64_t{static_cast<int32_t>(u(0, lane))} * static_cast<int32_t>(u(1, lane));
        write_lane64(w, in.dst[0], lane, static_cast<uint64_t>(p + static_cast<int64_t>(u64(2, lane))));
      });
    } else if (op == "v_add_max_i32"_op || op == "v_add_min_i32"_op || op == "v_add_max_u32"_op || op == "v_add_min_u32"_op) {
      // The sum held to the type's range, then the larger or smaller of it and the third source.
      const bool is_signed = name.find("_i32") != std::string::npos, is_max = name.find("_max_") != std::string::npos;
      each([&](uint32_t lane) {
        if (is_signed) {
          const int64_t sum = std::clamp<int64_t>(int64_t{static_cast<int32_t>(u(0, lane))} + static_cast<int32_t>(u(1, lane)),
                                                  INT32_MIN, INT32_MAX);
          const int64_t c = static_cast<int32_t>(u(2, lane));
          write_lane(w, in.dst[0], lane, static_cast<uint32_t>(is_max ? std::max(sum, c) : std::min(sum, c)));
        } else {
          const uint64_t sum = std::min<uint64_t>(uint64_t{u(0, lane)} + u(1, lane), 0xFFFFFFFFu);
          const uint64_t c = u(2, lane);
          write_lane(w, in.dst[0], lane, static_cast<uint32_t>(is_max ? std::max(sum, c) : std::min(sum, c)));
        }
      });
    } else if (op == "v_ashr_pk_i8_i32"_op || op == "v_ashr_pk_u8_i32"_op) {
      const bool is_signed = op == "v_ashr_pk_i8_i32"_op;
      each([&](uint32_t lane) {
        const uint32_t n = u(2, lane) & 31;
        write_lane(w, in.dst[0], lane,
                   sat8(static_cast<int32_t>(u(0, lane)) >> n, is_signed) | sat8(static_cast<int32_t>(u(1, lane)) >> n, is_signed) << 8);
      });
    } else if (op == "v_sat_pk4_i4_i8"_op || op == "v_sat_pk4_u4_u8"_op) {
      const bool is_signed = op == "v_sat_pk4_i4_i8"_op;
      each([&](uint32_t lane) {
        const uint32_t x = u(0, lane);
        uint32_t r = 0;
        for (uint32_t k = 0; k < 4; ++k) {
          const int64_t b = is_signed ? static_cast<int8_t>(x >> (8 * k)) : static_cast<int64_t>((x >> (8 * k)) & 0xFF);
          r |= (static_cast<uint32_t>(is_signed ? std::clamp<int64_t>(b, -8, 7) : std::clamp<int64_t>(b, 0, 15)) & 0xF) << (4 * k);
        }
        write_lane(w, in.dst[0], lane, r);
      });
    } else if (op == "v_tanh_f32"_op) {
      each([&](uint32_t lane) { write_float(w, in, lane, std::tanh(lane_float(w, in.src[0], lane))); });
    } else if (op == "v_tanh_f16"_op) {
      each([&](uint32_t lane) { write_half(w, in, lane, static_cast<_Float16>(std::tanh(static_cast<float>(lane_half(w, in.src[0], lane))))); });
    } else if (op == "v_tanh_bf16"_op) {
      each([&](uint32_t lane) { write_lane(w, in.dst[0], lane, to_bf16(std::tanh(bf(0, lane)))); });
    } else if (op == "v_rcp_bf16"_op || op == "v_sqrt_bf16"_op || op == "v_rsq_bf16"_op || op == "v_log_bf16"_op ||
               op == "v_exp_bf16"_op || op == "v_sin_bf16"_op || op == "v_cos_bf16"_op) {
      // The operand is a bfloat16 and so is the result; sine and cosine take the angle in turns, as the f32 forms do.
      each([&](uint32_t lane) {
        const float x = bf(0, lane);
        float r;
        if (op == "v_rcp_bf16"_op) r = 1.0f / x;
        else if (op == "v_sqrt_bf16"_op) r = std::sqrt(x);
        else if (op == "v_rsq_bf16"_op) r = 1.0f / std::sqrt(x);
        else if (op == "v_log_bf16"_op) r = std::log2(x);
        else if (op == "v_exp_bf16"_op) r = std::exp2(x);
        else if (op == "v_sin_bf16"_op) r = std::sin(x * 6.283185307179586f);
        else r = std::cos(x * 6.283185307179586f);
        write_lane(w, in.dst[0], lane, to_bf16(r));
      });
    } else if (op == "v_cvt_f16_fp8"_op || op == "v_cvt_f16_bf8"_op) {
      const F8& t = op == "v_cvt_f16_fp8"_op ? fp8() : bf8();
      each([&](uint32_t lane) {
        write_half(w, in, lane, static_cast<_Float16>(f8_to_float(u(0, lane) & 0xFF, t)));
      });
    } else if (op == "v_cvt_pk_f16_fp8"_op || op == "v_cvt_pk_f16_bf8"_op) {
      const F8& t = op == "v_cvt_pk_f16_fp8"_op ? fp8() : bf8();
      each([&](uint32_t lane) {
        const auto h = [&](uint32_t byte) {
          const _Float16 v = static_cast<_Float16>(f8_to_float(byte, t));
          uint16_t b;
          std::memcpy(&b, &v, 2);
          return uint32_t{b};
        };
        const uint32_t x = u(0, lane);
        write_lane(w, in.dst[0], lane, h(x & 0xFF) | h((x >> 8) & 0xFF) << 16);
      });
    } else {
      return false;
    }
    return true;
  }

  bool rdna_alu(Wave& w, const Inst& in, const OpName& op_in) {
    std::string name = op_in;
    if (name.size() > 4 && (name.compare(name.size() - 4, 4, "_e32") == 0 || name.compare(name.size() - 4, 4, "_e64") == 0))
      name.resize(name.size() - 4);
    const OpName op(name);
    const auto each = [&](auto&& body) {
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (w.exec >> lane & 1) body(lane);
    };
    const auto src = [&](uint32_t k, uint32_t lane) { return lane_src(w, in.src[k], lane); };
    const auto u16 = [&](uint32_t k, uint32_t lane) { return static_cast<uint16_t>(src(k, lane)); };
    const auto i16 = [&](uint32_t k, uint32_t lane) { return static_cast<int16_t>(src(k, lane)); };
    const auto f32 = [&](uint32_t k, uint32_t lane) { return lane_float(w, in.src[k], lane); };
    const auto i32 = [&](uint32_t k, uint32_t lane) { return static_cast<int32_t>(src(k, lane)); };
    if (op == "v_and_b16"_op || op == "v_or_b16"_op || op == "v_xor_b16"_op) {
      each([&](uint32_t lane) {
        const uint16_t a = u16(0, lane), b = u16(1, lane);
        write_lane(w, in.dst[0], lane, op == "v_and_b16"_op ? a & b : op == "v_or_b16"_op ? a | b : a ^ b);
      });
    } else if (op == "v_not_b16"_op) {
      each([&](uint32_t lane) { write_lane(w, in.dst[0], lane, static_cast<uint16_t>(~u16(0, lane))); });
    } else if (op == "v_cndmask_b16"_op) {
      // The lane's bit of the mask (src2) picks the second source.
      const uint64_t cond = scalar(w, in.src[2]);
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_bits(w, (cond >> lane) & 1 ? in.src[1] : in.src[0], lane) & 0xFFFF);
      });
    } else if (op == "v_permlane16_b32"_op || op == "v_permlanex16_b32"_op) {
      // Each lane of a row of 16 reads the lane of its row (permlane16) or of
      // the other row of its pair (permlanex16) that its 4 bits of the
      // 64-bit selector {src2, src1} name. A source lane that is off is read
      // anyway where op_sel's first bit (FI) says so; otherwise the lane
      // gets zero where its second (BOUND_CTRL) says so, and keeps its value
      // where neither does.
      const uint64_t sel = (scalar(w, in.src[1]) & 0xFFFFFFFFu) | (scalar(w, in.src[2]) & 0xFFFFFFFFu) << 32;
      const bool fi = in.op_sel & 1, bound = (in.op_sel >> 1) & 1;
      const bool cross = op == "v_permlanex16_b32"_op;
      uint32_t from[kLanes];
      for (uint32_t lane = 0; lane < kLanes; ++lane) from[lane] = lane_src(w, in.src[0], lane);
      each([&](uint32_t lane) {
        const uint32_t row = lane / 16 ^ (cross ? 1 : 0);
        const uint32_t s = row * 16 + static_cast<uint32_t>((sel >> (4 * (lane % 16))) & 0xF);
        if (fi || (w.exec >> s & 1)) write_lane(w, in.dst[0], lane, from[s]);
        else if (bound) write_lane(w, in.dst[0], lane, 0);
      });
    } else if (op == "v_mad_u16"_op || op == "v_mad_i16"_op) {
      each([&](uint32_t lane) {
        const uint32_t v = op == "v_mad_u16"_op ? u16(0, lane) * u16(1, lane) + u16(2, lane)
                                                : static_cast<uint32_t>(i16(0, lane) * i16(1, lane) + i16(2, lane));
        write_lane(w, in.dst[0], lane, v & 0xFFFF);
      });
    } else if (op == "v_fmac_f16"_op || op == "v_fmaak_f16"_op || op == "v_fmamk_f16"_op) {
      // fmac: the destination is the addend; fmaak: the constant is; fmamk:
      // the constant is the second factor.
      each([&](uint32_t lane) {
        const float a = static_cast<float>(lane_half(w, in.src[0], lane));
        float b, c;
        if (op == "v_fmamk_f16"_op) {
          b = static_cast<float>(lane_half(w, in.src[1], lane));
          c = static_cast<float>(lane_half(w, in.src[2], lane));
          std::swap(b, c);   // v_fmamk_f16 d, a, K, c: a * K + c
        } else {
          b = static_cast<float>(lane_half(w, in.src[1], lane));
          if (op == "v_fmaak_f16"_op) {
            c = static_cast<float>(lane_half(w, in.src[2], lane));
          } else {
            Operand d = in.dst[0];
            d.sel = d.half ? (d.hi ? 5 : 4) : 6;
            c = static_cast<float>(lane_half(w, d, lane));
          }
        }
        write_half(w, in, lane, static_cast<_Float16>(std::fma(a, b, c)));
      });
    } else if (op == "v_maxmin_f32"_op || op == "v_minmax_f32"_op) {
      // Two steps, each the IEEE-style min or max of two.
      each([&](uint32_t lane) {
        const float ab = op == "v_maxmin_f32"_op ? std::fmax(f32(0, lane), f32(1, lane)) : std::fmin(f32(0, lane), f32(1, lane));
        write_float(w, in, lane, op == "v_maxmin_f32"_op ? std::fmin(ab, f32(2, lane)) : std::fmax(ab, f32(2, lane)));
      });
    } else if (op == "v_maxmin_i32"_op || op == "v_minmax_i32"_op || op == "v_maxmin_u32"_op || op == "v_minmax_u32"_op) {
      const bool max_first = op == "v_maxmin_i32"_op || op == "v_maxmin_u32"_op;
      const bool is_signed = op == "v_maxmin_i32"_op || op == "v_minmax_i32"_op;
      each([&](uint32_t lane) {
        uint32_t v;
        if (is_signed) {
          const int32_t ab = max_first ? std::max(i32(0, lane), i32(1, lane)) : std::min(i32(0, lane), i32(1, lane));
          v = static_cast<uint32_t>(max_first ? std::min(ab, i32(2, lane)) : std::max(ab, i32(2, lane)));
        } else {
          const uint32_t ab = max_first ? std::max(src(0, lane), src(1, lane)) : std::min(src(0, lane), src(1, lane));
          v = max_first ? std::min(ab, src(2, lane)) : std::max(ab, src(2, lane));
        }
        write_lane(w, in.dst[0], lane, v);
      });
    } else if (op.rfind("v_s_", 0) == 0) {
      // RDNA4's transcendentals on a scalar, into a scalar register: the
      // vector unit's reciprocal, square root, exponent and logarithm, once.
      // (A 16-bit one reads and writes the low half.)
      const bool half = op.rfind("_f16") == op.size() - 4;
      const uint32_t raw = static_cast<uint32_t>(scalar(w, in.src[0]));
      double x;
      if (half) {
        _Float16 h;
        const uint16_t b16 = static_cast<uint16_t>(raw);
        std::memcpy(&h, &b16, 2);
        x = static_cast<double>(h);
      } else {
        x = as_float(raw);
      }
      if (in.src[0].neg) x = -x;
      if (in.src[0].abs) x = std::fabs(x);
      const std::string f = op.substr(4, op.size() - 8);   // exp, log, rcp, rsq, sqrt
      const double r = f == "exp" ? std::exp2(x) : f == "log" ? std::log2(x) : f == "rcp" ? 1.0 / x
                       : f == "rsq" ? 1.0 / std::sqrt(x) : f == "sqrt" ? std::sqrt(x)
                       : throw Error::make(Err::Unsupported, in.name, " is decoded but not implemented");
      if (half) {
        const _Float16 h = static_cast<_Float16>(r);
        uint16_t b16;
        std::memcpy(&b16, &h, 2);
        write_scalar(w, in.dst[0], b16);
      } else {
        write_scalar(w, in.dst[0], as_bits(static_cast<float>(r)));
      }
    } else if (op.rfind("v_wmma_", 0) == 0) {
      wmma(w, in, op);
    } else if (op == "v_xor3_b32"_op) {
      each([&](uint32_t lane) { write_lane(w, in.dst[0], lane, src(0, lane) ^ src(1, lane) ^ src(2, lane)); });
    } else if (op == "v_movrels_b32"_op || op == "v_movreld_b32"_op || op == "v_movrelsd_b32"_op) {
      // Relative addressing: M0 moves the source register (movrels), the
      // destination (movreld) or both on.
      const uint32_t by = w.m0;
      each([&](uint32_t lane) {
        const uint32_t from = in.src[0].index + (op == "v_movreld_b32"_op ? 0 : by);
        const uint32_t to = in.dst[0].index + (op == "v_movrels_b32"_op ? 0 : by);
        if (from >= kVgprs || to >= kVgprs) throw Error::make(Err::InvalidValue, op, " past the vector registers");
        w.vgpr[to][lane] = in.src[0].kind == OperandKind::Vgpr ? w.vgpr[from][lane] : src(0, lane);
      });
    } else {
      return false;
    }
    return true;
  }

  // RDNA's matrix instructions (v_wmma_*): D = A x B + C for one 16x16 tile
  // across a wave32, as AMD lays it out.
  //   gfx11: lane l holds row l % 16 of A and column l % 16 of B, all of K
  //     (the two halves of the wave hold the same, and the first is read);
  //     register i of lane l is D[2i + l / 16][l % 16]; a 16-bit result goes
  //     to the half of each register OP_SEL's bit 2 names.
  //   gfx12: lane l holds row (column) l % 16 and half of K, the half l / 16;
  //     element i of lane l is D[i + 8 * (l / 16)][l % 16], 16-bit results
  //     two to a register.
  // The integer forms read their inputs signed where NEG's bit for them is
  // set, and CLAMP saturates; gfx12's 8-bit floats are the OCP formats. Each
  // output's products and addend are summed exactly (double) and rounded once.
  void wmma(Wave& w, const Inst& in, const OpName& op) {
    if (w.lanes != 32 || static_cast<uint32_t>(w.exec) != 0xFFFFFFFFu)
      throw Error::make(Err::Unsupported, in.name, " with lanes switched off, or in a wave64, which this does not model");
    const bool g12 = gcn::is_gfx12(in.arch);
    const std::string name = op;   // v_wmma_<out>_16x16x<K>_<a>[_<b>]
    const std::string out = name.substr(7, name.find('_', 7) - 7);
    const size_t shape_at = name.find("16x16x");
    const uint32_t K = static_cast<uint32_t>(std::stoul(name.substr(shape_at + 6)));
    const std::string types = name.substr(name.find('_', shape_at) + 1);          // "f16", "iu8", "fp8_bf8", ...
    const std::string ta = types.substr(0, types.find('_')), tb = types.find('_') == std::string::npos ? ta : types.substr(types.find('_') + 1);
    const bool a_signed = in.neg_lo & 1, b_signed = (in.neg_lo >> 1) & 1;
    const uint32_t per_lane = g12 ? K / 2 : K;
    const auto element = [&](const Operand& o, const std::string& t, uint32_t lane, uint32_t e, bool is_signed) -> double {
      if (t == "f16" || t == "bf16") {
        const uint32_t bits = (word(w, o, e / 2, lane) >> (16 * (e % 2))) & 0xFFFF;
        if (t == "bf16") return static_cast<double>(as_float(bits << 16));
        _Float16 h;
        const uint16_t b16 = static_cast<uint16_t>(bits);
        std::memcpy(&h, &b16, 2);
        return static_cast<double>(h);
      }
      if (t == "iu8" || t == "fp8" || t == "bf8") {
        const uint32_t byte = (word(w, o, e / 4, lane) >> (8 * (e % 4))) & 0xFF;
        if (t == "fp8") return f8_to_float(byte, kOcpFp8);
        if (t == "bf8") return f8_to_float(byte, kOcpBf8);
        return is_signed ? static_cast<int8_t>(byte) : byte;
      }
      const uint32_t nib = (word(w, o, e / 8, lane) >> (4 * (e % 8))) & 0xF;   // iu4
      return is_signed ? static_cast<int32_t>(nib << 28) >> 28 : nib;
    };
    // Where A[row][k] (B[k][col]) is: which lane, and which of its values.
    const auto input = [&](const Operand& o, const std::string& t, uint32_t rc, uint32_t k, bool sg) {
      return g12 ? element(o, t, rc + 16 * (k / per_lane), k % per_lane, sg) : element(o, t, rc, k, sg);
    };
    const bool half_out = out == "f16" || out == "bf16";
    const bool high = (in.op_sel >> 2) & 1;
    // Where D[row][col] is: the lane, and the register and half of it.
    const auto place = [&](uint32_t row, uint32_t col, uint32_t* lane, uint32_t* reg, uint32_t* half) {
      if (g12) {
        *lane = col + 16 * (row / 8);
        const uint32_t i = row % 8;
        *reg = half_out ? i / 2 : i;
        *half = half_out ? i % 2 : 0;
      } else {
        *lane = col + 16 * (row % 2);
        *reg = row / 2;
        *half = half_out && high ? 1 : 0;
      }
    };
    const auto from_half = [&](uint32_t bits) -> double {
      if (out == "bf16") return static_cast<double>(as_float(bits << 16));
      _Float16 h;
      const uint16_t b16 = static_cast<uint16_t>(bits);
      std::memcpy(&h, &b16, 2);
      return static_cast<double>(h);
    };
    const auto c_value = [&](uint32_t lane, uint32_t reg, uint32_t half) -> double {
      const Operand& c = in.src[2];
      if (c.kind != OperandKind::Vgpr) {
        if (c.kind == OperandKind::InlineFloat) return c.fvalue;
        return static_cast<double>(static_cast<int32_t>(scalar(w, c)));
      }
      const uint32_t v = word(w, c, reg, lane);
      if (out == "f32") return static_cast<double>(as_float(v));
      if (out == "i32") return static_cast<double>(static_cast<int32_t>(v));
      return from_half(half ? v >> 16 : v & 0xFFFF);
    };
    // The result, into a copy first: D may be C, or A or B.
    std::array<std::array<uint32_t, 8>, 32> result{};
    for (uint32_t lane = 0; lane < 32; ++lane)
      for (uint32_t r = 0; r < 8; ++r) result[lane][r] = word(w, in.dst[0], r, lane);
    for (uint32_t row = 0; row < 16; ++row)
      for (uint32_t col = 0; col < 16; ++col) {
        uint32_t lane, reg, half;
        place(row, col, &lane, &reg, &half);
        double sum = c_value(lane, reg, half);
        int64_t isum = static_cast<int64_t>(sum);
        for (uint32_t k = 0; k < K; ++k) {
          const double a = input(in.src[0], ta, row, k, a_signed), b = input(in.src[1], tb, col, k, b_signed);
          if (out == "i32") isum += static_cast<int64_t>(a) * static_cast<int64_t>(b);
          else sum += a * b;
        }
        uint32_t& slot = result[lane][reg];
        if (out == "i32") {
          if (in.clamp) isum = std::clamp<int64_t>(isum, INT32_MIN, INT32_MAX);
          slot = static_cast<uint32_t>(isum);
        } else if (out == "f32") {
          slot = as_bits(static_cast<float>(sum));
        } else {
          uint32_t bits;
          if (out == "bf16") {
            const uint32_t f = as_bits(static_cast<float>(sum));
            bits = std::isnan(as_float(f)) ? 0x7FC0 : (f + 0x7FFF + ((f >> 16) & 1)) >> 16;   // to nearest even
          } else {
            const _Float16 h = static_cast<_Float16>(sum);
            uint16_t b16;
            std::memcpy(&b16, &h, 2);
            bits = b16;
          }
          slot = half ? (slot & 0xFFFF) | bits << 16 : (slot & 0xFFFF0000u) | bits;
        }
      }
    const uint32_t regs = in.dst[0].width;
    for (uint32_t lane = 0; lane < 32; ++lane)
      for (uint32_t r = 0; r < regs && r < 8; ++r) set_word(w, in.dst[0], r, lane, result[lane][r]);
  }

  // The integer, 16-bit, half-precision and double instructions PyTorch's
  // libraries use beside the ones above. Returns whether `op` was one.
  bool more_alu(Wave& w, const Inst& in, const OpName& op) {
    const auto each = [&](auto&& body) {
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (w.exec >> lane & 1) body(lane);
    };
    const auto u16 = [&](uint32_t k, uint32_t lane) { return static_cast<uint16_t>(lane_src(w, in.src[k], lane)); };
    const auto i16 = [&](uint32_t k, uint32_t lane) { return static_cast<int16_t>(lane_src(w, in.src[k], lane)); };
    const auto half = [&](uint32_t k, uint32_t lane) { return static_cast<float>(lane_half(w, in.src[k], lane)); };
    if (op == "v_nop"_op) {
    } else if (op == "v_cvt_f32_ubyte1_e32"_op || op == "v_cvt_f32_ubyte2_e32"_op || op == "v_cvt_f32_ubyte3_e32"_op) {
      const uint32_t at = 8 * static_cast<uint32_t>(op.substr(15, 1)[0] - '0');   // "v_cvt_f32_ubyte1_e32"[15]
      each([&](uint32_t lane) {
        write_float(w, in, lane, static_cast<float>((lane_src(w, in.src[0], lane) >> at) & 0xFF));
      });
    } else if (op == "v_cvt_flr_i32_f32_e32"_op) {
      each([&](uint32_t lane) {
        const float f = std::floor(lane_float(w, in.src[0], lane));
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(std::isnan(f) ? 0 : f <= -2147483648.0f ? INT32_MIN
                                                          : f >= 2147483648.0f   ? INT32_MAX
                                                                                 : static_cast<int32_t>(f)));
      });
    } else if (op == "v_ffbl_b32_e32"_op) {
      each([&](uint32_t lane) {   // the lowest set bit, or -1
        const uint32_t v = lane_src(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane, v ? static_cast<uint32_t>(__builtin_ctz(v)) : 0xFFFFFFFFu);
      });
    } else if (op == "v_ffbh_i32_e32"_op) {
      each([&](uint32_t lane) {   // the first bit from the top unlike the sign, or -1
        const uint32_t v = lane_src(w, in.src[0], lane), x = v >> 31 ? ~v : v;
        write_lane(w, in.dst[0], lane, x ? static_cast<uint32_t>(__builtin_clz(x)) : 0xFFFFFFFFu);
      });
    } else if (op == "v_frexp_mant_f64_e32"_op || op == "v_frexp_exp_i32_f64_e32"_op) {
      each([&](uint32_t lane) {   // as the float forms: an infinity or a NaN is its own mantissa, exponent 0
        const double x = lane_double(w, in.src[0], lane);
        int e = 0;
        const double m = std::isfinite(x) ? std::frexp(x, &e) : x;
        if (!std::isfinite(x)) e = 0;
        if (op == "v_frexp_mant_f64_e32"_op) write_double(w, in, lane, m);
        else write_lane(w, in.dst[0], lane, static_cast<uint32_t>(e));
      });
    } else if (op == "v_fract_f64_e32"_op) {
      each([&](uint32_t lane) {
        // x - floor(x), held below 1: the largest double under 1 for a tiny
        // negative x, where the difference would round up to 1.
        const double x = lane_double(w, in.src[0], lane);
        const double r = std::isinf(x) ? std::numeric_limits<double>::quiet_NaN()
                                       : std::isnan(x) ? x : std::fmin(x - std::floor(x), 0x1.fffffffffffffp-1);
        write_double(w, in, lane, r);
      });
    } else if (op == "v_cvt_f16_u16_e32"_op || op == "v_cvt_f16_i16_e32"_op) {
      each([&](uint32_t lane) {
        write_half(w, in, lane,
                   static_cast<_Float16>(op == "v_cvt_f16_u16_e32"_op ? static_cast<float>(u16(0, lane))
                                                                     : static_cast<float>(i16(0, lane))));
      });
    } else if (op == "v_cvt_u16_f16_e32"_op || op == "v_cvt_i16_f16_e32"_op) {
      // Toward zero, held to what 16 bits hold, and a NaN to zero.
      each([&](uint32_t lane) {
        const float f = half(0, lane);
        const float lo = op == "v_cvt_u16_f16_e32"_op ? 0.0f : -32768.0f, hi = op == "v_cvt_u16_f16_e32"_op ? 65535.0f : 32767.0f;
        const int32_t v = std::isnan(f) ? 0 : static_cast<int32_t>(std::trunc(std::clamp(f, lo, hi)));
        write_lane(w, in.dst[0], lane, static_cast<uint16_t>(v));
      });
    } else if (op == "v_rcp_f16_e32"_op || op == "v_sqrt_f16_e32"_op || op == "v_rsq_f16_e32"_op ||
               op == "v_log_f16_e32"_op || op == "v_exp_f16_e32"_op || op == "v_floor_f16_e32"_op ||
               op == "v_ceil_f16_e32"_op || op == "v_trunc_f16_e32"_op || op == "v_rndne_f16_e32"_op) {
      // Worked out in float, which holds a half exactly, and rounded once
      // to a half. The log and the exponential are base 2.
      each([&](uint32_t lane) {
        const float x = half(0, lane);
        const float r = op == "v_rcp_f16_e32"_op    ? 1.0f / x
                        : op == "v_sqrt_f16_e32"_op ? std::sqrt(x)
                        : op == "v_rsq_f16_e32"_op  ? 1.0f / std::sqrt(x)
                        : op == "v_log_f16_e32"_op  ? std::log2(x)
                        : op == "v_exp_f16_e32"_op  ? std::exp2(x)
                        : op == "v_floor_f16_e32"_op ? std::floor(x)
                        : op == "v_ceil_f16_e32"_op ? std::ceil(x)
                        : op == "v_trunc_f16_e32"_op ? std::trunc(x)
                                                     : std::nearbyint(x);
        write_half(w, in, lane, static_cast<_Float16>(r));
      });
    } else if (op == "v_subrev_f32_e32"_op) {
      each([&](uint32_t lane) { write_float(w, in, lane, lane_float(w, in.src[1], lane) - lane_float(w, in.src[0], lane)); });
    } else if (op == "v_subrev_f16_e32"_op) {
      each([&](uint32_t lane) { write_half(w, in, lane, static_cast<_Float16>(half(1, lane) - half(0, lane))); });
    } else if (op == "v_subrev_u16_e32"_op) {
      each([&](uint32_t lane) { write_lane(w, in.dst[0], lane, static_cast<uint16_t>(u16(1, lane) - u16(0, lane))); });
    } else if (op == "v_mul_hi_u32_u24_e32"_op) {
      each([&](uint32_t lane) {
        const uint64_t p = uint64_t{lane_src(w, in.src[0], lane) & 0xFFFFFF} * (lane_src(w, in.src[1], lane) & 0xFFFFFF);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(p >> 32));
      });
    } else if (op == "v_ashrrev_i16_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, static_cast<uint16_t>(i16(1, lane) >> (lane_src(w, in.src[0], lane) & 15)));
      });
    } else if (op == "v_max_u16_e32"_op || op == "v_min_u16_e32"_op) {
      each([&](uint32_t lane) {
        const uint16_t x = u16(0, lane), y = u16(1, lane);
        write_lane(w, in.dst[0], lane, op == "v_max_u16_e32"_op ? std::max(x, y) : std::min(x, y));
      });
    } else if (op == "v_max_i16_e32"_op || op == "v_min_i16_e32"_op) {
      each([&](uint32_t lane) {
        const int16_t x = i16(0, lane), y = i16(1, lane);
        write_lane(w, in.dst[0], lane, static_cast<uint16_t>(op == "v_max_i16_e32"_op ? std::max(x, y) : std::min(x, y)));
      });
    } else if (op == "v_xnor_b32_e32"_op) {
      each([&](uint32_t lane) { write_lane(w, in.dst[0], lane, ~(lane_src(w, in.src[0], lane) ^ lane_src(w, in.src[1], lane))); });
    } else if (op == "v_med3_u32"_op) {
      each([&](uint32_t lane) {
        const uint32_t x = lane_src(w, in.src[0], lane), y = lane_src(w, in.src[1], lane), z = lane_src(w, in.src[2], lane);
        write_lane(w, in.dst[0], lane, std::max(std::min(x, y), std::min(std::max(x, y), z)));
      });
    } else if (op == "v_mad_u32_u16"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, uint32_t{u16(0, lane)} * u16(1, lane) + lane_src(w, in.src[2], lane));
      });
    } else if (op == "v_min3_i16"_op || op == "v_max3_i16"_op) {
      each([&](uint32_t lane) {
        const int16_t x = i16(0, lane), y = i16(1, lane), z = i16(2, lane);
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(op == "v_min3_i16"_op ? std::min({x, y, z}) : std::max({x, y, z})));
      });
    } else if (op == "v_min3_u16"_op || op == "v_max3_u16"_op) {
      each([&](uint32_t lane) {
        const uint16_t x = u16(0, lane), y = u16(1, lane), z = u16(2, lane);
        write_lane(w, in.dst[0], lane, op == "v_min3_u16"_op ? std::min({x, y, z}) : std::max({x, y, z}));
      });
    } else if (op == "v_div_fixup_f16"_op) {
      // As the float form, over halves: the quotient, with what the
      // division sequence cannot do filled in.
      each([&](uint32_t lane) {
        const float q = half(0, lane), den = half(1, lane), num = half(2, lane);
        const float inf = std::numeric_limits<float>::infinity();
        float out = q;
        if (std::isnan(num) || std::isnan(den)) out = std::numeric_limits<float>::quiet_NaN();
        else if (den == 0) out = num == 0 ? std::numeric_limits<float>::quiet_NaN()
                                          : std::copysign(inf, num) * std::copysign(1.0f, den);
        else if (std::isinf(den)) out = std::isinf(num) ? std::numeric_limits<float>::quiet_NaN()
                                                       : std::copysign(0.0f, num) * std::copysign(1.0f, den);
        else if (std::isinf(num)) out = std::copysign(inf, num) * std::copysign(1.0f, den);
        write_half(w, in, lane, static_cast<_Float16>(out));
      });
    } else if (op == "v_trig_preop_f64"_op) {
      // 53 bits of 2/pi, from the 53 * n'th on (n the second source's low
      // five bits), further on for a large first source, scaled so that the
      // pieces a kernel asks for add up to 2/pi.
      each([&](uint32_t lane) {
        const uint64_t bits = as_bits(lane_double(w, in.src[0], lane));
        const int32_t exponent = static_cast<int32_t>((bits >> 52) & 0x7FF);
        uint32_t shift = (lane_src(w, in.src[1], lane) & 31) * 53;
        if (exponent > 1077) shift += static_cast<uint32_t>(exponent - 1077);
        int scale = -53 - static_cast<int>(shift);
        if (exponent >= 1968) scale += 128;
        write_double(w, in, lane, std::ldexp(static_cast<double>(two_over_pi_bits(shift)), scale));
      });
    } else if (op == "v_bfm_b32"_op) {
      each([&](uint32_t lane) {
        const uint32_t width = lane_src(w, in.src[0], lane) & 31, at = lane_src(w, in.src[1], lane) & 31;
        write_lane(w, in.dst[0], lane, ((1u << width) - 1) << at);
      });
    } else if (op == "v_cvt_pk_u16_u32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   std::min(lane_src(w, in.src[0], lane), 0xFFFFu) | std::min(lane_src(w, in.src[1], lane), 0xFFFFu) << 16);
      });
    } else if (op == "v_add_i32"_op || op == "v_sub_i32"_op) {
      // Signed, wrapping, or held to the int32 range where the clamp is asked.
      each([&](uint32_t lane) {
        const int64_t x = static_cast<int32_t>(lane_src(w, in.src[0], lane)), y = static_cast<int32_t>(lane_src(w, in.src[1], lane));
        int64_t r = op == "v_add_i32"_op ? x + y : x - y;
        if (in.clamp) r = std::clamp<int64_t>(r, INT32_MIN, INT32_MAX);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(r));
      });
    } else if (op.rfind("v_pk_", 0) == 0 && (op.find("_u16") != std::string::npos || op.find("_i16") != std::string::npos ||
                                               op.find("_b16") != std::string::npos)) {
      // Two 16-bit integers in one register, each its own arithmetic.
      const std::string name = op;
      each([&](uint32_t lane) {
        uint32_t r = 0;
        for (uint32_t h = 0; h < 2; ++h) {
          const uint16_t x = packed_bits(w, in, 0, h, lane), y = packed_bits(w, in, 1, h, lane);
          const int16_t sx = static_cast<int16_t>(x), sy = static_cast<int16_t>(y);
          int32_t v;
          if (name == "v_pk_mul_lo_u16") v = x * y;
          else if (name == "v_pk_add_u16" || name == "v_pk_add_i16") v = x + y;
          else if (name == "v_pk_sub_u16" || name == "v_pk_sub_i16") v = x - y;
          else if (name == "v_pk_lshlrev_b16") v = y << (x & 15);
          else if (name == "v_pk_lshrrev_b16") v = y >> (x & 15);
          else if (name == "v_pk_ashrrev_i16") v = sy >> (x & 15);
          else if (name == "v_pk_max_i16") v = std::max(sx, sy);
          else if (name == "v_pk_min_i16") v = std::min(sx, sy);
          else if (name == "v_pk_max_u16") v = std::max(x, y);
          else if (name == "v_pk_min_u16") v = std::min(x, y);
          else if (name == "v_pk_mad_u16") v = x * y + packed_bits(w, in, 2, h, lane);
          else if (name == "v_pk_mad_i16") v = sx * sy + static_cast<int16_t>(packed_bits(w, in, 2, h, lane));
          else throw Error::make(Err::Unsupported, name, " is decoded but not implemented");
          // The clamp holds an add or a subtract to what 16 bits hold.
          if (in.clamp && (name.find("add") != std::string::npos || name.find("sub") != std::string::npos ||
                           name.find("mad") != std::string::npos))
            v = name.find("_i16") != std::string::npos ? std::clamp(v, -32768, 32767) : std::clamp(v, 0, 65535);
          r |= uint32_t{static_cast<uint16_t>(v)} << (16 * h);
        }
        write_lane(w, in.dst[0], lane, r);
      });
    } else if (op == "v_dot2_f32_f16"_op) {
      // Two pairs of halves multiplied and added into a float: each product
      // exact in double, the sum rounded once.
      // neg_lo negates a source's low half and neg_hi its high one; the
      // addend is negated by its neg_lo bit.
      each([&](uint32_t lane) {
        const float c = lane_float(w, in.src[2], lane);
        double sum = (in.neg_lo >> 2) & 1 ? -double(c) : double(c);
        for (uint32_t h = 0; h < 2; ++h) {
          _Float16 a, b;
          const uint16_t ab = packed_bits(w, in, 0, h, lane), bb = packed_bits(w, in, 1, h, lane);
          std::memcpy(&a, &ab, 2);
          std::memcpy(&b, &bb, 2);
          const uint8_t neg = h ? in.neg_hi : in.neg_lo;
          sum += (neg & 1 ? -double(a) : double(a)) * (neg & 2 ? -double(b) : double(b));
        }
        // Its clamp bit does not hold the result to [0, 1], as other float
        // results' does: hip-tests' amd_mixed_dot({1, 3}, {3, 3}, 2, true)
        // is 14 on a card, and the compiler takes the clamped form for the
        // unclamped one.
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(sum)));
      });
    } else if (op == "v_swap_b32"_op) {
      each([&](uint32_t lane) {
        const uint32_t a = w.vgpr[in.dst[0].index][lane], b = w.vgpr[in.src[0].index][lane];
        w.vgpr[in.dst[0].index][lane] = b;
        w.vgpr[in.src[0].index][lane] = a;
      });
    } else if (op == "v_permlane32_swap_b32_e32"_op || op == "v_permlane32_swap_b32_e64"_op ||
               op == "v_permlane16_swap_b32_e32"_op || op == "v_permlane16_swap_b32_e64"_op) {
      // gfx950: the upper 32 lanes of the first register trade places with
      // the lower 32 of the second; or, 16 lanes to a row, each odd row of
      // the first with the even row before it in the second. A pair is
      // swapped where both its lanes are on.
      // Every lane, whatever EXEC says, as the ISA's pseudocode has it.
      const bool rows32 = op.find("permlane32") != std::string::npos;
      auto& d = w.vgpr[in.dst[0].index];
      auto& v = w.vgpr[in.src[0].index];
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const bool first = rows32 ? lane >= 32 : (lane / 16) % 2 == 1;
        if (first) std::swap(d[lane], v[rows32 ? lane - 32 : lane - 16]);
      }
    } else if (op == "v_prng_b32_e32"_op || op == "v_prng_b32_e64"_op) {
      // One step of the LFSR the CDNA4 ISA gives: shift left, and where the
      // top bit fell off, fold in 197.
      each([&](uint32_t lane) {
        const uint32_t x = lane_src(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane, (x << 1) ^ (x >> 31 ? 197u : 0u));
      });
    } else if (op == "v_cvt_f32_bf16_e32"_op || op == "v_cvt_f32_bf16_e64"_op) {
      // gfx950: a bfloat16 (the low half, or the high one op_sel names) as
      // the float it is the top half of.
      each([&](uint32_t lane) {
        const uint32_t x = lane_src(w, in.src[0], lane) >> (16 * (in.op_sel & 1));
        write_float(w, in, lane, as_float((x & 0xFFFF) << 16));
      });
    } else if (op == "v_accvgpr_mov_b32"_op) {
      each([&](uint32_t lane) { set_word(w, in.dst[0], 0, lane, lane_src(w, in.src[0], lane)); });
    } else {
      return false;
    }
    return true;
  }

  void vector_alu(Wave& w, const Inst& in) {
    // The sub-dword form of an instruction does what the short form does,
    // over the part of each register it names, and the cross-lane form does
    // it over the lanes it named, so both are the same arithmetic under the
    // name the short form has.
    const std::string as_short =
        in.sdwa   ? in.name.substr(0, in.name.size() - 5) + "_e32"
        : in.dpp  ? in.name.substr(0, in.name.size() - 4) + "_e32"
        : in.promoted ? in.name.substr(0, in.name.size() - 4) + "_e32"
        : in.name.size() > 4 && in.name.compare(in.name.size() - 4, 4, "_e64") == 0 &&
                (in.name.find("_u16") != std::string::npos || in.name.find("_b16") != std::string::npos)
                  ? in.name.substr(0, in.name.size() - 4) + "_e32"
                  : std::string();
    const OpName op(as_short.empty() ? in.name : as_short);
    // (A sub-dword instruction may write part of its destination and pad the
    // rest with zeroes, carry the part's sign into it, or keep what was
    // there: write_lane does each.)
    // op_sel on a 16-bit long form: a source's bit reads its high half,
    // which is the sub-dword selection SDWA makes, so the instruction runs
    // as that. (The 8-bit float conversions and v_pack read their own bits;
    // RDNA's v_permlane16 and v_permlanex16 take op_sel as FI and
    // BOUND_CTRL. Rewritten as halves, a permlanex16 with FI set -- Triton's
    // row maximum across a wave32, in vLLM's attention -- moved each lane's
    // high half, and -inf arrived as a tiny positive number.)
    if (in.enc == gcn::Enc::Vop3 && (in.op_sel & 7) && in.name.find("fp8") == std::string::npos &&
        in.name.find("bf8") == std::string::npos && in.name.rfind("v_permlane", 0) != 0) {
      const bool wide_third = in.name.find("u32_u16") != std::string::npos;   // v_mad_u32_u16's addend is 32 bits
      Inst x = in;
      for (uint32_t k = 0; k < x.src.size() && k < 3; ++k)
        if ((in.op_sel >> k) & 1) {
          if (k == 2 && wide_third) continue;
          if (x.src[k].kind != OperandKind::Vgpr)
            throw Error::make(Err::Unsupported, in.name, " takes the high half of a constant, which this does not model");
          x.src[k].sel = 5;
        }
      x.op_sel &= 8;
      vector_alu(w, x);
      return;
    }
    Narrowed narrowed(*this, in);
    if (in.arch == gcn::Target::Gfx1250 && cdna5_alu(w, in, op)) return;
    if (gcn::is_rdna(in.arch) && rdna_alu(w, in, op)) return;
    if (carry_alu(w, in)) return;
    if (more_alu(w, in, op)) return;
    if (op == "v_mov_b16_e32"_op || op == "v_mov_b16_e64"_op || op == "v_mov_b16"_op) {
      // RDNA: a half register, or the low half of a constant, into a half.
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (w.exec >> lane & 1) write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) & 0xFFFF);
      return;
    }
    if (op == "v_writelane_b32"_op) {
      // The one instruction here that names the lane it writes: a scalar
      // value into one lane of a register, whatever EXEC says.
      const uint32_t lane = static_cast<uint32_t>(scalar(w, in.src[1])) & 63;
      w.vgpr[in.dst[0].index][lane] = static_cast<uint32_t>(scalar(w, in.src[0]));
      return;
    }
    if (op == "v_readlane_b32"_op) {
      // And its reverse, one lane into a scalar register, which is just as
      // blind to EXEC. Unoptimized code spills scalars into lanes and reads
      // them back with every lane off, on the way out of a branch nobody took.
      const uint32_t which = static_cast<uint32_t>(scalar(w, in.src[1])) & (w.lanes - 1);
      write_scalar(w, in.dst[0], w.vgpr[in.src[0].index][which]);
      return;
    }
    if (op == "v_readfirstlane_b32"_op) {
      // The first active lane; with none active, lane 0.
      write_scalar(w, in.dst[0], w.vgpr[in.src[0].index][first_active(w)]);
      return;
    }
    // Which operation it is is decided once; its body then runs for each
    // lane EXEC names.
    const auto each = [&](auto&& body) {
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (w.exec >> lane & 1) body(lane);
    };
    if (op == "v_mov_b32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane));
      });
    } else if (op == "v_add_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, lane_float(w, in.src[0], lane) + lane_float(w, in.src[1], lane));
      });
    } else if (op == "v_mul_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, lane_float(w, in.src[0], lane) * lane_float(w, in.src[1], lane));
      });
    } else if (op == "v_lshlrev_b32_e32"_op) {
      each([&](uint32_t lane) {
        // The "rev" forms shift the second source by the first.
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 31));
      });
    } else if (op == "v_lshrrev_b32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) >> (lane_src(w, in.src[0], lane) & 31));
      });
    } else if (op == "v_ashrrev_i32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(static_cast<int32_t>(lane_src(w, in.src[1], lane)) >>
                                         (lane_src(w, in.src[0], lane) & 31)));
      });
    } else if (op == "v_lshlrev_b64"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane, lane_src64(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 63));
      });
    } else if (op == "v_lshl_add_u32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) << (lane_src(w, in.src[1], lane) & 31)) +
                       lane_src(w, in.src[2], lane));
      });
    } else if (op == "v_lshl_add_u64"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane,
                     (lane_src64(w, in.src[0], lane) << (lane_src(w, in.src[1], lane) & 63)) +
                         lane_src64(w, in.src[2], lane));
      });
    } else if (op == "v_ashrrev_i64"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane,
                     static_cast<uint64_t>(static_cast<int64_t>(lane_src64(w, in.src[1], lane)) >>
                                           (lane_src(w, in.src[0], lane) & 63)));
      });
    } else if (op == "v_and_b32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) & lane_src(w, in.src[1], lane));
      });
    } else if (op == "v_or_b32_e32"_op || op == "v_or_b32_e64"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) | lane_src(w, in.src[1], lane));
      });
    } else if (op == "v_xor_b32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) ^ lane_src(w, in.src[1], lane));
      });
    } else if (op == "v_add_u32_e32"_op || op == "v_add_u32_e64"_op) {
      // The VOP3 form's clamp saturates: an unsigned sum past 32 bits is
      // their maximum (as __clzll's sum of two leading-zero counts relies on).
      each([&](uint32_t lane) {
        const uint64_t sum = uint64_t{lane_src(w, in.src[0], lane)} + lane_src(w, in.src[1], lane);
        write_lane(w, in.dst[0], lane, in.clamp && sum > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(sum));
      });
    } else if (op == "v_sub_u32_e32"_op || op == "v_sub_u32_e64"_op) {
      // Clamped, a difference below zero is zero.
      each([&](uint32_t lane) {
        const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
        write_lane(w, in.dst[0], lane, in.clamp && b > a ? 0u : a - b);
      });
    } else if (op == "v_add3_u32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   lane_src(w, in.src[0], lane) + lane_src(w, in.src[1], lane) + lane_src(w, in.src[2], lane));
      });
    } else if (op == "v_mul_lo_u32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) * lane_src(w, in.src[1], lane));
      });
    } else if (op == "v_mul_hi_i32"_op) {
      each([&](uint32_t lane) {
        const int64_t p = static_cast<int64_t>(static_cast<int32_t>(lane_src(w, in.src[0], lane))) *
                          static_cast<int32_t>(lane_src(w, in.src[1], lane));
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(static_cast<uint64_t>(p) >> 32));
      });
    } else if (op == "v_sub_u16_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[0], lane) - lane_src(w, in.src[1], lane)));
      });
    } else if (op == "v_add_u16_e32"_op) {
      each([&](uint32_t lane) {
        // 16-bit arithmetic writes the low half of the destination and zeroes
        // the high half: the compiler leaves out the mask a widening would
        // otherwise need after one of these.
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[0], lane) + lane_src(w, in.src[1], lane)));
      });
    } else if (op == "v_lshlrev_b16_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 15)));
      });
    } else if (op == "v_mad_legacy_u16"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[0], lane) * lane_src(w, in.src[1], lane) +
                                         lane_src(w, in.src[2], lane)));
      });
    } else if (op == "v_accvgpr_read_b32"_op || op == "v_accvgpr_write_b32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane));
      });
    } else if (op == "v_cvt_f64_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane, as_bits(static_cast<double>(lane_float(w, in.src[0], lane))));
      });
    } else if (op == "v_cvt_f64_i32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane,
                     as_bits(static_cast<double>(static_cast<int32_t>(lane_src(w, in.src[0], lane)))));
      });
    } else if (op == "v_cvt_f64_u32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane, as_bits(static_cast<double>(lane_src(w, in.src[0], lane))));
      });
    } else if (op == "v_trunc_f64_e32"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane, std::trunc(lane_double(w, in.src[0], lane)));
      });
    } else if (op == "v_ceil_f64_e32"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane, std::ceil(lane_double(w, in.src[0], lane)));
      });
    } else if (op == "v_floor_f64_e32"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane, std::floor(lane_double(w, in.src[0], lane)));
      });
    } else if (op == "v_rndne_f64_e32"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane, std::nearbyint(lane_double(w, in.src[0], lane)));
      });
    } else if (op == "v_rsq_f64_e32"_op) {
      each([&](uint32_t lane) {
        // As with the reciprocal: the hardware's is a table and this is the
        // exact one, so the refinement the compiler builds around it lands on
        // the same answer either way.
        write_double(w, in, lane, 1.0 / std::sqrt(lane_double(w, in.src[0], lane)));
      });
    } else if (op == "v_min_f64"_op || op == "v_max_f64"_op) {
      each([&](uint32_t lane) {
        const double x = lane_double(w, in.src[0], lane), y = lane_double(w, in.src[1], lane);
        write_double(w, in, lane, op == "v_min_f64"_op ? std::fmin(x, y) : std::fmax(x, y));
      });
    } else if (op == "v_ldexp_f64"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane,
                     std::ldexp(lane_double(w, in.src[0], lane),
                                static_cast<int32_t>(lane_src(w, in.src[1], lane))));
      });
    } else if (op == "v_mov_b64_e32"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane, lane_src64(w, in.src[0], lane));
      });
    } else if (op == "v_floor_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, std::floor(lane_float(w, in.src[0], lane)));
      });
    } else if (op == "v_ceil_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, std::ceil(lane_float(w, in.src[0], lane)));
      });
    } else if (op == "v_fract_f32_e32"_op) {
      each([&](uint32_t lane) {
        // x - floor(x), held below 1 as v_fract_f64 is; an infinity is a NaN.
        const float x = lane_float(w, in.src[0], lane);
        write_float(w, in, lane,
                    std::isinf(x) ? std::numeric_limits<float>::quiet_NaN()
                                  : std::isnan(x) ? x : std::fmin(x - std::floor(x), 0x1.fffffep-1f));
      });
    } else if (op == "v_fract_f16_e32"_op) {
      each([&](uint32_t lane) {
        const float x = static_cast<float>(lane_half(w, in.src[0], lane));
        const float r = std::isinf(x) ? std::numeric_limits<float>::quiet_NaN()
                                      : std::isnan(x) ? x : std::fmin(x - std::floor(x), 0x1.ffcp-1f);
        write_half(w, in, lane, static_cast<_Float16>(r));
      });
    } else if (op == "v_rndne_f32_e32"_op) {
      each([&](uint32_t lane) {
        // To the nearest, and to the even one where it falls in the middle.
        write_float(w, in, lane, std::nearbyint(lane_float(w, in.src[0], lane)));
      });
    } else if (op == "v_sin_f32_e32"_op || op == "v_cos_f32_e32"_op) {
      each([&](uint32_t lane) {
        // The argument is in turns: a whole turn is 1.0, not 2pi.
        const double turns = static_cast<double>(lane_float(w, in.src[0], lane));
        const double radians = turns * 6.283185307179586476925286766559;
        write_float(w, in, lane,
                    static_cast<float>(op == "v_sin_f32_e32"_op ? std::sin(radians) : std::cos(radians)));
      });
    } else if (op == "v_ldexp_f32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane,
                    std::ldexp(lane_float(w, in.src[0], lane),
                               static_cast<int32_t>(lane_src(w, in.src[1], lane))));
      });
    } else if (op == "v_min3_f32"_op || op == "v_max3_f32"_op) {
      each([&](uint32_t lane) {
        const float x = lane_float(w, in.src[0], lane), y = lane_float(w, in.src[1], lane),
                    z = lane_float(w, in.src[2], lane);
        write_float(w, in, lane, op == "v_min3_f32"_op ? std::fmin(std::fmin(x, y), z)
                                                    : std::fmax(std::fmax(x, y), z));
      });
    } else if (op == "v_med3_f32"_op) {
      each([&](uint32_t lane) {
        // The middle one of three; with a NaN among them, the least, as
        // min3 gives it (a NaN losing to any number).
        const float x = lane_float(w, in.src[0], lane), y = lane_float(w, in.src[1], lane),
                    z = lane_float(w, in.src[2], lane);
        write_float(w, in, lane,
                    std::isnan(x) || std::isnan(y) || std::isnan(z)
                        ? std::fmin(std::fmin(x, y), z)
                        : std::fmax(std::fmin(x, y), std::fmin(std::fmax(x, y), z)));
      });
    } else if (op == "v_cvt_f32_fp8_e32"_op || op == "v_cvt_f32_bf8_e32"_op ||
               op == "v_cvt_pk_f32_fp8_e32"_op || op == "v_cvt_pk_f32_bf8_e32"_op) {
      // One 8-bit float widened, from the source's low byte, or two from its
      // low half -- the part SDWA picked, or the register's bottom.
      if (in.promoted && in.op_sel)
        throw Error::make(Err::Unsupported, in.name, " picks its byte with op_sel, which this does not model");
      const F8& t = op == "v_cvt_f32_fp8_e32"_op || op == "v_cvt_pk_f32_fp8_e32"_op ? fp8() : bf8();
      const bool pair = op == "v_cvt_pk_f32_fp8_e32"_op || op == "v_cvt_pk_f32_bf8_e32"_op;
      each([&](uint32_t lane) {
        const uint32_t v = lane_src(w, in.src[0], lane);
        set_word(w, in.dst[0], 0, lane, as_bits(f8_to_float(v, t)));
        if (pair) set_word(w, in.dst[0], 1, lane, as_bits(f8_to_float(v >> 8, t)));
      });
    } else if (op == "v_cvt_pk_fp8_f32"_op || op == "v_cvt_pk_bf8_f32"_op) {
      // Two floats narrowed into one half of the destination (op_sel's
      // bit 3 says the high one), the other half kept.
      const F8& t = op == "v_cvt_pk_fp8_f32"_op ? fp8() : bf8();
      each([&](uint32_t lane) {
        const uint32_t two = float_to_f8(lane_float(w, in.src[0], lane), t, in.clamp, nullptr) |
                             float_to_f8(lane_float(w, in.src[1], lane), t, in.clamp, nullptr) << 8;
        const uint32_t was = w.vgpr[in.dst[0].index][lane];
        write_lane(w, in.dst[0], lane, in.op_sel & 8 ? (was & 0xFFFFu) | two << 16 : (was & 0xFFFF0000u) | two);
      });
    } else if (op == "v_cvt_sr_fp8_f32"_op || op == "v_cvt_sr_bf8_f32"_op) {
      // One float narrowed, rounded by the second source's random bits,
      // into the byte op_sel's bits 2 and 3 name, the others kept.
      const F8& t = op == "v_cvt_sr_fp8_f32"_op ? fp8() : bf8();
      const uint32_t at = 8 * ((in.op_sel >> 2) & 3);
      each([&](uint32_t lane) {
        const uint32_t random = lane_src(w, in.src[1], lane);
        const uint32_t b = float_to_f8(lane_float(w, in.src[0], lane), t, in.clamp, &random);
        const uint32_t was = w.vgpr[in.dst[0].index][lane];
        write_lane(w, in.dst[0], lane, (was & ~(0xFFu << at)) | b << at);
      });
    } else if (op == "v_min3_i32"_op || op == "v_max3_i32"_op) {
      each([&](uint32_t lane) {
        const int32_t x = static_cast<int32_t>(lane_src(w, in.src[0], lane)),
                      y = static_cast<int32_t>(lane_src(w, in.src[1], lane)),
                      z = static_cast<int32_t>(lane_src(w, in.src[2], lane));
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(op == "v_min3_i32"_op ? std::min(std::min(x, y), z)
                                                            : std::max(std::max(x, y), z)));
      });
    } else if (op == "v_perm_b32"_op) {
      each([&](uint32_t lane) {
        // Four bytes chosen out of the eight the two sources make, the first
        // source holding the top four. Past those, a selector asks for the
        // sign of byte 1, 3, 5 or 7 spread over a byte (8 to 11), a zero
        // byte (12), or a byte of ones (13 on): Tensile's int8 GEMMs widen
        // bytes with them.
        const uint64_t bytes = static_cast<uint64_t>(lane_src(w, in.src[0], lane)) << 32 |
                               lane_src(w, in.src[1], lane);
        const uint32_t sel = lane_src(w, in.src[2], lane);
        uint32_t out = 0;
        for (uint32_t k = 0; k < 4; ++k) {
          const uint32_t which = (sel >> (8 * k)) & 0xFF;
          const uint32_t b = which < 8    ? (bytes >> (8 * which)) & 0xFF
                             : which < 12 ? ((bytes >> (8 * (2 * (which - 8) + 1) + 7)) & 1) * 0xFFu
                             : which == 12 ? 0u
                                           : 0xFFu;
          out |= b << (8 * k);
        }
        write_lane(w, in.dst[0], lane, out);
      });
    } else if (op == "v_or3_b32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   lane_src(w, in.src[0], lane) | lane_src(w, in.src[1], lane) | lane_src(w, in.src[2], lane));
      });
    } else if (op == "v_min_i32_e32"_op || op == "v_max_i32_e32"_op) {
      each([&](uint32_t lane) {
        const int32_t x = static_cast<int32_t>(lane_src(w, in.src[0], lane)),
                      y = static_cast<int32_t>(lane_src(w, in.src[1], lane));
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(op == "v_min_i32_e32"_op ? std::min(x, y) : std::max(x, y)));
      });
    } else if (op == "v_dot2_i32_i16"_op || op == "v_dot2_u32_u16"_op) {
      // Two pairs of 16-bit values (halves chosen as a packed instruction's
      // are) multiplied and added to the third source; clamped, the sum
      // saturates rather than wraps.
      const bool is_signed = op == "v_dot2_i32_i16"_op;
      each([&](uint32_t lane) {
        const uint32_t c = lane_src(w, in.src[2], lane);
        int64_t sum = is_signed ? int64_t{static_cast<int32_t>(c)} : int64_t{c};
        for (uint32_t h = 0; h < 2; ++h) {
          const uint16_t a = packed_bits(w, in, 0, h, lane), b = packed_bits(w, in, 1, h, lane);
          sum += is_signed ? int64_t{static_cast<int16_t>(a)} * static_cast<int16_t>(b) : int64_t{a} * b;
        }
        if (in.clamp) sum = is_signed ? std::clamp<int64_t>(sum, INT32_MIN, INT32_MAX) : std::clamp<int64_t>(sum, 0, UINT32_MAX);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(sum));
      });
    } else if (op == "v_dot4_u32_u8"_op || op == "v_dot8_i32_i4"_op || op == "v_dot8_u32_u4"_op) {
      // Four unsigned bytes, or eight 4-bit values, of each source multiplied
      // pairwise and added to the third.
      const bool nibbles = op != "v_dot4_u32_u8"_op, is_signed = op == "v_dot8_i32_i4"_op;
      const uint32_t bits = nibbles ? 4 : 8, n = 32 / bits, mask = (1u << bits) - 1;
      each([&](uint32_t lane) {
        const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
        const uint32_t c = lane_src(w, in.src[2], lane);
        int64_t sum = is_signed ? int64_t{static_cast<int32_t>(c)} : int64_t{c};
        for (uint32_t k = 0; k < n; ++k) {
          int64_t x = (a >> (bits * k)) & mask, y = (b >> (bits * k)) & mask;
          if (is_signed) {
            if (x & 8) x -= 16;
            if (y & 8) y -= 16;
          }
          sum += x * y;
        }
        if (in.clamp) sum = is_signed ? std::clamp<int64_t>(sum, INT32_MIN, INT32_MAX) : std::clamp<int64_t>(sum, 0, UINT32_MAX);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(sum));
      });
    } else if (op == "v_dot4_i32_i8"_op) {
      // Four signed bytes of each multiplied pairwise and added to the third
      // source; clamped, the sum saturates rather than wraps.
      each([&](uint32_t lane) {
        const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
        int64_t sum = static_cast<int32_t>(lane_src(w, in.src[2], lane));
        for (uint32_t k = 0; k < 4; ++k)
          sum += static_cast<int64_t>(static_cast<int8_t>(a >> (8 * k))) * static_cast<int8_t>(b >> (8 * k));
        if (in.clamp) sum = std::clamp<int64_t>(sum, INT32_MIN, INT32_MAX);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(sum));
      });
    } else if (op == "v_min3_u32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   std::min({lane_src(w, in.src[0], lane), lane_src(w, in.src[1], lane), lane_src(w, in.src[2], lane)}));
      });
    } else if (op == "v_lshrrev_b16_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[1], lane)) >> (lane_src(w, in.src[0], lane) & 15));
      });
    } else if (op == "v_min_f16_e32"_op || op == "v_max_f16_e32"_op) {
      // Where one is a NaN the other is kept, as fmin and fmax keep it.
      each([&](uint32_t lane) {
        const float x = static_cast<float>(lane_half(w, in.src[0], lane)),
                    y = static_cast<float>(lane_half(w, in.src[1], lane));
        write_half(w, in, lane, static_cast<_Float16>(op == "v_min_f16_e32"_op ? std::fmin(x, y) : std::fmax(x, y)));
      });
    } else if (op == "v_cvt_u32_f64_e32"_op) {
      // Toward zero, held to what 32 unsigned bits hold, and a NaN to zero.
      each([&](uint32_t lane) {
        const double d = lane_double(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane,
                   !(d > 0) ? 0u : d >= 4294967295.0 ? 0xFFFFFFFFu : static_cast<uint32_t>(d));
      });
    } else if (op == "v_mad_i32_i24"_op) {
      // The low 24 bits of each factor, signed.
      each([&](uint32_t lane) {
        const auto i24 = [](uint32_t v) { return static_cast<int32_t>(v << 8) >> 8; };
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(i24(lane_src(w, in.src[0], lane)) * i24(lane_src(w, in.src[1], lane)) +
                                         static_cast<int32_t>(lane_src(w, in.src[2], lane))));
      });
    } else if (op == "v_mad_i64_i32"_op) {
      sdst_bits = 0;
      each([&](uint32_t lane) {
        // A signed 32x32 product added to a signed 64-bit value, and whether
        // that overflowed.
        const __int128 p = static_cast<__int128>(static_cast<int32_t>(lane_src(w, in.src[0], lane))) *
                               static_cast<int32_t>(lane_src(w, in.src[1], lane)) +
                           static_cast<int64_t>(lane_src64(w, in.src[2], lane));
        write_lane64(w, in.dst[0], lane, static_cast<uint64_t>(p));
        if (p != static_cast<int64_t>(p)) sdst_bits |= uint64_t{1} << lane;
      });
      if (in.dst.size() > 1) write_scalar(w, in.dst[1], sdst_bits);
    } else if (op == "v_add_f16_e32"_op) {
      each([&](uint32_t lane) {
        write_half(w, in, lane, lane_half(w, in.src[0], lane) + lane_half(w, in.src[1], lane));
      });
    } else if (op == "v_sub_f16_e32"_op) {
      each([&](uint32_t lane) {
        write_half(w, in, lane, lane_half(w, in.src[0], lane) - lane_half(w, in.src[1], lane));
      });
    } else if (op == "v_mul_f16_e32"_op) {
      each([&](uint32_t lane) {
        write_half(w, in, lane, lane_half(w, in.src[0], lane) * lane_half(w, in.src[1], lane));
      });
    } else if (op == "v_fma_f16"_op) {
      each([&](uint32_t lane) {
        // A fused multiply-add: one rounding, which is what the C means by
        // fma and what the half the compiler folded into it expects.
        write_half(w, in, lane,
                   static_cast<_Float16>(std::fma(static_cast<float>(lane_half(w, in.src[0], lane)),
                                                  static_cast<float>(lane_half(w, in.src[1], lane)),
                                                  static_cast<float>(lane_half(w, in.src[2], lane)))));
      });
    } else if (op == "v_cvt_f16_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_half(w, in, lane, static_cast<_Float16>(lane_float(w, in.src[0], lane)));
      });
    } else if (op == "v_cvt_f32_f16_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(lane_half(w, in.src[0], lane))));
      });
    } else if (op == "v_mul_i32_i24_e32"_op) {
      each([&](uint32_t lane) {
        // The low 24 bits of each source, as signed numbers; the product wraps, so it is formed unsigned.
        const auto i24 = [](uint32_t v) { return static_cast<int32_t>(v << 8) >> 8; };
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(i24(lane_src(w, in.src[0], lane))) * static_cast<uint32_t>(i24(lane_src(w, in.src[1], lane))));
      });
    } else if (op == "v_mul_hi_i32_i24_e32"_op) {
      each([&](uint32_t lane) {
        // The high half of the 48-bit product of the sources' low 24 bits as signed numbers: what a compiler
        // emits for a multiply high of values it knows fit in 24 bits (llama.cpp's mul_mat_vec_q).
        const auto i24 = [](uint32_t v) { return static_cast<int64_t>(static_cast<int32_t>(v << 8) >> 8); };
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>((i24(lane_src(w, in.src[0], lane)) * i24(lane_src(w, in.src[1], lane))) >> 32));
      });
    } else if (op == "v_mul_lo_u16_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[0], lane) * lane_src(w, in.src[1], lane)));
      });
    } else if (op == "v_not_b32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, ~lane_src(w, in.src[0], lane));
      });
    } else if (op == "v_cvt_f32_ubyte0_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(lane_src(w, in.src[0], lane) & 0xFF)));
      });
    } else if (op == "v_rsq_f32_e32"_op) {
      each([&](uint32_t lane) {
        // As the reciprocal is: the host's exact answer, where a card's is a
        // table good to about one unit in the last place.
        write_float(w, in, lane, 1.0f / std::sqrt(lane_float(w, in.src[0], lane)));
      });
    } else if (op == "v_mul_f32_e64"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, lane_float(w, in.src[0], lane) * lane_float(w, in.src[1], lane));
      });
    } else if (op == "v_lshlrev_b32_e64"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 31));
      });
    } else if (op == "v_lshrrev_b64"_op) {
      each([&](uint32_t lane) {
        write_lane64(w, in.dst[0], lane, lane_src64(w, in.src[1], lane) >> (lane_src(w, in.src[0], lane) & 63));
      });
    } else if (op == "v_fmaak_f32"_op) {
      each([&](uint32_t lane) {
        // The constant the instruction carries is the addend, and comes last.
        write_float(w, in, lane,
                    std::fma(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane),
                             as_float(static_cast<uint32_t>(in.src[2].value))));
      });
    } else if (op == "v_mad_u32_u24"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) & 0xFFFFFFu) * (lane_src(w, in.src[1], lane) & 0xFFFFFFu) +
                       lane_src(w, in.src[2], lane));
      });
    } else if (op == "v_add_lshl_u32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) + lane_src(w, in.src[1], lane)) << (lane_src(w, in.src[2], lane) & 31));
      });
    } else if (op == "v_and_or_b32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) & lane_src(w, in.src[1], lane)) | lane_src(w, in.src[2], lane));
      });
    } else if (op == "v_pack_b32_f16"_op) {
      each([&](uint32_t lane) {
        // The low half of each source, the first below the second.
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) & 0xFFFFu) | (lane_src(w, in.src[1], lane) & 0xFFFFu) << 16);
      });
    } else if (op == "v_fma_mix_f32"_op || op == "v_fma_mixlo_f16"_op || op == "v_fma_mixhi_f16"_op) {
      each([&](uint32_t lane) {
        // Each source is a float, or -- where its op_sel_hi bit is set -- a
        // half, the one op_sel names. The multiply-add is a float's; the
        // mixlo and mixhi forms round it to a half and write it into one half
        // of the destination, leaving the other as it was.
        const auto source = [&](uint32_t k) {
          const Operand& o = in.src[k];
          if (!((in.op_sel_hi >> k) & 1)) return lane_float(w, o, lane);
          const uint32_t bits = o.kind == OperandKind::InlineFloat
                                    ? as_bits(static_cast<_Float16>(o.fvalue))
                                    : lane_src(w, o, lane) >> (((in.op_sel >> k) & 1) ? 16 : 0);
          float f = static_cast<float>(as_half(static_cast<uint16_t>(bits)));
          if (o.abs) f = std::fabs(f);
          return o.neg ? -f : f;
        };
        float r = std::fma(source(0), source(1), source(2));
        if (in.clamp) r = std::isnan(r) ? 0.0f : std::fmin(1.0f, std::fmax(0.0f, r));
        if (op == "v_fma_mix_f32"_op) {
          write_lane(w, in.dst[0], lane, as_bits(r));
          return;
        }
        const uint32_t bits = as_bits(static_cast<_Float16>(r));
        const uint32_t was = w.vgpr[in.dst[0].index][lane];
        w.vgpr[in.dst[0].index][lane] = op == "v_fma_mixlo_f16"_op ? (was & 0xFFFF0000u) | bits
                                                                : (was & 0x0000FFFFu) | bits << 16;
      });
    } else if (op == "v_bfi_b32"_op) {
      each([&](uint32_t lane) {
        // The bits the first source selects come from the second, the rest
        // from the third.
        const uint32_t m = lane_src(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane, (m & lane_src(w, in.src[1], lane)) | (~m & lane_src(w, in.src[2], lane)));
      });
    } else if (op == "v_alignbit_b32"_op) {
      each([&](uint32_t lane) {
        // Two registers side by side, the first above the second, and the 32
        // bits that start where the third says: a rotate, when both are one.
        const uint64_t pair = static_cast<uint64_t>(lane_src(w, in.src[0], lane)) << 32 | lane_src(w, in.src[1], lane);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(pair >> (lane_src(w, in.src[2], lane) & 31)));
      });
    } else if (op == "v_med3_i32"_op) {
      each([&](uint32_t lane) {
        const int32_t x = static_cast<int32_t>(lane_src(w, in.src[0], lane)),
                      y = static_cast<int32_t>(lane_src(w, in.src[1], lane)),
                      z = static_cast<int32_t>(lane_src(w, in.src[2], lane));
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(std::max(std::min(x, y), std::min(std::max(x, y), z))));
      });
    } else if (op == "v_frexp_mant_f32_e32"_op || op == "v_frexp_exp_i32_f32_e32"_op) {
      each([&](uint32_t lane) {
        // A float as a mantissa in [0.5, 1) and a power of two. An infinity
        // or a NaN has no such parts: the mantissa is the value itself and
        // the exponent zero, which is also what the host's frexp gives.
        const float x = lane_float(w, in.src[0], lane);
        int e = 0;
        const float m = std::isfinite(x) ? std::frexp(x, &e) : x;
        if (!std::isfinite(x)) e = 0;
        write_lane(w, in.dst[0], lane, op == "v_frexp_mant_f32_e32"_op ? as_bits(m) : static_cast<uint32_t>(e));
      });
    } else if (op == "v_dot8c_i32_i4_e32"_op) {
      each([&](uint32_t lane) {
        // Eight signed 4-bit pairs, added into the destination; it wraps.
        const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
        uint32_t sum = w.vgpr[in.dst[0].index][lane];
        for (uint32_t k = 0; k < 8; ++k) {
          int32_t x = (a >> (4 * k)) & 15, y = (b >> (4 * k)) & 15;
          if (x & 8) x -= 16;
          if (y & 8) y -= 16;
          sum += static_cast<uint32_t>(x * y);
        }
        write_lane(w, in.dst[0], lane, sum);
      });
    } else if (op == "v_dot2c_i32_i16_e32"_op) {
      each([&](uint32_t lane) {
        // Two signed 16-bit pairs, added into the destination; it wraps.
        const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
        uint32_t sum = w.vgpr[in.dst[0].index][lane];
        for (uint32_t h = 0; h < 2; ++h)
          sum += static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(a >> (16 * h))) *
                                       static_cast<int32_t>(static_cast<int16_t>(b >> (16 * h))));
        write_lane(w, in.dst[0], lane, sum);
      });
    } else if (op == "v_dot4c_i32_i8_e32"_op) {
      each([&](uint32_t lane) {
        // Four signed bytes times four, added into the destination. Nothing
        // clamps it: a sum past what 32 bits hold wraps.
        const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
        uint32_t sum = w.vgpr[in.dst[0].index][lane];
        for (uint32_t k = 0; k < 4; ++k)
          sum += static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(a >> (8 * k))) *
                                       static_cast<int32_t>(static_cast<int8_t>(b >> (8 * k))));
        write_lane(w, in.dst[0], lane, sum);
      });
    } else if (op == "v_dot2c_f32_bf16_e32"_op || op == "v_dot2_f32_bf16"_op) {
      // gfx950: two pairs of bfloat16s multiplied and added into a float,
      // the sum worked out in double and rounded once. The VOP2 form adds
      // into its destination; the packed one into its third source, with
      // op_sel choosing halves and neg_lo/neg_hi negating, as v_dot2_f32_f16.
      const bool into_dst = op == "v_dot2c_f32_bf16_e32"_op;
      each([&](uint32_t lane) {
        const auto bf = [](uint32_t bits16) { return static_cast<double>(as_float(bits16 << 16)); };
        double sum;
        if (into_dst) {
          const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
          sum = bf(a & 0xFFFF) * bf(b & 0xFFFF) + bf(a >> 16) * bf(b >> 16) +
                static_cast<double>(as_float(w.vgpr[in.dst[0].index][lane]));
        } else {
          const float c = lane_float(w, in.src[2], lane);
          sum = (in.neg_lo >> 2) & 1 ? -double(c) : double(c);
          for (uint32_t h = 0; h < 2; ++h) {
            const double a = bf(packed_bits(w, in, 0, h, lane)), b = bf(packed_bits(w, in, 1, h, lane));
            const uint8_t neg = h ? in.neg_hi : in.neg_lo;
            sum += (neg & 1 ? -a : a) * (neg & 2 ? -b : b);
          }
        }
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(sum)));
      });
    } else if (op == "v_dot2c_f32_f16_e32"_op) {
      each([&](uint32_t lane) {
        // Two pairs of halves multiplied and added into a float. Each product
        // is exact in a float; the sum is worked out in double and rounded
        // once, which is this model's reading of a dot product -- where the
        // sum fits a float exactly, any reading gives the same answer.
        const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
        const auto half = [](uint32_t v, uint32_t k) {
          _Float16 h;
          const uint16_t bits = static_cast<uint16_t>(v >> (16 * k));
          std::memcpy(&h, &bits, 2);
          return static_cast<double>(h);
        };
        const double sum = half(a, 0) * half(b, 0) + half(a, 1) * half(b, 1) +
                           static_cast<double>(as_float(w.vgpr[in.dst[0].index][lane]));
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(sum)));
      });
    } else if (op == "v_bitop3_b32"_op || op == "v_bitop3_b16"_op) {
      // Each result bit is the truth table's entry for that bit of the three
      // sources, the first the most significant of the index: 0xF0, 0xCC and
      // 0xAA as the sources give the table itself. The 16-bit form reads the
      // half op_sel names of each source, and writes the low half.
      const bool b16 = op == "v_bitop3_b16"_op;
      each([&](uint32_t lane) {
        uint32_t x[3];
        for (uint32_t k = 0; k < 3; ++k) {
          x[k] = lane_src(w, in.src[k], lane);
          if (b16) x[k] = (x[k] >> (16 * ((in.op_sel >> k) & 1))) & 0xFFFF;
        }
        uint32_t r = 0;
        for (uint32_t idx = 0; idx < 8; ++idx)
          if (in.bitop3 >> idx & 1)
            r |= (idx & 4 ? x[0] : ~x[0]) & (idx & 2 ? x[1] : ~x[1]) & (idx & 1 ? x[2] : ~x[2]);
        if (b16) r &= 0xFFFF;
        write_lane(w, in.dst[0], lane, r);
      });
    } else if (op == "v_cvt_pk_f16_f32"_op || op == "v_cvt_pk_bf16_f32"_op) {
      // Two floats to a packed pair, the first in the low half, each rounded
      // to nearest even.
      const bool bf = op == "v_cvt_pk_bf16_f32"_op;
      each([&](uint32_t lane) {
        const float a = lane_float(w, in.src[0], lane), b = lane_float(w, in.src[1], lane);
        const auto narrow = [&](float f) -> uint32_t {
          if (bf) return to_bf16(f);
          const _Float16 h = static_cast<_Float16>(f);
          uint16_t bits;
          std::memcpy(&bits, &h, 2);
          return bits;
        };
        write_lane(w, in.dst[0], lane, narrow(a) | narrow(b) << 16);
      });
    } else if (op == "v_cvt_pkrtz_f16_f32"_op) {
      each([&](uint32_t lane) {
        // Two floats narrowed to halves and packed, each rounded toward zero.
        write_lane(w, in.dst[0], lane,
                   half_toward_zero(lane_float(w, in.src[0], lane)) |
                       static_cast<uint32_t>(half_toward_zero(lane_float(w, in.src[1], lane))) << 16);
      });
    } else if (op == "v_xad_u32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) ^ lane_src(w, in.src[1], lane)) + lane_src(w, in.src[2], lane));
      });
    } else if (op == "v_bfrev_b32_e32"_op) {
      each([&](uint32_t lane) {
        uint32_t v = lane_src(w, in.src[0], lane), r = 0;
        for (uint32_t k = 0; k < 32; ++k) r |= ((v >> k) & 1) << (31 - k);
        write_lane(w, in.dst[0], lane, r);
      });
    } else if (op == "v_min_u32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, std::min(lane_src(w, in.src[0], lane), lane_src(w, in.src[1], lane)));
      });
    } else if (op == "v_lshl_or_b32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) << (lane_src(w, in.src[1], lane) & 31)) |
                       lane_src(w, in.src[2], lane));
      });
    } else if (op == "v_trunc_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, as_bits(std::trunc(lane_float(w, in.src[0], lane))));
      });
    } else if (op == "v_cvt_f32_f64_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(as_double(lane_src64(w, in.src[0], lane)))));
      });
    } else if (op == "v_cvt_i32_f64_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(static_cast<int32_t>(as_double(lane_src64(w, in.src[0], lane)))));
      });
    } else if (op == "v_mad_f32"_op || op == "v_mad_legacy_f32"_op || op == "v_mac_f32_e32"_op ||
               op == "v_mac_f32_e64"_op || op == "v_madmk_f32"_op) {
      // gfx90a's multiply-adds, which gfx940 dropped: the product rounded to
      // a float before the add, and denormals -- in or out -- flushed to
      // zero, which is why a compiler picks them only where denormals are
      // flushed anyway. The legacy form takes zero times anything, infinity
      // and NaN included, as zero. v_mac adds into its destination, and
      // v_madmk's middle source is the constant it carries.
      const bool legacy = op == "v_mad_legacy_f32"_op, mac = op == "v_mac_f32_e32"_op || op == "v_mac_f32_e64"_op,
                 madmk = op == "v_madmk_f32"_op;
      const auto ftz = [](float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x; };
      each([&](uint32_t lane) {
        const float a = ftz(lane_float(w, in.src[0], lane));
        const float b = ftz(madmk ? as_float(static_cast<uint32_t>(in.src[1].value)) : lane_float(w, in.src[1], lane));
        const float c = ftz(mac ? as_float(w.vgpr[in.dst[0].index][lane]) : lane_float(w, in.src[2], lane));
        const float product = legacy && (a == 0.0f || b == 0.0f) ? 0.0f : ftz(a * b);
        write_float(w, in, lane, ftz(product + c));
      });
    } else if (op == "v_fmamk_f32"_op) {
      each([&](uint32_t lane) {
        // The middle source is the constant the instruction carries.
        write_lane(w, in.dst[0], lane,
                   as_bits(std::fma(lane_float(w, in.src[0], lane), as_float(static_cast<uint32_t>(in.src[1].value)),
                                    lane_float(w, in.src[2], lane))));
      });
    } else if (op == "v_bfe_i32"_op) {
      each([&](uint32_t lane) {
        // The same bits as v_bfe_u32, with the top one carried into the rest.
        const uint32_t value = lane_src(w, in.src[0], lane), start = lane_src(w, in.src[1], lane) & 31,
                       width = lane_src(w, in.src[2], lane) & 31;
        uint32_t out = 0;
        if (width != 0) {
          out = width >= 32 ? value >> start : (value >> start) & ((1u << width) - 1);
          if (width < 32 && (out >> (width - 1) & 1)) out |= ~((1u << width) - 1);
        }
        write_lane(w, in.dst[0], lane, out);
      });
    } else if (op == "v_bfe_u32"_op) {
      each([&](uint32_t lane) {
        // The bits src2 wide starting at src1.
        const uint32_t value = lane_src(w, in.src[0], lane), start = lane_src(w, in.src[1], lane) & 31,
                       width = lane_src(w, in.src[2], lane) & 31;
        write_lane(w, in.dst[0], lane, width == 0 ? 0u : (value >> start) & ((width >= 32) ? ~0u : ((1u << width) - 1)));
      });
    } else if (op == "v_cvt_f32_i32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(static_cast<int32_t>(lane_src(w, in.src[0], lane)))));
      });
    } else if (op == "v_cvt_f32_u32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(lane_src(w, in.src[0], lane))));
      });
    } else if (op == "v_fmac_f32_e32"_op || op == "v_fmac_f32_e64"_op) {
      each([&](uint32_t lane) {
        // The destination is also the addend.
        write_float(w, in, lane,
                    std::fma(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane),
                             as_float(w.vgpr[in.dst[0].index][lane])));
      });
    } else if (op == "v_fma_f32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane,
                   as_bits(std::fma(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane),
                                    lane_float(w, in.src[2], lane))));
      });
    } else if (op == "v_cndmask_b32_e64"_op) {
      each([&](uint32_t lane) {
        // One lane's bit of the condition register picks a source, with its
        // modifiers -- which is how the compiler negates a float, or takes
        // its absolute value, as it selects it.
        const uint64_t cond = scalar(w, in.src[2]);
        write_lane(w, in.dst[0], lane, lane_bits(w, (cond >> lane) & 1 ? in.src[1] : in.src[0], lane));
      });
    } else if (op == "v_sub_f32_e32"_op || op == "v_sub_f32_e64"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, lane_float(w, in.src[0], lane) - lane_float(w, in.src[1], lane));
      });
    } else if (op == "v_min_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, std::fmin(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane)));
      });
    } else if (op == "v_max_f32_e32"_op || op == "v_max_f32_e64"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, std::fmax(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane)));
      });
    } else if (op == "v_add_f32_e64"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, lane_float(w, in.src[0], lane) + lane_float(w, in.src[1], lane));
      });
    } else if (op == "v_max_u32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, std::max(lane_src(w, in.src[0], lane), lane_src(w, in.src[1], lane)));
      });
    } else if (op == "v_subrev_u32_e32"_op) {
      each([&](uint32_t lane) {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) - lane_src(w, in.src[0], lane));
      });
    } else if (op == "v_mul_hi_u32"_op) {
      each([&](uint32_t lane) {
        const uint64_t p = static_cast<uint64_t>(lane_src(w, in.src[0], lane)) * lane_src(w, in.src[1], lane);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(p >> 32));
      });
    } else if (op == "v_bcnt_u32_b32"_op) {
      each([&](uint32_t lane) {
        // The set bits of the first source, counted into the second.
        write_lane(w, in.dst[0], lane,
                   lane_src(w, in.src[1], lane) + static_cast<uint32_t>(__builtin_popcount(lane_src(w, in.src[0], lane))));
      });
    } else if (op == "v_ffbh_u32_e32"_op) {
      each([&](uint32_t lane) {
        // The leading zeros, and -1 when there is no set bit at all.
        const uint32_t v = lane_src(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane, v ? static_cast<uint32_t>(__builtin_clz(v)) : 0xFFFFFFFFu);
      });
    } else if (op == "v_cvt_i32_f32_e32"_op) {
      each([&](uint32_t lane) {
        const float f = lane_float(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(std::isnan(f)          ? 0
                                         : f <= -2147483648.0f  ? INT32_MIN
                                         : f >= 2147483648.0f   ? INT32_MAX
                                                                : static_cast<int32_t>(f)));
      });
    } else if (op == "v_mul_u32_u24_e32"_op) {
      each([&](uint32_t lane) {
        // Only the low 24 bits of each source take part.
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) & 0xFFFFFF) * (lane_src(w, in.src[1], lane) & 0xFFFFFF));
      });
    } else if (op == "v_cvt_u32_f32_e32"_op) {
      each([&](uint32_t lane) {
        const float f = lane_float(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane,
                   std::isnan(f) || f <= 0 ? 0u : f >= 4294967296.0f ? 0xFFFFFFFFu : static_cast<uint32_t>(f));
      });
    } else if (op == "v_sqrt_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, std::sqrt(lane_float(w, in.src[0], lane)));
      });
    } else if (op == "v_exp_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, std::exp2(lane_float(w, in.src[0], lane)));
      });
    } else if (op == "v_log_f32_e32"_op) {
      each([&](uint32_t lane) {
        write_float(w, in, lane, std::log2(lane_float(w, in.src[0], lane)));
      });
    } else if (op == "v_cndmask_b32_e32"_op) {
      each([&](uint32_t lane) {
        // The sub-dword form's sources may carry modifiers too.
        const uint64_t cond = scalar(w, in.src[2]);
        write_lane(w, in.dst[0], lane, lane_bits(w, (cond >> lane) & 1 ? in.src[1] : in.src[0], lane));
      });
    } else if (op == "v_add_f64"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane, lane_double(w, in.src[0], lane) + lane_double(w, in.src[1], lane));
      });
    } else if (op == "v_mul_f64"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane, lane_double(w, in.src[0], lane) * lane_double(w, in.src[1], lane));
      });
    } else if (op == "v_fma_f64"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane,
                     std::fma(lane_double(w, in.src[0], lane), lane_double(w, in.src[1], lane),
                              lane_double(w, in.src[2], lane)));
      });
    } else if (op == "v_fmac_f64_e32"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane,
                     std::fma(lane_double(w, in.src[0], lane), lane_double(w, in.src[1], lane),
                              as_double(lane_src64(w, in.dst[0], lane))));
      });
    } else if (op == "v_rcp_f64_e32"_op) {
      each([&](uint32_t lane) {
        write_double(w, in, lane, 1.0 / lane_double(w, in.src[0], lane));
      });
    } else if (op == "v_pk_fma_f32"_op || op == "v_pk_add_f32"_op || op == "v_pk_mul_f32"_op) {
      each([&](uint32_t lane) {
        // Two floats in a register pair, each its own arithmetic -- both
        // worked out before either is written, since the destination may be
        // a source whose low register the high result reads.
        float r[2];
        for (uint32_t half = 0; half < 2; ++half) {
          const float x = packed_float(w, in, 0, half, lane), y = packed_float(w, in, 1, half, lane);
          r[half] = op == "v_pk_add_f32"_op   ? x + y
                    : op == "v_pk_mul_f32"_op ? x * y
                                           : std::fma(x, y, packed_float(w, in, 2, half, lane));
        }
        set_word(w, in.dst[0], 0, lane, as_bits(r[0]));
        set_word(w, in.dst[0], 1, lane, as_bits(r[1]));
      });
    } else if (op == "v_pk_mov_b32"_op) {
      // The low register from the first source and the high from the second,
      // each the register of its pair op_sel names.
      each([&](uint32_t lane) {
        const uint32_t lo = packed_word(w, in, 0, (in.op_sel >> 0) & 1, lane);
        const uint32_t hi = packed_word(w, in, 1, (in.op_sel >> 1) & 1, lane);
        set_word(w, in.dst[0], 0, lane, lo);
        set_word(w, in.dst[0], 1, lane, hi);
      });
    } else if (op == "v_pk_fma_f16"_op || op == "v_pk_add_f16"_op || op == "v_pk_mul_f16"_op ||
               op == "v_pk_min_f16"_op || op == "v_pk_max_f16"_op) {
      each([&](uint32_t lane) {
        // Two halves in one register, each its own arithmetic, done in float
        // -- which holds a product of two halves exactly -- and rounded once
        // to a half. Where one is a NaN, min and max keep the other.
        uint32_t r = 0;
        for (uint32_t half = 0; half < 2; ++half) {
          const float x = packed_half(w, in, 0, half, lane), y = packed_half(w, in, 1, half, lane);
          const float v = op == "v_pk_add_f16"_op   ? x + y
                          : op == "v_pk_mul_f16"_op ? x * y
                          : op == "v_pk_min_f16"_op ? std::fmin(x, y)
                          : op == "v_pk_max_f16"_op ? std::fmax(x, y)
                                                 : std::fma(x, y, packed_half(w, in, 2, half, lane));
          r |= static_cast<uint32_t>(as_bits(static_cast<_Float16>(v))) << (16 * half);
        }
        write_lane(w, in.dst[0], lane, r);
      });
    } else if (op == "v_rcp_f32_e32"_op || op == "v_rcp_iflag_f32_e32"_op) {
      each([&](uint32_t lane) {
        // The hardware's reciprocal is a table good to about one unit in the
        // last place; this is the exact one, so a program that refines it
        // (which is how the compiler divides) lands on the same answer. The
        // iflag form differs only in which exceptions it raises, and nothing
        // here raises any.
        write_float(w, in, lane, 1.0f / lane_float(w, in.src[0], lane));
      });
    } else if (op == "v_mbcnt_lo_u32_b32"_op || op == "v_mbcnt_hi_u32_b32"_op) {
      each([&](uint32_t lane) {
        // The lanes below this one that are set in the mask, counted into the
        // second source: how a wave numbers its active lanes.
        const uint32_t mask = lane_src(w, in.src[0], lane);
        const uint32_t below = op == "v_mbcnt_lo_u32_b32"_op
                                   ? (lane >= 32 ? 0xFFFFFFFFu : (lane ? (1u << lane) - 1 : 0))
                                   : (lane < 32 ? 0u : (1u << (lane - 32)) - 1);
        write_lane(w, in.dst[0], lane,
                   lane_src(w, in.src[1], lane) + static_cast<uint32_t>(__builtin_popcount(mask & below)));
      });
    } else if (op.rfind("v_div_", 0) == 0) {
      sdst_bits = 0;
      each([&](uint32_t lane) {
        divide_step(w, in, lane);
      });
      if (in.dst.size() > 1) write_scalar(w, in.dst[1], sdst_bits);
    } else if (op == "v_mad_u64_u32"_op) {
      sdst_bits = 0;
      each([&](uint32_t lane) {
        // A 32x32 product added to a 64-bit value, with the carry out.
        const unsigned __int128 p = static_cast<unsigned __int128>(lane_src(w, in.src[0], lane)) *
                                        lane_src(w, in.src[1], lane) +
                                    lane_src64(w, in.src[2], lane);
        write_lane64(w, in.dst[0], lane, static_cast<uint64_t>(p));
        if (static_cast<uint64_t>(p >> 64)) sdst_bits |= uint64_t{1} << lane;
      });
      if (in.dst.size() > 1) write_scalar(w, in.dst[1], sdst_bits);
    } else if (w.exec) {
      // With no lane to write, an instruction changes nothing, and is not refused.
      throw Error::make(Err::Unsupported, "vector instruction ", op, " is decoded but not implemented");
    }
  }

  // A comparison, one bit a lane. VOPC and VOP3 number the comparisons alike,
  // and the number says everything: which type (the high bits) and which
  // test (the low ones), in the ISA's order -- for floats F, LT, EQ, LE, GT,
  // LG, GE, O, U, NGE, NLG, NGT, NLE, NEQ, NLT, TRU, where the N forms are
  // the tests negated and so hold for a NaN; for integers F, LT, EQ, LE, GT,
  // NE, GE, T. The sub-dword form compares the parts of its sources it names.
  void compare(Wave& w, const Inst& in) {
    // The X forms, sixteen on from the others (the class tests' one on),
    // write their result to EXEC as well as where the others write it.
    const uint32_t raw = in.opcode & 0xFF;
    const bool writes_exec = (raw >= 0x30 && raw < 0x40) || (raw >= 0x50 && raw < 0x60) || (raw >= 0x70 && raw < 0x80) ||
                             (raw >= 0xB0 && raw < 0xC0) || (raw >= 0xD0 && raw < 0xE0) || raw >= 0xF0 ||
                             raw == 0x11 || raw == 0x13 || raw == 0x15;
    const uint32_t op = !writes_exec ? raw : raw < 0x20 ? raw - 1 : raw - 0x10;
    const auto each = [&](auto&& test) {
      uint64_t result = 0;
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (w.exec >> lane & 1 && test(lane)) result |= uint64_t{1} << lane;   // an inactive lane's bit reads 0
      // RDNA's v_cmpx writes EXEC alone; gfx9's writes its destination too.
      if (!in.dst.empty()) write_scalar(w, in.dst[0], result);
      if (writes_exec) w.exec = result;
    };
    const auto float_test = [](uint32_t t, auto x, auto y) {
      switch (t & 0xF) {
        case 0x0: return false;
        case 0x1: return x < y;
        case 0x2: return x == y;
        case 0x3: return x <= y;
        case 0x4: return x > y;
        case 0x5: return x < y || x > y;
        case 0x6: return x >= y;
        case 0x7: return x == x && y == y;   // neither a NaN, for a half as for a double
        case 0x8: return x != x || y != y;
        case 0x9: return !(x >= y);
        case 0xA: return !(x < y || x > y);
        case 0xB: return !(x > y);
        case 0xC: return !(x <= y);
        case 0xD: return !(x == y);
        case 0xE: return !(x < y);
        default: return true;
      }
    };
    const auto int_test = [](uint32_t t, auto x, auto y) {
      switch (t & 0x7) {
        case 0: return false;
        case 1: return x < y;
        case 2: return x == y;
        case 3: return x <= y;
        case 4: return x > y;
        case 5: return x != y;
        case 6: return x >= y;
        default: return true;
      }
    };
    const Operand &a = in.src[0], &b = in.src[1];
    if (op == 0x10) return each([&](uint32_t l) { return matches_class(lane_float(w, a, l), lane_src(w, b, l)); });
    if (op == 0x12) return each([&](uint32_t l) { return matches_class(lane_double(w, a, l), lane_src(w, b, l)); });
    if (op == 0x14) return each([&](uint32_t l) { return matches_class_half(lane_half(w, a, l), lane_src(w, b, l)); });
    {
      if (op >= 0x20 && op < 0x30)
        return each([&](uint32_t l) { return float_test(op, lane_half(w, a, l), lane_half(w, b, l)); });
      if (op >= 0x40 && op < 0x50)
        return each([&](uint32_t l) { return float_test(op, lane_float(w, a, l), lane_float(w, b, l)); });
      if (op >= 0x60 && op < 0x70)
        return each([&](uint32_t l) { return float_test(op, lane_double(w, a, l), lane_double(w, b, l)); });
      if (op >= 0xA0 && op < 0xA8)
        return each([&](uint32_t l) {
          return int_test(op, static_cast<int16_t>(lane_src(w, a, l)), static_cast<int16_t>(lane_src(w, b, l)));
        });
      if (op >= 0xA8 && op < 0xB0)
        return each([&](uint32_t l) {
          return int_test(op, static_cast<uint16_t>(lane_src(w, a, l)), static_cast<uint16_t>(lane_src(w, b, l)));
        });
      if (op >= 0xC0 && op < 0xC8)
        return each([&](uint32_t l) {
          return int_test(op, static_cast<int32_t>(lane_src(w, a, l)), static_cast<int32_t>(lane_src(w, b, l)));
        });
      if (op >= 0xC8 && op < 0xD0)
        return each([&](uint32_t l) { return int_test(op, lane_src(w, a, l), lane_src(w, b, l)); });
      if (op >= 0xE0 && op < 0xE8)
        return each([&](uint32_t l) {
          return int_test(op, static_cast<int64_t>(lane_src64(w, a, l)), static_cast<int64_t>(lane_src64(w, b, l)));
        });
      if (op >= 0xE8 && op < 0xF0)
        return each([&](uint32_t l) { return int_test(op, lane_src64(w, a, l), lane_src64(w, b, l)); });
    }
    throw Error::make(Err::Unsupported, "comparison ", in.name, " is decoded but not implemented");
  }

  // One work-item's private memory: its own block of the group's scratch.
  uint8_t* scratch_at(Group& g, const Wave& w, uint32_t lane, uint64_t offset, uint32_t bytes) {
    const uint64_t base = uint64_t{w.first_lane + lane} * g.scratch_per_lane;
    if (offset + bytes > g.scratch_per_lane || base + offset + bytes > g.scratch.size())
      throw Error::make(Err::InvalidValue, "a scratch access at ", offset, " is past the ", g.scratch_per_lane,
                        " bytes a work-item reserved");
    return &g.scratch[base + offset];
  }

  // LDS, which the work-group shares.
  // Which lane a lane reads under a swizzle pattern: four lanes choosing
  // among their own four, or, within a group of 32, the lane its own number
  // becomes once ANDed, ORed and XORed with the pattern's three masks.
  // Or, rotating, the lane so many further on (to the left) or back (to the
  // right) within its 32.
  static uint32_t swizzle_source(uint32_t pattern, uint32_t lane) {
    if ((pattern & 0xFF00) == 0x8000) return (lane & ~3u) + ((pattern >> (2 * (lane & 3))) & 3);
    if ((pattern & 0xE000) == 0xC000) {
      const uint32_t by = (pattern >> 5) & 0x1F, right = (pattern >> 10) & 1;
      return (lane & ~31u) | ((right ? lane - by : lane + by) & 31u);
    }
    const uint32_t and_mask = pattern & 0x1F, or_mask = (pattern >> 5) & 0x1F, xor_mask = (pattern >> 10) & 0x1F;
    return (lane & ~31u) | ((((lane & 31u) & and_mask) | or_mask) ^ xor_mask);
  }

  void lds_access(Wave& w, const Inst& in, Group& g) {
    const OpName op(in.name);
    // The two that move values between lanes read every lane's value before
    // any lane's result is written: the destination may be the very register
    // they read, and a lane further on must still see what was there.
    const bool across = op == "ds_bpermute_b32"_op || op == "ds_swizzle_b32"_op;
    std::array<uint32_t, kLanes> before{};
    if (across) {
      const Operand& data = op == "ds_bpermute_b32"_op ? in.src[1] : in.src[0];
      for (uint32_t lane = 0; lane < kLanes; ++lane) before[lane] = w.vgpr[data.index][lane];
    }
    if (op == "ds_read_b64_tr_b16"_op || op == "ds_read_b64_tr_b8"_op || op == "ds_read_b64_tr_b4"_op) {
      // gfx950's transposing reads. Each lane reads 64 bits from its address
      // -- `tile` elements of `bits` bits -- and the hardware hands them out
      // across the wave: destination lane l's element r is element l mod
      // tile of what lane A read, where A takes, from its low bit up, lane
      // bits t .. t+b-1, then r's t bits, then lane bits t+b and up (t =
      // log2 tile; b = 2, 1, 0 for 16-, 8-, 4-bit elements). This is the map
      // Triton's AMD backend lowers these instructions by (TargetFeatures:
      // CDNA4's leading register and lane bases, 0 and b). The ISA asks for
      // every lane to be on; every lane's read is made, and only the active
      // lanes written.
      const uint32_t bits = op == "ds_read_b64_tr_b16"_op ? 16 : op == "ds_read_b64_tr_b8"_op ? 8 : 4;
      const uint32_t tile = 64 / bits, t = static_cast<uint32_t>(__builtin_ctz(tile));
      const uint32_t b = bits == 16 ? 2 : bits == 8 ? 1 : 0;
      std::array<uint64_t, kLanes> read{};
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint64_t a = static_cast<uint32_t>(lane_src(w, in.src[0], lane) + static_cast<uint32_t>(in.offset));
        if (a + 8 <= g.lds.size()) std::memcpy(&read[lane], &g.lds[a], 8);
        else if (g.lds.empty())
          throw Error::make(Err::InvalidValue, "an LDS access at ", a, " in a work-group given no LDS: the launch ",
                            "did not pay for the LDS its kernel uses");
      }
      const uint64_t mask = (uint64_t{1} << bits) - 1;
      for (uint32_t l = 0; l < kLanes; ++l) {
        if (!(w.exec >> l & 1)) continue;
        uint64_t out = 0;
        for (uint32_t r = 0; r < tile; ++r) {
          const uint32_t src = ((l >> t) & ((1u << b) - 1)) | (r << b) | ((l >> (t + b)) << (b + t));
          const uint32_t c = l & (tile - 1);
          out |= ((read[src % kLanes] >> (c * bits)) & mask) << (r * bits);
        }
        set_word(w, in.dst[0], 0, l, static_cast<uint32_t>(out));
        set_word(w, in.dst[0], 1, l, static_cast<uint32_t>(out >> 32));
      }
      return;
    }
    if (op == "ds_permute_b32"_op) {
      // The other way round from bpermute: each lane sends its value to the
      // lane its address names, and a lane no one sent to gets zero. Where
      // two send to the same lane, the higher-numbered one's lands last.
      std::array<uint32_t, kLanes> sent{};
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (w.exec >> lane & 1)
          sent[((lane_src(w, in.src[0], lane) + static_cast<uint32_t>(in.offset)) >> 2) & (w.lanes - 1)] =
              lane_src(w, in.src[1], lane);
      for (uint32_t lane = 0; lane < kLanes; ++lane)
        if (w.exec >> lane & 1) write_lane(w, in.dst[0], lane, sent[lane]);
      return;
    }
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint32_t addr = across ? 0 : lane_src(w, in.src[0], lane);
      // Where an access lands. On a card, an access past the work-group's
      // LDS reads zero and its write goes nowhere: Tensile's GEMMs read from
      // far past the 64 KB a compute unit has to clear registers, and
      // rocFFT's kernels read a little past theirs in lanes whose results
      // are not kept. A work-group given no LDS at all, though, is almost
      // always a launch that did not pay for the LDS its kernel uses, and
      // is caught rather than read as zeroes.
      alignas(16) uint8_t outside[16];
      const auto at = [&](uint64_t offset, uint64_t bytes = 4) -> uint8_t* {
        // The address and the offset add in 32 bits, as the hardware adds
        // them: a register holding -8 with offset 64 reaches byte 56, which
        // rocFFT's real-to-complex kernels count on.
        const uint64_t a = static_cast<uint32_t>(addr + offset);
        if (a + bytes <= g.lds.size()) return &g.lds[a];
        if (g.lds.empty() && a < lds_limit(d))
          throw Error::make(Err::InvalidValue, "an LDS access at ", a, " in a work-group given no LDS: the launch ",
                            "did not pay for the LDS its kernel uses");
        std::memset(outside, 0, sizeof outside);
        return outside;
      };
      const uint64_t off = static_cast<uint64_t>(in.offset);
      if (op == "ds_write_b32"_op) {
        const uint32_t v = lane_src(w, in.src[1], lane);
        std::memcpy(at(static_cast<uint64_t>(in.offset)), &v, 4);
      } else if (op == "ds_read_b32"_op) {
        uint32_t v = 0;
        std::memcpy(&v, at(static_cast<uint64_t>(in.offset)), 4);
        write_lane(w, in.dst[0], lane, v);
      } else if (op == "ds_add_u32"_op) {
        // An atomic add in LDS: every lane's addition lands.
        uint32_t v = 0;
        std::memcpy(&v, at(static_cast<uint64_t>(in.offset)), 4);
        v += lane_src(w, in.src[1], lane);
        std::memcpy(at(static_cast<uint64_t>(in.offset)), &v, 4);
      } else if (op == "ds_bpermute_b32"_op) {
        // A lane reads what another lane holds: the address says which, in
        // bytes, and the source register is read across the wave.
        // A lane that is switched off does not give its value: "if src_lane selects a disabled
        // thread then zero is returned" (the MI300 ISA guide, DS_BPERMUTE_B32).
        const uint32_t from = ((lane_src(w, in.src[0], lane) + static_cast<uint32_t>(in.offset)) >> 2) & (w.lanes - 1);
        write_lane(w, in.dst[0], lane, w.exec >> from & 1 ? before[from] : 0);
      } else if (op == "ds_swizzle_b32"_op) {
        // The same: "thread_valid[j] ? thread_in[j] : 0".
        const uint32_t from = swizzle_source(static_cast<uint32_t>(in.offset), lane);
        write_lane(w, in.dst[0], lane, w.exec >> from & 1 ? before[from] : 0);
      } else if (op == "ds_add_f32"_op || op == "ds_add_rtn_f32"_op || op == "ds_min_f32"_op ||
                 op == "ds_min_rtn_f32"_op || op == "ds_max_f32"_op || op == "ds_max_rtn_f32"_op) {
        // Float atomics, lane by lane, as the integer ones are; the _rtn
        // forms hand back what was there.
        uint32_t was = 0;
        std::memcpy(&was, at(off), 4);
        const float a = as_float(was), b = as_float(lane_src(w, in.src[1], lane));
        const float now = op == "ds_add_f32"_op || op == "ds_add_rtn_f32"_op   ? a + b
                          : op == "ds_min_f32"_op || op == "ds_min_rtn_f32"_op ? std::fmin(a, b)
                                                                               : std::fmax(a, b);
        std::memcpy(at(off), &now, 4);
        if (op == "ds_add_rtn_f32"_op || op == "ds_min_rtn_f32"_op || op == "ds_max_rtn_f32"_op)
          write_lane(w, in.dst[0], lane, was);
      } else if (op == "ds_cmpst_b32"_op || op == "ds_cmpst_f32"_op || op == "ds_cmpst_rtn_f32"_op) {
        // The second value is stored where the first is found (compared as
        // floats for the _f32 forms); the _rtn form hands back what was there.
        uint32_t was = 0;
        std::memcpy(&was, at(off), 4);
        const uint32_t cmp = lane_src(w, in.src[1], lane);
        const bool equal = op == "ds_cmpst_b32"_op ? was == cmp : as_float(was) == as_float(cmp);
        if (equal) {
          const uint32_t v = lane_src(w, in.src[2], lane);
          std::memcpy(at(off), &v, 4);
        }
        if (op == "ds_cmpst_rtn_f32"_op) write_lane(w, in.dst[0], lane, was);
      } else if (op == "ds_write_b8"_op || op == "ds_write_b16"_op) {
        const uint32_t v = lane_src(w, in.src[1], lane);
        const uint64_t bytes = op == "ds_write_b8"_op ? 1 : 2;
        std::memcpy(at(off, bytes), &v, bytes);
      } else if (Half h; half_access(std::string_view(in.name).substr(3), &h)) {
        uint8_t* p = at(off, h.bytes);
        if (h.store) {
          const uint32_t v = half_store(lane_src(w, in.src[1], lane), h);
          std::memcpy(p, &v, h.bytes);
        } else {
          uint32_t raw = 0;
          std::memcpy(&raw, p, h.bytes);
          write_half_load(w, in, in.dst[0], lane, half_load(raw, h), h);
        }
      } else if (op == "ds_read_u8"_op || op == "ds_read_i8"_op || op == "ds_read_u16"_op) {
        const uint64_t bytes = op == "ds_read_u16"_op ? 2 : 1;
        uint32_t v = 0;
        std::memcpy(&v, at(off, bytes), bytes);
        if (op == "ds_read_i8"_op) v = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(v)));
        write_lane(w, in.dst[0], lane, v);
      } else if (op == "ds_write_b64"_op || op == "ds_write_b128"_op) {
        const uint32_t words = in.src[1].width;
        uint8_t* const a = at(off, 4 * words);
        for (uint32_t k = 0; k < words; ++k) {
          const uint32_t v = word(w, in.src[1], k, lane);
          std::memcpy(a + 4 * k, &v, 4);
        }
      } else if (op == "ds_read_b64"_op || op == "ds_read_b128"_op) {
        const uint32_t words = in.dst[0].width;
        uint8_t* const a = at(off, 4 * words);
        for (uint32_t k = 0; k < words; ++k) {
          uint32_t v = 0;
          std::memcpy(&v, a + 4 * k, 4);
          set_word(w, in.dst[0], k, lane, v);
        }
      } else if (op == "ds_read2st64_b32"_op) {
        // Two dwords, each offset by its own count of 64 dwords.
        uint32_t v0 = 0, v1 = 0;
        std::memcpy(&v0, at(uint64_t{static_cast<uint32_t>(in.offset)} * 64 * 4), 4);
        std::memcpy(&v1, at(uint64_t{static_cast<uint32_t>(in.offset1)} * 64 * 4), 4);
        set_word(w, in.dst[0], 0, lane, v0);
        set_word(w, in.dst[0], 1, lane, v1);
      } else if (op == "ds_sub_u32"_op || op == "ds_rsub_u32"_op || op == "ds_inc_u32"_op || op == "ds_dec_u32"_op ||
                 op == "ds_sub_rtn_u32"_op || op == "ds_rsub_rtn_u32"_op || op == "ds_inc_rtn_u32"_op ||
                 op == "ds_dec_rtn_u32"_op || op == "ds_min_rtn_i32"_op || op == "ds_max_rtn_i32"_op ||
                 op == "ds_min_rtn_u32"_op || op == "ds_max_rtn_u32"_op || op == "ds_and_rtn_b32"_op ||
                 op == "ds_or_rtn_b32"_op || op == "ds_xor_rtn_b32"_op || op == "ds_wrxchg_rtn_b32"_op) {
        // The rest of LDS's 32-bit read-modify-writes; the _rtn forms hand
        // back what was there.
        std::string what(in.name);
        const bool rtn = what.find("_rtn") != std::string::npos;
        if (rtn) what.erase(what.find("_rtn"), 4);
        uint32_t was = 0;
        std::memcpy(&was, at(off), 4);
        const uint32_t v = lane_src(w, in.src[1], lane);
        const int32_t sw = static_cast<int32_t>(was), sv = static_cast<int32_t>(v);
        uint32_t now = was;
        if (what == "ds_sub_u32") now = was - v;
        else if (what == "ds_rsub_u32") now = v - was;
        else if (what == "ds_inc_u32") now = was >= v ? 0 : was + 1;
        else if (what == "ds_dec_u32") now = was == 0 || was > v ? v : was - 1;
        else if (what == "ds_min_i32") now = static_cast<uint32_t>(std::min(sw, sv));
        else if (what == "ds_max_i32") now = static_cast<uint32_t>(std::max(sw, sv));
        else if (what == "ds_min_u32") now = std::min(was, v);
        else if (what == "ds_max_u32") now = std::max(was, v);
        else if (what == "ds_and_b32") now = was & v;
        else if (what == "ds_or_b32") now = was | v;
        else if (what == "ds_xor_b32") now = was ^ v;
        else if (what == "ds_wrxchg_b32") now = v;
        std::memcpy(at(off), &now, 4);
        if (rtn) write_lane(w, in.dst[0], lane, was);
      } else if (op == "ds_min_i32"_op || op == "ds_min_u32"_op || op == "ds_max_u32"_op || op == "ds_and_b32"_op ||
                 op == "ds_or_b32"_op || op == "ds_add_rtn_u32"_op) {
        uint32_t was = 0;
        std::memcpy(&was, at(off), 4);
        const uint32_t v = lane_src(w, in.src[1], lane);
        const uint32_t now = op == "ds_min_i32"_op ? static_cast<uint32_t>(std::min(static_cast<int32_t>(was), static_cast<int32_t>(v)))
                             : op == "ds_min_u32"_op ? std::min(was, v)
                             : op == "ds_max_u32"_op ? std::max(was, v)
                             : op == "ds_and_b32"_op ? was & v
                             : op == "ds_or_b32"_op  ? was | v
                                                     : was + v;
        std::memcpy(at(off), &now, 4);
        if (op == "ds_add_rtn_u32"_op) write_lane(w, in.dst[0], lane, was);
      } else if (op == "ds_cmpst_rtn_b32"_op) {
        // The second value is stored where the first is found; either way
        // the lane is told what was there.
        uint32_t was = 0;
        std::memcpy(&was, at(off), 4);
        if (was == lane_src(w, in.src[1], lane)) {
          const uint32_t v = lane_src(w, in.src[2], lane);
          std::memcpy(at(off), &v, 4);
        }
        write_lane(w, in.dst[0], lane, was);
      } else if (op == "ds_cmpst_rtn_b64"_op) {
        uint64_t was = 0;
        std::memcpy(&was, at(off, 8), 8);
        if (was == lane_src64(w, in.src[1], lane)) {
          const uint64_t v = lane_src64(w, in.src[2], lane);
          std::memcpy(at(off, 8), &v, 8);
        }
        write_lane64(w, in.dst[0], lane, was);
      } else if (op == "ds_write_b96"_op) {
        uint8_t* const a = at(off, 12);
        for (uint32_t k = 0; k < 3; ++k) {
          const uint32_t v = word(w, in.src[1], k, lane);
          std::memcpy(a + 4 * k, &v, 4);
        }
      } else if (op == "ds_read_i16"_op) {
        int16_t v = 0;
        std::memcpy(&v, at(off, 2), 2);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(static_cast<int32_t>(v)));
      } else if (op == "ds_add_u64"_op ||
                 op == "ds_sub_u64"_op ||
                 op == "ds_rsub_u64"_op ||
                 op == "ds_inc_u64"_op ||
                 op == "ds_dec_u64"_op ||
                 op == "ds_min_i64"_op ||
                 op == "ds_max_i64"_op ||
                 op == "ds_min_u64"_op ||
                 op == "ds_max_u64"_op ||
                 op == "ds_and_b64"_op ||
                 op == "ds_or_b64"_op ||
                 op == "ds_xor_b64"_op ||
                 op == "ds_add_f64"_op ||
                 op == "ds_min_f64"_op ||
                 op == "ds_max_f64"_op ||
                 op == "ds_add_rtn_u64"_op ||
                 op == "ds_sub_rtn_u64"_op ||
                 op == "ds_rsub_rtn_u64"_op ||
                 op == "ds_inc_rtn_u64"_op ||
                 op == "ds_dec_rtn_u64"_op ||
                 op == "ds_min_rtn_i64"_op ||
                 op == "ds_max_rtn_i64"_op ||
                 op == "ds_min_rtn_u64"_op ||
                 op == "ds_max_rtn_u64"_op ||
                 op == "ds_and_rtn_b64"_op ||
                 op == "ds_or_rtn_b64"_op ||
                 op == "ds_xor_rtn_b64"_op ||
                 op == "ds_wrxchg_rtn_b64"_op ||
                 op == "ds_min_rtn_f64"_op ||
                 op == "ds_max_rtn_f64"_op || op == "ds_add_rtn_f64"_op) {
        // LDS's 64-bit read-modify-writes; the _rtn forms hand back what
        // was there.
        std::string what(in.name);
        const bool rtn = what.find("_rtn") != std::string::npos;
        if (rtn) what.erase(what.find("_rtn"), 4);
        uint64_t was = 0;
        std::memcpy(&was, at(off, 8), 8);
        const uint64_t v = lane_src64(w, in.src[1], lane);
        const int64_t sw = static_cast<int64_t>(was), sv = static_cast<int64_t>(v);
        uint64_t now = was;
        if (what == "ds_add_u64") now = was + v;
        else if (what == "ds_sub_u64") now = was - v;
        else if (what == "ds_rsub_u64") now = v - was;
        else if (what == "ds_inc_u64") now = was >= v ? 0 : was + 1;
        else if (what == "ds_dec_u64") now = was == 0 || was > v ? v : was - 1;
        else if (what == "ds_min_i64") now = static_cast<uint64_t>(std::min(sw, sv));
        else if (what == "ds_max_i64") now = static_cast<uint64_t>(std::max(sw, sv));
        else if (what == "ds_min_u64") now = std::min(was, v);
        else if (what == "ds_max_u64") now = std::max(was, v);
        else if (what == "ds_and_b64") now = was & v;
        else if (what == "ds_or_b64") now = was | v;
        else if (what == "ds_xor_b64") now = was ^ v;
        else if (what == "ds_wrxchg_b64") now = v;
        else if (what == "ds_add_f64") now = as_bits(as_double(was) + as_double(v));
        else if (what == "ds_min_f64") now = as_bits(std::fmin(as_double(was), as_double(v)));
        else if (what == "ds_max_f64") now = as_bits(std::fmax(as_double(was), as_double(v)));
        std::memcpy(at(off, 8), &now, 8);
        if (rtn) write_lane64(w, in.dst[0], lane, was);
      } else if (op == "ds_xor_b32"_op || op == "ds_max_i32"_op) {
        uint32_t before = 0;
        std::memcpy(&before, at(static_cast<uint64_t>(in.offset)), 4);
        const uint32_t v = lane_src(w, in.src[1], lane);
        const uint32_t after = op == "ds_xor_b32"_op
                                   ? before ^ v
                                   : static_cast<uint32_t>(std::max(static_cast<int32_t>(before),
                                                                    static_cast<int32_t>(v)));
        std::memcpy(at(static_cast<uint64_t>(in.offset)), &after, 4);
      } else if (op == "ds_write2_b32"_op) {
        // Two words, each at its own offset, counted in words.
        const uint32_t v0 = lane_src(w, in.src[1], lane), v1 = lane_src(w, in.src[2], lane);
        std::memcpy(at(uint64_t{static_cast<uint32_t>(in.offset)} * 4), &v0, 4);
        std::memcpy(at(uint64_t{static_cast<uint32_t>(in.offset1)} * 4), &v1, 4);
      } else if (op == "ds_write2st64_b32"_op || op == "ds_write2_b64"_op || op == "ds_write2st64_b64"_op) {
        // Two values at two offsets, counted in values -- or, for the st64
        // forms, in 64s of them.
        const uint32_t words = in.src[1].width;
        const uint64_t unit = 4 * words * (op == "ds_write2_b64"_op ? 1 : 64);
        uint8_t* const a0 = at(uint64_t{static_cast<uint32_t>(in.offset)} * unit, 4 * words);
        uint8_t* const a1 = at(uint64_t{static_cast<uint32_t>(in.offset1)} * unit, 4 * words);
        for (uint32_t k = 0; k < words; ++k) {
          const uint32_t v0 = word(w, in.src[1], k, lane), v1 = word(w, in.src[2], k, lane);
          std::memcpy(a0 + 4 * k, &v0, 4);
          std::memcpy(a1 + 4 * k, &v1, 4);
        }
      } else if (op == "ds_read2_b64"_op || op == "ds_read2st64_b64"_op) {
        const uint64_t unit = 8 * (op == "ds_read2_b64"_op ? 1 : 64);
        uint8_t* const a0 = at(uint64_t{static_cast<uint32_t>(in.offset)} * unit, 8);
        uint8_t* const a1 = at(uint64_t{static_cast<uint32_t>(in.offset1)} * unit, 8);
        uint32_t v[4];
        std::memcpy(&v[0], a0, 8);
        std::memcpy(&v[2], a1, 8);
        for (uint32_t k = 0; k < 4; ++k) set_word(w, in.dst[0], k, lane, v[k]);
      } else if (op == "ds_read_b96"_op) {
        uint8_t* const a = at(off, 12);
        for (uint32_t k = 0; k < 3; ++k) {
          uint32_t v = 0;
          std::memcpy(&v, a + 4 * k, 4);
          set_word(w, in.dst[0], k, lane, v);
        }
      } else if (op == "ds_read2_b32"_op) {
        uint32_t v0 = 0, v1 = 0;
        std::memcpy(&v0, at(uint64_t{static_cast<uint32_t>(in.offset)} * 4), 4);
        std::memcpy(&v1, at(uint64_t{static_cast<uint32_t>(in.offset1)} * 4), 4);
        set_word(w, in.dst[0], 0, lane, v0);
        set_word(w, in.dst[0], 1, lane, v1);
      } else {
        throw Error::make(Err::Unsupported, "LDS instruction ", op, " is decoded but not implemented");
      }
    }
  }

  // What a load or store narrower than a register moves, and whether what it
  // loads keeps its sign: the name says so, and the three segments spell it
  // the same way.
  struct Narrow {
    uint32_t bytes = 0;
    bool sign = false;
  };
  static bool narrow(const std::string& op, Narrow* n) {
    const size_t at = op.rfind('_');
    if (at == std::string::npos) return false;
    const std::string what = op.substr(at + 1);
    if (what == "ubyte" || what == "byte") *n = {1, false};
    else if (what == "sbyte") *n = {1, true};
    else if (what == "ushort" || what == "short") *n = {2, false};
    else if (what == "sshort") *n = {2, true};
    else return false;
    return true;
  }

  // Two halves, or two bfloat16s, added pairwise: each sum rounded to
  // nearest, as a single half or bfloat16 add would be.
  static uint16_t to_bf16(float f) {
    uint32_t u = as_bits(f);
    if (std::isnan(f)) return static_cast<uint16_t>((u >> 16) | 0x40);
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<uint16_t>(u >> 16);
  }
  static uint32_t packed_add(uint32_t a, uint32_t b, bool bf16) {
    uint32_t out = 0;
    for (int k = 0; k < 2; ++k) {
      const uint16_t x = static_cast<uint16_t>(a >> (16 * k)), y = static_cast<uint16_t>(b >> (16 * k));
      uint16_t r;
      if (bf16) {
        r = to_bf16(as_float(uint32_t{x} << 16) + as_float(uint32_t{y} << 16));
      } else {
        _Float16 hx, hy;
        std::memcpy(&hx, &x, 2);
        std::memcpy(&hy, &y, 2);
        const _Float16 sum = static_cast<_Float16>(static_cast<float>(hx) + static_cast<float>(hy));
        std::memcpy(&r, &sum, 2);
      }
      out |= uint32_t{r} << (16 * k);
    }
    return out;
  }

  // The half-register forms (_d16): a byte or a short loaded into one half of
  // a register, or one half stored. A load clears the other half: with SRAM
  // ECC on, as it is on every card here, the whole register is written
  // (LLVM keeps the other half only where ECC is off). Tensile's tail loops
  // count on it, loading two halves into two registers and or-ing them. Read from the name
  // past its segment ("load_ubyte_d16_hi", "read_u16_d16", "write_b8_d16_hi").
  struct Half {
    uint32_t bytes = 2;
    bool sign = false, hi = false, store = false;
  };
  static bool half_access(std::string_view body, Half* h) {
    if (body.find("_d16") == std::string_view::npos) return false;
    h->hi = body.size() >= 3 && body.substr(body.size() - 3) == "_hi";
    h->store = body.rfind("store", 0) == 0 || body.rfind("write", 0) == 0;
    h->bytes = body.find("short") != std::string_view::npos || body.find("16_d16") != std::string_view::npos ? 2 : 1;
    h->sign = body.find("sbyte") != std::string_view::npos || body.find("i8") != std::string_view::npos;
    return true;
  }
  // What a load puts in its half: the byte widened to 16 bits, with its sign
  // where the name says so, or the short.
  static uint32_t half_load(uint64_t raw, const Half& h) {
    uint32_t v = static_cast<uint32_t>(raw) & (h.bytes == 1 ? 0xFFu : 0xFFFFu);
    if (h.bytes == 1 && h.sign) v = static_cast<uint16_t>(static_cast<int16_t>(static_cast<int8_t>(v)));
    return h.hi ? v << 16 : v;
  }
  // And what a store stores: the half it names.
  static uint32_t half_store(uint32_t v, const Half& h) { return h.hi ? v >> 16 : v; }
  // Where a 16-bit load's half goes. gfx942 and gfx90a, with SRAM ECC on,
  // write the whole register -- the other half zero, which LLVM knows (d16
  // loads do not preserve the unused bits there). RDNA, without ECC, keeps
  // the other half.
  void write_half_load(Wave& w, const Inst& in, const Operand& data, uint32_t lane, uint32_t v, const Half& h) {
    if (gcn::is_rdna(in.arch)) {
      const uint32_t mask = h.hi ? 0xFFFF0000u : 0x0000FFFFu;
      v = (word(w, data, 0, lane) & ~mask) | (v & mask);
    }
    write_lane(w, data, lane, v);
  }

  // What a narrow load puts in the register: the bytes it read, with the sign
  // carried into the rest where the name says so.
  static uint32_t widen(uint64_t raw, const Narrow& n) {
    if (!n.sign) return static_cast<uint32_t>(raw);
    return n.bytes == 1 ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)))
                        : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
  }

  // A flat address says for itself which memory it means: the shared
  // aperture is LDS, the private one the work-item's own memory, and
  // everything else is the device's. Most flat accesses reach only the
  // device's, and those are global accesses by another name.
  // Returns whether any lane's address was in LDS.
  // How many bytes a global or flat access moves per lane, for SCALE_OFFSET.
  static uint32_t access_bytes(const Inst& in) {
    const std::string_view body = std::string_view(in.name).substr(in.name.find('_') + 1);
    Narrow n;
    Half h;
    AtomicOp a;
    if (narrow(in.name, &n)) return n.bytes;
    if (half_access(body, &h)) return h.bytes;
    if ((body.rfind("load_dword", 0) == 0 || body.rfind("load_monitor_b", 0) == 0) && !in.dst.empty()) return 4 * in.dst[0].width;
    if (body.rfind("store_dword", 0) == 0 && in.src.size() > 1) return 4 * in.src[1].width;
    if (parse_atomic(body, &a)) return a.bytes;
    return 4;
  }
  // A lane's address in a flat access: a 64-bit one in a register pair, or -- gfx1250 -- a scalar pair's with a 32-bit
  // offset register added (scaled, where SCALE_OFFSET asks).
  uint64_t flat_address(Wave& w, const Inst& in, uint32_t lane, uint32_t bytes) {
    const uint64_t base = in.has_saddr ? scalar_field(w, in.saddr, true) + scaled_offset(in, lane_src(w, in.src[0], lane), bytes)
                                       : lane_src64(w, in.src[0], lane);
    return base + static_cast<uint64_t>(static_cast<int64_t>(in.offset));
  }

  bool flat_access(Wave& w, const Inst& in, Group& g) {
    const auto in_lds = [](uint64_t a) { return a >= kSharedBase && a < kSharedBase + kSharedSize; };
    const auto in_private = [](uint64_t a) { return a >= kPrivateBase && a < kPrivateBase + kPrivateSize; };
    const uint32_t bytes_per_lane = access_bytes(in);
    bool lds = false, priv = false;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint64_t addr = flat_address(w, in, lane, bytes_per_lane);
      lds = lds || in_lds(addr);
      priv = priv || in_private(addr);
    }
    if (!lds && !priv) {
      global_access(w, in);
      return false;
    }
    // Otherwise lane by lane, each to its own memory: loads and stores, of a
    // byte or a half or whole registers.
    const std::string_view body = std::string_view(in.name).substr(in.name.find('_') + 1);
    // An atomic, lane by lane: in LDS or private memory, which only this
    // work-group reaches, done in place; in the device's, as a global one.
    if (AtomicOp a; parse_atomic(body, &a)) {
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        if (!(w.exec >> lane & 1)) continue;
        const uint64_t addr = flat_address(w, in, lane, bytes_per_lane);
        uint64_t v = 0, expected = 0, before = 0;
        atomic_data(w, in, a, lane, &v, &expected);
        uint8_t* host = nullptr;
        if (in_lds(addr)) {
          const uint64_t where = addr - kSharedBase;
          if (where + a.bytes > g.lds.size())
            throw Error::make(Err::InvalidValue, "a flat atomic reaches LDS at ", where, ", past the ", g.lds.size(),
                              " bytes the kernel reserved");
          host = &g.lds[where];
        } else if (in_private(addr)) {
          host = scratch_at(g, w, lane, addr - kPrivateBase, a.bytes);
        }
        if (host) {
          std::memcpy(&before, host, a.bytes);
          const uint64_t after = atomic_result(a, before, v, expected);
          std::memcpy(host, &after, a.bytes);
        } else {
          before = atomic_rmw(addr, a, v, expected);
        }
        if (!in.dst.empty()) {
          if (a.bytes == 8) write_lane64(w, in.dst[0], lane, before);
          else write_lane(w, in.dst[0], lane, static_cast<uint32_t>(before));
        }
      }
      return lds;
    }
    Narrow n;
    const bool part = narrow(in.name, &n), storing = body.rfind("store", 0) == 0;
    if (!part && body.rfind("load_dword", 0) != 0 && body.rfind("store_dword", 0) != 0)
      throw Error::make(Err::Unsupported, in.name, " reaching LDS or private memory is not implemented");
    const uint32_t words = part ? 0 : storing ? in.src[1].width : in.dst[0].width;
    const uint32_t bytes = part ? n.bytes : 4 * words;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint64_t addr = flat_address(w, in, lane, bytes_per_lane);
      uint8_t* host = nullptr;   // where LDS or private memory keeps it; null for the device's
      if (in_lds(addr)) {
        const uint64_t where = addr - kSharedBase;
        if (where + bytes > g.lds.size())
          throw Error::make(Err::InvalidValue, "a flat access reaches LDS at ", where, ", past the ", g.lds.size(),
                            " bytes the kernel reserved");
        host = &g.lds[where];
      } else if (in_private(addr)) {
        host = scratch_at(g, w, lane, addr - kPrivateBase, bytes);
      }
      if (part && storing) {
        const uint32_t v = lane_src(w, in.src[1], lane);
        if (host) std::memcpy(host, &v, n.bytes);
        else store(addr, n.bytes, v);
      } else if (part) {
        uint64_t raw = 0;
        if (host) std::memcpy(&raw, host, n.bytes);
        else raw = load(addr, n.bytes);
        write_lane(w, in.dst[0], lane, widen(raw, n));
      } else {
        for (uint32_t k = 0; k < words; ++k) {
          if (storing) {
            const uint32_t v = word(w, in.src[1], k, lane);
            if (host) std::memcpy(host + 4 * k, &v, 4);
            else store(addr + 4 * k, 4, v);
          } else {
            uint32_t v = 0;
            if (host) std::memcpy(&v, host + 4 * k, 4);
            else v = static_cast<uint32_t>(load(addr + 4 * k, 4));
            set_word(w, in.dst[0], k, lane, v);
          }
        }
      }
    }
    return lds;
  }

  // A work-item's private memory, which a kernel spills into.
  void scratch_access(Wave& w, const Inst& in, Group& g) {
    const OpName op(in.name);
    const uint32_t words = op.rfind("scratch_store", 0) == 0 ? in.src[1].width : in.dst[0].width;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      // An offset in a register, one in a scalar register, and the
      // instruction's own, as many of them as it has.
      const uint64_t offset = (in.has_vaddr ? scaled_offset(in, lane_src(w, in.src[0], lane), 4 * words) : 0) +
                              (in.has_saddr ? scalar_field(w, in.saddr, false) : 0) + static_cast<uint64_t>(in.offset);
      // A narrow access moves one byte or two; every other one moves whole
      // registers.
      Narrow n;
      const bool part = narrow(op, &n);
      uint8_t* at = scratch_at(g, w, lane, offset, part ? n.bytes : 4 * words);
      // A byte or a half first: its name begins the same way a whole
      // register's does, so asking about the width has to come first.
      if (part && op.rfind("scratch_store", 0) == 0) {
        const uint32_t v = lane_src(w, in.src[1], lane);
        std::memcpy(at, &v, n.bytes);
      } else if (part) {
        uint64_t raw = 0;
        std::memcpy(&raw, at, n.bytes);
        write_lane(w, in.dst[0], lane, widen(raw, n));
      } else if (op.rfind("scratch_store", 0) == 0) {
        for (uint32_t k = 0; k < words; ++k) {
          const uint32_t v = word(w, in.src[1], k, lane);
          std::memcpy(at + 4 * k, &v, 4);
        }
      } else if (op.rfind("scratch_load", 0) == 0) {
        for (uint32_t k = 0; k < words; ++k) {
          uint32_t v = 0;
          std::memcpy(&v, at + 4 * k, 4);
          set_word(w, in.dst[0], k, lane, v);
        }
      } else {
        throw Error::make(Err::Unsupported, "scratch instruction ", op, " is decoded but not implemented");
      }
    }
  }

  // A load or a store through the private segment buffer (gfx90a's
  // scratch). The address the card forms is swizzled: from the resource and
  // the scalar offset, dword d of lane l's run is at d * 256 + l * 4 (index
  // stride 64, 4-byte elements, the lane's id added). Unswizzling that byte
  // address gives the work-item and its offset in its own private memory --
  // so a scalar offset that is a stack pointer scaled by the wave's size, as
  // LLVM keeps it, moves every lane's frame alike.
  void scratch_buffer_access(Wave& w, const Inst& in, Group& g, const Operand& data, const Operand& vaddr,
                             const Operand& soff) {
    const std::string_view body = std::string_view(in.name).substr(7);   // past "buffer_"
    const bool store_op = body.rfind("store", 0) == 0;
    Narrow n;
    const bool part = narrow(in.name, &n);
    Half h;
    const bool half = half_access(body, &h);
    if (!store_op && !part && !half && body.rfind("load_dword", 0) != 0)
      throw Error::make(Err::Unsupported, in.name, " through the private segment buffer, which this does not model");
    const uint64_t soffset = static_cast<uint32_t>(scalar(w, soff));
    const uint32_t words = half || part ? 1 : data.width;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint64_t offset = uint64_t{in.offen ? word(w, vaddr, 0, lane) : 0} + static_cast<uint32_t>(in.offset);
      const auto place = [&](uint64_t off, uint32_t bytes) -> uint8_t* {
        const uint64_t at = soffset + (off / 4) * 256 + lane * 4 + off % 4;
        const uint32_t who = static_cast<uint32_t>(at % 256 / 4);
        return scratch_at(g, w, who, at / 256 * 4 + at % 4, bytes);
      };
      for (uint32_t k = 0; k < words; ++k) {
        const uint32_t bytes = half ? h.bytes : part ? n.bytes : 4;
        uint8_t* p = place(offset + 4 * k, bytes);
        if (store_op) {
          const uint32_t v = half ? static_cast<uint32_t>(half_store(lane_src(w, data, lane), h))
                                  : word(w, data, k, lane);
          std::memcpy(p, &v, bytes);
        } else {
          uint32_t v = 0;
          std::memcpy(&v, p, bytes);
          if (half) write_half_load(w, in, data, lane, half_load(v, h), h);
          else if (part) write_lane(w, data, lane, widen(v, n));
          else set_word(w, data, k, lane, v);
        }
      }
    }
  }

  // ---- RDNA's image instructions --------------------------------------------------
  //
  // An image resource (T#, eight scalar registers) says where the texels are,
  // how they are laid out and what format they have; a sampler (S#, four)
  // says how a sample filters them and what a coordinate past an edge does
  // (vgpu/amd_image.hpp). Loads, stores and atomics take whole texel
  // coordinates; samples and gathers take them normalized to 0..1 (or in
  // texels, where the instruction or the sampler says unnormalized) and
  // filter, choosing the level by the LOD the instruction gives or works out.

  // One texel's four results, the resource's DST_SEL applied; `border`
  // where the coordinate fell outside and the sampler said to use its color.
  void fetch_texel(const image::Image& img, uint32_t level, int64_t x, int64_t y, int64_t z, uint32_t out[4]) {
    uint8_t bytes[16] = {};
    const uint32_t n = image::texel_bytes(img.format);
    const uint64_t addr = img.base + image::texel_offset(img, level, static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                                                        static_cast<uint32_t>(z));
    at(addr).read(addr, bytes, n);
    uint32_t raw[4];
    image::read_texel(img.format, bytes, raw);
    const bool ints = image::integer(img.format);
    for (int k = 0; k < 4; ++k) {
      switch (img.sel[k]) {
        case image::Sel::Zero: out[k] = 0; break;
        case image::Sel::One: out[k] = ints ? 1 : as_bits(1.0f); break;
        case image::Sel::X: out[k] = raw[0]; break;
        case image::Sel::Y: out[k] = raw[1]; break;
        case image::Sel::Z: out[k] = raw[2]; break;
        default: out[k] = raw[3]; break;
      }
    }
  }
  // Where an index along an axis of `n` texels lands, the sampler's mode for
  // that axis applied; false where it is the border.
  static bool wrap_index(image::Clamp mode, int64_t n, int64_t* i) {
    int64_t v = *i;
    switch (mode) {
      case image::Clamp::Wrap: v = ((v % n) + n) % n; break;
      case image::Clamp::Mirror: {
        const int64_t p = ((v % (2 * n)) + 2 * n) % (2 * n);
        v = p < n ? p : 2 * n - 1 - p;
        break;
      }
      case image::Clamp::ClampLastTexel: v = std::clamp<int64_t>(v, 0, n - 1); break;
      case image::Clamp::MirrorOnceLastTexel: v = std::clamp<int64_t>(v < 0 ? -v - 1 : v, 0, n - 1); break;
      case image::Clamp::ClampHalfBorder:
      case image::Clamp::ClampBorder:
        if (v < 0 || v >= n) return false;
        break;
      default:   // the mirror-once modes to the border
        v = v < 0 ? -v - 1 : v;
        if (v >= n) return false;
        break;
    }
    *i = v;
    return true;
  }
  void border_color(const image::Image& img, const image::Sampler& smp, uint32_t out[4]) {
    const bool ints = image::integer(img.format);
    const uint32_t one = ints ? 1 : as_bits(1.0f);
    out[0] = out[1] = out[2] = smp.border == 2 ? one : 0;
    out[3] = smp.border == 0 ? 0 : one;
  }
  // A texel through the sampler: its index along each axis wrapped, or the
  // border color.
  void sampled_texel(const image::Image& img, const image::Sampler& smp, uint32_t level, int64_t x, int64_t y,
                     int64_t z, uint32_t dims, uint32_t out[4]) {
    const int64_t n[3] = {image::level_width(img, level), image::level_height(img, level),
                          image::level_depth(img, level)};
    int64_t c[3] = {x, y, z};
    for (uint32_t k = 0; k < dims; ++k)
      if (!wrap_index(smp.clamp[k], n[k], &c[k])) return border_color(img, smp, out);
    fetch_texel(img, level, c[0], c[1], c[2], out);
  }
  // A sample at one level: `u`, `v`, `r` in texels (a coordinate a dimension
  // does not have is 0), `slice` the array layer. Point sampling reads the
  // texel the point is in; bilinear the two, four or eight around it,
  // weighted by how near each is.
  void sample_level(const image::Image& img, const image::Sampler& smp, uint32_t level, double u, double v, double r,
                    int64_t slice, uint32_t dims, bool linear, uint32_t out[4]) {
    const bool ints = image::integer(img.format);
    if (!linear || ints) {
      const int64_t x = static_cast<int64_t>(std::floor(u)), y = dims > 1 ? static_cast<int64_t>(std::floor(v)) : 0;
      const int64_t z = dims > 2 ? static_cast<int64_t>(std::floor(r)) : slice;
      return sampled_texel(img, smp, level, x, y, z, dims, out);
    }
    const double fu = u - 0.5, fv = v - 0.5, fr = r - 0.5;
    const int64_t x0 = static_cast<int64_t>(std::floor(fu)), y0 = static_cast<int64_t>(std::floor(fv)),
                  z0 = static_cast<int64_t>(std::floor(fr));
    const double a = fu - x0, b = fv - y0, c = fr - z0;
    double acc[4] = {0, 0, 0, 0};
    for (uint32_t corner = 0; corner < (1u << dims); ++corner) {
      const uint32_t dx = corner & 1, dy = (corner >> 1) & 1, dz = (corner >> 2) & 1;
      double weight = dx ? a : 1 - a;
      if (dims > 1) weight *= dy ? b : 1 - b;
      if (dims > 2) weight *= dz ? c : 1 - c;
      if (weight == 0) continue;
      uint32_t t[4];
      sampled_texel(img, smp, level, x0 + dx, dims > 1 ? y0 + dy : 0, dims > 2 ? z0 + dz : slice, dims, t);
      for (int k = 0; k < 4; ++k) acc[k] += weight * as_float(t[k]);
    }
    for (int k = 0; k < 4; ++k) out[k] = as_bits(static_cast<float>(acc[k]));
  }

  static image::Gen image_gen(gcn::Target t) {
    return t == gcn::Target::Gfx1030 ? image::Gen::Gfx10 : t == gcn::Target::Gfx1100 ? image::Gen::Gfx11 : image::Gen::Gfx12;
  }

  void image_access(Wave& w, const Inst& in) {
    if (!w.exec) return;
    const std::string& name = in.name;
    const bool gather = name.find("gather4") != std::string::npos;
    const bool sample = gather || name.find("sample") != std::string::npos || name == "image_get_lod";
    const bool store = name.rfind("image_store", 0) == 0;
    const bool atomic = name.rfind("image_atomic", 0) == 0;
    const bool resinfo = name == "image_get_resinfo";
    const size_t naddr = in.src.size() - (sample ? 3 : 2);
    const Operand& rsrc = in.src[1 + naddr];
    uint32_t t[8] = {};
    for (uint32_t k = 0; k < rsrc.width && k < 8; ++k) t[k] = w.sgpr[rsrc.index + k];
    const image::Image img = image::decode_image(t, image_gen(in.arch));
    image::Sampler smp;
    if (sample) {
      const Operand& so = in.src[2 + naddr];
      uint32_t sw[4];
      for (uint32_t k = 0; k < 4; ++k) sw[k] = w.sgpr[so.index + k];
      smp = image::decode_sampler(sw, image_gen(in.arch));
    }
    if (!resinfo && image::texel_bytes(img.format) == 0)
      throw Error::make(Err::Unsupported, name, " of an image whose format this does not read (data format ",
                        static_cast<int>(img.format.data), ")");
    const uint32_t dim = in.dim & 7;
    static const uint32_t kCoords[8] = {1, 2, 3, 3, 2, 3, 3, 4};
    static const uint32_t kSpatial[8] = {1, 2, 3, 2, 1, 2, 2, 2};   // coordinates that are positions, not layers
    // A load, store or atomic takes a cube's face as a layer (x, y, face), as
    // the hardware addresses it; sampling a cube, which picks the face from
    // a direction, is not modelled (ROCm's HIP makes no cube arrays).
    if ((dim == 3 && (sample || resinfo)) || dim >= 6)
      throw Error::make(Err::Unsupported, name, " of a cube or multisampled image, which this does not model");
    const uint32_t coords = kCoords[dim], spatial = kSpatial[dim];
    const bool layered = dim == 4 || dim == 5;
    const auto has = [&](const char* part) {
      const std::string p(part);
      for (size_t at = name.find(p); at != std::string::npos; at = name.find(p, at + 1)) {
        const size_t end = at + p.size();
        if (end == name.size() || name[end] == '_') return true;
      }
      return false;
    };
    // What the name says, worked out once rather than per lane.
    const bool has_o = has("_o"), has_b = has("_b"), has_c = has("_c"), has_d = has("_d"), has_l = has("_l"),
               has_lz = has("_lz"), has_mip = has("_mip"), has_cl = has("_cl"), has_g16 = has("_g16");
    // Every lane's address words, flattened across the address registers.
    struct Lane {
      uint32_t word[16] = {};
      uint32_t n = 0;
    };
    std::array<Lane, kLanes> lanes{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      for (size_t k = 1; k <= naddr; ++k)
        for (uint32_t d = 0; d < in.src[k].width && lanes[lane].n < 16; ++d)
          lanes[lane].word[lanes[lane].n++] = word(w, in.src[k], d, lane);
    }
    // Reads a lane's components in the order the ISA gives them: each group
    // (the offsets, the bias, the compare, each derivative vector, and the
    // coordinates with the LOD and clamp) starting a register, two to a
    // register with a16 (the derivatives with g16 too).
    struct Addr {
      double coord[4] = {0, 0, 0, 0};   // the coordinates, as floats (samples) or integers (the rest)
      double lod = 0, bias = 0, clamp = 0;
      double grad[2][3] = {};           // explicit derivatives: d/dx then d/dy of each spatial coordinate
      bool has_grad = false;
    };
    const bool a16 = in.a16, g16 = in.a16 || has_g16;
    const auto read_addr = [&](const Lane& l) {
      Addr a;
      uint32_t at = 0;
      const auto group = [&](uint32_t count, bool packed, bool floats, double* out) {
        for (uint32_t k = 0; k < count; ++k) {
          uint32_t v;
          if (packed) {
            v = (l.word[at + k / 2] >> (16 * (k & 1))) & 0xFFFF;
            if (floats) {
              _Float16 h;
              const uint16_t b = static_cast<uint16_t>(v);
              std::memcpy(&h, &b, 2);
              out[k] = static_cast<double>(h);
            } else {
              out[k] = static_cast<double>(v);
            }
          } else {
            v = l.word[at + k];
            out[k] = floats ? static_cast<double>(as_float(v)) : static_cast<double>(v);
          }
        }
        at += packed ? (count + 1) / 2 : count;
      };
      double scratch[8];
      if (sample) {
        if (has_o) group(1, false, false, scratch);   // texel offsets: not modelled yet, and rare
        if (has_b) group(1, a16, true, &a.bias);
        if (has_c) group(1, false, true, scratch);
        if (has_d) {
          a.has_grad = true;
          for (int v = 0; v < 2; ++v) group(spatial, g16, true, a.grad[v]);
        }
      }
      const bool lod = has_l || has_mip || resinfo;
      const bool cl = has_cl;
      double c[6];
      group(resinfo ? 1 : coords + (lod && !resinfo ? 1 : 0) + (cl ? 1 : 0), a16, sample, c);
      if (resinfo) {
        a.lod = c[0];
      } else {
        for (uint32_t k = 0; k < coords; ++k) a.coord[k] = c[k];
        if (lod) a.lod = c[coords];
        if (cl) a.clamp = c[coords + (lod ? 1 : 0)];
      }
      return a;
    };
    // The data: the channels DMASK names, in order, two to a register with d16.
    const auto write_result = [&](uint32_t lane, const uint32_t v[4], bool ints) {
      if (in.dst.empty()) return;
      uint32_t out[4] = {0, 0, 0, 0}, n = 0;
      for (int k = 0; k < 4; ++k)
        if (gather ? k < 4 : (in.dmask >> k) & 1) out[n++] = v[k];
      if (gather) n = 4;
      if (n == 0) out[n++] = v[0];
      const Operand& d = in.dst[0];
      if (in.d16) {
        for (uint32_t k = 0; k < n; k += 2) {
          uint32_t packed = 0;
          for (uint32_t h = 0; h < 2 && k + h < n; ++h) {
            uint32_t half;
            if (ints) {
              half = out[k + h] & 0xFFFF;
            } else {
              const _Float16 f = static_cast<_Float16>(as_float(out[k + h]));
              uint16_t b;
              std::memcpy(&b, &f, 2);
              half = b;
            }
            packed |= half << (16 * h);
          }
          set_word(w, d, k / 2, lane, packed);
        }
        if (in.tfe || in.lwe) set_word(w, d, (n + 1) / 2, lane, 0);
      } else {
        for (uint32_t k = 0; k < n; ++k) set_word(w, d, k, lane, out[k]);
        if (in.tfe || in.lwe) set_word(w, d, n, lane, 0);
      }
    };
    const uint32_t levels = img.last_level >= img.base_level ? img.last_level - img.base_level + 1 : 1;
    std::array<Addr, kLanes> addrs{};
    for (uint32_t lane = 0; lane < kLanes; ++lane)
      if (w.exec >> lane & 1) addrs[lane] = read_addr(lanes[lane]);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const Addr& a = addrs[lane];
      if (resinfo) {
        const uint32_t level = img.base_level + static_cast<uint32_t>(a.lod);
        const uint32_t v[4] = {image::level_width(img, level),
                               layered && dim == 4 ? img.depth : image::level_height(img, level),
                               dim == 2 ? image::level_depth(img, level) : layered ? img.depth : 1u, levels};
        write_result(lane, v, true);
        continue;
      }
      if (!sample) {
        // Whole texel coordinates, and the level for the _mip forms.
        const uint32_t level = img.base_level + (has_mip ? static_cast<uint32_t>(a.lod) : 0);
        const int64_t x = static_cast<int64_t>(a.coord[0]), y = coords > 1 ? static_cast<int64_t>(a.coord[1]) : 0;
        const int64_t z = coords > 2 ? static_cast<int64_t>(a.coord[2]) : 0;
        const int64_t zz = dim == 4 ? y : z, yy = dim == 4 ? 0 : y;   // a 1D array's layer is its second coordinate
        const bool inside = level <= img.last_level && x >= 0 && x < image::level_width(img, level) && yy >= 0 &&
                            yy < image::level_height(img, level) && zz >= 0 && zz < image::level_depth(img, level);
        const uint64_t addr =
            img.base + (inside ? image::texel_offset(img, level, static_cast<uint32_t>(x), static_cast<uint32_t>(yy),
                                                     static_cast<uint32_t>(zz))
                               : 0);
        const bool ints = image::integer(img.format);
        if (store) {
          if (!inside) continue;   // a store past the edge is dropped
          uint8_t bytes[16] = {};
          const uint32_t nbytes = image::texel_bytes(img.format);
          at(addr).read(addr, bytes, nbytes);
          uint32_t raw[4];
          image::read_texel(img.format, bytes, raw);
          uint32_t k = 0;
          for (int c = 0; c < 4; ++c) {
            if (!((in.dmask >> c) & 1)) continue;
            uint32_t v = in.d16 ? (word(w, in.src[0], k / 2, lane) >> (16 * (k & 1))) & 0xFFFF
                                : word(w, in.src[0], k, lane);
            if (in.d16 && !ints) {
              _Float16 h;
              const uint16_t b = static_cast<uint16_t>(v);
              std::memcpy(&h, &b, 2);
              v = as_bits(static_cast<float>(h));
            }
            raw[c] = v;
            ++k;
          }
          image::write_texel(img.format, raw, bytes);
          at(addr).write(addr, bytes, nbytes);
          continue;
        }
        if (atomic) {
          if (image::texel_bytes(img.format) != 4)
            throw Error::make(Err::Unsupported, name, " of an image whose texels are not 32 bits");
          const std::string_view body = std::string_view(name).substr(6);   // "atomic_add"
          AtomicOp op;
          if (!parse_atomic(body, &op)) throw Error::make(Err::Unsupported, name, " is decoded but not implemented");
          const uint64_t v = word(w, in.src[0], 0, lane);
          const uint64_t expected = op.rmw == Rmw::CmpSwap ? word(w, in.src[0], 1, lane) : 0;
          const uint64_t before = inside ? atomic_rmw(addr, op, v, expected) : 0;
          if (!in.dst.empty()) set_word(w, in.dst[0], 0, lane, static_cast<uint32_t>(before));
          continue;
        }
        uint32_t v[4] = {0, 0, 0, 0};
        if (inside) fetch_texel(img, level, x, yy, zz, v);
        write_result(lane, v, ints);
        continue;
      }
      // A sample, or a gather: the LOD from what the instruction gives, or
      // from how fast the coordinates change across the lane's quad.
      const uint32_t w0 = img.width, h0 = image::level_height(img, 0), d0 = image::level_depth(img, 0);
      const bool unnorm = in.unorm || smp.unnormalized;
      const double scale[3] = {unnorm ? 1.0 : double(w0), unnorm ? 1.0 : double(h0), unnorm ? 1.0 : double(d0)};
      double lod = 0;
      if (has_l) {
        lod = a.lod;
      } else if (!has_lz && !gather) {
        double dx[3] = {}, dy[3] = {};
        if (a.has_grad) {
          for (uint32_t k = 0; k < spatial; ++k) dx[k] = a.grad[0][k], dy[k] = a.grad[1][k];
        } else {
          const uint32_t qx = lane ^ 1, qy = lane ^ 2;
          const Addr& ax = (w.exec >> qx & 1) ? addrs[qx] : a;
          const Addr& ay = (w.exec >> qy & 1) ? addrs[qy] : a;
          const double sx = (lane & 1) ? -1 : 1, sy = (lane & 2) ? -1 : 1;
          for (uint32_t k = 0; k < spatial; ++k) dx[k] = sx * (ax.coord[k] - a.coord[k]), dy[k] = sy * (ay.coord[k] - a.coord[k]);
        }
        double px = 0, py = 0;
        for (uint32_t k = 0; k < spatial; ++k) px += dx[k] * scale[k] * dx[k] * scale[k], py += dy[k] * scale[k] * dy[k] * scale[k];
        const double rho = std::sqrt(std::max(px, py));
        lod = rho > 0 ? std::log2(rho) : -1e30;
      }
      lod += smp.lod_bias + a.bias;
      lod = std::clamp(lod, static_cast<double>(smp.min_lod), static_cast<double>(smp.max_lod));
      if (has_cl) lod = std::max(lod, a.clamp);
      const bool linear = lod <= 0 ? smp.mag_linear : smp.min_linear;
      const double top = static_cast<double>(levels - 1);
      const double u = a.coord[0], vv = coords > 1 ? a.coord[1] : 0, r = coords > 2 ? a.coord[2] : 0;
      const int64_t slice =
          dim == 4 ? std::clamp<int64_t>(std::llround(vv), 0, img.depth - 1)
          : dim == 5 ? std::clamp<int64_t>(std::llround(r), 0, img.depth - 1) : 0;
      const auto at_level = [&](uint32_t rel, uint32_t out[4]) {
        const uint32_t level = img.base_level + rel;
        const double lw = image::level_width(img, level), lh = image::level_height(img, level),
                     ld = image::level_depth(img, level);
        // In texels, kept to the 1/256 of a texel the hardware keeps: a
        // coordinate HIP rounds to a texel's edge (floor(x * w) / w, for
        // point sampling) lands on that edge, not a hair before it.
        const auto snap = [](double t) { return std::round(t * 256.0) / 256.0; };
        const double tu = snap(unnorm ? u : u * lw), tv = snap(unnorm ? vv : vv * lh), tr = snap(unnorm ? r : r * ld);
        if (gather) {
          // The four texels bilinear filtering would weigh, one channel of
          // each: (x0, y1), (x1, y1), (x1, y0), (x0, y0).
          const int64_t x0 = static_cast<int64_t>(std::floor(tu - 0.5)), y0 = static_cast<int64_t>(std::floor(tv - 0.5));
          uint32_t ch = 0;
          while (ch < 3 && !((in.dmask >> ch) & 1)) ++ch;
          const int64_t xs[4] = {x0, x0 + 1, x0 + 1, x0}, ys[4] = {y0 + 1, y0 + 1, y0, y0};
          for (int k = 0; k < 4; ++k) {
            uint32_t tex[4];
            sampled_texel(img, smp, level, xs[k], ys[k], slice, 2, tex);
            out[k] = tex[ch];
          }
          return;
        }
        sample_level(img, smp, level, tu, spatial > 1 ? tv : 0, spatial > 2 ? tr : 0, slice, spatial, linear, out);
      };
      uint32_t v[4];
      if (smp.mip_filter == 0 || levels == 1 || gather) {
        at_level(0, v);
      } else if (smp.mip_filter == 1) {
        at_level(static_cast<uint32_t>(std::clamp(std::floor(lod + 0.5), 0.0, top)), v);
      } else {
        const double l = std::clamp(lod, 0.0, top);
        const uint32_t l0 = static_cast<uint32_t>(std::floor(l)), l1 = std::min(l0 + 1, levels - 1);
        const double f = l - l0;
        uint32_t a0[4], a1[4];
        at_level(l0, a0);
        at_level(l1, a1);
        for (int k = 0; k < 4; ++k) v[k] = as_bits(static_cast<float>(as_float(a0[k]) * (1 - f) + as_float(a1[k]) * f));
      }
      write_result(lane, v, image::integer(img.format));
    }
  }

  // A load, a store or an atomic through a buffer resource: four scalar
  // registers giving the buffer's base address, the stride of its records and
  // how many there are (bytes, where the stride is zero). An access past the
  // end is not a fault but a nothing -- a load reads zero, a store or an
  // atomic is dropped -- which is what Tensile's GEMMs count on at the edges
  // of a matrix. The scalar offset counts against the bounds as the rest of
  // the offset does: Tensile's DGEMM moves the resource's base back and
  // walks the scalar offset past the end, and gets zeroes there only if it
  // counts. Each register's worth is checked on its own.
  // One lane's formatted buffer access: the element's channels, as many as
  // the name's x, xy, xyz or xyzw says, through the resource's DST_SEL on a
  // load (a channel the format lacks reading 0, alpha 1); half-width with
  // d16, into a register's high half with d16_hi. A store leaves the
  // channels it does not give as they were; outside the buffer a load reads
  // 0 and a store is dropped.
  void formatted_access(Wave& w, const Inst& in, std::string_view body, image::Format format, uint32_t d3,
                        const Operand& data, uint32_t lane, uint64_t addr, bool inside) {
    const bool storing = body.rfind("store", 0) == 0, d16 = body.find("_d16") != std::string_view::npos;
    const bool hi = body.ends_with("_hi_x");
    const std::string_view xyzw = body.substr(body.rfind('_') + 1);
    const uint32_t count = static_cast<uint32_t>(xyzw.size());
    const bool ints = image::integer(format);
    const uint32_t nbytes = image::texel_bytes(format);
    const auto to_half = [&](uint32_t v) -> uint32_t {
      if (ints) return v & 0xFFFF;
      const _Float16 f = static_cast<_Float16>(as_float(v));
      uint16_t b;
      std::memcpy(&b, &f, 2);
      return b;
    };
    const auto from_half = [&](uint32_t v) -> uint32_t {
      if (ints) return v & 0xFFFF;
      _Float16 f;
      const uint16_t b = static_cast<uint16_t>(v);
      std::memcpy(&f, &b, 2);
      return as_bits(static_cast<float>(f));
    };
    uint8_t bytes[16] = {};
    uint32_t raw[4] = {0, 0, 0, 0};
    if (inside) {
      at(addr).read(addr, bytes, nbytes);
      image::read_texel(format, bytes, raw);
    }
    if (storing) {
      if (!inside) return;
      for (uint32_t k = 0; k < count; ++k) {
        const uint32_t v = d16 ? (word(w, data, k / 2, lane) >> (16 * ((k & 1) | (hi ? 1 : 0)))) & 0xFFFF
                               : word(w, data, k, lane);
        raw[k] = d16 ? from_half(v) : v;
      }
      image::write_texel(format, raw, bytes);
      at(addr).write(addr, bytes, nbytes);
      return;
    }
    uint32_t out[4];
    for (uint32_t k = 0; k < 4; ++k) {
      switch (static_cast<image::Sel>((d3 >> (3 * k)) & 7)) {
        case image::Sel::Zero: out[k] = 0; break;
        case image::Sel::One: out[k] = ints ? 1 : as_bits(1.0f); break;
        case image::Sel::X: out[k] = raw[0]; break;
        case image::Sel::Y: out[k] = raw[1]; break;
        case image::Sel::Z: out[k] = raw[2]; break;
        default: out[k] = raw[3]; break;
      }
      if (!inside) out[k] = 0;
    }
    if (!d16) {
      for (uint32_t k = 0; k < count; ++k) set_word(w, data, k, lane, out[k]);
    } else if (hi) {
      set_word(w, data, 0, lane, (word(w, data, 0, lane) & 0xFFFF) | to_half(out[0]) << 16);
    } else {
      for (uint32_t k = 0; k < count; k += 2)
        set_word(w, data, k / 2, lane, to_half(out[k]) | (k + 1 < count ? to_half(out[k + 1]) << 16 : 0));
    }
  }

  void buffer_access(Wave& w, const Inst& in, Group& g) {
    if (!w.exec) return;
    const bool reads_data = in.dst.empty();   // a store, or an atomic
    const Operand& data = reads_data ? in.src[0] : in.dst[0];
    const Operand& vaddr = in.src[reads_data ? 1 : 0];
    const Operand& rsrc = in.src[reads_data ? 2 : 1];
    const Operand& soff = in.src[reads_data ? 3 : 2];
    const uint32_t d1 = w.sgpr[rsrc.index + 1], records = w.sgpr[rsrc.index + 2], d3 = w.sgpr[rsrc.index + 3];
    const uint64_t base = w.sgpr[rsrc.index] | static_cast<uint64_t>(d1 & 0xFFFF) << 32;
    if (base == kScratchResourceBase) return scratch_buffer_access(w, in, g, data, vaddr, soff);
    const uint32_t stride = (d1 >> 16) & 0x3FFF;
    // RDNA's resource (V#) keeps the same base, stride and record count, but
    // a 2-bit swizzle, a 7-bit format at bits 18:12 in place of gfx9's data
    // and number formats, and OOB_SELECT (bits 29:28), which says how an
    // access is checked against the buffer.
    const bool rdna = gcn::is_rdna(in.arch);
    if (rdna ? (d1 >> 30) != 0 : (d1 >> 31) != 0)
      throw Error::make(Err::Unsupported, in.name, " through a swizzled buffer, which this does not model");
    if ((d3 >> 23) & 1)
      throw Error::make(Err::Unsupported, in.name, " through a buffer that adds the lane's id, which this does not model");
    const uint64_t soffset = static_cast<uint32_t>(scalar(w, soff));
    const bool valid_format = rdna ? ((d3 >> 12) & 0x7F) != 0 : ((d3 >> 15) & 0xF) != 0;
    const uint32_t oob_select = (d3 >> 28) & 3;
    const std::string_view body = std::string_view(in.name).substr(in.enc == gcn::Enc::Mtbuf ? 8 : 7);   // past "buffer_"
    // A formatted access converts each element between its format (the
    // resource's, or MTBUF's own) and a register's floats or integers.
    image::Format format;
    const bool formatted = body.rfind("load_format_", 0) == 0 || body.rfind("store_format_", 0) == 0;
    if (formatted) {
      if (in.enc == gcn::Enc::Mtbuf) {
        format = image::from_code(in.format, image_gen(in.arch));
      } else if (rdna) {
        const uint32_t words[4] = {w.sgpr[rsrc.index], d1, records, d3};
        format = image::buffer_format(words, image_gen(in.arch));
      } else {   // gfx9's two fields: NUM_FORMAT [14:12], DATA_FORMAT [18:15]
        format.data = static_cast<image::Data>((d3 >> 15) & 0xF);
        format.num = static_cast<image::Num>((d3 >> 12) & 7);
      }
      if (image::texel_bytes(format) == 0)
        throw Error::make(Err::Unsupported, in.name, " of a format this does not convert (",
                          static_cast<int>(format.data), "/", static_cast<int>(format.num), ")");
    }
    Narrow n;
    const bool part = narrow(in.name, &n);
    const bool atomic = body.rfind("atomic_", 0) == 0;
    const bool returns = atomic && (in.cache & 1);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint32_t index = in.idxen ? word(w, vaddr, 0, lane) : 0;
      const uint64_t offset = uint64_t{in.offen ? word(w, vaddr, in.idxen ? 1 : 0, lane) : 0} +
                              static_cast<uint32_t>(in.offset);
      const uint64_t addr = base + soffset + offset + uint64_t{index} * stride;
      // A resource whose data format is 0 (BUF_DATA_FORMAT_INVALID) holds
      // nothing: every access is out of range, which MIOpen's hand-written
      // kernels use to switch a store off.
      const auto fits = [&](uint64_t at, uint64_t bytes) {
        if (!valid_format) return false;
        if (rdna) switch (oob_select) {
            case 0:   // structured: the index within the records, the offset within an element
              return stride ? index < records && at + bytes <= stride : soffset + at + bytes <= records;
            case 1: return index < records;              // the index alone
            case 2: return records != 0;                 // only an empty buffer is out of range
            default: return soffset + at + bytes <= records;   // raw: bytes, from the offset
          }
        return stride ? index < records : soffset + at + bytes <= records;
      };
      if (formatted) {
        formatted_access(w, in, body, format, d3, data, lane, addr, fits(offset, image::texel_bytes(format)));
      } else if (Half h; half_access(body, &h)) {
        if (h.store) {
          if (fits(offset, h.bytes)) store(addr, h.bytes, half_store(lane_src(w, data, lane), h));
        } else {
          write_half_load(w, in, data, lane, half_load(fits(offset, h.bytes) ? load(addr, h.bytes) : 0, h), h);
        }
      } else if (part) {
        if (body.rfind("store", 0) == 0) {
          if (fits(offset, n.bytes)) store(addr, n.bytes, lane_src(w, data, lane));
        } else {
          write_lane(w, data, lane, fits(offset, n.bytes) ? widen(load(addr, n.bytes), n) : 0);
        }
      } else if (body.rfind("load_dword", 0) == 0) {
        if (data.width <= kMaxWords && fits(offset, 4 * data.width)) {
          uint32_t words[kMaxWords];
          load_words(addr, data.width, words);
          for (uint32_t k = 0; k < data.width; ++k) set_word(w, data, k, lane, words[k]);
        } else {  // partly out of range: the words past the end read zero
          for (uint32_t k = 0; k < data.width; ++k)
            set_word(w, data, k, lane, fits(offset + 4 * k, 4) ? static_cast<uint32_t>(load(addr + 4 * k, 4)) : 0);
        }
      } else if (body.rfind("store_dword", 0) == 0) {
        if (data.width <= kMaxWords && fits(offset, 4 * data.width)) {
          uint32_t words[kMaxWords];
          for (uint32_t k = 0; k < data.width; ++k) words[k] = word(w, data, k, lane);
          store_words(addr, data.width, words);
        } else {  // partly out of range: the words past the end are dropped
          for (uint32_t k = 0; k < data.width; ++k)
            if (fits(offset + 4 * k, 4)) store(addr + 4 * k, 4, word(w, data, k, lane));
        }
      } else if (atomic) {
        const bool wide = body == "atomic_cmpswap_x2";
        const uint32_t bytes = wide ? 8 : 4;
        uint64_t before = 0;
        if (fits(offset, bytes)) {
          const auto guard = atomic_guard(addr);
          before = at(addr).load_scalar(addr, bytes);
          uint64_t after;
          if (body == "atomic_cmpswap_x2") {
            const uint64_t value = word(w, data, 0, lane) | uint64_t{word(w, data, 1, lane)} << 32;
            const uint64_t expected = word(w, data, 2, lane) | uint64_t{word(w, data, 3, lane)} << 32;
            after = before == expected ? value : before;
          } else if (body == "atomic_cmpswap") {
            after = before == word(w, data, 1, lane) ? word(w, data, 0, lane) : before;
          } else if (body == "atomic_add") {
            after = static_cast<uint32_t>(before + word(w, data, 0, lane));
          } else if (body == "atomic_swap") {
            after = word(w, data, 0, lane);
          } else if (body == "atomic_add_f32") {
            after = as_bits(as_float(static_cast<uint32_t>(before)) + as_float(word(w, data, 0, lane)));
          } else if (body == "atomic_pk_add_f16") {
            after = packed_add(static_cast<uint32_t>(before), word(w, data, 0, lane), false);
          } else if (const uint32_t old = static_cast<uint32_t>(before), v = word(w, data, 0, lane);
                     body == "atomic_sub") {
            after = old - v;
          } else if (body == "atomic_smin" || body == "atomic_smax") {
            const int32_t a = static_cast<int32_t>(old), b = static_cast<int32_t>(v);
            after = static_cast<uint32_t>(body == "atomic_smin" ? std::min(a, b) : std::max(a, b));
          } else if (body == "atomic_umin") {
            after = std::min(old, v);
          } else if (body == "atomic_umax") {
            after = std::max(old, v);
          } else if (body == "atomic_and") {
            after = old & v;
          } else if (body == "atomic_or") {
            after = old | v;
          } else if (body == "atomic_xor") {
            after = old ^ v;
          } else if (body == "atomic_inc") {
            // Counts up to the data, then wraps to zero.
            after = old >= v ? 0 : old + 1;
          } else if (body == "atomic_dec") {
            // Counts down to zero, then wraps to the data (as does anything
            // above it).
            after = old == 0 || old > v ? v : old - 1;
          } else {
            throw Error::make(Err::Unsupported, in.name, " is decoded but not implemented");
          }
          at(addr).store_scalar(addr, bytes, after);
        }
        if (returns) {
          set_word(w, data, 0, lane, static_cast<uint32_t>(before));
          if (wide) set_word(w, data, 1, lane, static_cast<uint32_t>(before >> 32));
        }
      } else {
        throw Error::make(Err::Unsupported, in.name, " is decoded but not implemented");
      }
    }
  }

  // Loads straight into LDS (global_load_lds_dword; gfx950's _dwordx3 and
  // _dwordx4): each lane reads its element from global memory and writes it
  // into the work-group's LDS at M0, plus the instruction's offset, plus the
  // lane's number times the element's size -- as LLVM's global_load_lds
  // intrinsics lay it out. No register is written.
  void global_load_lds(Wave& w, const Inst& in, Group& g) {
    const std::string& op = in.name;
    const uint32_t words = op.ends_with("_dwordx4") ? 4 : op.ends_with("_dwordx3") ? 3 : 1;
    const uint32_t bytes = 4 * words;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint64_t addr = (in.has_saddr ? scalar_field(w, in.saddr, true) + lane_src(w, in.src[0], lane)
                                          : lane_src64(w, in.src[0], lane)) +
                            static_cast<uint64_t>(static_cast<int64_t>(in.offset));
      const uint64_t at = uint64_t{w.m0} + static_cast<uint64_t>(static_cast<int64_t>(in.offset)) + uint64_t{lane} * bytes;
      if (at + bytes > g.lds.size())
        throw Error::make(Err::InvalidValue, op, " writes LDS at ", at, ", past the ", g.lds.size(),
                          " bytes the work-group has");
      uint32_t v[4];
      load_words(addr, words, v);
      std::memcpy(&g.lds[at], v, bytes);
    }
  }

  // A read-modify-write of global or flat memory: what it does, and how
  // wide it is.
  enum class Rmw { Add, Sub, And, Or, Xor, Swap, AddF32, SMin, UMin, SMax, UMax, PkAddF16, PkAddBf16, Inc, Dec,
                   AddF64, MinF64, MaxF64, MinF32, MaxF32, CmpSwap };
  struct AtomicOp {
    Rmw rmw = Rmw::Add;
    uint32_t bytes = 4;
  };
  // An atomic, read from the name past its segment ("atomic_add_x2"); false
  // for anything that is not one this knows.
  static bool parse_atomic(std::string_view body, AtomicOp* a) {
    if (body.rfind("atomic_", 0) != 0) return false;
    std::string_view what = body.substr(7);
    a->bytes = 4;
    if (what.size() > 3 && what.substr(what.size() - 3) == "_x2") {
      a->bytes = 8;
      what.remove_suffix(3);
    }
    static const std::pair<std::string_view, Rmw> kNames[] = {
        {"add", Rmw::Add},       {"sub", Rmw::Sub},         {"and", Rmw::And},           {"or", Rmw::Or},
        {"xor", Rmw::Xor},       {"swap", Rmw::Swap},       {"smin", Rmw::SMin},         {"umin", Rmw::UMin},
        {"smax", Rmw::SMax},     {"umax", Rmw::UMax},       {"inc", Rmw::Inc},           {"dec", Rmw::Dec},
        {"cmpswap", Rmw::CmpSwap}, {"add_f32", Rmw::AddF32}, {"pk_add_f16", Rmw::PkAddF16},
        {"pk_add_bf16", Rmw::PkAddBf16}, {"add_f64", Rmw::AddF64}, {"min_f64", Rmw::MinF64},
        {"max_f64", Rmw::MaxF64}, {"min_num_f64", Rmw::MinF64}, {"max_num_f64", Rmw::MaxF64},
        // RDNA's float min and max, as each generation spells them: gfx10's
        // fmin (fmin_x2 the double), gfx11's min_f32, gfx12's min_num_f32.
        {"fmin", Rmw::MinF32}, {"fmax", Rmw::MaxF32}, {"min_f32", Rmw::MinF32}, {"max_f32", Rmw::MaxF32},
        {"min_num_f32", Rmw::MinF32}, {"max_num_f32", Rmw::MaxF32}};
    for (auto [name, rmw] : kNames)
      if (what == name) {
        if (a->bytes == 8 && rmw == Rmw::MinF32) rmw = Rmw::MinF64;
        if (a->bytes == 8 && rmw == Rmw::MaxF32) rmw = Rmw::MaxF64;
        a->rmw = rmw;
        if (rmw == Rmw::AddF64 || rmw == Rmw::MinF64 || rmw == Rmw::MaxF64) a->bytes = 8;
        return !(a->bytes == 8 && (rmw == Rmw::AddF32 || rmw == Rmw::PkAddF16 || rmw == Rmw::PkAddBf16));
      }
    return false;
  }
  // What an atomic leaves where it found `before`, given its data `v` and,
  // for a compare-and-swap, what it must find. inc counts up to the data,
  // then wraps to zero; dec counts down to zero, then wraps to the data (as
  // does anything above it); the float min and max keep the number where the
  // other is a NaN.
  static uint64_t atomic_result(const AtomicOp& a, uint64_t before, uint64_t v, uint64_t expected) {
    if (a.bytes == 4) {
      const uint32_t b = static_cast<uint32_t>(before), x = static_cast<uint32_t>(v);
      const int32_t sb = static_cast<int32_t>(b), sx = static_cast<int32_t>(x);
      switch (a.rmw) {
        case Rmw::Add: return uint32_t(b + x);
        case Rmw::Sub: return uint32_t(b - x);
        case Rmw::And: return b & x;
        case Rmw::Or: return b | x;
        case Rmw::Xor: return b ^ x;
        case Rmw::Swap: return x;
        case Rmw::AddF32: return as_bits(as_float(b) + as_float(x));
        case Rmw::MinF32: return as_bits(std::fmin(as_float(b), as_float(x)));
        case Rmw::MaxF32: return as_bits(std::fmax(as_float(b), as_float(x)));
        case Rmw::SMin: return static_cast<uint32_t>(std::min(sb, sx));
        case Rmw::UMin: return std::min(b, x);
        case Rmw::SMax: return static_cast<uint32_t>(std::max(sb, sx));
        case Rmw::UMax: return std::max(b, x);
        case Rmw::Inc: return b >= x ? 0 : b + 1;
        case Rmw::Dec: return b == 0 || b > x ? x : b - 1;
        case Rmw::PkAddF16: return packed_add(b, x, false);
        case Rmw::PkAddBf16: return packed_add(b, x, true);
        case Rmw::CmpSwap: return b == static_cast<uint32_t>(expected) ? x : b;
        default: return b;
      }
    }
    const int64_t sb = static_cast<int64_t>(before), sx = static_cast<int64_t>(v);
    switch (a.rmw) {
      case Rmw::Add: return before + v;
      case Rmw::Sub: return before - v;
      case Rmw::And: return before & v;
      case Rmw::Or: return before | v;
      case Rmw::Xor: return before ^ v;
      case Rmw::Swap: return v;
      case Rmw::SMin: return static_cast<uint64_t>(std::min(sb, sx));
      case Rmw::UMin: return std::min(before, v);
      case Rmw::SMax: return static_cast<uint64_t>(std::max(sb, sx));
      case Rmw::UMax: return std::max(before, v);
      case Rmw::Inc: return before >= v ? 0 : before + 1;
      case Rmw::Dec: return before == 0 || before > v ? v : before - 1;
      case Rmw::AddF64: return as_bits(as_double(before) + as_double(v));
      case Rmw::MinF64: return as_bits(std::fmin(as_double(before), as_double(v)));
      case Rmw::MaxF64: return as_bits(std::fmax(as_double(before), as_double(v)));
      case Rmw::CmpSwap: return before == expected ? v : before;
      default: return before;
    }
  }
  // A lane's data for an atomic, and what a compare-and-swap must find: the
  // register (or pair) after the data.
  void atomic_data(const Wave& w, const Inst& in, const AtomicOp& a, uint32_t lane, uint64_t* v,
                   uint64_t* expected) const {
    if (a.bytes == 4) {
      *v = lane_src(w, in.src[1], lane);
      *expected = a.rmw == Rmw::CmpSwap ? w.vgpr[in.src[1].index + 1][lane] : 0;
    } else {
      *v = lane_src64(w, in.src[1], lane);
      *expected = a.rmw == Rmw::CmpSwap
                      ? (w.vgpr[in.src[1].index + 2][lane] | uint64_t{w.vgpr[in.src[1].index + 3][lane]} << 32)
                      : 0;
    }
  }
  // An atomic on memory a device reaches, returning what it found. Host
  // memory the device maps (pinned, registered, managed) is updated with the
  // CPU's own atomics there, so a host thread's atomics on it and a kernel's
  // all land; the rest under the lock that makes work-groups on other threads
  // take turns.
  uint64_t atomic_rmw(uint64_t addr, const AtomicOp& a, uint64_t v, uint64_t expected) {
    MemoryManager& m = at(addr);
    if (uint8_t* h = m.host_address(addr, a.bytes); h && reinterpret_cast<uintptr_t>(h) % a.bytes == 0) {
      if (a.bytes == 4) {
        std::atomic_ref<uint32_t> r(*reinterpret_cast<uint32_t*>(h));
        uint32_t before = r.load();
        while (!r.compare_exchange_weak(before, static_cast<uint32_t>(atomic_result(a, before, v, expected)))) {
        }
        return before;
      }
      std::atomic_ref<uint64_t> r(*reinterpret_cast<uint64_t*>(h));
      uint64_t before = r.load();
      while (!r.compare_exchange_weak(before, atomic_result(a, before, v, expected))) {
      }
      return before;
    }
    const auto guard = atomic_guard(addr);
    uint64_t before = 0;
    uint32_t valid = a.bytes;   // fewer on the last word of an allocation (word_tail)
    // The bytes of a word that runs past the allocation's end, one at a time
    // and each atomically, as every other access to device memory is: a
    // plain copy here raced with another work-group's load of the same word.
    const auto load_bytes = [&](uint32_t n) {
      uint64_t x = 0;
      for (uint32_t i = 0; i < n; ++i) x |= m.load_scalar(addr + i, 1) << (8 * i);
      return x;
    };
    const auto store_bytes = [&](uint64_t x, uint32_t n) {
      for (uint32_t i = 0; i < n; ++i) m.store_scalar(addr + i, 1, (x >> (8 * i)) & 0xFF);
    };
    try {
      before = m.load_scalar(addr, a.bytes);
    } catch (...) {
      valid = word_tail(m, addr, a.bytes);
      if (!valid) throw;
      before = load_bytes(valid);
    }
    const uint64_t after = atomic_result(a, before, v, expected);
    if (after != before) {
      if (valid == a.bytes) {
        try {
          m.store_scalar(addr, a.bytes, after);
        } catch (...) {
          valid = word_tail(m, addr, a.bytes);
          if (!valid) throw;
          store_bytes(after, valid);
        }
      } else {
        store_bytes(after, valid);
      }
    }
    return before;
  }

  // A 32-bit offset register's contribution to an address: as it is, or -- with gfx1250's SCALE_OFFSET -- as a
  // signed number times the size of what is moved.
  static uint64_t scaled_offset(const Inst& in, uint32_t reg, uint32_t bytes) {
    if (!in.scale_offset) return reg;
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(reg)) * bytes);
  }

  void global_access(Wave& w, const Inst& in) {
    if (!w.exec) return;   // no lane to reach memory for
    const std::string& op = in.name;
    // What the instruction does is settled once; its lanes then each do it.
    // A flat access that reaches only the device's memory comes here too, so
    // what it does is read from its name past the segment ("load_dwordx4").
    const std::string_view body = std::string_view(op).substr(op.find('_') + 1);
    enum class Kind { Narrow, HalfReg, Load, Store, StoreHi16, Atomic } kind;
    Half half;
    AtomicOp atomic;
    Narrow n;
    bool narrow_store = false;
    if (narrow(op, &n)) {
      kind = Kind::Narrow;
      narrow_store = body.rfind("store", 0) == 0;
    } else if (half_access(body, &half)) {
      kind = Kind::HalfReg;
    } else if (body.rfind("prefetch_", 0) == 0) {
      return;   // gfx1250's prefetch into the cache: there is none to warm
    } else if (body.rfind("load_dword", 0) == 0 || body.rfind("load_monitor_b", 0) == 0) {
      kind = Kind::Load;
    } else if (body.rfind("store_dword", 0) == 0) {
      kind = Kind::Store;
    } else if (parse_atomic(body, &atomic)) {
      kind = Kind::Atomic;
    } else {
      throw Error::make(Err::Unsupported, "memory instruction ", op, " is decoded but not implemented");
    }
    // Lanes of one load that read the same address read it once, as the card
    // reads it in one transaction: the compiler counts on a wave-uniform
    // address giving every lane the same value (it branches the whole wave
    // on it), and lane by lane, another work-group's store landing between
    // two lanes gave them different ones -- unsafeAtomicMax's compare-and-
    // swap loop then left lanes that never swapped holding the old value.
    uint64_t memo_addr = 0;
    bool memo = false;
    uint32_t memo_words[kMaxWords];
    uint64_t memo_value = 0;
    const auto load_once = [&](uint64_t addr, uint32_t bytes) {
      if (!memo || memo_addr != addr) memo_value = load(addr, bytes), memo_addr = addr, memo = true;
      return memo_value;
    };
    // gfx1250's SCALE_OFFSET multiplies the 32-bit offset register by what the access moves.
    const uint32_t access_bytes = kind == Kind::Narrow    ? n.bytes
                                  : kind == Kind::HalfReg ? half.bytes
                                  : kind == Kind::Load    ? 4 * in.dst[0].width
                                  : kind == Kind::Store   ? 4 * in.src[1].width
                                  : kind == Kind::StoreHi16 ? 2
                                                            : atomic.bytes;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      // The address is a 64-bit one in a register pair, or a scalar base with
      // a 32-bit offset per lane.
      const uint64_t offset_reg = in.has_saddr ? scaled_offset(in, lane_src(w, in.src[0], lane), access_bytes) : 0;
      const uint64_t addr = (in.has_saddr ? scalar_field(w, in.saddr, true) + offset_reg
                                          : lane_src64(w, in.src[0], lane)) +
                            static_cast<uint64_t>(static_cast<int64_t>(in.offset));
      switch (kind) {
        case Kind::Narrow:
          if (narrow_store) store(addr, n.bytes, lane_src(w, in.src[1], lane));
          else write_lane(w, in.dst[0], lane, widen(load_once(addr, n.bytes), n));
          break;
        case Kind::Load: {
          // One word, or two, or four: a register each, in order.
          if (in.dst[0].width > kMaxWords) throw Error::make(Err::Unsupported, "memory instruction ", op, " moves more than 16 words");
          if (!memo || memo_addr != addr) {
            load_words(addr, in.dst[0].width, memo_words);
            memo_addr = addr, memo = true;
          }
          for (uint32_t k = 0; k < in.dst[0].width; ++k) set_word(w, in.dst[0], k, lane, memo_words[k]);
          break;
        }
        case Kind::Store: {
          uint32_t words[kMaxWords];
          if (in.src[1].width > kMaxWords) throw Error::make(Err::Unsupported, "memory instruction ", op, " moves more than 16 words");
          for (uint32_t k = 0; k < in.src[1].width; ++k) words[k] = word(w, in.src[1], k, lane);
          store_words(addr, in.src[1].width, words);
          break;
        }
        case Kind::StoreHi16:
          store(addr, 2, lane_src(w, in.src[1], lane) >> 16);
          break;
        case Kind::HalfReg:
          if (half.store) store(addr, half.bytes, half_store(lane_src(w, in.src[1], lane), half));
          else write_half_load(w, in, in.dst[0], lane, half_load(load_once(addr, half.bytes), half), half);
          break;
        case Kind::Atomic: {
          // Lane by lane, which is what makes these atomic within a wave:
          // every lane's turn lands, whatever order they come in, and each is
          // told what it found.
          uint64_t v = 0, expected = 0;
          atomic_data(w, in, atomic, lane, &v, &expected);
          const uint64_t before = atomic_rmw(addr, atomic, v, expected);
          if (!in.dst.empty()) {
            if (atomic.bytes == 8) write_lane64(w, in.dst[0], lane, before);
            else write_lane(w, in.dst[0], lane, static_cast<uint32_t>(before));
          }
          break;
        }
      }
    }
  }

  // Which lane a lane reads its first source from, when the instruction says
  // another lane's. A row is sixteen lanes and a bank is four of those; a
  // shift, a rotate or a mirror stays inside its row, and the two broadcasts
  // carry the last lane of a row into the rows above it. False where there is
  // no such lane.
  //
  // The direction is not a guess: the sequence a wave adds itself up with
  // shifts by one, two, four and eight and ends with the total in the last
  // lane, which only comes out if each lane reads the lane below it.
  static bool dpp_source(uint32_t ctrl, uint32_t lane, uint32_t* from) {
    const uint32_t row = lane & ~15u, in_row = lane & 15u;
    if (ctrl <= 0xFF) {   // four lanes choosing among their own four
      *from = (lane & ~3u) + ((ctrl >> (2 * (lane & 3))) & 3);
      return true;
    }
    if (ctrl >= 0x101 && ctrl <= 0x10F) {   // row_shl: the lane above
      const uint32_t n = ctrl - 0x100;
      if (in_row + n > 15) return false;
      *from = row + in_row + n;
      return true;
    }
    if (ctrl >= 0x111 && ctrl <= 0x11F) {   // row_shr: the lane below
      const uint32_t n = ctrl - 0x110;
      if (n > in_row) return false;
      *from = row + in_row - n;
      return true;
    }
    if (ctrl >= 0x121 && ctrl <= 0x12F) {   // row_ror: the same, wrapping
      const uint32_t n = ctrl - 0x120;
      *from = row + ((in_row + 16 - n) & 15);
      return true;
    }
    // GCN's shifts and rotates of the whole wave by one lane (gone from RDNA):
    // wave_shl reads the lane above, wave_shr the lane below, as the row forms do.
    if (ctrl == 0x130) {   // wave_shl:1
      if (lane == kLanes - 1) return false;
      *from = lane + 1;
      return true;
    }
    if (ctrl == 0x134) {   // wave_rol:1
      *from = (lane + 1) % kLanes;
      return true;
    }
    if (ctrl == 0x138) {   // wave_shr:1
      if (lane == 0) return false;
      *from = lane - 1;
      return true;
    }
    if (ctrl == 0x13C) {   // wave_ror:1
      *from = (lane + kLanes - 1) % kLanes;
      return true;
    }
    if (ctrl == 0x140) {   // the row reversed
      *from = row + (15 - in_row);
      return true;
    }
    if (ctrl == 0x141) {   // each half of the row reversed
      *from = (lane & ~7u) + (7 - (lane & 7));
      return true;
    }
    if (ctrl == 0x142) {   // the last lane of the row below
      if (lane < 16) return false;
      *from = row - 1;
      return true;
    }
    if (ctrl == 0x143) {   // the last lane of the half of the wave below
      if (lane < 32) return false;
      *from = (lane & ~31u) - 1;
      return true;
    }
    if (ctrl >= 0x150 && ctrl <= 0x15F) {   // row_share (gfx90a's row_newbcast): lane n of the row
      *from = row + (ctrl - 0x150);
      return true;
    }
    if (ctrl >= 0x160 && ctrl <= 0x16F) {   // row_xmask: the lane of the row n away by XOR
      *from = row + (in_row ^ (ctrl - 0x160));
      return true;
    }
    throw Error::make(Err::Unsupported, "a cross-lane instruction reading across the whole wave rather than "
                                        "within a row, which this decodes but does not model");
  }

  // What each lane's first source comes to, and which lanes the instruction
  // writes at all: a row and a bank mask say which, and bound_ctrl says
  // whether a lane with no lane to read gets a zero or is left alone.
  uint64_t dpp_shuffle(const Wave& w, const Inst& in, std::array<uint32_t, kLanes>& values) {
    const Operand& o = in.src[0];
    uint64_t writes = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      values[lane] = 0;
      if (!(w.exec >> lane & 1)) continue;
      if (!((in.row_mask >> (lane >> 4)) & 1)) continue;
      if (!((in.bank_mask >> ((lane >> 2) & 3)) & 1)) continue;
      uint32_t from = 0;
      // (RDNA's FI reads a lane that is off as though it were on.)
      const bool there = dpp_source(in.dpp_ctrl, lane, &from) && (in.fi || ((w.exec >> from) & 1));
      if (!there && !in.bound_ctrl) continue;   // nothing to read, and nothing written
      values[lane] = there ? w.vgpr[o.index][from] : 0u;
      writes |= uint64_t{1} << lane;
    }
    return writes;
  }

  // The matrix instructions: D = A x B + C over the whole wave, with each
  // matrix spread across the 64 lanes -- or, for the multi-block forms,
  // several such products side by side, a block to a group of lanes. For a
  // block M rows by N columns with K in the product, lane l of the lanes a
  // block has, and g = l / M the group of M lanes it is in:
  //   A: row l % M, the K / groups values of k from g * (K / groups) on;
  //   B: column l % N, the same k;
  //   C and D: column l % N, and the rows given by out_row below.
  // For v_mfma_f32_16x16x16_f16 that arrangement is checked, not assumed:
  // rocWMMA -- AMD's library whose loads and stores put each element where
  // the hardware expects it -- is run through this and compared with the
  // same product worked out in C. The float and double forms are checked
  // through rocBLAS's GEMMs against a GEMM done on the host.
  //
  // Each output's products and addend are summed in double (or, for the
  // double forms, fused in order) and rounded once. Where every partial sum
  // is exact any order gives the same answer; where it is not, a card may
  // round differently. A wave with lanes switched off is refused: the
  // instruction works on all 64 at once, and what a card does with some of
  // them off is not something this can check.
  struct MatrixShape {
    uint32_t m, n, k, blocks;
    char in, out;   // 'h' a half, 'b' a bfloat16, 'c' a signed byte, 'e' an fp8, 'g' a bf8,
                    // 'f' a float, 'F' an xf32 (a float with a 10-bit mantissa), 'd' a double, 'i' an int32
    char in_b = 0;  // B's type, where it is not A's
  };
  static const MatrixShape* matrix_shape(const OpName& op) {
    static const MatrixShape f16_16x16x16{16, 16, 16, 1, 'h', 'f'}, f32_16x16x4{16, 16, 4, 1, 'f', 'f'},
        f32_32x32x2{32, 32, 2, 1, 'f', 'f'}, f32_16x16x1_4b{16, 16, 1, 4, 'f', 'f'},
        f32_4x4x1_16b{4, 4, 1, 16, 'f', 'f'}, f32_32x32x1_2b{32, 32, 1, 2, 'f', 'f'}, f64_16x16x4{16, 16, 4, 1, 'd', 'd'},
        bf16_16x16x16{16, 16, 16, 1, 'b', 'f'}, f16_32x32x8{32, 32, 8, 1, 'h', 'f'},
        bf16_32x32x8{32, 32, 8, 1, 'b', 'f'}, i8_32x32x16{32, 32, 16, 1, 'c', 'i'},
        f16_4x4x4_16b{4, 4, 4, 16, 'h', 'f'}, bf16_4x4x4_16b{4, 4, 4, 16, 'b', 'f'},
        xf32_16x16x8{16, 16, 8, 1, 'F', 'f'}, xf32_32x32x4{32, 32, 4, 1, 'F', 'f'},
        f16_16x16x4_4b{16, 16, 4, 4, 'h', 'f'}, bf16_16x16x4_4b{16, 16, 4, 4, 'b', 'f'},
        i8_16x16x32{16, 16, 32, 1, 'c', 'i'}, f8_16x16x32[4] = {{16, 16, 32, 1, 'g', 'f', 'g'},
                                                                 {16, 16, 32, 1, 'g', 'f', 'e'},
                                                                 {16, 16, 32, 1, 'e', 'f', 'g'},
                                                                 {16, 16, 32, 1, 'e', 'f', 'e'}},
        f8_32x32x16[4] = {{32, 32, 16, 1, 'g', 'f', 'g'},
                          {32, 32, 16, 1, 'g', 'f', 'e'},
                          {32, 32, 16, 1, 'e', 'f', 'g'},
                          {32, 32, 16, 1, 'e', 'f', 'e'}};
    // The 8-bit float forms, named for A's type then B's.
    if (op == "v_mfma_f32_16x16x32_bf8_bf8"_op) return &f8_16x16x32[0];
    if (op == "v_mfma_f32_16x16x32_bf8_fp8"_op) return &f8_16x16x32[1];
    if (op == "v_mfma_f32_16x16x32_fp8_bf8"_op) return &f8_16x16x32[2];
    if (op == "v_mfma_f32_16x16x32_fp8_fp8"_op) return &f8_16x16x32[3];
    if (op == "v_mfma_f32_32x32x16_bf8_bf8"_op) return &f8_32x32x16[0];
    if (op == "v_mfma_f32_32x32x16_bf8_fp8"_op) return &f8_32x32x16[1];
    if (op == "v_mfma_f32_32x32x16_fp8_bf8"_op) return &f8_32x32x16[2];
    if (op == "v_mfma_f32_32x32x16_fp8_fp8"_op) return &f8_32x32x16[3];
    if (op == "v_mfma_f32_16x16x16_f16"_op) return &f16_16x16x16;
    // The reduced-precision float forms (gfx940 to gfx942): a float's mantissa cut to 10 bits.
    if (op == "v_mfma_f32_16x16x8_xf32"_op) return &xf32_16x16x8;
    if (op == "v_mfma_f32_32x32x4_xf32"_op) return &xf32_32x32x4;
    if (op == "v_mfma_f32_16x16x4_f32"_op) return &f32_16x16x4;
    if (op == "v_mfma_f32_32x32x2_f32"_op) return &f32_32x32x2;
    if (op == "v_mfma_f32_16x16x1_4b_f32"_op) return &f32_16x16x1_4b;
    if (op == "v_mfma_f32_4x4x1_16b_f32"_op) return &f32_4x4x1_16b;
    if (op == "v_mfma_f32_32x32x1_2b_f32"_op) return &f32_32x32x1_2b;
    if (op == "v_mfma_f64_16x16x4_f64"_op) return &f64_16x16x4;
    if (op == "v_mfma_f32_16x16x16_bf16"_op) return &bf16_16x16x16;
    // gfx950's, with K doubled.
    static const MatrixShape f16_16x16x32{16, 16, 32, 1, 'h', 'f'}, f16_32x32x16{32, 32, 16, 1, 'h', 'f'},
        bf16_16x16x32{16, 16, 32, 1, 'b', 'f'}, bf16_32x32x16{32, 32, 16, 1, 'b', 'f'},
        i8_16x16x64{16, 16, 64, 1, 'c', 'i'}, i8_32x32x32{32, 32, 32, 1, 'c', 'i'},
        // 'x': a small float whose format the instruction's CBSZ or BLGP names.
        f8f6f4_16x16x128{16, 16, 128, 1, 'x', 'f', 'x'}, f8f6f4_32x32x64{32, 32, 64, 1, 'x', 'f', 'x'};
    // gfx90a's, which gfx940 dropped: int8 with K of 8 and 16, and bfloat16s
    // two to a lane.
    static const MatrixShape i8_32x32x8{32, 32, 8, 1, 'c', 'i'}, i8_16x16x16{16, 16, 16, 1, 'c', 'i'},
        bf16_32x32x2{32, 32, 2, 2, 'b', 'f'}, bf16_16x16x2{16, 16, 2, 4, 'b', 'f'},
        bf16_4x4x2{4, 4, 2, 16, 'b', 'f'}, bf16_32x32x4{32, 32, 4, 1, 'b', 'f'}, bf16_16x16x8{16, 16, 8, 1, 'b', 'f'},
        bf16_32x32x4_2b{32, 32, 4, 2, 'b', 'f'};
    static const MatrixShape f16_32x32x4_2b{32, 32, 4, 2, 'h', 'f'}, i8_32x32x4_2b{32, 32, 4, 2, 'c', 'i'},
        i8_16x16x4_4b{16, 16, 4, 4, 'c', 'i'}, i8_4x4x4_16b{4, 4, 4, 16, 'c', 'i'};
    if (op == "v_mfma_f32_32x32x4_2b_f16"_op) return &f16_32x32x4_2b;
    if (op == "v_mfma_i32_32x32x4_2b_i8"_op) return &i8_32x32x4_2b;
    if (op == "v_mfma_i32_16x16x4_4b_i8"_op) return &i8_16x16x4_4b;
    if (op == "v_mfma_i32_4x4x4_16b_i8"_op) return &i8_4x4x4_16b;
    if (op == "v_mfma_i32_32x32x8_i8"_op) return &i8_32x32x8;
    if (op == "v_mfma_i32_16x16x16_i8"_op) return &i8_16x16x16;
    if (op == "v_mfma_f32_32x32x2bf16"_op) return &bf16_32x32x2;
    if (op == "v_mfma_f32_16x16x2bf16"_op) return &bf16_16x16x2;
    if (op == "v_mfma_f32_4x4x2bf16"_op) return &bf16_4x4x2;
    if (op == "v_mfma_f32_32x32x4bf16"_op) return &bf16_32x32x4;
    if (op == "v_mfma_f32_16x16x8bf16"_op) return &bf16_16x16x8;
    if (op == "v_mfma_f32_32x32x4_2b_bf16"_op) return &bf16_32x32x4_2b;
    if (op == "v_mfma_i32_16x16x64_i8"_op) return &i8_16x16x64;
    if (op == "v_mfma_i32_32x32x32_i8"_op) return &i8_32x32x32;
    if (op == "v_mfma_f32_16x16x128_f8f6f4"_op) return &f8f6f4_16x16x128;
    if (op == "v_mfma_f32_32x32x64_f8f6f4"_op) return &f8f6f4_32x32x64;
    if (op == "v_mfma_f32_16x16x32_f16"_op) return &f16_16x16x32;
    if (op == "v_mfma_f32_32x32x16_f16"_op) return &f16_32x32x16;
    if (op == "v_mfma_f32_16x16x32_bf16"_op) return &bf16_16x16x32;
    if (op == "v_mfma_f32_32x32x16_bf16"_op) return &bf16_32x32x16;
    if (op == "v_mfma_f32_32x32x8_f16"_op) return &f16_32x32x8;
    if (op == "v_mfma_f32_32x32x8_bf16"_op) return &bf16_32x32x8;
    if (op == "v_mfma_i32_32x32x16_i8"_op) return &i8_32x32x16;
    if (op == "v_mfma_i32_16x16x32_i8"_op) return &i8_16x16x32;
    // The sparse forms, by the dense product each works out: A 2:4 sparse
    // along the K here, the lanes and output placed as a dense one's.
    static const MatrixShape f8_16x16x64[4] = {{16, 16, 64, 1, 'g', 'f', 'g'},
                                               {16, 16, 64, 1, 'g', 'f', 'e'},
                                               {16, 16, 64, 1, 'e', 'f', 'g'},
                                               {16, 16, 64, 1, 'e', 'f', 'e'}},
                             f8_32x32x32[4] = {{32, 32, 32, 1, 'g', 'f', 'g'},
                                               {32, 32, 32, 1, 'g', 'f', 'e'},
                                               {32, 32, 32, 1, 'e', 'f', 'g'},
                                               {32, 32, 32, 1, 'e', 'f', 'e'}},
                             f8_16x16x128[4] = {{16, 16, 128, 1, 'g', 'f', 'g'},
                                                {16, 16, 128, 1, 'g', 'f', 'e'},
                                                {16, 16, 128, 1, 'e', 'f', 'g'},
                                                {16, 16, 128, 1, 'e', 'f', 'e'}},
                             f8_32x32x64[4] = {{32, 32, 64, 1, 'g', 'f', 'g'},
                                               {32, 32, 64, 1, 'g', 'f', 'e'},
                                               {32, 32, 64, 1, 'e', 'f', 'g'},
                                               {32, 32, 64, 1, 'e', 'f', 'e'}},
                             f16_16x16x64{16, 16, 64, 1, 'h', 'f'}, f16_32x32x32{32, 32, 32, 1, 'h', 'f'},
                             bf16_16x16x64{16, 16, 64, 1, 'b', 'f'}, bf16_32x32x32{32, 32, 32, 1, 'b', 'f'},
                             i8_16x16x128{16, 16, 128, 1, 'c', 'i'}, i8_32x32x64{32, 32, 64, 1, 'c', 'i'};
    if (op == "v_smfmac_f32_16x16x32_f16"_op) return &f16_16x16x32;
    if (op == "v_smfmac_f32_32x32x16_f16"_op) return &f16_32x32x16;
    if (op == "v_smfmac_f32_16x16x32_bf16"_op) return &bf16_16x16x32;
    if (op == "v_smfmac_f32_32x32x16_bf16"_op) return &bf16_32x32x16;
    if (op == "v_smfmac_i32_16x16x64_i8"_op) return &i8_16x16x64;
    if (op == "v_smfmac_i32_32x32x32_i8"_op) return &i8_32x32x32;
    if (op == "v_smfmac_f32_16x16x64_bf8_bf8"_op) return &f8_16x16x64[0];
    if (op == "v_smfmac_f32_16x16x64_bf8_fp8"_op) return &f8_16x16x64[1];
    if (op == "v_smfmac_f32_16x16x64_fp8_bf8"_op) return &f8_16x16x64[2];
    if (op == "v_smfmac_f32_16x16x64_fp8_fp8"_op) return &f8_16x16x64[3];
    if (op == "v_smfmac_f32_32x32x32_bf8_bf8"_op) return &f8_32x32x32[0];
    if (op == "v_smfmac_f32_32x32x32_bf8_fp8"_op) return &f8_32x32x32[1];
    if (op == "v_smfmac_f32_32x32x32_fp8_bf8"_op) return &f8_32x32x32[2];
    if (op == "v_smfmac_f32_32x32x32_fp8_fp8"_op) return &f8_32x32x32[3];
    if (op == "v_smfmac_f32_16x16x64_f16"_op) return &f16_16x16x64;
    if (op == "v_smfmac_f32_32x32x32_f16"_op) return &f16_32x32x32;
    if (op == "v_smfmac_f32_16x16x64_bf16"_op) return &bf16_16x16x64;
    if (op == "v_smfmac_f32_32x32x32_bf16"_op) return &bf16_32x32x32;
    if (op == "v_smfmac_i32_16x16x128_i8"_op) return &i8_16x16x128;
    if (op == "v_smfmac_i32_32x32x64_i8"_op) return &i8_32x32x64;
    if (op == "v_smfmac_f32_16x16x128_bf8_bf8"_op) return &f8_16x16x128[0];
    if (op == "v_smfmac_f32_16x16x128_bf8_fp8"_op) return &f8_16x16x128[1];
    if (op == "v_smfmac_f32_16x16x128_fp8_bf8"_op) return &f8_16x16x128[2];
    if (op == "v_smfmac_f32_16x16x128_fp8_fp8"_op) return &f8_16x16x128[3];
    if (op == "v_smfmac_f32_32x32x64_bf8_bf8"_op) return &f8_32x32x64[0];
    if (op == "v_smfmac_f32_32x32x64_bf8_fp8"_op) return &f8_32x32x64[1];
    if (op == "v_smfmac_f32_32x32x64_fp8_bf8"_op) return &f8_32x32x64[2];
    if (op == "v_smfmac_f32_32x32x64_fp8_fp8"_op) return &f8_32x32x64[3];
    if (op == "v_mfma_f32_4x4x4_16b_f16"_op) return &f16_4x4x4_16b;
    if (op == "v_mfma_f32_4x4x4_16b_bf16"_op) return &bf16_4x4x4_16b;
    if (op == "v_mfma_f32_16x16x4_4b_f16"_op) return &f16_16x16x4_4b;
    if (op == "v_mfma_f32_16x16x4_4b_bf16"_op) return &bf16_16x16x4_4b;
    return nullptr;
  }
  // Which block, and which row of it, output value r of lane l is: a 16- or
  // 4-row block holds four rows a lane group, a 32-row one eight groups of
  // four rows spread over its two halves of the wave. A double's block holds
  // one row a lane group, the next four rows on: the vector width rocWMMA
  // gives a double accumulator on this architecture is 1, a float's 4.
  static void out_place(const MatrixShape& s, uint32_t lane, uint32_t r, uint32_t* block, uint32_t* row) {
    if (s.out == 'd') {        // one 16x16 block of doubles
      *block = 0;
      *row = lane / 16 + 4 * r;
    } else if (s.m == 4) {     // sixteen 4x4 blocks, one to each four lanes
      *block = lane / 4;
      *row = r;
    } else if (s.m == 32) {    // 32x32 blocks, sixteen registers to each
      *block = r / 16;
      *row = 8 * (r % 16 / 4) + 4 * (lane / 32) + r % 4;
    } else {                   // 16x16 blocks, four rows a register to each
      *block = r / 4;
      *row = 4 * (lane / 16) + r % 4;
    }
  }
  void matrix_multiply(Wave& w, const Inst& in) {
    const MatrixShape* shape =
        matrix_shape(OpName(in.scaled ? "v_mfma_" + in.name.substr(std::string("v_mfma_scale_").size()) : in.name));
    if (!shape) throw Error::make(Err::Unsupported, in.name, " is decoded but not implemented");
    const MatrixShape& s = *shape;
    // The sparse forms: the destination is the accumulator too, and the
    // third source the indices of A's values.
    const bool sparse = in.name.rfind("v_smfmac", 0) == 0;
    if (w.exec != ~uint64_t{0})
      throw Error::make(Err::Unsupported, in.name, " with lanes switched off (EXEC ", w.exec,
                        "), which this does not model");
    const auto reg = [&](const Operand& o, uint32_t k, uint32_t lane) -> uint32_t {
      if (o.kind == OperandKind::Agpr) return o.index + k < w.agpr.size() ? w.agpr[o.index + k][lane] : 0;
      if (o.kind == OperandKind::Vgpr) return w.vgpr[o.index + k][lane];
      // An inline constant, the same in every lane and register: a float
      // constant a float's bits in each, whatever the operand's width.
      if (o.kind == OperandKind::InlineFloat) return as_bits(static_cast<float>(o.fvalue));
      return static_cast<uint32_t>(scalar(w, o));
    };
    // Value e of a source's run in one lane: a half or a bfloat16 (two to a
    // register), a signed byte (four), a float or an int32, or a double (a
    // register pair). A bfloat16 is a float's top half, so it widens exactly.
    // A small float of `bits` bits, element e of a run packed from the first
    // register's lowest bit up: a 6-bit one may straddle two registers.
    const auto packed = [&](const Operand& o, uint32_t bits, uint32_t e, uint32_t lane) -> uint32_t {
      const uint32_t at = e * bits, word = at / 32, shift = at % 32;
      uint64_t v = reg(o, word, lane);
      if (shift + bits > 32 && word + 1 < o.width) v |= static_cast<uint64_t>(reg(o, word + 1, lane)) << 32;
      return static_cast<uint32_t>(v >> shift) & ((1u << bits) - 1);
    };
    const auto value = [&](const Operand& o, char type, uint32_t e, uint32_t lane) -> double {
      if (type == 'x') {   // an f8f6f4 source: its format is CBSZ's (A) or BLGP's (B)
        const uint32_t f = &o == &in.src[0] ? in.cbsz : in.blgp;
        const F8& t = f == 0 ? kOcpFp8 : f == 1 ? kOcpBf8 : f == 2 ? kFp6 : f == 3 ? kBf6 : kFp4;
        return f8_to_float(packed(o, static_cast<uint32_t>(t.bits), e, lane), t);
      }
      if (type == 'b') return static_cast<double>(as_float((reg(o, e / 2, lane) >> (16 * (e % 2))) << 16));
      if (type == 'c') return static_cast<double>(static_cast<int8_t>(reg(o, e / 4, lane) >> (8 * (e % 4))));
      if (type == 'e' || type == 'g') return f8_to_float(reg(o, e / 4, lane) >> (8 * (e % 4)), type == 'e' ? fp8() : bf8());
      if (type == 'i') return static_cast<double>(static_cast<int32_t>(reg(o, e, lane)));
      if (type == 'h') {
        _Float16 h;
        const uint16_t bits = static_cast<uint16_t>(reg(o, e / 2, lane) >> (16 * (e % 2)));
        std::memcpy(&h, &bits, 2);
        return static_cast<double>(h);
      }
      if (type == 'f') return static_cast<double>(as_float(reg(o, e, lane)));
      if (type == 'F') {
        // XF32: "32-bit floats but the mantissa truncated to 10 bits (not including the leading 1)"
        // (the MI300 ISA guide, V_MFMA_F32_16X16X8_XF32), denormals kept. Infinities and NaNs stay.
        uint32_t bits = reg(o, e, lane);
        if ((bits & 0x7F800000u) != 0x7F800000u) bits &= ~0x1FFFu;
        return static_cast<double>(as_float(bits));
      }
      if (o.kind == OperandKind::InlineFloat) return o.fvalue;
      return as_double(reg(o, 2 * e, lane) | static_cast<uint64_t>(reg(o, 2 * e + 1, lane)) << 32);
    };
    const uint32_t lanes_per_block = kLanes / s.blocks, groups = lanes_per_block / s.m, per_lane = s.k / groups;
    const uint32_t outs = s.m * s.n * s.blocks / kLanes;
    std::vector<double> a(size_t{s.blocks} * s.m * s.k), b(size_t{s.blocks} * s.k * s.n), c(size_t{s.blocks} * s.m * s.n);
    const auto A = [&](uint32_t bl, uint32_t i, uint32_t k) -> double& { return a[(size_t{bl} * s.m + i) * s.k + k]; };
    const auto B = [&](uint32_t bl, uint32_t k, uint32_t j) -> double& { return b[(size_t{bl} * s.k + k) * s.n + j]; };
    const auto C = [&](uint32_t bl, uint32_t i, uint32_t j) -> double& { return c[(size_t{bl} * s.m + i) * s.n + j]; };
    // gfx950's scaled forms: each lane's A and B values are its one row (or
    // column) and one block of 32 along K, and are scaled by 2^(e - 127),
    // e the E8M0 byte of the lane's scale register its selector names -- or
    // an inline float constant's exponent. 0xFF is the format's NaN.
    const auto scale_of = [&](const Operand& o, uint32_t sel, uint32_t lane) -> double {
      const uint32_t e = o.kind == OperandKind::InlineFloat
                             ? (as_bits(static_cast<float>(o.fvalue)) >> 23) & 0xFF
                             : (lane_src(w, o, lane) >> (8 * sel)) & 0xFF;
      return e == 0xFF ? std::numeric_limits<double>::quiet_NaN() : std::ldexp(1.0, static_cast<int>(e) - 127);
    };
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      const uint32_t bl = lane / lanes_per_block, within = lane % lanes_per_block, g = within / s.m;
      // With CBSZ, a block takes its A from block ABID of its group of 2^CBSZ:
      // the same place in that block's lanes.
      const uint32_t a_block = in.a_bcast_size ? (bl >> in.a_bcast_size << in.a_bcast_size) + in.a_bcast_id : bl;
      const uint32_t a_lane = a_block < s.blocks ? a_block * lanes_per_block + within : lane;
      const double sa = in.scaled ? scale_of(in.src[3], in.scale_sel & 3, lane) : 1.0;
      const double sb = in.scaled ? scale_of(in.src[4], in.scale_sel >> 2, lane) : 1.0;
      for (uint32_t e = 0; e < per_lane; ++e) {
        if (!sparse) A(bl, within % s.m, g * per_lane + e) = value(in.src[0], s.in, e, a_lane) * sa;
        // gfx950's sparse forms, whose B is eight registers, split it: the
        // first four hold the lane group's run of K in the first half, the
        // last four the same run K/2 on (the CDNA4 guide's layout tables),
        // where A's run is K's in one piece -- as hipSPARSELt lays both out.
        const uint32_t half = per_lane / 2;
        const uint32_t kb = sparse && in.src[1].width == 8 ? e / half * (s.k / 2) + g * half + e % half : g * per_lane + e;
        B(bl, kb, within % s.n) = value(in.src[1], s.in_b ? s.in_b : s.in, e, lane) * sb;
      }
      if (sparse) {
        // A lane's A holds two values of each four along K, packed; the
        // index register says, two bits a value, which of the four each is.
        // A lane's run of K has one bit of indices a value, so a register
        // holds 32 / per_lane sets of them: CBSZ zero lets ABID pick one.
        const uint32_t sets = 32 / per_lane, set = (in.cbsz == 0 ? in.abid : 0u) % sets;
        const uint32_t idx = reg(in.src[2], 0, lane) >> (set * per_lane);
        for (uint32_t v = 0; v < per_lane / 2; ++v)
          A(bl, within % s.m, g * per_lane + 4 * (v / 2) + ((idx >> (2 * v)) & 3)) = value(in.src[0], s.in, v, lane);
      }
      for (uint32_t r = 0; r < outs; ++r) {
        uint32_t ob = 0, row = 0;
        out_place(s, lane, r, &ob, &row);
        C(ob, row, lane % s.n) = value(sparse ? in.dst[0] : in.src[2], s.out, r, lane);
      }
    }
    for (uint32_t lane = 0; lane < kLanes; ++lane)
      for (uint32_t r = 0; r < outs; ++r) {
        uint32_t ob = 0, i = 0;
        out_place(s, lane, r, &ob, &i);
        const uint32_t j = lane % s.n;
        if (s.out == 'i') {
          // Integers are summed exactly, and wrap as the hardware's do.
          int64_t isum = static_cast<int64_t>(C(ob, i, j));
          for (uint32_t k = 0; k < s.k; ++k) isum += static_cast<int64_t>(A(ob, i, k)) * static_cast<int64_t>(B(ob, k, j));
          set_word(w, in.dst[0], r, lane, static_cast<uint32_t>(isum));
          continue;
        }
        double sum = C(ob, i, j);
        for (uint32_t k = 0; k < s.k; ++k) {
          if (s.out == 'd') sum = std::fma(A(ob, i, k), B(ob, k, j), sum);
          else sum += A(ob, i, k) * B(ob, k, j);
        }
        if (s.out == 'd') {
          const uint64_t bits = as_bits(sum);
          set_word(w, in.dst[0], 2 * r, lane, static_cast<uint32_t>(bits));
          set_word(w, in.dst[0], 2 * r + 1, lane, static_cast<uint32_t>(bits >> 32));
        } else {
          set_word(w, in.dst[0], r, lane, as_bits(static_cast<float>(sum)));
        }
      }
  }


  // What s_set_gpr_idx_on does to an instruction: each vector register
  // operand it enabled is moved on by M0's low byte.
  static void index_gprs(const Wave& w, Inst* in) {
    const uint32_t by = w.m0 & 0xFF;
    for (uint32_t k = 0; k < in->src.size() && k < 3; ++k)
      if ((w.gpr_idx >> k) & 1 && in->src[k].kind == OperandKind::Vgpr) in->src[k].index += by;
    if ((w.gpr_idx & 8) && !in->dst.empty() && in->dst[0].kind == OperandKind::Vgpr) in->dst[0].index += by;
  }

  // Runs one instruction. Returns false when the wave has stopped or parked
  // at a barrier, so the group can run another wave.
  // VGPU_TRACE_WAVE=1: each instruction a work-group's first wave runs, and
  // what lane 0 (VGPU_TRACE_LANE=n: lane n) of its destination holds after,
  // on stderr -- the view a debugger's single-step gives of one wave.
  struct Traced {
    const Machine& m;
    const Wave& w;
    const Inst& in;
    uint64_t pc;
    ~Traced() {
      static const uint32_t lane = [] {
        const char* t = std::getenv("VGPU_TRACE_LANE");
        return t && *t ? static_cast<uint32_t>(std::atoi(t)) & 63 : 0u;
      }();
      std::string after;
      char b[48];
      const auto show = [&](const Inst& x) {
        for (const Operand& o : x.dst) {
          if (o.kind == OperandKind::Vgpr) {
            for (uint32_t k = 0; k < o.width && k < 4; ++k) {
              std::snprintf(b, sizeof b, " v%u=%08x", o.index + k, w.vgpr[o.index + k][lane]);
              after += b;
            }
          } else if (o.kind == OperandKind::Sgpr) {
            for (uint32_t k = 0; k < o.width && k < 16; ++k) {
              std::snprintf(b, sizeof b, " s%u=%08x", o.index + k, w.sgpr[o.index + k]);
              after += b;
            }
          }
        }
      };
      show(in);
      for (const Inst& h : in.dual) show(h);
      std::snprintf(b, sizeof b, " exec=%llx", static_cast<unsigned long long>(w.exec));
      std::fprintf(stderr, "%6llx  %-56s%s%s\n", static_cast<unsigned long long>(pc - m.d.code_base),
                   gcn::to_text(in).c_str(), after.c_str(), b);
    }
  };
  static bool tracing() {
    static const bool on = [] {
      const char* t = std::getenv("VGPU_TRACE_WAVE");
      return t && *t && *t != '0';
    }();
    return on;
  }

  // Where the debugger (VGPU_DEBUG) stops a wave, what it sees of it.
  void debug_stop(Wave& w, Group& g) {
    debug::WaveView v;
    v.kernel = d.kernel ? d.kernel->name : std::string("?");
    v.pc = w.pc;
    v.offset = w.pc - d.code_base - (d.kernel ? d.kernel->entry : 0);
    std::copy(std::begin(g.id), std::end(g.id), v.group);
    v.wave = w.first_lane / w.lanes;
    v.lanes = w.lanes;
    v.exec = &w.exec;
    v.vcc = &w.vcc;
    v.scc = &w.scc;
    v.m0 = &w.m0;
    v.mode = &w.mode;
    v.sgpr = w.sgpr;
    v.sgprs = kSgprs;
    v.vgpr = w.vgpr;
    v.vgprs = kVgprs;
    v.lds = &g.lds;
    v.memory = &mem;
    v.disassemble = [this](uint64_t pc, uint32_t* size) -> std::string {
      try {
        const Inst& in = fetch(pc);
        *size = in.size;
        return gcn::to_text(in);
      } catch (const std::exception& e) {
        *size = 0;
        return std::string("(") + e.what() + ")";
      }
    };
    debug::stop(v, &w);
  }

  // The global wave sync (GWS) barrier, which gfx10's cooperative groups
  // sync a grid with: the kernel names the resource by M0's bits 21:16 plus
  // the instruction's offset, and passes (in its first active lane) how many
  // waves, less one, the barrier waits for; ROCm's runtime initializes the
  // resource before a cooperative launch. A wave arrives once, then gives way
  // to the others (as s_sleep does) and tries again on its next turn, until
  // that many have arrived; then all of them go on. A cooperative launch runs
  // its work-groups on one machine, which keeps the counts.
  struct GwsBarrier {
    uint64_t arrived = 0, generation = 0;
  };
  std::map<uint32_t, GwsBarrier> gws;
  bool gws_barrier(Wave& w, const Inst& in) {
    const OpName op(in.name);
    if (op == "ds_gws_init"_op) return true;   // the count comes with each barrier
    if (op != "ds_gws_barrier"_op)
      throw Error::make(Err::Unsupported, in.name, ": GWS semaphores are not modelled");
    if (!d.cooperative)
      throw Error::make(Err::Unsupported, "a GWS barrier outside a cooperative launch, whose work-groups are not all "
                                          "resident at once");
    if (!w.exec) return true;
    GwsBarrier& b = gws[((w.m0 >> 16) & 0x3F) + static_cast<uint32_t>(in.offset)];
    if (!w.gws_waiting) {
      const uint32_t lane = static_cast<uint32_t>(__builtin_ctzll(w.exec));
      const uint64_t waves = uint64_t{static_cast<uint32_t>(lane_src(w, in.src[0], lane))} + 1;
      if (++b.arrived >= waves) {
        b.arrived = 0;
        ++b.generation;
        return true;
      }
      w.gws_waiting = true;
      w.gws_generation = b.generation;
    } else if (b.generation != w.gws_generation) {
      w.gws_waiting = false;
      return true;
    }
    w.pc -= in.size;   // not yet: this instruction again on the wave's next turn
    return false;
  }

  static void issued_load(Wave& w) {
    if (w.loads_in_flight < 63) ++w.loads_in_flight;
  }
  // The number of loads an s_waitcnt lets stay outstanding: vmcnt, in the bits the target puts it
  // (gfx9 and gfx10: 3:0 and 15:14; gfx11: 15:10).
  static uint32_t vm_count_of_waitcnt(const Inst& in) {
    const uint32_t imm = in.simm;
    if (in.arch == gcn::Target::Gfx1100) return (imm >> 10) & 0x3F;
    return (imm & 0xF) | ((imm >> 14) & 3) << 4;
  }
  // A wait for loads: all but `allowed` of them must have arrived. A wave of a group with others to run
  // that has loads to wait for gives them a turn first (Wave::soft_yield; run_round).
  static bool waited_for_loads(Wave& w, const Group& g, const Inst&, uint32_t allowed) {
    if (w.loads_in_flight <= allowed) return true;
    w.loads_in_flight = allowed;
    if (g.waves.size() < 2) return true;
    w.soft_yield = true;
    return false;
  }

  bool step(Wave& w, Group& g) {
    if (debug::active() &&
        debug::should_stop(d.kernel ? d.kernel->name : std::string(), w.pc - d.code_base - (d.kernel ? d.kernel->entry : 0), &w))
      debug_stop(w, g);
    const Inst& in = fetch(w.pc);
    // The performance counters' instruction mix, worked out when decoded.
    {
      InstructionCounts& c = stats.counts;
      if (in.dual.empty()) {
        ++c.mix[static_cast<int>(in.mix)];
        c.mops[static_cast<int>(in.mops_type)] += in.mops;
      } else {
        for (const Inst& half : in.dual) ++c.mix[static_cast<int>(half.mix)];
      }
    }
    std::optional<Traced> traced;
    if (tracing() && w.first_lane == 0) traced.emplace(*this, w, in, w.pc);
    w.pc += in.size;
    ++stats.instructions;
    InstructionCounts& n = stats.counts;
    switch (in.enc) {
      case gcn::Enc::Sop1:
      case gcn::Enc::Sop2:
      case gcn::Enc::Sopk:
        ++n.salu;
        scalar_alu(w, in);
        return true;
      case gcn::Enc::Sopc:
        ++n.salu;
        scalar_compare(w, in);
        return true;
      case gcn::Enc::Smem:
        ++n.smem;
        scalar_load(w, in);
        return true;
      case gcn::Enc::Vop1:
      case gcn::Enc::Vop2:
      case gcn::Enc::Vop3:
      case gcn::Enc::Vop3p: {
        ++n.valu;
        // Under VGPR indexing, the operands it names read and write the
        // vector register M0's low byte further on.
        Inst indexed;
        const Inst& x = w.gpr_idx ? (indexed = in, index_gprs(w, &indexed), indexed) : in;
        // A comparison in its long form is still a comparison: it writes a
        // mask of the lanes that passed, not a value per lane.
        if (x.name.rfind("v_mfma", 0) == 0 || x.name.rfind("v_smfmac", 0) == 0) {
          // The matrix instructions ignore MODE: round to nearest even,
          // denormals kept, always.
          ++n.mfma;
          matrix_multiply(w, x);
          return true;
        }
        std::optional<FloatMode> fm;
        if (!FloatMode::is_default(w.mode)) fm.emplace(w.mode, x.name.find("f64") != std::string::npos);
        if (x.name.rfind("v_cmp", 0) == 0) compare(w, x);
        else if (x.dpp) cross_lane_alu(w, x);
        else vector_alu(w, x);
        return true;
      }
      case gcn::Enc::Vopd: {
        // RDNA's two instructions issued as one: both read their sources
        // before either writes. The compiler may pair a Y that reads X's
        // destination (v_dual_add_f32 v2, ... :: v_dual_lshlrev_b32 v3, 2, v2
        // shifts the old v2), so Y runs with X's destination as it was, and
        // X's result is put back after. (X cannot read Y's destination as
        // changed: it runs first. The two destinations always differ.)
        ++n.valu;
        std::optional<FloatMode> fm;
        if (!FloatMode::is_default(w.mode)) fm.emplace(w.mode, false);
        const Inst& x = in.dual[0];
        const Inst& y = in.dual[1];
        const uint32_t xd = x.dst.at(0).index, xw = std::max(1u, x.dst[0].width);   // (a 64-bit result is two registers)
        std::array<std::array<uint32_t, kLanes>, 2> before = {}, after = {};
        for (uint32_t k = 0; k < xw && k < 2; ++k)
          std::copy(std::begin(w.vgpr[xd + k]), std::end(w.vgpr[xd + k]), before[k].begin());
        vector_alu(w, x);
        bool y_reads_x = false;
        for (const Operand& o : y.src)
          y_reads_x = y_reads_x || (o.kind == OperandKind::Vgpr && o.index < xd + xw && xd < o.index + std::max(1u, o.width));
        if (!y_reads_x) {
          vector_alu(w, y);
          return true;
        }
        for (uint32_t k = 0; k < xw && k < 2; ++k) {
          std::copy(std::begin(w.vgpr[xd + k]), std::end(w.vgpr[xd + k]), after[k].begin());
          std::copy(before[k].begin(), before[k].end(), std::begin(w.vgpr[xd + k]));
        }
        vector_alu(w, y);
        for (uint32_t k = 0; k < xw && k < 2; ++k) std::copy(after[k].begin(), after[k].end(), std::begin(w.vgpr[xd + k]));
        return true;
      }
      case gcn::Enc::Vopc: {
        ++n.valu;
        std::optional<FloatMode> fm;
        if (!FloatMode::is_default(w.mode)) fm.emplace(w.mode, in.name.find("f64") != std::string::npos);
        compare(w, in);
        return true;
      }
      case gcn::Enc::Ds:
        ++n.lds;
        if (in.name.rfind("ds_gws_", 0) == 0) return gws_barrier(w, in);
        lds_access(w, in, g);
        return true;
      case gcn::Enc::Flat:
        // RDNA4's cache write-back and invalidate (global_wb, global_inv,
        // global_wbinv), which its fences compile to: every access here
        // reaches memory directly, so they order the host's threads and no more.
        if (OpName(in.name) == "global_inv"_op || OpName(in.name) == "global_wb"_op ||
            OpName(in.name) == "global_wbinv"_op) {
          std::atomic_thread_fence(std::memory_order_seq_cst);
          return true;
        }
        ++n.vmem;
        ++n.flat;
        if (in.name.find("_atomic") != std::string::npos) ++n.flat_atomic, ++n.vmem_wr;
        else if (in.name.find("_store") != std::string::npos) ++n.flat_write, ++n.vmem_wr;
        else ++n.flat_read, ++n.vmem_rd, issued_load(w);
        if (in.segment == Inst::Segment::Scratch) scratch_access(w, in, g);
        else if (in.segment == Inst::Segment::Flat) n.lds += flat_access(w, in, g);
        else if (in.name.find("_load_lds_") != std::string::npos) global_load_lds(w, in, g), ++n.lds;
        else global_access(w, in);
        return true;
      case gcn::Enc::Mubuf:
        ++n.vmem;
        // A write-back or an invalidate of the caches, which a fence compiles
        // to. Every access here reaches memory directly, so there is nothing
        // to write back and nothing stale to drop -- but work-groups on other
        // host threads see this thread's writes in the order a fence
        // promises only if the host is told to keep it.
        if (OpName(in.name) == "buffer_wbl2"_op || OpName(in.name) == "buffer_inv"_op ||
            OpName(in.name) == "buffer_wbinvl1"_op || OpName(in.name) == "buffer_wbinvl1_vol"_op ||
            OpName(in.name) == "buffer_invl2"_op || OpName(in.name) == "buffer_gl0_inv"_op ||
            OpName(in.name) == "buffer_gl1_inv"_op) {
          std::atomic_thread_fence(std::memory_order_seq_cst);
          return true;
        }
        if (in.name.find("_store") != std::string::npos || in.name.find("_atomic") != std::string::npos) ++n.vmem_wr;
        else ++n.vmem_rd, issued_load(w);
        buffer_access(w, in, g);
        return true;
      case gcn::Enc::Mtbuf:
        ++n.vmem;
        if (in.name.find("_store") != std::string::npos) ++n.vmem_wr;
        else ++n.vmem_rd, issued_load(w);
        buffer_access(w, in, g);
        return true;
      case gcn::Enc::Mimg:
        if (in.name.rfind("tensor_", 0) == 0)
          throw Error::make(Err::Unsupported, in.name, " (gfx1250's tensor data mover) is decoded but not modeled");
        ++n.vmem;
        if (in.name.find("_store") != std::string::npos || in.name.find("_atomic") != std::string::npos) ++n.vmem_wr;
        else ++n.vmem_rd, issued_load(w);
        image_access(w, in);
        return true;
      case gcn::Enc::Sopp: break;
      default:
        throw Error::make(Err::Unsupported, gcn::enc_name(in.enc), " is decoded but not implemented");
    }
    // The program-flow instructions.
    if (OpName(in.name) == "s_branch"_op || in.name.rfind("s_cbranch_", 0) == 0) ++n.branch;
    if (OpName(in.name) == "s_sendmsg"_op) ++n.sendmsg;
    if (OpName(in.name) == "s_endpgm"_op) {
      w.done = true;
      return false;
    }
    if (OpName(in.name) == "s_waitcnt"_op) return waited_for_loads(w, g, in, vm_count_of_waitcnt(in));
    if (OpName(in.name) == "s_nop"_op) return true;   // nothing is out of order here
    // RDNA's scheduling hints -- a clause of memory instructions, the delay
    // an ALU result needs, the dependency and prefetch controls -- change
    // nothing where every instruction completes before the next begins.
    if (OpName(in.name) == "s_clause"_op || OpName(in.name) == "s_delay_alu"_op ||
        OpName(in.name) == "s_waitcnt_depctr"_op || OpName(in.name) == "s_set_inst_prefetch_distance"_op)
      return true;
    // RDNA4's waits, a counter each (loads, stores, LDS, scalar memory, ...).
    if (OpName(in.name) == "s_wait_loadcnt"_op) return waited_for_loads(w, g, in, in.simm & 0x3F);
    if (in.name.rfind("s_wait_", 0) == 0 && in.name != "s_wait_event") return true;
    if (OpName(in.name) == "s_barrier_wait"_op) {
      w.at_barrier = true;
      ++stats.barriers;
      return false;
    }
    // Giving the wave's vector registers back as it ends (RDNA's
    // MSG_DEALLOC_VGPRS), before its s_endpgm.
    if (OpName(in.name) == "s_sendmsg"_op && gcn::is_rdna(in.arch) && in.simm == 3) return true;
    if (OpName(in.name) == "s_setprio"_op || OpName(in.name) == "s_setprio_inc_wg"_op) return true;   // waves are not scheduled by priority here
    // gfx1250 addresses up to 1024 vector registers by setting the top bits of every VGPR number; a program that
    // sets any is not one this runs yet.
    if (OpName(in.name) == "s_set_vgpr_msb"_op) {
      if (in.simm != 0)
        throw Error::make(Err::Unsupported, "s_set_vgpr_msb ", in.simm, ", which addresses vector registers past v255 and is not modeled");
      return true;
    }
    if (OpName(in.name) == "s_monitor_sleep"_op) return false;
    // No wave sleeps past its own s_sleep to be woken, and there is no
    // instruction cache to invalidate.
    if (OpName(in.name) == "s_wakeup"_op || OpName(in.name) == "s_icache_inv"_op) return true;
    if (OpName(in.name) == "s_set_gpr_idx_off"_op) {
      w.gpr_idx = 0;
      return true;
    }
    if (OpName(in.name) == "s_trap"_op) {
      // A real card's trap handler reports it to the queue and the runtime
      // aborts the launch; the launch fails here, and says why.
      throw Error::make(Err::Trap, "the kernel executed s_trap ", in.simm,
                        in.simm == 2 ? " (llvm.trap: __builtin_trap or abort() in device code)" : "");
    }
    if (OpName(in.name) == "s_barrier"_op) {
      w.at_barrier = true;
      ++stats.barriers;
      return false;
    }
    if (OpName(in.name) == "s_branch"_op) {
      w.pc = in.target;
      return true;
    }
    if (OpName(in.name) == "s_cbranch_execz"_op) {
      if (!w.exec) w.pc = in.target;
      return true;
    }
    if (OpName(in.name) == "s_cbranch_execnz"_op) {
      if (w.exec) w.pc = in.target;
      return true;
    }
    // A wave32 wave's VCC is its low half: the high half is an ordinary
    // register the compiler may keep anything in.
    const uint64_t vcc = w.lanes == 32 ? static_cast<uint32_t>(w.vcc) : w.vcc;
    if (OpName(in.name) == "s_cbranch_vccz"_op) {
      if (!vcc) w.pc = in.target;
      return true;
    }
    if (OpName(in.name) == "s_cbranch_vccnz"_op) {
      if (vcc) w.pc = in.target;
      return true;
    }
    // A wave waiting for something -- another work-group at a grid barrier,
    // the host answering a call -- sleeps; here it gives way to the others,
    // which is what lets a wave that spins on memory see the write it waits for.
    if (OpName(in.name) == "s_sleep"_op) return false;
    if (OpName(in.name) == "s_sendmsg"_op) {
      // The kernel raised the doorbell of its hostcall buffer and asks the
      // host for attention (MSG_INTERRUPT, the one message decoded): the
      // host answers now, and the wave, spinning on its packet, carries on.
      if (!d.hostcall)
        throw Error::make(Err::Unsupported, "the kernel asked the host for attention (s_sendmsg MSG_INTERRUPT), which ",
                          "is how device-side printf reaches the host, and the launch was given no hostcall buffer");
      d.hostcall->service();
      return true;
    }
    if (OpName(in.name) == "s_cbranch_scc0"_op) {
      if (!w.scc) w.pc = in.target;
      return true;
    }
    if (OpName(in.name) == "s_cbranch_scc1"_op) {
      if (w.scc) w.pc = in.target;
      return true;
    }
    throw Error::make(Err::Unsupported, "instruction ", in.name, " is decoded but not implemented");
  }
};

}  // namespace

namespace {

// The grid's size in work-items in dimension i: what the dispatch says, or
// its whole work-groups.
uint64_t grid_items(const Dispatch& d, int i) {
  return d.grid_items[i] ? d.grid_items[i] : uint64_t{d.groups[i]} * d.group_size[i];
}
// How many work-items work-group g has in dimension i: the full size, or in a
// grid that is not a whole number of groups, what is left for the last one.
uint32_t items_in_group(const Dispatch& d, int i, uint32_t g) {
  const uint64_t start = uint64_t{g} * d.group_size[i];
  return static_cast<uint32_t>(std::min<uint64_t>(d.group_size[i], grid_items(d, i) - start));
}

// What the runtime tells a kernel about the grid it is part of, written into
// the kernarg segment after the kernel's own arguments. The names and offsets
// are the code object's own (its metadata lists them); the values are this
// dispatch's.
void fill_hidden_arguments(const Dispatch& d, const Kernel& k, MemoryManager& mem) {
  const uint64_t threads_x = grid_items(d, 0), threads_y = grid_items(d, 1), threads_z = grid_items(d, 2);
  const uint32_t dims = d.groups[2] > 1 || d.group_size[2] > 1   ? 3
                        : d.groups[1] > 1 || d.group_size[1] > 1 ? 2
                                                                 : 1;
  for (const KernelArg& a : k.args) {
    if (!a.hidden()) continue;
    uint64_t value = 0;
    const std::string& kind = a.kind;
    if (kind == "hidden_block_count_x") value = d.groups[0];
    else if (kind == "hidden_block_count_y") value = d.groups[1];
    else if (kind == "hidden_block_count_z") value = d.groups[2];
    else if (kind == "hidden_group_size_x") value = d.group_size[0];
    else if (kind == "hidden_group_size_y") value = d.group_size[1];
    else if (kind == "hidden_group_size_z") value = d.group_size[2];
    else if (kind == "hidden_grid_size_x") value = threads_x;
    else if (kind == "hidden_grid_size_y") value = threads_y;
    else if (kind == "hidden_grid_size_z") value = threads_z;
    else if (kind == "hidden_grid_dims") value = dims;
    else if (kind == "hidden_shared_base") value = kSharedBase;
    else if (kind == "hidden_private_base") value = kPrivateBase;
    else if (kind == "hidden_dynamic_lds_size") value = d.dynamic_lds;
    else if (kind == "hidden_hostcall_buffer") value = d.hostcall ? d.hostcall->buffer() : 0;
    // The size of the last work-group in each dimension, where the grid is
    // not a whole number of them; zero where it is.
    else if (kind == "hidden_remainder_x") value = threads_x % d.group_size[0];
    else if (kind == "hidden_remainder_y") value = threads_y % d.group_size[1];
    else if (kind == "hidden_remainder_z") value = threads_z % d.group_size[2];
    // What a grid barrier counts on, in a cooperative launch; zero otherwise,
    // which is how a kernel tells that its grid cannot synchronize.
    else if (kind == "hidden_multigrid_sync_arg") value = d.grid_sync;
    // Everything else -- the global offsets, a heap for device malloc -- is
    // zero, and a kernel that needs one of those will say so by failing on a
    // null pointer rather than reading something made up.
    else continue;
    if (a.offset + a.size > k.kernarg_size) continue;
    for (uint32_t b = 0; b < a.size && b < 8; ++b)
      mem.store_scalar(d.kernarg + a.offset + b, 1, (value >> (8 * b)) & 0xFF);
  }
}

// The packet the hardware is given for a dispatch, which a kernel may read
// instead of its implicit arguments (the HSA kernel dispatch packet: its
// sizes, its segments, and where its arguments are).
uint64_t write_dispatch_packet(const Dispatch& d, const Kernel& k, MemoryManager& mem) {
  const uint32_t dims = d.groups[2] > 1 || d.group_size[2] > 1   ? 3
                        : d.groups[1] > 1 || d.group_size[1] > 1 ? 2
                                                                 : 1;
  std::vector<uint8_t> packet(64, 0);
  const auto put16 = [&](uint32_t at, uint16_t v) { std::memcpy(&packet[at], &v, 2); };
  const auto put32 = [&](uint32_t at, uint32_t v) { std::memcpy(&packet[at], &v, 4); };
  const auto put64 = [&](uint32_t at, uint64_t v) { std::memcpy(&packet[at], &v, 8); };
  put16(0, 2 << 0);                       // header: a kernel dispatch packet
  put16(2, static_cast<uint16_t>(dims));  // setup: how many dimensions the grid has
  put16(4, static_cast<uint16_t>(d.group_size[0]));
  put16(6, static_cast<uint16_t>(d.group_size[1]));
  put16(8, static_cast<uint16_t>(d.group_size[2]));
  put32(12, static_cast<uint32_t>(grid_items(d, 0)));
  put32(16, static_cast<uint32_t>(grid_items(d, 1)));
  put32(20, static_cast<uint32_t>(grid_items(d, 2)));
  put32(24, k.private_segment);
  put32(28, k.group_segment);
  put64(32, d.code_base + k.entry);
  put64(40, d.kernarg);
  const uint64_t where = mem.alloc(packet.size());
  mem.write(where, packet.data(), packet.size());
  return where;
}

// How many host threads a dispatch's work-groups are spread over: every
// core, or VGPU_THREADS. One runs them in order, one after another, which is
// what a kernel with a data race needs to come out the same every time; more
// is what the hardware already allows, since work-groups run in any order and
// at once.
unsigned worker_count(uint64_t groups) {
  unsigned want = 0;
  // Under the debugger, one thread: a wave stopped at a breakpoint stops the
  // dispatch, and everything runs in the same order every time.
  if (debug::active()) return 1;
  if (const char* t = std::getenv("VGPU_THREADS")) {
    const int v = std::atoi(t);
    want = v > 0 ? static_cast<unsigned>(v) : 1;
  } else {
    want = vgpu::host_cpus();  // the CPUs this process may use, quota included
  }
  if (want == 0) want = 1;
  if (want > groups) want = static_cast<unsigned>(groups);
  return want ? want : 1;
}

// A work-group's waves, set up as the hardware leaves them.
void set_up_group(Group& group, Machine& m, const Dispatch& d, uint64_t packet, uint64_t group_segment, uint32_t gx,
                  uint32_t gy, uint32_t gz) {
  const Kernel& k = *d.kernel;
  // The work-group's own shape: the dispatch's, or less in the last group of
  // a grid that is not a whole number of them. Its work-items are numbered
  // across that shape, as the hardware numbers a partial group's.
  const uint32_t size[3] = {items_in_group(d, 0, gx), items_in_group(d, 1, gy), items_in_group(d, 2, gz)};
  const uint64_t threads = uint64_t{size[0]} * size[1] * size[2];
  const uint32_t lanes = k.wave32 ? 32 : kLanes;
  const uint32_t waves_per_group = static_cast<uint32_t>((threads + lanes - 1) / lanes);
  // A card gives a work-group LDS in 512-byte granules (128 dwords, the
  // unit the descriptor counts it in), so a kernel reading a little past
  // what it asked for still reads its own LDS: rocFFT's kernels do.
  group.lds.assign(std::min<uint64_t>((group_segment + 511) / 512 * 512, lds_limit(d)), 0);
  // Each work-item's private memory. A kernel that spills says how much
  // it needs; the rest get none. The hardware gives it to every lane of a
  // wave, work-item or not: a function saving whole-wave registers turns
  // on every lane, past the group's last work-item too.
  group.scratch_per_lane = (k.private_segment + 3) & ~3u;
  group.scratch.assign(static_cast<size_t>(group.scratch_per_lane) * waves_per_group * lanes, 0);
  group.waves.resize(waves_per_group);
  group.id[0] = gx;
  group.id[1] = gy;
  group.id[2] = gz;
  for (uint32_t i = 0; i < waves_per_group; ++i) {
    Wave& w = group.waves[i];
    w.pc = d.code_base + k.entry;
    w.mode = k.mode;
    w.lanes = lanes;
    w.first_lane = i * lanes;
    w.group = gx + d.groups[0] * (gy + uint64_t{d.groups[1]} * gz);
    // The lanes this wave has of the work-group, which is short in the
    // last wave when the group is not a whole number of waves.
    const uint64_t left = threads - w.first_lane;
    w.exec = left >= lanes ? (lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1) : (uint64_t{1} << left) - 1;
    // What the hardware leaves in registers before the first
    // instruction: the user SGPRs the descriptor asked for, then the
    // work-group's id, and each lane's id in v0 (and v1, v2 where the
    // group has those dimensions).
    uint32_t at = 0;
    if (k.private_segment_buffer) {
      // gfx90a reaches scratch through a buffer resource the hardware sets
      // up, swizzled as a card's is: each work-item's dwords interleaved
      // across the wave's 64 lanes (ADD_TID, index stride 64, 4-byte
      // elements), from the wave's own offset (which is 0 here; each
      // work-item has a block of its own). buffer_access unswizzles it.
      m.set_sgpr(w, at, static_cast<uint32_t>(kScratchResourceBase));
      m.set_sgpr(w, at + 1, static_cast<uint32_t>(kScratchResourceBase >> 32) | 1u << 31);   // SWIZZLE_EN
      m.set_sgpr(w, at + 2, 0xFFFFFFFFu);                                                  // num_records
      m.set_sgpr(w, at + 3, 4u << 15 | 3u << 21 | 1u << 23);   // 32-bit data, index stride 64, ADD_TID
      at += 4;
    }
    if (k.dispatch_ptr) {
      m.set_sgpr64(w, at, packet);
      at += 2;
    }
    if (k.queue_ptr) at += 2;
    if (k.kernarg_segment_ptr) {
      m.set_sgpr64(w, at, d.kernarg);
      at += 2;
    }
    if (k.dispatch_id) at += 2;
    if (k.flat_scratch_init) at += 2;
    // Past any the descriptor reserves for preloaded arguments: a kernel
    // that preloads them loads them itself where the hardware has not (its
    // first 256 bytes do it, and the hardware skips them).
    at = std::max(at, k.user_sgpr_count);
    // RDNA4 gives the work-group's id in the trap handler's registers, and
    // the wave's number within the group in TTMP8's bits 25 to 29, where
    // LLVM reads it (llvm.amdgcn.wave.id). Without it every wave of a Triton
    // kernel took itself for the first, and only a group's first 32 work-items'
    // share of the work was done right.
    if (gcn::is_gfx12(m.target())) {
      w.ttmp[9] = gx;
      w.ttmp[7] = (gy & 0xFFFF) | gz << 16;
      w.ttmp[8] = (i & 0x1F) << 25;
    }
    if (k.group_id_x) m.set_sgpr(w, at++, gx);
    if (k.group_id_y) m.set_sgpr(w, at++, gy);
    if (k.group_id_z) m.set_sgpr(w, at++, gz);
    if (k.group_info) m.set_sgpr(w, at++, 0);
    if (k.private_wave_offset && m.target() == gcn::Target::Gfx90a) m.set_sgpr(w, at++, 0);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint64_t flat = w.first_lane + lane;
      const uint32_t x = static_cast<uint32_t>(flat % size[0]), y = static_cast<uint32_t>(flat / size[0] % size[1]),
                     z = static_cast<uint32_t>(flat / size[0] / size[1]);
      // From ABI version 5 a work-item's three ids are packed into v0,
      // ten bits each, and the kernel pulls them out; before it each id
      // had a register of its own -- except where the hardware packs them
      // whatever the ABI (gcn::packs_work_item_ids): a version 4 object
      // built for gfx90a, gfx942 or gfx1100 still reads them from v0.
      if (d.object->packed_work_item_id() || gcn::packs_work_item_ids(m.target())) {
        w.vgpr[0][lane] = x | y << 10 | z << 20;
      } else {
        w.vgpr[0][lane] = x;
        w.vgpr[1][lane] = y;
        w.vgpr[2][lane] = z;
      }
    }
    ++m.stats.waves;
    const uint64_t active = left >= lanes ? lanes : left;
    if (active == kLanes) ++m.stats.waves_eq64;
    else ++m.stats.waves_lt64;
    m.stats.waves_lt48 += active < 48;
    m.stats.waves_lt32 += active < 32;
    m.stats.waves_lt16 += active < 16;
  }

}

// One turn for each of a group's waves: each runs until it stops, parks at a
// barrier or yields (s_sleep, a wave waiting on something), and a barrier
// every unfinished wave has reached is released. Says whether every wave has
// stopped.
// A turn may be cut short after `slice` instructions (0: no limit), as the
// hardware interleaves the waves of groups resident together: a wave polling
// memory for another group's write, with no s_sleep in its loop, gives the
// other group its turn.
std::atomic<bool> g_abandoned{false};
[[noreturn]] void abandon() {
  throw Error::make(Err::DeviceLost, "the process is exiting, and the dispatch was abandoned");
}

bool run_round(Machine& m, Group& group, uint64_t slice = 0) {
  if (g_abandoned.load(std::memory_order_relaxed)) abandon();
  bool runnable = false;
  // A wave that waits for a global load gives the others its turn (Wave::soft_yield), as the load's
  // latency does on a card, and has the rest of the round after them; the turn counts across those
  // pauses, so a wave polling memory with loads in its loop still ends its round.
  for (Wave& w : group.waves) w.round_stopped = false, w.round_steps = 0;
  bool again;
  do {
    again = false;
    for (Wave& w : group.waves) {
      if (w.done || w.at_barrier || w.round_stopped) continue;
      runnable = true;
      uint64_t pc = w.pc;
      try {
        uint64_t n = w.round_steps;
        for (; m.step(w, group); ++n) {
          pc = w.pc;
          if (slice && n + 1 >= slice) break;
          if ((n & 4095) == 4095 && g_abandoned.load(std::memory_order_relaxed)) abandon();
        }
        w.round_steps = n;
      } catch (const Error& e) {
        throw m.at_instruction(e, pc);
      }
      if (w.soft_yield) {
        w.soft_yield = false;
        again = true;
      } else {
        w.round_stopped = true;
      }
    }
  } while (again);
  if (runnable) return false;
  // Every wave is stopped or waiting: release the barrier.
  bool any = false;
  for (Wave& w : group.waves)
    if (w.at_barrier) {
      w.at_barrier = false;
      any = true;
    }
  return !any;
}

// Whether a group's last round ended with a wave waiting (s_sleep, as a
// group polling a flag another group sets does) rather than at a barrier or
// done.
bool waiting(const Group& group) {
  for (const Wave& w : group.waves)
    if (!w.done && !w.at_barrier) return true;
  return false;
}

// Runs work-groups [begin, end) on one host thread. Each group runs until it
// finishes, waits (s_sleep) or has had its turn; the groups this thread holds take turns, and when every
// one of them is waiting, the next is started beside them -- as the hardware
// keeps many groups resident at once. hipBLASLt's Stream-K GEMMs have a group
// wait for another's partial tile: run one group at a time, one that waited on
// a group later on the same thread waited for ever. A kernel whose groups
// never wait runs one group at a time, as before.
template <typename GroupAt>
void run_groups(Machine& m, const Dispatch& d, uint64_t packet, uint64_t group_segment, uint64_t begin,
                uint64_t end, GroupAt group_at, const std::atomic<bool>* failed) {
  // A group's first turn is some four million instructions a wave: long
  // enough that a group that does not wait finishes in it, so the groups of
  // an ordinary kernel still run one at a time. A group still going after
  // that is waiting on another, or very long, and its later turns are short
  // (32768), so that a group polling another's flag gives way quickly. (A
  // short turn for every group held dozens of groups' registers at once and
  // ran PyTorch's checks many times slower; a long one for every turn made
  // vLLM's Stream-K GEMMs on MI350X six times slower.) The groups held at
  // once are bounded by their waves' registers (256 waves to a thread).
  constexpr uint64_t kFirstTurn = 1 << 22, kLaterTurn = 1 << 15;
  constexpr uint64_t kMaxResidentWaves = 256;
  struct Held {
    Group group;
    bool had_turn = false;
  };
  std::list<Held> resident;
  uint64_t next = begin, resident_waves = 0;
  const auto admit = [&] {
    uint32_t gx, gy, gz;
    group_at(next++, &gx, &gy, &gz);
    resident.emplace_back();
    set_up_group(resident.back().group, m, d, packet, group_segment, gx, gy, gz);
    resident_waves += resident.back().group.waves.size();
  };
  while (!resident.empty() || next < end) {
    if (failed && failed->load(std::memory_order_relaxed)) return;
    if (resident.empty()) admit();
    bool all_waiting = true;
    for (auto it = resident.begin(); it != resident.end();) {
      if (run_round(m, it->group, it->had_turn ? kLaterTurn : kFirstTurn)) {
        resident_waves -= it->group.waves.size();
        it = resident.erase(it);
        all_waiting = false;
        continue;
      }
      if (waiting(it->group)) it->had_turn = true;
      else all_waiting = false;
      ++it;
    }
    if (all_waiting && next < end && resident_waves < kMaxResidentWaves) admit();
  }
}

}  // namespace

std::mutex& memory_atomic_lock(uint64_t addr) {
  static std::array<std::mutex, 251> locks;
  return locks[(addr >> 2) % locks.size()];
}

void abandon_dispatches() { g_abandoned.store(true); }
bool dispatches_abandoned() { return g_abandoned.load(); }

DispatchStats execute(const Dispatch& d, MemoryManager& mem) {
  if (g_abandoned.load(std::memory_order_relaxed)) abandon();
  if (!d.object || !d.kernel) throw Error::make(Err::InvalidValue, "a dispatch needs a kernel");
  const Kernel& k = *d.kernel;
  // A wave is 64 lanes on CDNA. An RDNA kernel says in its descriptor
  // whether it was built for 32 or 64, and runs so whatever the device's
  // usual size.
  const bool rdna = gcn::is_rdna(gcn::target_of_mach(d.object->mach));
  if (!rdna && d.wave_size != kLanes)
    throw Error::make(Err::InvalidValue, "a CDNA wavefront is ", kLanes, " lanes, not ", d.wave_size);
  (void)rdna;
  const uint64_t threads = uint64_t{d.group_size[0]} * d.group_size[1] * d.group_size[2];
  if (!threads) throw Error::make(Err::InvalidValue, "a work-group has no work-items");
  for (int i = 0; i < 3; ++i)
    if (d.grid_items[i] && d.groups[i] != (d.grid_items[i] + d.group_size[i] - 1) / d.group_size[i])
      throw Error::make(Err::InvalidValue, "a grid of ", d.grid_items[i], " work-items is not ", d.groups[i],
                        " work-groups of ", d.group_size[i]);
  if (d.kernel_limits && k.max_flat_workgroup_size && threads > k.max_flat_workgroup_size)
    throw Error::make(Err::InvalidValue, "a work-group of ", threads, " work-items is past the ",
                      k.max_flat_workgroup_size, " this kernel allows");
  if (threads > 1024)
    throw Error::make(Err::InvalidValue, "a work-group of ", threads, " work-items is past the 1024 a CDNA work-group has");
  // What the work-group's LDS comes to: what the kernel reserved, and what
  // the launch added.
  const uint64_t group_segment = uint64_t{k.group_segment} + d.dynamic_lds;
  if (group_segment > lds_limit(d))
    throw Error::make(Err::InvalidValue, "a work-group asking for ", group_segment, " bytes of LDS is past the ",
                      lds_limit(d), " a work-group has on ", d.object->gfx950() ? "gfx950" : "gfx942");

  // What the kernel is told about its grid, and the packet it may read it
  // from. Both are written before any wave starts.
  if (d.kernarg && d.fill_hidden) fill_hidden_arguments(d, k, mem);
  uint64_t packet = 0;
  if (k.dispatch_ptr) packet = write_dispatch_packet(d, k, mem);

  const uint64_t groups = uint64_t{d.groups[0]} * d.groups[1] * d.groups[2];
  auto group_at = [&](uint64_t i, uint32_t* gx, uint32_t* gy, uint32_t* gz) {
    *gx = static_cast<uint32_t>(i % d.groups[0]);
    *gy = static_cast<uint32_t>(i / d.groups[0] % d.groups[1]);
    *gz = static_cast<uint32_t>(i / d.groups[0] / d.groups[1]);
  };
  std::unique_ptr<DecodeCache> own;
  DecodeCache* cache = d.decoded;
  if (!cache) cache = (own = std::make_unique<DecodeCache>(d.object->text.size())).get();
  DispatchStats total;
  // A kernel whose first instruction ends it does the same in every
  // work-group: each wave starts and stops, and nothing else happens. hip-tests
  // launches such empty kernels over the largest grids there are (2^31
  // work-groups and more). One group runs, and what it did is counted once
  // for each. Where the groups differ in shape (a partial last group), and
  // under the debugger or the wave trace, every group runs.
  bool same_shape = true;
  for (int i = 0; i < 3; ++i) same_shape &= !d.grid_items[i] || d.grid_items[i] % d.group_size[i] == 0;
  if (groups > 1 && same_shape && !debug::active() && !Machine::tracing()) {
    Machine m(d, mem, *cache);
    if (OpName(m.fetch(d.code_base + k.entry).name) == "s_endpgm"_op) {
      run_groups(m, d, packet, group_segment, 0, 1, group_at, nullptr);
      total = m.stats;
      total.scale(groups);
      if (packet) mem.free(packet);
      return total;
    }
  }
  const unsigned nthreads = d.cooperative ? 1 : worker_count(groups);
  if (d.cooperative) {
    // A cooperative launch's work-groups may wait on one another (a grid
    // barrier), so every one is resident at once, as the runtime promised when
    // it accepted the launch, and each takes a turn in order until all have
    // finished -- on one thread, so no group's turn ever waits on another's.
    Machine m(d, mem, *cache);
    std::vector<Group> all(groups);
    std::vector<bool> finished(groups, false);
    for (uint64_t i = 0; i < groups; ++i) {
      uint32_t gx, gy, gz;
      group_at(i, &gx, &gy, &gz);
      set_up_group(all[i], m, d, packet, group_segment, gx, gy, gz);
    }
    for (uint64_t left = groups; left;)
      for (uint64_t i = 0; i < groups; ++i)
        if (!finished[i] && run_round(m, all[i])) {
          finished[i] = true;
          --left;
        }
    total = m.stats;
  } else if (nthreads <= 1) {
    Machine m(d, mem, *cache);
    run_groups(m, d, packet, group_segment, 0, groups, group_at, nullptr);
    total = m.stats;
  } else {
    // Each thread a range of work-groups and a machine of its own; what they
    // share is device memory. The first failure stops the rest at their next
    // work-group, and is the one reported.
    std::vector<DispatchStats> per_thread(nthreads);
    std::vector<std::thread> threads_;
    std::mutex err_mu;
    std::exception_ptr first_error;
    std::atomic<bool> failed{false};
    for (unsigned t = 0; t < nthreads; ++t) {
      const uint64_t begin = groups * t / nthreads, end = groups * (t + 1) / nthreads;
      threads_.emplace_back([&, t, begin, end] {
        try {
          Machine m(d, mem, *cache);
          m.concurrent = true;
          run_groups(m, d, packet, group_segment, begin, end, group_at, &failed);
          per_thread[t] = m.stats;
        } catch (...) {
          failed = true;
          std::lock_guard<std::mutex> lock(err_mu);
          if (!first_error) first_error = std::current_exception();
        }
      });
    }
    for (std::thread& th : threads_) th.join();
    if (first_error) {
      if (packet) mem.free(packet);
      std::rethrow_exception(first_error);
    }
    for (const DispatchStats& p : per_thread) total.add(p);
  }
  if (packet) mem.free(packet);
  return total;
}

}  // namespace vgpu::amd
