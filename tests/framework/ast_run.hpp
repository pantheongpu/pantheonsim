// Build-and-run helpers for tests that make a kernel with ptx::KernelBuilder and compare its output with a host
// reference: a prologue that loads N buffer pointers and works out the thread's linear id, element addresses, and a
// run() that allocates the buffers, launches at a chosen warp width and reads them back.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/ptx/builder.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "ast_ops.hpp"

namespace astrun {

using namespace vgpu;
using namespace astops;

inline Type tu(uint32_t bits) { return Type{Type::Kind::U, bits}; }
inline Type ts(uint32_t bits) { return Type{Type::Kind::S, bits}; }
inline Type tf(uint32_t bits) { return Type{Type::Kind::F, bits}; }

// The buffer pointers (device addresses), this thread's linear id inside its block (x fastest) and its index in the
// whole grid (blocks laid out one after another, each with ntid.x * ntid.y * ntid.z threads).
struct Prologue {
  std::vector<Reg> ptr;
  Reg lin, gid;
};

inline Prologue prologue(ptx::KernelBuilder& kb, int nbuf) {
  Prologue p;
  for (int i = 0; i < nbuf; ++i) {
    const size_t pi = kb.add_param("buf" + std::to_string(i), tu(64));
    Reg r = kb.reg("%buf" + std::to_string(i), tu(64));
    kb.emit(ld(Space::Param, tu(64), r, kb.param_addr(pi)), "ld");
    p.ptr.push_back(r);
  }
  auto sreg = [&](const char* nm, Sreg s) {
    Reg r = kb.reg(nm, tu(32));
    kb.emit(mov(tu(32), r, SR(s)), "mov");
    return r;
  };
  const Reg tx = sreg("%tx", Sreg::TidX), ty = sreg("%ty", Sreg::TidY), tz = sreg("%tz", Sreg::TidZ);
  const Reg nx = sreg("%nx", Sreg::NtidX), ny = sreg("%ny", Sreg::NtidY), nz = sreg("%nz", Sreg::NtidZ);
  const Reg cx = sreg("%cx", Sreg::CtaidX);
  const Reg a = kb.temp(tu(32)), lin = kb.temp(tu(32)), tot = kb.temp(tu(32)), tot2 = kb.temp(tu(32)), gid = kb.temp(tu(32));
  kb.emit(madlo(tu(32), a, RO(tz), RO(ny), RO(ty)), "mad");
  kb.emit(madlo(tu(32), lin, RO(a), RO(nx), RO(tx)), "mad");
  kb.emit(ibin(IntBinOp::Mul, tu(32), tot, RO(nx), RO(ny)), "mul");
  kb.emit(ibin(IntBinOp::Mul, tu(32), tot2, RO(tot), RO(nz)), "mul");
  kb.emit(madlo(tu(32), gid, RO(cx), RO(tot2), RO(lin)), "mad");
  p.lin = lin;
  p.gid = gid;
  return p;
}

// The address of element `idx` of a buffer of `esz`-byte elements.
inline Addr elem(ptx::KernelBuilder& kb, const Reg& base, const Reg& idx, uint32_t esz) {
  const Reg off = kb.temp(tu(64)), at = kb.temp(tu(64));
  kb.emit(mulwide(false, off, RO(idx), IM(esz)), "mul");
  kb.emit(ibin(IntBinOp::Add, tu(64), at, RO(base), RO(off)), "add");
  return ptx::KernelBuilder::reg_addr(at);
}

// Loads element `idx` of `buf` as `ty` into a fresh register.
inline Reg load_elem(ptx::KernelBuilder& kb, const Reg& buf, const Reg& idx, Type ty) {
  const Reg v = kb.temp(ty);
  kb.emit(ld(Space::Global, ty, v, elem(kb, buf, idx, ty.bytes())), "ld");
  return v;
}
inline void store_elem(ptx::KernelBuilder& kb, const Reg& buf, const Reg& idx, Type ty, const Reg& v) {
  kb.emit(st(Space::Global, ty, elem(kb, buf, idx, ty.bytes()), RO(v)), "st");
}

// Launches `fn` (a kernel whose parameters are `bufs.size()` buffer pointers) and returns the buffers after the run.
inline std::vector<std::vector<uint8_t>> run(ptx::EntryFn fn, exec::LaunchConfig cfg, std::vector<std::vector<uint8_t>> bufs) {
  static runtime::Runtime rt(load_gpu("nvidia/h100"));   // one device for all the launches of a test
  auto& dev = rt.device(0);
  const std::string name = fn.name;
  ptx::ModuleBuilder mb;
  mb.add_entry(std::move(fn));
  const uint64_t mod = dev.load_built_module(mb.finish());
  const ptx::EntryFn* f = dev.get_function(mod, name);
  auto& m = dev.memory();
  std::vector<uint64_t> addr;
  std::vector<std::vector<uint8_t>> args;
  for (auto& b : bufs) {
    const uint64_t a = m.alloc(b.size() ? b.size() : 8);
    if (!b.empty()) m.write(a, b.data(), b.size());
    addr.push_back(a);
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &a, 8);
    args.push_back(arg);
  }
  dev.launch(*f, cfg, args, dev.symbols(mod));
  for (size_t i = 0; i < bufs.size(); ++i) {
    if (!bufs[i].empty()) m.read(addr[i], bufs[i].data(), bufs[i].size());
    m.free(addr[i]);
  }
  dev.unload_module(mod);
  return bufs;
}

template <class T>
std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  if (!b.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}
template <class T>
std::vector<T> from_bytes(const std::vector<uint8_t>& b) {
  std::vector<T> v(b.size() / sizeof(T));
  if (!v.empty()) std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
  return v;
}

// Deterministic pseudo-random bits (xorshift64*).
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
  uint64_t next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1Dull;
  }
};


// ---- host references for reductions ----------------------------------------------------------------------------------

// binary16, written here independently of the engine: by value.
inline double half_value(uint32_t h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  double v;
  if (e == 31) v = m ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
  else if (e == 0) v = std::ldexp(static_cast<double>(m), -24);
  else v = std::ldexp(static_cast<double>(m + 1024), e - 25);
  return (h & 0x8000) ? -v : v;
}
inline uint32_t half_from(double d) {
  if (std::isnan(d)) return 0x7FFF;
  const uint32_t sign = std::signbit(d) ? 0x8000u : 0u;
  const double a = std::fabs(d);
  if (a == 0) return sign;
  if (std::isinf(a) || a >= 65520.0) return sign | 0x7C00u;
  int e;
  std::frexp(a, &e);                                   // a in [2^(e-1), 2^e)
  int E = std::max(e - 1, -14);                        // the exponent the last place is scaled to
  const double r = std::nearbyint(std::ldexp(a, 10 - E));   // nearest even, in units of the last place
  uint64_t mant = static_cast<uint64_t>(r);
  if (E == -14 && mant < 1024) return sign | static_cast<uint32_t>(mant);   // subnormal
  if (mant >= 2048) { mant >>= 1; ++E; }
  if (E + 15 >= 31) return sign | 0x7C00u;
  return sign | (static_cast<uint32_t>(E + 15) << 10) | static_cast<uint32_t>(mant - 1024);
}

// An element type of a reduction: float or integer, signed or not, and its width.
struct TySpec {
  bool fl = false, sg = false;
  uint32_t bits = 32;
  Type type() const { return fl ? tf(bits) : sg ? ts(bits) : tu(bits); }
  std::string name() const { return std::string(fl ? "f" : sg ? "s" : "u") + std::to_string(bits); }
  uint64_t mask() const { return bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1; }
};
inline int64_t sext(uint64_t v, uint32_t bits) {
  if (bits >= 64) return static_cast<int64_t>(v);
  const uint64_t m = (uint64_t{1} << bits) - 1, sb = uint64_t{1} << (bits - 1);
  v &= m;
  return static_cast<int64_t>((v ^ sb) - sb);
}
inline bool is_nan_bits(const TySpec& t, uint64_t v) {
  if (t.bits == 16) return (v & 0x7C00) == 0x7C00 && (v & 0x3FF);
  if (t.bits == 32) return (v & 0x7F800000u) == 0x7F800000u && (v & 0x7FFFFFu);
  return (v & 0x7FF0000000000000ull) == 0x7FF0000000000000ull && (v & 0xFFFFFFFFFFFFFull);
}

// One step of a reduction, `a` then `b`, as the engine defines it (see OpRedux in ast.hpp).
inline uint64_t ref_combine(vgpu::ptx::ReduxOp op, const TySpec& t, uint64_t a, uint64_t b) {
  using vgpu::ptx::ReduxOp;
  const uint64_t m = t.mask();
  if (!t.fl) {
    a &= m;
    b &= m;
    switch (op) {
      case ReduxOp::Add: return (a + b) & m;
      case ReduxOp::Mul: return (a * b) & m;
      case ReduxOp::And: return a & b;
      case ReduxOp::Or: return a | b;
      case ReduxOp::Xor: return a ^ b;
      case ReduxOp::Min: return t.sg ? (sext(a, t.bits) < sext(b, t.bits) ? a : b) : (a < b ? a : b);
      case ReduxOp::Max: return t.sg ? (sext(a, t.bits) > sext(b, t.bits) ? a : b) : (a > b ? a : b);
    }
    return 0;
  }
  if (t.bits == 16) {
    const double x = half_value(static_cast<uint32_t>(a)), y = half_value(static_cast<uint32_t>(b));
    double r = 0;
    switch (op) {
      case ReduxOp::Add: r = x + y; break;
      case ReduxOp::Mul: r = x * y; break;
      case ReduxOp::Min: r = std::isnan(x) ? y : std::isnan(y) ? x : (x < y ? x : y); break;
      case ReduxOp::Max: r = std::isnan(x) ? y : std::isnan(y) ? x : (x > y ? x : y); break;
      default: break;
    }
    return half_from(r);
  }
  if (t.bits == 32) {
    const float x = std::bit_cast<float>(static_cast<uint32_t>(a)), y = std::bit_cast<float>(static_cast<uint32_t>(b));
    if (op == ReduxOp::Min || op == ReduxOp::Max) {
      if (std::isnan(x) && std::isnan(y)) return 0x7FFFFFFFu;
      if (std::isnan(x)) return b;
      if (std::isnan(y)) return a;
      if (x == y) {   // equal: the same bits, except zeros (-0 wins a min, +0 wins a max)
        if (x != 0) return a;
        const bool neg = op == ReduxOp::Min ? ((a | b) & 0x80000000u) != 0 : ((a & b) & 0x80000000u) != 0;
        return neg ? 0x80000000u : 0u;
      }
      return (op == ReduxOp::Min ? x < y : x > y) ? a : b;
    }
    const float r = op == ReduxOp::Add ? x + y : x * y;
    return std::isnan(r) ? 0x7FFFFFFFu : std::bit_cast<uint32_t>(r);
  }
  const double x = std::bit_cast<double>(a), y = std::bit_cast<double>(b);
  double r;
  switch (op) {
    case ReduxOp::Add: r = x + y; break;
    case ReduxOp::Mul: r = x * y; break;
    case ReduxOp::Min:
      r = std::isnan(x) ? y : std::isnan(y) ? x : (x == 0 && y == 0) ? (std::signbit(x) || std::signbit(y) ? -0.0 : 0.0) : (x < y ? x : y);
      break;
    default:
      r = std::isnan(x) ? y : std::isnan(y) ? x : (x == 0 && y == 0) ? (std::signbit(x) && std::signbit(y) ? -0.0 : 0.0) : (x > y ? x : y);
      break;
  }
  return std::bit_cast<uint64_t>(r);
}
inline uint64_t ref_identity(vgpu::ptx::ReduxOp op, const TySpec& t) {
  using vgpu::ptx::ReduxOp;
  const uint64_t ones = t.mask();
  if (t.fl) {
    const uint64_t one = t.bits == 16 ? 0x3C00u : t.bits == 32 ? 0x3F800000u : 0x3FF0000000000000ull;
    const uint64_t inf = t.bits == 16 ? 0x7C00u : t.bits == 32 ? 0x7F800000u : 0x7FF0000000000000ull;
    const uint64_t sign = uint64_t{1} << (t.bits - 1);
    return op == ReduxOp::Mul ? one : op == ReduxOp::Min ? inf : op == ReduxOp::Max ? (inf | sign) : 0;
  }
  switch (op) {
    case ReduxOp::Mul: return 1;
    case ReduxOp::And: return ones;
    case ReduxOp::Min: return t.sg ? ones >> 1 : ones;
    case ReduxOp::Max: return t.sg ? (ones >> 1) + 1 : 0;
    default: return 0;
  }
}
// Equal as the engine promises: the same bits, except that every NaN of a float type is the same value.
inline bool same_value(const TySpec& t, uint64_t got, uint64_t want) {
  got &= t.mask();
  want &= t.mask();
  if (t.fl && is_nan_bits(t, got) && is_nan_bits(t, want)) return true;
  return got == want;
}
// A random operand: integers mix small values and the extremes, floats mix ordinary values, zeros, infinities, NaN
// and subnormals (f16 min/max leave the signed zeros out: the engine takes them from the host's fmin/fmax).
inline uint64_t random_value(Rng& r, const TySpec& t, bool zeros = true) {
  const uint64_t pick = r.next() % 16;
  if (!t.fl) {
    if (pick == 0) return 0;
    if (pick == 1) return t.mask();
    if (pick == 2) return (t.mask() >> 1);                      // max signed
    if (pick == 3) return (t.mask() >> 1) + 1;                  // min signed
    if (pick < 10) return (r.next() % 33) & t.mask();           // small
    return r.next() & t.mask();
  }
  const uint32_t eb = t.bits == 16 ? 5 : t.bits == 32 ? 8 : 11, mb = t.bits - 1 - eb;
  const uint64_t emax = (uint64_t{1} << eb) - 1, bias = emax >> 1;
  auto make = [&](uint64_t sign, uint64_t e, uint64_t m) { return (sign << (t.bits - 1)) | (e << mb) | m; };
  const uint64_t sign = r.next() & 1, mant = r.next() & ((uint64_t{1} << mb) - 1);
  switch (pick) {
    case 0: return zeros ? make(sign, 0, 0) : make(sign, bias, mant);
    case 1: return make(sign, emax, 0);                    // infinity
    case 2: return make(sign, emax, (uint64_t{1} << (mb - 1)) | (mant >> 1));   // quiet NaN
    case 3: return make(sign, 0, mant | 1);                // subnormal
    case 4: return make(sign, emax - 1, mant);             // huge
    default: return make(sign, bias - 3 + r.next() % 7, mant);   // ordinary, magnitude 1/8 .. 16
  }
}

}  // namespace astrun
