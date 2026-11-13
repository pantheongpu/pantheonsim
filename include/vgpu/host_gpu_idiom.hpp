// Host-GPU exact mode: the IEEE square root idiom (sqrt.rn.f32: `sqrtf`, `__fsqrt_rn`), recognised in a kernel's SASS (Round sp68).
//
// MUFU.RSQ alone is not in the exact set: the card's reciprocal square root differs from the simulator's correctly rounded one on 10.7% of the inputs, by
// up to 2 ulp (Round sp44, B). But ptxas builds the IEEE square root from it. For sm_86 (nvcc 13.0 and 13.2 give the same SASS, Round sp67) the call site is
//
//     IADD3 Rt, Rx, -0xd000000, RZ                    the range check: Rx - 0x0d000000 above 0x727fffff (unsigned) is zero, a negative, a subnormal, a
//     MUFU.RSQ Ry, Rx                                  float below 2^-101, an infinity or a NaN: those go to the slow path
//     ISETP.GT.U32.AND Pp, PT, Rt, 0x727fffff, PT
//     @!Pp BRA fast
//     MOV Rret, <the address after the CALL>
//     CALL.REL.NOINC slow                              (the argument and the result in registers of ptxas's choice)
//     BRA join
//   fast:
//     FMUL.FTZ Rs, Ry, Rx                              s = y x, h = y / 2, the residual x - s s with one rounding, and s + residual h
//     FMUL.FTZ Rh, Ry, 0.5
//     FFMA Rd, -Rs, Rs, Rx
//     FFMA Rr, Rd, Rh, Rs
//   join:
//
// and the slow path a subroutine that returns x for a zero and an infinity, the NaN 0x7fffffff for a negative, x + 1 for a NaN, and for the rest scales x by
// 2^64, does the same four steps on it and scales the result by 2^-32. With the simulator's MUFU.RSQ both paths give the IEEE root for every input
// (tests/unit/test_host_gpu_ref.cpp, all 2^31 positive floats), and the card ran this code (tools/probes/hostgpu_sfu_exhaustive.cu sqrtrn) on all 2^32 inputs
// with 0 results different from the simulator's (Round sp44, A10G). That second fact is the card's, not the refinement's: with a reciprocal square root one
// ulp off the simulator's a few hundred of the 2^31 inputs get a wrong root (the same test), so the idiom is exact on a card only where its MUFU.RSQ is
// right for those inputs. That is why VGPU_HOST_GPU_SQRT is a switch, off by default: only the A10G (sm_86) has been run.
//
// How a kernel's code is recognised. Not by the instructions' order (ptxas schedules other work in between, predicates the slow path in one kernel and
// branches around it in another), but by what the code computes: from the IADD3 and the MUFU to the join, every path (both sides of each predicate, through
// the CALL into the subroutine and back) is executed on symbolic values. Each register then holds an expression of the registers at the start, as text
// (multiplications with their two factors sorted, so that the order of the operands does not matter, and the copies followed). The paths, each with the
// predicate values it took, must be exactly the paths of the reference: the probe's own SASS (the code the card ran, kReferenceWords below), with the register
// that held x renamed. Then
//   * a register that holds the reference's result on every path carries the square root out of the join;
//   * every other register or predicate whose value at the join depends on a MUFU (the raw reciprocal square root, the scaled one) must not be read after
//     the join on any path of the kernel before it is written (a register read later would carry the card's approximation);
//   * nothing in the region touches memory, other threads or uniform registers, a branch goes forward only, the subroutine returns to the CALL's next
//     instruction, and no branch from outside lands inside the region or inside the subroutine.
// Anything else is refused: a bare rsqrtf, a MUFU.RSQ without the refinement or without the range check, other constants, a slow path that is not this
// one or that touches memory, a result that is not the root. A kernel is in the exact set by this idiom when every instruction outside the set is a
// MUFU.RSQ, CALL or RET of a recognised site (plus the division idiom's when VGPU_HOST_GPU_DIV is on).
#pragma once

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vgpu/host_gpu.hpp"
#include "vgpu/sass/sass.hpp"

namespace vgpu::hostgpu::idiom {

using sass::Instr;
using sass::Kind;
using sass::Operand;

// ---- registers read and written, the code as a graph ---------------------------------------------------------------------------------------------

struct Regs {
  std::bitset<256> r;   // R0..R254 (RZ is never set)
  std::bitset<8> p;     // P0..P6 (PT is never set)
};

// Every R<n> named in an operand's printed text (the address of a memory operand, an indexed constant).
inline void regs_in_text(const std::string& t, std::bitset<256>& r) {
  for (size_t i = 0; i + 1 < t.size(); ++i) {
    if (t[i] != 'R' || (i > 0 && (std::isalnum(static_cast<unsigned char>(t[i - 1])) || t[i - 1] == '_'))) continue;
    size_t j = i + 1;
    unsigned n = 0;
    while (j < t.size() && t[j] >= '0' && t[j] <= '9') n = n * 10 + static_cast<unsigned>(t[j++] - '0');
    if (j > i + 1 && n < 255) r.set(n);
  }
}

// How many registers a source operand reads, where the decoder may give a width of 1 for a pair or a quad: rounded up, never down (a register counted as
// read that is not only makes the check below refuse more).
inline unsigned read_width(const Instr& ins, size_t k) {
  const std::string& m = ins.mnemonic;
  if (m == "HMMA" || m == "IMMA" || m == "DMMA" || m == "HGMMA" || m == "LDSM" || m == "MOVM") return 8;
  if (m == "IMAD" && mods_have(ins.mods, "WIDE")) return k == 2 ? 2 : 1;   // a 32 x 32 product plus a 64-bit addend
  if (m == "RET" || m == "BRX" || (!m.empty() && m[0] == 'D')) return 2;   // a return address; a double
  const bool memory = m == "LDG" || m == "STG" || m == "LDS" || m == "STS" || m == "LDL" || m == "STL" || m == "LD" || m == "ST" || m == "LDC" ||
                      m.rfind("ATOM", 0) == 0 || m.rfind("RED", 0) == 0 || m == "FRND";
  if (memory) {   // the data of a wide access (the address is the Mem operand)
    if (any_mod_part(ins.mods, [](const std::string& p) { return p.find("128") != std::string::npos; })) return 4;
    if (any_mod_part(ins.mods, [](const std::string& p) { return p.find("64") != std::string::npos; })) return 2;
    return 1;
  }
  if (m == "F2F" || m == "I2F" || m == "F2I" || m == "I2I" || m == "I2FP" || m == "F2IP") {   // the source's type is the last type printed
    std::string last;
    any_mod_part(ins.mods, [&](const std::string& p) {
      if (p.size() >= 2 && (p[0] == 'F' || p[0] == 'S' || p[0] == 'U') && std::isdigit(static_cast<unsigned char>(p[1]))) last = p;
      return false;
    });
    return last.find("64") != std::string::npos ? 2 : 1;
  }
  return 1;
}

inline Regs reads(const Instr& ins) {
  Regs g;
  if (!ins.guard_uniform && ins.guard != sass::kPT) g.p.set(ins.guard);
  for (size_t k = 0; k < ins.src.size(); ++k) {
    const Operand& o = ins.src[k];
    if ((o.kind == Kind::Reg || (o.kind == Kind::Text && (ins.mnemonic == "RET" || ins.mnemonic == "BRX"))) && o.reg != sass::kRZ)
      for (unsigned w = 0; w < std::max<unsigned>(read_width(ins, k), o.width) && o.reg + w < 255; ++w) g.r.set(o.reg + w);
    if (o.kind == Kind::Pred && o.reg != sass::kPT) g.p.set(o.reg);
    if (o.kind == Kind::Mem && o.reg != sass::kRZ) {
      g.r.set(o.reg);
      if (o.reg + 1 < 255 && o.text.find(".64") != std::string::npos) g.r.set(o.reg + 1);   // a 64-bit address reads the pair
    }
    if (o.kind == Kind::Mem || o.kind == Kind::Text || o.kind == Kind::CBank) regs_in_text(o.text, g.r);
  }
  return g;
}

// How many registers a destination is known to be, where the decoder may give 1 for a pair or a quad: only the forms that certainly write more (counting a
// register as written that is not would hide a read of an older value).
inline unsigned write_width(const Instr& ins) {
  const std::string& m = ins.mnemonic;
  if (m == "IMAD" && mods_have(ins.mods, "WIDE")) return 2;
  if (m == "DADD" || m == "DMUL" || m == "DFMA" || m == "DMNMX") return 2;
  if (m == "LDG" || m == "LDS" || m == "LDL" || m == "LD" || m == "LDC") {
    if (mods_have(ins.mods, "128")) return 4;
    if (mods_have(ins.mods, "64")) return 2;
  }
  return 1;
}

inline Regs writes(const Instr& ins) {
  Regs g;
  for (const Operand& o : ins.dst) {
    if (o.kind == Kind::Reg && o.reg != sass::kRZ)
      for (unsigned k = 0; k < std::max<unsigned>(o.kind == Kind::Reg && &o == &ins.dst[0] ? write_width(ins) : 1, o.width) && o.reg + k < 255; ++k)
        g.r.set(o.reg + k);
    if (o.kind == Kind::Pred && o.reg != sass::kPT) g.p.set(o.reg);
  }
  return g;
}

inline bool unconditional(const Instr& ins) { return !ins.guard_uniform && ins.guard == sass::kPT && !ins.guard_not; }

inline int64_t label_of(const Instr& ins) {
  for (const Operand& o : ins.src)
    if (o.kind == Kind::Label) return o.imm;
  return -1;
}

struct Code {
  const std::vector<Instr>& ins;
  std::map<uint64_t, size_t> at;   // pc -> index
  explicit Code(const std::vector<Instr>& c) : ins(c) {
    for (size_t i = 0; i < c.size(); ++i) at.emplace(c[i].pc, i);
  }
  long index(int64_t pc) const {
    const auto it = pc < 0 ? at.end() : at.find(static_cast<uint64_t>(pc));
    return it == at.end() ? -1 : static_cast<long>(it->second);
  }
};

// May the register (pred = false) or predicate `n` be read, on some path from instruction `from`, before it is written? Calls are followed into the
// subroutine and back; anything this cannot follow (a computed branch, a RET with no call seen, a deep call chain) counts as a read. A write under a
// predicate covers the reads under the same predicate until that predicate is written again (`@P2 FMUL R28 ...; @P2 FFMA ..., R28, ...` reads the new R28),
// and writes under P and under !P with P unchanged in between are a write.
inline bool may_be_read(const Code& c, size_t from, bool pred, unsigned n) {
  struct St { size_t i; std::vector<size_t> ret; uint16_t covered; };   // covered: bit 2 g + neg, the guards a write of `n` was made under
  std::vector<St> work{{from, {}, 0}};
  std::set<std::tuple<size_t, std::vector<size_t>, uint16_t>> seen;
  while (!work.empty()) {
    St s = std::move(work.back());
    work.pop_back();
    if (s.i >= c.ins.size()) continue;
    if (!seen.insert({s.i, s.ret, s.covered}).second) continue;
    if (seen.size() > 200000) return true;
    const Instr& ins = c.ins[s.i];
    const bool guarded = !ins.guard_uniform && ins.guard != sass::kPT;
    const unsigned gbit = guarded ? 2 * ins.guard + (ins.guard_not ? 1 : 0) : 0;
    const Regs rd = reads(ins);
    if ((pred ? rd.p.test(n) : rd.r.test(n)) && !(guarded && (s.covered >> gbit & 1))) return true;
    const Regs wr = writes(ins);
    const bool writes_n = pred ? wr.p.test(n) : wr.r.test(n);
    if (writes_n && unconditional(ins)) continue;
    for (unsigned q = 0; q < 7; ++q)   // a predicate written again: what was covered under it is not
      if (wr.p.test(q)) s.covered &= static_cast<uint16_t>(~(3u << (2 * q)));
    if (writes_n && guarded && !wr.p.test(ins.guard)) {
      s.covered |= static_cast<uint16_t>(1u << gbit);
      if ((s.covered >> (2 * ins.guard) & 3) == 3) continue;   // written under P and under !P with P unchanged in between: written
    }
    const std::string& m = ins.mnemonic;
    if (m == "BRA") {
      const long t = c.index(label_of(ins));
      if (t < 0) return true;
      // a branch under P is taken only when P holds: a write made under P is a write on that edge, one made under !P on the other
      const bool taken_written = guarded && (s.covered >> gbit & 1), fall_written = guarded && (s.covered >> (gbit ^ 1) & 1);
      if (!taken_written) work.push_back({static_cast<size_t>(t), s.ret, s.covered});
      if (!unconditional(ins) && !fall_written) work.push_back({s.i + 1, s.ret, s.covered});
    } else if (m == "EXIT") {
      if (!unconditional(ins)) work.push_back({s.i + 1, s.ret, s.covered});
    } else if (m == "CALL") {
      const long t = c.index(label_of(ins));
      if (t < 0 || s.ret.size() >= 4) return true;
      std::vector<size_t> r = s.ret;
      r.push_back(s.i + 1);
      work.push_back({static_cast<size_t>(t), std::move(r), s.covered});
    } else if (m == "RET") {
      if (s.ret.empty()) return true;
      std::vector<size_t> r = s.ret;
      const size_t back = r.back();
      r.pop_back();
      work.push_back({back, std::move(r), s.covered});
      if (!unconditional(ins)) work.push_back({s.i + 1, s.ret, s.covered});
    } else if (m == "BRX" || m == "JMX" || m == "JMP" || m == "RTT" || m == "BPT" || m == "CALLX" || m == "KILL") {
      return true;
    } else {
      work.push_back({s.i + 1, s.ret, s.covered});
    }
  }
  return false;
}

// ---- symbolic execution -----------------------------------------------------------------------------------------------------------------------------

struct Path {
  size_t i = 0;
  std::map<unsigned, std::string> r, p;          // registers and predicates written so far (the others hold "in(R<n>)", "in(P<n>)")
  std::vector<std::pair<std::string, bool>> cond;   // the predicate values this path took
  std::vector<size_t> ret;                        // return indices of the calls this path is in
  std::set<size_t> visited;
  unsigned steps = 0;
  unsigned x = 255;                               // the argument's register: read before it is written, it is "X"
};

inline std::string reg_value(const Path& s, unsigned n) {
  if (n == sass::kRZ) return "#0";
  const auto it = s.r.find(n);
  if (it != s.r.end()) return it->second;
  return n == s.x ? std::string("X") : "in(R" + std::to_string(n) + ")";
}
inline std::string pred_value(const Path& s, unsigned n) {
  if (n == sass::kPT) return "PT";
  const auto it = s.p.find(n);
  return it != s.p.end() ? it->second : "in(P" + std::to_string(n) + ")";
}

inline std::string hex(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
  return b;
}

// An operand's value as text; `sign` collects a leading minus separately (for the products), otherwise it is part of the text.
inline std::string operand_value(const Path& s, const Operand& o, bool* sign = nullptr) {
  std::string v;
  switch (o.kind) {
    case Kind::Reg:
      for (unsigned k = 0; k < std::max<unsigned>(1, o.width); ++k) v += (k ? ":" : "") + reg_value(s, o.reg == sass::kRZ ? o.reg : o.reg + k);
      break;
    case Kind::Pred: v = pred_value(s, o.reg); break;
    case Kind::Imm: v = "#" + std::to_string(o.imm); break;
    case Kind::FImm: v = "#f" + std::to_string(o.fwidth) + ":" + hex(o.fbits); break;
    case Kind::CBank: v = "c[" + std::to_string(o.reg) + "][" + std::to_string(o.imm) + "]"; break;
    default: v = o.text.empty() ? "?" : o.text; break;
  }
  if (o.abs) v = "|" + v + "|";
  if (o.bnot) v = "~" + v;
  if (!o.suffix.empty()) v += o.suffix;
  if (o.neg) {
    if (sign && o.kind != Kind::Pred) *sign = !*sign;
    else v = (o.kind == Kind::Pred ? "!" : "-") + v;
  }
  return v;
}

inline std::string joined_mods(const Instr& ins) {
  std::string m;
  for (const std::string& x : ins.mods) m += "." + x;
  return m;
}

// The value an instruction computes, as text. Copies are followed; the two factors of FMUL and FFMA and the two terms of FADD are sorted (IEEE products
// and sums do not depend on the order of their operands, and -a * b is -(a * b)).
inline std::string expression(const Path& s, const Instr& ins) {
  const std::string m = ins.mnemonic + joined_mods(ins);
  const auto& src = ins.src;
  if (ins.mnemonic == "MOV" && !src.empty()) return operand_value(s, src[0]);
  if (m == "IMAD.MOV.U32" && src.size() == 3 && src[0].kind == Kind::Reg && src[0].reg == sass::kRZ && src[1].kind == Kind::Reg && src[1].reg == sass::kRZ)
    return operand_value(s, src[2]);
  if ((ins.mnemonic == "FMUL" || ins.mnemonic == "FFMA") && src.size() >= 2) {
    bool neg = false;
    std::string a = operand_value(s, src[0], &neg), b = operand_value(s, src[1], &neg);
    if (b < a) std::swap(a, b);
    std::string e = m + (neg ? "(-" : "(") + a + "*" + b;
    if (ins.mnemonic == "FFMA" && src.size() >= 3) e += "+" + operand_value(s, src[2]);
    return e + ")";
  }
  if (ins.mnemonic == "FADD" && src.size() == 2) {
    std::string a = operand_value(s, src[0]), b = operand_value(s, src[1]);
    if (b < a) std::swap(a, b);
    return m + "(" + a + "+" + b + ")";
  }
  std::string e = m + "(";
  for (size_t k = 0; k < src.size(); ++k) e += (k ? "," : "") + operand_value(s, src[k]);
  return e + ")";
}

// Does a value's text name the argument (the token X)?
inline bool mentions_x(const std::string& v) {
  for (size_t at = v.find('X'); at != std::string::npos; at = v.find('X', at + 1)) {
    const bool before = at > 0 && (std::isalnum(static_cast<unsigned char>(v[at - 1])) || v[at - 1] == '_' || v[at - 1] == '.');
    const bool after = at + 1 < v.size() && (std::isalnum(static_cast<unsigned char>(v[at + 1])) || v[at + 1] == '_' || v[at + 1] == '.');
    if (!before && !after) return true;
  }
  return false;
}

struct Region {
  bool ok = false;
  const char* why = "";
  std::vector<Path> paths;   // the finished paths, at the join
};

// Runs every path from instruction `start` until it reaches instruction `join` outside any call, with register `x` (the argument) read as "X". No memory,
// barrier or other thread is allowed inside, and of the SFU forms only MUFU.RSQ.
inline Region run_region(const Code& c, size_t start, size_t join, unsigned x, unsigned max_paths = 64) {
  Region out;
  std::vector<Path> work(1);
  work[0].i = start;
  work[0].x = x;
  unsigned made = 1;
  const auto fail = [&](const char* why) {
    out.ok = false;
    out.why = why;
    out.paths.clear();
    return out;
  };
  while (!work.empty()) {
    Path s = std::move(work.back());
    work.pop_back();
    for (;;) {
      if (s.i == join && s.ret.empty()) {
        out.paths.push_back(std::move(s));
        break;
      }
      if (s.i >= c.ins.size()) return fail("a path runs off the end of the code");
      if (++s.steps > 400) return fail("a path too long for the idiom");
      const Instr& ins = c.ins[s.i];
      s.visited.insert(s.i);
      // the guard: a predicate computed from the argument that this path has not decided yet splits it in two. Any other predicate (one the code
      // around the idiom computed) guards an instruction that is not a branch as a choice: its results are "SEL(guard, new, old)".
      bool active = true;
      std::string select;   // the guard, when the instruction's results are a choice
      const std::string& mn = ins.mnemonic;
      if (ins.guard_uniform) return fail("an instruction under a uniform predicate");
      if (ins.guard != sass::kPT || ins.guard_not) {
        const std::string g = pred_value(s, ins.guard);
        bool value;
        const bool control = mn == "BRA" || mn == "CALL" || mn == "RET" || mn == "EXIT" || mn == "BSSY" || mn == "BSYNC" || mn == "BREAK";
        if (g == "PT") value = true;
        else if (!control && !mentions_x(g) && std::none_of(s.cond.begin(), s.cond.end(), [&](const auto& x) { return x.first == g; })) {
          value = true;
          select = (ins.guard_not ? "!" : "") + g;
        } else {
          const auto it = std::find_if(s.cond.begin(), s.cond.end(), [&](const auto& x) { return x.first == g; });
          if (it != s.cond.end()) value = it->second;
          else {
            if (++made > max_paths) return fail("too many paths");
            Path other = s;
            other.cond.emplace_back(g, false);
            work.push_back(std::move(other));
            s.cond.emplace_back(g, true);
            value = true;
          }
        }
        active = select.empty() && ins.guard_not ? !value : value;
      }
      if (!active) {
        ++s.i;
        continue;
      }
      if (mn == "BRA") {
        const long t = c.index(label_of(ins));
        if (t < 0 || c.ins[static_cast<size_t>(t)].pc <= ins.pc) return fail("a branch that is not forward");
        s.i = static_cast<size_t>(t);
        continue;
      }
      if (mn == "CALL") {
        const long t = c.index(label_of(ins));
        if (t < 0 || !mods_have(ins.mods, "REL") || s.ret.size() >= 2) return fail("a call that is not a relative call to the code");
        s.ret.push_back(s.i + 1);
        s.i = static_cast<size_t>(t);
        continue;
      }
      if (mn == "RET") {
        // RET.REL.NODEC Rk: the return address is the pair Rk, Rk+1, which must be the address after the CALL and 0
        if (s.ret.empty() || ins.src.empty() || (ins.src[0].kind != Kind::Reg && ins.src[0].kind != Kind::Text) || ins.src[0].reg + 1 >= sass::kRZ ||
            ins.src[0].imm != 0)
          return fail("a RET with no call");
        const uint64_t back = c.ins[s.ret.back()].pc;
        if (reg_value(s, ins.src[0].reg) != "#" + std::to_string(back) || reg_value(s, ins.src[0].reg + 1) != "#0")
          return fail("a RET that does not go back to its call");
        s.i = s.ret.back();
        s.ret.pop_back();
        continue;
      }
      if (mn == "BSSY" || mn == "BSYNC" || mn == "NOP" || mn == "YIELD") {
        ++s.i;
        continue;
      }
      if (mn == "MUFU") {
        if (ins.mods.size() != 1 || ins.mods[0] != "RSQ") return fail("an SFU instruction other than MUFU.RSQ in the idiom");
      } else if (classify_for_idiom(mn, ins.mods) != CodeInsn::Kind::Other || mn == "EXIT" || mn == "BRX" || mn == "BREAK" || mn == "WARPSYNC") {
        return fail("memory, a barrier, another thread or control flow inside the idiom");
      }
      for (const Operand& o : ins.dst)
        if (o.kind == Kind::UReg || o.kind == Kind::UPred) return fail("a uniform register written inside the idiom");
      const std::string e = expression(s, ins);
      // a value computed from a MUFU may only flow through the idiom's own arithmetic and copies, whose results are single registers
      const bool copy = mn == "MOV" || (mn == "IMAD" && joined_mods(ins) == ".MOV.U32" && ins.src.size() == 3 && ins.src[0].kind == Kind::Reg &&
                                        ins.src[0].reg == sass::kRZ && ins.src[1].kind == Kind::Reg && ins.src[1].reg == sass::kRZ);
      if (e.find("MUFU") != std::string::npos && mn != "MUFU" && mn != "FMUL" && mn != "FFMA" && mn != "FADD" && !copy)
        return fail("a MUFU value used by an instruction that is not the refinement");
      unsigned k = 0;
      std::vector<std::pair<unsigned, std::string>> rw, pw;
      for (const Operand& o : ins.dst) {
        const std::string v = ins.dst.size() == 1 ? e : e + "#" + std::to_string(k);
        ++k;
        if (o.kind == Kind::Reg && o.reg != sass::kRZ)
          for (unsigned w = 0; w < std::max<unsigned>(1, o.width); ++w) rw.emplace_back(o.reg + w, o.width > 1 ? v + "." + std::to_string(w) : v);
        else if (o.kind == Kind::Pred && o.reg != sass::kPT)
          pw.emplace_back(o.reg, v);
      }
      for (auto& [n, v] : rw) s.r[n] = select.empty() ? v : "SEL(" + select + "," + v + "," + reg_value(s, n) + ")";
      for (auto& [n, v] : pw) s.p[n] = select.empty() ? v : "SEL(" + select + "," + v + "," + pred_value(s, n) + ")";
      ++s.i;
    }
  }
  out.ok = true;
  return out;
}

// One path's outcome with the argument renamed: the predicate values it took (sorted) and the value of every register.
inline std::string renamed(std::string v, unsigned x) {
  const std::string from = "in(R" + std::to_string(x) + ")";
  for (size_t at = v.find(from); at != std::string::npos; at = v.find(from, at + 1)) v.replace(at, from.size(), "X");
  return v;
}
inline std::string path_key(const Path& s, unsigned x) {
  std::vector<std::string> c;
  for (const auto& [g, v] : s.cond) c.push_back((v ? "" : "!") + renamed(g, x));
  std::sort(c.begin(), c.end());
  std::string k;
  for (const std::string& t : c) k += t + ";";
  return k;
}

// ---- the reference: the probe's sqrt.rn, as the card ran it ------------------------------------------------------------------------------------------
// tools/probes/hostgpu_sfu_exhaustive.cu op sqrtrn (`__fsqrt_rn(x)`), nvcc 13.0.88 -arch=sm_86 (13.2.51 gives the same words, Round sp67): the call site
// at 0x530 to the join at 0x5f0 and the slow path at 0xec0. The card ran exactly this code over all 2^32 inputs with 0 differences (Round sp44, B). The
// words are the probe's own build, read with cuobjdump on the rig (an own kernel; docs/perf-design/data/sp68/README.md).
struct RefWord { uint64_t pc, lo, hi; };
inline constexpr RefWord kReferenceWords[] = {
    {0x0530, 0xf300000003007810ull, 0x000fe20007ffe0ffull},   // IADD3 R0, R3, -0xd000000, RZ
    {0x0540, 0x0000000300027308ull, 0x0000620000001400ull},   // MUFU.RSQ R2, R3
    {0x0550, 0x000000a000017945ull, 0x000fe40003800000ull},   // BSSY B1, 0x600
    {0x0560, 0x727fffff0000780cull, 0x000fda0003f04070ull},   // ISETP.GT.U32.AND P0, PT, R0, 0x727fffff, PT
    {0x0570, 0x0000003000008947ull, 0x000fea0003800000ull},   // @!P0 BRA 0x5b0
    {0x0580, 0x000005a0000a7802ull, 0x003fe40000000f00ull},   // MOV R10, 0x5a0
    {0x0590, 0x0000092000007944ull, 0x000fea0003c00000ull},   // CALL.REL.NOINC 0xec0
    {0x05a0, 0x0000004000007947ull, 0x000fea0003800000ull},   // BRA 0x5f0
    {0x05b0, 0x0000000302007220ull, 0x043fe20000410000ull},   // FMUL.FTZ R0, R2.reuse, R3
    {0x05c0, 0x3f00000002027820ull, 0x000fc60000410000ull},   // FMUL.FTZ R2, R2, 0.5
    {0x05d0, 0x0000000000077223ull, 0x000fc80000000103ull},   // FFMA R7, -R0, R0, R3
    {0x05e0, 0x0000000207007223ull, 0x000fe40000000000ull},   // FFMA R0, R7, R2, R0
    {0x05f0, 0x0000000000017941ull, 0x000fea0003800000ull},   // BSYNC B1                         (the join)
    {0x0ec0, 0x7fffffff03ff7812ull, 0x000fda000780c0ffull},   // LOP3.LUT P0, RZ, R3, 0x7fffffff, RZ, 0xc0, !PT
    {0x0ed0, 0x000000ffff028224ull, 0x000fe200078e0003ull},   // @!P0 IMAD.MOV.U32 R2, RZ, RZ, R3
    {0x0ee0, 0x0000011000008947ull, 0x000fea0003800000ull},   // @!P0 BRA 0x1000
    {0x0ef0, 0x000000ff0300720bull, 0x000fe20003f1e000ull},   // FSETP.GEU.FTZ.AND P0, PT, R3, RZ, PT
    {0x0f00, 0x000000ffff007224ull, 0x000fd800078e0003ull},   // IMAD.MOV.U32 R0, RZ, RZ, R3
    {0x0f10, 0x7fffffffff028424ull, 0x000fe200078e00ffull},   // @!P0 IMAD.MOV.U32 R2, RZ, RZ, 0x7fffffff
    {0x0f20, 0x000000d000008947ull, 0x000fea0003800000ull},   // @!P0 BRA 0x1000
    {0x0f30, 0x7f8000000000780bull, 0x000fda0003f1c200ull},   // FSETP.GTU.FTZ.AND P0, PT, |R0|, +INF , PT
    {0x0f40, 0x3f80000000020421ull, 0x000fe20000010000ull},   // @P0 FADD.FTZ R2, R0, 1
    {0x0f50, 0x000000a000000947ull, 0x000fea0003800000ull},   // @P0 BRA 0x1000
    {0x0f60, 0x7f8000000000780bull, 0x000fda0003f1d200ull},   // FSETP.NEU.FTZ.AND P0, PT, |R0|, +INF , PT
    {0x0f70, 0x5f80000000060823ull, 0x000fc800000000ffull},   // @P0 FFMA R6, R0, 1.84467440737095516160e+19, RZ
    {0x0f80, 0x0000000600070308ull, 0x000e240000001400ull},   // @P0 MUFU.RSQ R7, R6
    {0x0f90, 0x0000000706090220ull, 0x001fe20000410000ull},   // @P0 FMUL.FTZ R9, R6, R7
    {0x0fa0, 0x3f00000007070820ull, 0x000fc60000410000ull},   // @P0 FMUL.FTZ R7, R7, 0.5
    {0x0fb0, 0x800000ff09020221ull, 0x000fc80000010100ull},   // @P0 FADD.FTZ R2, -R9, -RZ
    {0x0fc0, 0x0000000209080223ull, 0x000fe20000000006ull},   // @P0 FFMA R8, R9, R2, R6
    {0x0fd0, 0x000000ffff028224ull, 0x000fc600078e0000ull},   // @!P0 IMAD.MOV.U32 R2, RZ, RZ, R0
    {0x0fe0, 0x0000000708070223ull, 0x000fc80000000009ull},   // @P0 FFMA R7, R8, R7, R9
    {0x0ff0, 0x2f80000007020820ull, 0x000fe20000410000ull},   // @P0 FMUL.FTZ R2, R7, 2.3283064365386962891e-10
    {0x1000, 0x000000ffff067224ull, 0x000fe400078e000aull},   // IMAD.MOV.U32 R6, RZ, RZ, R10
    {0x1010, 0x00000000ff077424ull, 0x000fe400078e00ffull},   // IMAD.MOV.U32 R7, RZ, RZ, 0x0
    {0x1020, 0x000000ffff007224ull, 0x000fe400078e0002ull},   // IMAD.MOV.U32 R0, RZ, RZ, R2
    {0x1030, 0xffffefc006007950ull, 0x000fea0003c3ffffull},   // RET.REL.NODEC R6 0x0
};
inline constexpr unsigned kReferenceX = 3, kReferenceResult = 0;
inline constexpr uint64_t kReferenceStart = 0x530, kReferenceJoin = 0x5f0;

// path key -> the result's value, of the reference (decoded and run once; sm_86, the architecture the card run was made on).
inline const std::map<std::string, std::string>& reference_paths() {
  static const std::map<std::string, std::string> ref = [] {
    std::vector<Instr> code;
    for (const RefWord& w : kReferenceWords) code.push_back(sass::decode(sass::Word{w.lo, w.hi}, w.pc, 86));
    const Code c(code);
    const Region g = run_region(c, static_cast<size_t>(c.index(kReferenceStart)), static_cast<size_t>(c.index(kReferenceJoin)), kReferenceX);
    std::map<std::string, std::string> m;
    for (const Path& s : g.paths) m[path_key(s, kReferenceX)] = renamed(reg_value(s, kReferenceResult), kReferenceX);
    return m;
  }();
  return ref;
}

// ---- one site, and a whole kernel -----------------------------------------------------------------------------------------------------------------------

struct Site {
  bool ok = false;
  const char* why = "";
  std::set<size_t> covered;   // the instructions the site's paths ran (its MUFUs, CALL and RET among them)
  size_t start = 0, join = 0;
  long callee = -1, callee_end = -1;   // the subroutine's first instruction and its RET
  int live = -1;                        // when refused for a value read after the join: the register (or 256 + the predicate)
};

inline constexpr unsigned kSqrtGuardWindow = 16;   // from the MUFU back to the IADD3 of the range check, and on to the guard branch
inline constexpr unsigned kSqrtSlowWindow = 12;    // from the guard branch to the slow side's branch to the join

// Is the MUFU.RSQ at index `m` the reciprocal square root of a sqrt.rn idiom?
inline Site sqrt_site(const Code& c, size_t m) {
  Site st;
  const auto fail = [&](const char* why) {
    st.ok = false;
    st.why = why;
    return st;
  };
  const Instr& mu = c.ins[m];
  if (mu.mnemonic != "MUFU" || mu.mods.size() != 1 || mu.mods[0] != "RSQ") return fail("not a MUFU.RSQ");
  if (!unconditional(mu) || mu.src.size() != 1 || mu.src[0].kind != Kind::Reg || mu.src[0].reg == sass::kRZ || mu.src[0].neg || mu.src[0].abs)
    return fail("a MUFU.RSQ under a predicate or of an operand that is not a plain register");
  const unsigned x = mu.src[0].reg;
  // the range check's IADD3 of x, at most kSqrtGuardWindow before the MUFU (or after it: then the region starts at the MUFU)
  size_t start = m;
  for (size_t k = 1; k <= kSqrtGuardWindow && k <= m; ++k) {
    const Instr& a = c.ins[m - k];
    if (a.mnemonic == "IADD3" && a.src.size() == 3 && a.src[0].kind == Kind::Reg && a.src[0].reg == x && !a.src[0].neg && a.src[1].kind == Kind::Imm &&
        static_cast<uint32_t>(a.src[1].imm) == 0xf3000000u) {
      start = m - k;
      break;
    }
    if (a.mnemonic == "BRA" || a.mnemonic == "CALL" || a.mnemonic == "RET" || a.mnemonic == "EXIT") break;
  }
  // the guard branch, then the slow side's unconditional branch to the join
  size_t g = c.ins.size();
  for (size_t k = m + 1; k < c.ins.size() && k <= m + kSqrtGuardWindow; ++k)
    if (c.ins[k].mnemonic == "BRA") { g = k; break; }
  if (g == c.ins.size() || unconditional(c.ins[g])) return fail("no guard branch after the MUFU.RSQ (a bare rsqrt)");
  const int64_t fast = label_of(c.ins[g]);
  long join = -1;
  for (size_t k = g + 1; k < c.ins.size() && k <= g + kSqrtSlowWindow; ++k)
    if (c.ins[k].mnemonic == "BRA" && unconditional(c.ins[k])) {
      const int64_t t = label_of(c.ins[k]);
      if (t > fast && fast > static_cast<int64_t>(c.ins[k].pc)) join = c.index(t);
      break;
    }
  if (join < 0) return fail("no slow side ending in a branch to the join");
  st.start = start;
  st.join = static_cast<size_t>(join);
  const Region r = run_region(c, start, static_cast<size_t>(join), x);
  if (!r.ok) return fail(r.why);
  // the paths must be the reference's, and one register must hold the reference's result on all of them
  const auto& ref = reference_paths();
  if (r.paths.size() != ref.size()) return fail("not the paths of the reference sqrt.rn");
  std::map<std::string, const Path*> by_key;
  for (const Path& s : r.paths) by_key[path_key(s, x)] = &s;
  for (const auto& [k, v] : ref)
    if (!by_key.count(k)) return fail("not the paths of the reference sqrt.rn");
  std::bitset<256> result;
  for (unsigned n = 0; n < 255; ++n) {
    bool all = true;
    for (const auto& [k, v] : ref) all = all && renamed(reg_value(*by_key[k], n), x) == v;
    result[n] = all;
  }
  if (result.none()) return fail("no register holds the square root at the join");
  // every other value that depends on a MUFU must be dead at the join
  for (const Path& s : r.paths) {
    for (const auto& [n, v] : s.r)
      if (!result.test(n) && v.find("MUFU") != std::string::npos && may_be_read(c, static_cast<size_t>(join), false, n)) {
        st.live = static_cast<int>(n);
        return fail("a register holding an unrefined MUFU value is read after the idiom");
      }
    for (const auto& [n, v] : s.p)
      if (v.find("MUFU") != std::string::npos && may_be_read(c, static_cast<size_t>(join), true, n)) {
        st.live = 256 + static_cast<int>(n);
        return fail("a predicate computed from a MUFU value is read after the idiom");
      }
  }
  // the region and the subroutine are entered only through the site's own instructions
  for (const Path& s : r.paths) st.covered.insert(s.visited.begin(), s.visited.end());
  long callee = -1, callee_end = -1;
  for (size_t i : st.covered)
    if (c.ins[i].mnemonic == "CALL") callee = c.index(label_of(c.ins[i]));
  for (size_t i : st.covered)
    if (c.ins[i].mnemonic == "RET") callee_end = static_cast<long>(i);
  const uint64_t lo = c.ins[start].pc, hi = c.ins[static_cast<size_t>(join)].pc;
  for (size_t i = 0; i < c.ins.size(); ++i) {
    const std::string& mn = c.ins[i].mnemonic;
    if ((mn != "BRA" && mn != "CALL" && mn != "BRX") || st.covered.count(i)) continue;
    const int64_t t = label_of(c.ins[i]);
    if (t > static_cast<int64_t>(lo) && t < static_cast<int64_t>(hi)) return fail("a branch from outside lands inside the idiom");
    // (another call of the subroutine's entry is another site's: the kernel is admitted only when every CALL belongs to a recognised site)
    const int64_t entry = callee >= 0 ? static_cast<int64_t>(c.ins[static_cast<size_t>(callee)].pc) : -1;
    if (callee >= 0 && callee_end >= 0 && t >= entry && t <= static_cast<int64_t>(c.ins[static_cast<size_t>(callee_end)].pc) && !(mn == "CALL" && t == entry))
      return fail("a branch or call from outside lands inside the slow path");
  }
  if (callee > 0) {
    const Instr& before = c.ins[static_cast<size_t>(callee) - 1];
    const bool falls = !(unconditional(before) && (before.mnemonic == "BRA" || before.mnemonic == "EXIT" || before.mnemonic == "RET"));
    if (falls) return fail("code before the slow path falls into it");
  }
  st.ok = true;
  st.start = start;
  st.join = static_cast<size_t>(join);
  st.callee = callee;
  st.callee_end = callee_end;
  return st;
}

// The division idiom's summary of a kernel (host_gpu.hpp division_idiom_only), with the instructions in `skip` (the sqrt sites') made ordinary.
inline std::vector<CodeInsn> division_summary(const std::vector<Instr>& code, const std::vector<char>& skip) {
  std::vector<CodeInsn> out;
  out.reserve(code.size());
  for (size_t i = 0; i < code.size(); ++i) {
    const Instr& ins = code[i];
    CodeInsn c;
    const bool nan_const = ins.mnemonic == "MUFU" && !ins.src.empty() && ins.src[0].kind == Kind::FImm && ins.src[0].fwidth == 32 &&
                           (ins.src[0].fbits & 0x7f800000u) == 0x7f800000u && (ins.src[0].fbits & 0x7fffffu) != 0;
    c.kind = skip[i] ? CodeInsn::Kind::Other : classify_for_idiom(ins.mnemonic, ins.mods, nan_const);
    c.pc = ins.pc;
    const auto reg_of = [&](size_t k) { return ins.src.size() > k && ins.src[k].kind == Kind::Reg && !ins.src[k].neg && !ins.src[k].abs ? static_cast<int>(ins.src[k].reg) : -1; };
    if (c.kind == CodeInsn::Kind::Call && !ins.src.empty() && ins.src[0].kind == Kind::Label) c.target = ins.src[0].imm;
    if (c.kind == CodeInsn::Kind::MufuRcp) c.src0 = reg_of(0);
    if (c.kind == CodeInsn::Kind::Fchk) { c.src0 = reg_of(0); c.src1 = reg_of(1); }
    out.push_back(c);
  }
  return out;
}

struct KernelVerdict {
  bool ok = false;
  const char* why = "";
  unsigned sqrt_sites = 0;
};

// Is every instruction of the kernel outside the exact set part of a recognised idiom? `sqrt`: VGPU_HOST_GPU_SQRT, `div`: VGPU_HOST_GPU_DIV. `why` names the
// first square root site that failed when one did, else what else kept the kernel out.
inline KernelVerdict idioms_only(const std::vector<Instr>& code, bool div, bool sqrt) {
  KernelVerdict v;
  const Code c(code);
  std::vector<char> covered(code.size(), 0);
  std::vector<std::pair<int64_t, int64_t>> callees;   // the sqrt slow paths, [first pc, RET pc]
  const char* site_why = "";
  if (sqrt)
    for (size_t i = 0; i < code.size(); ++i) {
      const Instr& mu = code[i];
      if (covered[i] || mu.mnemonic != "MUFU" || mu.mods.size() != 1 || mu.mods[0] != "RSQ" || mu.src.size() != 1 || mu.src[0].kind != Kind::Reg) continue;
      const Site s = sqrt_site(c, i);
      if (!s.ok) {
        if (!*site_why) site_why = s.why;
        continue;
      }
      ++v.sqrt_sites;
      for (size_t k : s.covered) covered[k] = 1;
      if (s.callee >= 0 && s.callee_end >= 0) callees.emplace_back(code[static_cast<size_t>(s.callee)].pc, code[static_cast<size_t>(s.callee_end)].pc);
    }
  bool need_div = false;
  for (size_t i = 0; i < code.size(); ++i) {
    if (covered[i] || exact_opcode_allowed(code[i].mnemonic, code[i].mods)) continue;
    const std::string& mn = code[i].mnemonic;
    if (div && (mn == "MUFU" || mn == "CALL" || mn == "RET")) {
      need_div = true;
      continue;
    }
    v.why = *site_why ? site_why : mn == "MUFU" ? "an SFU instruction that no idiom explains" : "an instruction outside the exact set";
    return v;
  }
  if (need_div) {
    // a call the division idiom takes must not go into a square root's slow path
    for (size_t i = 0; i < code.size(); ++i)
      if (!covered[i] && code[i].mnemonic == "CALL")
        for (const auto& [lo, hi] : callees)
          if (label_of(code[i]) >= lo && label_of(code[i]) <= hi) {
            v.why = "a call into a square root's slow path from outside its site";
            return v;
          }
    if (!division_idiom_only(division_summary(code, covered), &v.why)) {
      if (*site_why) v.why = site_why;
      return v;
    }
  }
  v.ok = true;
  return v;
}

}  // namespace vgpu::hostgpu::idiom
