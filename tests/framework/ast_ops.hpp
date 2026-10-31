// Small constructors for AST instructions, so a test that builds a kernel with ptx::KernelBuilder reads as a
// kernel and not as a list of aggregate initialisers whose field order has to be remembered.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "vgpu/ptx/ast.hpp"

namespace astops {

using namespace vgpu::ptx;

inline Type u32() { return Type{Type::Kind::U, 32}; }
inline Type u64() { return Type{Type::Kind::U, 64}; }
inline Type f32() { return Type{Type::Kind::F, 32}; }

inline Operand RO(const Reg& r) { return RegOperand{r}; }
inline Operand IM(int64_t v) { return ImmInt{v}; }
inline Operand F32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return ImmFloatBits{u, 32};
}
inline Operand SR(Sreg s) { return SregOperand{s, 0}; }
inline Operand SYM(const char* name) { return SymbolOperand{name}; }

inline OpMov mov(Type ty, const Reg& d, Operand s) {
  OpMov o;
  o.ty = ty;
  o.dst = d;
  o.src = std::move(s);
  return o;
}
inline OpLd ld(Space sp, Type ty, const Reg& d, const Addr& a) {
  OpLd o;
  o.space = sp;
  o.ty = ty;
  o.dsts = {d};
  o.addr = a;
  return o;
}
inline OpSt st(Space sp, Type ty, const Addr& a, Operand v) {
  OpSt o;
  o.space = sp;
  o.ty = ty;
  o.addr = a;
  o.srcs = {std::move(v)};
  return o;
}
inline OpIntBin ibin(IntBinOp op, Type ty, const Reg& d, Operand a, Operand b) {
  OpIntBin o;
  o.op = op;
  o.ty = ty;
  o.dst = d;
  o.a = std::move(a);
  o.b = std::move(b);
  return o;
}
inline OpMadLo madlo(Type ty, const Reg& d, Operand a, Operand b, Operand c) {
  OpMadLo o;
  o.ty = ty;
  o.dst = d;
  o.a = std::move(a);
  o.b = std::move(b);
  o.c = std::move(c);
  return o;
}
inline OpMulWide mulwide(bool is_signed, const Reg& d, Operand a, Operand b) {
  OpMulWide o;
  o.src_bits = 32;
  o.is_signed = is_signed;
  o.dst = d;
  o.a = std::move(a);
  o.b = std::move(b);
  return o;
}
inline OpSetp setp(CmpOp cmp, Type ty, const Reg& p, Operand a, Operand b) {
  OpSetp o;
  o.cmp = cmp;
  o.ty = ty;
  o.dst = p;
  o.a = std::move(a);
  o.b = std::move(b);
  return o;
}
inline OpFloatBin fbin(FloatBinOp op, const Reg& d, Operand a, Operand b) {
  OpFloatBin o;
  o.op = op;
  o.ty = Type{Type::Kind::F, 32};
  o.dst = d;
  o.a = std::move(a);
  o.b = std::move(b);
  return o;
}
inline OpBar barsync(int id) {
  OpBar o;
  o.id = ImmInt{id};
  return o;
}

}  // namespace astops
