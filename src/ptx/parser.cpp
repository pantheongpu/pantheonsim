// PTX parser for the VirtualGPU subset.
//
// Anything outside the subset throws Err::UnsupportedPtx naming the exact
// instruction, source line, and kernel — never a silent wrong answer.
#include "vgpu/ptx/parser.hpp"

#include <cstdlib>
#include <optional>
#include <unordered_map>

#include "lexer.hpp"
#include "vgpu/error.hpp"

namespace vgpu::ptx {
namespace {

const std::unordered_map<std::string, Sreg>& sreg_table() {
  static const std::unordered_map<std::string, Sreg> t = {
      {"%tid.x", Sreg::TidX},       {"%tid.y", Sreg::TidY},       {"%tid.z", Sreg::TidZ},
      {"%ntid.x", Sreg::NtidX},     {"%ntid.y", Sreg::NtidY},     {"%ntid.z", Sreg::NtidZ},
      {"%ctaid.x", Sreg::CtaidX},   {"%ctaid.y", Sreg::CtaidY},   {"%ctaid.z", Sreg::CtaidZ},
      {"%nctaid.x", Sreg::NctaidX}, {"%nctaid.y", Sreg::NctaidY}, {"%nctaid.z", Sreg::NctaidZ},
      {"%laneid", Sreg::LaneId},
  };
  return t;
}

std::vector<std::string> split_dots(const std::string& s) {
  std::vector<std::string> parts;
  std::string cur;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '.') {
      if (!cur.empty()) parts.push_back(cur);
      cur.clear();
    } else {
      cur += s[i];
    }
  }
  if (!cur.empty()) parts.push_back(cur);
  return parts;
}

std::optional<Type> parse_type_token(const std::string& part) {
  if (part == "pred") return Type{Type::Kind::Pred, 1};
  if (part.size() < 2) return std::nullopt;
  Type::Kind kind;
  switch (part[0]) {
    case 'b': kind = Type::Kind::B; break;
    case 'u': kind = Type::Kind::U; break;
    case 's': kind = Type::Kind::S; break;
    case 'f': kind = Type::Kind::F; break;
    default: return std::nullopt;
  }
  for (size_t i = 1; i < part.size(); ++i)
    if (!isdigit(static_cast<unsigned char>(part[i]))) return std::nullopt;
  uint32_t bits = static_cast<uint32_t>(std::atoi(part.c_str() + 1));
  if (bits != 8 && bits != 16 && bits != 32 && bits != 64) return std::nullopt;
  return Type{kind, bits};
}

class Parser {
 public:
  explicit Parser(const std::string& src) : toks_(lex(src)) {}

  Module parse_module() {
    Module m;
    while (!at_end()) {
      const Token& t = peek();
      if (t.kind != Token::Kind::Word) fail(t.line, "expected directive, got '" + t.text + "'");
      if (t.text == ".version") {
        next();
        m.version = expect_word("PTX version");
        continue;
      }
      if (t.text == ".target") {
        next();
        m.target = expect_word("target");
        while (peek_punct(",")) {  // e.g. ".target sm_90, debug"
          next();
          m.target += "," + expect_word("target option");
        }
        continue;
      }
      if (t.text == ".address_size") {
        next();
        std::string sz = expect_word("address size");
        m.address_size = static_cast<uint32_t>(std::atoi(sz.c_str()));
        if (m.address_size != 64)
          fail_unsupported(t.line, ".address_size " + sz, "", "only 64-bit PTX is supported");
        continue;
      }
      if (t.text == ".visible" || t.text == ".weak") {
        next();
        continue;  // linkage qualifier before .entry
      }
      if (t.text == ".entry") {
        next();
        m.entries.push_back(parse_entry(t.line));
        continue;
      }
      fail_unsupported(t.line, t.text, "",
                       "directive not in the implemented PTX subset (supported: .version .target "
                       ".address_size .visible .entry)");
    }
    return m;
  }

 private:
  [[noreturn]] void fail(size_t line, const std::string& msg) {
    throw Error::make(Err::PtxParse, "line ", line, ": ", msg);
  }

  [[noreturn]] void fail_unsupported(size_t line, const std::string& what, const std::string& kernel,
                                     const std::string& hint) {
    std::string k = kernel.empty() ? current_kernel_ : kernel;
    throw Error::make(Err::UnsupportedPtx, "unsupported PTX:\n  ", what, "\nat line ", line,
                      k.empty() ? "" : " in kernel '" + k + "'", hint.empty() ? "" : "\n  (" + hint + ")");
  }

  bool at_end() const { return toks_[pos_].kind == Token::Kind::End; }
  const Token& peek(size_t off = 0) const { return toks_[std::min(pos_ + off, toks_.size() - 1)]; }
  const Token& next() { return toks_[pos_++]; }

  bool peek_punct(const std::string& p, size_t off = 0) const {
    return peek(off).kind == Token::Kind::Punct && peek(off).text == p;
  }

  void expect_punct(const std::string& p) {
    const Token& t = next();
    if (t.kind != Token::Kind::Punct || t.text != p)
      fail(t.line, "expected '" + p + "', got '" + t.text + "'");
  }

  std::string expect_word(const std::string& what) {
    const Token& t = next();
    if (t.kind != Token::Kind::Word) fail(t.line, "expected " + what + ", got '" + t.text + "'");
    return t.text;
  }

  Type expect_type(const std::string& ctx) {
    const Token& t = next();
    if (t.kind == Token::Kind::Word && t.text.size() > 1 && t.text[0] == '.') {
      if (auto ty = parse_type_token(t.text.substr(1))) return *ty;
    }
    fail(t.line, "expected a type in " + ctx + ", got '" + t.text + "'");
  }

  EntryFn parse_entry(size_t line) {
    EntryFn fn;
    fn.name = expect_word("kernel name");
    current_kernel_ = fn.name;
    if (peek_punct("(")) {
      next();
      while (!peek_punct(")")) {
        std::string kw = expect_word("'.param'");
        if (kw != ".param") fail(peek().line, "expected .param, got '" + kw + "'");
        Type ty = expect_type("parameter declaration");
        // Skip optional pointer annotations: .ptr .global .align N
        while (peek().kind == Token::Kind::Word && peek().text[0] == '.') {
          std::string ann = next().text;
          if (ann == ".align") (void)expect_word("alignment");
          // .ptr / .global / .const etc.: functional no-ops for us
        }
        ParamDecl p;
        p.name = expect_word("parameter name");
        p.ty = ty;
        if (peek_punct("[")) fail_unsupported(peek().line, ".param array", fn.name, "array parameters");
        fn.params.push_back(std::move(p));
        if (peek_punct(",")) next();
      }
      expect_punct(")");
    }
    expect_punct("{");
    parse_body(fn);
    current_kernel_.clear();
    (void)line;
    return fn;
  }

  void parse_body(EntryFn& fn) {
    std::unordered_map<std::string, size_t> labels;
    std::vector<std::pair<size_t, std::string>> bra_fixups;  // instr idx -> label

    while (true) {
      const Token& t = peek();
      if (t.kind == Token::Kind::End) fail(t.line, "unexpected end of file inside kernel body");
      if (peek_punct("}")) {
        next();
        break;
      }
      if (t.kind == Token::Kind::Word && t.text == ".reg") {
        next();
        parse_reg_decl(fn);
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text[0] == '.') {
        fail_unsupported(t.line, t.text, fn.name,
                         t.text == ".shared" || t.text == ".local"
                             ? "shared/local memory declarations are not implemented yet"
                             : "directive not in the implemented subset");
      }
      // Label?
      if (t.kind == Token::Kind::Word && peek_punct(":", 1)) {
        if (!labels.emplace(t.text, fn.body.size()).second)
          fail(t.line, "duplicate label '" + t.text + "'");
        next();
        next();
        continue;
      }
      fn.body.push_back(parse_instruction(fn, bra_fixups));
    }

    for (auto& [idx, label] : bra_fixups) {
      auto it = labels.find(label);
      if (it == labels.end())
        fail(fn.body[idx].line, "branch to undefined label '" + label + "' in kernel '" + fn.name + "'");
      std::get<OpBra>(fn.body[idx].op).target = it->second;
    }
  }

  void parse_reg_decl(EntryFn& fn) {
    Type ty = expect_type(".reg declaration");
    while (true) {
      std::string name = expect_word("register name");
      if (peek_punct("<")) {  // parameterized: .reg .b32 %r<6> declares %r0..%r5
        next();
        std::string count = expect_word("register count");
        expect_punct(">");
        int n = std::atoi(count.c_str());
        for (int i = 0; i < n; ++i) fn.reg_decls[name + std::to_string(i)] = ty;
      } else {
        fn.reg_decls[name] = ty;
      }
      if (peek_punct(",")) {
        next();
        continue;
      }
      break;
    }
    expect_punct(";");
  }

  // ---- operands ----

  int64_t parse_int_literal(const std::string& w, size_t line) {
    try {
      if (w.size() > 2 && w[0] == '0' && (w[1] == 'x' || w[1] == 'X'))
        return static_cast<int64_t>(std::stoull(w.substr(2), nullptr, 16));
      return std::stoll(w);
    } catch (const std::exception&) {
      fail(line, "bad integer literal '" + w + "'");
    }
  }

  Operand parse_operand(const EntryFn& fn) {
    const Token& t = next();
    if (t.kind == Token::Kind::Punct && t.text == "-") {
      std::string w = expect_word("number after '-'");
      return ImmInt{-parse_int_literal(w, t.line)};
    }
    if (t.kind != Token::Kind::Word) fail(t.line, "expected operand, got '" + t.text + "'");
    const std::string& w = t.text;
    if (w[0] == '%') {
      auto it = sreg_table().find(w);
      if (it != sreg_table().end()) return SregOperand{it->second};
      return RegOperand{w};
    }
    if (isdigit(static_cast<unsigned char>(w[0]))) {
      // Float literals: 0fXXXXXXXX (f32 bits), 0dXXXXXXXXXXXXXXXX (f64 bits).
      if (w.size() == 10 && w[0] == '0' && (w[1] == 'f' || w[1] == 'F'))
        return ImmFloatBits{std::stoull(w.substr(2), nullptr, 16), 32};
      if (w.size() == 18 && w[0] == '0' && (w[1] == 'd' || w[1] == 'D'))
        return ImmFloatBits{std::stoull(w.substr(2), nullptr, 16), 64};
      return ImmInt{parse_int_literal(w, t.line)};
    }
    (void)fn;
    fail(t.line, "cannot parse operand '" + w + "'");
  }

  std::string expect_reg_operand(const std::string& ctx) {
    std::string w = expect_word(ctx);
    if (w[0] != '%') fail(peek().line, ctx + " must be a register, got '" + w + "'");
    return w;
  }

  Addr parse_addr(const EntryFn& fn) {
    expect_punct("[");
    Addr a;
    std::string base = expect_word("address base");
    if (base[0] == '%') {
      a.base = base;
      a.base_is_param = false;
    } else {
      // Parameter name (ld.param [pname]) — validate it exists.
      bool found = false;
      for (const auto& p : fn.params) found = found || p.name == base;
      if (!found) fail(peek().line, "unknown parameter '" + base + "' in address operand");
      a.base = base;
      a.base_is_param = true;
    }
    if (peek_punct("+")) {
      next();
      bool neg = false;
      if (peek_punct("-")) {
        next();
        neg = true;
      }
      std::string off = expect_word("address offset");
      a.offset = parse_int_literal(off, peek().line);
      if (neg) a.offset = -a.offset;
    }
    expect_punct("]");
    return a;
  }

  // ---- instructions ----

  Instr parse_instruction(const EntryFn& fn, std::vector<std::pair<size_t, std::string>>& bra_fixups) {
    Instr ins;
    size_t start_tok = pos_;
    ins.line = peek().line;

    if (peek_punct("@")) {
      next();
      ins.has_pred = true;
      if (peek_punct("!")) {
        next();
        ins.pred_negated = true;
      }
      ins.pred = expect_reg_operand("predicate register");
    }

    std::string opcode = expect_word("instruction opcode");
    std::vector<std::string> parts = split_dots(opcode);
    if (parts.empty()) fail(ins.line, "bad opcode '" + opcode + "'");

    auto unsupported = [&](const std::string& hint) -> Instr {
      fail_unsupported(ins.line, reconstruct_from(start_tok), fn.name, hint);
    };

    const std::string& op0 = parts[0];
    if (op0 == "ld" || op0 == "st") {
      Space space = Space::Generic;
      bool vec = false;
      Type ty{};
      bool have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "param") space = Space::Param;
        else if (p == "global") space = Space::Global;
        else if (p == "shared") space = Space::Shared;
        else if (p == "local") space = Space::Local;
        else if (p == "volatile" || p == "nc" || p == "relaxed" || p == "acquire" || p == "release" ||
                 p == "gpu" || p == "cta" || p == "sys")
          ;  // memory-order/scope/cache hints: functionally inert for our engine today
        else if (p == "v2" || p == "v4") vec = true;
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("unrecognized ld/st modifier '." + p + "'");
      }
      if (vec) return unsupported("vector ld/st (v2/v4) not implemented yet");
      if (!have_ty) fail(ins.line, "ld/st missing type: " + opcode);
      if (space == Space::Shared || space == Space::Local)
        return unsupported("shared/local memory not implemented yet");
      if (op0 == "ld") {
        OpLd op;
        op.space = space;
        op.ty = ty;
        op.dst = expect_reg_operand("ld destination");
        expect_punct(",");
        op.addr = parse_addr(fn);
        ins.op = op;
      } else {
        if (space == Space::Param) return unsupported("st.param is only used in device functions");
        OpSt op;
        op.space = space;
        op.ty = ty;
        op.addr = parse_addr(fn);
        expect_punct(",");
        op.src = parse_operand(fn);
        ins.op = op;
      }
    } else if (op0 == "mov") {
      if (parts.size() != 2) return unsupported("mov form");
      auto ty = parse_type_token(parts[1]);
      if (!ty) fail(ins.line, "mov missing type");
      OpMov op;
      op.ty = *ty;
      op.dst = expect_reg_operand("mov destination");
      expect_punct(",");
      op.src = parse_operand(fn);
      ins.op = op;
    } else if (op0 == "cvta") {
      // cvta.to.global.u64 (generic->global) and cvta.global.u64 (global->generic):
      // both are identity in VirtualGPU's flat address space.
      size_t i = 1;
      if (i < parts.size() && parts[i] == "to") ++i;
      if (i >= parts.size() || parts[i] != "global") return unsupported("only cvta[.to].global supported");
      ++i;
      auto ty = (i < parts.size()) ? parse_type_token(parts[i]) : std::nullopt;
      if (!ty) fail(ins.line, "cvta missing type");
      OpCvtaToGlobal op;
      op.ty = *ty;
      op.dst = expect_reg_operand("cvta destination");
      expect_punct(",");
      op.src = parse_operand(fn);
      ins.op = op;
    } else if (op0 == "add" || op0 == "sub" || op0 == "mul" || op0 == "min" || op0 == "max" ||
               op0 == "div" || op0 == "rem" || op0 == "and" || op0 == "or" || op0 == "xor" ||
               op0 == "shl" || op0 == "shr") {
      bool wide = false, lo = false, hi = false;
      Type ty{};
      bool have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "wide") wide = true;
        else if (p == "lo") lo = true;
        else if (p == "hi") hi = true;
        else if (p == "rn" || p == "ftz") ;  // round-to-nearest is our only float mode
        else if (p == "rz" || p == "rm" || p == "rp")
          return unsupported("non-default float rounding mode '." + p + "'");
        else if (p == "sat" || p == "approx" || p == "full")
          return unsupported("modifier '." + p + "' not implemented");
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("unrecognized modifier '." + p + "'");
      }
      if (!have_ty) fail(ins.line, opcode + " missing type");
      if (hi) return unsupported("mul.hi not implemented yet");
      if (ty.is_float()) {
        if (ty.bits != 32 && ty.bits != 64) return unsupported("only f32/f64 float math implemented");
        static const std::unordered_map<std::string, FloatBinOp> fops = {
            {"add", FloatBinOp::Add}, {"sub", FloatBinOp::Sub}, {"mul", FloatBinOp::Mul},
            {"min", FloatBinOp::Min}, {"max", FloatBinOp::Max}, {"div", FloatBinOp::Div}};
        auto it = fops.find(op0);
        if (it == fops.end()) return unsupported("float op '" + op0 + "'");
        OpFloatBin op;
        op.op = it->second;
        op.ty = ty;
        op.dst = expect_reg_operand("destination");
        expect_punct(",");
        op.a = parse_operand(fn);
        expect_punct(",");
        op.b = parse_operand(fn);
        ins.op = op;
      } else if (wide) {
        if (op0 != "mul" || ty.bits != 32) return unsupported("only mul.wide.{s32,u32} implemented");
        OpMulWide op;
        op.is_signed = ty.is_signed();
        op.dst = expect_reg_operand("destination");
        expect_punct(",");
        op.a = parse_operand(fn);
        expect_punct(",");
        op.b = parse_operand(fn);
        ins.op = op;
      } else {
        static const std::unordered_map<std::string, IntBinOp> iops = {
            {"add", IntBinOp::Add}, {"sub", IntBinOp::Sub}, {"mul", IntBinOp::Mul},
            {"min", IntBinOp::Min}, {"max", IntBinOp::Max}, {"div", IntBinOp::Div},
            {"rem", IntBinOp::Rem}, {"and", IntBinOp::And}, {"or", IntBinOp::Or},
            {"xor", IntBinOp::Xor}, {"shl", IntBinOp::Shl}, {"shr", IntBinOp::Shr}};
        auto it = iops.find(op0);
        if (it == iops.end()) return unsupported("integer op '" + op0 + "'");
        if (op0 == "mul" && !lo) return unsupported("plain mul on integers requires .lo/.wide/.hi");
        OpIntBin op;
        op.op = it->second;
        op.ty = ty;
        op.dst = expect_reg_operand("destination");
        expect_punct(",");
        op.a = parse_operand(fn);
        expect_punct(",");
        op.b = parse_operand(fn);
        ins.op = op;
      }
    } else if (op0 == "mad" || op0 == "fma") {
      Type ty{};
      bool have_ty = false, lo = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "lo") lo = true;
        else if (p == "rn" || p == "ftz") ;
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("unrecognized modifier '." + p + "'");
      }
      if (!have_ty) fail(ins.line, opcode + " missing type");
      std::string dst = expect_reg_operand("destination");
      expect_punct(",");
      Operand a = parse_operand(fn);
      expect_punct(",");
      Operand b = parse_operand(fn);
      expect_punct(",");
      Operand c = parse_operand(fn);
      if (ty.is_float()) {
        if (ty.bits != 32 && ty.bits != 64) return unsupported("only f32/f64 fma implemented");
        ins.op = OpFma{ty, dst, a, b, c};
      } else {
        if (!lo) return unsupported("integer mad requires .lo");
        ins.op = OpMadLo{ty, dst, a, b, c};
      }
    } else if (op0 == "setp") {
      if (parts.size() != 3) return unsupported("setp form (only setp.<cmp>.<type> is implemented)");
      static const std::unordered_map<std::string, CmpOp> cmps = {
          {"eq", CmpOp::Eq}, {"ne", CmpOp::Ne}, {"lt", CmpOp::Lt},
          {"le", CmpOp::Le}, {"gt", CmpOp::Gt}, {"ge", CmpOp::Ge}};
      auto it = cmps.find(parts[1]);
      if (it == cmps.end()) return unsupported("comparison '." + parts[1] + "'");
      auto ty = parse_type_token(parts[2]);
      if (!ty) fail(ins.line, "setp missing type");
      OpSetp op;
      op.cmp = it->second;
      op.ty = *ty;
      op.dst = expect_reg_operand("predicate destination");
      expect_punct(",");
      op.a = parse_operand(fn);
      expect_punct(",");
      op.b = parse_operand(fn);
      ins.op = op;
    } else if (op0 == "selp") {
      if (parts.size() != 2) return unsupported("selp form");
      auto ty = parse_type_token(parts[1]);
      if (!ty) fail(ins.line, "selp missing type");
      OpSelp op;
      op.ty = *ty;
      op.dst = expect_reg_operand("destination");
      expect_punct(",");
      op.a = parse_operand(fn);
      expect_punct(",");
      op.b = parse_operand(fn);
      expect_punct(",");
      op.pred = expect_reg_operand("selp predicate");
      ins.op = op;
    } else if (op0 == "bra") {
      // ".uni" (uniform hint) is accepted and ignored.
      std::string label = expect_word("branch target");
      bra_fixups.emplace_back(fn.body.size(), label);
      ins.op = OpBra{0, label};
    } else if (op0 == "bar" || op0 == "barrier") {
      bool sync_seen = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] == "sync") sync_seen = true;
        else if (parts[i] == "cta") ;
        else return unsupported("only bar.sync is implemented");
      }
      if (!sync_seen) return unsupported("only bar.sync is implemented");
      Operand which = parse_operand(fn);
      if (auto* imm = std::get_if<ImmInt>(&which); !imm || imm->value != 0)
        return unsupported("only barrier 0 (bar.sync 0) is implemented");
      if (peek_punct(",")) return unsupported("partial barriers (bar.sync 0, N) not implemented");
      ins.op = OpBar{};
    } else if (op0 == "ret" || op0 == "exit") {
      // ".uni" hint ignored.
      ins.op = OpRet{};
    } else {
      return unsupported("instruction not in the implemented PTX subset");
    }

    expect_punct(";");
    ins.text = reconstruct_from(start_tok);
    return ins;
  }

  // Rebuilds source-ish text from tokens for error messages. Joins the tokens
  // consumed so far plus (for mid-statement failures) the un-consumed remainder,
  // stopping at the statement's own ';'.
  std::string reconstruct_from(size_t start_tok) {
    std::string out;
    for (size_t i = start_tok; i < toks_.size(); ++i) {
      if (toks_[i].kind == Token::Kind::End || toks_[i].text == ";") break;
      if (!out.empty() && toks_[i].kind == Token::Kind::Word && out.back() != '@' && out.back() != '[')
        out += ' ';
      out += toks_[i].text;
    }
    return out;
  }

  std::vector<Token> toks_;
  size_t pos_ = 0;
  std::string current_kernel_;
};

}  // namespace

Module parse(const std::string& src) {
  Parser p(src);
  Module m = p.parse_module();
  if (m.entries.empty())
    throw Error::make(Err::PtxParse, "no .entry kernels found in PTX module");
  return m;
}

}  // namespace vgpu::ptx
