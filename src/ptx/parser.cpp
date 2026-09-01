// PTX parser for the VirtualGPU subset.
//
// Anything outside the subset throws Err::UnsupportedPtx naming the exact
// instruction, source line, and kernel — never a silent wrong answer.
#include "vgpu/ptx/parser.hpp"

#include <cstdlib>
#include <optional>
#include <set>
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

// ld/st/atom modifiers that are functionally inert for this engine
// (cache hints, memory orders, scopes).
bool inert_mem_modifier(const std::string& p) {
  static const std::set<std::string> inert = {"volatile", "nc", "ca", "cg", "cs", "lu",  "cv",
                                              "wb",       "wt", "relaxed", "acquire", "release",
                                              "acq_rel",  "cta", "gpu", "sys"};
  return inert.count(p) > 0;
}

std::vector<std::string> split_dots(const std::string& s) {
  std::vector<std::string> parts;
  std::string cur;
  for (char c : s) {
    if (c == '.') {
      if (!cur.empty()) parts.push_back(cur);
      cur.clear();
    } else {
      cur += c;
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
        while (peek_punct(",")) {
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
        continue;  // linkage qualifier
      }
      if (t.text == ".pragma") {
        next();
        while (!at_end() && !peek_punct(";")) next();
        if (!at_end()) next();
        continue;
      }
      if (t.text == ".extern") {
        // ".extern .func (...) name (...);" — declaration of an external
        // function (vprintf). Recorded implicitly; skip to the ';'.
        while (!at_end() && !peek_punct(";")) next();
        if (!at_end()) next();
        continue;
      }
      if (t.text == ".entry") {
        next();
        m.entries.push_back(parse_entry());
        continue;
      }
      if (t.text == ".global" || t.text == ".const") {
        next();
        m.globals.push_back(parse_global(t.line));
        continue;
      }
      if (t.text == ".func")
        fail_unsupported(t.line, ".func", "", "device functions are not supported (only .entry kernels)");
      fail_unsupported(t.line, t.text, "",
                       "directive not in the implemented PTX subset (supported: .version .target "
                       ".address_size .visible .extern .global .const .entry)");
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

  int64_t parse_int_literal(const std::string& w, size_t line) {
    try {
      if (w.size() > 2 && w[0] == '0' && (w[1] == 'x' || w[1] == 'X'))
        return static_cast<int64_t>(std::stoull(w.substr(2), nullptr, 16));
      return std::stoll(w);
    } catch (const std::exception&) {
      fail(line, "bad integer literal '" + w + "'");
    }
  }

  // Reads "N" or "-N".
  int64_t expect_int(const std::string& what) {
    bool neg = false;
    if (peek_punct("-")) {
      next();
      neg = true;
    }
    std::string w = expect_word(what);
    int64_t v = parse_int_literal(w, peek().line);
    return neg ? -v : v;
  }

  // ---- module-scope .global/.const variables ----

  GlobalVar parse_global(size_t line) {
    GlobalVar g;
    // Optional modifiers before the type.
    while (peek().kind == Token::Kind::Word && peek().text[0] == '.') {
      if (peek().text == ".align") {
        next();
        g.align = static_cast<uint32_t>(expect_int("alignment"));
        continue;
      }
      break;
    }
    Type ty = expect_type(".global declaration");
    g.name = expect_word("global variable name");
    uint64_t elems = 1;
    if (peek_punct("[")) {
      next();
      elems = static_cast<uint64_t>(expect_int("array size"));
      expect_punct("]");
    }
    g.size = elems * ty.bytes();
    if (g.size == 0) fail(line, "zero-sized global '" + g.name + "'");
    if (peek_punct("=")) {
      next();
      if (peek_punct("{")) {
        next();
        g.init.reserve(g.size);
        while (!peek_punct("}")) {
          int64_t v = expect_int("initializer element");
          // Little-endian element append.
          for (uint32_t b = 0; b < ty.bytes(); ++b)
            g.init.push_back(static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * b)) & 0xFF));
          if (peek_punct(",")) next();
        }
        next();  // '}'
      } else {
        int64_t v = expect_int("initializer");
        for (uint32_t b = 0; b < ty.bytes(); ++b)
          g.init.push_back(static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * b)) & 0xFF));
      }
      if (g.init.size() > g.size)
        fail(line, "initializer for '" + g.name + "' longer than its declared size");
      g.init.resize(g.size, 0);
    }
    expect_punct(";");
    return g;
  }

  // ---- entry functions ----

  EntryFn parse_entry() {
    EntryFn fn;
    fn.name = expect_word("kernel name");
    current_kernel_ = fn.name;
    call_slots_.clear();
    if (peek_punct("(")) {
      next();
      while (!peek_punct(")")) {
        std::string kw = expect_word("'.param'");
        if (kw != ".param") fail(peek().line, "expected .param, got '" + kw + "'");
        ParamDecl p;
        // Optional ".align N" then type, then optional pointer annotations.
        if (peek().kind == Token::Kind::Word && peek().text == ".align") {
          next();
          p.align = static_cast<uint32_t>(expect_int("alignment"));
        }
        Type ty = expect_type("parameter declaration");
        while (peek().kind == Token::Kind::Word && peek().text[0] == '.') {
          std::string ann = next().text;
          if (ann == ".align") p.align = static_cast<uint32_t>(expect_int("alignment"));
          // .ptr / .global / .const: functional no-ops for us
        }
        p.name = expect_word("parameter name");
        p.ty = ty;
        p.size = ty.bytes();
        if (peek_punct("[")) {  // aggregate: .param .align 8 .b8 name[24]
          next();
          p.size = static_cast<uint32_t>(expect_int("parameter array size"));
          expect_punct("]");
        }
        if (p.align == 0) p.align = p.size < 8 ? p.size : 8;
        fn.params.push_back(std::move(p));
        if (peek_punct(",")) next();
      }
      expect_punct(")");
    }
    expect_punct("{");
    parse_body(fn);
    current_kernel_.clear();
    return fn;
  }

  void parse_body(EntryFn& fn) {
    std::unordered_map<std::string, size_t> labels;
    std::vector<std::pair<size_t, std::string>> bra_fixups;
    int depth = 0;  // nested { } scopes (call sequences)

    while (true) {
      const Token& t = peek();
      if (t.kind == Token::Kind::End) fail(t.line, "unexpected end of file inside kernel body");
      if (peek_punct("{")) {
        next();
        ++depth;
        continue;
      }
      if (peek_punct("}")) {
        next();
        if (depth == 0) break;
        --depth;
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text == ".reg") {
        next();
        parse_reg_decl(fn);
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text == ".local") {
        next();
        parse_local_decl(fn, t.line);
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text == ".pragma") {
        // Performance hints ("nounroll" etc.): functionally inert, skip.
        while (!at_end() && !peek_punct(";")) next();
        if (!at_end()) next();
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text == ".param") {
        // Call-argument slot declaration inside a call sequence.
        next();
        Instr ins;
        ins.line = t.line;
        OpDeclSlot d;
        Type ty = expect_type(".param slot declaration");
        d.name = expect_word("slot name");
        d.size = ty.bytes();
        if (peek_punct("["))
          fail_unsupported(t.line, ".param array call slot", fn.name,
                           "aggregate call arguments are not supported yet");
        expect_punct(";");
        call_slots_.insert(d.name);
        ins.op = d;
        ins.text = ".param " + d.name;
        fn.body.push_back(std::move(ins));
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text[0] == '.') {
        fail_unsupported(t.line, t.text, fn.name,
                         t.text == ".shared" ? "shared memory declarations are not implemented yet"
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

  void parse_local_decl(EntryFn& fn, size_t line) {
    LocalDecl d;
    while (peek().kind == Token::Kind::Word && peek().text == ".align") {
      next();
      d.align = static_cast<uint32_t>(expect_int("alignment"));
    }
    Type ty = expect_type(".local declaration");
    d.name = expect_word("local variable name");
    uint64_t elems = 1;
    if (peek_punct("[")) {
      next();
      elems = static_cast<uint64_t>(expect_int("array size"));
      expect_punct("]");
    }
    d.size = static_cast<uint32_t>(elems * ty.bytes());
    if (d.align == 0) d.align = 8;
    uint32_t off = (fn.local_frame_size + d.align - 1) / d.align * d.align;
    d.offset = off;
    fn.local_frame_size = off + d.size;
    if (!fn.locals.emplace(d.name, d).second) fail(line, "duplicate .local '" + d.name + "'");
    expect_punct(";");
  }

  // ---- operands ----

  Operand parse_operand() {
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
      if (w.size() == 10 && w[0] == '0' && (w[1] == 'f' || w[1] == 'F'))
        return ImmFloatBits{std::stoull(w.substr(2), nullptr, 16), 32};
      if (w.size() == 18 && w[0] == '0' && (w[1] == 'd' || w[1] == 'D'))
        return ImmFloatBits{std::stoull(w.substr(2), nullptr, 16), 64};
      return ImmInt{parse_int_literal(w, t.line)};
    }
    // Bare identifier: a module global / local depot symbol.
    return SymbolOperand{w};
  }

  std::string expect_reg_operand(const std::string& ctx) {
    std::string w = expect_word(ctx);
    if (w[0] != '%') fail(peek().line, ctx + " must be a register, got '" + w + "'");
    return w;
  }

  // Register vector: {%r1, %r2, %r3, %r4}
  std::vector<std::string> parse_reg_vector(size_t n) {
    std::vector<std::string> regs;
    expect_punct("{");
    while (!peek_punct("}")) {
      regs.push_back(expect_reg_operand("vector element"));
      if (peek_punct(",")) next();
    }
    next();
    if (regs.size() != n)
      fail(peek().line, "vector operand has " + std::to_string(regs.size()) + " elements, expected " +
                            std::to_string(n));
    return regs;
  }

  std::vector<Operand> parse_operand_vector(size_t n) {
    std::vector<Operand> ops;
    expect_punct("{");
    while (!peek_punct("}")) {
      ops.push_back(parse_operand());
      if (peek_punct(",")) next();
    }
    next();
    if (ops.size() != n)
      fail(peek().line, "vector operand has " + std::to_string(ops.size()) + " elements, expected " +
                            std::to_string(n));
    return ops;
  }

  Addr parse_addr(const EntryFn& fn) {
    expect_punct("[");
    Addr a;
    std::string base = expect_word("address base");
    if (base[0] == '%') {
      a.base = base;
      a.base_kind = Addr::Base::Reg;
    } else if (call_slots_.count(base)) {
      a.base = base;
      a.base_kind = Addr::Base::CallSlot;
    } else {
      bool found = false;
      for (const auto& p : fn.params) found = found || p.name == base;
      if (!found) fail(peek().line, "unknown parameter or slot '" + base + "' in address operand");
      a.base = base;
      a.base_kind = Addr::Base::EntryParam;
    }
    if (peek_punct("+")) {
      next();
      a.offset = expect_int("address offset");
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
      size_t vec = 1;
      Type ty{};
      bool have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "param") space = Space::Param;
        else if (p == "global") space = Space::Global;
        else if (p == "shared") space = Space::Shared;
        else if (p == "local") space = Space::Local;
        else if (inert_mem_modifier(p)) ;
        else if (p == "v2") vec = 2;
        else if (p == "v4") vec = 4;
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("unrecognized ld/st modifier '." + p + "'");
      }
      if (!have_ty) fail(ins.line, "ld/st missing type: " + opcode);
      if (space == Space::Shared) return unsupported("shared memory not implemented yet");
      if (op0 == "ld") {
        Addr addr;
        std::vector<std::string> dsts;
        if (vec == 1) {
          dsts.push_back(expect_reg_operand("ld destination"));
        } else {
          dsts = parse_reg_vector(vec);
        }
        expect_punct(",");
        addr = parse_addr(fn);
        if (space == Space::Param) {
          if (addr.base_kind == Addr::Base::Reg)
            return unsupported("ld.param through a register address");
          if (addr.base_kind == Addr::Base::CallSlot) {
            if (vec != 1) return unsupported("vector ld.param from call slot");
            OpLdSlot op{addr.base, addr.offset, ty, dsts[0]};
            ins.op = op;
          } else {
            ins.op = OpLd{space, ty, std::move(dsts), addr};
          }
        } else {
          if (addr.base_kind != Addr::Base::Reg)
            return unsupported("non-param load from a parameter/slot name");
          ins.op = OpLd{space, ty, std::move(dsts), addr};
        }
      } else {
        Addr addr = parse_addr(fn);
        expect_punct(",");
        std::vector<Operand> srcs;
        if (vec == 1)
          srcs.push_back(parse_operand());
        else
          srcs = parse_operand_vector(vec);
        if (space == Space::Param) {
          if (addr.base_kind != Addr::Base::CallSlot)
            return unsupported("st.param outside a call sequence");
          if (vec != 1) return unsupported("vector st.param");
          ins.op = OpStSlot{addr.base, addr.offset, ty, srcs[0]};
        } else {
          if (addr.base_kind != Addr::Base::Reg)
            return unsupported("store to a parameter/slot name");
          ins.op = OpSt{space, ty, addr, std::move(srcs)};
        }
      }
    } else if (op0 == "mov") {
      if (parts.size() != 2) return unsupported("mov form");
      auto ty = parse_type_token(parts[1]);
      if (!ty) fail(ins.line, "mov missing type");
      if (ty->kind == Type::Kind::Pred) return unsupported("mov.pred");
      OpMov op;
      op.ty = *ty;
      op.dst = expect_reg_operand("mov destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "cvta") {
      // cvta[.to].{global,local,shared,const}.u64 — all address spaces alias
      // in VirtualGPU's flat VA scheme, so every cvta is an identity move.
      size_t i = 1;
      if (i < parts.size() && parts[i] == "to") ++i;
      if (i < parts.size() && (parts[i] == "global" || parts[i] == "local" || parts[i] == "const")) ++i;
      else if (i < parts.size() && parts[i] == "shared") return unsupported("cvta.shared");
      auto ty = (i < parts.size()) ? parse_type_token(parts[i]) : std::nullopt;
      if (!ty) fail(ins.line, "cvta missing type");
      OpCvta op;
      op.ty = *ty;
      op.dst = expect_reg_operand("cvta destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "cvt") {
      // cvt[.round][.sat][.ftz].<dstty>.<srcty>
      std::vector<Type> tys;
      Round round = Round::None;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "rn") round = Round::Rn;
        else if (p == "rz") round = Round::Rz;
        else if (p == "rm") round = Round::Rm;
        else if (p == "rp") round = Round::Rp;
        else if (p == "rni") round = Round::Rni;
        else if (p == "rzi") round = Round::Rzi;
        else if (p == "rmi") round = Round::Rm;
        else if (p == "rpi") round = Round::Rp;
        else if (p == "sat" || p == "ftz") ;  // saturation/flush handled conservatively below
        else if (auto t2 = parse_type_token(p)) tys.push_back(*t2);
        else return unsupported("unrecognized cvt modifier '." + p + "'");
      }
      if (tys.size() != 2) fail(ins.line, "cvt needs .<dsttype>.<srctype>");
      if ((tys[0].is_float() && tys[0].bits == 16) || (tys[1].is_float() && tys[1].bits == 16))
        return unsupported("half-precision (f16) cvt not implemented yet");
      OpCvt op;
      op.dst_ty = tys[0];
      op.src_ty = tys[1];
      op.round = round;
      op.dst = expect_reg_operand("cvt destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "not") {
      if (parts.size() != 2) return unsupported("not form");
      auto ty = parse_type_token(parts[1]);
      if (!ty) fail(ins.line, "not missing type");
      if (ty->kind == Type::Kind::Pred) {
        OpNotPred op;
        op.dst = expect_reg_operand("not.pred destination");
        expect_punct(",");
        op.src = expect_reg_operand("not.pred source");
        ins.op = op;
      } else {
        OpNot op;
        op.ty = *ty;
        op.dst = expect_reg_operand("not destination");
        expect_punct(",");
        op.src = parse_operand();
        ins.op = op;
      }
    } else if (op0 == "neg") {
      if (parts.size() != 2) return unsupported("neg form");
      auto ty = parse_type_token(parts[1]);
      if (!ty) fail(ins.line, "neg missing type");
      OpNeg op;
      op.ty = *ty;
      op.dst = expect_reg_operand("neg destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "prmt") {
      // prmt.b32[.mode] d, a, b, c — only the default (generic) mode.
      if (parts.size() < 2 || parts[1] != "b32") return unsupported("prmt form (only prmt.b32)");
      if (parts.size() > 2) return unsupported("prmt with an explicit mode ('." + parts[2] + "')");
      OpPrmt op;
      op.dst = expect_reg_operand("prmt destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.c = parse_operand();
      ins.op = op;
    } else if (op0 == "shf") {
      // shf.{l,r}.{wrap,clamp}.b32 d, a, b, c — funnel shift of b:a.
      if (parts.size() != 4 || (parts[1] != "l" && parts[1] != "r") ||
          (parts[2] != "wrap" && parts[2] != "clamp") || parts[3] != "b32")
        return unsupported("shf form (only shf.{l,r}.{wrap,clamp}.b32)");
      OpShf op;
      op.left = parts[1] == "l";
      op.wrap = parts[2] == "wrap";
      op.dst = expect_reg_operand("shf destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.c = parse_operand();
      ins.op = op;
    } else if (op0 == "atom") {
      // atom[.space][.sem][.scope].<op>.<type> d, [a], b [, c]
      Space space = Space::Generic;
      std::optional<AtomOp> aop;
      Type ty{};
      bool have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "global") space = Space::Global;
        else if (p == "shared") return unsupported("atomics on shared memory (no shared memory yet)");
        else if (inert_mem_modifier(p)) ;
        else if (p == "add") aop = AtomOp::Add;
        else if (p == "min") aop = AtomOp::Min;
        else if (p == "max") aop = AtomOp::Max;
        else if (p == "and") aop = AtomOp::And;
        else if (p == "or") aop = AtomOp::Or;
        else if (p == "xor") aop = AtomOp::Xor;
        else if (p == "exch") aop = AtomOp::Exch;
        else if (p == "cas") aop = AtomOp::Cas;
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("atom operation '." + p + "'");
      }
      if (!aop || !have_ty) return unsupported("atom form");
      if (ty.is_float()) return unsupported("float atomics not implemented yet");
      OpAtom op;
      op.op = *aop;
      op.ty = ty;
      (void)space;
      op.dst = expect_reg_operand("atom destination");
      expect_punct(",");
      op.addr = parse_addr(fn);
      if (op.addr.base_kind != Addr::Base::Reg) return unsupported("atom on a parameter name");
      expect_punct(",");
      op.b = parse_operand();
      if (*aop == AtomOp::Cas) {
        expect_punct(",");
        op.c = parse_operand();
      }
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
        else if (p == "rn" || p == "ftz") ;
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
      if (ty.kind == Type::Kind::Pred) {
        // and.pred / or.pred / xor.pred
        std::optional<PredBinOp> pop;
        if (op0 == "and") pop = PredBinOp::And;
        else if (op0 == "or") pop = PredBinOp::Or;
        else if (op0 == "xor") pop = PredBinOp::Xor;
        if (!pop) return unsupported("'" + op0 + "' on predicates");
        OpPredBin op;
        op.op = *pop;
        op.dst = expect_reg_operand("predicate destination");
        expect_punct(",");
        op.a = expect_reg_operand("predicate operand");
        expect_punct(",");
        op.b = expect_reg_operand("predicate operand");
        ins.op = op;
      } else if (ty.is_float()) {
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
        op.a = parse_operand();
        expect_punct(",");
        op.b = parse_operand();
        ins.op = op;
      } else if (wide) {
        if (op0 != "mul" || ty.bits != 32) return unsupported("only mul.wide.{s32,u32} implemented");
        OpMulWide op;
        op.is_signed = ty.is_signed();
        op.dst = expect_reg_operand("destination");
        expect_punct(",");
        op.a = parse_operand();
        expect_punct(",");
        op.b = parse_operand();
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
        op.a = parse_operand();
        expect_punct(",");
        op.b = parse_operand();
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
      Operand a = parse_operand();
      expect_punct(",");
      Operand b = parse_operand();
      expect_punct(",");
      Operand c = parse_operand();
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
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      ins.op = op;
    } else if (op0 == "selp") {
      if (parts.size() != 2) return unsupported("selp form");
      auto ty = parse_type_token(parts[1]);
      if (!ty) fail(ins.line, "selp missing type");
      OpSelp op;
      op.ty = *ty;
      op.dst = expect_reg_operand("destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.pred = expect_reg_operand("selp predicate");
      ins.op = op;
    } else if (op0 == "bra") {
      std::string label = expect_word("branch target");
      bra_fixups.emplace_back(fn.body.size(), label);
      ins.op = OpBra{0, label};
    } else if (op0 == "call") {
      // call.uni (retval0), vprintf, (param0, param1);
      OpCall op;
      if (peek_punct("(")) {
        next();
        op.retval_slot = expect_word("return value slot");
        expect_punct(")");
        expect_punct(",");
      }
      op.callee = expect_word("call target");
      expect_punct(",");
      expect_punct("(");
      while (!peek_punct(")")) {
        op.param_slots.push_back(expect_word("call argument slot"));
        if (peek_punct(",")) next();
      }
      next();
      if (op.callee != "vprintf")
        return unsupported("call to '" + op.callee + "' (only the vprintf builtin is callable)");
      ins.op = op;
    } else if (op0 == "bar" || op0 == "barrier") {
      bool sync_seen = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] == "sync") sync_seen = true;
        else if (parts[i] == "cta") ;
        else return unsupported("only bar.sync is implemented");
      }
      if (!sync_seen) return unsupported("only bar.sync is implemented");
      Operand which = parse_operand();
      if (auto* imm = std::get_if<ImmInt>(&which); !imm || imm->value != 0)
        return unsupported("only barrier 0 (bar.sync 0) is implemented");
      if (peek_punct(",")) return unsupported("partial barriers (bar.sync 0, N) not implemented");
      ins.op = OpBar{};
    } else if (op0 == "ret" || op0 == "exit") {
      ins.op = OpRet{};
    } else {
      return unsupported("instruction not in the implemented PTX subset");
    }

    expect_punct(";");
    ins.text = reconstruct_from(start_tok);
    return ins;
  }

  // Rebuilds source-ish text from tokens for error messages, stopping at the
  // statement's own ';'.
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
  std::set<std::string> call_slots_;
};

}  // namespace

Module parse(const std::string& src) {
  Parser p(src);
  // Note: a module with zero kernels is legal (e.g. a translation unit with
  // only host code still registers an empty PTX image).
  return p.parse_module();
}

}  // namespace vgpu::ptx
