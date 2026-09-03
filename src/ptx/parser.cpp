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
  if (part == "bf16") return Type{Type::Kind::BF, 16};
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
        // ".extern .shared ..." declares dynamic shared memory; ".extern .func"
        // declares an external function (vprintf), which we skip.
        if (peek(1).kind == Token::Kind::Word && peek(1).text == ".shared") {
          next();  // .extern
          next();  // .shared
          SharedDecl d = parse_shared_decl(t.line, /*allow_unsized=*/true);
          d.dynamic = d.size == 0;
          m.module_shared.push_back(std::move(d));
          continue;
        }
        while (!at_end() && !peek_punct(";")) next();
        if (!at_end()) next();
        continue;
      }
      if (t.text == ".shared") {
        next();
        m.module_shared.push_back(parse_shared_decl(t.line, /*allow_unsized=*/true));
        continue;
      }
      if (t.text == ".entry") {
        next();
        EntryFn fn;
        // A prototype (".entry name(params);") declares a kernel defined later
        // or elsewhere; only a definition carries a body worth keeping.
        if (parse_entry(&fn)) m.entries.push_back(std::move(fn));
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

  // PTX identifiers start with a letter, '_' or '$'; numeric literals (including
  // "0f3F800000" floats) start with a digit or a sign.
  static bool is_identifier_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$';
  }

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
      } else if (peek().kind == Token::Kind::Word && is_identifier_start(peek().text[0])) {
        // "= some_symbol": the initialiser is another symbol's address, which
        // only exists once the module is loaded. The lexer gives numbers and
        // identifiers the same token kind, so the first character is what
        // separates them -- "= 5" is a value, not a symbol named "5".
        g.init_symbol = next().text;
      } else {
        int64_t v = expect_int("initializer");
        for (uint32_t b = 0; b < ty.bytes(); ++b)
          g.init.push_back(static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * b)) & 0xFF));
      }
      if (g.init.size() > g.size)
        fail(line, "initializer for '" + g.name + "' longer than its declared size");
      // A symbol initialiser leaves no bytes here: the address is written by
      // the loader once every global has one.
      if (g.init_symbol.empty()) g.init.resize(g.size, 0);
    }
    expect_punct(";");
    return g;
  }

  // ---- entry functions ----

  // Returns false when this was a declaration rather than a definition, in
  // which case *out is not meaningful.
  bool parse_entry(EntryFn* out) {
    EntryFn& fn = *out;
    cur_fn_ = &fn;
    fn.name = expect_word("kernel name");
    current_kernel_ = fn.name;
    call_slots_.clear();
    declared_regs_.clear();
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
    // Performance directives that follow the signature. .maxntid/.reqntid are
    // functional (they bound the launch shape); .minnctapersm is a scheduling
    // hint we record but do not act on.
    while (peek().kind == Token::Kind::Word && peek().text[0] == '.') {
      std::string d = peek().text;
      if (d == ".maxntid" || d == ".reqntid") {
        next();
        std::array<uint32_t, 3> dims{1, 1, 1};
        for (int i = 0; i < 3; ++i) {
          dims[i] = static_cast<uint32_t>(expect_int("launch bound"));
          if (i < 2 && peek_punct(",")) next();
          else if (i < 2) break;
        }
        if (d == ".maxntid") fn.max_ntid = dims;
        else fn.req_ntid = dims;
      } else if (d == ".minnctapersm" || d == ".maxnreg" || d == ".maxnctapersm") {
        next();
        uint32_t v = static_cast<uint32_t>(expect_int("directive value"));
        if (d == ".minnctapersm") fn.min_ctas_per_sm = v;
      } else if (d == ".noreturn" || d == ".pragma") {
        next();
        while (!at_end() && !peek_punct(";") && !peek_punct("{")) next();
        if (peek_punct(";")) next();
      } else {
        break;
      }
    }
    if (peek_punct(";")) {  // prototype, no body
      next();
      current_kernel_.clear();
      cur_fn_ = nullptr;
      return false;
    }
    expect_punct("{");
    parse_body(fn);
    // Dynamic shared memory lives above every static allocation.
    for (auto& [name, d] : fn.shared)
      if (d.dynamic) d.offset = fn.static_shared_size;
    current_kernel_.clear();
    cur_fn_ = nullptr;
    return true;
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
      if (t.kind == Token::Kind::Word && t.text == ".shared") {
        next();
        SharedDecl d = parse_shared_decl(t.line, /*allow_unsized=*/true);
        place_shared(fn, std::move(d), t.line);
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text[0] == '.') {
        fail_unsupported(t.line, t.text, fn.name, "directive not in the implemented subset");
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
        for (int i = 0; i < n; ++i) {
          const std::string nm = name + std::to_string(i);
          fn.reg_decls[nm] = ty;
          declared_regs_.insert(nm);
          fn.reg_wide[nm] = ty.bits > 32 && ty.kind != Type::Kind::Pred;
          intern(nm);
        }
      } else {
        fn.reg_decls[name] = ty;
        declared_regs_.insert(name);
        fn.reg_wide[name] = ty.bits > 32 && ty.kind != Type::Kind::Pred;
        intern(name);
      }
      if (peek_punct(",")) {
        next();
        continue;
      }
      break;
    }
    expect_punct(";");
  }

  // ".shared .align N .b8 name[size];" — size may be omitted for the dynamic
  // (extern) form, whose extent comes from the launch's sharedMemBytes.
  SharedDecl parse_shared_decl(size_t line, bool allow_unsized) {
    SharedDecl d;
    while (peek().kind == Token::Kind::Word && peek().text == ".align") {
      next();
      d.align = static_cast<uint32_t>(expect_int("alignment"));
    }
    Type ty = expect_type(".shared declaration");
    d.name = expect_word("shared variable name");
    uint64_t elems = 0;
    bool sized = false;
    if (peek_punct("[")) {
      next();
      if (!peek_punct("]")) {
        elems = static_cast<uint64_t>(expect_int("array size"));
        sized = true;
      }
      expect_punct("]");
    } else {
      elems = 1;
      sized = true;
    }
    d.size = static_cast<uint32_t>(elems * ty.bytes());
    if (!sized) {
      if (!allow_unsized) fail(line, "unsized .shared array '" + d.name + "'");
      d.dynamic = true;
      d.size = 0;
    }
    if (d.align == 0) d.align = 8;
    expect_punct(";");
    return d;
  }

  void place_shared(EntryFn& fn, SharedDecl d, size_t line) {
    if (d.dynamic) {
      // Dynamic shared memory starts after all static allocations; the exact
      // offset is fixed up once the whole body has been parsed.
      fn.uses_dynamic_shared = true;
      d.offset = 0;
    } else {
      uint32_t off = (fn.static_shared_size + d.align - 1) / d.align * d.align;
      d.offset = off;
      fn.static_shared_size = off + d.size;
    }
    if (!fn.shared.emplace(d.name, d).second) fail(line, "duplicate .shared '" + d.name + "'");
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
      return RegOperand{intern(w)};
    }
    if (isdigit(static_cast<unsigned char>(w[0]))) {
      if (w.size() == 10 && w[0] == '0' && (w[1] == 'f' || w[1] == 'F'))
        return ImmFloatBits{std::stoull(w.substr(2), nullptr, 16), 32};
      if (w.size() == 18 && w[0] == '0' && (w[1] == 'd' || w[1] == 'D'))
        return ImmFloatBits{std::stoull(w.substr(2), nullptr, 16), 64};
      return ImmInt{parse_int_literal(w, t.line)};
    }
    // A bare identifier is an inline-asm register local if it was declared as
    // one; otherwise it names a module global or local depot.
    if (declared_regs_.count(w)) return RegOperand{intern(w)};
    return SymbolOperand{w};
  }

  // Interns a register name into the current kernel's dense numbering. The
  // interpreter indexes a flat register file with these ids instead of hashing
  // names at run time.
  // Interns a register into the file its declared width selects. Ids are
  // dense within each file, so both can be plain vectors.
  Reg intern(const std::string& name) {
    auto wit = cur_fn_->reg_wide.find(name);
    bool wide = wit != cur_fn_->reg_wide.end() && wit->second;
    auto it = cur_fn_->reg_ids.find(name);
    if (it == cur_fn_->reg_ids.end()) {
      uint32_t id = wide ? cur_fn_->num_regs64++ : cur_fn_->num_regs32++;
      it = cur_fn_->reg_ids.emplace(name, id).first;
      cur_fn_->reg_wide.emplace(name, wide);
      ++cur_fn_->num_regs;
    }
    return Reg{name, it->second, wide};
  }

  // Register operand. Inline asm may declare locals without the '%' sigil
  // (".reg .f16 low;"), so a bare identifier that was declared as a register
  // is accepted too.
  Reg expect_reg_operand(const std::string& ctx) {
    std::string w = expect_word(ctx);
    if (w[0] != '%' && !declared_regs_.count(w))
      fail(peek().line, ctx + " must be a register, got '" + w + "'");
    return intern(w);
  }

  // Register vector: {%r1, %r2, %r3, %r4}
  std::vector<Reg> parse_reg_vector(size_t n) {
    std::vector<Reg> regs;
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

  std::vector<Reg> parse_reg_vector_any() {
    std::vector<Reg> regs;
    expect_punct("{");
    while (!peek_punct("}")) {
      regs.push_back(expect_reg_operand("fragment register"));
      if (peek_punct(",")) next();
    }
    next();
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
    if (base[0] == '%' || declared_regs_.count(base)) {
      a.base = base;
      a.base_kind = Addr::Base::Reg;
      Reg r = intern(base);
      a.base_id = r.id;
      a.base_wide = r.wide;
    } else if (call_slots_.count(base)) {
      a.base = base;
      a.base_kind = Addr::Base::CallSlot;
    } else {
      bool is_param = false;
      for (const auto& p : fn.params) is_param = is_param || p.name == base;
      if (is_param) {
        a.base = base;
        a.base_kind = Addr::Base::EntryParam;
      } else {
        // A variable addressed by name (.shared/.local/.global). Resolution
        // happens at launch, when the module's symbol table is available.
        a.base = base;
        a.base_kind = Addr::Base::Symbol;
      }
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
      if (op0 == "ld") {
        Addr addr;
        std::vector<Reg> dsts;
        if (vec == 1) {
          dsts.push_back(expect_reg_operand("ld destination"));
        } else {
          dsts = parse_reg_vector(vec);
        }
        expect_punct(",");
        addr = parse_addr(fn);
        if (space == Space::Param) {
          // A register base is a parameter's address, taken with
          // "mov.b64 %rd, kernel_param_N". Parameters have addresses of their
          // own, so this loads back out of the parameter buffer.
          if (addr.base_kind == Addr::Base::Reg) {
            ins.op = OpLd{space, ty, std::move(dsts), addr};
          } else if (addr.base_kind == Addr::Base::CallSlot) {
            if (vec != 1) return unsupported("vector ld.param from call slot");
            OpLdSlot op{addr.base, addr.offset, ty, dsts[0]};
            ins.op = op;
          } else {
            ins.op = OpLd{space, ty, std::move(dsts), addr};
          }
        } else {
          if (addr.base_kind == Addr::Base::CallSlot || addr.base_kind == Addr::Base::EntryParam)
            return unsupported("non-param load through a parameter/slot name");
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
          if (addr.base_kind == Addr::Base::CallSlot || addr.base_kind == Addr::Base::EntryParam)
            return unsupported("store through a parameter/slot name");
          ins.op = OpSt{space, ty, addr, std::move(srcs)};
        }
      }
    } else if (op0 == "mov") {
      if (parts.size() != 2) return unsupported("mov form");
      auto ty = parse_type_token(parts[1]);
      if (!ty) fail(ins.line, "mov missing type");
      if (ty->kind == Type::Kind::Pred) {
        // Predicates live in their own register file, so this cannot go through
        // the value path below.
        OpMovPred op;
        op.dst = expect_reg_operand("mov.pred destination");
        expect_punct(",");
        op.src = parse_operand();
        ins.op = op;
      } else if (peek_punct("{")) {  // mov.bN {d0, d1, ...}, src  — unpack
        OpMovUnpack op;
        op.ty = *ty;
        next();
        while (!peek_punct("}")) {
          op.dsts.push_back(expect_reg_operand("mov destination element"));
          if (peek_punct(",")) next();
        }
        next();
        expect_punct(",");
        op.src = parse_operand();
        if (op.dsts.size() != 2 && op.dsts.size() != 4) return unsupported("mov unpack arity");
        ins.op = op;
      } else {
        Reg dst = expect_reg_operand("mov destination");
        expect_punct(",");
        if (peek_punct("{")) {  // mov.bN d, {s0, s1, ...}  — pack
          OpMovPack op;
          op.ty = *ty;
          op.dst = dst;
          next();
          while (!peek_punct("}")) {
            op.srcs.push_back(parse_operand());
            if (peek_punct(",")) next();
          }
          next();
          if (op.srcs.size() != 2 && op.srcs.size() != 4) return unsupported("mov pack arity");
          ins.op = op;
        } else {
          OpMov op;
          op.ty = *ty;
          op.dst = dst;
          op.src = parse_operand();
          ins.op = op;
        }
      }
    } else if (op0 == "cvta") {
      // cvta[.to].<space>.<type>
      size_t i = 1;
      bool to_space = false;
      if (i < parts.size() && parts[i] == "to") {
        to_space = true;
        ++i;
      }
      Space space = Space::Generic;
      if (i < parts.size()) {
        if (parts[i] == "global") { space = Space::Global; ++i; }
        else if (parts[i] == "local") { space = Space::Local; ++i; }
        else if (parts[i] == "shared") { space = Space::Shared; ++i; }
        else if (parts[i] == "const") { space = Space::Global; ++i; }
        else if (parts[i] == "param") { space = Space::Param; ++i; }
      }
      auto ty = (i < parts.size()) ? parse_type_token(parts[i]) : std::nullopt;
      if (!ty) fail(ins.line, "cvta missing type");
      OpCvta op;
      op.ty = *ty;
      op.space = space;
      op.to_space = to_space;
      op.dst = expect_reg_operand("cvta destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "cvt") {
      // cvt[.round][.sat][.ftz].<dstty>.<srcty>
      std::vector<Type> tys;
      std::string packed;  // "f16x2"/"bf16x2": two f32 sources packed into one register
      Round round = Round::None;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "rn") round = Round::Rn;
        else if (p == "rz") round = Round::Rz;
        else if (p == "rm") round = Round::Rm;
        else if (p == "rp") round = Round::Rp;
        else if (p == "rni") round = Round::Rni;
        else if (p == "rzi") round = Round::Rzi;
        else if (p == "rmi") round = Round::Rmi;
        else if (p == "rpi") round = Round::Rpi;
        else if (p == "sat" || p == "ftz") ;  // saturation/flush handled conservatively below
        else if (p == "f16x2" || p == "bf16x2") packed = p;
        else if (auto t2 = parse_type_token(p)) tys.push_back(*t2);
        else return unsupported("unrecognized cvt modifier '." + p + "'");
      }
      if (!packed.empty()) {
        // cvt.rn.f16x2.f32 d, a, b -- two f32 converted and packed, a high, b low.
        if (tys.size() != 1 || tys[0].bits != 32 || !tys[0].is_float())
          return unsupported("cvt to " + packed + " from a source other than f32");
        OpCvtF16x2 op;
        op.bf16 = packed[0] == 'b';
        op.dst = expect_reg_operand("cvt destination");
        expect_punct(",");
        op.a = parse_operand();
        expect_punct(",");
        op.b = parse_operand();
        ins.op = op;
        expect_punct(";");
        return ins;
      }
      if (tys.size() != 2) fail(ins.line, "cvt needs .<dsttype>.<srctype>");
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
    } else if ((op0 == "add" || op0 == "sub" || op0 == "mul" || op0 == "fma" || op0 == "neg") &&
               opcode.find("f16x2") != std::string::npos) {
      // Packed half2 arithmetic.
      for (size_t i = 1; i < parts.size(); ++i)
        if (parts[i] != "f16x2" && parts[i] != "rn" && parts[i] != "ftz" && parts[i] != "sat" &&
            parts[i] != "rz" && parts[i] != "rm" && parts[i] != "rp")
          return unsupported("f16x2 modifier '." + parts[i] + "'");
      Reg dst = expect_reg_operand("destination");
      expect_punct(",");
      Operand a = parse_operand();
      if (op0 == "neg") {
        ins.op = OpF16x2Neg{dst, a};
      } else {
        expect_punct(",");
        Operand b = parse_operand();
        if (op0 == "fma") {
          expect_punct(",");
          Operand c = parse_operand();
          ins.op = OpF16x2Fma{dst, a, b, c};
        } else {
          FloatBinOp fop = op0 == "add"   ? FloatBinOp::Add
                           : op0 == "sub" ? FloatBinOp::Sub
                                          : FloatBinOp::Mul;
          ins.op = OpF16x2Bin{fop, dst, a, b};
        }
      }
    } else if (op0 == "wmma") {
      // wmma.mma.sync.aligned.<alayout>.<blayout>.m16n16k16.f32.f32 {d}, {a}, {b}, {c};
      // wmma.store.d.sync.aligned.<layout>.m16n16k16[.space].f32 [addr], {d}, stride;
      if (parts.size() < 2) return unsupported("wmma form");
      const std::string& kind = parts[1];
      std::vector<MatLayout> layouts;
      Space space = Space::Generic;
      bool shape_ok = false;
      for (size_t i = 2; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync" || p == "aligned" || p == "d" || p == "f32") ;
        else if (p == "row") layouts.push_back(MatLayout::Row);
        else if (p == "col") layouts.push_back(MatLayout::Col);
        else if (p == "m16n16k16") shape_ok = true;
        else if (p == "global") space = Space::Global;
        else if (p == "shared") space = Space::Shared;
        else if (p == "f16") ;
        else return unsupported("wmma modifier '." + p + "' (only m16n16k16 f32 is implemented)");
      }
      if (!shape_ok) return unsupported("only the m16n16k16 wmma shape is implemented");
      if (kind == "mma") {
        if (layouts.size() != 2) return unsupported("wmma.mma needs both A and B layouts");
        OpWmmaMma op;
        op.alayout = layouts[0];
        op.blayout = layouts[1];
        op.d = parse_reg_vector_any();
        expect_punct(",");
        op.a = parse_reg_vector_any();
        expect_punct(",");
        op.b = parse_reg_vector_any();
        expect_punct(",");
        op.c = parse_reg_vector_any();
        if (op.d.size() != 8 || op.a.size() != 8 || op.b.size() != 8 || op.c.size() != 8)
          return unsupported("wmma.mma fragment arity (expected 8 registers each)");
        ins.op = op;
      } else if (kind == "store") {
        OpWmmaStore op;
        op.layout = layouts.empty() ? MatLayout::Row : layouts[0];
        op.space = space;
        op.addr = parse_addr(fn);
        expect_punct(",");
        {
          std::vector<Reg> regs = parse_reg_vector_any();
          for (auto& r : regs) op.src.push_back(Operand{RegOperand{r}});
        }
        if (op.src.size() != 8) return unsupported("wmma.store fragment arity");
        expect_punct(",");
        op.stride = parse_operand();
        ins.op = op;
      } else {
        return unsupported("wmma." + kind + " is not implemented (only .mma and .store.d)");
      }
    } else if (op0 == "abs") {
      if (parts.size() < 2) return unsupported("abs form");
      auto ty = parse_type_token(parts.back());
      if (!ty) fail(ins.line, "abs missing type");
      OpAbs op;
      op.ty = *ty;
      op.dst = expect_reg_operand("abs destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "ex2" || op0 == "lg2" || op0 == "sin" || op0 == "cos" || op0 == "sqrt" ||
               op0 == "rsqrt" || op0 == "rcp" || op0 == "tanh") {
      static const std::unordered_map<std::string, MathOp> mops = {
          {"ex2", MathOp::Ex2},     {"lg2", MathOp::Lg2},   {"sin", MathOp::Sin},
          {"cos", MathOp::Cos},     {"sqrt", MathOp::Sqrt}, {"rsqrt", MathOp::Rsqrt},
          {"rcp", MathOp::Rcp},     {"tanh", MathOp::Tanh}};
      Type ty{};
      bool have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        // approx/rn/rz/ftz/full select precision on hardware; VirtualGPU always
        // computes at host precision (documented divergence).
        if (p == "approx" || p == "rn" || p == "rz" || p == "rm" || p == "rp" || p == "ftz" ||
            p == "full")
          ;
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("unrecognized modifier '." + p + "'");
      }
      if (!have_ty) fail(ins.line, opcode + " missing type");
      if (!ty.is_float() || (ty.bits != 32 && ty.bits != 64))
        return unsupported("only f32/f64 " + op0 + " is implemented");
      OpMath op;
      op.op = mops.at(op0);
      op.ty = ty;
      op.dst = expect_reg_operand("destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "bfe" || op0 == "bfi") {
      auto ty = parse_type_token(parts.back());
      if (!ty || parts.size() != 2) return unsupported(op0 + " form");
      Reg dst = expect_reg_operand("destination");
      expect_punct(",");
      Operand a = parse_operand();
      expect_punct(",");
      Operand b = parse_operand();
      expect_punct(",");
      Operand c = parse_operand();
      if (op0 == "bfe") {
        ins.op = OpBfe{*ty, dst, a, b, c};
      } else {
        expect_punct(",");
        Operand d = parse_operand();
        ins.op = OpBfi{*ty, dst, a, b, c, d};
      }
    } else if (op0 == "brev") {
      auto ty = parse_type_token(parts.back());
      if (!ty || parts.size() != 2) return unsupported("brev form");
      OpBrev op;
      op.ty = *ty;
      op.dst = expect_reg_operand("destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "redux") {
      // redux.sync.<op>.<type> d, a, membermask
      std::optional<ReduxOp> rop;
      Type ty{};
      bool have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync") ;
        else if (p == "add") rop = ReduxOp::Add;
        else if (p == "min") rop = ReduxOp::Min;
        else if (p == "max") rop = ReduxOp::Max;
        else if (p == "and") rop = ReduxOp::And;
        else if (p == "or") rop = ReduxOp::Or;
        else if (p == "xor") rop = ReduxOp::Xor;
        else if (auto t2 = parse_type_token(p)) { ty = *t2; have_ty = true; }
        else return unsupported("redux modifier '." + p + "'");
      }
      if (!rop || !have_ty) return unsupported("redux form");
      if (ty.is_float()) return unsupported("redux on float types");
      OpRedux op;
      op.op = *rop;
      op.ty = ty;
      op.dst = expect_reg_operand("redux destination");
      expect_punct(",");
      op.src = parse_operand();
      expect_punct(",");
      (void)parse_operand();  // membermask; the active mask already carries it
      ins.op = op;
    } else if (op0 == "copysign") {
      auto ty = parse_type_token(parts.back());
      if (!ty || !ty->is_float()) return unsupported("copysign form (float types only)");
      OpCopysign op;
      op.ty = *ty;
      op.dst = expect_reg_operand("copysign destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      ins.op = op;
    } else if (op0 == "dp4a") {
      // dp4a.atype.btype d, a, b, c
      if (parts.size() != 3) return unsupported("dp4a form");
      const bool as = parts[1] == "s32", au = parts[1] == "u32";
      const bool bs = parts[2] == "s32", bu = parts[2] == "u32";
      if ((!as && !au) || (!bs && !bu)) return unsupported("dp4a operand types");
      OpDp4a op;
      op.a_signed = as;
      op.b_signed = bs;
      op.dst = expect_reg_operand("dp4a destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.c = parse_operand();
      ins.op = op;
    } else if (op0 == "bmsk") {
      if (parts.size() < 2) return unsupported("bmsk form");
      bool wrap = false;
      for (size_t i = 1; i + 1 < parts.size(); ++i) {
        if (parts[i] == "wrap") wrap = true;
        else if (parts[i] != "clamp") return unsupported("bmsk mode '." + parts[i] + "'");
      }
      if (parts.back() != "b32") return unsupported("bmsk type (only .b32)");
      OpBmsk op;
      op.wrap = wrap;
      op.dst = expect_reg_operand("bmsk destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      ins.op = op;
    } else if (op0 == "popc" || op0 == "clz") {
      auto ty = parse_type_token(parts.back());
      if (!ty || parts.size() != 2) return unsupported(op0 + " form");
      OpPopcClz op;
      op.popc = (op0 == "popc");
      op.ty = *ty;
      op.dst = expect_reg_operand("destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "shfl") {
      // shfl.sync.<mode>.b32 d[|p], a, b, c, membermask;
      ShflMode mode;
      bool have_mode = false, sync = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync") sync = true;
        else if (p == "up") { mode = ShflMode::Up; have_mode = true; }
        else if (p == "down") { mode = ShflMode::Down; have_mode = true; }
        else if (p == "bfly") { mode = ShflMode::Bfly; have_mode = true; }
        else if (p == "idx") { mode = ShflMode::Idx; have_mode = true; }
        else if (p == "b32") ;
        else return unsupported("shfl modifier '." + p + "'");
      }
      if (!have_mode) return unsupported("shfl needs a mode (.up/.down/.bfly/.idx)");
      if (!sync) return unsupported("the deprecated non-.sync shfl is not implemented");
      OpShfl op;
      op.mode = mode;
      op.dst = expect_reg_operand("shfl destination");
      if (peek_punct("|")) {  // optional predicate destination
        next();
        op.pred_dst = expect_reg_operand("shfl predicate destination");
      }
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.c = parse_operand();
      expect_punct(",");
      op.member_mask = parse_operand();
      ins.op = op;
    } else if (op0 == "vote") {
      // vote.sync.{all,any,uni}.pred d, p, membermask;  vote.sync.ballot.b32 d, p, mask;
      VoteMode mode = VoteMode::All;
      bool ballot = false, have_mode = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync") ;
        else if (p == "all") { mode = VoteMode::All; have_mode = true; }
        else if (p == "any") { mode = VoteMode::Any; have_mode = true; }
        else if (p == "uni") { mode = VoteMode::Uni; have_mode = true; }
        else if (p == "ballot") { mode = VoteMode::Ballot; ballot = true; have_mode = true; }
        else if (p == "pred" || p == "b32") ;
        else return unsupported("vote modifier '." + p + "'");
      }
      if (!have_mode) return unsupported("vote needs a mode");
      OpVote op;
      op.mode = mode;
      op.ballot = ballot;
      op.dst = expect_reg_operand("vote destination");
      expect_punct(",");
      if (peek_punct("!")) {
        next();
        op.negate_src = true;
      }
      op.src = expect_reg_operand("vote predicate source");
      if (peek_punct(",")) {
        next();
        (void)parse_operand();  // membermask
      }
      ins.op = op;
    } else if (op0 == "neg") {
      // neg carries the same accuracy modifiers as the other float ops --
      // neg.ftz.f32 is ordinary in generated code. .ftz flushes denormals to
      // zero, which negation cannot turn into a wrong sign, so the exact result
      // stays within what the modifier promises. Take the type from the last
      // component rather than requiring it to be the only one, as abs does.
      if (parts.size() < 2) return unsupported("neg form");
      for (size_t i = 1; i + 1 < parts.size(); ++i)
        if (parts[i] != "ftz")
          return unsupported("neg modifier '." + parts[i] + "'");
      auto ty = parse_type_token(parts.back());
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
        else if (p == "shared") space = Space::Shared;
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
      // Float atomics are what a reduction, a gradient accumulation, or an
      // embedding backward pass is built out of, so they are not optional for
      // ML work. CUDA exposes add/exch/min/max on float and double; the
      // bitwise ops and CAS are integer-only there too, and a program that
      // wants CAS on a float does it through .b32.
      if (ty.is_float()) {
        if (*aop != AtomOp::Add && *aop != AtomOp::Exch && *aop != AtomOp::Min &&
            *aop != AtomOp::Max)
          return unsupported("atom." + std::string(*aop == AtomOp::Cas ? "cas" : "bitwise") +
                             " on a float type (CUDA has no such instruction; use .b32)");
        if (ty.bits != 32 && ty.bits != 64)
          return unsupported("float atomics are implemented for f32 and f64; f16/bf16 atomics "
                             "are not yet");
      }
      OpAtom op;
      op.op = *aop;
      op.ty = ty;
      op.space = space;
      op.dst = expect_reg_operand("atom destination");
      expect_punct(",");
      op.addr = parse_addr(fn);
      if (op.addr.base_kind == Addr::Base::CallSlot || op.addr.base_kind == Addr::Base::EntryParam)
        return unsupported("atom through a parameter/slot name");
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
      FRound frnd = FRound::Nearest;
      Type ty{};
      bool have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "wide") wide = true;
        else if (p == "lo") lo = true;
        else if (p == "hi") hi = true;
        else if (p == "rn" || p == "ftz") ;
        else if (p == "rz") frnd = FRound::Zero;
        else if (p == "rm") frnd = FRound::MinusInf;
        else if (p == "rp") frnd = FRound::PlusInf;
        // .approx and .full ask for a faster, less accurate result -- div.approx
        // is what __fdividef compiles to, and ML kernels use it constantly.
        // Both have a documented error bound, and the exact IEEE result falls
        // inside it, so computing exactly satisfies the contract. This is the
        // same policy the SFU transcendentals already follow: correct to better
        // than hardware, never bit-identical to it.
        else if (p == "approx" || p == "full") ;
        // .sat clamps the result into [0,1]. That is a semantic change, not an
        // accuracy one, so ignoring it would silently produce wrong numbers.
        else if (p == "sat")
          return unsupported("modifier '.sat' not implemented");
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("unrecognized modifier '." + p + "'");
      }
      if (!have_ty) fail(ins.line, opcode + " missing type");
      if (hi) {
        if (op0 != "mul") return unsupported("'." + op0 + ".hi' is not implemented");
        if (ty.is_float()) return unsupported("mul.hi on floats");
        OpMulHi op;
        op.ty = ty;
        op.dst = expect_reg_operand("destination");
        expect_punct(",");
        op.a = parse_operand();
        expect_punct(",");
        op.b = parse_operand();
        ins.op = op;
        expect_punct(";");
        ins.text = reconstruct_from(start_tok);
        return ins;
      }
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
        if (ty.bits != 16 && ty.bits != 32 && ty.bits != 64)
          return unsupported("only f16/f32/f64 float math implemented");
        static const std::unordered_map<std::string, FloatBinOp> fops = {
            {"add", FloatBinOp::Add}, {"sub", FloatBinOp::Sub}, {"mul", FloatBinOp::Mul},
            {"min", FloatBinOp::Min}, {"max", FloatBinOp::Max}, {"div", FloatBinOp::Div}};
        auto it = fops.find(op0);
        if (it == fops.end()) return unsupported("float op '" + op0 + "'");
        OpFloatBin op;
        op.round = frnd;
        op.op = it->second;
        op.ty = ty;
        op.dst = expect_reg_operand("destination");
        expect_punct(",");
        op.a = parse_operand();
        expect_punct(",");
        op.b = parse_operand();
        ins.op = op;
      } else if (wide) {
        // .wide doubles the operand width: 16-bit sources give a 32-bit result,
        // 32-bit give 64. Quantized kernels use the 16-bit form for byte-pair
        // arithmetic, so restricting this to 32-bit blocked them.
        if (op0 != "mul" || (ty.bits != 32 && ty.bits != 16))
          return unsupported("only mul.wide.{s16,u16,s32,u32} implemented");
        OpMulWide op;
        op.src_bits = ty.bits;
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
      bool have_ty = false, lo = false, wide = false, hi = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "lo") lo = true;
        else if (p == "wide") wide = true;
        else if (p == "hi") hi = true;
        // Rounding modes: VirtualGPU always computes at host precision
        // (round-to-nearest) — a documented divergence, see ARCHITECTURE.md.
        else if (p == "rn" || p == "rz" || p == "rm" || p == "rp" || p == "ftz" || p == "sat") ;
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("unrecognized modifier '." + p + "'");
      }
      if (!have_ty) fail(ins.line, opcode + " missing type");
      Reg dst = expect_reg_operand("destination");
      expect_punct(",");
      Operand a = parse_operand();
      expect_punct(",");
      Operand b = parse_operand();
      expect_punct(",");
      Operand c = parse_operand();
      if (ty.is_float()) {
        if (ty.bits != 32 && ty.bits != 64) return unsupported("only f32/f64 fma implemented");
        ins.op = OpFma{ty, dst, a, b, c};
      } else if (wide) {
        if (ty.bits != 32) return unsupported("only mad.wide.{s32,u32} implemented");
        ins.op = OpMadWide{ty.is_signed(), dst, a, b, c};
      } else if (hi) {
        ins.op = OpMadHi{ty, dst, a, b, c};
      } else {
        if (!lo) return unsupported("integer mad requires .lo or .wide");
        ins.op = OpMadLo{ty, dst, a, b, c};
      }
    } else if (op0 == "setp") {
      // setp.<cmp>[.ftz].<type> — drop the flush-to-zero qualifier.
      if (parts.size() == 4 && parts[2] == "ftz") parts.erase(parts.begin() + 2);
      if (parts.size() != 3) return unsupported("setp form (only setp.<cmp>.<type> is implemented)");
      static const std::unordered_map<std::string, CmpOp> cmps = {
          {"eq", CmpOp::Eq},   {"ne", CmpOp::Ne},   {"lt", CmpOp::Lt},
          {"le", CmpOp::Le},   {"gt", CmpOp::Gt},   {"ge", CmpOp::Ge},
          // Unsigned integer forms share the ordered comparators; the operand
          // type already selects signed vs unsigned interpretation.
          {"lo", CmpOp::Lt},   {"ls", CmpOp::Le},   {"hi", CmpOp::Gt},
          {"hs", CmpOp::Ge},
          // Float unordered (NaN-true) forms and the NaN tests.
          {"equ", CmpOp::Equ}, {"neu", CmpOp::Neu}, {"ltu", CmpOp::Ltu},
          {"leu", CmpOp::Leu}, {"gtu", CmpOp::Gtu}, {"geu", CmpOp::Geu},
          {"num", CmpOp::Num}, {"nan", CmpOp::Nan}};
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
    } else if (op0 == "trap") {
      ins.op = OpTrap{};
    } else if (op0 == "membar" || op0 == "fence") {
      // Blocks execute their instructions in order and device atomics are
      // serialized by a lock, so every prior write is already visible to
      // whoever could observe it. There is no reordering here to fence against.
      ins.op = OpBar{};
    } else if (op0 == "nanosleep") {
      // A backoff hint. Consuming it as a no-op is correct; the operand is a
      // duration nothing here can meaningfully honour.
      (void)parse_operand();
      ins.op = OpBar{};  // nothing to do; treated as a barrier-free no-op
    } else if (op0 == "bar" || op0 == "barrier") {
      bool sync_seen = false, warp_scope = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] == "sync") sync_seen = true;
        else if (parts[i] == "cta") ;
        else if (parts[i] == "warp") warp_scope = true;
        else return unsupported("only bar.sync and bar.warp.sync are implemented");
      }
      if (!sync_seen) return unsupported("only bar.sync is implemented");
      if (warp_scope) {
        // __syncwarp. A warp executes its lanes in lockstep here and diverged
        // paths reconverge at the earliest common pc, so the lanes named by the
        // mask are already synchronised by the time this is reached. The
        // operand is the member mask, which nothing needs to consume.
        (void)parse_operand();
        ins.op = OpBar{};
        expect_punct(";");
        return ins;
      }
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
  std::set<std::string> declared_regs_;  // every .reg name in the current kernel
  EntryFn* cur_fn_ = nullptr;           // receives interned register ids
};

}  // namespace

Module parse(const std::string& src) {
  Parser p(src);
  // Note: a module with zero kernels is legal (e.g. a translation unit with
  // only host code still registers an empty PTX image).
  Module m = p.parse_module();
  // Module-scope .shared variables are per-block storage available to every
  // kernel, so give each entry its own slot in that kernel's shared frame.
  for (auto& fn : m.entries) {
    for (const auto& md : m.module_shared) {
      if (fn.shared.count(md.name)) continue;
      SharedDecl d = md;
      if (d.dynamic) {
        fn.uses_dynamic_shared = true;
        d.offset = fn.static_shared_size;
      } else {
        uint32_t off = (fn.static_shared_size + d.align - 1) / d.align * d.align;
        d.offset = off;
        fn.static_shared_size = off + d.size;
      }
      fn.shared.emplace(d.name, d);
    }
    for (auto& [name, d] : fn.shared)
      if (d.dynamic) d.offset = fn.static_shared_size;
  }
  return m;
}

}  // namespace vgpu::ptx
