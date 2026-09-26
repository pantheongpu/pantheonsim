// PTX parser for the VirtualGPU subset.
//
// Anything outside the subset throws Err::UnsupportedPtx naming the exact
// instruction, source line, and kernel — never a silent wrong answer.
#include "vgpu/ptx/parser.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <set>
#include <stdexcept>
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
      {"%lanemask_eq", Sreg::LaneMaskEq},
      {"%lanemask_lt", Sreg::LaneMaskLt},
      {"%lanemask_le", Sreg::LaneMaskLe},
      {"%lanemask_gt", Sreg::LaneMaskGt},
      {"%lanemask_ge", Sreg::LaneMaskGe},
      {"%warpid", Sreg::WarpId},
      {"%nwarpid", Sreg::NWarpId},
      {"%clock", Sreg::Clock},
      {"%clock_hi", Sreg::ClockHi},
      {"%clock64", Sreg::Clock64},
      {"%globaltimer", Sreg::GlobalTimer},
      {"%globaltimer_lo", Sreg::GlobalTimerLo},
      {"%globaltimer_hi", Sreg::GlobalTimerHi},
      {"%smid", Sreg::SmId},
      {"%nsmid", Sreg::NSmId},
      {"%dynamic_smem_size", Sreg::DynamicSmemSize},
      {"%total_smem_size", Sreg::TotalSmemSize},
      {"%gridid", Sreg::GridId},
      {"%clusterid.x", Sreg::ClusterIdX},
      {"%clusterid.y", Sreg::ClusterIdY},
      {"%clusterid.z", Sreg::ClusterIdZ},
      {"%nclusterid.x", Sreg::NClusterIdX},
      {"%nclusterid.y", Sreg::NClusterIdY},
      {"%nclusterid.z", Sreg::NClusterIdZ},
      {"%cluster_ctaid.x", Sreg::ClusterCtaIdX},
      {"%cluster_ctaid.y", Sreg::ClusterCtaIdY},
      {"%cluster_ctaid.z", Sreg::ClusterCtaIdZ},
      {"%cluster_nctaid.x", Sreg::ClusterNCtaIdX},
      {"%cluster_nctaid.y", Sreg::ClusterNCtaIdY},
      {"%cluster_nctaid.z", Sreg::ClusterNCtaIdZ},
      {"%cluster_ctarank", Sreg::ClusterCtaRank},
      {"%cluster_nctarank", Sreg::ClusterNCtaRank},
      {"%is_explicit_cluster", Sreg::IsExplicitCluster},
  };
  return t;
}

// ld/st/atom modifiers that are functionally inert for this engine
// (cache hints, memory orders, scopes).
bool inert_mem_modifier(const std::string& p) {
  static const std::set<std::string> inert = {"volatile", "nc", "ca", "cg", "cs", "lu",  "cv",
                                              "wb",       "wt", "relaxed", "acquire", "release",
                                              "acq_rel",  "cta", "gpu", "sys",
                                              // .noftz keeps subnormal halves rather than
                                              // flushing them, and this engine never flushes,
                                              // so it asks for the behaviour already in place.
                                              "noftz",
                                              // Cluster scope on an ordinary access is a
                                              // visibility promise, not a different address.
                                              "cluster"};
  return inert.count(p) > 0;
}

// Cache hints (.L2::128B, .L1::no_allocate, .L2::cache_hint) tell the hardware
// how far to prefetch and what to keep resident. They change how fast a load
// is, never what it returns, so an interpreter drops them -- and must not
// mistake one for a modifier it does not know.
bool is_cache_hint(const std::string& part) {
  return part.rfind("L1::", 0) == 0 || part.rfind("L2::", 0) == 0;
}

// ".shared::cta" is the explicit spelling of the ".shared" every kernel here
// already means. ".shared::cluster" is a different space -- memory in another
// block of a thread-block cluster -- so that one is left intact to be rejected
// by name rather than quietly treated as ordinary shared memory.
std::string normalize_scope(std::string part) {
  const std::string cta = "::cta";
  if (part.size() > cta.size() && part.compare(part.size() - cta.size(), cta.size(), cta) == 0)
    part.resize(part.size() - cta.size());
  return part;
}

// Dynamic shared memory lives above every static allocation, at the
// alignment its declaration states. Every extern .shared names the same
// bytes, so they share one offset. Starting it at the unaligned end of the
// static data put a TMA destination declared .align 16 at offset 8.
void place_dynamic_shared(EntryFn& fn) {
  uint32_t align = 1;
  for (const auto& [name, d] : fn.shared)
    if (d.dynamic) align = std::max(align, d.align);
  fn.dynamic_shared_offset = (fn.static_shared_size + align - 1) / align * align;
  for (auto& [name, d] : fn.shared)
    if (d.dynamic) d.offset = fn.dynamic_shared_offset;
}

std::vector<std::string> split_dots(const std::string& s) {
  std::vector<std::string> parts;
  std::string cur;
  for (char c : s) {
    if (c == '.') {
      if (!cur.empty() && !is_cache_hint(cur)) parts.push_back(normalize_scope(cur));
      cur.clear();
    } else {
      cur += c;
    }
  }
  if (!cur.empty() && !is_cache_hint(cur)) parts.push_back(normalize_scope(cur));
  return parts;
}

// Is this a SIMD video mnemonic? The name is v<op><lanes>, so the operation is
// everything between the leading v and the trailing lane count.
const char* video_simd_base(const std::string& op0) {
  static const char* kOps[] = {"add", "sub", "absdiff", "min", "max", "avrg"};
  const std::string base = op0.substr(1, op0.size() - 2);
  for (const char* o : kOps)
    if (base == o) return o;
  return nullptr;
}

// The comparison suffixes, shared by setp and set.
const std::unordered_map<std::string, CmpOp>& cmp_table() {
  static const std::unordered_map<std::string, CmpOp> t = {
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
  return t;
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
        version_ = m.version;
        continue;
      }
      if (t.text == ".target") {
        next();
        m.target = expect_word("target");
        while (peek_punct(",")) {
          next();
          m.target += "," + expect_word("target option");
        }
        target_ = m.target;
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
      // Linkage qualifiers. ".common" is a tentative definition -- zero
      // initialized, and merged with any other definition of the same symbol at
      // link time. Once a module is loaded there is nothing left to merge with,
      // so it declares exactly what ".global" does. Numba emits one per kernel.
      if (t.text == ".visible" || t.text == ".weak" || t.text == ".common") {
        next();
        continue;
      }
      if (t.text == ".pragma") {
        next();
        while (!at_end() && !peek_punct(";")) next();
        if (!at_end()) next();
        continue;
      }
      // ".file" names a source file for the ".loc" markers inside kernels. It
      // runs to the end of its line rather than to a semicolon.
      if (t.text == ".file" || t.text == ".loc") {
        size_t line = t.line;
        next();
        while (!at_end() && peek().line == line) next();
        continue;
      }
      // DWARF sections: a nested brace block of raw bytes. Skipped wholesale --
      // nothing here consumes debug info, and the contents are not PTX.
      if (t.text == ".section") {
        next();
        while (!at_end() && !peek_punct("{")) next();
        if (peek_punct("{")) {
          next();
          int depth = 1;
          while (!at_end() && depth > 0) {
            if (peek_punct("{")) ++depth;
            else if (peek_punct("}")) --depth;
            next();
          }
        }
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
      if (t.text == ".func" || t.text == ".weak" || t.text == ".visible") {
        // ".weak .func" and ".visible .func" are the linkage-qualified forms.
        // A qualifier on anything else falls through to the error below.
        size_t look = 0;
        if (t.text != ".func") {
          if (peek(1).kind != Token::Kind::Word || peek(1).text != ".func") {
            fail_unsupported(t.line, t.text, "", "linkage qualifier on an unsupported directive");
          }
          look = 1;
        }
        (void)look;
        next();
        if (t.text != ".func") next();  // consume ".func" after the qualifier
        auto fn = std::make_shared<EntryFn>();
        if (parse_device_func(fn.get())) m.funcs.push_back(std::move(fn));
        continue;
      }
      fail_unsupported(t.line, t.text, "",
                       "directive not in the implemented PTX subset (supported: .version .target "
                       ".address_size .visible .weak .common .extern .global .const .entry .file .loc "
                       ".section)");
    }
    resolve_calls(m);
    return m;
  }

 private:
  // Point every call at the function it names. Done after the whole module is
  // parsed because PTX declares prototypes first and a call can precede the
  // definition -- resolving eagerly would refuse a forward reference that is
  // perfectly well defined thirty lines later.
  void resolve_calls(Module& m) {
    std::unordered_map<std::string, std::shared_ptr<EntryFn>> by_name;
    for (auto& f : m.funcs) by_name[f->name] = f;
    // `with_tables` is false for the device functions themselves. Giving every
    // EntryFn a list of shared_ptrs to all of them makes each function hold a
    // shared_ptr to itself -- a reference cycle, so no .func is ever freed and
    // every module leaks its whole body. Nothing needs it there anyway: an
    // indirect call resolves against the *kernel's* table, which is the one
    // the interpreter reads.
    auto fix = [&](EntryFn& fn, bool with_tables) {
      if (with_tables) {
        fn.module_funcs.assign(m.funcs.begin(), m.funcs.end());
        fn.module_entry_names.clear();
        for (const auto& e : m.entries) fn.module_entry_names.push_back(e.name);
      }
      for (Instr& ins : fn.body) {
        auto* call = std::get_if<OpCall>(&ins.op);
        if (!call || call->indirect || call->callee.empty()) continue;
        if (call->callee == "vprintf" || call->callee == "__assertfail" ||
            call->callee == "malloc" || call->callee == "free")
          continue;
        // An unresolved name is left unresolved rather than refused here. A
        // separately compiled build links in CUDA's device-runtime library,
        // which declares functions the driver supplies (cnpGetLastError and
        // friends) and defines them nowhere in the module -- so refusing at
        // parse time rejected a whole program because a library function
        // nobody calls had no body. Execution reports it if it is ever
        // reached, which is the point at which it actually matters.
        auto it = by_name.find(call->callee);
        if (it != by_name.end()) call->target = it->second;
      }
    };
    for (auto& e : m.entries) fix(e, /*with_tables=*/true);
    for (auto& f : m.funcs) fix(*f, /*with_tables=*/false);
  }

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

  int64_t parse_int_literal(const std::string& word, size_t line) {
    // Every character has to be part of the number. std::stoll stops at the
    // first one that is not, so "12abc" used to be read as 12. The one tail a
    // number may carry is a C integer suffix -- nvcc writes "0x3fb8aa3bU" --
    // which is taken off before the digits are checked.
    std::string w = word;
    size_t suffix = 0;
    while (suffix < 3 && suffix < w.size() - 1 &&
           std::strchr("uUlL", w[w.size() - 1 - suffix]) != nullptr)
      ++suffix;
    const std::string tail = w.substr(w.size() - suffix);
    static const char* const kSuffixes[] = {"", "u", "l", "ul", "lu", "ll", "ull", "llu"};
    std::string lower = tail;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    bool known = false;
    for (const char* k : kSuffixes) known = known || lower == k;
    if (known) w.resize(w.size() - suffix);
    auto whole = [&](size_t used, size_t len) {
      if (!known || used != len) throw std::invalid_argument(word);
    };
    try {
      size_t used = 0;
      if (w.size() > 2 && w[0] == '0' && (w[1] == 'x' || w[1] == 'X')) {
        const std::string digits = w.substr(2);
        const uint64_t v = std::stoull(digits, &used, 16);
        whole(used, digits.size());
        return static_cast<int64_t>(v);
      }
      // Decimal immediates above INT64_MAX are legal PTX for .u64/.b64 operands
      // -- 9223372036854775808 is the sign-bit mask a float negation uses, and
      // ptxas prints it in decimal. Fall back to an unsigned parse and keep the
      // bit pattern; the instruction's type decides how it is read.
      try {
        const int64_t v = std::stoll(w, &used);
        whole(used, w.size());
        return v;
      } catch (const std::out_of_range&) {
        const uint64_t v = std::stoull(w, &used);
        whole(used, w.size());
        return static_cast<int64_t>(v);
      }
    } catch (const std::exception&) {
      fail(line, "bad integer literal '" + word + "'");
    }
  }

  // Negation of a literal, through unsigned arithmetic: "-9223372036854775808"
  // is INT64_MIN, and negating it as a signed value is undefined.
  static int64_t negate(int64_t v) { return static_cast<int64_t>(uint64_t{0} - static_cast<uint64_t>(v)); }

  // A float immediate's bits: 0fXXXXXXXX or 0dXXXXXXXXXXXXXXXX. A malformed one
  // used to escape as a bare std::invalid_argument with no line to go on.
  uint64_t parse_float_bits(const std::string& w, size_t line) {
    try {
      size_t used = 0;
      const uint64_t v = std::stoull(w.substr(2), &used, 16);
      if (used == w.size() - 2) return v;
    } catch (const std::exception&) {
    }
    fail(line, "bad float literal '" + w + "'");
  }

  // A count that sizes something: an array, a register bank. Negative counts
  // turned into enormous unsigned ones.
  uint64_t expect_count(const std::string& what) {
    const size_t line = peek().line;
    const int64_t v = expect_int(what);
    if (v < 0) fail(line, what + " cannot be negative");
    return static_cast<uint64_t>(v);
  }

  // Alignment is a power of two, and a sensible one; anything else made the
  // layout arithmetic divide by zero or round to nonsense.
  uint32_t expect_align() {
    const size_t line = peek().line;
    const int64_t v = expect_int("alignment");
    if (v < 1 || v > 65536 || (v & (v - 1)) != 0)
      fail(line, "alignment must be a power of two from 1 to 65536, got " + std::to_string(v));
    return static_cast<uint32_t>(v);
  }

  // The bytes a type occupies in memory. A predicate (.pred) has no storage
  // size, and as a parameter, a variable or a load type its zero size became a
  // zero alignment and a division by zero at launch.
  uint32_t storage_bytes(Type ty, size_t line, const std::string& what) {
    if (ty.bytes() == 0) fail(line, what + " has type " + ty.str() + ", which has no storage size");
    return ty.bytes();
  }

  // elems * bytes, refused when it does not fit in `limit`. Computed in 32 bits,
  // a .shared array of 2^30 32-bit elements wrapped to size 0 and aliased the
  // next array, and the profile's shared-memory limit never saw it.
  uint64_t checked_size(uint64_t elems, uint32_t bytes, uint64_t limit, size_t line,
                        const std::string& what) {
    if (bytes != 0 && elems > limit / bytes)
      fail(line, what + " is too large (" + std::to_string(elems) + " elements of " +
                     std::to_string(bytes) + " bytes)");
    return elems * bytes;
  }

  // An initializer value: an integer, or a float's bits written 0fXXXXXXXX or
  // 0dXXXXXXXXXXXXXXXX. Float initializers used to go through the integer
  // parse, which stopped at the 'f' and stored 0 -- so a .f32 global declared
  // "= 0f3F800000" (1.0) started as 0.0.
  int64_t expect_init_value(const std::string& what) {
    const Token& t = peek();
    if (t.kind == Token::Kind::Word && t.text.size() > 2 && t.text[0] == '0' &&
        ((t.text.size() == 10 && (t.text[1] == 'f' || t.text[1] == 'F')) ||
         (t.text.size() == 18 && (t.text[1] == 'd' || t.text[1] == 'D')))) {
      const std::string w = next().text;
      return static_cast<int64_t>(parse_float_bits(w, t.line));
    }
    return expect_int(what);
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
    return neg ? negate(v) : v;
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
        g.align = expect_align();
        continue;
      }
      break;
    }
    Type ty = expect_type(".global declaration");
    g.name = expect_word("global variable name");
    uint64_t elems = 1;
    if (peek_punct("[")) {
      next();
      elems = expect_count("array size");
      expect_punct("]");
    }
    g.size = checked_size(elems, storage_bytes(ty, line, "global '" + g.name + "'"), uint64_t{1} << 40, line,
                          "global '" + g.name + "'");
    if (g.size == 0) fail(line, "zero-sized global '" + g.name + "'");
    if (peek_punct("=")) {
      next();
      if (peek_punct("{")) {
        next();
        g.init.reserve(g.size);
        while (!peek_punct("}")) {
          // An element may be a symbol rather than a number -- a table of
          // function pointers is "{f, g, h}". Its slot is zeroed here and the
          // address written by the loader, which is the same treatment the
          // scalar "= symbol" form gets.
          if (peek().kind == Token::Kind::Word && peek().text == "generic" &&
              peek(1).kind == Token::Kind::Punct && peek(1).text == "(") {
            // "generic(sym)" casts a symbol's address into the generic window.
            // Every address in this model is already generic, so the cast is
            // the identity and only the symbol matters.
            next();
            next();
            g.init_symbols.push_back({g.init.size(), expect_word("symbol in generic()")});
            expect_punct(")");
            g.init.resize(g.init.size() + ty.bytes(), 0);
          } else if (peek().kind == Token::Kind::Word && is_identifier_start(peek().text[0])) {
            g.init_symbols.push_back({g.init.size(), next().text});
            g.init.resize(g.init.size() + ty.bytes(), 0);
          } else {
            int64_t v = expect_init_value("initializer element");
            // Little-endian element append.
            for (uint32_t b = 0; b < ty.bytes(); ++b)
              g.init.push_back(static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * b)) & 0xFF));
          }
          if (peek_punct(",")) next();
        }
        next();  // '}'
      } else if (peek().kind == Token::Kind::Word && peek().text == "generic" &&
                 peek(1).kind == Token::Kind::Punct && peek(1).text == "(") {
        next();
        next();
        g.init_symbols.push_back({0, expect_word("symbol in generic()")});
        expect_punct(")");
      } else if (peek().kind == Token::Kind::Word && is_identifier_start(peek().text[0])) {
        // "= some_symbol": the initialiser is another symbol's address, which
        // only exists once the module is loaded. The lexer gives numbers and
        // identifiers the same token kind, so the first character is what
        // separates them -- "= 5" is a value, not a symbol named "5".
        g.init_symbols.push_back({0, next().text});
      } else {
        int64_t v = expect_init_value("initializer");
        for (uint32_t b = 0; b < ty.bytes(); ++b)
          g.init.push_back(static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * b)) & 0xFF));
      }
      if (g.init.size() > g.size)
        fail(line, "initializer for '" + g.name + "' longer than its declared size");
      // A symbol initialiser leaves no bytes here: the address is written by
      // the loader once every global has one.
      g.init.resize(g.size, 0);
    }
    expect_punct(";");
    return g;
  }

  // ---- entry functions ----

  // Returns false when this was a declaration rather than a definition, in
  // which case *out is not meaningful.
  // .func [(.param .type func_retval0)] name ( .param .type name_param_0, ... )
  // followed by a body, or by ";" for a prototype.
  //
  // The parameters and the return value are handled as *call slots*, not as a
  // launch parameter buffer: every lane of a warp passes its own arguments, so
  // there is no single byte buffer to read them from. Registering the names in
  // call_slots_ before the body is parsed is what makes "ld.param [x_param_0]"
  // inside the function compile to a slot read rather than a kernel-parameter
  // read.
  bool parse_device_func(EntryFn* out) {
    EntryFn& fn = *out;
    cur_fn_ = &fn;
    fn.is_device_func = true;
    call_slots_.clear();
    declared_regs_.clear();
    if (peek_punct("(")) {
      next();
      const std::string kw = expect_word("'.param'");
      if (kw != ".param") fail(peek().line, "expected .param in a .func return value");
      if (peek().kind == Token::Kind::Word && peek().text == ".align") {
        next();
        expect_int("alignment");
      }
      const Type rty = expect_type("return value declaration");
      fn.retval_slot_name = expect_word("return value name");
      // An aggregate return: ".param .align 4 .b8 func_retval0[16]". The
      // element type is .b8 and the bracket carries the byte count.
      if (peek_punct("[")) {
        next();
        fn.retval_bytes = static_cast<uint32_t>(expect_int("return value size")) * rty.bytes();
        expect_punct("]");
      } else {
        fn.retval_bytes = rty.bytes();
      }
      call_slots_.insert(fn.retval_slot_name);
      expect_punct(")");
    }
    fn.name = expect_word("device function name");
    current_kernel_ = fn.name;
    if (peek_punct("(")) {
      next();
      while (!peek_punct(")")) {
        const std::string kw = expect_word("'.param'");
        if (kw != ".param") fail(peek().line, "expected .param in a .func signature");
        if (peek().kind == Token::Kind::Word && peek().text == ".align") {
          next();
          expect_int("alignment");
        }
        const Type pty = expect_type("parameter declaration");
        while (peek().kind == Token::Kind::Word && peek().text[0] == '.') next();  // ptr annotations
        const std::string pname = expect_word("parameter name");
        uint32_t pbytes = pty.bytes();
        if (peek_punct("[")) {  // a struct or array passed by value
          next();
          pbytes = static_cast<uint32_t>(expect_int("parameter size")) * pty.bytes();
          expect_punct("]");
        }
        fn.param_slot_bytes.push_back(pbytes);
        fn.param_slot_names.push_back(pname);
        call_slots_.insert(pname);
        if (peek_punct(",")) next();
      }
      next();
    }
    if (peek_punct(";")) {  // a prototype, not a definition
      next();
      current_kernel_.clear();
      cur_fn_ = nullptr;
      return false;
    }
    // .noreturn and friends may sit between the signature and the body.
    while (peek().kind == Token::Kind::Word && peek().text[0] == '.' && !peek_punct("{")) {
      if (peek().text == ".noreturn" || peek().text == ".pragma") {
        next();
        while (!at_end() && !peek_punct(";") && !peek_punct("{")) next();
        if (peek_punct(";")) next();
        continue;
      }
      break;
    }
    expect_punct("{");
    parse_body(fn);
    current_kernel_.clear();
    cur_fn_ = nullptr;
    return true;
  }

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
          p.align = expect_align();
        }
        Type ty = expect_type("parameter declaration");
        while (peek().kind == Token::Kind::Word && peek().text[0] == '.') {
          std::string ann = next().text;
          if (ann == ".align") p.align = expect_align();
          // .ptr / .global / .const: functional no-ops for us
        }
        p.name = expect_word("parameter name");
        p.ty = ty;
        p.size = storage_bytes(ty, peek().line, "parameter '" + p.name + "'");
        if (peek_punct("[")) {  // aggregate: .param .align 8 .b8 name[24]
          next();
          const size_t line = peek().line;
          p.size = static_cast<uint32_t>(checked_size(expect_count("parameter array size"), 1, UINT32_MAX, line,
                                                      "parameter '" + p.name + "'"));
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
      } else if (d == ".reqnctapercluster") {
        // __cluster_dims__(x,y,z). Fewer than three values means the trailing
        // dimensions are 1, the same shorthand .maxntid uses.
        next();
        std::array<uint32_t, 3> dims{1, 1, 1};
        for (int i = 0; i < 3; ++i) {
          dims[i] = static_cast<uint32_t>(expect_int("cluster dimension"));
          if (i < 2 && peek_punct(",")) next();
          else if (i < 2) break;
        }
        fn.req_cluster = dims;
      } else if (d == ".explicitcluster") {
        next();
        fn.explicit_cluster = true;
      } else if (d == ".maxclusterrank") {
        // A ceiling on cluster size for occupancy, not a shape. Recorded
        // nowhere because nothing here schedules by it, but it must be
        // consumed or the token stream desynchronizes.
        next();
        expect_int("directive value");
      } else if (d == ".minnctapersm" || d == ".maxnreg" || d == ".maxnctapersm") {
        next();
        uint32_t v = static_cast<uint32_t>(expect_int("directive value"));
        if (d == ".minnctapersm") fn.min_ctas_per_sm = v;
        if (d == ".maxnreg") fn.max_nreg = v;
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
    place_dynamic_shared(fn);
    current_kernel_.clear();
    cur_fn_ = nullptr;
    return true;
  }

  void parse_body(EntryFn& fn) {
    reg_scopes_.clear();
    // Labels are scoped to the { } block that defines them (PTX ISA 4.x
    // "Statements"), and inline asm relies on it: CUTLASS's barrier waits are
    // each a block with its own LAB_WAIT and DONE, so one kernel holds many.
    // A label is keyed by name and the block it was defined in, and a branch
    // takes the innermost definition visible from where it stands.
    std::unordered_map<std::string, size_t> labels;
    std::vector<std::pair<size_t, std::string>> bra_fixups;
    std::vector<std::vector<int>> fixup_scopes;   // the scope chain at each branch
    std::vector<int> scopes{0};                   // innermost last
    int next_scope = 1;
    auto key = [](const std::string& name, int scope) { return name + '\x01' + std::to_string(scope); };

    while (true) {
      const Token& t = peek();
      if (t.kind == Token::Kind::End) fail(t.line, "unexpected end of file inside kernel body");
      if (peek_punct("{")) {
        next();
        scopes.push_back(next_scope++);
        reg_scopes_.emplace_back();
        continue;
      }
      if (peek_punct("}")) {
        next();
        if (scopes.size() == 1) break;
        scopes.pop_back();
        reg_scopes_.pop_back();
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text == ".reg") {
        next();
        parse_reg_decl(fn);
        continue;
      }
      // ".reg.b16 hl, hu;" -- the type glued to the directive with no space.
      // CUDA's own headers write it this way inside inline asm: h2exp, h2log
      // and the rest of the half2 math family all begin with a line like that,
      // so a kernel calling any of them arrives here. The lexer sees one word,
      // and splitting it is the whole fix.
      if (t.kind == Token::Kind::Word && t.text.size() > 4 &&
          t.text.compare(0, 5, ".reg.") == 0) {
        const std::string type_part = t.text.substr(5);
        auto ty = parse_type_token(type_part);
        if (!ty)
          fail_unsupported(t.line, t.text, fn.name,
                           "register type '." + type_part + "' is not implemented");
        next();
        parse_reg_decl_of_type(fn, *ty);
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
        // ".param .align 4 .b8 retval0[16];" -- an aggregate call slot. The
        // alignment is a layout hint the slot does not need (it is a private
        // byte buffer, not device memory), but it must be consumed.
        if (peek().kind == Token::Kind::Word && peek().text == ".align") {
          next();
          expect_int("slot alignment");
        }
        Type ty = expect_type(".param slot declaration");
        d.name = expect_word("slot name");
        d.size = ty.bytes();
        if (peek_punct("[")) {
          next();
          d.size = static_cast<uint32_t>(expect_int("slot size")) * ty.bytes();
          expect_punct("]");
        }
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
      // Debug information. ".loc" marks a source line and ".pragma" carries
      // hints for the code generator; neither changes what the kernel computes,
      // and both appear in anything compiled with line tables -- Triton emits a
      // ".loc" per statement.
      if (t.kind == Token::Kind::Word && t.text == ".callprototype") {
        // "proto : .callprototype (.param .b32 _) _ (.param .b32 _, ...);"
        // The label was already consumed as a label. The signature would let a
        // call be type-checked, which nothing here does, so it is dropped --
        // but it must be consumed or the token stream desynchronizes.
        next();
        while (!at_end() && !peek_punct(";")) next();
        if (peek_punct(";")) next();
        continue;
      }
      if (t.kind == Token::Kind::Word && (t.text == ".loc" || t.text == ".pragma")) {
        next();
        while (!at_end() && !peek_punct(";") && peek().line == t.line) next();
        if (peek_punct(";")) next();
        continue;
      }
      if (t.kind == Token::Kind::Word && t.text[0] == '.') {
        fail_unsupported(t.line, t.text, fn.name, "directive not in the implemented subset");
      }
      // Label?
      if (t.kind == Token::Kind::Word && peek_punct(":", 1)) {
        if (!labels.emplace(key(t.text, scopes.back()), fn.body.size()).second)
          fail(t.line, "duplicate label '" + t.text + "'");
        next();
        next();
        continue;
      }
      fn.body.push_back(parse_instruction(fn, bra_fixups));
      while (fixup_scopes.size() < bra_fixups.size()) fixup_scopes.push_back(scopes);
    }

    for (size_t f = 0; f < bra_fixups.size(); ++f) {
      const auto& [idx, label] = bra_fixups[f];
      const std::vector<int>& chain = fixup_scopes[f];
      auto it = labels.end();
      for (size_t i = chain.size(); i-- > 0 && it == labels.end();) it = labels.find(key(label, chain[i]));
      if (it == labels.end())
        fail(fn.body[idx].line, "branch to undefined label '" + label + "' in kernel '" + fn.name + "'");
      std::get<OpBra>(fn.body[idx].op).target = it->second;
    }
  }

  static constexpr int64_t kMaxRegisterBank = 1 << 16;

  // Declares one register. Its id is fixed the first time the name is interned,
  // in the register file its width selects. A second declaration at another
  // width -- or a 64-bit declaration of a name already used as a 32-bit
  // register -- used to change the width and keep the narrow id, so a 64-bit
  // store wrote past the end of the 64-bit register file (and a value written
  // before the declaration read back as 0).
  // The name a register reference resolves to: the innermost open block's
  // own declaration if it has one, else the name itself.
  const std::string& scoped(const std::string& name) const {
    for (auto it = reg_scopes_.rbegin(); it != reg_scopes_.rend(); ++it) {
      auto f = it->find(name);
      if (f != it->end()) return f->second;
    }
    return name;
  }

  void declare_reg(EntryFn& fn, const std::string& source_name, Type ty, size_t line) {
    // Registers are scoped to the { } block that declares them, like labels,
    // and a declaration inside a block hides an outer one of the same name
    // until the block closes. CUDA's __syncthreads_and is inline asm that
    // declares its own %p1 and %p2 in braces; nvcc's code after the block uses
    // its own %p2, and taking the two as one register sent CUTLASS's split-K
    // semaphore wait round its loop without ever re-reading the semaphore.
    // A declaration that hides nothing keeps its name, so a block-local like
    // CUTLASS's "p" is interned exactly as before.
    std::string nm = source_name;
    if (!reg_scopes_.empty()) {
      auto& inner = reg_scopes_.back();
      if (auto f = inner.find(source_name); f != inner.end()) {
        nm = f->second;   // declared again in the same block
      } else if (fn.reg_decls.count(scoped(source_name))) {
        nm = source_name + "{" + std::to_string(++shadow_count_) + "}";
        inner[source_name] = nm;
      }
    }
    const bool wide = ty.bits > 32 && ty.kind != Type::Kind::Pred;
    if (fn.reg_ids.count(nm)) {
      auto was = fn.reg_wide.find(nm);
      const bool was_wide = was != fn.reg_wide.end() && was->second;
      if (was_wide != wide)
        fail(line, "register '" + nm + "' is declared " + ty.str() + " but was already " +
                       (was_wide ? "declared 64-bit" : "used or declared as a 32-bit register"));
    }
    fn.reg_decls[nm] = ty;
    declared_regs_.insert(source_name);
    declared_regs_.insert(nm);
    fn.reg_wide[nm] = wide;
    intern(nm);
  }

  void parse_reg_decl(EntryFn& fn) {
    parse_reg_decl_of_type(fn, expect_type(".reg declaration"));
  }

  void parse_reg_decl_of_type(EntryFn& fn, Type ty) {
    while (true) {
      std::string name = expect_word("register name");
      if (peek_punct("<")) {  // parameterized: .reg .b32 %r<6> declares %r0..%r5
        next();
        std::string count = expect_word("register count");
        const size_t line = peek().line;
        expect_punct(">");
        // Strictly a number, and bounded: std::atoi took "abc" as 0, and an
        // unbounded count was an unbounded allocation (%r<5000000> took 1.5 GB).
        const int64_t n = parse_int_literal(count, line);
        if (n < 1 || n > kMaxRegisterBank)
          fail(line, "register count must be 1 to " + std::to_string(kMaxRegisterBank) + ", got '" + count + "'");
        for (int64_t i = 0; i < n; ++i) declare_reg(fn, name + std::to_string(i), ty, line);
      } else {
        declare_reg(fn, name, ty, peek().line);
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
      d.align = expect_align();
    }
    Type ty = expect_type(".shared declaration");
    d.name = expect_word("shared variable name");
    uint64_t elems = 0;
    bool sized = false;
    if (peek_punct("[")) {
      next();
      if (!peek_punct("]")) {
        elems = expect_count("array size");
        sized = true;
      }
      expect_punct("]");
    } else {
      elems = 1;
      sized = true;
    }
    d.size = static_cast<uint32_t>(checked_size(elems, storage_bytes(ty, line, ".shared '" + d.name + "'"),
                                                UINT32_MAX, line, ".shared '" + d.name + "'"));
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
      const uint64_t off = (uint64_t{fn.static_shared_size} + d.align - 1) / d.align * d.align;
      if (off + d.size > UINT32_MAX) fail(line, ".shared memory of this kernel is too large");
      d.offset = static_cast<uint32_t>(off);
      fn.static_shared_size = static_cast<uint32_t>(off + d.size);
    }
    if (!fn.shared.emplace(d.name, d).second) fail(line, "duplicate .shared '" + d.name + "'");
  }

  void parse_local_decl(EntryFn& fn, size_t line) {
    LocalDecl d;
    while (peek().kind == Token::Kind::Word && peek().text == ".align") {
      next();
      d.align = expect_align();
    }
    Type ty = expect_type(".local declaration");
    d.name = expect_word("local variable name");
    uint64_t elems = 1;
    if (peek_punct("[")) {
      next();
      elems = expect_count("array size");
      expect_punct("]");
    }
    d.size = static_cast<uint32_t>(checked_size(elems, storage_bytes(ty, line, ".local '" + d.name + "'"),
                                                UINT32_MAX, line, ".local '" + d.name + "'"));
    if (d.align == 0) d.align = 8;
    const uint64_t off = (uint64_t{fn.local_frame_size} + d.align - 1) / d.align * d.align;
    if (off + d.size > UINT32_MAX) fail(line, ".local frame of this kernel is too large");
    d.offset = static_cast<uint32_t>(off);
    fn.local_frame_size = static_cast<uint32_t>(off + d.size);
    if (!fn.locals.emplace(d.name, d).second) fail(line, "duplicate .local '" + d.name + "'");
    expect_punct(";");
  }

  // ---- operands ----

  Operand parse_operand() {
    const Token& t = next();
    if (t.kind == Token::Kind::Punct && t.text == "-") {
      std::string w = expect_word("number after '-'");
      return ImmInt{negate(parse_int_literal(w, t.line))};
    }
    if (t.kind != Token::Kind::Word) fail(t.line, "expected operand, got '" + t.text + "'");
    const std::string& w = t.text;
    if (w[0] == '%') {
      auto it = sreg_table().find(w);
      if (it != sreg_table().end()) return SregOperand{it->second};
      // %envreg0 .. %envreg31 all read as zero; they are only distinguished by
      // number for a driver that sets them, and this one does not.
      // %envreg0..31 is a bank the driver fills in before the launch. Which
      // one is asked for matters: a cooperative launch puts the address of its
      // grid-barrier workspace in %envreg1 and %envreg2.
      if (w.rfind("%envreg", 0) == 0)
        return SregOperand{Sreg::EnvReg,
                           static_cast<uint32_t>(std::atoi(w.c_str() + 7))};
      // A %-name that was never declared is a special register this engine does
      // not implement, not a register that happens to be unwritten. Letting it
      // through as an ordinary register made %lanemask_le read as zero, which
      // turned CUB's radix sort into a store four bytes below its shared array
      // -- a silent wrong answer where an unimplemented instruction would have
      // said so plainly.
      if (!declared_regs_.count(w))
        fail_unsupported(t.line, w, cur_fn_ ? cur_fn_->name : std::string(),
                         "special register '" + w + "'");
      return RegOperand{intern(w)};
    }
    if (isdigit(static_cast<unsigned char>(w[0]))) {
      if (w.size() == 10 && w[0] == '0' && (w[1] == 'f' || w[1] == 'F'))
        return ImmFloatBits{parse_float_bits(w, t.line), 32};
      if (w.size() == 18 && w[0] == '0' && (w[1] == 'd' || w[1] == 'D'))
        return ImmFloatBits{parse_float_bits(w, t.line), 64};
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
  Reg intern(const std::string& source_name) {
    const std::string& name = scoped(source_name);
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

  // A braced list of any length. Texture coordinates come this way: the count
  // is fixed by the geometry, not by the syntax.
  std::vector<Operand> parse_operand_vector_any() {
    std::vector<Operand> ops;
    expect_punct("{");
    while (!peek_punct("}")) {
      ops.push_back(parse_operand());
      if (peek_punct(",")) next();
    }
    next();
    return ops;
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
      a.base_kind = Addr::Base::Reg;
      Reg r = intern(base);
      a.base = r.name;
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

  // The module's .target line, for the instructions only one target has.
  std::string target_;
  // And its .version, for the one instruction whose operand changed meaning.
  std::string version_;

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

    // Interned here rather than at each instruction's own construction site:
    // there are dozens of those and the first attempt reached two of them, so
    // the histogram counted almost nothing. This is the one point every
    // instruction passes through with its mnemonic in hand.
    ins.opcode_id = intern_opcode(parts[0]);

    const std::string& op0 = parts[0];
    if ((op0 == "st" || op0 == "red") && parts.size() > 1 && parts[1] == "async") {
      // st.async{.weak}.shared::cluster.mbarrier::complete_tx::bytes{.v2,.v4}.type [a], b, [mbar]
      // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.op.type [a], b, [mbar]
      OpStAsync op;
      op.red = op0 == "red";
      size_t vec = 1;
      bool have_ty = false, have_op = false, cluster_space = false, mbar = false;
      for (size_t i = 2; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "shared::cluster") cluster_space = true;
        else if (p == "mbarrier::complete_tx::bytes") mbar = true;
        else if (p == "weak" || p == "relaxed" || p == "cluster") ;
        else if (!op.red && p == "v2") vec = 2;
        else if (!op.red && p == "v4") vec = 4;
        else if (op.red && p == "add") { op.op = AtomOp::Add; have_op = true; }
        else if (op.red && p == "min") { op.op = AtomOp::Min; have_op = true; }
        else if (op.red && p == "max") { op.op = AtomOp::Max; have_op = true; }
        else if (op.red && p == "and") { op.op = AtomOp::And; have_op = true; }
        else if (op.red && p == "or") { op.op = AtomOp::Or; have_op = true; }
        else if (op.red && p == "xor") { op.op = AtomOp::Xor; have_op = true; }
        else if (op.red && p == "inc") { op.op = AtomOp::Inc; have_op = true; }
        else if (op.red && p == "dec") { op.op = AtomOp::Dec; have_op = true; }
        else if (auto t = parse_type_token(p)) { op.ty = *t; have_ty = true; }
        else return unsupported(op0 + ".async modifier '." + p + "'");
      }
      if (!cluster_space || !mbar)
        return unsupported(op0 + ".async is implemented for .shared::cluster with "
                                 ".mbarrier::complete_tx::bytes");
      if (!have_ty || (op.red && !have_op)) return unsupported(op0 + ".async form");
      if (op.ty.bits != 32 && op.ty.bits != 64) return unsupported(op0 + ".async of 32- and 64-bit types only");
      op.addr = parse_addr(fn);
      expect_punct(",");
      if (vec == 1) op.srcs.push_back(parse_operand());
      else op.srcs = parse_operand_vector_any();
      if (op.srcs.size() != vec) return unsupported("st.async vector arity");
      expect_punct(",");
      op.mbar = parse_addr(fn);
      for (const Addr* a : {&op.addr, &op.mbar})
        if (a->base_kind == Addr::Base::CallSlot || a->base_kind == Addr::Base::EntryParam)
          return unsupported(op0 + ".async through a parameter/slot name");
      ins.op = op;
    } else if (op0 == "ld" || op0 == "st") {
      Space space = Space::Generic;
      size_t vec = 1;
      Type ty{};
      bool have_ty = false;
      bool acquire = false, release = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "acquire") acquire = true;
        else if (p == "release") release = true;
        else if (p == "param") space = Space::Param;
        else if (p == "global") space = Space::Global;
        // __constant__ variables are parsed into the module's globals, so a
        // constant-bank read is a global read of a range nothing writes. The
        // read-only-ness is a promise the program makes, not one this engine
        // has to enforce -- a kernel that writes there is already invalid.
        else if (p == "const") space = Space::Global;
        // A .shared::cluster address names its block itself (see mapa), so
        // it is a shared access like any other once decoded.
        else if (p == "shared" || p == "shared::cluster") space = Space::Shared;
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
      storage_bytes(ty, ins.line, opcode);
      if (op0 == "ld") {
        Addr addr;
        std::vector<Reg> dsts;
        // A one-element braced list is legal PTX for a scalar load, and it is
        // what Triton's inline-asm loads look like: "ld.global.b32 { %r1 },
        // [ %rd1 ];". The braces carry no meaning the vector count does not.
        if (vec == 1 && !peek_punct("{")) {
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
        if (vec == 1 && !peek_punct("{"))
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
      // The ordering the kernel asked for, on whichever op this became.
      if (auto* l = std::get_if<OpLd>(&ins.op)) l->acquire = acquire;
      if (auto* st = std::get_if<OpSt>(&ins.op)) st->release = release;
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
        else if (parts[i] == "shared" || parts[i] == "shared::cluster") { space = Space::Shared; ++i; }
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
    } else if (op0 == "cvt" && parts.size() > 1 && parts[1] == "pack") {
      // cvt.pack.sat.{u16,s16}.s32 d, a, b
      // cvt.pack.sat.{u8,s8,u4,s4,u2,s2}.s32.b32 d, a, b, c
      if (parts.size() < 5 || parts[2] != "sat" || parts[4] != "s32")
        return unsupported("cvt.pack form (expected cvt.pack.sat.<type>.s32[.b32])");
      const std::string& to = parts[3];
      OpCvtPack op;
      if (to.size() < 2 || (to[0] != 'u' && to[0] != 's'))
        return unsupported("cvt.pack to ." + to);
      op.is_signed = to[0] == 's';
      op.bits = static_cast<uint32_t>(std::atoi(to.c_str() + 1));
      if (op.bits != 16 && op.bits != 8 && op.bits != 4 && op.bits != 2)
        return unsupported("cvt.pack to ." + to);
      op.has_c = op.bits != 16;
      if (op.has_c != (parts.size() == 6 && parts[5] == "b32") || parts.size() > 6)
        return unsupported(op.has_c ? "cvt.pack to 8, 4 or 2 bits takes .b32 and a c operand"
                                    : "cvt.pack to 16 bits takes no c operand");
      op.dst = expect_reg_operand("cvt.pack destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      if (op.has_c) {
        expect_punct(",");
        op.c = parse_operand();
      }
      ins.op = op;
    } else if (op0 == "cvt") {
      // cvt[.round][.sat][.ftz].<dstty>.<srcty>
      std::vector<Type> tys;
      std::string packed;  // "f16x2"/"bf16x2": two f32 sources packed into one register
      std::string fp8;     // "e4m3x2"/"e5m2x2": the FP8 side of the conversion
      bool satfinite = false, sat = false, ftz = false;
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
        else if (p == "sat") sat = true;
        else if (p == "ftz") ftz = true;
        else if (p == "f16x2" || p == "bf16x2") packed = p;
        else if (p == "e4m3x2" || p == "e5m2x2") fp8 = p;
        else if (p == "satfinite") satfinite = true;
        else if (auto t2 = parse_type_token(p)) tys.push_back(*t2);
        else return unsupported("unrecognized cvt modifier '." + p + "'");
      }
      if (!fp8.empty()) {
        OpCvtFp8 op;
        op.e5m2 = fp8[1] == '5';
        op.satfinite = satfinite;
        // Which side of the dot the fp8 type sat on decides the direction, and
        // `packed`/`tys` carry whatever the other side was.
        const size_t fp8_pos = opcode.find(fp8);
        const size_t other_pos = packed.empty() ? std::string::npos : opcode.find(packed);
        op.to_fp8 = other_pos == std::string::npos ? !tys.empty() : fp8_pos < other_pos;
        op.bf16 = !packed.empty() && packed[0] == 'b';
        if (!packed.empty()) {
          op.src_f32_pair = false;
        } else {
          if (tys.size() != 1 || tys[0].bits != 32 || !tys[0].is_float())
            return unsupported("cvt between " + fp8 + " and a type other than f32/f16x2/bf16x2");
          op.src_f32_pair = true;
        }
        if (!op.to_fp8 && op.src_f32_pair)
          return unsupported("cvt from " + fp8 + " to f32 (PTX unpacks to f16x2)");
        op.dst = expect_reg_operand("cvt destination");
        expect_punct(",");
        op.a = parse_operand();
        if (op.to_fp8 && op.src_f32_pair) {
          expect_punct(",");
          op.b = parse_operand();
        }
        ins.op = op;
        expect_punct(";");
        return ins;
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
      op.sat = sat;
      op.ftz = ftz;
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
    } else if ((op0 == "add" || op0 == "sub" || op0 == "mul" || op0 == "fma" || op0 == "neg" ||
                op0 == "abs" || op0 == "min" || op0 == "max") &&
               (opcode.find("f16") != std::string::npos ||
                opcode.find("bf16") != std::string::npos)) {
      // Half-precision arithmetic in all four shapes: f16, f16x2, bf16, bf16x2.
      bool is_bf = false, is_packed = false, saw_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p2 = parts[i];
        if (p2 == "f16") { saw_ty = true; }
        else if (p2 == "f16x2") { saw_ty = true; is_packed = true; }
        else if (p2 == "bf16") { saw_ty = true; is_bf = true; }
        else if (p2 == "bf16x2") { saw_ty = true; is_bf = true; is_packed = true; }
        else if (p2 == "rn" || p2 == "ftz" || p2 == "sat" || p2 == "rz" || p2 == "rm" ||
                 p2 == "rp" || p2 == "NaN" || p2 == "xorsign" || p2 == "abs") ;
        else return unsupported("half-precision modifier '." + p2 + "'");
      }
      if (!saw_ty) return unsupported("half-precision form without a type");
      Reg dst = expect_reg_operand("destination");
      expect_punct(",");
      Operand a = parse_operand();
      if (op0 == "neg" || op0 == "abs") {
        ins.op = OpF16x2Neg{is_bf, is_packed, op0 == "abs", dst, a};
      } else {
        expect_punct(",");
        Operand b = parse_operand();
        if (op0 == "fma") {
          expect_punct(",");
          Operand c = parse_operand();
          ins.op = OpF16x2Fma{is_bf, is_packed, dst, a, b, c};
        } else {
          FloatBinOp fop = op0 == "add"   ? FloatBinOp::Add
                           : op0 == "sub" ? FloatBinOp::Sub
                           : op0 == "min" ? FloatBinOp::Min
                           : op0 == "max" ? FloatBinOp::Max
                                          : FloatBinOp::Mul;
          ins.op = OpF16x2Bin{fop, is_bf, is_packed, dst, a, b};
        }
      }
    } else if (op0 == "wmma") {
      // wmma.load.{a,b,c}.sync.aligned.<layout>.<shape>[.space].<type> {r...}, [p] [, stride]
      // wmma.store.d.sync.aligned.<layout>.<shape>[.space].<type> [p], {r...} [, stride]
      // wmma.mma[.op.popc].sync.aligned.<alayout>.<blayout>.<shape>[.rnd].<types> {d}, {a}, {b}, {c}
      if (parts.size() < 2) return unsupported("wmma form");
      const std::string& kind = parts[1];
      std::vector<MatLayout> layouts;
      Space space = Space::Generic;
      char frag = 0;
      WmmaShape shape;
      bool have_shape = false, satfinite = false, popc = false, b1_and = false, b1_xor = false;
      FRound rnd = FRound::Nearest;
      std::vector<WmmaType> types;
      static const std::unordered_map<std::string, WmmaShape> kShapes = {
          {"m16n16k16", {16, 16, 16}}, {"m8n32k16", {8, 32, 16}}, {"m32n8k16", {32, 8, 16}},
          {"m16n16k8", {16, 16, 8}},   {"m8n8k4", {8, 8, 4}},     {"m8n8k32", {8, 8, 32}},
          {"m8n8k128", {8, 8, 128}}};
      static const std::unordered_map<std::string, WmmaType> kTypes = {
          {"f16", WmmaType::F16}, {"bf16", WmmaType::BF16}, {"tf32", WmmaType::TF32},
          {"f32", WmmaType::F32}, {"f64", WmmaType::F64},   {"s8", WmmaType::S8},
          {"u8", WmmaType::U8},   {"s4", WmmaType::S4},     {"u4", WmmaType::U4},
          {"b1", WmmaType::B1},   {"s32", WmmaType::S32}};
      for (size_t i = 2; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync" || p == "aligned") ;
        else if (p == "a" || p == "b" || p == "c" || p == "d") frag = p[0];
        else if (p == "row") layouts.push_back(MatLayout::Row);
        else if (p == "col") layouts.push_back(MatLayout::Col);
        else if (auto sh = kShapes.find(p); sh != kShapes.end()) { shape = sh->second; have_shape = true; }
        else if (auto ty = kTypes.find(p); ty != kTypes.end()) types.push_back(ty->second);
        else if (p == "global") space = Space::Global;
        else if (p == "shared") space = Space::Shared;
        else if (p == "satfinite") satfinite = true;
        else if (p == "popc") popc = true;
        else if (p == "xor") b1_xor = true;
        else if (p == "and") b1_and = true;
        else if (p == "rn") rnd = FRound::Nearest;
        else if (p == "rz") rnd = FRound::Zero;
        else if (p == "rm") rnd = FRound::MinusInf;
        else if (p == "rp") rnd = FRound::PlusInf;
        else return unsupported("wmma modifier '." + p + "'");
      }
      if (!have_shape) return unsupported("wmma needs a shape");
      const bool k16 = shape.k == 16 && (shape == WmmaShape{16, 16, 16} || shape == WmmaShape{8, 32, 16} ||
                                         shape == WmmaShape{32, 8, 16});
      // Which types a multiplicand or accumulator of this shape may have.
      auto ab_ok = [&](WmmaType t) {
        if (k16) return t == WmmaType::F16 || t == WmmaType::BF16 || t == WmmaType::S8 || t == WmmaType::U8;
        if (shape == WmmaShape{16, 16, 8}) return t == WmmaType::TF32;
        if (shape == WmmaShape{8, 8, 4}) return t == WmmaType::F64;
        if (shape == WmmaShape{8, 8, 32}) return t == WmmaType::S4 || t == WmmaType::U4;
        return t == WmmaType::B1;
      };
      auto acc_ok = [&](WmmaType t) {
        if (k16) return t == WmmaType::F16 || t == WmmaType::F32 || t == WmmaType::S32;
        if (shape == WmmaShape{16, 16, 8}) return t == WmmaType::F32;
        if (shape == WmmaShape{8, 8, 4}) return t == WmmaType::F64;
        return t == WmmaType::S32;
      };
      const bool legacy_shape = shape == WmmaShape{16, 16, 16} || shape == WmmaShape{16, 16, 8};
      auto arity = [&](char f, WmmaType t, size_t have) -> bool { return wmma_geom(f, shape, t).regs == have; };
      // The stride is optional: the matrix's own leading dimension otherwise.
      auto stride_or_default = [&](char f, WmmaType t, MatLayout lay) -> Operand {
        if (peek_punct(",")) {
          next();
          return parse_operand();
        }
        const WmmaGeom g = wmma_geom(f, shape, t);
        return Operand{ImmInt{static_cast<int64_t>(lay == MatLayout::Row ? g.cols : g.rows)}};
      };
      if (kind == "load") {
        if (frag != 'a' && frag != 'b' && frag != 'c') return unsupported("wmma.load needs .a, .b or .c");
        if (types.size() != 1) return unsupported("wmma.load takes one element type");
        const WmmaType t = types[0];
        if (frag == 'c' ? !acc_ok(t) : !ab_ok(t)) return unsupported("wmma.load of this type at this shape");
        OpWmmaLoad op;
        op.which = frag == 'a' ? OpWmmaLoad::Which::A : frag == 'b' ? OpWmmaLoad::Which::B : OpWmmaLoad::Which::C;
        op.layout = layouts.empty() ? MatLayout::Row : layouts[0];
        op.space = space;
        op.shape = shape;
        op.type = t;
        if ((t == WmmaType::S4 || t == WmmaType::U4 || t == WmmaType::B1) &&
            op.layout != (frag == 'a' ? MatLayout::Row : MatLayout::Col))
          return unsupported("sub-byte and single-bit wmma loads A row-major and B column-major");
        // The layouts this simulator has always used stay as they were.
        if (frag == 'c') op.generic = !(legacy_shape && t == WmmaType::F32);
        else op.generic = !((shape == WmmaShape{16, 16, 16} && (t == WmmaType::F16 || t == WmmaType::BF16)) ||
                            t == WmmaType::TF32);
        op.f32 = frag == 'c';
        op.elem = t == WmmaType::BF16 ? WmmaElem::BF16 : t == WmmaType::TF32 ? WmmaElem::TF32 : WmmaElem::F16;
        op.dsts = parse_reg_vector_any();
        if (!arity(frag, t, op.dsts.size()))
          return unsupported("wmma.load fragment arity (expected " +
                             std::to_string(wmma_geom(frag, shape, t).regs) + " registers)");
        expect_punct(",");
        op.addr = parse_addr(fn);
        op.stride = stride_or_default(frag, t, op.layout);
        ins.op = op;
      } else if (kind == "store") {
        if (frag != 'd') return unsupported("wmma.store stores .d");
        if (types.size() != 1 || !acc_ok(types[0])) return unsupported("wmma.store of this type at this shape");
        OpWmmaStore op;
        op.layout = layouts.empty() ? MatLayout::Row : layouts[0];
        op.space = space;
        op.shape = shape;
        op.type = types[0];
        op.generic = !(legacy_shape && op.type == WmmaType::F32);
        op.addr = parse_addr(fn);
        expect_punct(",");
        for (auto& r : parse_reg_vector_any()) op.src.push_back(Operand{RegOperand{r}});
        if (!arity('d', op.type, op.src.size()))
          return unsupported("wmma.store fragment arity (expected " +
                             std::to_string(wmma_geom('d', shape, op.type).regs) + " registers)");
        op.stride = stride_or_default('d', op.type, op.layout);
        ins.op = op;
      } else if (kind == "mma") {
        if (layouts.size() != 2) return unsupported("wmma.mma needs both A and B layouts");
        OpWmmaMma op;
        op.alayout = layouts[0];
        op.blayout = layouts[1];
        op.shape = shape;
        // f16 multiplicands leave their type implicit: .dtype.ctype only.
        if (types.size() == 2) {
          op.dtype = types[0];
          op.atype = op.btype = WmmaType::F16;
          op.ctype = types[1];
        } else if (types.size() == 4) {
          op.dtype = types[0];
          op.atype = types[1];
          op.btype = types[2];
          op.ctype = types[3];
        } else if (types.size() == 3) {
          // .dtype.atype.btype with the accumulator left off, which the parser
          // has always taken to mean ctype = dtype (every non-f16 form has
          // them equal anyway).
          op.dtype = op.ctype = types[0];
          op.atype = types[1];
          op.btype = types[2];
        } else {
          return unsupported("wmma.mma types (.dtype.ctype, or .dtype.atype.btype.ctype)");
        }
        const WmmaType at = op.atype;
        const bool fp16 = at == WmmaType::F16, integer = at == WmmaType::S8 || at == WmmaType::U8 ||
                                                   at == WmmaType::S4 || at == WmmaType::U4;
        if (!ab_ok(at) || op.btype != at) return unsupported("wmma.mma: A and B must have the same type, valid for the shape");
        const bool acc_valid =
            fp16 ? (op.dtype == WmmaType::F16 || op.dtype == WmmaType::F32) &&
                       (op.ctype == WmmaType::F16 || op.ctype == WmmaType::F32)
            : at == WmmaType::F64 ? op.dtype == WmmaType::F64 && op.ctype == WmmaType::F64
            : (at == WmmaType::BF16 || at == WmmaType::TF32) ? op.dtype == WmmaType::F32 && op.ctype == WmmaType::F32
                                                             : op.dtype == WmmaType::S32 && op.ctype == WmmaType::S32;
        if (!acc_valid) return unsupported("wmma.mma accumulator types for these multiplicands");
        if (satfinite && !integer && at != WmmaType::B1)
          return unsupported("wmma.mma .satfinite is for integer multiplicands (removed for floating point in PTX 6.5)");
        if (at == WmmaType::B1) {
          if (!popc || b1_and == b1_xor) return unsupported("single-bit wmma.mma needs .xor.popc or .and.popc");
          if (satfinite) return unsupported("single-bit wmma.mma has no .satfinite");
        } else if (popc || b1_and || b1_xor) {
          return unsupported(".popc is for single-bit wmma.mma");
        }
        if (rnd != FRound::Nearest && at != WmmaType::F64) return unsupported("a rounding mode on a non-f64 wmma.mma");
        if ((at == WmmaType::S4 || at == WmmaType::U4 || at == WmmaType::B1) &&
            (op.alayout != MatLayout::Row || op.blayout != MatLayout::Col))
          return unsupported("sub-byte and single-bit wmma.mma is .row.col");
        op.satfinite = satfinite;
        op.b1_and = b1_and;
        op.rnd = rnd;
        op.generic = !((shape == WmmaShape{16, 16, 16} && (fp16 || at == WmmaType::BF16) &&
                        op.ctype == WmmaType::F32 && op.dtype == WmmaType::F32) ||
                       at == WmmaType::TF32);
        op.elem = at == WmmaType::BF16 ? WmmaElem::BF16 : at == WmmaType::TF32 ? WmmaElem::TF32 : WmmaElem::F16;
        op.d = parse_reg_vector_any();
        expect_punct(",");
        op.a = parse_reg_vector_any();
        expect_punct(",");
        op.b = parse_reg_vector_any();
        expect_punct(",");
        op.c = parse_reg_vector_any();
        if (!arity('a', op.atype, op.a.size()) || !arity('b', op.btype, op.b.size()) ||
            !arity('c', op.ctype, op.c.size()) || !arity('d', op.dtype, op.d.size()))
          return unsupported("wmma.mma fragment arity for this shape and these types");
        ins.op = op;
      } else {
        return unsupported("wmma." + kind + " is not implemented (only .load, .mma and .store.d)");
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
      bool packed_half = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        // approx/rn/rz/ftz/full select precision on hardware; VirtualGPU always
        // computes at host precision (documented divergence).
        if (p == "approx" || p == "rn" || p == "rz" || p == "rm" || p == "rp" || p == "ftz" ||
            p == "full")
          ;
        else if (p == "f16x2") { ty = Type{Type::Kind::F, 16}; have_ty = true; packed_half = true; }
        else if (p == "bf16x2") { ty = Type{Type::Kind::BF, 16}; have_ty = true; packed_half = true; }
        else if (auto t2 = parse_type_token(p)) {
          ty = *t2;
          have_ty = true;
        } else return unsupported("unrecognized modifier '." + p + "'");
      }
      if (!have_ty) fail(ins.line, opcode + " missing type");
      if (!ty.is_real() || (ty.bits != 16 && ty.bits != 32 && ty.bits != 64))
        return unsupported(op0 + " on '" + parts.back() + "'");
      OpMath op;
      op.op = mops.at(op0);
      op.ty = ty;
      op.packed = packed_half;
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
    } else if (op0 == "ldmatrix") {
      uint32_t count = 0;
      bool trans = false, shape_ok = false, b16 = false, shared_space = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync" || p == "aligned") ;
        else if (p == "m8n8") shape_ok = true;
        else if (p == "x1") count = 1;
        else if (p == "x2") count = 2;
        else if (p == "x4") count = 4;
        else if (p == "trans") trans = true;
        else if (p == "b16") b16 = true;
        else if (p == "shared") shared_space = true;
        else if (p == "cta") ;  // scope qualifier on .shared::cta
        else return unsupported("ldmatrix modifier '." + p + "'");
      }
      if (!shape_ok || !count || !b16)
        return unsupported("only ldmatrix.m8n8.x{1,2,4}.b16 is implemented");
      OpLdMatrix op;
      op.count = count;
      op.trans = trans;
      op.shared_space = shared_space;
      op.dsts = parse_reg_vector_any();
      if (op.dsts.size() != count) return unsupported("ldmatrix destination arity");
      expect_punct(",");
      op.addr = parse_addr(fn);
      ins.op = op;
    } else if (op0 == "stmatrix") {
      // The same modifiers as ldmatrix, and the same restriction: the 8x8 b16
      // shapes, which is what this engine's matrix fragments are.
      uint32_t count = 0;
      bool trans = false, shape_ok = false, b16 = false, shared_space = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync" || p == "aligned") ;
        else if (p == "m8n8") shape_ok = true;
        else if (p == "x1") count = 1;
        else if (p == "x2") count = 2;
        else if (p == "x4") count = 4;
        else if (p == "trans") trans = true;
        else if (p == "b16") b16 = true;
        else if (p == "shared") shared_space = true;
        else if (p == "cta") ;  // scope qualifier on .shared::cta
        else return unsupported("stmatrix modifier '." + p + "'");
      }
      if (!shape_ok || !count || !b16)
        return unsupported("only stmatrix.m8n8.x{1,2,4}.b16 is implemented");
      OpStMatrix op;
      op.count = count;
      op.trans = trans;
      op.shared_space = shared_space;
      op.addr = parse_addr(fn);
      expect_punct(",");
      op.srcs = parse_operand_vector_any();
      if (op.srcs.size() != count) return unsupported("stmatrix source arity");
      ins.op = op;
    } else if (op0 == "mma") {
      // mma.sync.aligned.m16n8kK.row.col.<d>.<a>.<b>.<c>
      // mma.sp[::ordered_metadata].sync.aligned.m16n8kK.row.col.<d>.<a>.<b>.<c> d, a, b, c, e, f
      uint32_t k = 0;
      std::vector<std::string> types;
      bool row_col = false, saw_row = false, sparse = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync" || p == "aligned") ;
        else if (p == "sp" || p == "sp::ordered_metadata") sparse = true;
        else if (p == "row") saw_row = true;
        else if (p == "col") { if (saw_row) row_col = true; }
        else if (p == "m16n8k8") k = 8;
        else if (p == "m16n8k16") k = 16;
        else if (p == "m16n8k32") k = 32;
        else if (p == "f32" || p == "f16" || p == "bf16" || p == "tf32" || p == "s32" ||
                 p == "s8" || p == "u8")
          types.push_back(p);
        else return unsupported("mma modifier '." + p + "' (shape or type not implemented)");
      }
      if (!k) return unsupported("only the m16n8k{8,16,32} mma shapes are implemented");
      if (!row_col) return unsupported("only mma .row.col is implemented");
      if (types.size() != 4) return unsupported("mma needs .<dtype>.<atype>.<btype>.<ctype>");
      OpMma op;
      op.k = k;
      const std::string& at = types[1];
      if (at == "f16") op.ab_type = MmaElem::F16;
      else if (at == "bf16") op.ab_type = MmaElem::BF16;
      else if (at == "tf32") op.ab_type = MmaElem::TF32;
      else if (at == "s8") { op.ab_type = MmaElem::S8; op.ab_signed = true; }
      else if (at == "u8") { op.ab_type = MmaElem::U8; op.ab_signed = false; }
      else return unsupported("mma operand type '." + at + "'");
      if (types[1] != types[2]) return unsupported("mma with mixed A and B types");
      op.acc_f16 = types[0] == "f16";
      op.acc_int = types[0] == "s32";
      if (sparse && op.ab_type != MmaElem::F16 && op.ab_type != MmaElem::BF16)
        return unsupported("mma.sp with ." + at + " (only .f16 and .bf16 sparse A are implemented)");
      if (sparse && k == 8) return unsupported("mma.sp.m16n8k8 (f16/bf16 sparse A is k16 or k32)");
      op.sparse = sparse;
      op.d = parse_reg_vector_any();
      expect_punct(",");
      op.a = parse_reg_vector_any();
      expect_punct(",");
      op.b = parse_reg_vector_any();
      expect_punct(",");
      op.c = parse_reg_vector_any();
      if (sparse) {
        expect_punct(",");
        op.meta = parse_operand();
        expect_punct(",");
        op.selector = parse_operand();
        const auto* sel = std::get_if<ImmInt>(&op.selector);
        if (!sel || sel->value < 0 || sel->value > 1)
          return unsupported("mma.sp's sparsity selector is an immediate 0 or 1 for .f16/.bf16");
      }
      if (op.d.size() != op.c.size()) return unsupported("mma D and C arity differ");
      ins.op = op;
    } else if (op0 == "wgmma") {
      // wgmma.fence.sync.aligned;  wgmma.commit_group.sync.aligned;
      // wgmma.wait_group.sync.aligned N;
      // wgmma.mma_async.sync.aligned.m64nNkK.<dtype>.<atype>.<btype>[.satfinite]
      //     d, a-desc|{a}, b-desc, scale-d[, imm-scale-a, imm-scale-b[, imm-trans-a], imm-trans-b];
      if (parts.size() < 2) return unsupported("wgmma form");
      // Arch-specific: sm_90a and nothing else, not sm_90 and not Blackwell
      // (which replaced it with tcgen05). ptxas refuses it anywhere else, so
      // a module that uses it under another target is not a real program.
      if (target_.rfind("sm_90a", 0) != 0)
        fail(ins.line, "wgmma requires .target sm_90a; this module targets " +
                           (target_.empty() ? std::string("nothing") : target_));
      OpWgmma op;
      const std::string& what = parts[1];
      if (what == "fence") op.kind = WgmmaKind::Fence;
      else if (what == "commit_group") op.kind = WgmmaKind::Commit;
      else if (what == "wait_group") op.kind = WgmmaKind::Wait;
      else if (what == "mma_async") op.kind = WgmmaKind::Mma;
      else return unsupported("wgmma." + what);
      std::vector<std::string> types;
      bool shape = false;
      for (size_t i = 2; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        unsigned mm = 0, nn = 0, kk = 0;
        char tail = 0;
        if (p == "sync" || p == "aligned") ;
        else if (p == "satfinite") op.satfinite = true;
        else if (p == "sp") return unsupported("wgmma.mma_async.sp (sparse A)");
        else if (std::sscanf(p.c_str(), "m%un%uk%u%c", &mm, &nn, &kk, &tail) == 3) {
          if (mm != 64 || nn == 0 || nn > 256 || nn % 8 != 0)
            return unsupported("wgmma shape '." + p + "'");
          op.n = nn;
          op.k = kk;
          shape = true;
        } else if (p == "f16" || p == "bf16" || p == "tf32" || p == "f32" || p == "s32" ||
                   p == "e4m3" || p == "e5m2" || p == "s8" || p == "u8")
          types.push_back(p);
        else if (p == "b1" || p == "and" || p == "popc")
          return unsupported("wgmma single-bit (.b1) forms");
        else return unsupported("wgmma modifier '." + p + "'");
      }
      if (op.kind == WgmmaKind::Wait) {
        const Operand n = parse_operand();
        const auto* imm = std::get_if<ImmInt>(&n);
        if (!imm || imm->value < 0) return unsupported("wgmma.wait_group needs a constant count");
        op.wait_n = static_cast<uint32_t>(imm->value);
      }
      if (op.kind != WgmmaKind::Mma) {
        ins.op = op;
      } else {
        if (!shape || types.size() != 3)
          return unsupported("wgmma.mma_async needs .m64nNkK.<dtype>.<atype>.<btype>");
        auto elem = [&](const std::string& t, WgmmaElem* out) {
          if (t == "f16") *out = WgmmaElem::F16;
          else if (t == "bf16") *out = WgmmaElem::BF16;
          else if (t == "tf32") *out = WgmmaElem::TF32;
          else if (t == "e4m3") *out = WgmmaElem::E4M3;
          else if (t == "e5m2") *out = WgmmaElem::E5M2;
          else if (t == "s8") *out = WgmmaElem::S8;
          else if (t == "u8") *out = WgmmaElem::U8;
          else return false;
          return true;
        };
        if (!elem(types[1], &op.a_type) || !elem(types[2], &op.b_type))
          return unsupported("wgmma multiplicand types ." + types[1] + "." + types[2]);
        if (types[0] == "f16") op.d_type = WgmmaAcc::F16;
        else if (types[0] == "f32") op.d_type = WgmmaAcc::F32;
        else if (types[0] == "s32") op.d_type = WgmmaAcc::S32;
        else return unsupported("wgmma accumulator type ." + types[0]);
        // The combinations the ISA defines (9.7.17.3), and the K each implies.
        const bool is16 = op.a_type == WgmmaElem::F16 || op.a_type == WgmmaElem::BF16;
        const bool fp8 = op.a_type == WgmmaElem::E4M3 || op.a_type == WgmmaElem::E5M2;
        const bool int8 = op.a_type == WgmmaElem::S8 || op.a_type == WgmmaElem::U8;
        const bool b_fp8 = op.b_type == WgmmaElem::E4M3 || op.b_type == WgmmaElem::E5M2;
        const bool b_int8 = op.b_type == WgmmaElem::S8 || op.b_type == WgmmaElem::U8;
        bool ok = false;
        uint32_t want_k = 0;
        if (is16) {
          ok = op.a_type == op.b_type &&
               (op.d_type == WgmmaAcc::F32 || (op.a_type == WgmmaElem::F16 && op.d_type == WgmmaAcc::F16));
          want_k = 16;
        } else if (op.a_type == WgmmaElem::TF32) {
          ok = op.b_type == WgmmaElem::TF32 && op.d_type == WgmmaAcc::F32;
          want_k = 8;
        } else if (fp8) {
          ok = b_fp8 && op.d_type != WgmmaAcc::S32;
          want_k = 32;
        } else if (int8) {
          ok = b_int8 && op.d_type == WgmmaAcc::S32;
          want_k = 32;
          // The integer shapes skip some N: 8, 16, 24, 32, then multiples of 16.
          if (op.n > 32 && op.n % 16 != 0) ok = false;
        }
        if (!ok || op.k != want_k)
          return unsupported("wgmma.mma_async." + types[0] + "." + types[1] + "." + types[2] +
                             " with k" + std::to_string(op.k) + " is not a form the ISA defines");
        if (op.satfinite && !int8) return unsupported("wgmma .satfinite outside the integer forms");
        op.d = parse_reg_vector_any();
        const size_t want_d = op.d_type == WgmmaAcc::F16 ? op.n / 4 : op.n / 2;
        if (op.d.size() != want_d)
          return unsupported("wgmma accumulator arity (expected " + std::to_string(want_d) + ")");
        expect_punct(",");
        if (peek_punct("{")) {
          op.a_regs = true;
          op.a = parse_reg_vector_any();
          if (op.a.size() != 4) return unsupported("wgmma A fragment arity (expected 4)");
        } else {
          op.a_desc = parse_operand();
        }
        expect_punct(",");
        op.b_desc = parse_operand();
        expect_punct(",");
        op.scale_d = parse_operand();
        // The immediates that follow depend on the form: floats take the two
        // negate flags, and the 16-bit forms add the transposes -- trans-b only
        // when A is in registers, since a register fragment has no major-ness.
        std::vector<int> imms;
        while (peek_punct(",")) {
          next();
          const Operand o = parse_operand();
          const auto* imm = std::get_if<ImmInt>(&o);
          if (!imm) return unsupported("wgmma scale/transpose arguments must be immediates");
          imms.push_back(static_cast<int>(imm->value));
        }
        const size_t want_imms = int8 ? 0u : (is16 ? (op.a_regs ? 3u : 4u) : 2u);
        if (imms.size() != want_imms)
          return unsupported("wgmma.mma_async takes " + std::to_string(want_imms) +
                             " immediate arguments after scale-d in this form");
        if (!int8) {
          op.scale_a = imms[0];
          op.scale_b = imms[1];
          if ((op.scale_a != 1 && op.scale_a != -1) || (op.scale_b != 1 && op.scale_b != -1))
            return unsupported("wgmma imm-scale-a/imm-scale-b must be 1 or -1");
        }
        if (is16) {
          if (op.a_regs) op.trans_b = imms[2];
          else { op.trans_a = imms[2]; op.trans_b = imms[3]; }
          if ((op.trans_a != 0 && op.trans_a != 1) || (op.trans_b != 0 && op.trans_b != 1))
            return unsupported("wgmma imm-trans-a/imm-trans-b must be 0 or 1");
        }
        ins.op = op;
      }
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
    } else if (op0 == "dp4a" || op0 == "dp2a") {
      // dp4a.atype.btype d, a, b, c
      // dp2a.{lo,hi}.atype.btype d, a, b, c
      OpDp4a op;
      op.two = op0 == "dp2a";
      const size_t t = op.two ? 2 : 1;
      if (parts.size() != t + 2) return unsupported(op0 + " form");
      if (op.two) {
        if (parts[1] != "lo" && parts[1] != "hi") return unsupported("dp2a needs .lo or .hi");
        op.hi = parts[1] == "hi";
      }
      const bool as = parts[t] == "s32", au = parts[t] == "u32";
      const bool bs = parts[t + 1] == "s32", bu = parts[t + 1] == "u32";
      if ((!as && !au) || (!bs && !bu)) return unsupported(op0 + " operand types");
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
    } else if (op0.size() > 3 && op0[0] == 'v' &&
               (op0.back() == '2' || op0.back() == '4') &&
               video_simd_base(op0) != nullptr) {
      // vadd4 / vsub4 / vabsdiff4 / vmin4 / vmax4 / vavrg4, and the 2-way forms.
      OpVideoSimd op;
      op.lanes = op0.back() == '4' ? 4u : 2u;
      const std::string base = op0.substr(1, op0.size() - 2);
      if (base == "add") op.op = VideoOp::Add;
      else if (base == "sub") op.op = VideoOp::Sub;
      else if (base == "absdiff") op.op = VideoOp::AbsDiff;
      else if (base == "min") op.op = VideoOp::Min;
      else if (base == "max") op.op = VideoOp::Max;
      else if (base == "avrg") op.op = VideoOp::Avrg;
      else return unsupported("SIMD video op '" + op0 + "'");
      std::vector<std::string> tys;
      for (size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] == "sat") { op.sat = true; continue; }
        // The secondary-operation forms fold the lanes into a scalar with c,
        // which is a different instruction wearing the same name.
        if (parts[i] == "add" || parts[i] == "min" || parts[i] == "max")
          return unsupported(op0 + " with a secondary '." + parts[i] + "' operation");
        tys.push_back(parts[i]);
      }
      if (tys.size() != 3) return unsupported(op0 + " form (expected .dtype.atype.btype)");
      auto sgn = [](const std::string& t) { return !t.empty() && t[0] == 's'; };
      op.d_signed = sgn(tys[0]);
      op.a_signed = sgn(tys[1]);
      op.b_signed = sgn(tys[2]);
      op.dst = expect_reg_operand(op0 + " destination");
      expect_punct(",");
      op.a = parse_operand();
      // The byte/halfword selector forms pick which lane of the source each
      // lane reads. Nothing emits them here yet, and guessing at the selection
      // would silently permute the result.
      if (peek_punct(".")) return unsupported(op0 + " with a lane selector");
      expect_punct(",");
      op.b = parse_operand();
      if (peek_punct(".")) return unsupported(op0 + " with a lane selector");
      expect_punct(",");
      op.c = parse_operand();
      ins.op = op;
    } else if (op0 == "bfind") {
      OpBfind op;
      size_t ti = 1;
      if (parts.size() > 1 && parts[1] == "shiftamt") { op.shiftamt = true; ti = 2; }
      if (ti >= parts.size()) return unsupported("bfind form (expected bfind[.shiftamt].type)");
      auto bty = parse_type_token(parts[ti]);
      if (!bty || (bty->bits != 32 && bty->bits != 64))
        return unsupported("bfind type '." + parts[ti] + "' (only 32- and 64-bit)");
      op.ty = *bty;
      op.dst = expect_reg_operand("bfind destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "elect") {
      if (parts.size() != 2 || parts[1] != "sync")
        return unsupported("elect form (expected elect.sync)");
      OpElect op;
      // The syntax is "d|p" or just "|p": the leader's lane id, the predicate,
      // or both. A bare "|" means only the predicate is wanted.
      if (!peek_punct("|")) op.dst = expect_reg_operand("elect destination");
      if (peek_punct("|")) {
        next();
        op.pred_dst = expect_reg_operand("elect predicate destination");
      }
      expect_punct(",");
      op.membermask = parse_operand();
      ins.op = op;
    } else if (op0 == "tensormap" && parts.size() > 1 && parts[1] == "replace") {
      // tensormap.replace.tile.<field>{.global|.shared::cta}.b1024.{b32|b64} [addr], {ord,} new_val
      OpTensormapReplace op;
      bool tile = false, b1024 = false, have_field = false, b64 = false, have_ty = false;
      static const std::unordered_map<std::string, TmapField> fields = {
          {"global_address", TmapField::GlobalAddress}, {"rank", TmapField::Rank},
          {"box_dim", TmapField::BoxDim}, {"global_dim", TmapField::GlobalDim},
          {"global_stride", TmapField::GlobalStride}, {"element_stride", TmapField::ElementStride},
          {"elemtype", TmapField::ElemType}, {"interleave_layout", TmapField::InterleaveLayout},
          {"swizzle_mode", TmapField::SwizzleMode}, {"swizzle_atomicity", TmapField::SwizzleAtomicity},
          {"fill_mode", TmapField::FillMode}};
      for (size_t i = 2; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "tile") tile = true;
        else if (auto f = fields.find(p); f != fields.end()) { op.field = f->second; have_field = true; }
        else if (p == "global") op.space = Space::Global;
        else if (p == "shared") op.space = Space::Shared;
        else if (p == "b1024") b1024 = true;
        else if (p == "b32") have_ty = true;
        else if (p == "b64") { have_ty = true; b64 = true; }
        else return unsupported("tensormap.replace modifier '." + p + "'");
      }
      if (!tile || !have_field || !b1024 || !have_ty)
        return unsupported("tensormap.replace form (expected tensormap.replace.tile.<field>.b1024.<type>)");
      // Arch-specific, like wgmma: sm_90a, and the a and f targets after it.
      {
        const char last = target_.empty() ? ' ' : target_.back();
        int sm = 0;
        std::sscanf(target_.c_str(), "sm_%d", &sm);
        if (sm < 90 || (last != 'a' && last != 'f'))
          fail(ins.line, "tensormap.replace requires an arch-specific target (.target sm_90a or "
                         "later a/f targets); this module targets " + (target_.empty() ? std::string("nothing") : target_));
      }
      const bool wide_field = op.field == TmapField::GlobalAddress || op.field == TmapField::GlobalStride;
      if (b64 != wide_field)
        return unsupported("tensormap.replace takes .b64 for global_address and global_stride and "
                           ".b32 for every other field");
      const bool per_dim = op.field == TmapField::BoxDim || op.field == TmapField::GlobalDim ||
                           op.field == TmapField::GlobalStride || op.field == TmapField::ElementStride;
      const bool field3 = op.field >= TmapField::ElemType;
      op.addr = parse_addr(fn);
      expect_punct(",");
      if (per_dim) {
        auto ord = parse_operand();
        auto* imm = std::get_if<ImmInt>(&ord);
        if (!imm || imm->value < 0 || imm->value > 4)
          return unsupported("tensormap.replace's ordinal is an immediate from 0 to 4");
        op.ord = static_cast<uint32_t>(imm->value);
        expect_punct(",");
      }
      op.value = parse_operand();
      if (field3 && !std::holds_alternative<ImmInt>(op.value))
        return unsupported("tensormap.replace's ." + parts[3] + " value is an immediate");
      // The ISA does not say what unit global_stride is in, and it changed:
      // CuTe hands it the stride in bytes when compiled by CUDA 12.5 or later
      // and the stride shifted right by 4 before that ("4 LSBs are not
      // included", cute/arch/copy_sm90_desc.hpp). CUDA 12.5 is PTX ISA 8.5,
      // so the module's .version decides.
      if (op.field == TmapField::GlobalStride) {
        int major = 0, minor = 0;
        std::sscanf(version_.c_str(), "%d.%d", &major, &minor);
        op.stride_in_16b = major < 8 || (major == 8 && minor < 5);
      }
      if (op.addr.base_kind == Addr::Base::CallSlot)
        return unsupported("tensormap.replace through a call slot");
      ins.op = op;
    } else if (op0 == "tensormap" && parts.size() > 1 && parts[1] == "cp_fenceproxy") {
      // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.<scope>.sync.aligned [dst], [src], 128
      bool global = false, shared = false;
      for (size_t i = 2; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "global") global = true;
        else if (p == "shared") shared = true;
        else if (p == "tensormap::generic" || p == "release" || p == "sync" || p == "aligned" ||
                 inert_mem_modifier(p)) ;
        else return unsupported("tensormap.cp_fenceproxy modifier '." + p + "'");
      }
      if (!global || !shared)
        return unsupported("tensormap.cp_fenceproxy copies from .shared::cta to .global");
      OpTensormapCopy op;
      op.dst = parse_addr(fn);
      expect_punct(",");
      op.src = parse_addr(fn);
      expect_punct(",");
      auto size = parse_operand();
      auto* imm = std::get_if<ImmInt>(&size);
      if (!imm || imm->value != 128)
        return unsupported("tensormap.cp_fenceproxy copies 128 bytes, the size of a tensor map");
      for (const Addr* a : {&op.dst, &op.src})
        if (a->base_kind == Addr::Base::CallSlot)
          return unsupported("tensormap.cp_fenceproxy through a call slot");
      ins.op = op;
    } else if (op0 == "mapa" || op0 == "getctarank") {
      // mapa{.shared::cluster}.{u32,u64} d, a, rank
      // getctarank{.shared::cluster}.{u32,u64} d, a
      bool shared = false, wide = false, have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] == "shared::cluster") shared = true;
        else if (parts[i] == "u32") have_ty = true;
        else if (parts[i] == "u64") { have_ty = true; wide = true; }
        else return unsupported(op0 + " modifier '." + parts[i] + "'");
      }
      if (!have_ty) return unsupported(op0 + " needs .u32 or .u64");
      if (op0 == "mapa") {
        OpMapa op;
        op.generic = !shared;
        op.wide = wide;
        op.dst = expect_reg_operand("mapa destination");
        expect_punct(",");
        op.src = parse_operand();
        expect_punct(",");
        op.rank = parse_operand();
        ins.op = op;
      } else {
        OpGetCtaRank op;
        op.generic = !shared;
        op.dst = expect_reg_operand("getctarank destination");
        expect_punct(",");
        op.src = parse_operand();
        ins.op = op;
      }
    } else if (op0 == "isspacep") {
      if (parts.size() != 2) return unsupported("isspacep form (expected isspacep.space)");
      OpIsSpacep op;
      if (parts[1] == "global") op.space = Space::Global;
      else if (parts[1] == "shared") op.space = Space::Shared;
      else if (parts[1] == "shared::cluster") { op.space = Space::Shared; op.cluster = true; }
      else if (parts[1] == "local") op.space = Space::Local;
      else if (parts[1] == "const") op.space = Space::Global;
      else return unsupported("isspacep space '." + parts[1] + "'");
      op.dst = expect_reg_operand("isspacep destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "griddepcontrol" || op0 == "setmaxnreg") {
      // Both are scheduling directives with no effect on what a kernel
      // computes. griddepcontrol orders a grid against its predecessor, and
      // launches here are synchronous, so the predecessor has already finished
      // by the time this executes -- .wait has nothing to wait for and
      // .launch_dependents nothing to release. setmaxnreg reshapes a
      // warpgroup's register budget, and there is no architectural register
      // file to reshape.
      while (!at_end() && !peek_punct(";")) next();
      ins.op = OpNop{};
    } else if (op0 == "mbarrier") {
      // mbarrier.<op>[.parity][.space][.sem][.scope].b64 ...
      OpMbarrier op;
      bool have_op = false;
      bool saw_shared = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p2 = parts[i];
        if (p2 == "init") { op.op = MbarOp::Init; have_op = true; }
        else if (p2 == "expect_tx" && have_op && op.op == MbarOp::Arrive) op.expect_tx = true;
        else if (p2 == "expect_tx") { op.op = MbarOp::ExpectTx; have_op = true; }
        else if (p2 == "complete_tx") { op.op = MbarOp::CompleteTx; have_op = true; }
        else if (p2 == "noComplete") op.no_complete = true;
        else if (p2 == "shared::cluster") op.cluster = true;
        else if (p2.rfind("phase_type::", 0) == 0)
          return unsupported("mbarrier ." + p2 + " (the report-carrying phase types)");
        else if (p2 == "inval") { op.op = MbarOp::Inval; have_op = true; }
        else if (p2 == "arrive") { op.op = MbarOp::Arrive; have_op = true; }
        else if (p2 == "arrive_drop") { op.op = MbarOp::ArriveDrop; have_op = true; }
        else if (p2 == "test_wait") { op.op = MbarOp::TestWait; have_op = true; }
        else if (p2 == "try_wait") { op.op = MbarOp::TryWait; have_op = true; }
        else if (p2 == "pending_count") { op.op = MbarOp::PendingCount; have_op = true; }
        else if (p2 == "parity") op.parity = true;
        else if (p2 == "shared") saw_shared = true;
        else if (p2 == "b64") ;
        else if (inert_mem_modifier(p2)) ;
        else return unsupported("mbarrier modifier '." + p2 + "'");
      }
      if (!have_op) return unsupported("mbarrier form");
      (void)saw_shared;  // an mbarrier is a shared object whether or not it says so
      // A barrier in another block can be arrived at and have bytes counted
      // against it; initializing, invalidating and waiting are for its own
      // block (9.7.13.15).
      if (op.cluster && op.op != MbarOp::Arrive && op.op != MbarOp::ArriveDrop &&
          op.op != MbarOp::ExpectTx && op.op != MbarOp::CompleteTx)
        return unsupported("mbarrier on .shared::cluster is defined for arrive, arrive_drop, "
                           "expect_tx and complete_tx only");
      // init, inval and the transaction counts take no destination; everything
      // else writes one -- or names the sink, `_`, to discard it.
      if (op.op != MbarOp::Init && op.op != MbarOp::Inval && op.op != MbarOp::ExpectTx &&
          op.op != MbarOp::CompleteTx) {
        if (peek().text == "_") next();
        else if (op.cluster)
          // Another block's phase means nothing to this one, so the ISA has
          // a remote arrive discard its state.
          return unsupported("mbarrier.arrive on .shared::cluster returns no state; its "
                             "destination must be the sink `_`");
        else op.dst = expect_reg_operand("mbarrier destination");
        expect_punct(",");
      }
      op.addr = parse_addr(fn);
      if (op.addr.base_kind == Addr::Base::CallSlot || op.addr.base_kind == Addr::Base::EntryParam)
        return unsupported("mbarrier through a parameter/slot name");
      if (peek_punct(",")) {
        next();
        if (op.op == MbarOp::TestWait || op.op == MbarOp::TryWait) {
          op.state = parse_operand();
          op.have_state = true;
          // try_wait's optional suspend-time hint: how long a thread may sleep
          // before re-testing, which changes nothing a wait returns.
          if (op.op == MbarOp::TryWait && peek_punct(",")) {
            next();
            (void)parse_operand();
          }
        } else {
          op.count = parse_operand();
          op.have_count = true;
        }
      }
      if (op.op == MbarOp::Init && !op.have_count)
        return unsupported("mbarrier.init without an expected arrival count");
      if ((op.op == MbarOp::ExpectTx || op.op == MbarOp::CompleteTx || op.expect_tx) && !op.have_count)
        return unsupported("mbarrier transaction count missing");
      if (op.op == MbarOp::ExpectTx || op.op == MbarOp::CompleteTx || op.expect_tx)
        if (peek_punct(",")) return unsupported("mbarrier multicast (a ctaMask operand)");
      ins.op = op;
    } else if (op0 == "match") {
      // match.any.sync.b32 d, a, membermask
      // match.all.sync.b32 d|p, a, membermask
      if (parts.size() != 4 || parts[2] != "sync")
        return unsupported("match form (expected match.{any,all}.sync.b{32,64})");
      OpMatch op;
      if (parts[1] == "any") op.all = false;
      else if (parts[1] == "all") op.all = true;
      else return unsupported("match mode '." + parts[1] + "'");
      auto mty = parse_type_token(parts[3]);
      if (!mty || (mty->bits != 32 && mty->bits != 64))
        return unsupported("match type '." + parts[3] + "' (only .b32 and .b64)");
      op.dst = expect_reg_operand("match destination");
      if (op.all && peek_punct("|")) {
        next();
        op.pred_dst = expect_reg_operand("match.all predicate destination");
      }
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.membermask = parse_operand();
      ins.op = op;
    } else if (op0 == "mul24") {
      if (parts.size() != 3 || (parts[1] != "lo" && parts[1] != "hi"))
        return unsupported("mul24 form (expected mul24.{lo,hi}.{u32,s32})");
      OpMul24 op;
      op.hi = parts[1] == "hi";
      if (parts[2] == "s32") op.is_signed = true;
      else if (parts[2] != "u32") return unsupported("mul24 type '." + parts[2] + "'");
      op.dst = expect_reg_operand("mul24 destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      ins.op = op;
    } else if (op0 == "szext") {
      if (parts.size() != 3 || (parts[1] != "clamp" && parts[1] != "wrap"))
        return unsupported("szext form (expected szext.{clamp,wrap}.{u32,s32})");
      OpSzext op;
      op.wrap = parts[1] == "wrap";
      if (parts[2] == "s32") op.is_signed = true;
      else if (parts[2] != "u32") return unsupported("szext type '." + parts[2] + "'");
      op.dst = expect_reg_operand("szext destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      ins.op = op;
    } else if (op0 == "fns") {
      if (parts.size() != 2 || parts[1] != "b32")
        return unsupported("fns form (expected fns.b32)");
      OpFns op;
      op.dst = expect_reg_operand("fns destination");
      expect_punct(",");
      op.mask = parse_operand();
      expect_punct(",");
      op.base = parse_operand();
      expect_punct(",");
      op.offset = parse_operand();
      ins.op = op;
    } else if (op0 == "vabsdiff") {
      // The scalar form is |a-b|+c, which is sad's definition, so it shares
      // the node. The SIMD forms (vabsdiff2/vabsdiff4) split the operands into
      // halves or bytes and are a different instruction with a similar name.
      if (parts.size() != 4) return unsupported("vabsdiff form (expected vabsdiff.dtype.atype.btype)");
      OpSad op;
      auto vty = parse_type_token(parts[1]);
      if (!vty) return unsupported("vabsdiff type '." + parts[1] + "'");
      op.ty = *vty;
      op.dst = expect_reg_operand("vabsdiff destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.c = parse_operand();
      ins.op = op;
    } else if (op0 == "lop3") {
      // lop3.b32 d, a, b, c, immLut
      if (parts.size() < 2 || parts[1] != "b32") return unsupported("lop3 form (only lop3.b32)");
      OpLop3 op;
      op.dst = expect_reg_operand("lop3 destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.c = parse_operand();
      expect_punct(",");
      op.lut = static_cast<uint8_t>(expect_int("lop3 immLut"));
      ins.op = op;
    } else if (op0 == "slct") {
      // slct.dtype.stype d, a, b, c. Only the selector's type matters to the
      // comparison; the data type just says how wide the result is.
      if (parts.size() != 3) return unsupported("slct form (expected slct.dtype.stype)");
      OpSlct op;
      auto slct_ty = parse_type_token(parts[1]);
      if (!slct_ty) return unsupported("slct data type '." + parts[1] + "'");
      op.ty = *slct_ty;
      if (parts[2] == "s32") op.c_is_float = false;
      else if (parts[2] == "f32") op.c_is_float = true;
      else return unsupported("slct selector type '." + parts[2] + "' (only .s32 and .f32)");
      op.dst = expect_reg_operand("slct destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.c = parse_operand();
      ins.op = op;
    } else if (op0 == "testp") {
      if (parts.size() != 3) return unsupported("testp form (expected testp.op.ftype)");
      OpTestp op;
      if (parts[1] == "finite") op.op = TestpOp::Finite;
      else if (parts[1] == "infinite") op.op = TestpOp::Infinite;
      else if (parts[1] == "number") op.op = TestpOp::Number;
      else if (parts[1] == "notanumber") op.op = TestpOp::NotANumber;
      else if (parts[1] == "normal") op.op = TestpOp::Normal;
      else if (parts[1] == "subnormal") op.op = TestpOp::Subnormal;
      else return unsupported("testp predicate '." + parts[1] + "'");
      auto testp_ty = parse_type_token(parts[2]);
      if (!testp_ty || !testp_ty->is_float() ||
          (testp_ty->bits != 32 && testp_ty->bits != 64))
        return unsupported("testp type '." + parts[2] + "' (only .f32 and .f64)");
      op.ty = *testp_ty;
      op.dst = expect_reg_operand("testp destination");
      expect_punct(",");
      op.a = parse_operand();
      ins.op = op;
    } else if (op0 == "sad") {
      if (parts.size() != 2) return unsupported("sad form (expected sad.type)");
      OpSad op;
      auto sad_ty = parse_type_token(parts[1]);
      if (!sad_ty) return unsupported("sad type '." + parts[1] + "'");
      op.ty = *sad_ty;
      op.dst = expect_reg_operand("sad destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      expect_punct(",");
      op.c = parse_operand();
      ins.op = op;
    } else if (op0 == "prefetch" || op0 == "prefetchu" || op0 == "createpolicy" ||
               op0 == "applypriority" || op0 == "discard") {
      // Cache-management hints. Every one of these says where data should be
      // kept or how long, and nothing about what a load returns -- the same
      // reason .L2::128B and .lu are already dropped. There is no cache model
      // here, so honouring them and ignoring them produce identical results,
      // and refusing the kernel over one would fail it for a performance note.
      //
      // createpolicy writes a register (the policy handle), so it cannot be
      // skipped outright: an unwritten destination would be read later and
      // diagnosed as read-before-write. It gets a zero handle, which is what a
      // policy nothing consults is worth.
      if (op0 == "createpolicy") {
        OpMov mv;
        mv.ty = Type{Type::Kind::B, 64};
        mv.dst = expect_reg_operand("createpolicy destination");
        mv.src = ImmInt{0};
        ins.op = mv;
        while (!at_end() && !peek_punct(";")) next();
      } else {
        while (!at_end() && !peek_punct(";")) next();
        ins.op = OpNop{};
      }
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
    } else if (op0 == "atom" || op0 == "red") {
      // atom[.space][.sem][.scope].<op>.<type> d, [a], b [, c]
      // red[.space][.sem][.scope].<op>.<type>    [a], b        -- same, no d.
      const bool discards = (op0 == "red");
      Space space = Space::Generic;
      std::optional<AtomOp> aop;
      Type ty{};
      bool have_ty = false;
      bool packed_half = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "global") space = Space::Global;
        else if (p == "shared" || p == "shared::cluster") space = Space::Shared;
        else if (inert_mem_modifier(p)) ;
        else if (p == "add") aop = AtomOp::Add;
        else if (p == "min") aop = AtomOp::Min;
        else if (p == "max") aop = AtomOp::Max;
        else if (p == "and") aop = AtomOp::And;
        else if (p == "or") aop = AtomOp::Or;
        else if (p == "xor") aop = AtomOp::Xor;
        else if (p == "exch") aop = AtomOp::Exch;
        else if (p == "cas") aop = AtomOp::Cas;
        else if (p == "inc") aop = AtomOp::Inc;
        else if (p == "dec") aop = AtomOp::Dec;
        else if (p == "f16x2") { ty = Type{Type::Kind::F, 16}; have_ty = true; packed_half = true; }
        else if (p == "bf16x2") { ty = Type{Type::Kind::BF, 16}; have_ty = true; packed_half = true; }
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
      if ((*aop == AtomOp::Inc || *aop == AtomOp::Dec) &&
          !(ty.kind == Type::Kind::U && ty.bits == 32))
        return unsupported("atom.inc/.dec are defined for .u32 only");
      if (ty.is_float()) {
        if (*aop != AtomOp::Add && *aop != AtomOp::Exch && *aop != AtomOp::Min &&
            *aop != AtomOp::Max)
          return unsupported("atom." + std::string(*aop == AtomOp::Cas ? "cas" : "bitwise") +
                             " on a float type (CUDA has no such instruction; use .b32)");
        if (ty.bits == 16 && *aop != AtomOp::Add)
          return unsupported("only atom.add is defined for f16/bf16");
        if (ty.bits != 16 && ty.bits != 32 && ty.bits != 64)
          return unsupported("float atomics are implemented for f16, bf16, f32 and f64");
      }
      if (discards && *aop == AtomOp::Cas)
        return unsupported("red.cas (a compare-and-swap whose result is discarded "
                           "cannot report whether it swapped)");
      OpAtom op;
      op.op = *aop;
      op.ty = ty;
      op.space = space;
      op.discards_result = discards;
      op.packed_half = packed_half;
      if (!discards) {
        op.dst = expect_reg_operand("atom destination");
        expect_punct(",");
      }
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
               op0 == "shl" || op0 == "shr" || op0 == "addc" || op0 == "subc") {
      // "addc"/"subc" are "add"/"sub" that also read the carry bit.
      const bool carry_in = (op0 == "addc" || op0 == "subc");
      const std::string base_op = carry_in ? op0.substr(0, 3) : op0;
      bool carry_out = false;
      bool nan_propagate = false;
      bool wide = false, lo = false, hi = false;
      FRound frnd = FRound::Nearest;
      Type ty{};
      bool have_ty = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "wide") wide = true;
        else if (p == "lo") lo = true;
        else if (p == "hi") hi = true;
        else if (p == "cc") carry_out = true;
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
        // min.NaN/max.NaN propagate a NaN operand instead of returning the
        // other one. That is exactly what fmin/fmax do NOT do, so it cannot be
        // dropped -- it is handled at execution, and recorded here.
        else if (p == "NaN") nan_propagate = true;
        // .xorsign.abs takes the magnitude and xors the signs; a semantic
        // change rather than an accuracy one.
        else if (p == "xorsign" || p == "abs")
          return unsupported("modifier '." + p + "' on " + op0 + " is not implemented");
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
        if (base_op != "mul") return unsupported("'." + base_op + ".hi' is not implemented");
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
        ins.opcode_id = intern_opcode(parts[0]);
        return ins;
      }
      if (ty.kind == Type::Kind::Pred) {
        // and.pred / or.pred / xor.pred
        std::optional<PredBinOp> pop;
        if (base_op == "and") pop = PredBinOp::And;
        else if (base_op == "or") pop = PredBinOp::Or;
        else if (base_op == "xor") pop = PredBinOp::Xor;
        if (!pop) return unsupported("'" + base_op + "' on predicates");
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
        auto it = fops.find(base_op);
        if (it == fops.end()) return unsupported("float op '" + base_op + "'");
        OpFloatBin op;
        op.round = frnd;
        op.op = it->second;
        op.nan_propagate = nan_propagate;
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
        if (base_op != "mul" || (ty.bits != 32 && ty.bits != 16))
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
        auto it = iops.find(base_op);
        if (it == iops.end()) return unsupported("integer op '" + base_op + "'");
        if (base_op == "mul" && !lo) return unsupported("plain mul on integers requires .lo/.wide/.hi");
        if ((carry_in || carry_out) && base_op != "add" && base_op != "sub")
          return unsupported("the carry bit is only defined for add/sub/mad");
        OpIntBin op;
        op.op = it->second;
        op.ty = ty;
        op.carry_in = carry_in;
        op.carry_out = carry_out;
        op.dst = expect_reg_operand("destination");
        expect_punct(",");
        op.a = parse_operand();
        expect_punct(",");
        op.b = parse_operand();
        ins.op = op;
      }
    } else if (op0 == "mad" || op0 == "fma" || op0 == "madc") {
      bool carry_in = (op0 == "madc");
      bool carry_out = false;
      Type ty{};
      bool have_ty = false, lo = false, wide = false, hi = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "lo") lo = true;
        else if (p == "wide") wide = true;
        else if (p == "hi") hi = true;
        else if (p == "cc") carry_out = true;
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
      if ((carry_in || carry_out) && !lo && !hi)
        return unsupported("the carry bit is only defined for mad.lo and mad.hi");
      if (ty.is_float()) {
        if (carry_in || carry_out) return unsupported("the carry bit is integer-only");
        if (ty.bits != 32 && ty.bits != 64) return unsupported("only f32/f64 fma implemented");
        ins.op = OpFma{ty, dst, a, b, c};
      } else if (wide) {
        if (ty.bits != 32) return unsupported("only mad.wide.{s32,u32} implemented");
        ins.op = OpMadWide{ty.is_signed(), dst, a, b, c};
      } else if (hi) {
        ins.op = OpMadHi{ty, dst, a, b, c, carry_in, carry_out};
      } else {
        if (!lo) return unsupported("integer mad requires .lo or .wide");
        ins.op = OpMadLo{ty, dst, a, b, c, carry_in, carry_out};
      }
    } else if (op0 == "set") {
      // set.<cmp>[.ftz].<dtype>.<stype> d, a, b
      std::vector<std::string> ps = parts;
      for (size_t i = 2; i < ps.size();)
        if (ps[i] == "ftz") ps.erase(ps.begin() + i); else ++i;
      if (ps.size() != 4) return unsupported("set form (expected set.<cmp>.<dtype>.<stype>)");
      auto it = cmp_table().find(ps[1]);
      if (it == cmp_table().end()) return unsupported("comparison '." + ps[1] + "'");
      OpSet op;
      op.cmp = it->second;
      auto dt = ps[2] == "f16x2"  ? std::optional<Type>{Type{Type::Kind::F, 16}}
              : ps[2] == "bf16x2" ? std::optional<Type>{Type{Type::Kind::BF, 16}}
                                  : parse_type_token(ps[2]);
      auto st = ps[3] == "f16x2"  ? std::optional<Type>{Type{Type::Kind::F, 16}}
              : ps[3] == "bf16x2" ? std::optional<Type>{Type{Type::Kind::BF, 16}}
                                  : parse_type_token(ps[3]);
      if (!dt || !st) return unsupported("set types '." + ps[2] + "." + ps[3] + "'");
      const bool dpack = ps[2] == "f16x2" || ps[2] == "bf16x2";
      const bool spack = ps[3] == "f16x2" || ps[3] == "bf16x2";
      if (dpack != spack)
        return unsupported("set with only one side packed ('." + ps[2] + "." + ps[3] + "')");
      op.dty = *dt;
      op.sty = *st;
      op.packed = dpack;
      op.dst = expect_reg_operand("set destination");
      expect_punct(",");
      op.a = parse_operand();
      expect_punct(",");
      op.b = parse_operand();
      ins.op = op;
    } else if (op0 == "setp") {
      // setp.<cmp>[.ftz].<type> — drop the flush-to-zero qualifier.
      if (parts.size() == 4 && parts[2] == "ftz") parts.erase(parts.begin() + 2);
      if (parts.size() != 3) return unsupported("setp form (only setp.<cmp>.<type> is implemented)");
      auto it = cmp_table().find(parts[1]);
      if (it == cmp_table().end()) return unsupported("comparison '." + parts[1] + "'");
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
      // The target is either a name or a register holding a function address.
      if (peek().kind == Token::Kind::Word && !peek().text.empty() && peek().text[0] == '%') {
        op.indirect = true;
        op.target_reg = expect_reg_operand("indirect call target");
      } else {
        op.callee = expect_word("call target");
      }
      expect_punct(",");
      expect_punct("(");
      while (!peek_punct(")")) {
        op.param_slots.push_back(expect_word("call argument slot"));
        if (peek_punct(",")) next();
      }
      next();
      // An indirect call names its prototype after the arguments.
      if (peek_punct(",")) {
        next();
        expect_word("call prototype name");
      }
      // Whether this names a builtin or a device function defined elsewhere in
      // the module is settled after parsing, by resolve_calls.
      ins.op = op;
    } else if (op0 == "activemask") {
      OpActiveMask op;
      op.dst = expect_reg_operand("activemask destination");
      ins.op = op;
    } else if (op0 == "trap") {
      ins.op = OpTrap{};
    } else if (op0 == "brkpt") {
      // Libraries put one on paths they consider unreachable (CuTe's invalid
      // control path does), so refusing the whole kernel for containing it
      // turned away programs that never execute it.
      OpTrap op;
      op.breakpoint = true;
      ins.op = op;
    } else if (op0 == "tex") {
      // tex.<geom>[.level|.grad].v4.<dtype>.<ctype> {d,d,d,d}, [obj, {c,...}]
      uint32_t dims = 0;
      TexGeom geom = TexGeom::D1;
      bool level = false;
      Type dtype{}, ctype{};
      bool have_d = false;
      std::vector<std::string> types;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "1d") { dims = 1; geom = TexGeom::D1; }
        else if (p == "2d") { dims = 2; geom = TexGeom::D2; }
        else if (p == "3d") { dims = 3; geom = TexGeom::D3; }
        else if (p == "a1d") { dims = 1; geom = TexGeom::A1D; }
        else if (p == "a2d") { dims = 2; geom = TexGeom::A2D; }
        else if (p == "cube") { dims = 3; geom = TexGeom::Cube; }
        else if (p == "acube") { dims = 3; geom = TexGeom::ACube; }
        else if (p == "v4") ;
        else if (p == "2dms" || p == "a2dms")
          return unsupported("multi-sample textures are not implemented");
        else if (p == "level") level = true;
        else if (p == "base") ;
        else if (p == "grad")
          return unsupported("tex.grad: the level of detail a GPU derives from gradients goes "
                             "through its approximate log2 and length units, which are not "
                             "documented, so it is refused rather than approximated (tex.level "
                             "and plain fetches of mipmapped textures are implemented)");
        else if (auto t2 = parse_type_token(p)) types.push_back(p);
        else return unsupported("tex modifier '." + p + "'");
      }
      if (!dims) return unsupported("tex geometry");
      if (types.size() != 2) return unsupported("tex needs a destination and a coordinate type");
      dtype = *parse_type_token(types[0]);
      ctype = *parse_type_token(types[1]);
      have_d = true;
      (void)have_d;
      OpTex op;
      op.geom = geom;
      op.dims = dims;
      op.dtype = dtype;
      op.ctype = ctype;
      if ((geom == TexGeom::Cube || geom == TexGeom::ACube) && !ctype.is_float())
        return unsupported("a cubemap fetch takes float coordinates");
      op.dsts = parse_reg_vector_any();
      if (op.dsts.size() != 4) return unsupported("tex destination arity (ptxas emits .v4)");
      expect_punct(",");
      expect_punct("[");
      op.obj = parse_operand();
      expect_punct(",");
      op.coords = parse_operand_vector_any();
      const bool indexed = geom == TexGeom::A1D || geom == TexGeom::A2D || geom == TexGeom::ACube;
      if (op.coords.size() < dims + (indexed ? 1 : 0))
        return unsupported("tex coordinate count does not match its geometry");
      expect_punct("]");
      if (level) {
        op.level = true;
        expect_punct(",");
        op.lod = parse_operand();
      }
      ins.op = std::move(op);
    } else if (op0 == "suld" || op0 == "sust") {
      // suld.b.<geom>.<type>.<clamp> {d,...}, [obj, {x,y}]
      // sust.b.<geom>.<type>.<clamp> [obj, {x,y}], {s,...}
      uint32_t dims = 0, bytes = 0;
      bool layered = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "b") ;              // byte-addressed, the only form ptxas emits
        else if (p == "p") return unsupported("suld/sust '.p' (formatted) is not implemented");
        else if (p == "1d") dims = 1;
        else if (p == "2d") dims = 2;
        else if (p == "3d") dims = 3;
        else if (p == "a1d") { dims = 1; layered = true; }
        else if (p == "a2d") { dims = 2; layered = true; }
        // Out-of-range policy. ".trap" is what a surface access compiles to by
        // default and is the only one implemented: ".clamp" and ".zero" change
        // the result rather than the diagnostics, so accepting them silently
        // would answer a different question.
        else if (p == "trap") ;
        else if (p == "clamp" || p == "zero")
          return unsupported("suld/sust '." + p + "' out-of-range policy is not implemented");
        else if (p == "v2" || p == "v4") ;
        else if (p == "b8") bytes = 1;
        else if (p == "b16") bytes = 2;
        else if (p == "b32") bytes = 4;
        else if (p == "b64") bytes = 8;
        else return unsupported(op0 + " modifier '." + p + "'");
      }
      if (!dims) return unsupported(op0 + " geometry");
      if (!bytes) return unsupported(op0 + " component width");
      if (op0 == "suld") {
        OpSuld op;
        op.dims = dims;
        op.layered = layered;
        op.bytes = bytes;
        op.dsts = parse_reg_vector_any();
        expect_punct(",");
        expect_punct("[");
        op.obj = parse_operand();
        expect_punct(",");
        op.coords = parse_operand_vector_any();
        expect_punct("]");
        if (op.coords.size() < dims + (layered ? 1 : 0)) return unsupported("suld coordinate count");
        ins.op = std::move(op);
      } else {
        OpSust op;
        op.dims = dims;
        op.layered = layered;
        op.bytes = bytes;
        expect_punct("[");
        op.obj = parse_operand();
        expect_punct(",");
        op.coords = parse_operand_vector_any();
        expect_punct("]");
        expect_punct(",");
        op.srcs = parse_operand_vector_any();
        if (op.coords.size() < dims + (layered ? 1 : 0)) return unsupported("sust coordinate count");
        ins.op = std::move(op);
      }
    } else if (op0 == "membar" || op0 == "fence") {
      // A host memory fence, not a no-op. This used to reason that blocks run
      // their instructions in order so there was nothing to fence against --
      // true of one thread, but blocks run on several, and on a weak-memory host
      // (ARM: Graviton, or a GH200's own Grace) the hardware reorders across
      // them. CUB's decoupled look-back depends on exactly this fence. It is
      // still not a barrier: this used to be OpBar, which made it wait for every
      // warp in the block.
      // fence.proxy.tensormap::generic takes the map's address and size; the
      // other proxy fences take nothing. Either way it orders, and orders only.
      while (!at_end() && !peek_punct(";")) next();
      ins.op = OpFence{};
    } else if (op0 == "nanosleep") {
      // A backoff hint. Consuming it as a no-op is correct; the operand is a
      // duration nothing here can meaningfully honour.
      (void)parse_operand();
      ins.op = OpNop{};
    } else if (op0 == "movmatrix") {
      bool trans = false, b16 = false, shape = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "sync" || p == "aligned") ;
        else if (p == "trans") trans = true;
        else if (p == "m8n8") shape = true;
        else if (p == "b16") b16 = true;
        else return unsupported("movmatrix modifier '." + p + "'");
      }
      // Without .trans it would be a plain move, and the only shape and element
      // width the ISA defines for it are m8n8.b16.
      if (!trans) return unsupported("movmatrix without .trans");
      if (!shape || !b16) return unsupported("only movmatrix.m8n8.b16 exists");
      OpMovMatrix op;
      op.dst = expect_reg_operand("movmatrix destination");
      expect_punct(",");
      op.src = parse_operand();
      ins.op = op;
    } else if (op0 == "cp" && parts.size() > 3 && parts[1] == "reduce" && parts[2] == "async" &&
               parts[3] == "bulk") {
      // cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.op.type [d], [s], size, [mbar]
      // cp.reduce.async.bulk.global.shared::cta.bulk_group.op{.noftz}.type [d], [s], size{, policy}
      // cp.reduce.async.bulk.tensor.Nd.global.shared::cta.op{.tile}.bulk_group [tmap, {c...}], [s]{, policy}
      OpBulkCopy op;
      op.reduce = true;
      std::vector<std::string> spaces;
      bool mbar_completion = false, group_completion = false, have_op = false, have_ty = false,
           noftz = false;
      for (size_t i = 4; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "tensor") op.tensor = true;
        else if (p.size() == 2 && p[1] == 'd' && p[0] >= '1' && p[0] <= '5') op.dims = p[0] - '0';
        else if (p == "shared" || p == "shared::cluster" || p == "global") spaces.push_back(p);
        else if (p == "mbarrier::complete_tx::bytes") mbar_completion = true;
        else if (p == "bulk_group") group_completion = true;
        else if (p == "noftz") noftz = true;
        else if (p == "tile" || p == "relaxed" || inert_mem_modifier(p)) ;
        else if (p == "im2col_no_offs") op.im2col = true;
        else if (p == "add") { op.red_op = AtomOp::Add; have_op = true; }
        else if (p == "min") { op.red_op = AtomOp::Min; have_op = true; }
        else if (p == "max") { op.red_op = AtomOp::Max; have_op = true; }
        else if (p == "inc") { op.red_op = AtomOp::Inc; have_op = true; }
        else if (p == "dec") { op.red_op = AtomOp::Dec; have_op = true; }
        else if (p == "and") { op.red_op = AtomOp::And; have_op = true; }
        else if (p == "or") { op.red_op = AtomOp::Or; have_op = true; }
        else if (p == "xor") { op.red_op = AtomOp::Xor; have_op = true; }
        else if (auto t = parse_type_token(p); t && !op.tensor) { op.red_ty = *t; have_ty = true; }
        else return unsupported("cp.reduce.async.bulk modifier '." + p + "' (overrides and multimem "
                                "are not implemented)");
      }
      if (!have_op) return unsupported("cp.reduce.async.bulk needs a reduction operation");
      if (op.tensor ? !op.dims : !have_ty) return unsupported("cp.reduce.async.bulk form");
      if (op.im2col && (!op.tensor || op.dims < 3))
        return unsupported("im2col_no_offs is a mode of 3D to 5D tensor reductions");
      if (spaces.size() != 2 || spaces[1] != "shared")
        return unsupported("cp.reduce.async.bulk reads from .shared::cta");
      const bool to_cluster = spaces[0] == "shared::cluster";
      if (to_cluster && (op.tensor || !mbar_completion))
        return unsupported("cp.reduce.async.bulk into .shared::cluster is a plain reduction "
                           "completing on an mbarrier");
      if (!to_cluster && !group_completion)
        return unsupported("cp.reduce.async.bulk into global memory completes in a bulk group");
      // The combinations the ISA's tables allow (9.7.10.28.4.2); the tensor
      // form's type comes from its map and is checked when it runs.
      if (!op.tensor) {
        const Type t = op.red_ty;
        const bool i32 = t.bits == 32 && !t.is_real(), i64 = t.bits == 64 && !t.is_real();
        const bool real = t.is_real();
        bool ok = false;
        switch (op.red_op) {
          case AtomOp::Add:
            ok = to_cluster ? (i32 && t.kind != Type::Kind::B) || (t.kind == Type::Kind::U && t.bits == 64)
                            : (t.kind != Type::Kind::B && (i32 || (t.kind == Type::Kind::U && i64))) || real;
            break;
          case AtomOp::Min:
          case AtomOp::Max:
            ok = to_cluster ? i32 && t.kind != Type::Kind::B
                            : ((i32 || i64) && t.kind != Type::Kind::B) || (real && t.bits == 16);
            break;
          case AtomOp::Inc:
          case AtomOp::Dec: ok = i32 && t.kind == Type::Kind::U; break;
          default: ok = to_cluster ? i32 : (i32 || i64); break;   // the bitwise ops
        }
        if (!ok)
          return unsupported("cp.reduce.async.bulk has no " + t.str() + " form of this operation");
        if (real && t.bits == 16 && op.red_op == AtomOp::Add && !noftz)
          return unsupported("cp.reduce.async.bulk.add on .f16 and .bf16 requires .noftz");
      }
      op.to_shared = to_cluster;
      op.shared_to_shared = to_cluster;
      if (op.tensor) {
        expect_punct("[");
        op.tmap = parse_operand();
        expect_punct(",");
        op.coords = parse_operand_vector_any();
        expect_punct("]");
        expect_punct(",");
        op.smem = parse_addr(fn);
        if (op.coords.size() != op.dims)
          return unsupported("cp.reduce.async.bulk.tensor coordinate count does not match ." +
                             std::to_string(op.dims) + "d");
      } else {
        // The destination is the side the reduction writes: global memory
        // (`gmem`, as a bulk store) or another block's shared memory (`smem`,
        // with the source in `gmem`, as a shared-to-shared bulk copy).
        if (to_cluster) op.smem = parse_addr(fn);
        else op.gmem = parse_addr(fn);
        expect_punct(",");
        if (to_cluster) op.gmem = parse_addr(fn);
        else op.smem = parse_addr(fn);
        expect_punct(",");
        op.size = parse_operand();
        if (to_cluster) {
          expect_punct(",");
          op.mbar = parse_addr(fn);
        }
      }
      if (peek_punct(",")) {   // a cache policy
        next();
        (void)parse_operand();
      }
      for (const Addr* a : {&op.smem, &op.gmem, &op.mbar})
        if (a->base_kind == Addr::Base::CallSlot)
          return unsupported("cp.reduce.async.bulk through a call slot");
      ins.op = op;
    } else if (op0 == "cp" && parts.size() > 2 && parts[1] == "async" && parts[2] == "bulk") {
      // Hopper's bulk copies (TMA). Forms, after cp.async.bulk:
      //   .commit_group / .wait_group[.read] N
      //   .prefetch[.tensor]...                      an L2 hint
      //   [.tensor.Nd].<dst>.<src>...  operands per direction below
      if (parts.size() > 3 && parts[3] == "commit_group") {
        ins.op = OpBulkGroup{};
      } else if (parts.size() > 3 && parts[3] == "wait_group") {
        for (size_t i = 4; i < parts.size(); ++i)
          if (parts[i] != "read") return unsupported("cp.async.bulk.wait_group modifier '." + parts[i] + "'");
        Operand n = parse_operand();
        auto* imm = std::get_if<ImmInt>(&n);
        if (!imm || imm->value < 0)
          return unsupported("cp.async.bulk.wait_group needs a non-negative immediate");
        OpBulkGroup g;
        g.wait = true;
        g.keep = static_cast<uint32_t>(imm->value);
        ins.op = g;
      } else if (parts.size() > 3 && parts[3] == "prefetch") {
        // Brings data into L2 ahead of a later copy; there is no cache here,
        // so it changes nothing -- like prefetch itself.
        while (!at_end() && !peek_punct(";")) next();
        ins.op = OpNop{};
      } else {
        OpBulkCopy op;
        std::vector<std::string> spaces;
        std::string im2col_mode;
        bool mbar_completion = false, group_completion = false;
        for (size_t i = 3; i < parts.size(); ++i) {
          const std::string& p = parts[i];
          if (p == "tensor") op.tensor = true;
          else if (p.size() == 2 && p[1] == 'd' && p[0] >= '1' && p[0] <= '5') op.dims = p[0] - '0';
          else if (p == "shared" || p == "shared::cluster" || p == "global") spaces.push_back(p);
          else if (p == "mbarrier::complete_tx::bytes") mbar_completion = true;
          else if (p == "bulk_group") group_completion = true;
          else if (p == "tile" || p == "weak" || p == "b128" || inert_mem_modifier(p)) ;
          else if (p == "multicast::cluster" || p == "multicast::cluster::16b") op.multicast = true;
          else if (p == "im2col" || p == "im2col_no_offs") { op.im2col = true; im2col_mode = p; }
          else return unsupported("cp.async.bulk modifier '." + p + "' (gather/scatter, im2col::w, "
                                  "masks, overrides and reports are not implemented)");
        }
        if (spaces.size() != 2) return unsupported("cp.async.bulk needs a destination and a source space");
        if (op.tensor && !op.dims) return unsupported("cp.async.bulk.tensor needs .1d to .5d");
        const bool g2s = spaces[1] == "global" && spaces[0] != "global";
        const bool s2g = spaces[0] == "global" && spaces[1] == "shared";
        const bool s2s = spaces[0] == "shared::cluster" && spaces[1] == "shared";
        if (!g2s && !s2g && !s2s)
          return unsupported("cp.async.bulk from ." + spaces[1] + " to ." + spaces[0]);
        if (s2s && (op.tensor || op.multicast))
          return unsupported("cp.async.bulk between shared memories is a plain, unicast copy");
        op.to_shared = g2s || s2s;
        op.shared_to_shared = s2s;
        if (op.im2col) {
          if (!op.tensor || op.dims < 3)
            return unsupported("im2col is a mode of 3D to 5D tensor copies");
          // A load takes the offsets (.im2col); a store has none to take.
          if (g2s != (im2col_mode == "im2col"))
            return unsupported("tensor loads take .im2col and stores .im2col_no_offs");
        }
        if (op.to_shared && !mbar_completion)
          return unsupported("a cp.async.bulk load completes on an mbarrier (.mbarrier::complete_tx::bytes)");
        if (s2g && !group_completion)
          return unsupported("a cp.async.bulk store completes in a bulk group (.bulk_group)");
        // [tensorMap, {c0, ...}]
        auto parse_tensor = [&]() {
          expect_punct("[");
          op.tmap = parse_operand();
          expect_punct(",");
          op.coords = parse_operand_vector_any();
          expect_punct("]");
        };
        if (op.to_shared) {
          op.smem = parse_addr(fn);
          expect_punct(",");
          if (op.tensor) parse_tensor();
          else {
            op.gmem = parse_addr(fn);
            expect_punct(",");
            op.size = parse_operand();
          }
          expect_punct(",");
          op.mbar = parse_addr(fn);
          if (op.im2col) {
            expect_punct(",");
            op.im2col_offsets = parse_operand_vector_any();
            if (op.im2col_offsets.size() != op.dims - 2)
              return unsupported("an im2col load takes one offset per spatial dimension (" +
                                 std::to_string(op.dims - 2) + ")");
          }
          if (op.multicast) {
            expect_punct(",");
            op.cta_mask = parse_operand();
          }
        } else {
          if (op.tensor) parse_tensor();
          else op.gmem = parse_addr(fn);
          expect_punct(",");
          op.smem = parse_addr(fn);
          if (!op.tensor) {
            expect_punct(",");
            op.size = parse_operand();
          }
        }
        // A trailing cache policy (from .L2::cache_hint, which the splitter
        // drops): a hint, read by nothing.
        if (peek_punct(",")) {
          next();
          (void)parse_operand();
        }
        if (op.tensor && op.coords.size() != op.dims)
          return unsupported("cp.async.bulk.tensor coordinate count does not match ." +
                             std::to_string(op.dims) + "d");
        for (const Addr* a : {&op.smem, &op.gmem, &op.mbar})
          if (a->base_kind == Addr::Base::CallSlot)
            return unsupported("cp.async.bulk through a call slot");
        ins.op = op;
      }
    } else if (op0 == "cp" && parts.size() > 1 && parts[1] == "async") {
      // The group operations first: they carry no addresses.
      if (parts.size() > 2 && parts[2] == "commit_group") {
        OpCpAsyncGroup g;
        g.kind = OpCpAsyncGroup::Kind::Commit;
        ins.op = g;
      } else if (parts.size() > 2 && parts[2] == "wait_all") {
        OpCpAsyncGroup g;
        g.kind = OpCpAsyncGroup::Kind::WaitAll;
        ins.op = g;
      } else if (parts.size() > 3 && parts[2] == "mbarrier" && parts[3] == "arrive") {
        // cp.async.mbarrier.arrive[.noinc].shared.b64 [bar]
        OpCpAsyncGroup g;
        g.kind = OpCpAsyncGroup::Kind::MbarrierArrive;
        for (size_t i = 4; i < parts.size(); ++i)
          if (parts[i] == "noinc") g.noinc = true;
        g.bar = parse_addr(fn);
        if (g.bar.base_kind == Addr::Base::CallSlot || g.bar.base_kind == Addr::Base::EntryParam)
          return unsupported("cp.async.mbarrier.arrive through a parameter/slot name");
        ins.op = g;
      } else if (parts.size() > 2 && parts[2] == "wait_group") {
        Operand n = parse_operand();
        auto* imm = std::get_if<ImmInt>(&n);
        if (!imm || imm->value < 0)
          return unsupported("cp.async.wait_group needs a non-negative immediate");
        OpCpAsyncGroup g;
        g.kind = OpCpAsyncGroup::Kind::WaitGroup;
        g.keep = static_cast<uint32_t>(imm->value);
        ins.op = g;
      } else {
        // cp.async.<ca|cg>.shared.global [dst], [src], cp-size{, src-size};
        bool cg = false, ca = false, shared_seen = false, global_seen = false;
        for (size_t i = 2; i < parts.size(); ++i) {
          const std::string& p = parts[i];
          if (p == "cg") cg = true;
          else if (p == "ca") ca = true;
          else if (p == "shared") shared_seen = true;
          else if (p == "global") global_seen = true;
          else return unsupported("cp.async modifier '." + p + "'");
        }
        if (!shared_seen || !global_seen)
          return unsupported("cp.async must name .shared and .global");
        if (!cg && !ca) return unsupported("cp.async needs .ca or .cg");
        OpCpAsync op;
        op.dst = parse_addr(fn);
        expect_punct(",");
        op.src = parse_addr(fn);
        expect_punct(",");
        {
          Operand n = parse_operand();
          auto* imm = std::get_if<ImmInt>(&n);
          if (!imm || (imm->value != 4 && imm->value != 8 && imm->value != 16))
            return unsupported("cp.async copy size must be 4, 8 or 16 bytes");
          op.bytes = static_cast<uint32_t>(imm->value);
        }
        // .cg exists only for 16-byte copies; .ca covers 4, 8 and 16. Saying so
        // beats copying the right bytes under a modifier that cannot mean this.
        if (cg && op.bytes != 16) return unsupported("cp.async.cg is 16 bytes only");
        if (peek_punct(",")) {
          next();
          op.have_src_size = true;
          op.src_size = parse_operand();
        }
        ins.op = op;
      }
    } else if (op0 == "barrier" && parts.size() > 2 && parts[1] == "cluster") {
      // barrier.cluster.arrive[.release|.relaxed][.aligned];
      // barrier.cluster.wait[.acquire][.aligned];
      OpClusterBarrier op;
      if (parts[2] == "arrive") op.wait = false;
      else if (parts[2] == "wait") op.wait = true;
      else return unsupported("barrier.cluster." + parts[2]);
      for (size_t i = 3; i < parts.size(); ++i)
        if (parts[i] != "aligned" && !inert_mem_modifier(parts[i]))
          return unsupported("barrier.cluster modifier '." + parts[i] + "'");
      ins.op = op;
    } else if ((op0 == "bar" || op0 == "barrier") &&
               std::find(parts.begin(), parts.end(), "red") != parts.end()) {
      // bar.red.<op>.<type> d, 0, [!]p
      std::optional<BarRedOp> rop;
      bool pred_ty = false, u32_ty = false;
      for (size_t i = 2; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p == "and") rop = BarRedOp::And;
        else if (p == "or") rop = BarRedOp::Or;
        else if (p == "popc") rop = BarRedOp::Popc;
        else if (p == "pred") pred_ty = true;
        else if (p == "u32") u32_ty = true;
        else if (p == "cta" || p == "red" || p == "aligned") ;
        else return unsupported("bar.red modifier '." + p + "'");
      }
      if (!rop) return unsupported("bar.red needs .and, .or or .popc");
      if (*rop == BarRedOp::Popc ? !u32_ty : !pred_ty)
        return unsupported("bar.red type does not match its operation");
      OpBarRed op;
      op.op = *rop;
      op.dst = expect_reg_operand("bar.red destination");
      expect_punct(",");
      {
        Operand which = parse_operand();
        if (auto* imm = std::get_if<ImmInt>(&which); !imm || imm->value != 0)
          return unsupported("only barrier 0 is implemented");
      }
      expect_punct(",");
      if (peek_punct("!")) { next(); op.negate_src = true; }
      op.src = expect_reg_operand("bar.red source predicate");
      ins.op = op;
    } else if (op0 == "bar" || op0 == "barrier") {
      bool sync_seen = false, arrive_seen = false, warp_scope = false;
      for (size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] == "sync") sync_seen = true;
        else if (parts[i] == "arrive") arrive_seen = true;
        else if (parts[i] == "cta" || parts[i] == "aligned") ;
        else if (parts[i] == "warp") warp_scope = true;
        else return unsupported("bar modifier '." + parts[i] + "'");
      }
      if (sync_seen == arrive_seen) return unsupported("bar needs one of .sync and .arrive");
      if (warp_scope && arrive_seen) return unsupported("bar.warp.arrive");
      if (warp_scope) {
        // __syncwarp. A warp executes its lanes in lockstep here and diverged
        // paths reconverge at the earliest common pc, so the lanes named by the
        // mask are already synchronised by the time this is reached. The
        // operand is the member mask, which nothing needs to consume.
        (void)parse_operand();
        OpBar op;
        op.warp = true;
        ins.op = op;
        expect_punct(";");
        return ins;
      }
      OpBar op;
      op.arrive = arrive_seen;
      op.id = parse_operand();
      if (auto* imm = std::get_if<ImmInt>(&op.id); imm && (imm->value < 0 || imm->value > 15))
        return unsupported("a CTA has barriers 0 to 15");
      if (peek_punct(",")) {
        next();
        op.count = parse_operand();
        op.have_count = true;
      }
      if (op.arrive && !op.have_count) return unsupported("bar.arrive needs a thread count");
      ins.op = op;
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
  // Registers declared inside a nested { } block that shadow an outer
  // declaration, innermost block last: source name -> the name it is interned
  // under while the block is open.
  std::vector<std::unordered_map<std::string, std::string>> reg_scopes_;
  int shadow_count_ = 0;
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
    place_dynamic_shared(fn);
  }
  return m;
}

}  // namespace vgpu::ptx
