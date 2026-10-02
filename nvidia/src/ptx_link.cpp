#include "ptx_link.hpp"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace vgpu::cuda {

// A linked-in piece's file-scope names -- module variables and functions
// not declared .visible, .extern or .weak -- belong to that piece alone,
// the way an object file's local symbols do. Pasted together they did
// not: every translation unit names its first string literal $str, and
// CUDA's device-runtime library, always one of the pieces, has a $str of
// its own ("cudaSuccess"). Its definition won, so a program's string
// literal read as the library's (NanoVDB's device strcmp returned 17 for
// two equal strings: 't' - 'c'). Renaming each linked-in piece's locals
// keeps them apart. nvcc's __nv_static_ names are already unique, and the
// host registers them by name, so they are left alone.
void localize_ptx(std::string& text, size_t piece) {
  auto id_char = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
  };
  auto starts = [](const std::string& line, const char* word) {
    return line.compare(0, std::strlen(word), word) == 0;
  };
  std::set<std::string> locals;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    const std::string line = text.substr(pos, eol - pos);
    pos = eol + 1;
    // Module-scope declarations start in column 0; a function's own
    // .shared or .local is indented and stays where it is.
    const bool func = starts(line, ".func");
    if (!func && !starts(line, ".global") && !starts(line, ".const")) continue;
    std::string name;
    size_t i = func ? 5 : 0;
    if (func) {
      while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
      if (i < line.size() && line[i] == '(') {  // the return parameter list
        const size_t close = line.find(')', i);
        if (close == std::string::npos) continue;
        i = close + 1;
      }
      while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
      size_t j = i;
      while (j < line.size() && id_char(line[j])) ++j;
      name = line.substr(i, j - i);
    } else {
      // ".global .align 8 .u64 name[8] = {...};": the first token that is
      // neither a directive nor a number.
      std::istringstream toks(line);
      std::string tok;
      while (toks >> tok) {
        if (tok[0] == '.' || std::isdigit(static_cast<unsigned char>(tok[0]))) continue;
        size_t j = 0;
        while (j < tok.size() && id_char(tok[j])) ++j;
        name = tok.substr(0, j);
        break;
      }
    }
    if (!name.empty() && name.compare(0, 12, "__nv_static_") != 0) locals.insert(name);
  }
  if (locals.empty()) return;
  const std::string suffix = "$vgpu" + std::to_string(piece);
  std::string out;
  out.reserve(text.size() + locals.size() * 16);
  for (size_t k = 0; k < text.size();) {
    const char c = text[k];
    // An identifier starts with a letter, '_' or '$'; a register (after
    // '%') or a number is copied through untouched.
    if ((std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$') &&
        (k == 0 || (!id_char(text[k - 1]) && text[k - 1] != '%'))) {
      size_t j = k;
      while (j < text.size() && id_char(text[j])) ++j;
      const std::string word = text.substr(k, j - k);
      out += word;
      if (locals.count(word)) out += suffix;
      k = j;
    } else {
      out += c;
      ++k;
    }
  }
  text = std::move(out);
}


bool parse_arch(const std::string& arch, uint32_t* number, char* suffix) {
  size_t i = 0;
  if (arch.compare(0, 3, "sm_") == 0) i = 3;
  else if (arch.compare(0, 8, "compute_") == 0) i = 8;
  else return false;
  uint32_t n = 0;
  const size_t digits = i;
  while (i < arch.size() && std::isdigit(static_cast<unsigned char>(arch[i])))
    n = n * 10 + static_cast<uint32_t>(arch[i++] - '0');
  if (i == digits) return false;
  char s = 0;
  if (i < arch.size() && (arch[i] == 'a' || arch[i] == 'f')) s = arch[i++];
  if (i != arch.size()) return false;
  *number = n;
  *suffix = s;
  return true;
}

bool ptx_module_target(const std::string& ptx, uint32_t* number, char* suffix) {
  size_t at = 0;
  while ((at = ptx.find(".target", at)) != std::string::npos) {
    // Only the directive, at the start of a line -- not a comment mentioning it.
    if (at == 0 || ptx[at - 1] == '\n' || ptx[at - 1] == ' ' || ptx[at - 1] == '\t') {
      size_t b = at + 7;
      while (b < ptx.size() && (ptx[b] == ' ' || ptx[b] == '\t')) ++b;
      size_t e = b;
      while (e < ptx.size() && (std::isalnum(static_cast<unsigned char>(ptx[e])) || ptx[e] == '_')) ++e;
      return parse_arch(ptx.substr(b, e - b), number, suffix);
    }
    at += 7;
  }
  return false;
}

bool target_runs_on(uint32_t target, char target_suffix, uint32_t arch, char arch_suffix) {
  if (target_suffix == 'a') return arch == target && arch_suffix == 'a';
  if (target_suffix == 'f') return arch / 10 == target / 10 && arch >= target && arch_suffix != 0;
  return target <= arch;
}

namespace {

// One top-level construct of a PTX module: a header directive, a declaration
// or definition, or something carried through as written.
struct Item {
  enum class Kind { Header, Debug, Entry, Func, Var, Other } kind = Kind::Other;
  enum class Link { None, Visible, Extern, Weak, Common } link = Link::None;
  std::string text;
  std::string name;
  bool definition = false;
  bool shared = false;   // a .shared variable: .extern there is dynamic shared memory, not a symbol
  bool removed = false;
};

bool id_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$'; }

bool word_at(const std::string& t, size_t i, const char* w) {
  const size_t n = std::strlen(w);
  return t.compare(i, n, w) == 0 && (i + n == t.size() || !id_char(t[i + n]));
}

// The head of a construct -- everything before its body, initializer or
// semicolon -- as words and single punctuation characters.
std::vector<std::string> head_tokens(const std::string& head) {
  std::vector<std::string> out;
  for (size_t i = 0; i < head.size();) {
    const char c = head[i];
    if (std::isspace(static_cast<unsigned char>(c))) {
      ++i;
    } else if (id_char(c) || c == '.' || c == '%') {
      size_t j = i + 1;
      while (j < head.size() && (id_char(head[j]) || head[j] == '.' || head[j] == ':')) ++j;
      out.push_back(head.substr(i, j - i));
      i = j;
    } else {
      out.push_back(std::string(1, c));
      ++i;
    }
  }
  return out;
}

void classify(Item& it) {
  // The head ends where the body, the initializer or the statement does.
  size_t end = it.text.size();
  for (size_t i = 0; i < it.text.size(); ++i)
    if (it.text[i] == '{' || it.text[i] == '=' || it.text[i] == ';') {
      end = i;
      break;
    }
  const std::vector<std::string> toks = head_tokens(it.text.substr(0, end));
  const bool has_body = end < it.text.size() && it.text[end] == '{';
  size_t i = 0;
  for (; i < toks.size(); ++i) {
    if (toks[i] == ".visible") it.link = Item::Link::Visible;
    else if (toks[i] == ".extern") it.link = Item::Link::Extern;
    else if (toks[i] == ".weak") it.link = Item::Link::Weak;
    else if (toks[i] == ".common") it.link = Item::Link::Common;
    else break;
  }
  if (i == toks.size()) return;
  const std::string& what = toks[i];
  auto first_name = [&](size_t from) {
    int depth = 0;
    for (size_t k = from; k < toks.size(); ++k) {
      const std::string& t = toks[k];
      if (t == "(") ++depth;
      else if (t == ")") --depth;
      else if (depth == 0 && !t.empty() && t[0] != '.' && t[0] != '%' && id_char(t[0]) &&
               !std::isdigit(static_cast<unsigned char>(t[0])))
        return t;
    }
    return std::string{};
  };
  if (what == ".entry") {
    it.kind = Item::Kind::Entry;
    it.name = first_name(i + 1);
    it.definition = has_body;
  } else if (what == ".func") {
    it.kind = Item::Kind::Func;
    size_t k = i + 1;
    if (k < toks.size() && toks[k] == "(") {   // the return parameter list
      int depth = 0;
      for (; k < toks.size(); ++k) {
        if (toks[k] == "(") ++depth;
        else if (toks[k] == ")" && --depth == 0) break;
      }
      ++k;
    }
    it.name = first_name(k);
    it.definition = has_body;
  } else if (what == ".global" || what == ".const" || what == ".shared" || what == ".local" ||
             what == ".tex" || what == ".texref" || what == ".samplerref" || what == ".surfref") {
    it.kind = Item::Kind::Var;
    it.shared = what == ".shared";
    it.name = first_name(i + 1);
    it.definition = it.link != Item::Link::Extern;
  } else if (what == ".section") {
    it.kind = Item::Kind::Debug;
  }
}

// Splits a module into its top-level constructs. Comments between them are
// dropped; inside a construct everything is kept as written.
std::vector<Item> split_items(const std::string& t) {
  std::vector<Item> items;
  const size_t n = t.size();
  size_t i = 0;
  auto skip_comment = [&](size_t& j) {
    if (t.compare(j, 2, "//") == 0) {
      while (j < n && t[j] != '\n') ++j;
      return true;
    }
    if (t.compare(j, 2, "/*") == 0) {
      const size_t e = t.find("*/", j + 2);
      j = e == std::string::npos ? n : e + 2;
      return true;
    }
    return false;
  };
  for (;;) {
    while (i < n && (std::isspace(static_cast<unsigned char>(t[i])) || skip_comment(i)))
      if (i < n && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
    if (i >= n) break;
    Item it;
    const size_t start = i;
    if (word_at(t, i, ".version") || word_at(t, i, ".target") || word_at(t, i, ".address_size") ||
        word_at(t, i, ".file")) {
      // Line directives: no semicolon, the line is the directive.
      while (i < n && t[i] != '\n') ++i;
      it.kind = word_at(t, start, ".file") ? Item::Kind::Debug : Item::Kind::Header;
      it.text = t.substr(start, i - start);
      items.push_back(std::move(it));
      continue;
    }
    int brace = 0, paren = 0;
    bool initializer = false, body = false;
    while (i < n) {
      if (skip_comment(i)) continue;
      const char c = t[i];
      if (c == '"') {
        for (++i; i < n && t[i] != '"'; ++i)
          if (t[i] == '\\') ++i;
        ++i;
        continue;
      }
      ++i;
      if (c == '(') ++paren;
      else if (c == ')') --paren;
      else if (c == '=' && brace == 0) initializer = true;
      else if (c == '{') {
        if (brace++ == 0 && !initializer) body = true;
      } else if (c == '}') {
        if (--brace == 0 && body) break;
      } else if (c == ';' && brace == 0 && paren <= 0) {
        break;
      }
    }
    it.text = t.substr(start, i - start);
    classify(it);
    items.push_back(std::move(it));
  }
  return items;
}

// The calls ptxas answers itself rather than the linker: printf's, the
// device heap's and assert's. A program that declares them links alone.
bool provided_by_ptxas(const std::string& name) {
  return name == "vprintf" || name == "malloc" || name == "free" || name == "__assertfail";
}

std::pair<int, int> version_of(const std::string& header_line) {
  int major = 0, minor = 0;
  std::sscanf(header_line.c_str(), ".version %d.%d", &major, &minor);
  return {major, minor};
}

}  // namespace

PtxLinkResult link_ptx(const std::vector<PtxInput>& inputs, const std::string& target) {
  PtxLinkResult r;
  std::vector<std::vector<Item>> modules;
  std::pair<int, int> version{0, 0};
  std::string first_target;
  for (size_t m = 0; m < inputs.size(); ++m) {
    std::string text = inputs[m].text;
    while (!text.empty() && text.back() == '\0') text.pop_back();
    if (m > 0) localize_ptx(text, m);
    modules.push_back(split_items(text));
    for (const Item& it : modules.back()) {
      if (it.kind != Item::Kind::Header) continue;
      if (it.text.compare(0, 8, ".version") == 0) version = std::max(version, version_of(it.text));
      if (it.text.compare(0, 7, ".target") == 0 && first_target.empty()) first_target = it.text;
    }
  }

  // One definition per external symbol: strong beats weak, first beats later.
  struct Def {
    size_t module, item;
    bool strong;
  };
  std::map<std::string, Def> defs;
  for (size_t m = 0; m < modules.size(); ++m) {
    for (size_t k = 0; k < modules[m].size(); ++k) {
      Item& it = modules[m][k];
      if (!it.definition || it.shared || it.name.empty()) continue;
      const bool external = it.kind == Item::Kind::Entry || it.link == Item::Link::Visible ||
                            it.link == Item::Link::Weak || it.link == Item::Link::Common;
      if (!external) continue;
      const bool strong = it.link != Item::Link::Weak && it.link != Item::Link::Common;
      auto found = defs.find(it.name);
      if (found == defs.end()) {
        defs[it.name] = {m, k, strong};
      } else if (strong && found->second.strong) {
        r.errors += "error   : Multiple definition of '" + it.name + "' in '" + inputs[m].name +
                    "', first defined in '" + inputs[found->second.module].name + "'\n";
        it.removed = true;
      } else if (strong) {
        modules[found->second.module][found->second.item].removed = true;
        found->second = {m, k, true};
      } else {
        it.removed = true;
      }
    }
  }

  bool undefined = false;
  for (size_t m = 0; m < modules.size(); ++m) {
    std::set<std::string> reported;
    for (const Item& it : modules[m]) {
      if (it.link != Item::Link::Extern || it.shared || it.name.empty()) continue;
      if (it.kind != Item::Kind::Func && it.kind != Item::Kind::Var) continue;
      if (defs.count(it.name) || provided_by_ptxas(it.name) || !reported.insert(it.name).second)
        continue;
      r.errors += "error   : Undefined reference to '" + it.name + "' in '" + inputs[m].name + "'\n";
      undefined = true;
    }
  }
  if (undefined) return r;

  // One module, laid out the way ptxas needs it -- it refuses a definition
  // that follows an .extern declaration of the same symbol, and a use before
  // any declaration. So: a prototype of every function first, then every
  // module variable, then the functions and kernels, in input order; and
  // the .extern declarations only of what nothing here defines (vprintf and
  // the rest of ptxas's own), once each.
  std::ostringstream out;
  out << "//\n// Linked by VirtualGPU from " << inputs.size() << " PTX module"
      << (inputs.size() == 1 ? "" : "s") << "\n//\n\n";
  if (version.first) out << ".version " << version.first << "." << version.second << "\n";
  if (!target.empty()) out << ".target " << target << "\n";
  else if (!first_target.empty()) out << first_target << "\n";
  out << ".address_size 64\n";
  std::set<std::string> defined_funcs, declared;
  out << "\n";
  for (const auto& mod : modules)
    for (const Item& it : mod) {
      if (it.removed || it.kind != Item::Kind::Func || !it.definition) continue;
      std::string head = it.text.substr(0, it.text.find('{'));
      while (!head.empty() && std::isspace(static_cast<unsigned char>(head.back()))) head.pop_back();
      out << head << ";\n";
      defined_funcs.insert(it.name);
    }
  out << "\n";
  for (const auto& mod : modules)
    for (const Item& it : mod) {
      if (it.removed) continue;
      const bool extern_decl = it.link == Item::Link::Extern &&
                               (it.kind == Item::Kind::Var || it.kind == Item::Kind::Func);
      if (extern_decl) {
        if (!it.shared && defs.count(it.name)) continue;      // defined in this link
        if (!declared.insert(it.name).second) continue;       // already declared
        out << it.text << "\n";
      } else if (it.kind == Item::Kind::Var) {
        out << it.text << "\n";
      }
    }
  for (size_t m = 0; m < modules.size(); ++m) {
    out << "\n// " << inputs[m].name << "\n";
    for (const Item& it : modules[m]) {
      if (it.removed || it.kind == Item::Kind::Header || it.kind == Item::Kind::Debug ||
          it.kind == Item::Kind::Var || it.link == Item::Link::Extern)
        continue;
      // A prototype of a function defined here was written above.
      if (it.kind == Item::Kind::Func && !it.definition && defined_funcs.count(it.name)) continue;
      out << it.text << "\n";
    }
  }
  r.ptx = out.str();
  r.ok = true;
  return r;
}

}  // namespace vgpu::cuda
