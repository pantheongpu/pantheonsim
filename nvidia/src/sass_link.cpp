// The device linker for machine code: relocatable cubins -- what nvcc -rdc
// and -dc compile device code to, SASS with its relocations unapplied -- into
// one linked cubin, which VirtualGPU's SASS loader (src/sass/module.cpp) runs.
//
// Clean-room. The format is ELF; what is CUDA's own in it was established
// from NVIDIA's public tools and black-box links: cuobjdump -elf listings
// (section and symbol tables, .nv.info attributes, relocation tables) of
// relocatable cubins that CUDA 12.0's and 13.0's nvcc wrote for sm_75 to
// sm_120, and of what NVIDIA's libnvJitLink 13.0 linked them into on an RTX
// 3060, compared byte for byte at each relocated field. No NVIDIA binary was
// disassembled.
//
// What a link does:
//
// - Symbols. One definition per external symbol; a weak one (templates,
//   __cuda_sm70_* helpers that every module carries) gives way to a strong
//   one or to the first weak one; a second strong definition is reported
//   ("Multiple definition of ...") and dropped, and the link succeeds, as
//   NVIDIA's does. A reference nothing defines fails the link ("Undefined
//   reference to ..."), except the functions the driver supplies (vprintf,
//   malloc, free, __assertfail, the device runtime's) and weak references
//   (__UFT*), which are 0. File-scope names stay their module's: a local
//   function or object whose name another symbol has is renamed, because the
//   loader resolves relocations by name.
//
// - Sections. Each function keeps its own code section (.text.<name>) and
//   attribute section; kernels keep their parameter bank (.nv.constant0.<k>).
//   Module data is merged by name, each module's part at its own alignment:
//   .nv.global (zero-filled), .nv.global.init, and the constant banks
//   (.nv.constant3 holds __constant__ variables).
//
// - Shared memory, which only the link can lay out. In a relocatable cubin a
//   shared variable's symbol value is its alignment, not an offset; device
//   functions' variables are in .nv_debug.shared, a kernel's own in
//   .nv.shared.<kernel>, and extern __shared__ arrays are undefined. Measured
//   on NVIDIA's links: two device functions' variables share offsets unless
//   some kernel reaches both (f2's 64 bytes and f3's 12 both at 0, f1's after
//   f2's in the kernel that calls both); a kernel's own variables follow the
//   functions' it reaches; the dynamic array starts at the end of the static
//   ones rounded up to 16, and the kernel's .nv.shared size is that start.
//   Here: each function's variables are placed in order at the lowest offset
//   clear of every placed variable a common kernel reaches, which gives
//   those offsets.
//
// - Relocations. Those whose value is known once the module is laid out are
//   applied here, those that need a load address are kept for the loader,
//   as NVIDIA's output keeps them. By type number (the ELF r_type; names as
//   cuobjdump prints them), with the bits each writes, read off the linked
//   output (value = symbol + addend; a REL entry has none, and the field's old
//   contents are replaced):
//     applied   59 R_CUDA_ABS16_32          value, bits 32-47
//               55 R_CUDA_ABS32_32          value, bits 32-63
//               74 R_CUDA_ABS24_40          value, bits 40-63
//               64 R_CUDA_CONST_FIELD19_40  value / 4 at bits 40-53, bank at 54-58 (sm_75-sm_89)
//               66 R_CUDA_CONST_FIELD21_38  value at bits 38-53, bank at 54-58 (sm_90)
//              115 R_CUDA_CONST_FIELD22_37  value at bits 37-53, bank at 54-58 (sm_100, sm_120)
//              (the first three against a shared variable or a constant-bank offset; the
//               CONST_FIELDs against a constant, its bank from .nv.constant<N>)
//              114 R_CUDA_ABS56_16_34 against __UFT_OFFSET: an indirect call's
//                  offset into a function table this link does not build, 0
//     kept      2 R_CUDA_64, 4 R_CUDA_G64, 35 R_CUDA_FUNC_DESC_64 (data),
//               56/57 R_CUDA_ABS32_LO_32/HI_32,
//               58 R_CUDA_ABS47_34 and 75 R_CUDA_ABS55_16_34 (calls)
//     converted 112/113 R_CUDA_UNIFIED32_LO_32/HI_32 to 56/57, 102 R_CUDA_UNIFIED
//               to 2: a function's address, which NVIDIA's link turns into its
//               absolute address in the same way
//     dropped   68 and 69, which mark where the driver may insert a yield --
//               NVIDIA's link left the instruction as it was -- and everything in
//               debug sections, which are not kept
//   Any other type fails the link by name, never left unapplied.
//
// - Attributes. A kernel's register count (EIATTR_REGCOUNT) is the most any
//   function it may call uses -- NVIDIA's link raised a 28-register kernel's
//   to the 35 of a function it calls, and left the code section's own count
//   -- and its minimum stack its frame plus the deepest callee's (unknown,
//   0xffffffff, through recursion). The call graph and each kernel's list of
//   driver functions it calls are rewritten for the new symbol table.
//
// Not linked: debug information (-G's .debug_* and .nv_debug_* sections are
// dropped, so a debugger has no line table for linked code), the mercury
// sections sm_100's cubins carry for the driver to re-finalize, and
// texture/surface references. The output carries no program headers, which
// VirtualGPU's loader does not read.
#include "sass_link.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace vgpu::cuda {
namespace {

constexpr uint32_t kShtProgbits = 1, kShtSymtab = 2, kShtStrtab = 3, kShtRela = 4, kShtNobits = 8, kShtRel = 9;
constexpr uint32_t kShtCudaInfo = 0x70000000, kShtCudaCallgraph = 0x70000001;
constexpr uint8_t kSttObject = 1, kSttFunc = 2, kSttSection = 3, kSttCudaObject = 13;
constexpr uint8_t kStoManaged = 4;
// st_other's memory-space bits on a CUDA object: 0x20 global, 0x40 shared,
// 0x80 constant (an undefined extern __shared__ array has 0x40).
bool extern_shared(const struct Sym& s);
constexpr uint8_t kBindLocal = 0, kBindGlobal = 1, kBindWeak = 2;
constexpr uint8_t kStoEntry = 0x10;   // a kernel's symbol: st_other bit 4

struct LinkError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Sec {
  std::string name;
  uint32_t type = 0, link = 0, info = 0;
  uint64_t flags = 0, size = 0, align = 1, entsize = 0;
  std::vector<uint8_t> bytes;
};
struct Sym {
  std::string name;
  uint8_t info = 0, other = 0;
  uint16_t shndx = 0;
  uint64_t value = 0, size = 0;
  uint8_t bind() const { return info >> 4; }
  uint8_t type() const { return info & 0xf; }
};
bool extern_shared(const Sym& s) { return s.shndx == 0 && (s.other & 0xe0) == 0x40; }
struct Rel {
  uint32_t target = 0;   // the section it patches
  uint64_t offset = 0;
  uint32_t type = 0, sym = 0;
  int64_t addend = 0;
  bool rela = false;
};
struct In {
  std::string label;
  std::vector<uint8_t> header;
  uint32_t sm = 0;
  std::vector<Sec> secs;
  std::vector<Sym> syms;
  std::vector<Rel> rels;
};

template <class T>
T rd(const std::vector<uint8_t>& d, uint64_t off) {
  if (off > d.size() || sizeof(T) > d.size() - off) throw LinkError("read past the end");
  T v;
  std::memcpy(&v, d.data() + off, sizeof v);
  return v;
}
std::string rdstr(const std::vector<uint8_t>& d, uint64_t off) {
  if (off >= d.size()) throw LinkError("string offset past the end");
  const auto* p = static_cast<const uint8_t*>(std::memchr(d.data() + off, 0, d.size() - off));
  if (!p) throw LinkError("unterminated string");
  return std::string(reinterpret_cast<const char*>(d.data() + off), p - (d.data() + off));
}

In read_input(const SassLinkInput& input) {
  const std::vector<uint8_t>& d = input.cubin;
  In in;
  in.label = input.label;
  if (d.size() < 64 || d[0] != 0x7f || d[1] != 'E' || d[2] != 'L' || d[3] != 'F' || d[4] != 2)
    throw LinkError("not a 64-bit ELF cubin");
  in.header.assign(d.begin(), d.begin() + 64);
  const uint32_t eflags = rd<uint32_t>(d, 0x30);
  in.sm = d[7] == 0x33 ? eflags & 0xff : (eflags >> 8) & 0xff;
  const uint64_t shoff = rd<uint64_t>(d, 0x28);
  const uint16_t shentsize = rd<uint16_t>(d, 0x3a), shnum = rd<uint16_t>(d, 0x3c), shstrndx = rd<uint16_t>(d, 0x3e);
  if (shentsize < 64 || shnum == 0 || shstrndx >= shnum) throw LinkError("bad section header table");
  struct H {
    uint32_t name, type;
    uint64_t flags, off, size;
    uint32_t link, info;
    uint64_t align, entsize;
  };
  std::vector<H> h(shnum);
  for (uint16_t i = 0; i < shnum; ++i) {
    const uint64_t o = shoff + uint64_t{i} * shentsize;
    h[i] = {rd<uint32_t>(d, o), rd<uint32_t>(d, o + 4), rd<uint64_t>(d, o + 8), rd<uint64_t>(d, o + 24),
            rd<uint64_t>(d, o + 32), rd<uint32_t>(d, o + 40), rd<uint32_t>(d, o + 44), rd<uint64_t>(d, o + 48),
            rd<uint64_t>(d, o + 56)};
  }
  in.secs.resize(shnum);
  for (uint16_t i = 0; i < shnum; ++i) {
    Sec& s = in.secs[i];
    s.name = rdstr(d, h[shstrndx].off + h[i].name);
    s.type = h[i].type;
    s.flags = h[i].flags;
    s.size = h[i].size;
    s.link = h[i].link;
    s.info = h[i].info;
    s.align = h[i].align ? h[i].align : 1;
    s.entsize = h[i].entsize;
    // Zero-filled kinds hold no bytes in the file: NOBITS, and CUDA's own
    // types for .nv.global (0x70000007) and shared memory (0x7000000a),
    // whose offsets point at nothing.
    if (i != 0 && s.type != kShtNobits && s.type != 0x70000007 && s.type != 0x7000000a) {
      if (h[i].off > d.size() || h[i].size > d.size() - h[i].off)
        throw LinkError("section " + s.name + " runs past the end of the image");
      s.bytes.assign(d.begin() + h[i].off, d.begin() + h[i].off + h[i].size);
    }
  }
  for (uint16_t i = 0; i < shnum; ++i) {
    if (h[i].type != kShtSymtab) continue;
    if (h[i].link >= shnum || h[i].entsize < 24) throw LinkError("bad symbol table");
    const std::vector<uint8_t>& strtab = in.secs[h[i].link].bytes;
    for (uint64_t k = 0; k < h[i].size / h[i].entsize; ++k) {
      const uint64_t o = h[i].off + k * h[i].entsize;
      Sym s;
      s.name = rdstr(strtab, rd<uint32_t>(d, o));
      s.info = rd<uint8_t>(d, o + 4);
      s.other = rd<uint8_t>(d, o + 5);
      s.shndx = rd<uint16_t>(d, o + 6);
      s.value = rd<uint64_t>(d, o + 8);
      s.size = rd<uint64_t>(d, o + 16);
      in.syms.push_back(std::move(s));
    }
  }
  for (uint16_t i = 0; i < shnum; ++i) {
    if (h[i].type != kShtRel && h[i].type != kShtRela) continue;
    const uint64_t es = h[i].type == kShtRela ? 24 : 16;
    for (uint64_t o = h[i].off; o + es <= h[i].off + h[i].size; o += es) {
      Rel r;
      r.target = h[i].info;
      r.offset = rd<uint64_t>(d, o);
      const uint64_t inf = rd<uint64_t>(d, o + 8);
      r.type = static_cast<uint32_t>(inf);
      r.sym = static_cast<uint32_t>(inf >> 32);
      r.rela = es == 24;
      if (r.rela) r.addend = rd<int64_t>(d, o + 16);
      if (r.sym >= in.syms.size() || r.target >= shnum) throw LinkError("a relocation names a bad symbol or section");
      in.rels.push_back(r);
    }
  }
  return in;
}

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// ".nv.constant<N>" -> N, or -1. With a suffix (".nv.constant0.<kernel>") the
// bank belongs to that function.
int bank_of(const std::string& name, std::string* owner) {
  if (!starts(name, ".nv.constant")) return -1;
  size_t i = 12;
  int n = 0;
  bool digits = false;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    n = n * 10 + (name[i++] - '0');
    digits = true;
  }
  if (!digits) return -1;
  if (i == name.size()) {
    owner->clear();
    return n;
  }
  if (name[i] != '.') return -1;
  *owner = name.substr(i + 1);
  return n;
}

// The functions the driver supplies to device code, which a link leaves for
// the loader to resolve: printf, the device heap, assert, the graph calls and
// the device runtime's system calls.
bool driver_function(const std::string& n) {
  static const std::set<std::string> kNames = {"vprintf", "malloc", "free", "__assertfail",
                                               "cudaGraphSetConditional", "cudaGraphLaunch"};
  return kNames.count(n) || starts(n, "__cuda_syscall") || starts(n, "__cudaCDP") || starts(n, "cnp");
}

uint64_t align_up(uint64_t v, uint64_t a) { return a > 1 ? (v + a - 1) / a * a : v; }

void put_bits(std::vector<uint8_t>& b, uint64_t off, unsigned pos, unsigned len, uint64_t value) {
  if (off > b.size() || (pos + len + 7) / 8 > b.size() - off) throw LinkError("a relocation past the end of its section");
  for (unsigned k = 0; k < len; ++k) {
    const unsigned bit = pos + k;
    uint8_t& byte = b[off + bit / 8];
    const uint8_t m = static_cast<uint8_t>(1u << (bit % 8));
    byte = static_cast<uint8_t>(((value >> k) & 1) ? (byte | m) : (byte & ~m));
  }
}
uint64_t get_bits(const std::vector<uint8_t>& b, uint64_t off, unsigned pos, unsigned len) {
  if (off > b.size() || (pos + len + 7) / 8 > b.size() - off) throw LinkError("a relocation past the end of its section");
  uint64_t v = 0;
  for (unsigned k = 0; k < len; ++k) {
    const unsigned bit = pos + k;
    if ((b[off + bit / 8] >> (bit % 8)) & 1) v |= uint64_t{1} << k;
  }
  return v;
}

const char* reloc_name(uint32_t t) {
  switch (t) {
    case 2: return "R_CUDA_64";
    case 4: return "R_CUDA_G64";
    case 35: return "R_CUDA_FUNC_DESC_64";
    case 55: return "R_CUDA_ABS32_32";
    case 56: return "R_CUDA_ABS32_LO_32";
    case 57: return "R_CUDA_ABS32_HI_32";
    case 58: return "R_CUDA_ABS47_34";
    case 59: return "R_CUDA_ABS16_32";
    case 64: return "R_CUDA_CONST_FIELD19_40";
    case 66: return "R_CUDA_CONST_FIELD21_38";
    case 74: return "R_CUDA_ABS24_40";
    case 75: return "R_CUDA_ABS55_16_34";
    case 102: return "R_CUDA_UNIFIED";
    case 112: return "R_CUDA_UNIFIED32_LO_32";
    case 113: return "R_CUDA_UNIFIED32_HI_32";
    case 114: return "R_CUDA_ABS56_16_34";
    case 115: return "R_CUDA_CONST_FIELD22_37";
    default: return "an unknown type";
  }
}

// .nv.info records: {format, attribute, payload}. Format 4 (EIFMT_SVAL)
// carries a 16-bit length; the others a 2-byte value.
struct Rec {
  uint8_t fmt, attr;
  std::vector<uint8_t> payload;
};
std::vector<Rec> records(const std::vector<uint8_t>& b) {
  std::vector<Rec> out;
  size_t i = 0;
  while (i + 4 <= b.size()) {
    Rec r{b[i], b[i + 1], {}};
    if (r.fmt == 4) {
      const uint16_t n = static_cast<uint16_t>(b[i + 2] | (b[i + 3] << 8));
      if (i + 4 + n > b.size()) throw LinkError(".nv.info record runs past its section");
      r.payload.assign(b.begin() + static_cast<long>(i + 4), b.begin() + static_cast<long>(i + 4 + n));
      i += 4 + n;
    } else {
      r.payload.assign(b.begin() + static_cast<long>(i + 2), b.begin() + static_cast<long>(i + 4));
      i += 4;
    }
    out.push_back(std::move(r));
  }
  return out;
}
void emit(std::vector<uint8_t>& out, const Rec& r) {
  out.push_back(r.fmt);
  out.push_back(r.attr);
  if (r.fmt == 4) {
    out.push_back(static_cast<uint8_t>(r.payload.size()));
    out.push_back(static_cast<uint8_t>(r.payload.size() >> 8));
  }
  out.insert(out.end(), r.payload.begin(), r.payload.end());
}
uint32_t u32at(const std::vector<uint8_t>& p, size_t o) {
  uint32_t v;
  std::memcpy(&v, &p[o], 4);
  return v;
}
void put32(std::vector<uint8_t>& p, size_t o, uint32_t v) { std::memcpy(&p[o], &v, 4); }

constexpr uint8_t kAttrParamCbank = 0x0a, kAttrExterns = 0x0f, kAttrFrameSize = 0x11, kAttrMinStack = 0x12,
                  kAttrMaxStack = 0x23, kAttrRegcount = 0x2f;

// ---- the link ------------------------------------------------------------------

struct OutSym {
  std::string name;
  uint8_t info = 0, other = 0;
  int section = -1;   // output section index, -1 undefined
  uint64_t value = 0, size = 0;
};
struct OutRel {
  uint64_t offset;
  uint32_t type;
  uint32_t sym;   // output symbol index
  int64_t addend;
};
struct OutSec {
  Sec s;
  std::vector<OutRel> rels;
  int sym = -1;   // its section symbol
};

class Linker {
 public:
  Linker(std::vector<In> ins, uint32_t arch) : in_(std::move(ins)), arch_(arch) {}
  SassLinkResult run();

 private:
  // Where an input symbol went.
  struct Where {
    int out_sym = -1;        // output symbol, if it has one
    int section = -1;        // output section it lives in
    uint64_t value = 0;      // its value there
  };
  using Key = std::pair<size_t, uint32_t>;   // (input, symbol index)

  void resolve_globals();
  void place_sections();
  void lay_out_shared();
  void make_symbols();
  void relocate();
  void write_info();
  std::vector<uint8_t> write_elf();

  int add_section(Sec s) {
    out_.push_back({std::move(s), {}, -1});
    return static_cast<int>(out_.size() - 1);
  }
  // The definition an input's reference resolves to: itself when local or
  // defined there, else the global table's entry, else none.
  std::pair<size_t, uint32_t> definition(size_t i, uint32_t j) const {
    const Sym& s = in_[i].syms[j];
    if (s.bind() == kBindLocal || (s.shndx != 0 && s.type() != kSttSection && kept_def(i, j))) return {i, j};
    if (s.type() == kSttSection) return {i, j};
    const auto g = globals_.find(s.name);
    if (g != globals_.end()) return g->second;
    return {SIZE_MAX, 0};
  }
  bool kept_def(size_t i, uint32_t j) const {
    const auto g = globals_.find(in_[i].syms[j].name);
    return g != globals_.end() && g->second == Key{i, j};
  }
  bool is_shared_sym(size_t i, uint32_t j) const {
    const Sym& s = in_[i].syms[j];
    if (s.shndx == 0) return extern_shared(s);
    if (s.shndx >= in_[i].secs.size()) return false;
    const std::string& n = in_[i].secs[s.shndx].name;
    return n == ".nv_debug.shared" || starts(n, ".nv.shared.");
  }
  // The function whose code a section of input i is.
  std::string function_of(size_t i, uint32_t sec) const {
    const std::string& n = in_[i].secs[sec].name;
    return starts(n, ".text.") ? n.substr(6) : std::string();
  }

  std::vector<In> in_;
  uint32_t arch_;
  std::string errors_;
  std::map<std::string, Key> globals_;
  std::vector<OutSec> out_;
  std::map<std::string, int> merged_;                 // merged section name -> output section
  std::map<std::pair<size_t, uint32_t>, int> sec_map_;        // (input, section) -> output section
  std::map<std::pair<size_t, uint32_t>, uint64_t> sec_base_;  // where it starts there
  std::map<std::pair<size_t, uint32_t>, std::string> fn_name_;   // (input, text section) -> output function name
  std::map<Key, Where> where_;
  std::vector<OutSym> syms_;
  std::map<std::string, int> fn_sym_;   // output function name -> output symbol
  // Shared memory: each shared variable's offset, each extern-shared user's
  // dynamic base, each kernel's static size.
  std::map<Key, uint64_t> shared_off_;
  std::map<std::string, uint64_t> dyn_base_;      // by function
  std::map<std::string, uint64_t> kernel_shared_;  // by kernel
  std::map<std::string, std::set<std::string>> calls_;   // caller -> callees (output names)
  std::set<std::string> address_taken_;
  std::vector<std::string> indirect_calls_;   // a function per indirect call it makes
  std::set<std::string> kernels_;
  std::map<std::string, std::set<std::string>> externs_used_;   // kernel/function -> driver functions
  std::map<std::string, int> driver_sym_;   // the driver's functions, undefined, by name
  uint32_t first_global_ = 1;
  int info_sec_ = -1, callgraph_sec_ = -1;
  std::vector<std::pair<int, std::string>> fn_info_;       // .nv.info.<fn> sections and their functions
  std::vector<std::pair<int, std::string>> shared_secs_;   // .nv.shared.<kernel> sections and their kernels
  std::map<std::string, int> note_;   // the toolkit notes kept
  std::map<std::string, int> late_section_syms_;   // section symbols of sections write_elf makes
  std::vector<std::pair<int, std::string>> prototypes_;   // .nv.prototype: output symbol, prototype
};

void Linker::resolve_globals() {
  for (size_t i = 0; i < in_.size(); ++i) {
    for (uint32_t j = 0; j < in_[i].syms.size(); ++j) {
      const Sym& s = in_[i].syms[j];
      if (s.name.empty() || s.bind() == kBindLocal || s.shndx == 0 || s.type() == kSttSection) continue;
      // A shared variable defined in a module is its own (file scope).
      const auto it = globals_.find(s.name);
      if (it == globals_.end()) {
        globals_[s.name] = {i, j};
        continue;
      }
      const Sym& prev = in_[it->second.first].syms[it->second.second];
      if (s.bind() == kBindWeak) continue;            // a weak one gives way
      if (prev.bind() == kBindWeak) {                 // to a strong one
        it->second = {i, j};
        continue;
      }
      errors_ += "error   : Multiple definition of '" + s.name + "' in '" + in_[i].label + "', first defined in '" +
                 in_[it->second.first].label + "'\n";
    }
  }
  // Undefined references, per module, in name order as NVIDIA's lists them.
  bool undefined = false;
  for (size_t i = 0; i < in_.size(); ++i) {
    std::set<std::string> missing;
    for (const Sym& s : in_[i].syms) {
      if (s.name.empty() || s.shndx != 0 || s.bind() == kBindLocal || s.type() == kSttSection) continue;
      if (globals_.count(s.name) || s.bind() == kBindWeak || extern_shared(s)) continue;
      if (driver_function(s.name)) continue;
      missing.insert(s.name);
    }
    for (const std::string& n : missing) errors_ += "error   : Undefined reference to '" + n + "' in '" + in_[i].label + "'\n";
    undefined = undefined || !missing.empty();
  }
  if (undefined) throw LinkError("");
}

void Linker::place_sections() {
  for (size_t i = 0; i < in_.size(); ++i) {
    const In& in = in_[i];
    for (uint32_t k = 1; k < in.secs.size(); ++k) {
      const Sec& s = in.secs[k];
      const std::string& n = s.name;
      if (s.type == kShtSymtab || s.type == kShtStrtab || s.type == kShtRel || s.type == kShtRela) continue;
      // The toolkit notes, from the first module: NVIDIA's driver reads the
      // toolkit version from .note.nv.cuinfo, and refuses an image without.
      if (n == ".note.nv.tkinfo" || n == ".note.nv.cuinfo") {
        if (i == 0) note_[n] = add_section(s);
        continue;
      }
      if (n == ".nv.info" || n == ".nv.callgraph" || n == ".nv.prototype" || n == ".nv.rel.action" ||
          n == ".nv.compat" || n == ".nv_debug.shared" || starts(n, ".debug") || starts(n, ".nv_debug") ||
          starts(n, ".note") || starts(n, ".nv.merc") || starts(n, ".nv.capmerc") || starts(n, ".nv.shared.") ||
          starts(n, ".nv.info."))
        continue;   // rebuilt below, laid out by the link, or not kept
      std::string owner;
      const int bank = bank_of(n, &owner);
      if (starts(n, ".text.")) {
        // A function's code: kept if this module's definition of the
        // function is the one linked.
        const std::string fn = n.substr(6);
        uint32_t def = UINT32_MAX;
        for (uint32_t j = 0; j < in.syms.size(); ++j)
          if (in.syms[j].name == fn && in.syms[j].shndx == k && in.syms[j].type() == kSttFunc) def = j;
        if (def == UINT32_MAX) throw LinkError("'" + in.label + "': " + n + " defines no function of its name");
        const Sym& d = in.syms[def];
        if (d.bind() != kBindLocal && !kept_def(i, def)) continue;   // a dropped duplicate
        std::string name = fn;
        if (d.bind() == kBindLocal) {
          // A file-scope function: renamed when another module has one of
          // the same name.
          bool clash = globals_.count(fn) != 0 || fn_sym_.count(fn) != 0;
          for (const auto& [key, out_name] : fn_name_) clash = clash || out_name == fn;
          if (clash) name = fn + "$vgpu" + std::to_string(i);
        }
        fn_name_[{i, k}] = name;
        Sec o = s;
        o.name = ".text." + name;
        sec_map_[{i, k}] = add_section(std::move(o));
        sec_base_[{i, k}] = 0;
        fn_sym_[name] = -1;
        if (d.other & kStoEntry) kernels_.insert(name);
        continue;
      }
      if (bank > 0 && owner.empty()) {
        // A module constant bank: merged.
      } else if (n == ".nv.global" || n == ".nv.global.init") {
      } else if (bank >= 0) {
        // A function's own bank (.nv.constant0.<kernel>): with its function.
        continue;   // placed with the function below
      } else {
        throw LinkError("'" + in.label + "' has a section VirtualGPU's SASS linker does not know: " + n);
      }
      auto m = merged_.find(n);
      if (m == merged_.end()) {
        Sec o;
        o.name = n;
        o.type = n == ".nv.global" ? kShtNobits : kShtProgbits;
        o.flags = s.flags;
        o.align = 1;
        m = merged_.emplace(n, add_section(std::move(o))).first;
      }
      Sec& o = out_[static_cast<size_t>(m->second)].s;
      const uint64_t base = align_up(o.size, s.align);
      o.align = std::max(o.align, s.align);
      if (o.type != kShtNobits) {
        o.bytes.resize(base, 0);
        if (s.bytes.empty()) o.bytes.resize(base + s.size, 0);
        else o.bytes.insert(o.bytes.end(), s.bytes.begin(), s.bytes.end());
      }
      o.size = base + s.size;
      sec_map_[{i, k}] = m->second;
      sec_base_[{i, k}] = base;
    }
  }
  // Function-owned banks, now that the functions' names are known.
  for (size_t i = 0; i < in_.size(); ++i) {
    const In& in = in_[i];
    for (uint32_t k = 1; k < in.secs.size(); ++k) {
      std::string owner;
      const int bank = bank_of(in.secs[k].name, &owner);
      if (bank < 0 || owner.empty()) continue;
      // Its function: the .text section of the same name in this module.
      std::string name;
      for (const auto& [key, n] : fn_name_)
        if (key.first == i && in.secs[key.second].name == ".text." + owner) name = n;
      if (name.empty()) continue;   // a dropped duplicate's
      Sec o = in.secs[k];
      o.name = ".nv.constant" + std::to_string(bank) + "." + name;
      o.type = kShtProgbits;
      sec_map_[{i, k}] = add_section(std::move(o));
      sec_base_[{i, k}] = 0;
    }
  }
}

void Linker::lay_out_shared() {
  // Who calls whom, by output name, from every module's call graph -- edges
  // between -1 and -2 markers, pairs of symbol indices -- and which functions
  // have their address taken (a non-call relocation against one).
  for (size_t i = 0; i < in_.size(); ++i) {
    const In& in = in_[i];
    const auto name_of = [&](uint32_t j) -> std::string {
      if (j >= in.syms.size()) return {};
      const auto [di, dj] = definition(i, j);
      if (di == SIZE_MAX) return {};
      const Sym& d = in_[di].syms[dj];
      const auto it = fn_name_.find({di, d.shndx});
      return it != fn_name_.end() ? it->second : std::string();
    };
    // .nv.callgraph is four lists, each opened by a (0, -n) marker: -1 the
    // calls (caller, callee); -2 the functions whose address is taken (f, 1);
    // -3 each indirect call (its function, 1), once per call; -4 the targets
    // an indirect call may reach (caller, target), which the link fills in.
    for (const Sec& s : in.secs) {
      if (s.name != ".nv.callgraph") continue;
      uint32_t list = 0;
      for (size_t o = 0; o + 8 <= s.bytes.size(); o += 8) {
        const uint32_t a = u32at(s.bytes, o), b = u32at(s.bytes, o + 4);
        if (a == 0) {
          list = ~b + 1;   // -1 -> 1, -2 -> 2, ...
          continue;
        }
        const std::string from = name_of(a);
        if (from.empty()) continue;
        if (list == 1) {
          const std::string to = name_of(b);
          if (!to.empty()) calls_[from].insert(to);
        } else if (list == 2) {
          address_taken_.insert(from);
        } else if (list == 3) {
          indirect_calls_.push_back(from);
        }
      }
    }
    for (const Rel& r : in.rels) {
      if (r.type == 58 || r.type == 75) continue;
      if (!sec_map_.count({i, r.target})) continue;   // debug information names every function
      const Sym& s = in.syms[r.sym];
      if (s.type() != kSttFunc) continue;
      const std::string to = name_of(r.sym);
      // A function's own return addresses are relocations against itself.
      const auto from = fn_name_.find({i, r.target});
      if (!to.empty() && (from == fn_name_.end() || from->second != to)) address_taken_.insert(to);
    }
  }
  // Each kernel's reach: itself, what it calls, and every function whose
  // address is taken (it may be called through a pointer).
  std::map<std::string, std::set<std::string>> reach;
  for (const std::string& k : kernels_) {
    std::set<std::string>& r = reach[k];
    std::vector<std::string> todo{k};
    todo.insert(todo.end(), address_taken_.begin(), address_taken_.end());
    while (!todo.empty()) {
      const std::string f = todo.back();
      todo.pop_back();
      if (!r.insert(f).second) continue;
      if (const auto c = calls_.find(f); c != calls_.end()) todo.insert(todo.end(), c->second.begin(), c->second.end());
    }
  }
  // The variables of device functions (.nv_debug.shared), and which functions
  // use them.
  struct Var {
    Key key;
    uint64_t align, size;
    std::set<std::string> kernels;
  };
  std::vector<Var> vars;
  std::map<Key, std::set<std::string>> users;
  std::map<std::string, bool> uses_dynamic;
  for (size_t i = 0; i < in_.size(); ++i)
    for (const Rel& r : in_[i].rels) {
      const std::string fn = function_of(i, r.target);
      if (fn.empty() || !is_shared_sym(i, r.sym)) continue;
      const auto it = fn_name_.find({i, r.target});
      if (it == fn_name_.end()) continue;
      const Sym& s = in_[i].syms[r.sym];
      if (s.shndx == 0) uses_dynamic[it->second] = true;   // extern __shared__
      else users[{i, r.sym}].insert(it->second);
    }
  for (size_t i = 0; i < in_.size(); ++i)
    for (uint32_t j = 0; j < in_[i].syms.size(); ++j) {
      const Sym& s = in_[i].syms[j];
      if (s.shndx == 0 || s.shndx >= in_[i].secs.size() || in_[i].secs[s.shndx].name != ".nv_debug.shared" ||
          s.type() == kSttSection)
        continue;
      Var v{{i, j}, std::max<uint64_t>(s.value, 1), s.size, {}};
      for (const std::string& f : users[{i, j}])
        for (const auto& [k, r] : reach)
          if (r.count(f)) v.kernels.insert(k);
      vars.push_back(std::move(v));
    }
  std::vector<std::pair<uint64_t, const Var*>> placed;
  for (const Var& v : vars) {
    uint64_t off = 0;
    for (bool moved = true; moved;) {
      moved = false;
      for (const auto& [o, u] : placed) {
        bool common = false;
        for (const std::string& k : v.kernels) common = common || u->kernels.count(k);
        if (common && off < o + u->size && o < off + v.size) {
          off = align_up(o + u->size, v.align);
          moved = true;
        }
      }
    }
    placed.emplace_back(off, &v);
    shared_off_[v.key] = off;
  }
  // Each kernel's own variables (.nv.shared.<kernel> of its module), after
  // those of the functions it reaches.
  std::map<std::string, uint64_t> static_end;
  for (const auto& [k, r] : reach) {
    uint64_t end = 0;
    for (const auto& [o, v] : placed)
      if (v->kernels.count(k)) end = std::max(end, o + v->size);
    static_end[k] = end;
  }
  for (size_t i = 0; i < in_.size(); ++i)
    for (uint32_t j = 0; j < in_[i].syms.size(); ++j) {
      const Sym& s = in_[i].syms[j];
      if (s.shndx == 0 || s.shndx >= in_[i].secs.size() || s.type() == kSttSection) continue;
      const std::string& sec = in_[i].secs[s.shndx].name;
      if (!starts(sec, ".nv.shared.")) continue;
      std::string kernel;
      for (const auto& [key, n] : fn_name_)
        if (key.first == i && in_[i].secs[key.second].name == ".text." + sec.substr(11)) kernel = n;
      if (kernel.empty()) continue;
      uint64_t& end = static_end[kernel];
      const uint64_t off = align_up(end, std::max<uint64_t>(s.value, 1));
      shared_off_[{i, j}] = off;
      end = off + s.size;
    }
  // The dynamic array: one start for every kernel that reaches a function
  // using it, so the code they share agrees -- the largest static end among
  // them, rounded up to 16.
  std::map<std::string, std::string> group;   // kernel -> its group's first kernel
  std::function<std::string(const std::string&)> find = [&](const std::string& k) -> std::string {
    auto it = group.find(k);
    if (it == group.end() || it->second == k) return k;
    return it->second = find(it->second);
  };
  std::map<std::string, std::vector<std::string>> dyn_users_by_kernel;
  for (const auto& [f, yes] : uses_dynamic) {
    std::string first;
    for (const auto& [k, r] : reach) {
      if (!r.count(f)) continue;
      dyn_users_by_kernel[k].push_back(f);
      if (first.empty()) first = find(k);
      else group[find(k)] = first;
    }
  }
  std::map<std::string, uint64_t> group_base;
  for (const auto& [k, fs] : dyn_users_by_kernel) {
    uint64_t& b = group_base[find(k)];
    b = std::max(b, align_up(static_end[k], 16));
  }
  for (const auto& [k, r] : reach) {
    uint64_t size = static_end[k];
    if (dyn_users_by_kernel.count(k)) {
      size = group_base[find(k)];
      for (const std::string& f : dyn_users_by_kernel[k]) dyn_base_[f] = size;
    }
    // From sm_90 a kernel that uses shared memory has the driver's reserved
    // 1 KiB counted in its section too, though its variables still start at
    // 0: NVIDIA's link made a 0x960-byte array's section 0xd60, and an
    // extern-only kernel's 0x400.
    if (arch_ >= 90 && (size > 0 || dyn_users_by_kernel.count(k))) size += 0x400;
    kernel_shared_[k] = size;
  }
}

void Linker::make_symbols() {
  syms_.push_back({});   // index 0
  // Section symbols: one per output section, in the order the modules' own
  // come (NVIDIA's link keeps that order), interleaved with their locals.
  const auto section_symbol = [&](int s) {
    if (out_[static_cast<size_t>(s)].sym >= 0) return;
    out_[static_cast<size_t>(s)].sym = static_cast<int>(syms_.size());
    syms_.push_back({out_[static_cast<size_t>(s)].s.name, static_cast<uint8_t>((kBindLocal << 4) | kSttSection), 0, s, 0, 0});
  };
  // Every name an output symbol has, to keep file-scope names apart.
  std::set<std::string> taken;
  for (const auto& [n, key] : globals_) taken.insert(n);
  for (const auto& [key, n] : fn_name_) taken.insert(n);
  std::map<std::string, int> local_uses;
  for (int pass = 0; pass < 2; ++pass) {   // locals first, as ELF requires
    for (size_t i = 0; i < in_.size(); ++i)
      for (uint32_t j = 0; j < in_[i].syms.size(); ++j) {
        const Sym& s = in_[i].syms[j];
        if (s.type() == kSttSection) {
          if (pass == 0) {
            const auto m = sec_map_.find({i, s.shndx});
            if (m != sec_map_.end()) {
              section_symbol(m->second);
              where_[{i, j}] = {out_[static_cast<size_t>(m->second)].sym, m->second, sec_base_[{i, s.shndx}]};
            }
          }
          continue;
        }
        if (s.name.empty() || s.shndx == 0) continue;
        const bool local = s.bind() == kBindLocal;
        if ((pass == 0) != local) continue;
        if (!local && !kept_def(i, j)) continue;
        if (is_shared_sym(i, j)) continue;   // shared offsets are applied, the symbols not kept
        if (s.name == "_param") continue;    // a kernel's parameters' place in its bank: not kept either
        const auto m = sec_map_.find({i, s.shndx});
        if (m == sec_map_.end()) continue;   // in a section not kept
        OutSym o;
        o.name = s.name;
        if (s.type() == kSttFunc) {
          const auto fn = fn_name_.find({i, s.shndx});
          if (fn != fn_name_.end() && s.value == 0) o.name = fn->second;
        } else if (local && (taken.count(s.name) || local_uses[s.name]++)) {
          o.name = s.name + "$vgpu" + std::to_string(i) + "_" + std::to_string(j);
        }
        if (local) taken.insert(o.name);
        // A linked image's objects are plain ELF objects: NVIDIA's link turns
        // CUDA's object type (13) into STT_OBJECT and keeps of st_other only
        // the managed bit (4), and its driver refused an image that kept them.
        o.info = s.info;
        o.other = s.other;
        if (s.type() == kSttCudaObject) {
          o.info = static_cast<uint8_t>((s.bind() << 4) | kSttObject);
          o.other = static_cast<uint8_t>(s.other & kStoManaged);
        }
        o.section = m->second;
        o.value = s.value + sec_base_[{i, s.shndx}];
        o.size = s.size;
        const int idx = static_cast<int>(syms_.size());
        where_[{i, j}] = {idx, m->second, o.value};
        if (s.type() == kSttFunc) fn_sym_[o.name] = idx;
        syms_.push_back(std::move(o));
      }
    if (pass == 0) {
      for (size_t k = 0; k < out_.size(); ++k) section_symbol(static_cast<int>(k));
      // And for the sections the link makes, whose indices come later.
      bool protos = false;
      for (const In& in : in_)
        for (const Sec& x : in.secs) protos = protos || (x.name == ".nv.prototype" && !x.bytes.empty());
      for (const char* n : {".nv.callgraph", ".nv.prototype", ".nv.rel.action"}) {
        if ((n[4] == 'p' && !protos) || (n[4] == 'r' && arch_ >= 100)) continue;
        late_section_syms_[n] = static_cast<int>(syms_.size());
        syms_.push_back({n, static_cast<uint8_t>((kBindLocal << 4) | kSttSection), 0, -1, 0, 0});
      }
      first_global_ = static_cast<uint32_t>(syms_.size());
    }
  }
  // The driver's functions, undefined, in the order the modules name them.
  std::vector<std::string> drivers;
  for (const In& in : in_)
    for (const Sym& s : in.syms)
      if (s.shndx == 0 && driver_function(s.name) && !globals_.count(s.name) &&
          std::find(drivers.begin(), drivers.end(), s.name) == drivers.end())
        drivers.push_back(s.name);
  for (const std::string& n : drivers) {
    driver_sym_[n] = static_cast<int>(syms_.size());
    syms_.push_back({n, static_cast<uint8_t>((kBindGlobal << 4) | kSttFunc), 0, -1, 0, 0});
  }
}

void Linker::relocate() {
  for (size_t i = 0; i < in_.size(); ++i) {
    In& in = in_[i];
    for (const Rel& r : in.rels) {
      const auto m = sec_map_.find({i, r.target});
      if (m == sec_map_.end()) continue;   // a section not kept (debug, a dropped duplicate)
      OutSec& os = out_[static_cast<size_t>(m->second)];
      const uint64_t at = r.offset + sec_base_[{i, r.target}];
      const Sym& s = in.syms[r.sym];
      const std::string fn = function_of(i, r.target);
      const std::string out_fn = fn.empty() ? std::string() : fn_name_[{i, r.target}];
      if (r.type == 68 || r.type == 69) continue;   // where the driver may yield: left as it is
      const auto fail = [&](const std::string& why) {
        throw LinkError("'" + in.label + "': relocation " + std::to_string(r.type) + " (" + reloc_name(r.type) +
                        ") against '" + s.name + "' in " + in.secs[r.target].name + ": " + why);
      };
      // The symbol's link-time value, where it has one: a shared variable's
      // offset, or an offset into a constant bank (and the bank).
      bool link_time = false;
      uint64_t value = 0;
      int bank = -1;
      const auto [di, dj] = definition(i, r.sym);
      if (is_shared_sym(i, r.sym)) {
        link_time = true;
        if (s.shndx == 0) {
          const auto d = dyn_base_.find(out_fn);
          value = d != dyn_base_.end() ? d->second : 0;
        } else {
          value = shared_off_[{i, r.sym}];
        }
      } else if (di != SIZE_MAX) {
        const Sym& d = in_[di].syms[dj];
        std::string owner;
        if (d.shndx != 0 && d.shndx < in_[di].secs.size()) {
          bank = bank_of(in_[di].secs[d.shndx].name, &owner);
          if (bank > 0 && owner.empty()) {
            link_time = true;
            value = d.value + sec_base_[{di, d.shndx}];
          } else {
            bank = -1;
          }
        }
      }
      std::vector<uint8_t>& b = os.s.bytes;
      const auto apply = [&](unsigned pos, unsigned len, uint64_t v) { put_bits(b, at, pos, len, v); };
      // A REL entry's field is not an addend: NVIDIA's link replaced an STS
      // offset field that held 0x200 with the variable's offset alone.
      const uint64_t addend = r.rela ? static_cast<uint64_t>(r.addend) : 0;
      switch (r.type) {
        case 59:
        case 55:
        case 74: {
          if (!link_time) fail("its value is an address, which only the loader knows");
          const unsigned pos = r.type == 74 ? 40 : 32, len = r.type == 59 ? 16 : r.type == 55 ? 32 : 24;
          apply(pos, len, value + addend);
          continue;
        }
        case 64:
        case 66:
        case 115: {
          if (bank < 0) fail("it is not in a constant bank");
          const unsigned pos = r.type == 64 ? 40 : r.type == 66 ? 38 : 37, len = 54 - pos;
          const uint64_t off = value + addend;
          apply(pos, len, r.type == 64 ? off >> 2 : off);
          apply(54, 5, static_cast<uint64_t>(bank));
          continue;
        }
        case 114:
          if (s.name != "__UFT_OFFSET") fail("only __UFT_OFFSET is known");
          continue;   // the function table's offset: no table, 0
        case 2:
        case 4:
        case 35:
        case 56:
        case 57:
        case 58:
        case 75:
        case 102:
        case 112:
        case 113:
          break;
        default:
          fail("VirtualGPU's SASS linker does not know this relocation");
      }
      if (link_time) fail("it is a shared or constant-bank offset, which the loader cannot place");
      uint32_t type = r.type == 112 ? 56 : r.type == 113 ? 57 : r.type == 102 ? 2 : r.type;
      int64_t kept_addend = r.addend;
      if (!r.rela && (type == 2 || type == 4 || type == 35)) {   // data, REL: the addend is the 64-bit field
        kept_addend = static_cast<int64_t>(get_bits(b, at, 0, 64));
        put_bits(b, at, 0, 64, 0);
      }
      int target = -1;
      if (di == SIZE_MAX) {
        if (s.bind() == kBindWeak) continue;   // a weak reference to nothing: 0
        const auto d = driver_sym_.find(s.name);
        if (d == driver_sym_.end()) fail("nothing defines it");
        target = d->second;
        if (!out_fn.empty()) externs_used_[out_fn].insert(s.name);
      } else {
        const auto w = where_.find({di, dj});
        if (w == where_.end()) fail("its definition is not in the linked image");
        target = w->second.out_sym;
        // A section symbol: the section's part from this module starts later.
        if (in_[di].syms[dj].type() == kSttSection) kept_addend += static_cast<int64_t>(w->second.value);
      }
      os.rels.push_back({at, type, static_cast<uint32_t>(target), kept_addend});
    }
  }
}

void Linker::write_info() {
  // Per-function attributes from each module's .nv.info, by output symbol.
  std::map<std::string, uint32_t> regs, frame;
  std::vector<uint8_t> module_records;
  bool copied_flags = false;
  for (size_t i = 0; i < in_.size(); ++i) {
    for (const Sec& s : in_[i].secs) {
      if (s.name != ".nv.info") continue;
      for (const Rec& rec : records(s.bytes)) {
        if (rec.fmt != 4) {   // module-wide flags (EIFMT_HVAL): once
          if (!copied_flags) emit(module_records, rec);
          continue;
        }
        if (rec.payload.size() < 8) continue;
        const uint32_t sym = u32at(rec.payload, 0), val = u32at(rec.payload, 4);
        if (sym >= in_[i].syms.size()) continue;
        const Sym& fs = in_[i].syms[sym];
        const auto fn = fn_name_.find({i, fs.shndx});
        if (fn == fn_name_.end() || fs.type() != kSttFunc) continue;
        if (rec.attr == kAttrRegcount) regs[fn->second] = val;
        if (rec.attr == kAttrFrameSize) frame[fn->second] = val;
      }
      copied_flags = true;
    }
  }
  // A kernel's register count covers every function it may call, and its
  // stack the deepest chain of frames (0xffffffff through recursion).
  std::map<std::string, uint64_t> stack;
  std::set<std::string> on_path;
  bool recursive = false;
  std::function<uint64_t(const std::string&)> depth = [&](const std::string& f) -> uint64_t {
    if (const auto it = stack.find(f); it != stack.end()) return it->second;
    if (!on_path.insert(f).second) {
      recursive = true;
      return 0;
    }
    uint64_t deepest = 0;
    std::set<std::string> callees = calls_[f];
    for (const std::string& c : callees) deepest = std::max(deepest, depth(c));
    on_path.erase(f);
    return stack[f] = frame[f] + deepest;
  };
  std::vector<uint8_t> info = module_records;
  const auto sval = [&](uint8_t attr, uint32_t sym, uint32_t val) {
    Rec r{4, attr, std::vector<uint8_t>(8)};
    put32(r.payload, 0, sym);
    put32(r.payload, 4, val);
    emit(info, r);
  };
  for (const auto& [name, idx] : fn_sym_) {
    if (idx < 0) continue;
    uint32_t r = regs[name];
    uint64_t st = frame[name];
    if (kernels_.count(name)) {
      std::set<std::string> seen;
      std::vector<std::string> todo{name};
      todo.insert(todo.end(), address_taken_.begin(), address_taken_.end());
      while (!todo.empty()) {
        const std::string f = todo.back();
        todo.pop_back();
        if (!seen.insert(f).second) continue;
        r = std::max(r, regs[f]);
        if (const auto c = calls_.find(f); c != calls_.end()) todo.insert(todo.end(), c->second.begin(), c->second.end());
      }
      recursive = false;
      stack.clear();
      st = depth(name);
      for (const std::string& f : address_taken_) st = std::max<uint64_t>(st, frame[name] + depth(f));
    }
    sval(kAttrFrameSize, static_cast<uint32_t>(idx), frame[name]);
    sval(kAttrRegcount, static_cast<uint32_t>(idx), r);
    if (kernels_.count(name))
      sval(kAttrMinStack, static_cast<uint32_t>(idx), recursive ? 0xffffffffu : static_cast<uint32_t>(st));
    else {
      recursive = false;
      const uint64_t d = depth(name);
      sval(kAttrMaxStack, static_cast<uint32_t>(idx), recursive ? 0xffffffffu : static_cast<uint32_t>(d));
    }
  }
  Sec s;
  s.name = ".nv.info";
  s.type = kShtCudaInfo;
  s.align = 4;
  s.bytes = std::move(info);
  s.size = s.bytes.size();
  info_sec_ = add_section(std::move(s));

  // Each function's own attributes, with its symbol references rewritten: its
  // parameter bank's section symbol, and the driver functions it calls.
  for (size_t i = 0; i < in_.size(); ++i) {
    for (uint32_t k = 1; k < in_[i].secs.size(); ++k) {
      const Sec& src = in_[i].secs[k];
      if (!starts(src.name, ".nv.info.")) continue;
      std::string name;
      uint32_t text = 0;
      for (const auto& [key, n] : fn_name_)
        if (key.first == i && in_[i].secs[key.second].name == ".text." + src.name.substr(9)) {
          name = n;
          text = key.second;
        }
      if (name.empty()) continue;
      std::vector<uint8_t> out;
      // In the reverse of the module's order, as NVIDIA's link writes them.
      std::vector<Rec> recs = records(src.bytes);
      std::reverse(recs.begin(), recs.end());
      for (Rec rec : recs) {
        if (rec.fmt == 4 && rec.attr == kAttrParamCbank && rec.payload.size() >= 4) {
          const uint32_t sym = u32at(rec.payload, 0);
          const auto w = sym < in_[i].syms.size() ? where_.find({i, sym}) : where_.end();
          put32(rec.payload, 0, w != where_.end() ? static_cast<uint32_t>(w->second.out_sym) : 0);
        } else if (rec.fmt == 4 && rec.attr == kAttrExterns) {
          // The driver's functions it calls, in the module's order; what
          // the link resolved is no longer external.
          std::vector<uint8_t> kept;
          for (size_t o = 0; o + 4 <= rec.payload.size(); o += 4) {
            const uint32_t sym = u32at(rec.payload, o);
            if (sym >= in_[i].syms.size()) continue;
            const auto d = driver_sym_.find(in_[i].syms[sym].name);
            if (d == driver_sym_.end() || definition(i, sym).first != SIZE_MAX) continue;
            kept.resize(kept.size() + 4);
            put32(kept, kept.size() - 4, static_cast<uint32_t>(d->second));
          }
          if (kept.empty()) continue;
          rec.payload = std::move(kept);
        }
        emit(out, rec);
      }
      Sec o = src;
      o.name = ".nv.info." + name;
      o.bytes = std::move(out);
      o.size = o.bytes.size();
      const int idx = add_section(std::move(o));
      fn_info_.push_back({idx, name});
      (void)text;
    }
  }
  // Each kernel's shared memory.
  for (const auto& [k, size] : kernel_shared_) {
    if (size == 0) continue;
    Sec s;
    s.name = ".nv.shared." + k;
    s.type = kShtNobits;
    s.flags = 0x43;
    s.align = 16;
    s.size = size;
    shared_secs_.push_back({add_section(std::move(s)), k});
  }
  // The call graph.
  std::vector<uint8_t> cg;
  const auto pair = [&](uint32_t a, uint32_t b) {
    cg.resize(cg.size() + 8);
    put32(cg, cg.size() - 8, a);
    put32(cg, cg.size() - 4, b);
  };
  pair(0, 0xffffffffu);
  for (const auto& [from, tos] : calls_) {
    const auto f = fn_sym_.find(from);
    if (f == fn_sym_.end() || f->second < 0) continue;
    for (const std::string& to : tos) {
      const auto t = fn_sym_.find(to);
      if (t != fn_sym_.end() && t->second >= 0) pair(static_cast<uint32_t>(f->second), static_cast<uint32_t>(t->second));
    }
  }
  for (const auto& [from, es] : externs_used_) {
    const auto f = fn_sym_.find(from);
    if (f == fn_sym_.end() || f->second < 0) continue;
    for (const std::string& e : es) pair(static_cast<uint32_t>(f->second), static_cast<uint32_t>(driver_sym_[e]));
  }
  // The functions whose address is taken; each indirect call; and every
  // function an indirect call may reach, which NVIDIA's link lists for each
  // caller, and without which its driver did not find the caller.
  const auto sym_of = [&](const std::string& n) -> int {
    const auto f = fn_sym_.find(n);
    return f != fn_sym_.end() ? f->second : -1;
  };
  pair(0, 0xfffffffeu);
  for (const std::string& f : address_taken_)
    if (sym_of(f) >= 0) pair(static_cast<uint32_t>(sym_of(f)), 1);
  pair(0, 0xfffffffdu);
  std::set<std::string> indirect;
  for (const std::string& f : indirect_calls_)
    if (sym_of(f) >= 0) {
      pair(static_cast<uint32_t>(sym_of(f)), 1);
      indirect.insert(f);
    }
  pair(0, 0xfffffffcu);
  for (const std::string& f : indirect)
    for (const std::string& t : address_taken_)
      if (sym_of(t) >= 0) pair(static_cast<uint32_t>(sym_of(f)), static_cast<uint32_t>(sym_of(t)));
  Sec c;
  c.name = ".nv.callgraph";
  c.type = kShtCudaCallgraph;
  c.align = 4;
  c.entsize = 8;
  c.bytes = std::move(cg);
  c.size = c.bytes.size();
  callgraph_sec_ = add_section(std::move(c));

  // The prototypes of the functions each module calls, (symbol, string):
  // the calling convention of each, which the driver reads for the ones it
  // supplies (malloc, vprintf). One per function, the first module's.
  std::set<int> have;
  for (size_t i = 0; i < in_.size(); ++i)
    for (const Sec& src : in_[i].secs) {
      if (src.name != ".nv.prototype" || src.link >= in_[i].secs.size()) continue;
      const std::vector<uint8_t>& strtab = in_[i].secs[in_[i].secs[src.link].link].bytes;
      for (size_t o = 0; o + 8 <= src.bytes.size(); o += 8) {
        const uint32_t sym = u32at(src.bytes, o), str = u32at(src.bytes, o + 4);
        if (sym >= in_[i].syms.size()) continue;
        int out = -1;
        const auto [di, dj] = definition(i, sym);
        if (di != SIZE_MAX) {
          if (const auto w = where_.find({di, dj}); w != where_.end()) out = w->second.out_sym;
        } else if (const auto d = driver_sym_.find(in_[i].syms[sym].name); d != driver_sym_.end()) {
          out = d->second;
        }
        if (out < 0 || !have.insert(out).second) continue;
        try {
          prototypes_.emplace_back(out, rdstr(strtab, str));
        } catch (const LinkError&) {
        }
      }
    }
}

std::vector<uint8_t> Linker::write_elf() {
  // The sections in the order NVIDIA's link writes them: the tables, the
  // notes, the attributes, the call graph and prototypes, the relocations,
  // then what is loaded -- parameter banks, constant banks, code, data, and
  // the zero-filled kinds last.
  struct Item {
    int rank;
    Sec s;
    int out = -1;      // the output section it is, if one
    int rel_of = -1;   // the output section it relocates, if a relocation section
  };
  const auto rank_of = [](const Sec& x) {
    std::string owner;
    if (starts(x.name, ".note")) return 0;
    if (x.name == ".nv.info") return 1;
    if (starts(x.name, ".nv.info.")) return 2;
    if (x.type == kShtCudaCallgraph) return 3;
    if (bank_of(x.name, &owner) >= 0) return owner.empty() ? 8 : 7;
    if (starts(x.name, ".text.")) return 9;
    if (x.name == ".nv.global.init") return 10;
    if (starts(x.name, ".nv.shared.")) return 11;
    return 12;
  };
  std::vector<Item> items;
  for (size_t s = 0; s < out_.size(); ++s) items.push_back({rank_of(out_[s].s), out_[s].s, static_cast<int>(s), -1});
  for (OutSec& o : out_)   // by offset, as NVIDIA's link lists them
    std::stable_sort(o.rels.begin(), o.rels.end(), [](const OutRel& x, const OutRel& y) { return x.offset < y.offset; });
  // Relocations: those with no addend in a .rel section, the rest in a .rela.
  for (size_t s = 0; s < out_.size(); ++s) {
    for (const bool with_addend : {false, true}) {
      Sec r;
      r.name = (with_addend ? ".rela" : ".rel") + out_[s].s.name;
      r.type = with_addend ? kShtRela : kShtRel;
      r.flags = 0x40;
      r.align = 8;
      r.entsize = with_addend ? 24 : 16;
      r.link = 3;
      for (const OutRel& x : out_[s].rels) {
        if ((x.addend != 0) != with_addend) continue;
        const uint64_t inf = (uint64_t{x.sym} << 32) | x.type;
        const size_t at = r.bytes.size();
        r.bytes.resize(at + r.entsize);
        std::memcpy(&r.bytes[at], &x.offset, 8);
        std::memcpy(&r.bytes[at + 8], &inf, 8);
        if (with_addend) std::memcpy(&r.bytes[at + 16], &x.addend, 8);
      }
      if (!r.bytes.empty()) items.push_back({6, std::move(r), -1, static_cast<int>(s)});
    }
  }
  // Before sm_100 NVIDIA's link describes one relocation kind for the driver
  // in .nv.rel.action, the same 16 bytes in every image -- cuobjdump reads
  // them as "R_CUDA_CONST_FIELD22_37 : EIVALUE_SYM_KIND_ADDR,0,0,17,37,0,5,54"
  // (type 115, then the value's bits 0-16 to bits 37-53 and the bank's 5 to
  // 54). Kept as NVIDIA's link writes it.
  if (arch_ < 100) {
    Sec act;
    act.name = ".nv.rel.action";
    act.type = 0x7000000b;   // CUDA_RELOCINFO
    act.align = 8;
    act.entsize = 8;
    act.bytes = {0x73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x11, 0x25, 0, 0x05, 0x36};
    items.push_back({5, std::move(act), -1, -1});
  }
  int proto_item = -1;
  if (!prototypes_.empty()) {
    Sec proto;
    proto.name = ".nv.prototype";
    proto.type = 0x70000002;   // CUDA_PROTOTYPE
    proto.align = 4;
    proto.entsize = 8;
    proto.link = 3;
    items.push_back({4, std::move(proto), -1, -1});
    proto_item = static_cast<int>(items.size() - 1);
  }
  std::vector<size_t> order(items.size());
  for (size_t k = 0; k < order.size(); ++k) order[k] = k;
  std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return items[x].rank < items[y].rank; });
  std::vector<Sec> secs(4);   // null, .shstrtab, .strtab, .symtab
  secs[1].name = ".shstrtab";
  secs[1].type = kShtStrtab;
  secs[2].name = ".strtab";
  secs[2].type = kShtStrtab;
  secs[3].name = ".symtab";
  secs[3].type = kShtSymtab;
  secs[3].align = 8;
  secs[3].entsize = 24;
  secs[3].link = 2;
  std::vector<int> index(out_.size(), 0);
  size_t proto_at = 0;
  for (size_t k : order) {
    if (items[k].out >= 0) index[static_cast<size_t>(items[k].out)] = static_cast<int>(secs.size());
    if (static_cast<int>(k) == proto_item) proto_at = secs.size();
    secs.push_back(items[k].s);
  }
  for (size_t k : order)
    if (items[k].rel_of >= 0)
      for (Sec& x : secs)
        if (x.name == items[k].s.name && x.type == items[k].s.type)
          x.info = static_cast<uint32_t>(index[static_cast<size_t>(items[k].rel_of)]);
  // Links and infos that name sections or symbols. A function's attribute,
  // parameter-bank and shared-memory sections name its code section by
  // index (sh_info), as NVIDIA's link writes them.
  std::map<std::string, uint32_t> text_index;
  for (size_t s = 0; s < out_.size(); ++s)
    if (starts(out_[s].s.name, ".text.")) text_index[out_[s].s.name.substr(6)] = static_cast<uint32_t>(index[s]);
  for (size_t s = 0; s < out_.size(); ++s) {
    Sec& o = secs[static_cast<size_t>(index[s])];
    if (starts(o.name, ".text.")) {
      const auto f = fn_sym_.find(o.name.substr(6));
      o.link = 3;
      o.info = (o.info & 0xff000000u) | static_cast<uint32_t>(f != fn_sym_.end() && f->second >= 0 ? f->second : 0);
    } else if (o.type == kShtCudaInfo || o.type == kShtCudaCallgraph) {
      o.link = 3;
    }
    std::string owner;
    if (bank_of(o.name, &owner) >= 0 && !owner.empty()) o.info = text_index[owner];
  }
  if (note_.count(".note.nv.cuinfo") && note_.count(".note.nv.tkinfo"))
    secs[static_cast<size_t>(index[static_cast<size_t>(note_[".note.nv.cuinfo"])])].link =
        static_cast<uint32_t>(index[static_cast<size_t>(note_[".note.nv.tkinfo"])]);
  for (const auto& [sec, name] : fn_info_) {
    Sec& o = secs[static_cast<size_t>(index[static_cast<size_t>(sec)])];
    o.info = text_index[name];
    o.flags |= 0x40;
  }
  for (const auto& [sec, name] : shared_secs_)
    secs[static_cast<size_t>(index[static_cast<size_t>(sec)])].info = text_index[name];
  // String and symbol tables.
  const auto add_str = [](Sec& t, const std::string& s) {
    if (t.bytes.empty()) t.bytes.push_back(0);
    const uint32_t at = static_cast<uint32_t>(t.bytes.size());
    t.bytes.insert(t.bytes.end(), s.begin(), s.end());
    t.bytes.push_back(0);
    return at;
  };
  std::vector<uint32_t> name_off(secs.size(), 0);
  for (size_t s = 1; s < secs.size(); ++s) name_off[s] = add_str(secs[1], secs[s].name);
  Sec& strtab = secs[2];
  Sec& syms = secs[3];
  if (strtab.bytes.empty()) strtab.bytes.push_back(0);
  // The prototypes' strings first, as NVIDIA's link places them.
  if (proto_at) {
    Sec& proto = secs[proto_at];
    for (const auto& [sym, text] : prototypes_) {
      const uint32_t at = add_str(strtab, text);
      proto.bytes.resize(proto.bytes.size() + 8);
      put32(proto.bytes, proto.bytes.size() - 8, static_cast<uint32_t>(sym));
      put32(proto.bytes, proto.bytes.size() - 4, at);
    }
  }
  for (const OutSym& o : syms_) {
    const uint32_t n = o.name.empty() ? 0 : add_str(strtab, o.name);
    std::vector<uint8_t> e(24, 0);
    std::memcpy(&e[0], &n, 4);
    e[4] = o.info;
    e[5] = o.other;
    uint16_t shndx = o.section < 0 ? 0 : static_cast<uint16_t>(index[static_cast<size_t>(o.section)]);
    if (o.section < 0 && (o.info & 0xf) == kSttSection)
      for (size_t x = 4; x < secs.size(); ++x)
        if (secs[x].name == o.name) shndx = static_cast<uint16_t>(x);
    std::memcpy(&e[6], &shndx, 2);
    std::memcpy(&e[8], &o.value, 8);
    std::memcpy(&e[16], &o.size, 8);
    syms.bytes.insert(syms.bytes.end(), e.begin(), e.end());
  }
  syms.info = first_global_;
  for (Sec& x : secs) {
    if (x.type != kShtNobits) x.size = x.bytes.size();
  }
  // The image: header, sections, section headers.
  std::vector<uint8_t> img(64, 0);
  std::memcpy(img.data(), in_[0].header.data(), 16);
  const uint16_t et_exec = 2;
  std::memcpy(&img[0x10], &et_exec, 2);
  std::memcpy(&img[0x12], &in_[0].header[0x12], 6);   // machine, version
  uint32_t flags;
  std::memcpy(&flags, &in_[0].header[0x30], 4);
  if (img[7] == 0x33) flags = (flags & ~0xff00ffu) | arch_ | (arch_ << 16);
  else flags = (flags & ~0xff00u) | (arch_ << 8);
  std::memcpy(&img[0x30], &flags, 4);
  std::vector<uint64_t> offs(secs.size(), 0);
  for (size_t s = 1; s < secs.size(); ++s) {
    if (secs[s].type == kShtNobits) continue;
    const uint64_t a = std::max<uint64_t>(secs[s].align, 1);
    img.resize(align_up(img.size(), a), 0);
    offs[s] = img.size();
    img.insert(img.end(), secs[s].bytes.begin(), secs[s].bytes.end());
  }
  img.resize(align_up(img.size(), 8), 0);
  const uint64_t shoff = img.size();
  for (size_t s = 0; s < secs.size(); ++s) {
    std::vector<uint8_t> h(64, 0);
    const Sec& x = secs[s];
    if (s != 0) {
      std::memcpy(&h[0], &name_off[s], 4);
      std::memcpy(&h[4], &x.type, 4);
      std::memcpy(&h[8], &x.flags, 8);
      std::memcpy(&h[24], &offs[s], 8);
      std::memcpy(&h[32], &x.size, 8);
      std::memcpy(&h[40], &x.link, 4);
      std::memcpy(&h[44], &x.info, 4);
      std::memcpy(&h[48], &x.align, 8);
      std::memcpy(&h[56], &x.entsize, 8);
    }
    img.insert(img.end(), h.begin(), h.end());
  }
  const uint16_t ehsize = 64, shentsize = 64, shnum = static_cast<uint16_t>(secs.size()), shstrndx = 1;
  std::memcpy(&img[0x28], &shoff, 8);
  std::memcpy(&img[0x34], &ehsize, 2);
  std::memcpy(&img[0x3a], &shentsize, 2);
  std::memcpy(&img[0x3c], &shnum, 2);
  std::memcpy(&img[0x3e], &shstrndx, 2);
  return img;
}

SassLinkResult Linker::run() {
  SassLinkResult res;
  try {
    resolve_globals();
    place_sections();
    lay_out_shared();
    make_symbols();
    relocate();
    write_info();
    res.cubin = write_elf();
    res.ok = true;
  } catch (const LinkError& e) {
    if (*e.what()) errors_ += std::string("error   : ") + e.what() + "\n";
  }
  res.errors = errors_;
  return res;
}

}  // namespace

bool cubin_relocatable(const void* data, size_t size) {
  const auto* p = static_cast<const uint8_t*>(data);
  return size >= 64 && p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F' && p[4] == 2 && p[0x10] == 1 &&
         p[0x11] == 0;
}

SassLinkResult link_sass(const std::vector<SassLinkInput>& inputs, uint32_t arch) {
  std::vector<In> ins;
  SassLinkResult bad;
  for (const SassLinkInput& in : inputs) {
    try {
      ins.push_back(read_input(in));
    } catch (const LinkError& e) {
      bad.errors += "error   : '" + in.label + "' is not a cubin VirtualGPU can link: " + e.what() + "\n";
      return bad;
    }
  }
  if (ins.empty()) {
    bad.errors += "error   : nothing to link\n";
    return bad;
  }
  return Linker(std::move(ins), arch).run();
}

}  // namespace vgpu::cuda
