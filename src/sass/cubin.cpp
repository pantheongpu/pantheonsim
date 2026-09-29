// Cubin (ELF) parsing; see vgpu/sass/cubin.hpp for the layout. A cubin comes
// from a file or a fatbin a program hands the driver, so every offset and
// size in it is checked before it is used.
#include "vgpu/sass/cubin.hpp"

#include <algorithm>
#include <cstring>

#include "vgpu/error.hpp"

namespace vgpu::sass {

namespace {

[[noreturn]] void bad(const std::string& why) { throw Error(Err::InvalidValue, "cubin: " + why); }

struct Reader {
  const uint8_t* data;
  size_t size;
  template <class T>
  T at(uint64_t off) const {
    if (off > size || sizeof(T) > size - off) bad("read past the end of the image");
    T v;
    std::memcpy(&v, data + off, sizeof v);
    return v;
  }
  std::string cstr(uint64_t off) const {
    if (off >= size) bad("string offset past the end of the image");
    const auto* end = static_cast<const uint8_t*>(std::memchr(data + off, 0, size - off));
    if (!end) bad("unterminated string");
    return std::string(reinterpret_cast<const char*>(data + off), end - (data + off));
  }
};

struct Shdr {
  uint32_t name, type;
  uint64_t flags, addr, offset, size;
  uint32_t link, info;
  uint64_t align, entsize;
};

constexpr uint32_t kShtSymtab = 2, kShtRela = 4, kShtNobits = 8, kShtRel = 9;

// .nv.info attributes this reads (the rest are skipped).
enum Attr : uint8_t {
  kMaxThreads = 0x05,
  kParamCbank = 0x0a,
  kExterns = 0x0f,
  kFrameSize = 0x11,
  kMinStackSize = 0x12,
  kKparamInfo = 0x17,
  kCbankParamSize = 0x19,
  kExitInstrOffsets = 0x1c,
  kRegcount = 0x2f,
  kNumBarriers = 0x4c,
};

struct Record {
  uint8_t format, attr;
  const uint8_t* payload;
  uint16_t size;   // payload bytes (2 for the fixed-size formats)
};

std::vector<Record> records(const std::vector<uint8_t>& blob) {
  std::vector<Record> out;
  size_t i = 0;
  while (i + 4 <= blob.size()) {
    const uint8_t fmt = blob[i], attr = blob[i + 1];
    if (fmt == 4) {
      uint16_t n;
      std::memcpy(&n, &blob[i + 2], 2);
      if (i + 4 + n > blob.size()) bad(".nv.info record runs past its section");
      out.push_back({fmt, attr, &blob[i + 4], n});
      i += 4 + n;
    } else {
      out.push_back({fmt, attr, &blob[i + 2], 2});
      i += 4;
    }
  }
  return out;
}

uint32_t u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

}  // namespace

bool is_cubin(const void* data, size_t size) {
  const auto* p = static_cast<const uint8_t*>(data);
  return size >= 64 && p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F' && p[4] == 2 /* 64-bit */ &&
         p[7] == 0x41 /* the CUDA OS ABI */;
}

Cubin parse_cubin(const uint8_t* data, size_t size) {
  if (!is_cubin(data, size)) bad("not a 64-bit CUDA ELF image");
  const Reader r{data, size};
  Cubin c;
  c.sm = static_cast<int>((r.at<uint32_t>(0x30) >> 8) & 0xff);
  const uint64_t shoff = r.at<uint64_t>(0x28);
  const uint16_t shentsize = r.at<uint16_t>(0x3a), shnum = r.at<uint16_t>(0x3c), shstrndx = r.at<uint16_t>(0x3e);
  if (shentsize < 64 || shnum == 0 || shstrndx >= shnum) bad("bad section header table");
  std::vector<Shdr> sh(shnum);
  for (uint16_t i = 0; i < shnum; ++i) {
    const uint64_t o = shoff + static_cast<uint64_t>(i) * shentsize;
    sh[i] = {r.at<uint32_t>(o), r.at<uint32_t>(o + 4), r.at<uint64_t>(o + 8), r.at<uint64_t>(o + 16),
             r.at<uint64_t>(o + 24), r.at<uint64_t>(o + 32), r.at<uint32_t>(o + 40), r.at<uint32_t>(o + 44),
             r.at<uint64_t>(o + 48), r.at<uint64_t>(o + 56)};
    if (sh[i].type != kShtNobits && (sh[i].offset > size || sh[i].size > size - sh[i].offset))
      bad("section " + std::to_string(i) + " runs past the end of the image");
  }
  const uint64_t shstr = sh[shstrndx].offset;
  std::vector<std::string> names(shnum);
  for (uint16_t i = 0; i < shnum; ++i) names[i] = r.cstr(shstr + sh[i].name);

  // Sections by index; relocation sections are attached to what they patch.
  std::vector<int> as_section(shnum, -1);
  for (uint16_t i = 0; i < shnum; ++i) {
    if (sh[i].type == kShtRel || sh[i].type == kShtRela || sh[i].type == kShtSymtab || i == 0) continue;
    CubinSection s;
    s.name = names[i];
    s.size = sh[i].size;
    if (sh[i].type != kShtNobits) s.bytes.assign(data + sh[i].offset, data + sh[i].offset + sh[i].size);
    as_section[i] = static_cast<int>(c.sections.size());
    c.section_index[s.name] = c.sections.size();
    c.sections.push_back(std::move(s));
  }

  // Symbols.
  std::vector<std::string> sym_names;
  for (uint16_t i = 0; i < shnum; ++i) {
    if (sh[i].type != kShtSymtab) continue;
    if (sh[i].link >= shnum) bad("symbol table names a bad string table");
    const uint64_t strtab = sh[sh[i].link].offset;
    const uint64_t n = sh[i].entsize ? sh[i].size / sh[i].entsize : 0;
    for (uint64_t k = 0; k < n; ++k) {
      const uint64_t o = sh[i].offset + k * sh[i].entsize;
      CubinSymbol s;
      s.name = r.cstr(strtab + r.at<uint32_t>(o));
      const uint8_t info = r.at<uint8_t>(o + 4);
      const uint16_t shndx = r.at<uint16_t>(o + 6);
      s.value = r.at<uint64_t>(o + 8);
      s.size = r.at<uint64_t>(o + 16);
      s.function = (info & 0xf) == 2;
      s.global = (info >> 4) == 1;
      if (shndx != 0 && shndx < shnum) s.section = names[shndx];
      sym_names.push_back(s.name);
      c.symbols.push_back(std::move(s));
    }
  }

  // Relocations: .rel.<section> (offset, info) and .rela.<section> (+ addend).
  for (uint16_t i = 0; i < shnum; ++i) {
    if (sh[i].type != kShtRel && sh[i].type != kShtRela) continue;
    if (sh[i].info >= shnum || as_section[sh[i].info] < 0) continue;   // e.g. .rel.debug_frame
    CubinSection& target = c.sections[static_cast<size_t>(as_section[sh[i].info])];
    const uint64_t es = sh[i].type == kShtRela ? 24 : 16;
    for (uint64_t o = sh[i].offset; o + es <= sh[i].offset + sh[i].size; o += es) {
      CubinReloc rel;
      rel.offset = r.at<uint64_t>(o);
      const uint64_t info = r.at<uint64_t>(o + 8);
      rel.type = static_cast<uint32_t>(info & 0xffffffff);
      const uint64_t sym = info >> 32;
      if (sym >= sym_names.size()) bad("relocation names a symbol past the table");
      rel.symbol = sym_names[sym];
      if (es == 24) rel.addend = r.at<int64_t>(o + 16);
      target.relocs.push_back(std::move(rel));
    }
  }

  // Kernels: every .text.<name> whose symbol is a global function with a
  // .nv.info.<name> section (device functions have code but no info).
  for (const CubinSection& s : c.sections) {
    if (s.name.rfind(".nv.info.", 0) != 0) continue;
    CubinKernel k;
    k.name = s.name.substr(9);
    k.text_section = ".text." + k.name;
    if (!c.section(k.text_section)) continue;
    for (const Record& rec : records(s.bytes)) {
      switch (rec.attr) {
        case kParamCbank:
          if (rec.size >= 8) {
            const uint32_t v = u32(rec.payload + 4);
            k.param_base = v & 0xffff;
            k.param_size = v >> 16;
          }
          break;
        case kCbankParamSize:
          k.param_size = static_cast<uint32_t>(rec.payload[0] | (rec.payload[1] << 8));
          break;
        case kKparamInfo:
          if (rec.size >= 12) {
            CubinParam p;
            p.ordinal = u32(rec.payload + 4) & 0xffff;
            p.offset = u32(rec.payload + 4) >> 16;
            p.size = (u32(rec.payload + 8) >> 18) & 0x3fff;
            k.params.push_back(p);
          }
          break;
        case kMaxThreads:
          if (rec.size >= 4) k.max_threads = u32(rec.payload);
          break;
        case kExitInstrOffsets:
          for (uint16_t o = 0; o + 4 <= rec.size; o += 4) k.exit_offsets.push_back(u32(rec.payload + o));
          break;
        case kNumBarriers: k.barriers = rec.payload[0]; break;
        default: break;
      }
    }
    std::sort(k.params.begin(), k.params.end(),
              [](const CubinParam& a, const CubinParam& b) { return a.ordinal < b.ordinal; });
    if (const CubinSection* shm = c.section(".nv.shared." + k.name)) k.shared_bytes = shm->size;
    c.kernels.push_back(std::move(k));
  }

  // Module-wide attributes name functions by symbol index: register counts,
  // frame and stack sizes, and the external functions called.
  if (const CubinSection* info = c.section(".nv.info")) {
    for (const Record& rec : records(info->bytes)) {
      if (rec.format != 4 || rec.size < 8) {
        if (rec.attr == kExterns && rec.format == 4)
          for (uint16_t o = 0; o + 4 <= rec.size; o += 4)
            if (u32(rec.payload + o) < sym_names.size()) c.externs.push_back(sym_names[u32(rec.payload + o)]);
        continue;
      }
      const uint32_t sym = u32(rec.payload), val = u32(rec.payload + 4);
      if (rec.attr == kExterns) {
        for (uint16_t o = 0; o + 4 <= rec.size; o += 4)
          if (u32(rec.payload + o) < sym_names.size()) c.externs.push_back(sym_names[u32(rec.payload + o)]);
        continue;
      }
      if (sym >= sym_names.size()) continue;
      for (CubinKernel& k : c.kernels) {
        if (k.name != sym_names[sym]) continue;
        if (rec.attr == kRegcount) k.regs = val;
        if (rec.attr == kFrameSize) k.frame_size = val;
        if (rec.attr == kMinStackSize) k.min_stack = val;
      }
    }
  }
  // The call graph: pairs of symbol indices (caller, callee); negative ones
  // are markers.
  if (const CubinSection* cg = c.section(".nv.callgraph")) {
    for (size_t o = 0; o + 8 <= cg->bytes.size(); o += 8) {
      const uint32_t a = u32(&cg->bytes[o]), b = u32(&cg->bytes[o + 4]);
      if (a < sym_names.size() && b < sym_names.size() && a != 0) c.calls.emplace_back(sym_names[a], sym_names[b]);
    }
  }
  // A function the module calls but does not define is an undefined symbol.
  for (const CubinSymbol& sym : c.symbols)
    if (sym.section.empty() && !sym.name.empty() &&
        std::find(c.externs.begin(), c.externs.end(), sym.name) == c.externs.end())
      c.externs.push_back(sym.name);
  return c;
}

}  // namespace vgpu::sass
