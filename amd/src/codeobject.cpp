#include "vgpu/amd_codeobject.hpp"

#include <cstring>
#include <algorithm>
#include <map>

#include "vgpu/error.hpp"

namespace vgpu::amd {
namespace {

// ---- The bytes ------------------------------------------------------------------

// Reads little-endian fields, refusing anything that runs past the end rather
// than reading whatever follows: a code object is a file, and files are
// truncated.
struct Reader {
  const std::string& b;
  const std::string& origin;

  void need(uint64_t at, uint64_t n) const {
    if (at + n > b.size() || at + n < at)
      throw Error::make(Err::ProfileParse, origin, ": truncated at byte ", at);
  }
  uint8_t u8(uint64_t at) const {
    need(at, 1);
    return static_cast<uint8_t>(b[at]);
  }
  uint16_t u16(uint64_t at) const { return static_cast<uint16_t>(u8(at) | u8(at + 1) << 8); }
  uint32_t u32(uint64_t at) const { return u16(at) | static_cast<uint32_t>(u16(at + 2)) << 16; }
  uint64_t u64(uint64_t at) const { return u32(at) | static_cast<uint64_t>(u32(at + 4)) << 32; }
  std::string str(uint64_t at, uint64_t n) const {
    need(at, n);
    return b.substr(at, n);
  }
  // A NUL-terminated string in a table.
  std::string cstr(uint64_t at) const {
    std::string s;
    for (uint64_t i = at; u8(i); ++i) s += static_cast<char>(u8(i));
    return s;
  }
};

// ---- MessagePack ----------------------------------------------------------------
//
// MessagePack numbers are big-endian, unlike every field of the ELF around
// them, so they are read here rather than through Reader.
//
// The metadata note is a MessagePack document (msgpack.org's format): maps of
// strings to values. Only what the metadata uses is read -- maps, arrays,
// strings, integers and booleans -- and anything else is an error rather than
// a value quietly misread.
struct Msgpack {
  enum class Kind { Int, Str, Array, Map, Bool, Nil };
  Kind kind = Kind::Nil;
  int64_t i = 0;
  bool b = false;
  std::string s;
  std::vector<Msgpack> list;
  std::vector<std::pair<Msgpack, Msgpack>> map;

  const Msgpack* at(const std::string& key) const {
    for (const auto& [k, v] : map)
      if (k.kind == Kind::Str && k.s == key) return &v;
    return nullptr;
  }
  int64_t number(const std::string& key, int64_t fallback = 0) const {
    const Msgpack* v = at(key);
    return v && v->kind == Kind::Int ? v->i : fallback;
  }
  std::string text(const std::string& key) const {
    const Msgpack* v = at(key);
    return v && v->kind == Kind::Str ? v->s : "";
  }
};

// A big-endian field of `n` bytes.
uint64_t be(const Reader& r, uint64_t at, uint32_t n) {
  uint64_t v = 0;
  for (uint32_t i = 0; i < n; ++i) v = v << 8 | r.u8(at + i);
  return v;
}

Msgpack parse_msgpack(const Reader& r, uint64_t& at, uint64_t end, int depth = 0) {
  if (depth > 32) throw Error::make(Err::ProfileParse, r.origin, ": metadata nested past 32 deep");
  if (at >= end) throw Error::make(Err::ProfileParse, r.origin, ": metadata ends early");
  Msgpack v;
  const uint8_t tag = r.u8(at++);
  const auto bytes = [&](uint64_t n) {
    const std::string s = r.str(at, n);
    at += n;
    return s;
  };
  const auto fill_array = [&](uint64_t n) {
    v.kind = Msgpack::Kind::Array;
    for (uint64_t i = 0; i < n; ++i) v.list.push_back(parse_msgpack(r, at, end, depth + 1));
  };
  const auto fill_map = [&](uint64_t n) {
    v.kind = Msgpack::Kind::Map;
    for (uint64_t i = 0; i < n; ++i) {
      Msgpack key = parse_msgpack(r, at, end, depth + 1);
      v.map.emplace_back(std::move(key), parse_msgpack(r, at, end, depth + 1));
    }
  };
  if (tag <= 0x7F) {   // positive fixint
    v.kind = Msgpack::Kind::Int;
    v.i = tag;
  } else if (tag >= 0xE0) {   // negative fixint
    v.kind = Msgpack::Kind::Int;
    v.i = static_cast<int8_t>(tag);
  } else if ((tag & 0xF0) == 0x80) {
    fill_map(tag & 0x0F);
  } else if ((tag & 0xF0) == 0x90) {
    fill_array(tag & 0x0F);
  } else if ((tag & 0xE0) == 0xA0) {
    v.kind = Msgpack::Kind::Str;
    v.s = bytes(tag & 0x1F);
  } else {
    switch (tag) {
      case 0xC0: v.kind = Msgpack::Kind::Nil; break;
      case 0xC2:
      case 0xC3:
        v.kind = Msgpack::Kind::Bool;
        v.b = tag == 0xC3;
        break;
      case 0xCC: v.kind = Msgpack::Kind::Int; v.i = static_cast<int64_t>(be(r, at, 1)); at += 1; break;
      case 0xCD: v.kind = Msgpack::Kind::Int; v.i = static_cast<int64_t>(be(r, at, 2)); at += 2; break;
      case 0xCE: v.kind = Msgpack::Kind::Int; v.i = static_cast<int64_t>(be(r, at, 4)); at += 4; break;
      case 0xCF: v.kind = Msgpack::Kind::Int; v.i = static_cast<int64_t>(be(r, at, 8)); at += 8; break;
      case 0xD0: v.kind = Msgpack::Kind::Int; v.i = static_cast<int8_t>(be(r, at, 1)); at += 1; break;
      case 0xD1: v.kind = Msgpack::Kind::Int; v.i = static_cast<int16_t>(be(r, at, 2)); at += 2; break;
      case 0xD2: v.kind = Msgpack::Kind::Int; v.i = static_cast<int32_t>(be(r, at, 4)); at += 4; break;
      case 0xD3: v.kind = Msgpack::Kind::Int; v.i = static_cast<int64_t>(be(r, at, 8)); at += 8; break;
      case 0xD9: { const uint32_t n = static_cast<uint32_t>(be(r, at, 1)); at += 1; v.kind = Msgpack::Kind::Str; v.s = bytes(n); break; }
      case 0xDA: { const uint32_t n = static_cast<uint32_t>(be(r, at, 2)); at += 2; v.kind = Msgpack::Kind::Str; v.s = bytes(n); break; }
      case 0xDB: { const uint32_t n = static_cast<uint32_t>(be(r, at, 4)); at += 4; v.kind = Msgpack::Kind::Str; v.s = bytes(n); break; }
      case 0xDC: { const uint32_t n = static_cast<uint32_t>(be(r, at, 2)); at += 2; fill_array(n); break; }
      case 0xDD: { const uint32_t n = static_cast<uint32_t>(be(r, at, 4)); at += 4; fill_array(n); break; }
      case 0xDE: { const uint32_t n = static_cast<uint32_t>(be(r, at, 2)); at += 2; fill_map(n); break; }
      case 0xDF: { const uint32_t n = static_cast<uint32_t>(be(r, at, 4)); at += 4; fill_map(n); break; }
      default:
        throw Error::make(Err::ProfileParse, r.origin, ": metadata has a MessagePack type this does not read (0x",
                          std::to_string(tag), ")");
    }
  }
  return v;
}

// ---- ELF ------------------------------------------------------------------------

constexpr uint16_t kEmAmdgpu = 224;
constexpr uint32_t kNtAmdgpuMetadata = 32;

struct Section {
  std::string name;
  uint32_t type = 0;
  uint64_t flags = 0, addr = 0, offset = 0, size = 0, entsize = 0;
  uint32_t link = 0;
};

// The kernel descriptor, 64 bytes, as the ABI lays it out.
struct Descriptor {
  uint32_t group_segment = 0, private_segment = 0, kernarg_size = 0;
  int64_t entry_offset = 0;
  uint32_t rsrc1 = 0, rsrc2 = 0;
  uint16_t properties = 0;
};

}  // namespace

CodeObject load_code_object(const std::string& bytes, const std::string& origin) {
  const Reader r{bytes, origin};
  if (bytes.size() < 64 || bytes.compare(0, 4, "\x7F" "ELF") != 0)
    throw Error::make(Err::ProfileParse, origin, ": not an ELF file");
  if (r.u8(4) != 2 || r.u8(5) != 1)
    throw Error::make(Err::ProfileParse, origin, ": not a 64-bit little-endian ELF");
  if (r.u16(18) != kEmAmdgpu)
    throw Error::make(Err::ProfileParse, origin, ": not an AMDGPU code object (e_machine ", r.u16(18), ")");

  // The section headers, and their names.
  const uint64_t shoff = r.u64(0x28);
  const uint16_t shentsize = r.u16(0x3A), shnum = r.u16(0x3C), shstrndx = r.u16(0x3E);
  const auto header = [&](uint16_t i) {
    const uint64_t at = shoff + uint64_t{shentsize} * i;
    Section s;
    s.type = r.u32(at + 4);
    s.flags = r.u64(at + 8);
    s.addr = r.u64(at + 16);
    s.offset = r.u64(at + 24);
    s.size = r.u64(at + 32);
    s.link = r.u32(at + 40);
    s.entsize = r.u64(at + 56);
    return std::pair{s, r.u32(at)};   // the section and its name's offset
  };
  if (shstrndx >= shnum) throw Error::make(Err::ProfileParse, origin, ": no section names");
  const uint64_t shstr = header(shstrndx).first.offset;
  std::vector<Section> sections;
  for (uint16_t i = 0; i < shnum; ++i) {
    auto [s, name_off] = header(i);
    s.name = r.cstr(shstr + name_off);
    sections.push_back(std::move(s));
  }
  const auto section = [&](const std::string& name) -> const Section* {
    for (const auto& s : sections)
      if (s.name == name) return &s;
    return nullptr;
  };

  CodeObject out;
  if (const Section* text = section(".text")) {
    r.need(text->offset, text->size);
    out.text.assign(bytes.begin() + static_cast<long>(text->offset),
                    bytes.begin() + static_cast<long>(text->offset + text->size));
    out.text_addr = text->addr;
  } else {
    throw Error::make(Err::ProfileParse, origin, ": no .text section");
  }

  // A linked code object -- what hipcc embeds in a program -- is laid out
  // already: every section it loads has its address, and the code reaches its
  // constants and variables relative to where it is itself (s_getpc_b64 and a
  // fixed distance). So the whole of it goes on the device as one image, each
  // section at its own address from wherever the image starts.
  out.linked = r.u16(16) != 1 /* ET_REL */;
  if (out.linked) {
    uint64_t end = 0;
    for (const Section& sec : sections)
      if (sec.flags & 2 /* SHF_ALLOC */) end = std::max(end, sec.addr + sec.size);
    out.image.assign(end, 0);
    for (const Section& sec : sections) {
      if (!(sec.flags & 2) || sec.type == 8 /* SHT_NOBITS */ || !sec.size) continue;
      r.need(sec.offset, sec.size);
      std::memcpy(out.image.data() + sec.addr, bytes.data() + sec.offset, sec.size);
    }
  }

  // An object not yet linked has no addresses: its initialised variables and
  // zeroed ones are laid out one after the other, and a loader places this on
  // the device and tells the code where (place_globals).
  std::map<uint16_t, uint64_t> data_at;   // section index -> where it starts in the image
  for (uint16_t i = 0; i < shnum && !out.linked; ++i) {
    const Section& sec = sections[i];
    const bool bss = sec.type == 8 /* SHT_NOBITS */;
    if (sec.name != ".data" && sec.name != ".bss" && !(bss && sec.name.rfind(".bss", 0) == 0)) continue;
    const uint64_t align = 16;
    uint64_t at = (out.data.size() + align - 1) & ~(align - 1);
    out.data.resize(at + sec.size, 0);
    if (!bss) {
      r.need(sec.offset, sec.size);
      std::memcpy(out.data.data() + at, bytes.data() + sec.offset, sec.size);
    }
    data_at[i] = at;
  }

  // Every kernel's descriptor, by the symbol that names it ("<kernel>.kd"),
  // and where each kernel's code is, from the function symbol of its name.
  // The descriptor's entry offset is a relocation in an object that has not
  // been linked, so the symbol is what says where the code starts.
  std::map<std::string, Descriptor> descriptors;
  std::map<std::string, std::pair<uint64_t, uint64_t>> code;   // name -> address, size
  std::map<std::string, uint64_t> symbol_at;                   // a global's offset in the data image
  std::map<uint32_t, std::string> symbol_name;                 // by index, for the relocations
  for (const auto& s : sections) {
    if (s.type != 2 /* SHT_SYMTAB */ || !s.entsize) continue;
    if (s.link >= sections.size()) continue;
    const uint64_t strtab = sections[s.link].offset;
    for (uint64_t at = s.offset; at + s.entsize <= s.offset + s.size; at += s.entsize) {
      const std::string name = r.cstr(strtab + r.u32(at));
      symbol_name[static_cast<uint32_t>((at - s.offset) / s.entsize)] = name;
      const uint8_t type = r.u8(at + 4) & 0xF;
      const uint16_t shndx = r.u16(at + 6);
      const uint64_t value = r.u64(at + 8), size = r.u64(at + 16);
      if (type == 2 /* STT_FUNC */ && !name.empty()) code[name] = {value, size};
      // A variable in the module's memory, and where it lands in the image.
      const bool loaded = out.linked && shndx < sections.size() && (sections[shndx].flags & 2);
      if (type == 1 /* STT_OBJECT */ && !name.empty() && (data_at.count(shndx) || loaded)) {
        GlobalVar g;
        g.name = name;
        g.offset = out.linked ? value : data_at[shndx] + value;   // linked: its address in the image
        g.size = size;
        out.globals.push_back(std::move(g));
        symbol_at[name] = g.offset;
      }
      if (name.size() < 4 || name.compare(name.size() - 3, 3, ".kd") != 0) continue;
      if (shndx >= sections.size() || size < 64) continue;
      const uint64_t kd = sections[shndx].offset + (value - sections[shndx].addr);
      Descriptor d;
      d.group_segment = r.u32(kd + 0);
      d.private_segment = r.u32(kd + 4);
      d.kernarg_size = r.u32(kd + 8);
      d.entry_offset = static_cast<int64_t>(r.u64(kd + 16));
      d.rsrc1 = r.u32(kd + 48);
      d.rsrc2 = r.u32(kd + 52);
      d.properties = r.u16(kd + 56);
      descriptors[name.substr(0, name.size() - 3)] = d;
    }
  }

  // What the code has to be told once the module is placed: where each global
  // ended up. The compiler leaves room for the address and a relocation
  // saying which global it meant.
  for (const Section& sec : sections) {
    if (sec.type != 4 /* SHT_RELA */ || sec.name != ".rela.text" || !sec.entsize) continue;
    for (uint64_t at = sec.offset; at + sec.entsize <= sec.offset + sec.size; at += sec.entsize) {
      const uint64_t where = r.u64(at);
      const uint64_t info = r.u64(at + 8);
      const int64_t addend = static_cast<int64_t>(r.u64(at + 16));
      const uint32_t kind = static_cast<uint32_t>(info), sym = static_cast<uint32_t>(info >> 32);
      // R_AMDGPU_REL32_LO and _HI: the halves of an address relative to the
      // instruction that reads it.
      if (kind != 10 && kind != 11) continue;
      const auto named = symbol_name.find(sym);
      if (named == symbol_name.end()) continue;
      if (const auto found = symbol_at.find(named->second); found != symbol_at.end()) {
        out.relocations.push_back({where, found->second, addend, kind == 11, false});
        continue;
      }
      // A call: the symbol is a function in this module's own code.
      const auto called = code.find(named->second);
      if (called == code.end())
        throw Error::make(Err::ProfileParse, origin, ": the code refers to ", named->second,
                          ", which is not a variable or a function this module defines");
      out.relocations.push_back({where, called->second.first, addend, kind == 11, true});
    }
  }

  // The metadata note: a MessagePack map, with the kernels under
  // "amdhsa.kernels" and the target under "amdhsa.target". An object linked
  // from several -- Tensile's libraries are hundreds of kernels linked into
  // one -- keeps each one's note, so every note's kernels are the object's.
  std::vector<Msgpack> notes;
  for (const auto& s : sections) {
    if (s.type != 7 /* SHT_NOTE */) continue;
    for (uint64_t at = s.offset; at + 12 <= s.offset + s.size;) {
      const uint32_t namesz = r.u32(at), descsz = r.u32(at + 4), type = r.u32(at + 8);
      const uint64_t name_at = at + 12, desc_at = name_at + ((namesz + 3) & ~3u);
      const std::string name = r.str(name_at, namesz ? namesz - 1 : 0);
      if (type == kNtAmdgpuMetadata && name == "AMDGPU") {
        uint64_t p = desc_at;
        Msgpack note = parse_msgpack(r, p, desc_at + descsz);
        if (const Msgpack* k = note.at("amdhsa.kernels"); k && k->kind == Msgpack::Kind::Array)
          notes.push_back(std::move(note));
      }
      at = desc_at + ((descsz + 3) & ~3u);
    }
  }
  if (notes.empty()) throw Error::make(Err::ProfileParse, origin, ": no AMDGPU metadata note naming its kernels");
  const Msgpack& metadata = notes.front();
  if (const Msgpack* t = metadata.at("amdhsa.target"); t && t->kind == Msgpack::Kind::Str) {
    out.target = t->s;
    out.isa = out.target.substr(out.target.rfind('-') + 1);
  }
  if (const Msgpack* v = metadata.at("amdhsa.version");
      v && v->kind == Msgpack::Kind::Array && v->list.size() >= 2) {
    out.abi_major = static_cast<uint32_t>(v->list[0].i);
    out.abi_minor = static_cast<uint32_t>(v->list[1].i);
  }

  for (const Msgpack& note : notes)
  for (const Msgpack& k : note.at("amdhsa.kernels")->list) {
    if (k.kind != Msgpack::Kind::Map) continue;
    Kernel kern;
    kern.name = k.text(".name");
    if (kern.name.empty()) throw Error::make(Err::ProfileParse, origin, ": a kernel in the metadata has no name");
    kern.kernarg_size = static_cast<uint32_t>(k.number(".kernarg_segment_size"));
    kern.kernarg_align = static_cast<uint32_t>(k.number(".kernarg_segment_align", 8));
    kern.group_segment = static_cast<uint32_t>(k.number(".group_segment_fixed_size"));
    kern.private_segment = static_cast<uint32_t>(k.number(".private_segment_fixed_size"));
    kern.sgpr_count = static_cast<uint32_t>(k.number(".sgpr_count"));
    kern.vgpr_count = static_cast<uint32_t>(k.number(".vgpr_count"));
    kern.agpr_count = static_cast<uint32_t>(k.number(".agpr_count"));
    kern.max_flat_workgroup_size = static_cast<uint32_t>(k.number(".max_flat_workgroup_size", 1024));
    kern.wavefront_size = static_cast<uint32_t>(k.number(".wavefront_size", 64));
    if (const Msgpack* args = k.at(".args"); args && args->kind == Msgpack::Kind::Array)
      for (const Msgpack& a : args->list) {
        KernelArg arg;
        arg.offset = static_cast<uint32_t>(a.number(".offset"));
        arg.size = static_cast<uint32_t>(a.number(".size"));
        arg.kind = a.text(".value_kind");
        arg.address_space = a.text(".address_space");
        kern.args.push_back(std::move(arg));
      }
    // The descriptor: where the code starts, and which user SGPRs the
    // hardware loads before it (kernel_code_properties).
    std::string symbol = k.text(".symbol");   // "<kernel>.kd", where the metadata gives it
    if (symbol.size() > 3 && symbol.compare(symbol.size() - 3, 3, ".kd") == 0) symbol.resize(symbol.size() - 3);
    const auto it = descriptors.find(symbol.empty() ? kern.name : symbol);
    if (it == descriptors.end())
      throw Error::make(Err::ProfileParse, origin, ": kernel ", kern.name, " has no descriptor symbol");
    const Descriptor& d = it->second;
    if (const auto at_code = code.find(kern.name); at_code != code.end()) {
      kern.entry = at_code->second.first;
      kern.size = at_code->second.second;
    } else {
      kern.entry = static_cast<uint64_t>(static_cast<int64_t>(out.text_addr) + d.entry_offset);
    }
    if (kern.entry < out.text_addr || kern.entry - out.text_addr > out.text.size())
      throw Error::make(Err::ProfileParse, origin, ": kernel ", kern.name, " starts outside .text");
    if (!kern.kernarg_size) kern.kernarg_size = d.kernarg_size;
    if (!kern.group_segment) kern.group_segment = d.group_segment;
    if (!kern.private_segment) kern.private_segment = d.private_segment;
    kern.private_segment_buffer = d.properties & (1u << 0);
    kern.dispatch_ptr = d.properties & (1u << 1);
    kern.queue_ptr = d.properties & (1u << 2);
    kern.kernarg_segment_ptr = d.properties & (1u << 3);
    kern.dispatch_id = d.properties & (1u << 4);
    kern.flat_scratch_init = d.properties & (1u << 5);
    kern.user_sgpr_count = 2 * kern.private_segment_buffer + 2 * kern.dispatch_ptr + 2 * kern.queue_ptr +
                           2 * kern.kernarg_segment_ptr + 2 * kern.dispatch_id + 2 * kern.flat_scratch_init;
    out.kernels.push_back(std::move(kern));
  }
  if (out.kernels.empty()) throw Error::make(Err::ProfileParse, origin, ": no kernels");
  return out;
}

const Kernel* find_kernel(const CodeObject& o, const std::string& name) {
  for (const Kernel& k : o.kernels)
    if (k.name == name) return &k;
  return nullptr;
}

}  // namespace vgpu::amd

namespace vgpu::amd {

void place_globals(CodeObject& o, uint64_t base) {
  if (o.placed && o.data_base != base)
    throw Error::make(Err::InvalidValue, "this module's globals are already at another address");
  for (const Relocation& rel : o.relocations) {
    // The address the code needs, relative to the instruction that reads it,
    // which is how a kernel reaches a global: the program counter plus a
    // constant. Where the code reaches another part of itself -- a call --
    // the distance is the same wherever the module was placed, and the base
    // does not come into it.
    const uint64_t symbol = rel.in_text ? o.text_addr + rel.symbol : base + rel.symbol;
    const uint64_t place = o.text_addr + rel.at;
    const uint64_t value = symbol + static_cast<uint64_t>(rel.addend) - place;
    const uint32_t half = static_cast<uint32_t>(rel.high ? value >> 32 : value);
    if (rel.at + 4 > o.text.size())
      throw Error::make(Err::ProfileParse, "a relocation points past the end of the code");
    for (uint32_t b = 0; b < 4; ++b) o.text[rel.at + b] = static_cast<uint8_t>(half >> (8 * b));
  }
  o.placed = true;
  o.data_base = base;
}

const GlobalVar* find_global(const CodeObject& o, const std::string& name) {
  for (const GlobalVar& g : o.globals)
    if (g.name == name) return &g;
  return nullptr;
}

}  // namespace vgpu::amd
