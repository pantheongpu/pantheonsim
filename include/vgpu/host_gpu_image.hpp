// Host-GPU mode: the cubin the card loads for a module that has variables (Round sp66, docs/perf-design/12-host-gpu-turbo.md section 3).
//
// The card runs the module's own code, but where that code finds a `__device__` or `__constant__` variable is decided by whoever loads the
// cubin: the simulator put each variable in its own device memory, at an address of the window the card mirrors; the real driver would put
// its own copy somewhere else. Two copies of one variable cannot both be the truth: a value set with cudaMemcpyToSymbol, a pointer taken
// with cudaGetSymbolAddress, a kernel of the simulator that wrote it, all name the simulator's copy. So the card is given a cubin whose
// references to the variables are the simulator's addresses: every relocation whose symbol lies in a section of variables is applied here,
// with the address the simulator gave that section, exactly as the simulator's loader applied it (src/sass/module.cpp), and taken out of
// the image, so that the driver leaves it as it is. The driver still makes its own copies of the variables (nothing addresses them) and
// resolves every other relocation (code addresses, vprintf).
//
// What this does not move: a `__constant__` variable read with c[bank][offset] is read from the card's constant bank, which no relocation
// names. The mode copies the simulator's bytes into that bank before every launch of such a module (src/runtime/host_gpu.cpp).
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vgpu/sass/reloc_patch.hpp"

namespace vgpu::hostgpu {

struct CardImage {
  bool ok = false;
  std::string why;              // !ok: why no such image can be made (the module's kernels then stay in the simulator)
  std::vector<uint8_t> bytes;   // ok: the cubin to load on the card
  unsigned applied = 0;         // relocations applied here and taken out
};

// `var_va(section)`: the simulator's address of a section whose variables the card must address where the simulator holds them, or nullopt
// for any other section (whose relocations are left to the driver).
inline CardImage card_image_at_sim_addresses(const uint8_t* img, size_t n, const std::function<std::optional<uint64_t>(const std::string&)>& var_va) {
  CardImage out;
  const auto fail = [&](std::string why) {
    out.ok = false;
    out.why = std::move(why);
    out.bytes.clear();
    return out;
  };
  const auto rd = [&](uint64_t at, unsigned len, uint64_t* v) {
    if (at > n || len > n - at) return false;
    uint64_t x = 0;
    std::memcpy(&x, img + at, len);   // (little-endian, as the cubin)
    *v = x;
    return true;
  };
  uint64_t shoff = 0, shentsize = 0, shnum = 0, shstrndx = 0, cls = 0;
  if (n < 64 || std::memcmp(img, "\x7f" "ELF", 4) != 0 || !rd(4, 1, &cls) || cls != 2) return fail("not a 64-bit ELF image");
  if (!rd(0x28, 8, &shoff) || !rd(0x3a, 2, &shentsize) || !rd(0x3c, 2, &shnum) || !rd(0x3e, 2, &shstrndx) || shentsize < 64 || shnum == 0 ||
      shstrndx >= shnum || shoff > n || shnum * shentsize > n - shoff)
    return fail("a bad section header table");
  struct Sh {
    uint64_t name, type, offset, size, link, info, entsize;
  };
  std::vector<Sh> sh(shnum);
  for (uint64_t i = 0; i < shnum; ++i) {
    const uint64_t o = shoff + i * shentsize;
    Sh& s = sh[i];
    rd(o, 4, &s.name);
    rd(o + 4, 4, &s.type);
    rd(o + 24, 8, &s.offset);
    rd(o + 32, 8, &s.size);
    rd(o + 40, 4, &s.link);
    rd(o + 44, 4, &s.info);
    rd(o + 56, 8, &s.entsize);
    if (s.type != 8 && (s.offset > n || s.size > n - s.offset)) return fail("a section runs past the end of the image");   // (8: SHT_NOBITS)
  }
  const auto name_of = [&](uint64_t i) {
    std::string s;
    for (uint64_t at = sh[shstrndx].offset + sh[i].name; at < n && img[at]; ++at) s += static_cast<char>(img[at]);
    return s;
  };
  out.bytes.assign(img, img + n);
  constexpr uint64_t kRel = 9, kRela = 4, kSymtab = 2, kNobits = 8;
  for (uint64_t i = 0; i < shnum; ++i) {
    if (sh[i].type != kRel && sh[i].type != kRela) continue;
    const uint64_t es = sh[i].type == kRela ? 24 : 16;
    const uint64_t dst = sh[i].info, symtab = sh[i].link;
    if (dst == 0 || dst >= shnum || symtab >= shnum || sh[symtab].type != kSymtab || sh[symtab].entsize < 24) continue;   // (the loader's rule: what it does not patch)
    const std::string dst_name = name_of(dst);
    const bool code = dst_name.rfind(".text.", 0) == 0;
    uint64_t kept = 0;   // entries left for the driver, moved down over the ones applied
    for (uint64_t o = sh[i].offset; o + es <= sh[i].offset + sh[i].size; o += es) {
      uint64_t r_offset = 0, r_info = 0, r_addend = 0;
      rd(o, 8, &r_offset);
      rd(o + 8, 8, &r_info);
      if (es == 24) rd(o + 16, 8, &r_addend);
      const uint64_t sym = r_info >> 32, type = r_info & 0xffffffffu;
      std::optional<uint64_t> base;
      uint64_t st_value = 0, st_shndx = 0;
      if (sym * sh[symtab].entsize < sh[symtab].size) {
        const uint64_t so = sh[symtab].offset + sym * sh[symtab].entsize;
        rd(so + 6, 2, &st_shndx);
        rd(so + 8, 8, &st_value);
        if (st_shndx != 0 && st_shndx < shnum) base = var_va(name_of(st_shndx));
      }
      if (!base) {
        if (kept * es != o - sh[i].offset) std::memmove(&out.bytes[sh[i].offset + kept * es], &out.bytes[o], es);
        ++kept;
        continue;
      }
      // A relocation against a variable: applied here as the simulator's loader applies it.
      if (sh[dst].type == kNobits) return fail("a relocation into " + dst_name + ", which has no bytes");
      const uint64_t value = *base + st_value + r_addend;
      if (code && sass::code_reloc_type(static_cast<uint32_t>(type))) {
        if (r_offset > sh[dst].size || sh[dst].size - r_offset < 16) return fail("a relocation past the end of " + dst_name);
        sass::patch_code_word(&out.bytes[sh[dst].offset + r_offset], static_cast<uint32_t>(type), value);
      } else if (!code && (type == 2 || type == 4)) {   // R_CUDA_64, R_CUDA_G64
        if (r_offset > sh[dst].size || sh[dst].size - r_offset < 8) return fail("a relocation past the end of " + dst_name);
        std::memcpy(&out.bytes[sh[dst].offset + r_offset], &value, 8);
      } else {
        return fail("relocation type " + std::to_string(type) + " of a variable in " + dst_name);
      }
      ++out.applied;
    }
    // The table is shorter by the entries applied: its size in the section header says so (the bytes after it are left, unread).
    const uint64_t new_size = kept * es;
    std::memcpy(&out.bytes[shoff + i * shentsize + 32], &new_size, 8);
  }
  out.ok = true;
  return out;
}

}  // namespace vgpu::hostgpu
