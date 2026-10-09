// Which cubin the driver runs from a fatbin: sm_XYa SASS runs on XY alone, so
// on a compute capability 10.3 (B300) or 10.7 (Vera Rubin) profile an sm_100a
// cubin is no candidate and the fatbin's PTX is what runs. pick_cubin used to
// take the sm_100a image there, because it ignored the arch-specific flag, and
// the load then refused it instead of falling back.
//
// The cubins are built here: an ELF header, a section string table and the
// .nv.compat section whose attribute 0x09 holds the flag (the layout nvcc
// writes, measured on CUDA 12.0's and 13.0's output for sm_90 to sm_120, with
// and without the "a"). They carry no code, which is all pick_cubin needs.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fatbin.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/cubin.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::cuda::FatbinImage;

namespace {

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A linked (ET_EXEC) CUDA 13 cubin for sm_<sm>, arch-specific or not.
std::string make_cubin(uint32_t sm, bool arch_specific) {
  const std::string names = std::string("\0.shstrtab\0.nv.compat\0", 21);   // .shstrtab at 1, .nv.compat at 11
  std::string compat;
  const uint8_t recs[] = {0x02, 0x09, static_cast<uint8_t>(arch_specific ? 1 : 0), 0x00,   // the flag
                          0x02, 0x02, 0x01, 0x00};                                          // another record
  compat.assign(reinterpret_cast<const char*>(recs), sizeof recs);

  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 8) out.push_back('\0');
  const uint64_t compat_off = out.size();
  out += compat;
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name);
    put<uint32_t>(out, type);
    put<uint64_t>(out, 0);      // flags
    put<uint64_t>(out, 0);      // addr
    put<uint64_t>(out, off);
    put<uint64_t>(out, size);
    put<uint32_t>(out, 0);      // link
    put<uint32_t>(out, 0);      // info
    put<uint64_t>(out, 4);      // align
    put<uint64_t>(out, 0);      // entsize
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());      // .shstrtab, SHT_STRTAB
  shdr(11, 1, compat_off, compat.size());   // .nv.compat
  out[0] = 0x7f;
  out[1] = 'E';
  out[2] = 'L';
  out[3] = 'F';
  out[4] = 2;       // 64-bit
  out[5] = 1;
  out[6] = 1;
  out[7] = 0x41;    // the CUDA OS ABI
  out[8] = 8;       // ABI version 8: e_flags' second byte names the architecture
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 3, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (sm << 8);
  std::memcpy(&out[16], &type, 2);
  std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8);
  std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2);
  std::memcpy(&out[0x3c], &shnum, 2);
  std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

FatbinImage cubin_image(uint32_t sm, bool arch_specific) {
  FatbinImage im;
  im.kind = cuda::kFatbinElf;
  im.arch = sm;
  im.major = 8;
  im.data = make_cubin(sm, arch_specific);
  return im;
}

FatbinImage ptx_image(uint32_t sm, const char* target) {
  FatbinImage im;
  im.kind = cuda::kFatbinPtx;
  im.arch = sm;
  im.major = 8;
  im.minor = 8;
  im.name = "k.ptx";
  im.data = std::string(".version 8.8\n.target ") + target + "\n.address_size 64\n";
  return im;
}

uint32_t cc_of(const char* id) {
  const DeviceProfile p = load_gpu(id);
  return static_cast<uint32_t>(p.cc_major * 10 + p.cc_minor);
}

bool picks(const std::string& fatbin, uint32_t cc, const std::string& cubin) {
  return cuda::pick_cubin(fatbin.data(), cc) == cubin;
}

}  // namespace

VTEST(the_flag_is_read_from_the_cubin) {
  for (uint32_t sm : {90u, 100u, 103u, 120u}) {
    const std::string a = make_cubin(sm, true), plain = make_cubin(sm, false);
    VCHECK(sass::cubin_arch_specific(reinterpret_cast<const uint8_t*>(a.data()), a.size()));
    VCHECK(!sass::cubin_arch_specific(reinterpret_cast<const uint8_t*>(plain.data()), plain.size()));
    VCHECK(sass::parse_cubin(reinterpret_cast<const uint8_t*>(a.data()), a.size()).arch_specific);
    VCHECK(!sass::parse_cubin(reinterpret_cast<const uint8_t*>(plain.data()), plain.size()).arch_specific);
  }
  const std::string junk(100, 'x');
  VCHECK(!sass::cubin_arch_specific(reinterpret_cast<const uint8_t*>(junk.data()), junk.size()));
  std::string cut = make_cubin(100, true);
  cut.resize(cut.size() - 40);   // the section table is cut short
  VCHECK(!sass::cubin_arch_specific(reinterpret_cast<const uint8_t*>(cut.data()), cut.size()));
}

VTEST(an_sm_100a_cubin_is_no_candidate_on_a_b300_or_a_vera_rubin) {
  unsetenv("VGPU_SASS");
  const uint32_t b200 = cc_of("nvidia/b200"), b300 = cc_of("nvidia/b300"), vr200 = cc_of("nvidia/vr200");
  VCHECK(b200 == 100);
  VCHECK(b300 == 103);
  VCHECK(vr200 == 107);
  // nvcc -arch=sm_100a: the sm_100a cubin and compute_100a PTX.
  const FatbinImage cubin = cubin_image(100, true);
  const std::string only_a = cuda::write_fatbin({cubin, ptx_image(100, "sm_100a")});
  VCHECK(picks(only_a, b200, cubin.data));
  VCHECK(picks(only_a, b300, ""));
  VCHECK(picks(only_a, vr200, ""));
  // With plain PTX beside it (-gencode arch=compute_100a,code=sm_100a
  // -gencode arch=compute_100,code=compute_100): no cubin on the newer parts,
  // and the PTX is what the load takes.
  const std::string with_ptx = cuda::write_fatbin({cubin, ptx_image(100, "sm_100")});
  VCHECK(picks(with_ptx, b200, cubin.data));
  for (uint32_t cc : {b300, vr200}) {
    VCHECK(picks(with_ptx, cc, ""));
    const auto ptxs = cuda::extract_ptx(with_ptx.data(), with_ptx.size());
    VCHECK(ptxs.size() == 1);
    VCHECK(cuda::pick_ptx(ptxs, cc) == 0);
  }
}

VTEST(a_plain_or_family_cubin_still_runs_on_the_newer_minors) {
  unsetenv("VGPU_SASS");
  const uint32_t b200 = cc_of("nvidia/b200"), b300 = cc_of("nvidia/b300"), vr200 = cc_of("nvidia/vr200");
  const FatbinImage plain = cubin_image(100, false);   // sm_100 and sm_100f write the same flag
  const std::string fb = cuda::write_fatbin({plain});
  VCHECK(picks(fb, b200, plain.data));
  VCHECK(picks(fb, b300, plain.data));
  VCHECK(picks(fb, vr200, plain.data));
  // Not across a major version, and not to an older device.
  VCHECK(picks(fb, cc_of("nvidia/h100"), ""));
  VCHECK(picks(fb, cc_of("nvidia/rtx5090"), ""));
  const std::string fb103 = cuda::write_fatbin({cubin_image(103, false)});
  VCHECK(picks(fb103, b200, ""));
}

VTEST(of_two_cubins_for_one_architecture_the_arch_specific_one_wins_where_it_runs) {
  unsetenv("VGPU_SASS");
  const uint32_t b200 = cc_of("nvidia/b200"), b300 = cc_of("nvidia/b300");
  const FatbinImage plain = cubin_image(100, false), spec = cubin_image(100, true);
  const std::string fb = cuda::write_fatbin({plain, spec});
  VCHECK(picks(fb, b200, spec.data));
  VCHECK(picks(fb, b300, plain.data));   // the a image does not run there: the plain one does
  const std::string reversed = cuda::write_fatbin({spec, plain});
  VCHECK(picks(reversed, b200, spec.data));
  VCHECK(picks(reversed, b300, plain.data));
  // A newer minor's cubin beats an older one's, both plain.
  const FatbinImage newer = cubin_image(103, false);
  VCHECK(picks(cuda::write_fatbin({plain, newer}), b300, newer.data));
  VCHECK(picks(cuda::write_fatbin({newer, plain}), b200, plain.data));
}
