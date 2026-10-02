// The PTX linker behind nvJitLink, and the fatbin writer and container
// readers behind nvFatbin and nvJitLink's file inputs -- checked without a
// toolkit, so hosted CI covers them on every pull request. The PTX modules are
// nvjitlink_paths' (jitlink_ptx.inc), whose link an RTX 3060 runs through
// NVIDIA's nvJitLink and VirtualGPU's alike; here the linked module also runs
// on a simulated A100.
#include <cstring>
#include <string>
#include <vector>

#include "fatbin.hpp"
#include "ptx_link.hpp"
#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

#include "../../nvidia/tests/e2e/jitlink_ptx.inc"

using namespace vgpu;
using vgpu::cuda::FatbinImage;
using vgpu::cuda::PtxInput;

namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

size_t count(const std::string& hay, const std::string& needle) {
  size_t n = 0;
  for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + 1)) ++n;
  return n;
}

// Runs the linked module's k over 32 threads on a simulated A100 and checks
// every thread's answer.
bool runs_k(const std::string& ptx) {
  runtime::Runtime rt(load_gpu("nvidia/a100"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  if (!fn) return false;
  const uint64_t out = dev.memory().alloc(32 * 4);
  exec::LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
  for (uint32_t t = 0; t < 32; ++t)
    if (dev.memory().load_scalar(out + 4 * t, 4) != static_cast<uint32_t>(expected(static_cast<int>(t))))
      return false;
  return true;
}

// A 64-bit ELF relocatable file with the given sections (after the null one
// and .shstrtab), as nvcc's host objects are laid out.
std::string elf_object(uint16_t machine, const std::vector<std::pair<std::string, std::string>>& sections) {
  std::string shstr(1, '\0');
  std::vector<uint32_t> name_at;
  for (const auto& s : sections) {
    name_at.push_back(static_cast<uint32_t>(shstr.size()));
    shstr += s.first;
    shstr.push_back('\0');
  }
  const uint32_t shstr_name = static_cast<uint32_t>(shstr.size());
  shstr += ".shstrtab";
  shstr.push_back('\0');
  std::string body;
  std::vector<uint64_t> offsets;
  for (const auto& s : sections) {
    while (body.size() % 8) body.push_back('\0');
    offsets.push_back(64 + body.size());
    body += s.second;
  }
  while (body.size() % 8) body.push_back('\0');
  const uint64_t shstr_off = 64 + body.size();
  body += shstr;
  while (body.size() % 8) body.push_back('\0');
  const uint64_t shoff = 64 + body.size();
  const uint16_t shnum = static_cast<uint16_t>(sections.size() + 2);
  std::string f(64, '\0');
  f[0] = 0x7f, f[1] = 'E', f[2] = 'L', f[3] = 'F', f[4] = 2, f[5] = 1, f[6] = 1;
  auto put = [](std::string& s, size_t at, uint64_t v, int n) {
    for (int i = 0; i < n; ++i) s[at + i] = static_cast<char>((v >> (8 * i)) & 0xff);
  };
  put(f, 16, 1, 2);   // ET_REL
  put(f, 18, machine, 2);
  put(f, 0x28, shoff, 8);
  put(f, 0x34, 64, 2);
  put(f, 0x3a, 64, 2);
  put(f, 0x3c, shnum, 2);
  put(f, 0x3e, shnum - 1, 2);
  f += body;
  auto header = [&](uint32_t name, uint64_t off, uint64_t size) {
    std::string h(64, '\0');
    put(h, 0, name, 4);
    put(h, 4, 1, 4);   // SHT_PROGBITS
    put(h, 0x18, off, 8);
    put(h, 0x20, size, 8);
    return h;
  };
  f += std::string(64, '\0');
  for (size_t i = 0; i < sections.size(); ++i) f += header(name_at[i], offsets[i], sections[i].second.size());
  f += header(shstr_name, shstr_off, shstr.size());
  return f;
}

FatbinImage ptx_image(uint32_t arch, const std::string& text, uint16_t kind = vgpu::cuda::kFatbinPtx) {
  FatbinImage im;
  im.kind = kind;
  im.arch = arch;
  im.major = 7;
  im.minor = 1;
  im.name = "lib";
  im.data = text;
  return im;
}

}  // namespace

VTEST(linked_modules_resolve_across_inputs_and_run) {
  auto r = cuda::link_ptx({{"main.ptx", kMain}, {"lib.ptx", kLib}}, "sm_80");
  VCHECK(r.ok);
  VCHECK_EQ(r.errors, std::string{});
  // ptxas refuses a definition after an .extern declaration of it, so the
  // declarations of what the link defines are gone; the second module's
  // file-scope twice and scale have names of their own.
  VCHECK(r.ptx.find(".extern .func") == std::string::npos);
  VCHECK(r.ptx.find(".extern .global") == std::string::npos);
  VCHECK(r.ptx.find("twice$vgpu1") != std::string::npos);
  VCHECK(r.ptx.find("scale$vgpu1") != std::string::npos);
  VCHECK(r.ptx.find(".target sm_80") != std::string::npos);
  VCHECK(runs_k(r.ptx));
}

VTEST(an_undefined_reference_fails_the_link_with_its_name) {
  auto r = cuda::link_ptx({{"main.ptx", kMain}}, "sm_80");
  VCHECK(!r.ok);
  VCHECK_CONTAINS(r.errors, "Undefined reference to 'add_base' in 'main.ptx'");
  VCHECK_CONTAINS(r.errors, "Undefined reference to 'base' in 'main.ptx'");
  // What ptxas provides itself is not a reference to resolve.
  const std::string printf_user = R"(
.version 7.1
.target sm_80
.address_size 64
.extern .func  (.param .b32 func_retval0) vprintf
(
	.param .b64 vprintf_param_0,
	.param .b64 vprintf_param_1
)
;
.visible .entry p()
{
	ret;
}
)";
  auto p = cuda::link_ptx({{"p.ptx", printf_user}}, "sm_80");
  VCHECK(p.ok);
  VCHECK_EQ(count(p.ptx, ".extern .func"), size_t{1});
}

VTEST(the_first_strong_definition_wins_and_a_second_is_reported) {
  // As NVIDIA's nvJitLink: the second is named and dropped, and the link
  // succeeds.
  auto r = cuda::link_ptx({{"main.ptx", kMain}, {"lib.ptx", kLib}, {"dup.ptx", kDup}}, "sm_80");
  VCHECK(r.ok);
  VCHECK_CONTAINS(r.errors, "Multiple definition of 'add_base' in 'dup.ptx', first defined in 'lib.ptx'");
  VCHECK(r.ptx.find("mov.u32 \t%r1, -1;") == std::string::npos);
  VCHECK(runs_k(r.ptx));
}

VTEST(a_strong_definition_beats_an_earlier_weak_one) {
  auto r = cuda::link_ptx({{"main.ptx", kMain}, {"weak.ptx", kWeak}, {"lib.ptx", kLib}}, "sm_80");
  VCHECK(r.ok);
  VCHECK_EQ(r.errors, std::string{});
  VCHECK(r.ptx.find("-2;") == std::string::npos);
  VCHECK(runs_k(r.ptx));
  // Weak alone is a definition like any other.
  auto w = cuda::link_ptx({{"main.ptx", kMain}, {"weak.ptx", kWeak}, {"lib.ptx", kLib}, {"weak2", kWeak}},
                          "sm_80");
  VCHECK(w.ok);
}

VTEST(a_noreturn_function_is_declared_ahead_of_its_definition) {
  // The linked module declares every function before any use, attributes
  // and all: ".visible .func stop() .noreturn;".
  const std::string user = R"(
.version 7.1
.target sm_80
.address_size 64
.extern .func stop() .noreturn;
.visible .entry k()
{
	ret;
}
)";
  const std::string lib = R"(
.version 7.1
.target sm_80
.address_size 64
.visible .func stop() .noreturn
{
	trap;
}
)";
  auto r = cuda::link_ptx({{"user.ptx", user}, {"lib.ptx", lib}}, "sm_80");
  VCHECK(r.ok);
  VCHECK_CONTAINS(r.ptx, ".visible .func stop() .noreturn;");
  ptx::Module m = ptx::parse(r.ptx);
  VCHECK_EQ(m.funcs.size(), size_t{1});
  VCHECK_EQ(m.entries.size(), size_t{1});
}

VTEST(the_link_carries_the_newest_ptx_version) {
  auto r = cuda::link_ptx({{"main.ptx", kMain}, {"lib.ptx", kLib}, {"hopper.ptx", kHopper}}, "");
  VCHECK(r.ok);
  VCHECK(r.ptx.find(".version 7.8") != std::string::npos);
  VCHECK(r.ptx.find(".target sm_80") != std::string::npos);   // the first input's, with no -arch
}

VTEST(architectures_and_targets) {
  uint32_t n = 0;
  char s = 0;
  VCHECK(cuda::parse_arch("sm_86", &n, &s) && n == 86 && s == 0);
  VCHECK(cuda::parse_arch("compute_90a", &n, &s) && n == 90 && s == 'a');
  VCHECK(cuda::parse_arch("sm_100f", &n, &s) && n == 100 && s == 'f');
  VCHECK(!cuda::parse_arch("86", &n, &s));
  VCHECK(!cuda::parse_arch("sm_", &n, &s));
  VCHECK(!cuda::parse_arch("sm_86x", &n, &s));
  VCHECK(cuda::ptx_module_target(kHopper, &n, &s) && n == 90);
  // What ptxas compiles for what.
  VCHECK(cuda::target_runs_on(80, 0, 86, 0));
  VCHECK(!cuda::target_runs_on(90, 0, 86, 0));
  VCHECK(cuda::target_runs_on(90, 'a', 90, 'a'));
  VCHECK(!cuda::target_runs_on(90, 'a', 90, 0));
  VCHECK(cuda::target_runs_on(100, 'f', 103, 'f'));
  VCHECK(!cuda::target_runs_on(100, 'f', 120, 'f'));
}

VTEST(a_written_fatbin_reads_back_entry_for_entry) {
  std::vector<FatbinImage> in;
  in.push_back(ptx_image(80, kLib));
  in.push_back(ptx_image(90, kHopper, vgpu::cuda::kFatbinRelocPtx));
  FatbinImage elf;
  elf.kind = vgpu::cuda::kFatbinElf;
  elf.arch = 86;
  elf.name = "cubin";
  elf.data = std::string("\x7f" "ELF", 4) + std::string(60, '\x01');
  in.push_back(elf);
  in[0].options = "-O3";
  const std::string image = cuda::write_fatbin(in);
  VCHECK_EQ(image.size() % 8, size_t{0});
  size_t used = 0;
  auto out = cuda::extract_images(image.data(), image.size(), &used);
  VCHECK_EQ(used, image.size());
  VCHECK_EQ(out.size(), size_t{3});
  VCHECK_EQ(out[0].kind, vgpu::cuda::kFatbinPtx);
  VCHECK_EQ(out[0].arch, 80u);
  VCHECK_EQ(out[0].name, std::string("lib"));
  VCHECK_EQ(out[1].kind, vgpu::cuda::kFatbinRelocPtx);
  VCHECK_EQ(out[2].arch, 86u);
  VCHECK(out[2].data.compare(0, elf.data.size(), elf.data) == 0);
  // The PTX reader takes both PTX kinds -- the driver JITs nvFatbinAddReloc's
  // relocatable PTX like any other, as an RTX 3060's does -- and skips SASS.
  auto ptx = cuda::extract_ptx(image.data(), image.size());
  VCHECK_EQ(ptx.size(), size_t{2});
  VCHECK_EQ(ptx[0].text, std::string(kLib));
  VCHECK_EQ(ptx[1].arch, 90u);
  // An empty fatbin is its container header.
  VCHECK_EQ(cuda::write_fatbin({}).size(), size_t{16});
}

VTEST(an_entry_that_will_not_decompress_is_kept_unless_it_is_ptx) {
  // nvcc flags its LTO-IR entries 0x18011, and their payload is not a zstd
  // frame; reading the fatbin must not fail over it.
  std::vector<FatbinImage> in = {ptx_image(80, kLib)};
  FatbinImage lto;
  lto.kind = vgpu::cuda::kFatbinLtoIr;
  lto.arch = 80;
  lto.data = "not a zstd frame";
  in.push_back(lto);
  std::string image = cuda::write_fatbin(in);
  // Set the zstd flag on the second entry (its flags are at +40 of its header).
  auto first = cuda::extract_images(image.data(), image.size());
  VCHECK_EQ(first.size(), size_t{2});
  const size_t second = image.rfind("not a zstd frame") - 64 - 8;   // header + name
  image[second + 41] = static_cast<char>(0x80);
  auto out = cuda::extract_images(image.data(), image.size());
  VCHECK_EQ(out.size(), size_t{2});
  VCHECK(out[1].stored);
}

VTEST(blobs_are_told_apart_by_their_first_bytes) {
  using cuda::BlobKind;
  const std::string ptx = "//\n// Generated\n//\n\n.version 8.0\n.target sm_80\n";
  VCHECK(cuda::classify_blob(ptx.data(), ptx.size()) == BlobKind::Ptx);
  const std::string fatbin = cuda::write_fatbin({ptx_image(80, kLib)});
  VCHECK(cuda::classify_blob(fatbin.data(), fatbin.size()) == BlobKind::Fatbin);
  const std::string cubin = elf_object(190, {});
  VCHECK(cuda::classify_blob(cubin.data(), cubin.size()) == BlobKind::Cubin);
  const std::string object = elf_object(62, {});
  VCHECK(cuda::classify_blob(object.data(), object.size()) == BlobKind::HostObject);
  const std::string ar = "!<arch>\n";
  VCHECK(cuda::classify_blob(ar.data(), ar.size()) == BlobKind::Archive);
  const unsigned char ltoir[8] = {0xed, 0x43, 0x4e, 0x7f, 1, 0x41, 2, 0x62};
  VCHECK(cuda::classify_blob(ltoir, sizeof ltoir) == BlobKind::LtoIr);
  const std::string text = "this is not device code";
  VCHECK(cuda::classify_blob(text.data(), text.size()) == BlobKind::Unknown);
}

VTEST(a_cubins_architecture_is_read_from_either_elf_abi) {
  // e_flags as CUDA 12.0 (ABI 7) and CUDA 13.0 (ABI 8) write them.
  std::string cubin = elf_object(190, {});
  auto with = [&](uint8_t abi, uint32_t flags) {
    std::string c = cubin;
    c[8] = static_cast<char>(abi);
    std::memcpy(&c[0x30], &flags, 4);
    return cuda::cubin_arch(c.data(), c.size());
  };
  VCHECK_EQ(with(7, 0x560556), 86u);
  VCHECK_EQ(with(8, 0x6005604), 86u);
  VCHECK_EQ(with(8, 0x6005a04), 90u);
  VCHECK_EQ(with(8, 0x6007802), 120u);
  const std::string object = elf_object(62, {});
  VCHECK_EQ(cuda::cubin_arch(object.data(), object.size()), 0u);
}

VTEST(a_host_objects_fatbins_are_found_by_section) {
  const std::string a = cuda::write_fatbin({ptx_image(80, kLib)});
  const std::string b = cuda::write_fatbin({ptx_image(90, kHopper)});
  const std::string object =
      elf_object(62, {{".text", "\xc3"}, {"__nv_module_id", std::string("_abc_lib_cu\0", 12)},
                      {"__nv_relfatbin", a + b}, {".nv_fatbin", a}});
  auto all = cuda::host_object_fatbins(object.data(), object.size());
  VCHECK_EQ(all.size(), size_t{3});
  VCHECK_EQ(all[0], a);
  VCHECK_EQ(all[1], b);
  auto reloc = cuda::host_object_fatbins(object.data(), object.size(), /*relocatable_only=*/true);
  VCHECK_EQ(reloc.size(), size_t{2});
  bool id = false;
  for (const auto& [name, contents] : cuda::elf_sections(object.data(), object.size()))
    id = id || (name == "__nv_module_id" && std::string(contents.c_str()) == "_abc_lib_cu");
  VCHECK(id);
  // No device code: nothing, and no error.
  const std::string plain = elf_object(62, {{".text", "\xc3"}});
  VCHECK(cuda::host_object_fatbins(plain.data(), plain.size()).empty());
  // A section table that does not fit is refused, not read.
  std::string cut = object.substr(0, object.size() - 64);
  auto err = VCAPTURE(Error, cuda::host_object_fatbins(cut.data(), cut.size()));
  VCHECK(err.code() == Err::InvalidValue);
}

VTEST(archive_members_with_long_names) {
  auto header = [](const std::string& name, size_t size) {
    char h[61];
    std::snprintf(h, sizeof h, "%-16s%-12s%-6s%-6s%-8s%-10zu`\n", name.c_str(), "0", "0", "0", "644", size);
    return std::string(h, 60);
  };
  const std::string long_names = "a_rather_long_member_name.o/\n";
  std::string ar = "!<arch>\n";
  ar += header("/", 4) + std::string(4, '\0');
  ar += header("//", long_names.size()) + long_names;
  if (ar.size() % 2) ar.push_back('\n');
  ar += header("/0", 3) + "abc" + "\n";   // an odd size is padded to even
  ar += header("short.o/", 2) + "xy";
  auto members = cuda::archive_members(ar.data(), ar.size());
  VCHECK_EQ(members.size(), size_t{2});
  VCHECK_EQ(members[0].first, std::string("a_rather_long_member_name.o"));
  VCHECK_EQ(members[0].second, std::string("abc"));
  VCHECK_EQ(members[1].first, std::string("short.o"));
  VCHECK_EQ(members[1].second, std::string("xy"));
  std::string truncated = ar.substr(0, ar.size() - 1);
  auto err = VCAPTURE(Error, cuda::archive_members(truncated.data(), truncated.size()));
  VCHECK(err.code() == Err::InvalidValue);
}

VTEST_MAIN
