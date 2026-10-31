// The code object reader (amd/src/codeobject.cpp) gives, byte for byte, what it gave before it was made faster: every code
// object the repository holds (the objects and hsaco files of amd/tests/data, the executables and libraries hipcc built, with
// the code objects in their offload bundles) is read, and a digest of everything the reader returns (the code, the image, the
// kernels and their arguments, the variables, the relocations) -- or of the text of the error it threw -- is compared with
// amd/tests/data/codeobject_digests.txt.
//
// The digests were recorded from the build before the change (VGPU_CODEOBJECT_UPDATE=<file> writes them). A code object
// that is not in the repository (ROCm's own, hundreds of kernels in one) is checked the same way by running the old and the
// new reader on it and comparing: VGPU_CODEOBJECT_EXTRA=<file or directory> prints a digest per object for that.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "vgpu/amd_bundle.hpp"
#include "vgpu/amd_codeobject.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;
namespace fs = std::filesystem;

namespace {

struct Digest {
  uint64_t h = 1469598103934665603ull;
  void bytes(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  }
  template <typename T>
  void num(T v) {
    const uint64_t x = static_cast<uint64_t>(v);
    bytes(&x, 8);
  }
  void str(const std::string& s) {
    num(s.size());
    bytes(s.data(), s.size());
  }
};

uint64_t digest_of(const amd::CodeObject& o) {
  Digest d;
  d.str(o.target), d.str(o.isa);
  d.num(o.abi_major), d.num(o.abi_minor);
  d.num(o.text.size()), d.bytes(o.text.data(), o.text.size()), d.num(o.text_addr);
  d.num(o.kernels.size());
  for (const amd::Kernel& k : o.kernels) {
    d.str(k.name);
    d.num(k.entry), d.num(k.size), d.num(k.descriptor), d.num(k.kernarg_size), d.num(k.kernarg_align);
    d.num(k.group_segment), d.num(k.private_segment), d.num(k.sgpr_count), d.num(k.vgpr_count), d.num(k.agpr_count);
    d.num(k.max_flat_workgroup_size);
    for (uint32_t c : k.cluster_dims) d.num(c);
    d.num(k.wavefront_size), d.num(k.uniform_work_group_size);
    d.num(k.args.size());
    for (const amd::KernelArg& a : k.args) d.num(a.offset), d.num(a.size), d.str(a.kind), d.str(a.address_space);
    d.num(k.kernarg_segment_ptr), d.num(k.dispatch_ptr), d.num(k.queue_ptr), d.num(k.dispatch_id), d.num(k.flat_scratch_init);
    d.num(k.private_segment_buffer), d.num(k.user_sgpr_count), d.num(k.kernarg_preload_length), d.num(k.kernarg_preload_offset);
    d.num(k.group_id_x), d.num(k.group_id_y), d.num(k.group_id_z), d.num(k.group_info), d.num(k.private_wave_offset);
    d.num(k.mode), d.num(k.wave32);
  }
  d.num(o.data.size()), d.bytes(o.data.data(), o.data.size());
  d.num(o.globals.size());
  for (const amd::GlobalVar& g : o.globals) d.str(g.name), d.num(g.offset), d.num(g.size);
  d.num(o.relocations.size());
  for (const amd::Relocation& r : o.relocations)
    d.num(r.at), d.num(r.symbol), d.num(r.addend), d.num(r.high), d.num(r.in_text), d.num(r.wide);
  d.num(o.linked);
  d.num(o.image.size()), d.bytes(o.image.data(), o.image.size());
  d.num(o.dyn_relocations.size());
  for (const auto& r : o.dyn_relocations) d.num(r.at), d.num(r.value), d.num(r.bytes), d.num(r.high);
  d.num(o.mach);
  return d.h;
}

// The digest of reading `bytes`, or of the text of the error.
uint64_t read_digest(const std::string& bytes, const std::string& origin) {
  try {
    return digest_of(amd::load_code_object(bytes, origin));
  } catch (const std::exception& e) {
    Digest d;
    d.str("error");
    d.str(e.what());
    return d.h;
  }
}

std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}

// Every code object in a file: the file itself where it is an ELF, and each of the offload bundles found in it.
void digests_of_file(const fs::path& p, const std::string& label, std::map<std::string, uint64_t>* out) {
  const std::string bytes = slurp(p);
  if (bytes.size() > 4 && bytes.compare(0, 4, "\x7F" "ELF") == 0) {
    // An executable carries bundles; an AMDGPU object is one itself.
    uint16_t machine = 0;
    if (bytes.size() > 20) std::memcpy(&machine, bytes.data() + 18, 2);
    if (machine == 224) {
      (*out)[label] = read_digest(bytes, label);
      return;
    }
  }
  if (bytes.size() > 24 && amd::is_bundle(reinterpret_cast<const uint8_t*>(bytes.data()))) {
    // A bundle that is the whole file, as ROCm's libraries ship theirs (compressed, or not).
    try {
      const auto bundle = amd::read_bundle(reinterpret_cast<const uint8_t*>(bytes.data()), "");
      if (bundle)
        for (const auto& [target, code] : bundle->targets)
          (*out)[label + ":" + target] = read_digest(std::string(code), label);
    } catch (const std::exception&) {
    }
    return;
  }
  static const std::string_view kMagic = "__CLANG_OFFLOAD_BUNDLE__";
  size_t from = 0;
  int n = 0;
  while ((from = bytes.find(kMagic, from)) != std::string::npos) {
    try {
      const auto bundle = amd::read_bundle(reinterpret_cast<const uint8_t*>(bytes.data() + from), "");
      if (bundle)
        for (const auto& [target, code] : bundle->targets)
          (*out)[label + "#" + std::to_string(n) + ":" + target] = read_digest(std::string(code), label);
    } catch (const std::exception&) {
      // (a bundle that is not whole: nothing to compare)
    }
    ++n;
    from += kMagic.size();
  }
}

}  // namespace

VTEST(every_code_object_in_the_repository_reads_as_it_was_recorded) {
  const fs::path root = fs::path(VGPU_SOURCE_DIR) / "amd" / "tests";
  std::map<std::string, uint64_t> got;
  for (const char* dir : {"data", "hipcc", "libraries", "hsa", "openmp", "hipsparselt", "rocblas"}) {
    if (!fs::exists(root / dir)) continue;
    std::vector<fs::path> files;
    for (const auto& e : fs::recursive_directory_iterator(root / dir))
      if (e.is_regular_file()) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const fs::path& p : files) digests_of_file(p, fs::relative(p, root).generic_string(), &got);
  }
  VCHECK(got.size() > 20);
  const std::string recorded = (root / "data" / "codeobject_digests.txt").string();
  if (const char* update = std::getenv("VGPU_CODEOBJECT_UPDATE"); update && *update) {
    std::ofstream out(update);
    for (const auto& [name, d] : got) out << name << "\t" << std::hex << d << "\n";
    std::printf("recorded %zu code objects in %s\n", got.size(), update);
    return;
  }
  std::map<std::string, uint64_t> want;
  {
    std::ifstream in(recorded);
    if (!in) throw vtest::Failure("no recorded digests at " + recorded);
    for (std::string line; std::getline(in, line);) {
      const size_t tab = line.rfind('\t');
      if (tab == std::string::npos) continue;
      want[line.substr(0, tab)] = std::stoull(line.substr(tab + 1), nullptr, 16);
    }
  }
  uint32_t differ = 0;
  for (const auto& [name, d] : want) {
    const auto it = got.find(name);
    if (it == got.end() || it->second != d) {
      if (differ++ < 20) std::fprintf(stderr, "  %s: %s\n", name.c_str(), it == got.end() ? "not read" : "differs");
    }
  }
  VCHECK_EQ(differ, 0u);
  VCHECK_EQ(got.size(), want.size());
}

VTEST(code_objects_outside_the_repository_print_their_digests_on_request) {
  const char* extra = std::getenv("VGPU_CODEOBJECT_EXTRA");
  if (!extra || !*extra) return;
  std::map<std::string, uint64_t> got;
  const fs::path p(extra);
  if (fs::is_directory(p)) {
    std::vector<fs::path> files;
    for (const auto& e : fs::recursive_directory_iterator(p))
      if (e.is_regular_file()) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const fs::path& f : files) digests_of_file(f, f.filename().string(), &got);
  } else {
    digests_of_file(p, p.filename().string(), &got);
  }
  for (const auto& [name, d] : got) std::printf("DIGEST %s %016llx\n", name.c_str(), static_cast<unsigned long long>(d));
}

VTEST_MAIN
