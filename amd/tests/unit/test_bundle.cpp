// Device code as ROCm's libraries carry it: offload bundles with a code
// object per target, compressed with zstd (what clang writes today) or zlib
// (what it wrote before), and a device choosing the one built for it by its
// target's features. And a code object linked from several, as Tensile links
// hundreds of kernels into one, each bringing its own metadata note.
//
// The fixtures are built by amd/tests/data/build.sh: three bundles of the
// same three objects, for gfx90a, gfx942 with XNACK on and gfx942 with it
// off, and two hand-written kernels linked into one shared object.
#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_bundle.hpp"
#include "vgpu/amd_codeobject.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

std::string file(const std::string& name) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no fixture at " + path);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}

const uint8_t* bytes(const std::string& s) { return reinterpret_cast<const uint8_t*>(s.data()); }

// Each bundle carries these three, whatever it was compressed with.
void check_contents(const std::string& bundle_file) {
  const std::string raw = file(bundle_file);
  VCHECK(amd::is_bundle(bytes(raw)));
  const std::unique_ptr<amd::Bundle> b = amd::read_bundle(bytes(raw));
  VCHECK(b != nullptr);
  VCHECK_EQ(amd::target_list(*b), std::string("gfx90a, gfx942:xnack+, gfx942:xnack-"));
  VCHECK(b->targets.at("gfx90a") == file("asm_sopk.gfx942.o"));
  VCHECK(b->targets.at("gfx942:xnack+") == file("asm_scalar.gfx942.o"));
  VCHECK(b->targets.at("gfx942:xnack-") == file("asm_vector.gfx942.o"));
}

}  // namespace

VTEST(a_bundle_is_read_plain_or_compressed_with_zstd_or_zlib) {
  check_contents("bundle.bin");
  check_contents("bundle_zstd.bin");
  check_contents("bundle_zlib.bin");
}

VTEST(a_device_runs_the_code_built_for_its_target_and_features) {
  const std::string raw = file("bundle_zstd.bin");
  const std::unique_ptr<amd::Bundle> b = amd::read_bundle(bytes(raw));
  const auto chosen = [&](const std::string& device) -> std::string {
    const std::string_view* code = amd::code_for(*b, device);
    return code ? std::string(*code) : std::string("none");
  };
  // An MI300X runs with XNACK off: the code built for that, not for XNACK on.
  VCHECK(chosen("gfx942:sramecc+:xnack-") == file("asm_vector.gfx942.o"));
  VCHECK(chosen("gfx942:sramecc+:xnack+") == file("asm_scalar.gfx942.o"));
  // Code that names no feature runs either way.
  VCHECK(chosen("gfx90a:sramecc+:xnack-") == file("asm_sopk.gfx942.o"));
  // And a device the bundle has nothing for is given nothing.
  VCHECK_EQ(chosen("gfx950:sramecc+:xnack-"), std::string("none"));
}

// Read for a device, a bundle keeps only the code for that device's
// processor -- a library's carries every target, and RCCL's inflates to 3 GB
// -- while still saying what else it carries.
VTEST(a_bundle_read_for_a_device_keeps_only_that_processors_code) {
  for (const char* name : {"bundle.bin", "bundle_zstd.bin", "bundle_zlib.bin"}) {
    const std::string raw = file(name);
    const std::unique_ptr<amd::Bundle> b = amd::read_bundle(bytes(raw), "gfx942:sramecc+:xnack-");
    VCHECK(b != nullptr);
    VCHECK_EQ(b->targets.size(), size_t{2});
    VCHECK(b->targets.count("gfx90a") == 0);
    VCHECK(*amd::code_for(*b, "gfx942:sramecc+:xnack-") == file("asm_vector.gfx942.o"));
    VCHECK(*amd::code_for(*b, "gfx942:sramecc+:xnack+") == file("asm_scalar.gfx942.o"));
    VCHECK_EQ(amd::target_list(*b), std::string("gfx90a, gfx942:xnack+, gfx942:xnack-"));
    if (std::string(name) != "bundle.bin")   // compressed: only what is kept is held
      VCHECK_EQ(b->inflated.size(), file("asm_vector.gfx942.o").size() + file("asm_scalar.gfx942.o").size());
    const std::unique_ptr<amd::Bundle> other = amd::read_bundle(bytes(raw), "gfx950:sramecc+:xnack-");
    VCHECK(other->targets.empty());
    VCHECK(amd::code_for(*other, "gfx950:sramecc+:xnack-") == nullptr);
  }
}

// Code built for a generic processor runs on each member of its family, as
// LLVM defines them: gfx9-4-generic on gfx940, gfx941, gfx942 and gfx950. A
// processor's own code comes first, where its features fit.
VTEST(a_device_runs_its_familys_generic_code_when_the_bundle_has_none_of_its_own) {
  const std::string raw = file("bundle_generic.bin");
  const std::unique_ptr<amd::Bundle> b = amd::read_bundle(bytes(raw));
  VCHECK_EQ(amd::target_list(*b), std::string("gfx9-4-generic, gfx942:xnack-"));
  const auto chosen = [&](const std::unique_ptr<amd::Bundle>& from, const std::string& device) -> std::string {
    const std::string_view* code = amd::code_for(*from, device);
    return code ? std::string(*code) : std::string("none");
  };
  VCHECK(chosen(b, "gfx942:sramecc+:xnack-") == file("asm_vector.gfx942.o"));   // its own
  VCHECK(chosen(b, "gfx942:sramecc+:xnack+") == file("asm_sopk.gfx942.o"));     // its own does not fit
  VCHECK(chosen(b, "gfx950:sramecc+:xnack-") == file("asm_sopk.gfx942.o"));     // the family's
  VCHECK_EQ(chosen(b, "gfx90a:sramecc+:xnack-"), std::string("none"));          // in no family here
  VCHECK_EQ(chosen(b, "gfx1100"), std::string("none"));
  // Read for a device, the family's code is kept too.
  const std::unique_ptr<amd::Bundle> for950 = amd::read_bundle(bytes(raw), "gfx950:sramecc+:xnack-");
  VCHECK_EQ(for950->targets.size(), size_t{1});
  VCHECK(chosen(for950, "gfx950:sramecc+:xnack-") == file("asm_sopk.gfx942.o"));
}

// A compressed bundle cut short, or one whose header claims more than it
// inflates to, is refused rather than read as far as it goes.
VTEST(a_damaged_compressed_bundle_is_refused) {
  std::string raw = file("bundle_zstd.bin");
  std::string cut = raw.substr(0, raw.size() / 2);
  // Keep the header's own sizes: the compressed stream just stops.
  bool refused = false;
  try {
    std::string padded = cut + std::string(raw.size() - cut.size(), '\0');
    amd::read_bundle(bytes(padded), "gfx942:sramecc+:xnack-");
  } catch (const std::exception&) {
    refused = true;
  }
  VCHECK(refused);
}

// Only zlib (0) and zstd (1) are methods clang's offload bundler defines, and only the header
// versions 2 and 3 are read: anything else is refused by name rather than read as the nearest.
VTEST(a_compressed_bundle_of_an_unknown_method_or_version_is_refused_by_name) {
  auto refusal = [&](const std::string& fixture, size_t at, uint16_t value) {
    std::string raw = file(fixture);
    raw.replace(at, 2, std::string(reinterpret_cast<const char*>(&value), 2));
    try {
      amd::read_bundle(bytes(raw), "gfx942:sramecc+:xnack-");
    } catch (const Error& e) {
      return e.code() == Err::Unsupported ? std::string(e.what()) : std::string("wrong kind of error");
    }
    return std::string("not refused");
  };
  VCHECK(refusal("bundle_zstd.bin", 6, 2).find("method (2)") != std::string::npos);
  VCHECK(refusal("bundle_zlib.bin", 6, 7).find("method (7)") != std::string::npos);
  VCHECK(refusal("bundle_zstd.bin", 4, 4).find("format version 4") != std::string::npos);
  VCHECK(refusal("bundle_zlib.bin", 4, 1).find("format version 1") != std::string::npos);
}

VTEST(what_is_not_a_bundle_is_not_read_as_one) {
  const std::string elf = file("asm_vector.gfx942.o");
  VCHECK(!amd::is_bundle(bytes(elf)));
  VCHECK(amd::read_bundle(bytes(elf)) == nullptr);
}

// ---- A bundle larger than the inflating buffer ------------------------------------
//
// A library's bundle inflates to tens of megabytes, far past the few megabytes the
// table of contents is read from, and the code kept for a device is inflated straight
// into its place (pantheonsim #565). The bundles here are made at run time, compressed
// with the system's zstd, with entries before, after and across the first 4 MB, and
// what is kept must be byte for byte what was put in -- for one entry, several, all of
// them, none, and for two entries that share bytes (which take the copying path) -- and the same with
// VGPU_LOAD_PLAIN=1, the reading from before.
namespace {

struct Part {
  std::string triple;
  uint64_t offset = 0, size = 0;
};

// Bytes that compress somewhat (runs and repeats) but not to nothing.
std::string pattern(uint64_t n, uint32_t seed) {
  std::string s(n, '\0');
  uint32_t x = seed * 2654435761u + 1;
  for (uint64_t i = 0; i < n; ++i) {
    x ^= x << 13, x ^= x >> 17, x ^= x << 5;
    s[i] = static_cast<char>((i % 64) < 40 ? (x & 0xFF) : (i >> 7) & 0xFF);
  }
  return s;
}

// An uncompressed bundle of `parts` (whose offsets are set here unless `fixed`), and the payload each one names.
std::string make_bundle(std::vector<Part>& parts, const std::vector<std::string>& payloads, bool fixed) {
  std::string toc = "__CLANG_OFFLOAD_BUNDLE__";
  const auto u64 = [](std::string& s, uint64_t v) { s.append(reinterpret_cast<const char*>(&v), 8); };
  uint64_t header = 32;
  for (const Part& p : parts) header += 24 + p.triple.size();
  uint64_t at = (header + 4095) & ~uint64_t{4095};
  for (size_t i = 0; i < parts.size() && !fixed; ++i) {
    parts[i].offset = at;
    parts[i].size = payloads[i].size();
    at = (at + parts[i].size + 4095) & ~uint64_t{4095};
  }
  u64(toc, parts.size());
  for (const Part& p : parts) u64(toc, p.offset), u64(toc, p.size), u64(toc, p.triple.size()), toc += p.triple;
  uint64_t end = toc.size();
  for (const Part& p : parts) end = std::max(end, p.offset + p.size);
  std::string b(end, '\0');
  std::memcpy(b.data(), toc.data(), toc.size());
  for (size_t i = 0; i < parts.size(); ++i)
    if (!payloads[i].empty()) std::memcpy(b.data() + parts[i].offset, payloads[i].data(), payloads[i].size());
  return b;
}

// `plain` compressed as clang's offload bundler writes it: "CCOB", version 2 or 3, method 1 (zstd).
std::string ccob(const std::string& plain, uint16_t version) {
  using Bound = size_t (*)(size_t);
  using Compress = size_t (*)(void*, size_t, const void*, size_t, int);
  void* lib = dlopen("libzstd.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) throw vtest::Failure("no libzstd.so.1 to compress with");
  const auto bound = reinterpret_cast<Bound>(dlsym(lib, "ZSTD_compressBound"));
  const auto compress = reinterpret_cast<Compress>(dlsym(lib, "ZSTD_compress"));
  std::string z(bound(plain.size()), '\0');
  z.resize(compress(z.data(), z.size(), plain.data(), plain.size(), 1));
  std::string out = "CCOB";
  const uint16_t method = 1;
  out.append(reinterpret_cast<const char*>(&version), 2);
  out.append(reinterpret_cast<const char*>(&method), 2);
  const uint64_t header = version == 2 ? 24 : 32, total = header + z.size(), size = plain.size(), hash = 0;
  if (version == 2) {
    const uint32_t t = static_cast<uint32_t>(total), n = static_cast<uint32_t>(size);
    out.append(reinterpret_cast<const char*>(&t), 4);
    out.append(reinterpret_cast<const char*>(&n), 4);
  } else {
    out.append(reinterpret_cast<const char*>(&total), 8);
    out.append(reinterpret_cast<const char*>(&size), 8);
  }
  out.append(reinterpret_cast<const char*>(&hash), 8);
  return out + z;
}

}  // namespace

VTEST(a_large_compressed_bundle_keeps_exactly_the_bytes_of_the_entries_a_device_runs) {
  const std::string kHip = "hipv4-amdgcn-amd-amdhsa--";
  std::vector<std::string> payloads = {pattern(3u << 20, 1), pattern((5u << 20) + 123, 2), pattern((2u << 20) + 7, 3),
                                       pattern(4u << 20, 4), std::string()};
  std::vector<Part> parts = {{kHip + "gfx90a"}, {kHip + "gfx942:xnack+"}, {kHip + "gfx1100"}, {kHip + "gfx942:xnack-"},
                             {"host-x86_64-unknown-linux-gnu-"}};
  const std::string plain = make_bundle(parts, payloads, false);
  VCHECK(plain.size() > (8u << 20));   // entries before, across and past the first 4 MB
  for (const char* old_reading : {"0", "1"}) {
    ::setenv("VGPU_LOAD_PLAIN", old_reading, 1);
    for (uint16_t version : {uint16_t{2}, uint16_t{3}}) {
      const std::string packed = ccob(plain, version);
      struct Case {
        const char* device;
        std::vector<size_t> kept;
      };
      for (const Case& c : {Case{"gfx942:sramecc+:xnack-", {1, 3}}, Case{"gfx90a:sramecc+:xnack-", {0}},
                            Case{"gfx1100", {2}}, Case{"", {0, 1, 2, 3}}, Case{"gfx950:sramecc+:xnack-", {}}}) {
        const std::unique_ptr<amd::Bundle> b = amd::read_bundle(bytes(packed), c.device);
        VCHECK(b != nullptr);
        VCHECK_EQ(b->targets.size(), c.kept.size());
        uint64_t held = 0;
        for (size_t i : c.kept) {
          const std::string target = parts[i].triple.substr(kHip.size());
          VCHECK(b->targets.count(target) == 1);
          VCHECK(b->targets.at(target) == payloads[i]);
          held += payloads[i].size();
        }
        VCHECK_EQ(b->inflated.size(), held);
        VCHECK_EQ(amd::target_list(*b), std::string("gfx1100, gfx90a, gfx942:xnack+, gfx942:xnack-"));
      }
    }
    // Two entries that share bytes: the second gfx942 entry starts inside the first.
    std::vector<Part> shared = parts;
    shared[3].offset = shared[1].offset + (1u << 20);
    shared[3].size = 3u << 20;
    std::vector<std::string> shared_payloads = payloads;
    shared_payloads[3].clear();   // (its bytes are the first entry's)
    const std::string overlapping = make_bundle(shared, shared_payloads, true);
    const std::unique_ptr<amd::Bundle> b = amd::read_bundle(bytes(ccob(overlapping, 3)), "gfx942:sramecc+:xnack-");
    VCHECK_EQ(b->targets.size(), size_t{2});
    VCHECK(b->targets.at("gfx942:xnack+") == payloads[1]);
    VCHECK(b->targets.at("gfx942:xnack-") == payloads[1].substr(1u << 20, 3u << 20));
  }
  ::unsetenv("VGPU_LOAD_PLAIN");
}

VTEST(a_linked_object_has_every_kernel_each_note_names) {
  const std::string so = file("linked.gfx942.hsaco");
  const amd::CodeObject o = amd::load_code_object(so, "linked.gfx942.hsaco");
  VCHECK_EQ(o.kernels.size(), size_t{3});   // two from one note, one from the other
  VCHECK(amd::find_kernel(o, "scalar") != nullptr);
  VCHECK(amd::find_kernel(o, "group_z") != nullptr);
  VCHECK(amd::find_kernel(o, "vector") != nullptr);
}

VTEST_MAIN
