// Device code as ROCm's libraries carry it: offload bundles with a code
// object per target, compressed with zstd (what clang writes today) or zlib
// (what it wrote before), and a device choosing the one built for it by its
// target's features. And a code object linked from several, as Tensile links
// hundreds of kernels into one, each bringing its own metadata note.
//
// The fixtures are built by amd/tests/data/build.sh: three bundles of the
// same three objects, for gfx90a, gfx942 with XNACK on and gfx942 with it
// off, and two hand-written kernels linked into one shared object.
#include <fstream>
#include <iterator>
#include <string>

#include "vgpu/amd_bundle.hpp"
#include "vgpu/amd_codeobject.hpp"
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

VTEST(what_is_not_a_bundle_is_not_read_as_one) {
  const std::string elf = file("asm_vector.gfx942.o");
  VCHECK(!amd::is_bundle(bytes(elf)));
  VCHECK(amd::read_bundle(bytes(elf)) == nullptr);
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
