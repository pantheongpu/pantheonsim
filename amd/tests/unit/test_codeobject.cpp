// Reading an AMD GPU code object: the ELF a HIP program hands the driver.
//
// The object under test is amd/tests/data/vector_add.gfx942.o, built by clang
// for gfx942 (amd/tests/data/build.sh). Every value checked here is one the
// AMDGPU ABI fixes, and llvm-readelf reports the same for this file.
#include <fstream>
#include <iterator>
#include <string>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

std::string fixture() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/vector_add.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}

// What load_code_object says about bytes it will not read.
std::string refused(const std::string& bytes) {
  try {
    amd::load_code_object(bytes, "test.o");
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

VTEST(a_code_object_names_its_kernel_and_its_arguments) {
  const amd::CodeObject o = amd::load_code_object(fixture(), "vector_add.gfx942.o");
  VCHECK_EQ(o.target, std::string("amdgcn-amd-amdhsa--gfx942"));
  VCHECK_EQ(o.isa, std::string("gfx942"));
  VCHECK(o.text.size() >= 0x80);        // the kernels' instructions
  VCHECK_EQ(o.kernels.size(), 2u);

  const amd::Kernel* k = amd::find_kernel(o, "vector_add");
  VCHECK(k != nullptr);
  VCHECK_EQ(k->entry, o.text_addr);     // its code starts at the front of .text
  VCHECK(k->size >= 0x40 && k->size <= o.text.size());
  VCHECK_EQ(k->wavefront_size, 64u);    // a CDNA wave
  VCHECK_EQ(k->max_flat_workgroup_size, 1024u);
  VCHECK_EQ(k->group_segment, 0u);      // no LDS
  VCHECK_EQ(k->private_segment, 0u);    // no scratch
  VCHECK(k->sgpr_count > 0 && k->vgpr_count > 0);
  VCHECK(amd::find_kernel(o, "no_such_kernel") == nullptr);

  // (const float* a, const float* b, float* out, int n), packed as the ABI
  // lays out the kernarg segment: three pointers then the count.
  VCHECK_EQ(k->kernarg_size, 28u);
  VCHECK_EQ(k->kernarg_align, 8u);
  VCHECK_EQ(k->args.size(), 4u);
  for (size_t i = 0; i < 3; ++i) {
    VCHECK_EQ(k->args[i].offset, static_cast<uint32_t>(8 * i));
    VCHECK_EQ(k->args[i].size, 8u);
    VCHECK_EQ(k->args[i].kind, std::string("global_buffer"));
    VCHECK(!k->args[i].hidden());
  }
  VCHECK_EQ(k->args[3].offset, 24u);
  VCHECK_EQ(k->args[3].size, 4u);
  VCHECK_EQ(k->args[3].kind, std::string("by_value"));
}

VTEST(a_kernel_that_uses_lds_reserves_it) {
  const amd::CodeObject o = amd::load_code_object(fixture(), "vector_add.gfx942.o");
  const amd::Kernel* k = amd::find_kernel(o, "reduce_sum");
  VCHECK(k != nullptr);
  // (const float* in, float* out, int n, float scale), and 256 floats of LDS
  // for the reduction, which the launch must give the work-group.
  VCHECK_EQ(k->group_segment, 1024u);
  VCHECK_EQ(k->kernarg_size, 24u);
  VCHECK_EQ(k->args.size(), 4u);
  VCHECK_EQ(k->args[2].kind, std::string("by_value"));
  VCHECK_EQ(k->args[3].kind, std::string("by_value"));
  VCHECK_EQ(k->args[3].offset, 20u);
  // The second kernel's code starts after the first's, in the same .text,
  // and the two do not overlap.
  const amd::Kernel* first = amd::find_kernel(o, "vector_add");
  VCHECK(k->entry >= first->entry + first->size);
  VCHECK(k->entry + k->size <= o.text_addr + o.text.size());
}

VTEST(the_descriptor_says_what_the_hardware_loads_before_the_first_instruction) {
  const amd::CodeObject o = amd::load_code_object(fixture(), "vector_add.gfx942.o");
  const amd::Kernel& k = o.kernels[0];
  // This kernel reads its arguments, so it asks for the kernarg segment
  // pointer, and nothing else: two user SGPRs, s[0:1].
  VCHECK(k.kernarg_segment_ptr);
  VCHECK_EQ(k.user_sgpr_count, 2u);
  VCHECK(!k.dispatch_ptr);
  VCHECK(!k.queue_ptr);
  VCHECK(!k.flat_scratch_init);
  VCHECK(!k.private_segment_buffer);
}

VTEST(bytes_that_are_not_a_code_object_are_refused_by_name) {
  VCHECK_CONTAINS(refused(""), "not an ELF file");
  VCHECK_CONTAINS(refused(std::string(4096, 'x')), "not an ELF file");

  std::string elf = fixture();
  // An x86-64 ELF is an ELF, and not this.
  std::string host = elf;
  host[18] = 62;   // EM_X86_64
  VCHECK_CONTAINS(refused(host), "not an AMDGPU code object");

  // A 32-bit one, and a file that stops in the middle.
  std::string thirty_two = elf;
  thirty_two[4] = 1;
  VCHECK_CONTAINS(refused(thirty_two), "not a 64-bit little-endian ELF");
  VCHECK(!refused(elf.substr(0, elf.size() / 2)).empty());

  // The metadata note carries the kernels: without it there is nothing to run.
  std::string no_note = elf;
  const std::string marker = "AMDGPU";
  if (const size_t at = no_note.find(marker); at != std::string::npos) no_note[at] = 'X';
  VCHECK_CONTAINS(refused(no_note), "no AMDGPU metadata note");
}

VTEST_MAIN
