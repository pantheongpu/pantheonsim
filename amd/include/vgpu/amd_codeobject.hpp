// AMD GPU code objects: what a HIP program hands the driver to run.
//
// A code object is an ELF64 for the AMDGPU-HSA ABI. Its .text holds the
// kernels' machine code; each kernel has a 64-byte kernel descriptor in
// .rodata saying where its code starts and what it needs; and a note,
// NT_AMDGPU_METADATA, carries a MessagePack document naming the kernels and
// laying out their arguments. This reads all three, as the ROCm loader does,
// and is the front of AMD execution: the decoder takes the code from here.
//
// The layout is the public AMDGPU ABI (LLVM's AMDGPUUsage documentation);
// nothing here is taken from AMD's runtime.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu::amd {

// One of a kernel's arguments, as the metadata lays it out in the kernarg
// segment. `kind` is the ABI's value_kind: "global_buffer" and "by_value" are
// the program's own arguments; the "hidden_*" kinds are what the compiler
// adds for the runtime to fill in (the grid's shape, the heap, the queue).
struct KernelArg {
  uint32_t offset = 0;
  uint32_t size = 0;
  std::string kind;
  std::string address_space;   // "global", "generic", ... where the ABI gives one
  bool hidden() const { return kind.rfind("hidden_", 0) == 0; }
};

// What the compiler asks of the hardware before a wave starts: which user
// SGPRs it wants preloaded, and how many registers the kernel uses.
struct Kernel {
  std::string name;                       // "vector_add"
  uint64_t entry = 0;                     // where its code starts in the object's .text
  uint64_t size = 0;                      // how long its code is, where the object says
  uint32_t kernarg_size = 0;
  uint32_t kernarg_align = 8;
  uint32_t group_segment = 0;             // LDS the kernel reserves, bytes
  uint32_t private_segment = 0;           // scratch per work-item, bytes
  uint32_t sgpr_count = 0, vgpr_count = 0, agpr_count = 0;
  uint32_t max_flat_workgroup_size = 0;
  uint32_t wavefront_size = 64;
  std::vector<KernelArg> args;
  // From the kernel descriptor's kernel_code_properties: the user SGPRs the
  // hardware loads before the first instruction. The kernarg segment pointer
  // is the one a launch must place.
  bool kernarg_segment_ptr = false, dispatch_ptr = false, queue_ptr = false;
  bool dispatch_id = false, flat_scratch_init = false, private_segment_buffer = false;
  uint32_t user_sgpr_count = 0;
};

struct CodeObject {
  std::string target;                     // "amdgcn-amd-amdhsa--gfx942", where the note gives it
  std::string isa;                        // "gfx942"
  std::vector<uint8_t> text;              // the .text section
  uint64_t text_addr = 0;                 // its address, which kernel entries are relative to
  std::vector<Kernel> kernels;
};

// Reads a code object. Throws Err::ProfileParse naming what is wrong: not an
// ELF, not AMDGPU, a note that is not MessagePack, a kernel whose descriptor
// is missing.
CodeObject load_code_object(const std::string& bytes, const std::string& origin);

// The kernel of that name, or null.
const Kernel* find_kernel(const CodeObject& o, const std::string& name);

}  // namespace vgpu::amd
