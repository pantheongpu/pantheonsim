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
  // Where its kernel descriptor is: in a linked object, an address from the
  // image's start, which placed on a device is what HSA calls the kernel
  // object -- the handle an AQL dispatch packet names the kernel by.
  uint64_t descriptor = 0;
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
  // Which of the work-group's ids the hardware puts in scalar registers
  // after the user ones (COMPUTE_PGM_RSRC2's enable bits): only those, one
  // after another, so a kernel that asks for x and z finds z where y would
  // have been.
  bool group_id_x = true, group_id_y = true, group_id_z = true;
};

// A variable the kernels share: a __device__ global. It lives in the module's
// own memory, which a loader places on the device.
struct GlobalVar {
  std::string name;
  uint64_t offset = 0;   // where it is in the module's data image
  uint64_t size = 0;
};

// One relocation in the code: where a global's address has to be written once
// the module has been placed.
struct Relocation {
  uint64_t at = 0;       // where in .text
  uint64_t symbol = 0;   // the symbol's offset in the data image, or in .text
  int64_t addend = 0;
  bool high = false;     // the top half of the address, rather than the bottom
  // A call reaches a function the same way a kernel reaches a global, but
  // both ends are in .text, so the distance between them is known as soon as
  // the module is read and the loader fills it in then.
  bool in_text = false;
};

struct CodeObject {
  std::string target;                     // "amdgcn-amd-amdhsa--gfx942", where the note gives it
  std::string isa;                        // "gfx942"
  // The code object ABI the metadata declares. It says where a work-item
  // finds its id: from version 5 (1.2) the three are packed into v0, and
  // before it each has a register of its own.
  uint32_t abi_major = 1, abi_minor = 0;
  bool packed_work_item_id() const { return abi_major > 1 || abi_minor >= 2; }
  std::vector<uint8_t> text;              // the .text section
  uint64_t text_addr = 0;                 // its address, which kernel entries are relative to
  std::vector<Kernel> kernels;
  // The module's own memory: its initialised variables, then the zeroed ones.
  // A loader places this on the device and calls place_globals.
  std::vector<uint8_t> data;
  std::vector<GlobalVar> globals;
  std::vector<Relocation> relocations;
  bool placed = false;
  uint64_t data_base = 0;   // where the data image was placed, once it was
  // A linked object (what hipcc builds): everything it loads, laid out by
  // address from zero -- its code, its constants, its variables. A loader
  // copies this to the device, and that base plus an address is where each
  // thing is; the code finds its constants and variables that way, relative
  // to itself. Empty for an object not yet linked, whose variables are `data`.
  bool linked = false;
  std::vector<uint8_t> image;
  // The processor the code was built for: the ELF header's e_flags machine
  // field (EF_AMDGPU_MACH), 0x4c for gfx942 and 0x4f for gfx950. Where the
  // same instruction means different things on the two -- an 8-bit float is
  // FNUZ on gfx942 and OCP on gfx950 -- this says which.
  uint32_t mach = 0;
  bool gfx950() const { return mach == 0x4f; }
};

// Writes the addresses of the module's globals into its code, for a data
// image placed at `base`. A kernel reaches a global by adding a constant to
// the program counter, and that constant is what this fills in: before it,
// the code has zeros there. Throws if called twice with different bases.
void place_globals(CodeObject& o, uint64_t base);
// The global of that name, or null.
const GlobalVar* find_global(const CodeObject& o, const std::string& name);

// Reads a code object. Throws Err::ProfileParse naming what is wrong: not an
// ELF, not AMDGPU, a note that is not MessagePack, a kernel whose descriptor
// is missing.
CodeObject load_code_object(const std::string& bytes, const std::string& origin);

// The kernel of that name, or null.
const Kernel* find_kernel(const CodeObject& o, const std::string& name);

}  // namespace vgpu::amd
