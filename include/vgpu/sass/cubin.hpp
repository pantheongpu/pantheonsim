// A cubin: the ELF image ptxas produces, holding one module's SASS. Parsed
// into what loading and launching a kernel needs. The format is NVIDIA's and
// unpublished; everything here was read from cubins the local toolkit built,
// checked against what cuobjdump -elf prints for the same files:
//
//   .text.<kernel>            the kernel's code (128-bit instructions)
//   .nv.info.<kernel>         attributes: parameter layout, registers, stack
//   .nv.info                  module-wide attributes (register counts, frames)
//   .nv.constant0.<kernel>    bank 0's template: the driver's launch fields
//                             (block and grid sizes, windows, stack) and the
//                             parameters, from 0x160 (0x210 from sm_90)
//   .nv.constant2.<kernel>    constants ptxas made for that kernel
//   .nv.constant3             the module's __constant__ variables
//   .nv.constantN             other module banks (4: an address table)
//   .nv.shared.<kernel>       the kernel's static shared memory (NOBITS)
//   .nv.global / .init        the module's __device__ variables
//   .rel(a).<section>         relocations: R_CUDA_64 (type 2) writes a
//                             symbol's 64-bit address
//
// .nv.info records are {u8 format, u8 attribute, ...}: formats 1-3 (NVAL,
// BVAL, HVAL) are four bytes in all, format 4 (SVAL) has a u16 size and that
// many bytes of payload.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vgpu::sass {

struct CubinParam {
  uint32_t ordinal = 0;
  uint32_t offset = 0;   // within the parameter area
  uint32_t size = 0;
};

struct CubinReloc {
  uint64_t offset = 0;          // in the section it patches
  std::string symbol;
  uint32_t type = 0;            // 2: R_CUDA_64
  int64_t addend = 0;
};

struct CubinSection {
  std::string name;
  std::vector<uint8_t> bytes;   // empty for NOBITS
  uint64_t size = 0;            // bytes.size(), or the NOBITS size
  std::vector<CubinReloc> relocs;
};

struct CubinSymbol {
  std::string name;
  std::string section;          // the section it lives in ("" when undefined)
  uint64_t value = 0;           // offset within that section
  uint64_t size = 0;
  bool function = false;
  bool global = false;
};

struct CubinKernel {
  std::string name;
  std::string text_section;     // ".text.<name>"
  uint32_t param_base = 0x160;  // where parameters start in bank 0
  uint32_t param_size = 0;
  std::vector<CubinParam> params;   // by ordinal
  uint32_t regs = 0;
  uint32_t frame_size = 0;      // bytes of local memory the kernel's frame needs
  uint32_t min_stack = 0;
  uint32_t max_threads = 0;     // __launch_bounds__, 0 when none
  uint32_t barriers = 0;
  uint64_t shared_bytes = 0;    // static shared memory
  std::vector<uint32_t> exit_offsets;
  std::array<uint32_t, 3> cluster{0, 0, 0};   // __cluster_dims__, zeros when none
  bool explicit_cluster = false;              // must be launched with a cluster
};

struct Cubin {
  int sm = 0;                   // 86, 90, 100, ...
  bool arch_specific = false;   // an sm_XYa image: runs on XY only
  std::vector<CubinSection> sections;
  std::map<std::string, size_t> section_index;
  std::vector<CubinSymbol> symbols;
  std::vector<CubinKernel> kernels;
  std::vector<std::string> externs;   // functions the module calls but does not define (vprintf, malloc)
  // .nv.callgraph: who calls whom, by symbol name (externs included).
  std::vector<std::pair<std::string, std::string>> calls;

  const CubinSection* section(const std::string& name) const {
    const auto it = section_index.find(name);
    return it == section_index.end() ? nullptr : &sections[it->second];
  }
};

// Parses a cubin. Throws vgpu::Error naming what is wrong with a malformed
// image; every offset and size read from the file is checked against it.
Cubin parse_cubin(const uint8_t* data, size_t size);

// True when `data` starts like an ELF cubin.
bool is_cubin(const void* data, size_t size);

}  // namespace vgpu::sass
