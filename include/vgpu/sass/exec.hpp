// Loading a cubin onto a simulated device and running its kernels' SASS.
//
// A loaded module is the cubin's code, decoded once, plus the device memory
// the driver would set up for it: every constant bank but bank 0 (bank 0 is
// written per launch: the launch fields and the parameters), the module's
// __device__ variables, and relocations resolved against those addresses.
// Functions the module calls but does not define -- vprintf, malloc, free,
// __assertfail -- get addresses the executor recognises and runs itself.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/sass/cubin.hpp"
#include "vgpu/sass/sass.hpp"

namespace vgpu::sass {

// Generic-address windows. The same values the PTX interpreter uses, so a
// pointer means the same thing to either engine.
inline constexpr uint64_t kSharedWindow = 0x6ffe'0000'0000ull;
inline constexpr uint64_t kLocalWindow = 0x6fff'0000'0000ull;
// Where the module's code sits for function pointers and CALL.ABS, and the
// addresses of the functions the executor provides.
inline constexpr uint64_t kCodeBase = 0x6ffb'0000'0000ull;
inline constexpr uint64_t kBuiltinBase = 0x6ffa'0000'0000ull;

// One .text section: a kernel's code, or a device function's.
struct Code {
  std::string section;
  uint64_t base = 0;             // its address in the code window
  std::vector<Instr> instrs;     // instruction i is at byte offset 16 * i
};

struct Module {
  Cubin cubin;
  int sm = 0;
  std::vector<Code> code;                     // one per .text section
  std::map<std::string, size_t> code_index;   // section name -> code
  std::map<std::string, size_t> kernel_index; // kernel name -> cubin.kernels
  // Constant banks backed by device memory: module banks (".nv.constantN")
  // and per-kernel ones (".nv.constantN.<kernel>"), by section name.
  std::map<std::string, uint64_t> bank_va;
  std::map<std::string, uint64_t> symbol_va;  // __device__ and __constant__ variables
  std::map<std::string, uint64_t> symbol_size;
  std::vector<uint64_t> allocations;          // freed on unload
  std::map<uint64_t, std::string> builtins;   // magic address -> vprintf, malloc, ...

  const Code* code_at(uint64_t addr) const;   // the section holding a code address
};

// Loads a cubin (checked against the device: it must be one the profile's
// architecture runs). Throws vgpu::Error on a malformed image or one this
// executor cannot run.
std::shared_ptr<Module> load(const uint8_t* image, size_t size, MemoryManager& mem, const DeviceProfile& profile);
void unload(Module& m, MemoryManager& mem);

// Whether a cubin built for `cubin_sm` (arch_specific: an sm_XYa image) runs
// on a device of compute capability `device_sm`, by the driver's rule: the
// same major and a minor at least as new, or exactly the architecture for an
// 'a' image.
bool runs_on(int cubin_sm, bool arch_specific, int device_sm);

// Every function a kernel can reach through calls, from the call graph --
// externs such as vprintf and malloc included.
std::vector<std::string> reachable(const Module& m, const std::string& kernel);

// Runs one kernel of a loaded module over the grid.
exec::LaunchStats launch(const Module& m, const std::string& kernel, const exec::LaunchConfig& cfg,
                         const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                         const DeviceProfile& profile);

}  // namespace vgpu::sass
