// Loading a cubin onto a simulated device and running its kernels' SASS.
//
// A loaded module is the cubin's code, each section decoded the first time
// it runs, plus the device memory
// the driver would set up for it: every constant bank but bank 0 (bank 0 is
// written per launch: the launch fields and the parameters), the module's
// __device__ variables, and relocations resolved against those addresses.
// Functions the module calls but does not define -- vprintf, malloc, free,
// __assertfail -- get addresses the executor recognises and runs itself.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>
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
// Constant bank 0 of the running kernel, read-only: what a pointer to a
// __grid_constant__ parameter points into (bank 0 holds the window's
// address for the kernel to add a parameter's offset to).
inline constexpr uint64_t kParamWindow = 0x6ffd'0000'0000ull;
// Where the user's __constant__ data (constant bank 3) shows in the global address space. ptxas forms a
// pointer to a __constant__ variable as bank 0's word at 0x50 plus the variable's offset
// (`IMAD.WIDE R2, R3, R2, c[0x0][0x50]`, `ULDC.64 UR4, c[0x0][0x50]`) and reads it with an ordinary
// global load. It is the lowest of the windows, so it is also the bound below which an address is global.
inline constexpr uint64_t kConstWindow = 0x6ff9'0000'0000ull;
inline constexpr uint64_t kBuiltinBase = 0x6ffa'0000'0000ull;

// One .text section: a kernel's code, or a device function's.
// Decoded the first time one of its instructions is fetched: a PyTorch cubin
// holds hundreds of kernels, of which a program launches a few, and decoded
// instructions take many times the bytes they are decoded from.
struct Code {
  std::string section;
  uint64_t base = 0;             // its address in the code window
  size_t count = 0;              // instructions; instruction i is at byte offset 16 * i
  int sm = 0;
  const std::vector<uint8_t>* bytes = nullptr;   // the section's, in the module's cubin
  const Instr& instr(size_t i) const;
 private:
  struct Decoded {
    std::once_flag once;
    std::atomic<bool> ready{false};   // set once decoding is done: the fast check before call_once
    std::vector<Instr> instrs;
  };
  std::unique_ptr<Decoded> decoded_ = std::make_unique<Decoded>();
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
  std::vector<std::string> managed;           // the __managed__ ones among them
  std::map<std::string, uint64_t> section_va; // banks and variables, by section name
  std::vector<uint64_t> allocations;          // freed on unload
  std::map<uint64_t, std::string> builtins;   // magic address -> vprintf, malloc, ...
  bool device_launches = false;               // it can launch kernels (dynamic parallelism)

  const Code* code_at(uint64_t addr) const;   // the section holding a code address
};

// Loads a cubin (checked against the device: it must be one the profile's
// architecture runs). Throws vgpu::Error on a malformed image or one this
// executor cannot run.
std::shared_ptr<Module> load(const uint8_t* image, size_t size, MemoryManager& mem, const DeviceProfile& profile);
void unload(Module& m, MemoryManager& mem);
// Moves variable `name` to `va` (a __managed__ one onto managed memory): every
// relocation against it, in a constant bank, a variable's initialiser or the
// code, is applied again with the new address. Done before the module's first
// launch, as its code is decoded when it first runs.
void rebind(Module& m, MemoryManager& mem, const std::string& name, uint64_t va);

// Whether a cubin built for `cubin_sm` (arch_specific: an sm_XYa image) runs
// on a device of compute capability `device_sm`, by the driver's rule: the
// same major and a minor at least as new, or exactly the architecture for an
// 'a' image.
bool runs_on(int cubin_sm, bool arch_specific, int device_sm);

// Every function a kernel can reach through calls, from the call graph --
// externs such as vprintf and malloc included.
std::vector<std::string> reachable(const Module& m, const std::string& kernel);

// The first instruction in a cubin's code that this executor cannot run
// (one it does not decode, or an op or form it does not implement yet),
// described; empty when it runs all of it. Decodes only: no device needed.
std::string unsupported(const uint8_t* image, size_t size);

// Runs one kernel of a loaded module over the grid.
exec::LaunchStats launch(const Module& m, const std::string& kernel, const exec::LaunchConfig& cfg,
                         const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                         const DeviceProfile& profile);

}  // namespace vgpu::sass
