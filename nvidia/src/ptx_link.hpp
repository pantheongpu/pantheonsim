// Linking PTX modules into one, the way the device linker links relocatable
// device code -- except that the result is PTX, since PTX is what VirtualGPU
// executes. Shared by the runtime (an -rdc build's relocatable pieces) and by
// libnvJitLink.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vgpu::cuda {

// Renames a linked-in piece's file-scope names -- module variables and
// functions not declared .visible, .extern or .weak -- so that they belong to
// that piece alone, the way an object file's local symbols do. `piece` makes
// the suffix ($vgpu<piece>) unique among the pieces being linked.
void localize_ptx(std::string& text, size_t piece);

// "sm_86" -> 86, 0; "sm_90a" -> 90, 'a'; "compute_100f" -> 100, 'f'. False
// for anything else.
bool parse_arch(const std::string& arch, uint32_t* number, char* suffix);

// A module's ".target sm_XY[a|f]" line, parsed as parse_arch parses an
// architecture. False when the module has none.
bool ptx_module_target(const std::string& ptx, uint32_t* number, char* suffix);

// Whether PTX written for `target` can be compiled for `arch`: a plain target
// for itself and every later architecture, a family target (f) within its
// major version for an architecture-specific or family build, an
// architecture-specific target (a) for exactly that architecture's own
// specific build -- which is what ptxas accepts.
bool target_runs_on(uint32_t target, char target_suffix, uint32_t arch, char arch_suffix);

struct PtxInput {
  std::string name;   // what diagnostics call it: a file name, or the caller's label
  std::string text;
};

struct PtxLinkResult {
  bool ok = false;
  std::string ptx;      // the linked module, when ok
  std::string errors;   // one line per problem, in nvJitLink's error-log form
};

// Links `inputs` in order into one module for `target` ("sm_86"; empty keeps
// the first input's .target). Each input after the first has its file-scope
// names localized. Across inputs a symbol with external linkage -- a .entry,
// or a .visible, .weak or .common function or variable -- has one definition:
// a strong one wins over a .weak or .common one, the first of several weak
// ones wins, and a second strong one is reported ("Multiple definition") and
// dropped, keeping the first, which is what NVIDIA's nvJitLink does (the link
// still succeeds there). An .extern function or variable that nothing defines
// fails the link ("Undefined reference"), except the calls ptxas itself
// provides (vprintf, malloc, free, __assertfail). The result carries the
// newest .version among the inputs, and drops their debug sections and .file
// tables, whose numbering would collide.
PtxLinkResult link_ptx(const std::vector<PtxInput>& inputs, const std::string& target);

}  // namespace vgpu::cuda
