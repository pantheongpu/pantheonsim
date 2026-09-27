// A debugger for AMD kernels on the simulator: breakpoints, single-stepping,
// and a wave's registers, LDS and memory, gdb's way.
//
// It is switched on by VGPU_DEBUG: /dev/tty (or "tty") to type commands at a
// prompt, or the path of a file of commands to run unattended -- which is how
// a test drives it. `vgpu debug [-x file] -- ./program` sets it up. While it
// is on, work-groups run on one host thread, so a wave stopped at a
// breakpoint stops its whole dispatch and the rest runs in a repeatable order.
//
// The commands (each also by its first letter where gdb has one):
//   break KERNEL[+OFFSET]   stop every wave of a kernel reaching OFFSET (bytes
//                           from its first instruction; the start by default)
//   tbreak ...              the same, once
//   delete [N]              drop breakpoint N, or all of them
//   info breakpoints        list them
//   continue                run until the next breakpoint
//   step [N]                run N instructions of this wave (1 by default)
//   where                   the kernel, offset, work-group, wave and EXEC
//   disas [N]               N instructions from here (5 by default)
//   info registers          PC, EXEC, VCC, SCC, M0, MODE and the SGPRs
//   print[/x|/d|/f] REG     a register, every lane: v5, v[4:5], s3, s[0:1],
//                           vcc, exec, m0 (/x hex, the default; /d signed; /f float)
//   lane N                  print only lane N (lane all: every lane again)
//   x/N ADDR | x/N lds:ADDR N dwords of device memory, or of the group's LDS
//   set REG[LANE] = VALUE   change a register (VALUE hex, decimal or a float)
//   quit                    stop the program
// The executor runs the debugger only where this says it is on, and costs
// nothing otherwise.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vgpu {
class MemoryManager;
}

namespace vgpu::amd::debug {

// What a stopped wave lets the debugger see and change. Pointers into the
// executor's own state, valid while the debugger has the wave.
struct WaveView {
  std::string kernel;
  uint64_t pc = 0;           // the instruction about to run
  uint64_t offset = 0;       // pc less the kernel's first instruction
  uint32_t group[3] = {};    // the work-group's id
  uint32_t wave = 0;         // which wave of the group
  uint32_t lanes = 64;       // 32 for a wave32 wave
  uint64_t* exec = nullptr;
  uint64_t* vcc = nullptr;
  bool* scc = nullptr;
  uint32_t* m0 = nullptr;
  uint32_t* mode = nullptr;
  uint32_t* sgpr = nullptr;
  uint32_t sgprs = 0;
  uint32_t (*vgpr)[64] = nullptr;
  uint32_t vgprs = 0;
  std::vector<uint8_t>* lds = nullptr;
  MemoryManager* memory = nullptr;
  // The instruction at an address, as the assembler writes it, and its size
  // in bytes (0 where there is none).
  std::function<std::string(uint64_t pc, uint32_t* size)> disassemble;
};

// Whether VGPU_DEBUG asked for a debugger.
bool active();

// Called before each instruction a wave runs, while active(): whether the
// wave (`who`, the executor's own identity for it) stops here -- at a
// breakpoint, or at the end of a step. Cheap: the executor builds a view
// only where this says so.
bool should_stop(const std::string& kernel, uint64_t offset, const void* who);

// The wave has stopped: take commands until one lets it go on. Throws if told
// to quit.
void stop(const WaveView& wave, const void* who);

}  // namespace vgpu::amd::debug
