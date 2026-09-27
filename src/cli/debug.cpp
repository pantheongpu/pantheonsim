// `vgpu debug` — run a program with the AMD kernel debugger on.
//
//   vgpu debug [-x FILE] [vgpu run options] -- ./program [args]
//
// With -x, the commands come from FILE, unattended; without, from the
// terminal, at a (vgpu) prompt whenever a wave stops. Everything else is
// `vgpu run`'s: the checks that the program will reach the simulator, and
// then the program itself, exec'd in place. What the debugger does is in
// amd/include/vgpu/amd_debug.hpp.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int cmd_run(const std::vector<std::string>& args);

int cmd_debug(const std::vector<std::string>& args) {
  std::vector<std::string> rest;
  std::string commands;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--") {
      rest.insert(rest.end(), args.begin() + static_cast<long>(i), args.end());
      break;
    }
    if (args[i] == "-x" || args[i] == "--commands") {
      if (i + 1 >= args.size()) {
        std::fprintf(stderr, "vgpu debug: %s needs a file of commands\n", args[i].c_str());
        return 2;
      }
      commands = args[++i];
      continue;
    }
    if (args[i] == "--help" || args[i] == "-h") {
      std::printf(
          "usage: vgpu debug [-x FILE] [vgpu run options] -- ./program [args]\n\n"
          "Runs an AMD program on the simulator with the kernel debugger on: commands\n"
          "from FILE (-x), or at a (vgpu) prompt on the terminal. break KERNEL[+OFFSET],\n"
          "continue, step [N], where, disas [N], info registers, print[/x|/d|/f] REG,\n"
          "lane N, x/N ADDR, x/N lds:ADDR, set REG[LANE] = VALUE, delete [N], quit.\n");
      return 0;
    }
    rest.push_back(args[i]);
  }
  ::setenv("VGPU_DEBUG", commands.empty() ? "/dev/tty" : commands.c_str(), 1);
  // One host thread, so a stopped wave stops its dispatch.
  ::setenv("VGPU_THREADS", "1", 1);
  return cmd_run(rest);
}
