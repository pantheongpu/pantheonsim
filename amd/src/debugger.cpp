// The AMD kernel debugger (vgpu/amd_debug.hpp): breakpoints, stepping, and a
// stopped wave's registers, LDS and memory, driven by commands from a
// terminal or a file.
#include "vgpu/amd_debug.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"

namespace vgpu::amd::debug {
namespace {

struct Breakpoint {
  int id = 0;
  std::string kernel;   // a kernel whose name contains this
  uint64_t offset = 0;
  bool once = false;
  bool live = true;
};

class Debugger {
 public:
  Debugger() {
    const char* d = std::getenv("VGPU_DEBUG");
    if (!d || !*d) return;
    on_ = true;
    const std::string where = d;
    if (where == "tty" || where == "/dev/tty") {
      tty_.open("/dev/tty");
      interactive_ = tty_.good();
      if (!interactive_) std::fprintf(stderr, "vgpu debug: no terminal to read commands from\n");
    } else {
      std::ifstream f(where);
      if (!f) {
        std::fprintf(stderr, "vgpu debug: cannot read commands from %s\n", where.c_str());
      } else {
        for (std::string line; std::getline(f, line);) script_.push_back(line);
      }
    }
    // The commands before the first "continue" set up breakpoints, before
    // any kernel runs.
    commands_until_go(nullptr, nullptr);
  }

  bool on() const { return on_; }

  bool should_stop(const std::string& kernel, uint64_t offset, const void* who) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (quit_) throw Error::make(Err::Unsupported, "the debugger was told to quit");
    if (stepping_ && who == stepper_) {
      if (--steps_left_ == 0) return true;
      return false;
    }
    for (Breakpoint& b : breaks_)
      if (b.live && b.offset == offset && kernel.find(b.kernel) != std::string::npos) return true;
    return false;
  }

  void stop(const WaveView& w, const void* who) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    stepping_ = false;
    for (Breakpoint& b : breaks_)
      if (b.live && b.offset == w.offset && w.kernel.find(b.kernel) != std::string::npos) {
        say("Breakpoint " + std::to_string(b.id) + ", ");
        if (b.once) b.live = false;
        break;
      }
    where(w);
    commands_until_go(&w, who);
    if (quit_) throw Error::make(Err::Unsupported, "the debugger was told to quit");
  }

 private:
  void say(const std::string& s) {
    std::fputs(s.c_str(), stdout);
    std::fflush(stdout);
  }

  // The next command: from the terminal, or the script. None where the
  // script is done, which lets the program run to its end.
  bool next_command(std::string* line) {
    if (interactive_) {
      say("(vgpu) ");
      return static_cast<bool>(std::getline(tty_, *line));
    }
    if (at_ >= script_.size()) return false;
    *line = script_[at_++];
    say("(vgpu) " + *line + "\n");
    return true;
  }

  // Runs commands until one lets the program go on (continue, step, quit).
  // With no wave -- before any kernel -- only the ones that need none.
  void commands_until_go(const WaveView* w, const void* who) {
    std::string line;
    while (next_command(&line)) {
      std::istringstream in(line);
      std::string cmd;
      if (!(in >> cmd) || cmd[0] == '#') continue;
      std::string rest;
      std::getline(in, rest);
      rest = trim(rest);
      if (cmd == "continue" || cmd == "c" || cmd == "run" || cmd == "r") return;
      if (cmd == "quit" || cmd == "q" || cmd == "kill") {
        quit_ = true;
        return;
      }
      if (cmd == "break" || cmd == "b" || cmd == "tbreak") {
        add_break(rest, cmd == "tbreak");
      } else if (cmd == "delete" || cmd == "d") {
        for (Breakpoint& b : breaks_)
          if (rest.empty() || std::to_string(b.id) == rest) b.live = false;
      } else if (cmd == "info" && (rest == "breakpoints" || rest == "b")) {
        for (const Breakpoint& b : breaks_)
          if (b.live) say(std::to_string(b.id) + "  " + b.kernel + "+0x" + hex(b.offset) + (b.once ? " (once)" : "") + "\n");
      } else if (cmd == "lane") {
        lane_ = rest == "all" || rest.empty() ? -1 : std::stoi(rest);
      } else if (!w) {
        say("no wave is stopped: '" + cmd + "' needs one\n");
      } else if (cmd == "step" || cmd == "s" || cmd == "stepi" || cmd == "si" || cmd == "next" || cmd == "n") {
        steps_left_ = rest.empty() ? 1 : std::stoull(rest);
        stepping_ = steps_left_ > 0;
        stepper_ = who;
        return;
      } else if (cmd == "where" || cmd == "bt" || cmd == "backtrace") {
        where(*w);
      } else if (cmd == "disas" || cmd == "disassemble") {
        disas(*w, rest.empty() ? 5 : std::stoul(rest));
      } else if (cmd == "info" && (rest == "registers" || rest == "reg" || rest == "r")) {
        registers(*w);
      } else if (cmd.rfind("print", 0) == 0 || cmd.rfind("p", 0) == 0) {
        const size_t slash = cmd.find('/');
        print(*w, rest, slash == std::string::npos ? 'x' : cmd[slash + 1]);
      } else if (cmd.rfind("x", 0) == 0) {
        examine(*w, cmd, rest);
      } else if (cmd == "set") {
        set(*w, rest);
      } else {
        say("unknown command '" + cmd + "'\n");
      }
    }
    // Out of commands: run to the end, stopping nowhere else.
    for (Breakpoint& b : breaks_) b.live = false;
  }

  static std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
  }
  static std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%llx", static_cast<unsigned long long>(v));
    return b;
  }
  static uint64_t number(const std::string& s) { return std::stoull(s, nullptr, 0); }

  void add_break(const std::string& spec, bool once) {
    if (spec.empty()) {
      say("break needs a kernel: break KERNEL[+OFFSET]\n");
      return;
    }
    Breakpoint b;
    b.id = static_cast<int>(breaks_.size()) + 1;
    b.once = once;
    const size_t plus = spec.find('+');
    b.kernel = spec.substr(0, plus);
    if (plus != std::string::npos) b.offset = number(spec.substr(plus + 1));
    breaks_.push_back(b);
    say("Breakpoint " + std::to_string(b.id) + " at " + b.kernel + "+0x" + hex(b.offset) + "\n");
  }

  void where(const WaveView& w) {
    uint32_t size = 0;
    char b[256];
    std::snprintf(b, sizeof b, "%s+0x%llx, work-group (%u,%u,%u), wave %u, exec 0x%llx\n  %s\n", w.kernel.c_str(),
                  static_cast<unsigned long long>(w.offset), w.group[0], w.group[1], w.group[2], w.wave,
                  static_cast<unsigned long long>(*w.exec), w.disassemble(w.pc, &size).c_str());
    say(b);
  }

  void disas(const WaveView& w, uint32_t n) {
    uint64_t pc = w.pc;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t size = 0;
      const std::string text = w.disassemble(pc, &size);
      say((i == 0 ? "=> " : "   ") + std::string("0x") + hex(w.offset + (pc - w.pc)) + ":  " + text + "\n");
      if (!size) break;
      pc += size;
    }
  }

  void registers(const WaveView& w) {
    char b[128];
    std::snprintf(b, sizeof b, "pc     +0x%llx\nexec   0x%016llx\nvcc    0x%016llx\nscc    %d\nm0     0x%08x\nmode   0x%08x\n",
                  static_cast<unsigned long long>(w.offset), static_cast<unsigned long long>(*w.exec),
                  static_cast<unsigned long long>(*w.vcc), *w.scc ? 1 : 0, *w.m0, *w.mode);
    say(b);
    for (uint32_t i = 0; i < w.sgprs; i += 8) {
      std::string row;
      std::snprintf(b, sizeof b, "s%-4u ", i);
      row = b;
      for (uint32_t k = i; k < i + 8 && k < w.sgprs; ++k) {
        std::snprintf(b, sizeof b, " %08x", w.sgpr[k]);
        row += b;
      }
      say(row + "\n");
    }
  }

  // A register name: v5, v[4:5], s3, s[0:1], vcc, exec, m0.
  struct Reg {
    char kind = 0;   // 'v', 's', or 0 for the special ones
    uint32_t first = 0, count = 1;
    std::string special;
  };
  static bool parse_reg(const std::string& t, Reg* r) {
    if (t == "vcc" || t == "exec" || t == "m0" || t == "scc") {
      r->special = t;
      return true;
    }
    if (t.size() < 2 || (t[0] != 'v' && t[0] != 's')) return false;
    r->kind = t[0];
    if (t[1] == '[') {
      const size_t colon = t.find(':'), close = t.find(']');
      if (colon == std::string::npos || close == std::string::npos) return false;
      r->first = static_cast<uint32_t>(std::stoul(t.substr(2, colon - 2)));
      r->count = static_cast<uint32_t>(std::stoul(t.substr(colon + 1, close - colon - 1))) - r->first + 1;
      return true;
    }
    r->first = static_cast<uint32_t>(std::stoul(t.substr(1)));
    return true;
  }

  std::string value_text(uint64_t v, char format, bool wide) {
    char b[48];
    if (format == 'd') std::snprintf(b, sizeof b, "%lld", static_cast<long long>(wide ? static_cast<int64_t>(v) : static_cast<int32_t>(v)));
    else if (format == 'f') {
      if (wide) {
        double d;
        std::memcpy(&d, &v, 8);
        std::snprintf(b, sizeof b, "%.17g", d);
      } else {
        float f;
        const uint32_t u = static_cast<uint32_t>(v);
        std::memcpy(&f, &u, 4);
        std::snprintf(b, sizeof b, "%.9g", static_cast<double>(f));
      }
    } else if (wide) std::snprintf(b, sizeof b, "0x%016llx", static_cast<unsigned long long>(v));
    else std::snprintf(b, sizeof b, "0x%08x", static_cast<uint32_t>(v));
    return b;
  }

  void print(const WaveView& w, const std::string& what, char format) {
    Reg r;
    if (!parse_reg(what, &r)) {
      say("print what? (v5, v[4:5], s3, s[0:1], vcc, exec, m0, scc)\n");
      return;
    }
    if (!r.special.empty()) {
      const uint64_t v = r.special == "vcc" ? *w.vcc : r.special == "exec" ? *w.exec : r.special == "m0" ? *w.m0 : (*w.scc ? 1 : 0);
      say(what + " = " + value_text(v, format, r.special == "vcc" || r.special == "exec") + "\n");
      return;
    }
    const bool wide = r.count == 2;
    if (r.kind == 's') {
      if (r.first + r.count > w.sgprs) return say("past the scalar registers\n");
      std::string out = what + " =";
      if (wide) out += " " + value_text(w.sgpr[r.first] | static_cast<uint64_t>(w.sgpr[r.first + 1]) << 32, format, true);
      else
        for (uint32_t k = 0; k < r.count; ++k) out += " " + value_text(w.sgpr[r.first + k], format, false);
      say(out + "\n");
      return;
    }
    if (r.first + r.count > w.vgprs) return say("past the vector registers\n");
    const auto lane_value = [&](uint32_t lane) {
      if (wide) return value_text(w.vgpr[r.first][lane] | static_cast<uint64_t>(w.vgpr[r.first + 1][lane]) << 32, format, true);
      std::string s;
      for (uint32_t k = 0; k < r.count; ++k) s += (k ? " " : "") + value_text(w.vgpr[r.first + k][lane], format, false);
      return s;
    };
    if (lane_ >= 0) {
      say(what + "[" + std::to_string(lane_) + "] = " + lane_value(static_cast<uint32_t>(lane_)) + "\n");
      return;
    }
    // Eight lanes to a row (four for a pair), a lane switched off marked *.
    std::string out = what + " =\n";
    const uint32_t per_row = r.count > 1 ? 4 : 8;
    bool any_off = false;
    for (uint32_t row = 0; row < w.lanes; row += per_row) {
      char b[24];
      std::snprintf(b, sizeof b, "  [%2u-%2u]", row, row + per_row - 1);
      out += b;
      for (uint32_t lane = row; lane < row + per_row && lane < w.lanes; ++lane) {
        const bool off = !(*w.exec >> lane & 1);
        any_off = any_off || off;
        out += std::string(off ? " *" : "  ") + lane_value(lane);
      }
      out += "\n";
    }
    say(out + (any_off ? "  (* : lane switched off)\n" : ""));
  }

  // x/N ADDR: N dwords of device memory; x/N lds:ADDR, of the group's LDS.
  void examine(const WaveView& w, const std::string& cmd, const std::string& rest) {
    const size_t slash = cmd.find('/');
    const uint32_t n = slash == std::string::npos ? 1 : static_cast<uint32_t>(std::stoul(cmd.substr(slash + 1)));
    const bool lds = rest.rfind("lds:", 0) == 0;
    const uint64_t addr = number(lds ? rest.substr(4) : rest);
    std::string out;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t v = 0;
      const uint64_t at = addr + 4 * i;
      if (lds) {
        if (at + 4 > w.lds->size()) {
          out += "(past the group's " + std::to_string(w.lds->size()) + " bytes of LDS)\n";
          break;
        }
        std::memcpy(&v, w.lds->data() + at, 4);
      } else {
        try {
          w.memory->read(at, &v, 4);
        } catch (const std::exception& e) {
          out += std::string(e.what()) + "\n";
          break;
        }
      }
      if (i % 4 == 0) out += (i ? "\n" : "") + std::string(lds ? "lds:0x" : "0x") + hex(at) + ":";
      char b[16];
      std::snprintf(b, sizeof b, " %08x", v);
      out += b;
    }
    say(out + "\n");
  }

  // set v5 = 3, set v5[2] = 0x40400000, set s3 = 1.5, set exec = 0xff
  void set(const WaveView& w, const std::string& rest) {
    const size_t eq = rest.find('=');
    if (eq == std::string::npos) return say("set REG[LANE] = VALUE\n");
    std::string name = trim(rest.substr(0, eq));
    const std::string value = trim(rest.substr(eq + 1));
    int lane = -1;
    if (const size_t open = name.rfind('['); open != std::string::npos && name[0] == 'v' && name[1] != '[') {
      lane = std::stoi(name.substr(open + 1));
      name = name.substr(0, open);
    }
    uint64_t v;
    if (value.find('.') != std::string::npos) {
      const float f = std::stof(value);
      uint32_t u;
      std::memcpy(&u, &f, 4);
      v = u;
    } else {
      v = number(value);
    }
    Reg r;
    if (!parse_reg(name, &r)) return say("set what?\n");
    if (r.special == "exec") *w.exec = v;
    else if (r.special == "vcc") *w.vcc = v;
    else if (r.special == "m0") *w.m0 = static_cast<uint32_t>(v);
    else if (r.special == "scc") *w.scc = v != 0;
    else if (r.kind == 's' && r.first < w.sgprs) w.sgpr[r.first] = static_cast<uint32_t>(v);
    else if (r.kind == 'v' && r.first < w.vgprs) {
      for (uint32_t l = 0; l < w.lanes; ++l)
        if (lane < 0 || static_cast<int>(l) == lane) w.vgpr[r.first][l] = static_cast<uint32_t>(v);
    } else return say("set what?\n");
  }

  bool on_ = false, interactive_ = false, quit_ = false;
  std::ifstream tty_;
  std::vector<std::string> script_;
  size_t at_ = 0;
  std::vector<Breakpoint> breaks_;
  bool stepping_ = false;
  uint64_t steps_left_ = 0;
  const void* stepper_ = nullptr;
  int lane_ = -1;
  std::recursive_mutex mu_;
};

Debugger& debugger() {
  static Debugger d;
  return d;
}

}  // namespace

bool active() {
  static const bool on = [] {
    const char* d = std::getenv("VGPU_DEBUG");
    return d && *d;
  }();
  return on && debugger().on();
}

bool should_stop(const std::string& kernel, uint64_t offset, const void* who) {
  return debugger().should_stop(kernel, offset, who);
}

void stop(const WaveView& wave, const void* who) { debugger().stop(wave, who); }

}  // namespace vgpu::amd::debug
