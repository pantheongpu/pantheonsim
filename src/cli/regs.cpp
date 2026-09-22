// `vgpu regs`: a device's registers, as bring-up and debug work reaches them --
// listed from the register database, read and decoded field by field, written
// with each register's semantics, dumped whole, and the log of who touched them.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.hpp"
#include "machine.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/regs.hpp"
#include "vgpu/telemetry.hpp"

namespace {

int usage(FILE* to) {
  std::fprintf(to,
               "usage: vgpu regs list [--space S] [--status done|model]\n"
               "       vgpu regs read [--space S] [--gpu N] REGISTER [--size 1|2|4]\n"
               "       vgpu regs write [--space S] [--gpu N] REGISTER VALUE [--size 1|2|4]\n"
               "       vgpu regs dump [--space S] [--gpu N] [--extended]\n"
               "       vgpu regs log [--space S] [--gpu N] [--last K]\n"
               "       vgpu regs export [PROFILE...] [--out DIR]\n"
               "\n"
               "A device's registers, from the register database: each register's offset,\n"
               "width, access and what backs it. --space is config (default), the PCI\n"
               "configuration space (registers/pci-config.yaml), or mmio, an AMD GPU's\n"
               "registers behind BAR5 (registers/amd-mmio.yaml): engine status and the SMU\n"
               "mailbox. REGISTER is a name from `list` or an offset (0x08a); a read decodes\n"
               "its fields. A write follows the register's access: rw bits are kept, a 1\n"
               "written to an rw1c status bit clears it until its cause recurs, and a base\n"
               "address register reads back its size after all ones are written. Writes are\n"
               "shared by every process on the machine, and every access is logged with the\n"
               "process that made it. --gpu defaults to 0.\n"
               "\n"
               "Every GPU model's registers and their power-on values are kept in the\n"
               "repository, registers/gpus/<vendor>/<model>.yaml, and every simulated GPU of\n"
               "the model starts from them. `export` writes those files from the database and\n"
               "the profiles: to stdout, or with --out as DIR/<vendor>/<model>.yaml. With no\n"
               "PROFILE, every built-in one.\n");
  return to == stdout ? 0 : 2;
}

bool parse_u32(const std::string& s, uint32_t* out) {
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s.c_str(), &end, 0);
  if (s.empty() || s[0] == '-' || errno || !end || *end || v > 0xFFFFFFFFull) return false;
  *out = static_cast<uint32_t>(v);
  return true;
}

// "9:4 negotiated_link_width" or "12 slot_clock": the bits a field covers.
bool field_bits(const std::string& f, uint32_t* hi, uint32_t* lo, std::string* name) {
  const size_t sp = f.find(' ');
  if (sp == std::string::npos) return false;
  const std::string range = f.substr(0, sp);
  *name = f.substr(sp + 1);
  const size_t colon = range.find(':');
  char* end = nullptr;
  *hi = static_cast<uint32_t>(std::strtoul(range.c_str(), &end, 10));
  if (end == range.c_str()) return false;
  *lo = colon == std::string::npos ? *hi : static_cast<uint32_t>(std::strtoul(range.c_str() + colon + 1, nullptr, 10));
  return *hi < 32 && *lo <= *hi;
}

// `at` is where the register is on this device, which for a capability's
// register depends on the device's capability chain.
void print_register(const vgpu::regs::Register& r, uint32_t at, uint32_t v) {
  const int digits = static_cast<int>(r.width / 4);
  std::printf("0x%03x  %-28s = 0x%0*x   (%s, %s)\n", at, r.name.c_str(), digits, v,
              vgpu::regs::access_name(r.access), r.status.c_str());
  if (!r.source.empty()) std::printf("       from %s\n", r.source.c_str());
  if (!r.measured.empty()) std::printf("       measured: %s\n", r.measured.c_str());
  for (const std::string& f : r.fields) {
    uint32_t hi = 0, lo = 0;
    std::string name;
    if (!field_bits(f, &hi, &lo, &name)) continue;
    const uint32_t width = hi - lo + 1;
    const uint32_t fv = (v >> lo) & (width >= 32 ? 0xFFFFFFFFu : ((1u << width) - 1));
    const std::string bits = hi == lo ? std::to_string(hi) : std::to_string(hi) + ":" + std::to_string(lo);
    std::printf("       %6s %-40s %u\n", bits.c_str(), name.c_str(), fv);
  }
}

// `vgpu regs export`: each profile's registers file, from its first device at
// power-on.
int export_files(const std::vector<std::string>& args) {
  std::string out_dir;
  std::vector<std::string> profiles;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--out") {
      if (i + 1 >= args.size()) {
        std::fprintf(stderr, "vgpu regs export: --out needs a directory\n");
        return 2;
      }
      out_dir = args[++i];
    } else if (args[i].rfind("--", 0) == 0) {
      std::fprintf(stderr, "vgpu regs export: unknown option '%s'\n", args[i].c_str());
      return 2;
    } else {
      profiles.push_back(args[i]);
    }
  }
  if (profiles.empty()) profiles = vgpu::available_gpus();
  try {
    for (const std::string& p : profiles) {
      const vgpu::DeviceProfile prof = vgpu::load_gpu(p);
      vgpu::telemetry::DeviceSample d{};
      vgpu::telemetry::describe_device(prof, 0, &d);
      const std::string text = vgpu::regs::export_registers(prof.id, d);
      if (out_dir.empty()) {
        std::fputs(text.c_str(), stdout);
        continue;
      }
      const std::filesystem::path path = std::filesystem::path(out_dir) / (prof.id + ".yaml");
      std::filesystem::create_directories(path.parent_path());
      std::ofstream f(path, std::ios::trunc);
      f << text;
      if (!f) {
        std::fprintf(stderr, "vgpu regs export: could not write %s\n", path.c_str());
        return 1;
      }
      std::printf("wrote %s\n", path.c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vgpu regs export: %s\n", e.what());
    return 2;
  }
  return 0;
}

}  // namespace

int cmd_regs(const std::vector<std::string>& args) {
  if (args.empty()) return usage(stderr);
  const std::string& verb = args[0];
  if (verb == "-h" || verb == "--help" || verb == "help") return usage(stdout);
  if (verb != "list" && verb != "read" && verb != "write" && verb != "dump" && verb != "log" && verb != "export") {
    std::fprintf(stderr, "vgpu regs: unknown action '%s' (list, read, write, dump, log or export)\n", verb.c_str());
    return 2;
  }
  if (verb == "export") return export_files(std::vector<std::string>(args.begin() + 1, args.end()));
  long long gpu = 0, size = 0, last = 20;
  bool extended = false;
  std::string status;
  vgpu::regs::Space space = vgpu::regs::Space::Config;
  std::vector<std::string> pos;
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    const bool takes = a == "--gpu" || a == "--size" || a == "--last" || a == "--status" || a == "--space";
    if (takes && i + 1 >= args.size()) {
      std::fprintf(stderr, "vgpu regs: %s needs a value\n", a.c_str());
      return 2;
    }
    if (a == "--gpu") {
      if (!vgpu::cli::parse_int(args[++i], 0, 63, &gpu)) {
        std::fprintf(stderr, "vgpu regs: --gpu needs an index, got '%s'\n", args[i].c_str());
        return 2;
      }
    } else if (a == "--size") {
      if (!vgpu::cli::parse_int(args[++i], 1, 4, &size) || size == 3) {
        std::fprintf(stderr, "vgpu regs: --size is 1, 2 or 4, got '%s'\n", args[i].c_str());
        return 2;
      }
    } else if (a == "--last") {
      if (!vgpu::cli::parse_int(args[++i], 1, vgpu::regs::kLogEntries, &last)) {
        std::fprintf(stderr, "vgpu regs: --last is 1 to %u\n", vgpu::regs::kLogEntries);
        return 2;
      }
    } else if (a == "--status") {
      status = args[++i];
    } else if (a == "--space") {
      if (!vgpu::regs::parse_space(args[++i], &space)) {
        std::fprintf(stderr, "vgpu regs: --space is config or mmio, got '%s'\n", args[i].c_str());
        return 2;
      }
    } else if (a == "--extended") {
      extended = true;
    } else if (a == "-h" || a == "--help") {
      return usage(stdout);
    } else if (a.rfind("--", 0) == 0) {
      std::fprintf(stderr, "vgpu regs: unknown option '%s'\n", a.c_str());
      return 2;
    } else {
      pos.push_back(a);
    }
  }
  const size_t want_pos = verb == "read" ? 1 : verb == "write" ? 2 : 0;
  if (pos.size() != want_pos) {
    std::fprintf(stderr, "vgpu regs %s: takes %zu argument%s\n", verb.c_str(), want_pos,
                 want_pos == 1 ? "" : "s");
    return 2;
  }

  try {
    // "mmio" is the chosen GPU's own MMIO space, whichever vendor's.
    vgpu::telemetry::Shared snap{};
    const bool have_machine = vgpu::cli::read_machine(&snap);
    if (have_machine && gpu < snap.device_count) space = vgpu::regs::resolve_space(snap.devices[gpu], space);
    if (verb == "list") {
      // A capability's registers sit where each card's capability chain puts
      // the capability; OFFSET is the generic layout's.
      std::printf("%-7s %-5s %-6s %-5s %-28s %-22s %s\n", "OFFSET", "WIDTH", "ACCESS", "CAP", "REGISTER", "BACKING",
                  "STATUS");
      for (const auto& r : vgpu::regs::registers(space)) {
        if (!status.empty() && r.status != status) continue;
        std::printf("0x%05x %-5u %-6s %-5s %-28s %-22s %s\n", r.offset, r.width, vgpu::regs::access_name(r.access),
                    r.capability.empty() ? "-" : r.capability.c_str(), r.name.c_str(), r.backing.empty() ? "reset value" : r.backing.c_str(), r.status.c_str());
      }
      return 0;
    }

    if (!have_machine) return 1;
    if (gpu >= snap.device_count) {
      std::fprintf(stderr, "vgpu regs: there is no GPU %lld (this machine has %u)\n", gpu, snap.device_count);
      return 2;
    }
    const auto& d = snap.devices[gpu];
    if (!vgpu::regs::has_space(d, space)) {
      std::fprintf(stderr,
                   "vgpu regs: GPU %lld (%s) has no %s registers modelled: an AMD GPU's are, and an NVIDIA "
                   "GPU's where a card of its model has been measured\n",
                   gpu, d.name, vgpu::regs::space_name(space));
      return 2;
    }
    vgpu::regs::RegisterSpace cs(space, d);

    if (verb == "log") {
      const auto log = vgpu::regs::access_log(d.uuid, space);
      const size_t from = log.size() > static_cast<size_t>(last) ? log.size() - static_cast<size_t>(last) : 0;
      for (size_t i = from; i < log.size(); ++i) {
        const auto& e = log[i];
        const std::time_t t = static_cast<std::time_t>(e.time_ns / 1000000000ull);
        std::tm tm{};
        localtime_r(&t, &tm);
        char when[32];
        std::strftime(when, sizeof when, "%H:%M:%S", &tm);
        const auto* r = cs.at(e.offset);
        if (e.size > 4)
          std::printf("%s.%03llu  %-16s pid %-7u read   0x%03x-0x%03x (%u bytes)\n", when,
                      static_cast<unsigned long long>(e.time_ns / 1000000 % 1000), e.process.c_str(), e.pid,
                      e.offset, e.offset + e.size - 1, e.size);
        else
          std::printf("%s.%03llu  %-16s pid %-7u %-6s 0x%03x %-24s %u byte%s = 0x%0*x\n", when,
                      static_cast<unsigned long long>(e.time_ns / 1000000 % 1000), e.process.c_str(), e.pid,
                      e.write ? "write" : "read", e.offset, r ? r->name.c_str() : "", e.size, e.size == 1 ? "" : "s",
                      static_cast<int>(e.size * 2), e.value);
      }
      return 0;
    }

    if (verb == "dump" && space == vgpu::regs::Space::AmdMmio) {
      // Half a megabyte of mostly undeclared space: the declared registers.
      for (const auto& r : vgpu::regs::registers(space)) print_register(r, r.offset, cs.read(r.offset, 4));
      return 0;
    }
    if (verb == "dump") {
      const auto img = cs.image(extended ? vgpu::regs::kConfigSize : 256);
      for (size_t row = 0; row < img.size() / 16; ++row) {
        std::printf("%03zx:", row * 16);
        for (size_t col = 0; col < 16; ++col) std::printf(" %02x", img[row * 16 + col]);
        std::printf("\n");
      }
      return 0;
    }

    // A name, found where this device has it, or an offset on this device.
    const vgpu::regs::Register* r = nullptr;
    uint32_t offset = 0;
    if (parse_u32(pos[0], &offset)) {
      if (offset >= vgpu::regs::space_size(space)) {
        std::fprintf(stderr, "vgpu regs: 0x%x is past the end of the space\n", offset);
        return 2;
      }
      r = cs.at(offset);
    } else if ((r = vgpu::regs::find(space, pos[0]))) {
      offset = cs.offset_of(*r);
      if (offset == vgpu::regs::RegisterSpace::kAbsent) {
        std::fprintf(stderr, "vgpu regs: GPU %lld has no %s capability, so no %s\n", gpu, r->capability.c_str(),
                     r->name.c_str());
        return 2;
      }
    } else {
      std::fprintf(stderr, "vgpu regs: no register '%s'; `vgpu regs list` names them\n", pos[0].c_str());
      return 2;
    }
    // A register's own width unless --size says otherwise; a 24-bit register
    // is read as the dword around it, as software reads it.
    uint32_t sz = size ? static_cast<uint32_t>(size) : r ? (r->width == 24 ? 4 : r->width / 8) : 4;
    if (r && r->width == 24 && !size) offset &= ~3u;
    if (verb == "read") {
      const uint32_t v = cs.read(offset, sz);
      const vgpu::regs::Register* at = cs.at(offset);
      if (r && r->width == 24) at = r;
      if (at && sz * 8 >= at->width)
        print_register(*at, cs.offset_of(*at), at == r && r->width == 24 ? v >> 8 : v);
      else std::printf("0x%03x = 0x%0*x\n", offset, static_cast<int>(sz * 2), v);
      return 0;
    }
    uint32_t value = 0;
    if (!parse_u32(pos[1], &value) || (sz < 4 && value >> (sz * 8))) {
      std::fprintf(stderr, "vgpu regs: '%s' is not a %u-byte value\n", pos[1].c_str(), sz);
      return 2;
    }
    cs.write(offset, sz, value);
    if (r && r->width != 24) print_register(*r, cs.offset_of(*r), cs.value(*r));
    return 0;
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "vgpu regs: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vgpu regs: %s\n", e.what());
    return 1;
  }
}
