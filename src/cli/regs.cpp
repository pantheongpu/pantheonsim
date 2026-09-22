// `vgpu regs`: a device's registers, as bring-up and debug work reaches them --
// listed from the register database, read and decoded field by field, written
// with each register's semantics, dumped whole, and the log of who touched them.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.hpp"
#include "machine.hpp"
#include "vgpu/regs.hpp"
#include "vgpu/telemetry.hpp"

namespace {

int usage(FILE* to) {
  std::fprintf(to,
               "usage: vgpu regs list [--status done|model]\n"
               "       vgpu regs read [--gpu N] REGISTER [--size 1|2|4]\n"
               "       vgpu regs write [--gpu N] REGISTER VALUE [--size 1|2|4]\n"
               "       vgpu regs dump [--gpu N] [--extended]\n"
               "       vgpu regs log [--gpu N] [--last K]\n"
               "\n"
               "A device's PCI configuration space, from the register database\n"
               "(registers/pci-config.yaml): each register's offset, width, access and what\n"
               "backs it. REGISTER is a name from `list` or an offset (0x08a); a read decodes\n"
               "its fields. A write follows the register's access: rw bits are kept, a 1\n"
               "written to an rw1c status bit clears it until its cause recurs, and a base\n"
               "address register reads back its size after all ones are written. Writes are\n"
               "shared by every process on the machine, and every access is logged with the\n"
               "process that made it. --gpu defaults to 0.\n");
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

void print_register(const vgpu::regs::Register& r, uint32_t v) {
  const int digits = static_cast<int>(r.width / 4);
  std::printf("0x%03x  %-28s = 0x%0*x   (%s, %s)\n", r.offset, r.name.c_str(), digits, v,
              vgpu::regs::access_name(r.access), r.status.c_str());
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

std::string offset_key(uint32_t offset) {
  char b[8];
  std::snprintf(b, sizeof b, "0x%03x", offset);
  return b;
}

}  // namespace

int cmd_regs(const std::vector<std::string>& args) {
  if (args.empty()) return usage(stderr);
  const std::string& verb = args[0];
  if (verb == "-h" || verb == "--help" || verb == "help") return usage(stdout);
  if (verb != "list" && verb != "read" && verb != "write" && verb != "dump" && verb != "log") {
    std::fprintf(stderr, "vgpu regs: unknown action '%s' (list, read, write, dump or log)\n", verb.c_str());
    return 2;
  }
  long long gpu = 0, size = 0, last = 20;
  bool extended = false;
  std::string status;
  std::vector<std::string> pos;
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    const bool takes = a == "--gpu" || a == "--size" || a == "--last" || a == "--status";
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
    if (verb == "list") {
      std::printf("%-6s %-5s %-5s %-28s %-22s %s\n", "OFFSET", "WIDTH", "ACCESS", "REGISTER", "BACKING", "STATUS");
      for (const auto& r : vgpu::regs::config_registers()) {
        if (!status.empty() && r.status != status) continue;
        std::printf("0x%03x  %-5u %-6s %-28s %-22s %s\n", r.offset, r.width, vgpu::regs::access_name(r.access),
                    r.name.c_str(), r.backing.empty() ? "reset value" : r.backing.c_str(), r.status.c_str());
      }
      return 0;
    }

    vgpu::telemetry::Shared snap{};
    if (!vgpu::cli::read_machine(&snap)) return 1;
    if (gpu >= snap.device_count) {
      std::fprintf(stderr, "vgpu regs: there is no GPU %lld (this machine has %u)\n", gpu, snap.device_count);
      return 2;
    }
    const auto& d = snap.devices[gpu];

    if (verb == "log") {
      const auto log = vgpu::regs::access_log(d.uuid);
      const size_t from = log.size() > static_cast<size_t>(last) ? log.size() - static_cast<size_t>(last) : 0;
      for (size_t i = from; i < log.size(); ++i) {
        const auto& e = log[i];
        const std::time_t t = static_cast<std::time_t>(e.time_ns / 1000000000ull);
        std::tm tm{};
        localtime_r(&t, &tm);
        char when[32];
        std::strftime(when, sizeof when, "%H:%M:%S", &tm);
        const auto* r = vgpu::regs::find_config(offset_key(e.offset));
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

    vgpu::regs::ConfigSpace cs(d);
    if (verb == "dump") {
      const auto img = cs.image(extended ? vgpu::regs::kConfigSize : 256);
      for (size_t row = 0; row < img.size() / 16; ++row) {
        std::printf("%03zx:", row * 16);
        for (size_t col = 0; col < 16; ++col) std::printf(" %02x", img[row * 16 + col]);
        std::printf("\n");
      }
      return 0;
    }

    const vgpu::regs::Register* r = vgpu::regs::find_config(pos[0]);
    uint32_t offset = 0;
    if (r) {
      offset = r->offset;
    } else if (!parse_u32(pos[0], &offset) || offset >= vgpu::regs::kConfigSize) {
      std::fprintf(stderr, "vgpu regs: no register '%s'; `vgpu regs list` names them\n", pos[0].c_str());
      return 2;
    }
    // A register's own width unless --size says otherwise; a 24-bit register
    // is read as the dword around it, as software reads it.
    uint32_t sz = size ? static_cast<uint32_t>(size) : r ? (r->width == 24 ? 4 : r->width / 8) : 4;
    if (r && r->width == 24 && !size) offset &= ~3u;
    if (verb == "read") {
      const uint32_t v = cs.read(offset, sz);
      const vgpu::regs::Register* at = vgpu::regs::find_config(offset_key(offset));
      if (r && r->width == 24) at = r;
      if (at && sz * 8 >= at->width) print_register(*at, at == r && r->width == 24 ? v >> 8 : v);
      else std::printf("0x%03x = 0x%0*x\n", offset, static_cast<int>(sz * 2), v);
      return 0;
    }
    uint32_t value = 0;
    if (!parse_u32(pos[1], &value) || (sz < 4 && value >> (sz * 8))) {
      std::fprintf(stderr, "vgpu regs: '%s' is not a %u-byte value\n", pos[1].c_str(), sz);
      return 2;
    }
    cs.write(offset, sz, value);
    if (r && r->width != 24) print_register(*r, cs.value(*r));
    return 0;
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "vgpu regs: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vgpu regs: %s\n", e.what());
    return 1;
  }
}
