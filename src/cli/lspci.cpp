// lspci for the simulated GPUs.
//
// A session's lspci renders the devices through the host's real lspci when
// there is one (`lspci -F` over `vgpu smi --lspci-dump`), with a copy of the
// host's PCI ID database that also names the simulated cards it predates
// (`vgpu smi --lspci-ids FILE`, read with `lspci -i`). Where the host has no
// lspci at all -- a container image without pciutils -- `vgpu smi --lspci`
// prints what lspci would, for the options scripts use: -n/-nn, -D, -d, -s,
// -k, -v, -m/-mm, -x to -xxxx, -t and --version.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "machine.hpp"
#include "vgpu/regs.hpp"
#include "vgpu/telemetry.hpp"

namespace {

// Where distributions keep the PCI ID database lspci names devices from.
const char* const kIdsPaths[] = {"/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids", "/usr/share/pci.ids",
                                 "/usr/local/share/pci.ids"};

// VGPU_PCI_IDS names another (a path that does not exist: none at all, as on
// a machine without pciutils).
std::string host_ids_path() {
  if (const char* env = std::getenv("VGPU_PCI_IDS"); env) return std::ifstream(env).good() ? env : "";
  for (const char* p : kIdsPaths)
    if (std::ifstream(p).good()) return p;
  return "";
}

// The names the database gives: vendors, devices (vendor << 16 | device),
// subsystems (vendor, device, subvendor, subdevice), classes and subclasses
// (class << 8 | subclass, and the class alone at kClassOnly | class).
constexpr uint32_t kClassOnly = 0x10000;   // above every class << 8 | subclass
struct Names {
  std::map<uint32_t, std::string> vendor, device, cls;
  std::map<uint32_t, std::string> progif;   // class << 16 | subclass << 8 | prog-if
  std::map<uint64_t, std::string> subsystem;
};

// The simulated cards' names as the current PCI ID database spells them, for
// a host whose copy is older than the card (or has none).
const std::map<uint32_t, const char*> kKnownDevices = {
    {0x1002740cu, "Aldebaran/MI200 [Instinct MI250X/MI250]"},
    {0x100274a1u, "Aqua Vanjaram [Instinct MI300X]"},
    {0x100274a5u, "Aqua Vanjaram [Instinct MI325X]"},
    {0x100273bfu, "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"},
    {0x1002744cu, "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]"},
    {0x10027550u, "Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]"},
};

void add_defaults(Names* n) {
  n->vendor.emplace(0x1002, "Advanced Micro Devices, Inc. [AMD/ATI]");
  n->vendor.emplace(0x10de, "NVIDIA Corporation");
  n->cls.emplace(kClassOnly | 0x03, "Display controller");
  n->cls.emplace(0x0300, "VGA compatible controller");
  n->progif.emplace(0x030000, "VGA controller");
  n->cls.emplace(0x0302, "3D controller");
  n->cls.emplace(0x0380, "Display controller");
  n->cls.emplace(kClassOnly | 0x12, "Processing accelerators");
  n->cls.emplace(0x1200, "Processing accelerators");
}

Names load_names(const std::string& path) {
  Names n;
  std::ifstream in(path);
  std::string line;
  uint32_t vendor = 0, device = 0, cls = 0, sub = 0;
  bool in_classes = false, have_vendor = false, have_device = false;
  auto hex = [](const std::string& s, size_t at, size_t len) {
    return static_cast<uint32_t>(std::strtoul(s.substr(at, len).c_str(), nullptr, 16));
  };
  auto rest = [](const std::string& s, size_t at) {
    while (at < s.size() && s[at] == ' ') ++at;
    return s.substr(at);
  };
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (line.rfind("C ", 0) == 0 && line.size() > 4) {   // C 03  Display controller
      in_classes = true;
      cls = hex(line, 2, 2);
      n.cls[kClassOnly | cls] = rest(line, 4);
      continue;
    }
    if (in_classes) {
      if (line[0] == '\t' && line.size() > 3 && line[1] != '\t') {
        sub = hex(line, 1, 2);
        n.cls[cls << 8 | sub] = rest(line, 3);
      } else if (line.size() > 4 && line[0] == '\t' && line[1] == '\t') {
        n.progif[cls << 16 | sub << 8 | hex(line, 2, 2)] = rest(line, 4);
      }
      continue;
    }
    if (line[0] != '\t' && line.size() > 5 && std::isxdigit(static_cast<unsigned char>(line[0]))) {
      vendor = hex(line, 0, 4);
      have_vendor = true;
      have_device = false;
      n.vendor[vendor] = rest(line, 4);
    } else if (have_vendor && line.size() > 6 && line[0] == '\t' && line[1] != '\t') {
      device = hex(line, 1, 4);
      have_device = true;
      n.device[vendor << 16 | device] = rest(line, 5);
    } else if (have_device && line.size() > 12 && line[0] == '\t' && line[1] == '\t') {
      const uint64_t sv = hex(line, 2, 4), sd = hex(line, 7, 4);
      n.subsystem[static_cast<uint64_t>(vendor << 16 | device) << 32 | sv << 16 | sd] = rest(line, 11);
    }
  }
  return n;
}

// A device's name: the database's, the known name, or the model's own.
std::string device_name(const Names& n, uint32_t vendor, uint32_t device, const char* model) {
  if (auto it = n.device.find(vendor << 16 | device); it != n.device.end()) return it->second;
  if (auto it = kKnownDevices.find(vendor << 16 | device); it != kKnownDevices.end()) return it->second;
  std::string m = model;
  for (const char* p : {"AMD ", "NVIDIA "})
    if (m.rfind(p, 0) == 0) m = m.substr(std::strlen(p));
  return m;
}

struct Dev {
  uint32_t domain = 0, bus = 0, slot = 0, func = 0;
  std::vector<uint8_t> cfg;
  const vgpu::telemetry::DeviceSample* d = nullptr;
  uint32_t word(uint32_t at) const { return cfg[at] | cfg[at + 1] << 8; }
};

std::vector<Dev> devices(const vgpu::telemetry::Shared& s, uint32_t bytes) {
  std::vector<Dev> out;
  for (uint32_t i = 0; i < s.device_count; ++i) {
    Dev v;
    v.d = &s.devices[i];
    // "00000000:01:00.0" (NVML's eight-digit domain) or "0000:01:00.0".
    unsigned dom = 0, bus = 0, slot = 0, fn = 0;
    std::sscanf(v.d->bus_id, "%x:%x:%x.%x", &dom, &bus, &slot, &fn);
    v.domain = dom, v.bus = bus, v.slot = slot, v.func = fn;
    vgpu::regs::ConfigSpace cs(*v.d);
    v.cfg = cs.image(bytes);
    out.push_back(std::move(v));
  }
  return out;
}

int usage(FILE* f) {
  std::fprintf(f,
               "Usage: lspci [<switches>]\n\n"
               "VirtualGPU's lspci, for a machine without pciutils: the simulated GPUs.\n\n"
               "Display options:\n"
               "-m\t\tProduce machine-readable output (single -m for an obsolete format)\n"
               "-t\t\tShow bus tree\n\n"
               "Display options:\n"
               "-v\t\tBe verbose (-vv or -vvv for higher verbosity)\n"
               "-k\t\tShow kernel drivers handling each device\n"
               "-x\t\tShow hex-dump of the standard part of the config space\n"
               "-xxx\t\tShow hex-dump of the whole config space (dangerous; root only)\n"
               "-xxxx\t\tShow hex-dump of the 4096-byte extended config space (root only)\n"
               "-D\t\tAlways show domain numbers\n\n"
               "Resolving of device ID's to names:\n"
               "-n\t\tShow numeric ID's\n"
               "-nn\t\tShow both textual and numeric ID's (names & numbers)\n\n"
               "Selection of devices:\n"
               "-s [[[[<domain>]:]<bus>]:][<slot>][.[<func>]]\tShow only devices in selected slots\n"
               "-d [<vendor>]:[<device>][:<class>]\t\tShow only devices with specified ID's\n");
  return f == stdout ? 0 : 2;
}

// -s [[[[<domain>]:]<bus>]:][<slot>][.[<func>]]: empty parts match anything.
bool slot_matches(const std::string& spec, const Dev& v) {
  std::string s = spec, fn;
  if (const size_t dot = s.find('.'); dot != std::string::npos) {
    fn = s.substr(dot + 1);
    s = s.substr(0, dot);
  }
  std::vector<std::string> parts;
  std::stringstream ss(s);
  for (std::string p; std::getline(ss, p, ':');) parts.push_back(p);
  if (!s.empty() && s.back() == ':') parts.push_back("");
  std::string dom, bus, slot;
  if (parts.size() == 3) dom = parts[0], bus = parts[1], slot = parts[2];
  else if (parts.size() == 2) bus = parts[0], slot = parts[1];
  else if (parts.size() == 1) slot = parts[0];
  auto eq = [](const std::string& want, uint32_t have) {
    return want.empty() || want == "*" || std::strtoul(want.c_str(), nullptr, 16) == have;
  };
  return eq(dom, v.domain) && eq(bus, v.bus) && eq(slot, v.slot) && eq(fn, v.func);
}

// -d [<vendor>]:[<device>][:<class>]
bool id_matches(const std::string& spec, const Dev& v) {
  std::vector<std::string> parts;
  std::stringstream ss(spec);
  for (std::string p; std::getline(ss, p, ':');) parts.push_back(p);
  auto eq = [](const std::vector<std::string>& p, size_t k, uint32_t have) {
    return k >= p.size() || p[k].empty() || std::strtoul(p[k].c_str(), nullptr, 16) == have;
  };
  const uint32_t cls = v.cfg[0x0b] << 8 | v.cfg[0x0a];
  return eq(parts, 0, v.word(0)) && eq(parts, 1, v.word(2)) && eq(parts, 2, cls);
}

std::string size_text(uint64_t b) {
  static const char* kUnits[] = {"", "K", "M", "G", "T"};
  int u = 0;
  while (b >= 1024 && (b % 1024) == 0 && u < 4) b /= 1024, ++u;
  return std::to_string(b) + kUnits[u];
}

// The capability list, one line each, as lspci -v shows it: the standard
// ones from the pointer at 0x34, then the extended ones from 0x100.
void print_capabilities(const std::vector<uint8_t>& c) {
  auto w16 = [&](uint32_t at) { return static_cast<uint32_t>(c[at] | c[at + 1] << 8); };
  auto w32 = [&](uint32_t at) { return w16(at) | w16(at + 2) << 16; };
  const auto flag = [](bool b) { return b ? '+' : '-'; };
  if (!(w16(0x06) & 0x10)) return;   // no capability list
  uint32_t at = c[0x34] & ~3u;
  for (int guard = 0; at >= 0x40 && at < 0x100 && guard < 48; ++guard) {
    const uint32_t id = c[at];
    char b[160];
    switch (id) {
      case 0x01: std::snprintf(b, sizeof b, "Power Management version %u", w16(at + 2) & 7); break;
      case 0x05: {
        const uint32_t ctl = w16(at + 2);
        std::snprintf(b, sizeof b, "MSI: Enable%c Count=%u/%u Maskable%c 64bit%c", flag(ctl & 1), 1u << ((ctl >> 4) & 7),
                      1u << ((ctl >> 1) & 7), flag(ctl & 0x100), flag(ctl & 0x80));
        break;
      }
      case 0x10: {
        static const char* kTypes[] = {"Endpoint", "Legacy Endpoint", "?", "?", "Root Port", "Upstream Port",
                                       "Downstream Port", "PCI-Express to PCI/PCI-X Bridge", "PCI/PCI-X to PCI-Express Bridge",
                                       "Root Complex Integrated Endpoint", "Root Complex Event Collector"};
        const uint32_t caps = w16(at + 2), type = (caps >> 4) & 0xF;
        std::snprintf(b, sizeof b, "Express %s, MSI %02x", type < 11 ? kTypes[type] : "Unknown type", (caps >> 9) & 0x1F);
        break;
      }
      case 0x11: {
        const uint32_t ctl = w16(at + 2);
        std::snprintf(b, sizeof b, "MSI-X: Enable%c Count=%u Masked%c", flag(ctl & 0x8000), (ctl & 0x7FF) + 1, flag(ctl & 0x4000));
        break;
      }
      case 0x09: std::snprintf(b, sizeof b, "Vendor Specific Information: Len=%02x <?>", c[at + 2]); break;
      default: std::snprintf(b, sizeof b, "#%02x [%04x]", id, w16(at + 2)); break;
    }
    std::printf("\tCapabilities: [%x] %s\n", at, b);
    at = c[at + 1] & ~3u;
  }
  if (c.size() < 0x104 || !w32(0x100)) return;
  static const std::map<uint32_t, const char*> kExtended = {
      {0x0001, "Advanced Error Reporting"}, {0x0002, "Virtual Channel"}, {0x0003, "Device Serial Number"},
      {0x000b, "Vendor Specific Information"}, {0x000d, "Access Control Services"},
      {0x000e, "Alternative Routing-ID Interpretation (ARI)"}, {0x0010, "Single Root I/O Virtualization (SR-IOV)"},
      {0x0015, "Physical Resizable BAR"}, {0x0018, "Latency Tolerance Reporting"}, {0x0019, "Secondary PCI Express"},
      {0x001e, "L1 PM Substates"}, {0x0023, "Designated Vendor-Specific"}, {0x0025, "Data Link Feature"},
      {0x0026, "Physical Layer 16.0 GT/s"}, {0x0027, "Lane Margining at the Receiver"}, {0x002a, "Physical Layer 32.0 GT/s"}};
  at = 0x100;
  for (int guard = 0; at >= 0x100 && at + 4 <= c.size() && guard < 64; ++guard) {
    const uint32_t hdr = w32(at), id = hdr & 0xFFFF;
    if (!hdr) break;
    const auto it = kExtended.find(id);
    if (it != kExtended.end()) std::printf("\tCapabilities: [%x] %s\n", at, it->second);
    else std::printf("\tCapabilities: [%x] Extended Capability ID %#x\n", at, id);
    at = (hdr >> 20) & ~3u;
  }
}

}  // namespace

// `vgpu smi --lspci-ids OUT`: the host's PCI ID database with the simulated
// cards it does not know added under their vendors, or (with no database on
// the host) one naming just them.
int cmd_lspci_ids(const std::string& out_path) {
  vgpu::telemetry::Shared snap{};
  if (!vgpu::cli::read_machine(&snap)) return 1;
  const std::string host = host_ids_path();
  const Names known = host.empty() ? Names{} : load_names(host);
  std::map<uint32_t, std::map<uint32_t, std::string>> missing;   // vendor -> device -> name
  for (uint32_t i = 0; i < snap.device_count; ++i) {
    const auto& d = snap.devices[i];
    const uint32_t vendor = d.pci_device_id & 0xFFFF, device = d.pci_device_id >> 16;
    if (!known.device.count(vendor << 16 | device)) missing[vendor][device] = device_name(Names{}, vendor, device, d.name);
  }
  std::ofstream out(out_path);
  if (!out) {
    std::fprintf(stderr, "vgpu smi: cannot write %s\n", out_path.c_str());
    return 1;
  }
  Names defaults;
  add_defaults(&defaults);
  auto device_lines = [&](uint32_t vendor) {
    std::string s;
    for (const auto& [dev, name] : missing[vendor]) {
      char b[16];
      std::snprintf(b, sizeof b, "\t%04x  ", dev);
      s += b + name + "\n";
    }
    return s;
  };
  if (host.empty()) {
    for (const auto& [vendor, devs] : missing) {
      char b[16];
      std::snprintf(b, sizeof b, "%04x  ", vendor);
      out << b << defaults.vendor[vendor] << "\n" << device_lines(vendor);
    }
    out << "C 03  Display controller\n\t00  VGA compatible controller\n\t02  3D controller\n\t80  Display controller\n"
        << "C 12  Processing accelerators\n";
    return 0;
  }
  // Each missing device goes straight after its vendor's line.
  std::ifstream in(host);
  std::string line;
  while (std::getline(in, line)) {
    out << line << "\n";
    if (line.size() > 5 && line[0] != '\t' && line[0] != '#' && line[0] != 'C' && line[4] == ' ') {
      const uint32_t vendor = static_cast<uint32_t>(std::strtoul(line.substr(0, 4).c_str(), nullptr, 16));
      if (missing.count(vendor)) out << device_lines(vendor);
    }
  }
  return 0;
}

// `vgpu smi --lspci [lspci options]`: lspci's listing of the simulated GPUs.
int cmd_lspci(const std::vector<std::string>& args) {
  int numeric = 0, verbose = 0, machine = 0, hex = 0;
  bool domain = false, kernel = false, tree = false;
  std::string slot_spec, id_spec;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--version") {
      std::printf("lspci version 3.10.0 (VirtualGPU)\n");
      return 0;
    }
    if (a == "--help" || a == "-h" || a == "-?") return usage(stdout);
    if (a == "-s" || a == "-d" || a == "-i" || a == "-p" || a == "-A" || a == "-O" || a == "-F" || a == "-H") {
      if (i + 1 >= args.size()) {
        std::fprintf(stderr, "lspci: option requires an argument -- '%c'\n", a[1]);
        return usage(stderr);
      }
      if (a == "-s") slot_spec = args[++i];
      else if (a == "-d") id_spec = args[++i];
      else ++i;   // a database, access method or dump: this lspci has its own devices
      continue;
    }
    if (a.size() < 2 || a[0] != '-') {
      std::fprintf(stderr, "lspci: unknown argument '%s'\n", a.c_str());
      return usage(stderr);
    }
    for (size_t k = 1; k < a.size(); ++k) {
      switch (a[k]) {
        case 'n': ++numeric; break;
        case 'v': ++verbose; break;
        case 'm': ++machine; break;
        case 'x': ++hex; break;
        case 'D': domain = true; break;
        case 'k': kernel = true; break;
        case 't': tree = true; break;
        case 'b': case 'P': case 'q': case 'Q': case 'M': case 'G': break;   // accepted, no effect here
        default:
          std::fprintf(stderr, "lspci: invalid option -- '%c'\n", a[k]);
          return usage(stderr);
      }
    }
  }
  vgpu::telemetry::Shared snap{};
  if (!vgpu::cli::read_machine(&snap)) return 1;
  const uint32_t bytes = hex >= 4 ? vgpu::regs::kConfigSize : hex == 3 ? 256 : 64;
  std::vector<Dev> all = devices(snap, vgpu::regs::kConfigSize);
  const std::string ids = host_ids_path();
  Names names = ids.empty() ? Names{} : load_names(ids);
  add_defaults(&names);

  std::vector<const Dev*> shown;
  for (const Dev& v : all)
    if ((slot_spec.empty() || slot_matches(slot_spec, v)) && (id_spec.empty() || id_matches(id_spec, v)))
      shown.push_back(&v);

  if (tree) {
    // Each GPU is on a bus of its own, below a root port lspci does not see
    // here: one branch per bus.
    for (size_t k = 0; k < shown.size(); ++k) {
      const char* lead = shown.size() == 1 ? "-" : k == 0 ? "-+-" : k + 1 == shown.size() ? " \\-" : " +-";
      std::printf("%s[%04x:%02x]---%02x.%x\n", lead, shown[k]->domain, shown[k]->bus, shown[k]->slot, shown[k]->func);
    }
    return 0;
  }

  const bool show_domain = domain || std::any_of(all.begin(), all.end(), [](const Dev& v) { return v.domain != 0; });
  for (const Dev* v : shown) {
    const uint32_t vendor = v->word(0), device = v->word(2), svendor = v->word(0x2c), sdevice = v->word(0x2e);
    const uint32_t base = v->cfg[0x0b], sub = v->cfg[0x0a], progif = v->cfg[0x09], rev = v->cfg[0x08];
    char addr[32];
    if (show_domain) std::snprintf(addr, sizeof addr, "%04x:%02x:%02x.%x", v->domain, v->bus, v->slot, v->func);
    else std::snprintf(addr, sizeof addr, "%02x:%02x.%x", v->bus, v->slot, v->func);
    // A name, a number, or both, as -n and -nn ask.
    auto named = [&](const std::string& name, const char* fmt, uint32_t id) {
      char num[16];
      std::snprintf(num, sizeof num, fmt, id);
      if (numeric == 1) return std::string(num);
      if (numeric >= 2) return name + " [" + num + "]";
      return name;
    };
    std::string cls_name;
    if (auto it = names.cls.find(base << 8 | sub); it != names.cls.end()) cls_name = it->second;
    else if (auto it2 = names.cls.find(kClassOnly | base); it2 != names.cls.end()) cls_name = it2->second;
    else {
      char b[32];
      std::snprintf(b, sizeof b, "Class %04x", base << 8 | sub);
      cls_name = b;
    }
    const std::string vendor_name = names.vendor.count(vendor) ? names.vendor.at(vendor) : "Unknown vendor";
    const std::string dev_name = device_name(names, vendor, device, v->d->name);
    // The subsystem: its own entry, or -- where it is the device itself -- the
    // device's name, as pciutils has it.
    std::string sub_name;
    const uint64_t skey = static_cast<uint64_t>(vendor << 16 | device) << 32 | svendor << 16 | sdevice;
    if (auto it = names.subsystem.find(skey); it != names.subsystem.end()) sub_name = it->second;
    else if (svendor == vendor && sdevice == device) sub_name = dev_name;
    else {
      char b[32];
      std::snprintf(b, sizeof b, "Device %04x", sdevice);
      sub_name = b;
    }
    const std::string svendor_name = names.vendor.count(svendor) ? names.vendor.at(svendor) : "Unknown vendor";
    const char* driver = std::strcmp(v->d->vendor, "amd") == 0 ? "amdgpu" : "nvidia";
    char rev_text[8], prog_text[8];
    std::snprintf(rev_text, sizeof rev_text, "%02x", rev);
    std::snprintf(prog_text, sizeof prog_text, "%02x", progif);

    if (machine && verbose) {   // -vmm: one record per device
      std::printf("Slot:\t%s\n", addr);
      std::printf("Class:\t%s\n", named(cls_name, "%04x", base << 8 | sub).c_str());
      std::printf("Vendor:\t%s\n", named(vendor_name, "%04x", vendor).c_str());
      std::printf("Device:\t%s\n", named(dev_name, "%04x", device).c_str());
      std::printf("SVendor:\t%s\n", named(svendor_name, "%04x", svendor).c_str());
      std::printf("SDevice:\t%s\n", named(sub_name, "%04x", sdevice).c_str());
      if (rev) std::printf("Rev:\t%s\n", rev_text);
      std::printf("ProgIf:\t%s\n", prog_text);
      if (kernel) std::printf("Driver:\t%s\nModule:\t%s\n", driver, driver);
      std::printf("\n");
      continue;
    }
    if (machine) {   // -mm: one quoted line per device
      auto q = [](const std::string& t) { return "\"" + t + "\""; };
      std::string line = std::string(addr) + " " + q(named(cls_name, "%04x", base << 8 | sub)) + " " +
                         q(named(vendor_name, "%04x", vendor)) + " " + q(named(dev_name, "%04x", device));
      if (rev) line += std::string(" -r") + rev_text;
      line += std::string(" -p") + prog_text;
      line += " " + q(named(svendor_name, "%04x", svendor)) + " " + q(named(sub_name, "%04x", sdevice));
      std::printf("%s\n", line.c_str());
      continue;
    }

    std::string line = std::string(addr) + " ";
    if (numeric == 1) {
      char b[64];
      std::snprintf(b, sizeof b, "%04x: %04x:%04x", base << 8 | sub, vendor, device);
      line += b;
    } else {
      line += named(cls_name, "%04x", base << 8 | sub) + ": " + vendor_name + " " + dev_name;
      if (numeric >= 2) {
        char b[32];
        std::snprintf(b, sizeof b, " [%04x:%04x]", vendor, device);
        line += b;
      }
    }
    if (rev) line += std::string(" (rev ") + rev_text + ")";
    // The programming interface: where it is not zero, or -- verbose -- where
    // the database has a name for it, the name too.
    {
      const auto it = names.progif.find(base << 16 | sub << 8 | progif);
      const bool name = verbose && it != names.progif.end();
      if (name) line += std::string(" (prog-if ") + prog_text + " [" + it->second + "])";
      else if (progif) line += std::string(" (prog-if ") + prog_text + ")";
    }
    std::printf("%s\n", line.c_str());
    if (verbose || kernel) {
      if (numeric == 1) std::printf("\tSubsystem: %04x:%04x\n", svendor, sdevice);
      else {
        std::string t = svendor_name + " " + sub_name;
        if (numeric >= 2) {
          char b[32];
          std::snprintf(b, sizeof b, " [%04x:%04x]", svendor, sdevice);
          t += b;
        }
        std::printf("\tSubsystem: %s\n", t.c_str());
      }
    }
    if (verbose) {
      const uint32_t irq = v->cfg[0x3c] == 0xff ? 0 : v->cfg[0x3c];
      std::printf("\tFlags: bus master, fast devsel, latency 0%s\n",
                  irq ? (", IRQ " + std::to_string(irq)).c_str() : "");
      vgpu::regs::ConfigSpace cs(*v->d);
      for (int b = 0; b < 6; ++b) {
        const vgpu::regs::Bar bar = vgpu::regs::bar(cs, *v->d, b);
        if (!bar.size) continue;
        if (bar.io) {
          std::printf("\tI/O ports at %llx [size=%s]\n", static_cast<unsigned long long>(bar.base), size_text(bar.size).c_str());
          continue;
        }
        std::printf("\tMemory at %llx (%s, %s) [size=%s]\n", static_cast<unsigned long long>(bar.base),
                    bar.is64 ? "64-bit" : "32-bit", bar.prefetchable ? "prefetchable" : "non-prefetchable",
                    size_text(bar.size).c_str());
      }
      print_capabilities(v->cfg);
    }
    if (verbose || kernel) std::printf("\tKernel driver in use: %s\n\tKernel modules: %s\n", driver, driver);
    if (hex) {
      for (uint32_t row = 0; row < bytes / 16; ++row) {
        std::printf("%02x:", row * 16);
        for (uint32_t col = 0; col < 16; ++col) std::printf(" %02x", v->cfg[row * 16 + col]);
        std::printf("\n");
      }
    }
    if (verbose || hex) std::printf("\n");
  }
  return 0;
}
