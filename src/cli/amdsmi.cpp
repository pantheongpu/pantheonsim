// VirtualGPU's drop-in amd-smi: the commands health tools run against AMD
// GPUs -- list, metric --ecc/--ecc-blocks and ras --cper -- answering from the
// same machine state rocm-smi and nvidia-smi read. The output follows amd-smi's
// documented shapes: JSON is a list with one object per GPU keyed by "gpu", and
// the human-readable form is its upper-cased keys, indented by four.
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "args.hpp"
#include "machine.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/telemetry.hpp"

namespace {

// A value in amd-smi's output: a count, a string, or named children.
struct Node {
  std::string key;
  std::string text;
  bool number = false;
  std::vector<Node> kids;
  bool object = false;
};
Node num(const std::string& key, uint64_t v) { return {key, std::to_string(v), true, {}, false}; }
Node str(const std::string& key, const std::string& v) { return {key, v, false, {}, false}; }
Node obj(const std::string& key, std::vector<Node> kids) { return {key, "", false, std::move(kids), true}; }

std::string quoted(const std::string& s) {
  std::string out = "\"";
  for (char ch : s) {
    if (ch == '"' || ch == '\\') out += '\\';
    out += ch;
  }
  return out + "\"";
}

void print_json(const Node& n, int depth, bool last) {
  const std::string pad(static_cast<size_t>(depth) * 4, ' ');
  std::printf("%s%s: ", pad.c_str(), quoted(n.key).c_str());
  if (!n.object) {
    std::printf("%s%s\n", n.number ? n.text.c_str() : quoted(n.text).c_str(), last ? "" : ",");
    return;
  }
  std::printf("{\n");
  for (size_t i = 0; i < n.kids.size(); ++i) print_json(n.kids[i], depth + 1, i + 1 == n.kids.size());
  std::printf("%s}%s\n", pad.c_str(), last ? "" : ",");
}

// One list entry per GPU, each an object whose first key is "gpu".
void print_gpus_json(const std::vector<std::pair<uint32_t, std::vector<Node>>>& gpus) {
  std::printf("[\n");
  for (size_t g = 0; g < gpus.size(); ++g) {
    std::printf("    {\n        \"gpu\": %u%s\n", gpus[g].first, gpus[g].second.empty() ? "" : ",");
    const auto& kids = gpus[g].second;
    for (size_t i = 0; i < kids.size(); ++i) print_json(kids[i], 2, i + 1 == kids.size());
    std::printf("    }%s\n", g + 1 == gpus.size() ? "" : ",");
  }
  std::printf("]\n");
}

std::string upper(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  return s;
}

void print_human(const Node& n, int depth) {
  const std::string pad(static_cast<size_t>(depth) * 4, ' ');
  if (!n.object) {
    std::printf("%s%s: %s\n", pad.c_str(), upper(n.key).c_str(), n.text.c_str());
    return;
  }
  std::printf("%s%s:\n", pad.c_str(), upper(n.key).c_str());
  for (const Node& k : n.kids) print_human(k, depth + 1);
}

void print_gpus_human(const std::vector<std::pair<uint32_t, std::vector<Node>>>& gpus) {
  for (const auto& [index, kids] : gpus) {
    std::printf("GPU: %u\n", index);
    for (const Node& k : kids) print_human(k, 1);
    std::printf("\n");
  }
}

int usage(FILE* to) {
  std::fprintf(to,
               "usage: amd-smi list [-g GPU ...] [--json]\n"
               "       amd-smi metric [-e] [-k] [-g GPU ...] [--json]\n"
               "       amd-smi ras --cper [--severity SEVERITY ...] [-g GPU ...]\n"
               "       amd-smi version\n"
               "\n"
               "VirtualGPU's drop-in amd-smi, answering from the same machine state as rocm-smi.\n"
               "Of metric, only the ECC counts are modelled: -e the totals, -k per RAS block.\n"
               "ras --cper lists the machine's recent ECC error records; --folder is not\n"
               "modelled, since no CPER record files are written.\n");
  return to == stdout ? 0 : 2;
}

bool is_amd(const vgpu::telemetry::DeviceSample& d) { return std::strcmp(d.vendor, "amd") == 0; }

// "00000000:03:00.0" as amdgpu writes it: a four-digit domain.
std::string bdf(const vgpu::telemetry::DeviceSample& d) {
  const std::string b = d.bus_id;
  return b.size() > 4 && b.compare(0, 4, "0000") == 0 ? b.substr(4) : b;
}

// The RAS blocks amd-smi counts from sysfs, in the order it lists them. Device
// memory is the UMC (memory controller); the on-chip memories are GFX, as in
// rocm-smi's --showrasinfo.
constexpr const char* kBlocks[] = {"UMC", "SDMA", "GFX", "MMHUB", "PCIE_BIF", "HDP", "XGMI_WAFL"};

std::vector<Node> ecc_totals(const vgpu::telemetry::DeviceSample& d, const vgpu::ras::Counters& c) {
  using vgpu::ras::Severity;
  if (!d.ecc_enabled)
    // What amd-smi prints when the driver has no ECC count to give.
    return {str("total_correctable_count", "N/A"), str("total_uncorrectable_count", "N/A"),
            str("cache_correctable_count", "N/A"), str("cache_uncorrectable_count", "N/A")};
  const auto umc = [&](Severity s) {
    return c.ecc[static_cast<uint32_t>(s)][static_cast<uint32_t>(vgpu::ras::Location::DeviceMemory)];
  };
  return {num("total_correctable_count", c.ecc_total(Severity::Corrected)),
          num("total_uncorrectable_count", c.ecc_total(Severity::Uncorrected)),
          num("total_deferred_count", 0),
          num("cache_correctable_count", c.ecc_total(Severity::Corrected) - umc(Severity::Corrected)),
          num("cache_uncorrectable_count", c.ecc_total(Severity::Uncorrected) - umc(Severity::Uncorrected))};
}

Node ecc_blocks(const vgpu::telemetry::DeviceSample& d, const vgpu::ras::Counters& c) {
  using vgpu::ras::Severity;
  if (!d.ecc_enabled) return str("ecc_blocks", "N/A");
  std::vector<Node> blocks;
  for (const char* b : kBlocks) {
    const auto count = [&](Severity s) -> uint64_t {
      const uint64_t memory = c.ecc[static_cast<uint32_t>(s)][static_cast<uint32_t>(vgpu::ras::Location::DeviceMemory)];
      if (std::strcmp(b, "UMC") == 0) return memory;
      if (std::strcmp(b, "GFX") == 0) return c.ecc_total(s) - memory;
      return 0;
    };
    blocks.push_back(obj(b, {num("correctable_count", count(Severity::Corrected)),
                             num("uncorrectable_count", count(Severity::Uncorrected)),
                             num("deferred_count", 0)}));
  }
  return obj("ecc_blocks", std::move(blocks));
}

// CPER records for a GPU's recent ECC errors, oldest first: an uncorrectable
// error in device memory is recorded as non-fatal (the page is retired and the
// device goes on), a corrected one as corrected.
struct Cper {
  uint64_t time_ns;
  bool uncorrected;
};
std::vector<Cper> cper_records(const std::string& uuid) {
  std::vector<Cper> out;
  uint64_t after = 0;
  vgpu::ras::Event e{};
  while (vgpu::ras::next_event(uuid, vgpu::ras::kEventSingleBitEcc | vgpu::ras::kEventDoubleBitEcc,
                               &after, &e))
    out.push_back({e.time_ns, e.type == vgpu::ras::kEventDoubleBitEcc});
  return out;
}

std::string cper_time(uint64_t ns) {
  const std::time_t t = static_cast<std::time_t>(ns / 1000000000ull);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y/%m/%d %H:%M:%S", &tm);
  return buf;
}

}  // namespace

int cmd_amd_smi(const std::vector<std::string>& args) {
  if (args.empty()) return usage(stderr);
  const std::string& cmd = args[0];
  if (cmd == "-h" || cmd == "--help" || cmd == "help") return usage(stdout);
  if (cmd != "list" && cmd != "metric" && cmd != "ras" && cmd != "version") {
    std::fprintf(stderr, "amd-smi: '%s' is not modelled by VirtualGPU (list, metric, ras, version)\n",
                 cmd.c_str());
    return 2;
  }

  bool json = false, csv = false, ecc = false, blocks = false, cper = false;
  std::vector<std::string> gpu_args, severities;
  for (size_t i = 1; i < args.size(); ++i) {
    std::string a = args[i];
    std::string inline_value;
    if (const size_t eq = a.find('='); a.rfind("--", 0) == 0 && eq != std::string::npos) {
      inline_value = a.substr(eq + 1);
      a = a.substr(0, eq);
    }
    // Options that take one or more values run to the next option.
    auto values = [&](std::vector<std::string>* into) {
      if (!inline_value.empty()) {
        into->push_back(inline_value);
        return;
      }
      while (i + 1 < args.size() && args[i + 1].rfind("-", 0) != 0) into->push_back(args[++i]);
    };
    if (a == "--json") json = true;
    else if (a == "--csv") csv = true;
    else if (a == "-g" || a == "--gpu") values(&gpu_args);
    else if (cmd == "metric" && (a == "-e" || a == "--ecc")) ecc = true;
    else if (cmd == "metric" && (a == "-k" || a == "--ecc-blocks")) blocks = true;
    else if (cmd == "ras" && a == "--cper") cper = true;
    else if (cmd == "ras" && a == "--severity") values(&severities);
    else if (a == "-h" || a == "--help") return usage(stdout);
    else if (cmd == "metric" && a.rfind("-", 0) == 0) {
      std::fprintf(stderr, "amd-smi metric: %s is not modelled by VirtualGPU; -e and -k are\n", a.c_str());
      return 2;
    } else if (cmd == "ras" && (a == "--folder" || a == "--follow" || a == "--file-limit" || a == "--afid")) {
      std::fprintf(stderr, "amd-smi ras: %s is not modelled: VirtualGPU writes no CPER record files\n",
                   a.c_str());
      return 2;
    } else {
      std::fprintf(stderr, "amd-smi %s: unrecognized argument '%s'\n", cmd.c_str(), a.c_str());
      return 2;
    }
  }
  if (csv) {
    std::fprintf(stderr, "amd-smi: --csv is not modelled by VirtualGPU; use --json\n");
    return 2;
  }

  if (cmd == "version") {
    std::printf("AMDSMI Tool: VirtualGPU | AMDSMI Library version: N/A | ROCm version: N/A\n");
    return 0;
  }

  vgpu::telemetry::Shared snap{};
  if (!vgpu::cli::read_machine(&snap)) return 1;
  vgpu::drop_lost_amd(&snap);
  std::vector<uint32_t> amd;
  for (uint32_t i = 0; i < snap.device_count; ++i)
    if (is_amd(snap.devices[i])) amd.push_back(i);
  if (amd.empty()) {
    std::fprintf(stderr, "amd-smi: no AMD GPU in this machine (it has %s)\n",
                 snap.device_count ? snap.devices[0].name : "none");
    return 1;
  }
  std::vector<uint32_t> sel;
  bool all = gpu_args.empty();
  for (const std::string& g : gpu_args) {
    if (g == "all") {
      all = true;
      continue;
    }
    long long n = -1;
    if (!vgpu::cli::parse_int(g, 0, static_cast<long long>(amd.size()) - 1, &n)) {
      std::fprintf(stderr, "amd-smi: GPU '%s' is not one of 0-%zu\n", g.c_str(), amd.size() - 1);
      return 2;
    }
    sel.push_back(static_cast<uint32_t>(n));
  }
  if (all) {
    sel.clear();
    for (uint32_t i = 0; i < amd.size(); ++i) sel.push_back(i);
  }

  if (cmd == "ras") {
    if (!cper) {
      std::fprintf(stderr, "amd-smi ras: name --cper\n");
      return 2;
    }
    bool want_fatal = severities.empty(), want_uncorrected = severities.empty(),
         want_corrected = severities.empty();
    for (const std::string& s : severities) {
      if (s == "all") want_fatal = want_uncorrected = want_corrected = true;
      else if (s == "fatal") want_fatal = true;
      else if (s == "nonfatal" || s == "nonfatal-uncorrected") want_uncorrected = true;
      else if (s == "nonfatal-corrected" || s == "corrected") want_corrected = true;
      else {
        std::fprintf(stderr, "amd-smi ras: unknown --severity '%s'\n", s.c_str());
        return 2;
      }
    }
    (void)want_fatal;   // nothing injected is fatal
    // amd-smi prints this table whatever format was asked for.
    std::printf("WARNING: No CPER files will be dumped unless --folder=<folder_name> is specified "
                "and cper entries exist.\n");
    std::printf("%-20s %-7s %-20s\n", "timestamp", "gpu_id", "severity");
    for (uint32_t g : sel) {
      std::vector<Cper> records;
      try {
        records = cper_records(snap.devices[amd[g]].uuid);
      } catch (const std::exception&) {
      }
      for (const Cper& r : records) {
        if (r.uncorrected ? !want_uncorrected : !want_corrected) continue;
        std::printf("%-20s %-7u %-20s\n", cper_time(r.time_ns).c_str(), g,
                    r.uncorrected ? "NONFATAL-UNCORRECTED" : "NONFATAL-CORRECTED");
      }
    }
    return 0;
  }

  std::vector<std::pair<uint32_t, std::vector<Node>>> gpus;
  for (uint32_t g : sel) {
    const auto& d = snap.devices[amd[g]];
    std::vector<Node> kids;
    if (cmd == "list") {
      std::string uuid = d.uuid;
      if (uuid.rfind("GPU-", 0) == 0) uuid = uuid.substr(4);
      // KFD's id for the GPU is a hash; this one is stable per device. Node 0
      // is the CPU, so the GPUs start at 1.
      uint64_t kfd = 1469598103934665603ull;
      for (char ch : uuid) kfd = (kfd ^ static_cast<unsigned char>(ch)) * 1099511628211ull;
      kids = {str("bdf", bdf(d)), str("uuid", uuid), num("kfd_id", kfd % 65536), num("node_id", g + 1),
              num("partition_id", 0)};
    } else {
      vgpu::ras::Counters c{};
      try {
        c = vgpu::ras::read(d.uuid).since_load;
      } catch (const std::exception&) {
      }
      const bool both = !ecc && !blocks;   // metric alone: every modelled metric
      if (ecc || both) kids.push_back(obj("ecc", ecc_totals(d, c)));
      if (blocks || both) kids.push_back(ecc_blocks(d, c));
    }
    gpus.emplace_back(g, std::move(kids));
  }
  if (json) print_gpus_json(gpus);
  else print_gpus_human(gpus);
  return 0;
}
