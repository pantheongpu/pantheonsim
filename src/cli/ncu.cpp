// `vgpu ncu` -- the part of Nsight Compute's command line a tool drives, over
// VirtualGPU's own counters. Inside `vgpu shell` it is what `ncu` runs.
//
// NVIDIA's ncu cannot profile a simulated GPU: it reaches the driver through
// undocumented internal interfaces this clean-room project does not implement,
// and says it "failed to connect to the CUDA driver". What it is used for is
// still worth having here -- Pantheon's --profile runs `ncu --query-metrics`,
// keeps the metrics the device lists, and then runs its workload under
// `ncu --csv --metrics ...` -- so this answers the same commands with the
// counters the engine records for every launch.
//
// It reports a metric only when the engine counts exactly what Nsight
// Compute's definition counts. Those are the ones fixed by the program's own
// addresses: how many warp-level global and shared loads and stores ran, how
// many 32-byte sectors and requests the global ones touched, how many of those
// bytes were asked for, and how many extra passes shared-memory bank conflicts
// forced. Everything that needs a timing, cache or DRAM model -- cycles, stall
// reasons, hit rates, throughput -- is left out of --query-metrics, so a tool
// that asks sees it is not there rather than being handed an invented number.
// Local-memory counts are left out too: on a GPU they include register spills,
// which PTX does not have.
#include <cxxabi.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "machine.hpp"
#include "vgpu/telemetry.hpp"

namespace {

// One served metric: its full Nsight Compute name, what it is, its unit, and
// how to compute it from a launch record.
struct Metric {
  const char* name;
  const char* unit;
  const char* description;
  double (*value)(const std::map<std::string, double>& r);
};

double get(const std::map<std::string, double>& r, const char* key) {
  const auto it = r.find(key);
  return it == r.end() ? 0.0 : it->second;
}

double bytes_per_sector_pct(double bytes, double sectors) {
  return sectors > 0 ? 100.0 * bytes / (sectors * 32.0) : 0.0;
}

const std::vector<Metric>& metrics() {
  static const std::vector<Metric> m = {
      {"smsp__inst_executed_op_global_ld.sum", "inst",
       "# of warp-level global load instructions executed",
       [](const auto& r) { return get(r, "global_requests_ld"); }},
      {"smsp__inst_executed_op_global_st.sum", "inst",
       "# of warp-level global store instructions executed",
       [](const auto& r) { return get(r, "global_requests_st"); }},
      {"smsp__inst_executed_op_shared_ld.sum", "inst",
       "# of warp-level shared load instructions executed",
       [](const auto& r) { return get(r, "shared_requests_ld"); }},
      {"smsp__inst_executed_op_shared_st.sum", "inst",
       "# of warp-level shared store instructions executed",
       [](const auto& r) { return get(r, "shared_requests_st"); }},
      {"l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "request",
       "# of requests for global loads",
       [](const auto& r) { return get(r, "global_requests_ld"); }},
      {"l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "request",
       "# of requests for global stores",
       [](const auto& r) { return get(r, "global_requests_st"); }},
      {"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "sector",
       "# of 32-byte sectors requested for global loads",
       [](const auto& r) { return get(r, "global_sectors_ld"); }},
      {"l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "sector",
       "# of 32-byte sectors requested for global stores",
       [](const auto& r) { return get(r, "global_sectors_st"); }},
      {"smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", "%",
       "% of the bytes in each global-load sector that the threads asked for",
       [](const auto& r) {
         return bytes_per_sector_pct(get(r, "global_bytes_ld"), get(r, "global_sectors_ld"));
       }},
      {"smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct", "%",
       "% of the bytes in each global-store sector that the threads wrote",
       [](const auto& r) {
         return bytes_per_sector_pct(get(r, "global_bytes_st"), get(r, "global_sectors_st"));
       }},
      {"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "",
       "# of extra passes shared loads needed because of bank conflicts",
       [](const auto& r) { return get(r, "shared_bank_conflicts_ld"); }},
      {"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "",
       "# of extra passes shared stores needed because of bank conflicts",
       [](const auto& r) { return get(r, "shared_bank_conflicts_st"); }},
  };
  return m;
}

const Metric* find_metric(const std::string& name) {
  for (const Metric& m : metrics())
    if (name == m.name) return &m;
  return nullptr;
}

// The name without its rollup (".sum", ".pct", ...), which is how
// --query-metrics lists metrics unless asked for every suffix.
std::string base_name(const std::string& full) {
  const auto dot = full.rfind('.');
  return dot == std::string::npos ? full : full.substr(0, dot);
}

// A launch record is one line of flat JSON written by the runtime (see
// record_counters in runtime.cpp): numbers, three-number arrays, and the
// kernel name. This reads exactly that shape and nothing more general.
struct Launch {
  std::map<std::string, double> values;
  std::string kernel;
  unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
  int device = 0, pid = 0;
};

bool parse_launch(const std::string& line, Launch* out) {
  size_t i = 0;
  const auto skip = [&] { while (i < line.size() && (line[i] == ' ' || line[i] == ',')) ++i; };
  if (line.empty() || line[0] != '{') return false;
  i = 1;
  while (i < line.size()) {
    skip();
    if (line[i] == '}') return true;
    if (line[i] != '"') return false;
    const size_t kend = line.find('"', i + 1);
    if (kend == std::string::npos) return false;
    const std::string key = line.substr(i + 1, kend - i - 1);
    i = kend + 1;
    if (i >= line.size() || line[i] != ':') return false;
    ++i;
    if (line[i] == '"') {
      const size_t vend = line.find('"', i + 1);
      if (vend == std::string::npos) return false;
      if (key == "kernel") out->kernel = line.substr(i + 1, vend - i - 1);
      i = vend + 1;
    } else if (line[i] == '[') {
      const size_t vend = line.find(']', i);
      if (vend == std::string::npos) return false;
      unsigned v[3] = {1, 1, 1};
      std::sscanf(line.substr(i + 1, vend - i - 1).c_str(), "%u,%u,%u", &v[0], &v[1], &v[2]);
      if (key == "grid") std::copy(v, v + 3, out->grid);
      if (key == "block") std::copy(v, v + 3, out->block);
      i = vend + 1;
    } else {
      char* end = nullptr;
      const double v = std::strtod(line.c_str() + i, &end);
      if (end == line.c_str() + i) return false;
      out->values[key] = v;
      if (key == "device") out->device = static_cast<int>(v);
      if (key == "pid") out->pid = static_cast<int>(v);
      i = static_cast<size_t>(end - line.c_str());
    }
  }
  return false;
}

std::string demangle(const std::string& name) {
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> d(
      abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free);
  return status == 0 && d ? std::string(d.get()) : name;
}

std::string csv(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += '"';
    out += c;
  }
  return out + "\"";
}

std::string format_value(double v, const char* unit) {
  char buf[64];
  if (std::strcmp(unit, "%") == 0)
    std::snprintf(buf, sizeof buf, "%.2f", v);
  else
    std::snprintf(buf, sizeof buf, "%.0f", v);
  return buf;
}

void usage(std::FILE* to) {
  std::fprintf(to,
               "usage: ncu [options] <application> [application arguments]\n"
               "       ncu --query-metrics [--query-metrics-mode base|suffix|all] [--devices N]\n"
               "\n"
               "VirtualGPU's ncu: Nsight Compute's command line, answered from the counters\n"
               "the simulator records for every kernel launch. Supported options:\n"
               "  --csv                      report as CSV\n"
               "  --metrics a,b,...          metrics to report (default: every one listed)\n"
               "  --set NAME, --section NAME accepted; sections need a timing model and are\n"
               "                             not produced, so only the metrics are reported\n"
               "  -c, --launch-count N       report the first N matching launches\n"
               "  -s, --launch-skip N        skip the first N matching launches\n"
               "  -k, --kernel-name NAME     only launches of this kernel (regex:... for a regex)\n"
               "  --devices N[,M]            only launches on these devices\n"
               "  --log-file PATH            write the report there instead of stdout\n"
               "  --query-metrics            list the metrics this can report, with what each is\n"
               "  -o, --export               refused: .ncu-rep is Nsight Compute's own format\n"
               "Anything that would need cycles, stalls or cache behaviour is not listed:\n"
               "see `vgpu counters` for what is counted and why the rest is not.\n");
}

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(item);
  return out;
}

int query_metrics(const std::string& mode, const std::set<int>& devices) {
  vgpu::telemetry::Shared snap{};
  if (!vgpu::read_machine(&snap)) return 1;
  for (uint32_t d = 0; d < snap.device_count; ++d) {
    if (!devices.empty() && !devices.count(static_cast<int>(d))) continue;
    const auto& dev = snap.devices[d];
    std::printf("Device %s (%s)\n", dev.name, dev.architecture);
    std::printf("%-66s %s\n", "-----------------------------------------------------------------",
                "--------------------------------------------------------------------");
    std::printf("%-66s %s\n", "Metric Name", "Metric Description");
    std::set<std::string> shown;
    for (const Metric& m : metrics()) {
      const std::string name = mode == "all" || mode == "suffix" ? m.name : base_name(m.name);
      if (!shown.insert(name).second) continue;
      std::printf("%-66s %s (counted by VirtualGPU from the executed PTX)\n", name.c_str(),
                  m.description);
    }
    std::printf("\n");
  }
  return 0;
}

}  // namespace

int cmd_ncu(const std::vector<std::string>& args) {
  bool csv_out = false, query = false;
  std::string query_mode = "base", log_file, kernel_filter;
  std::vector<std::string> wanted;
  std::set<int> devices;
  long launch_count = -1, launch_skip = 0;
  size_t i = 0;
  const auto value = [&](const std::string& opt) -> std::string {
    if (i + 1 >= args.size()) {
      std::fprintf(stderr, "==ERROR== %s needs a value\n", opt.c_str());
      std::exit(1);
    }
    return args[++i];
  };
  for (; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a.empty() || a[0] != '-') break;   // the application starts here
    if (a == "--") { ++i; break; }
    if (a == "--version" || a == "-v" || a == "-V") {
      std::printf("VirtualGPU ncu: Nsight Compute command-line subset over simulated counters\n");
      return 0;
    }
    if (a == "-h" || a == "--help") { usage(stdout); return 0; }
    if (a == "--csv") csv_out = true;
    else if (a == "--query-metrics") query = true;
    else if (a == "--query-metrics-mode") query_mode = value(a);
    else if (a == "--metrics") for (const auto& m : split_commas(value(a))) wanted.push_back(m);
    else if (a == "--devices") {
      const std::string v = value(a);
      if (v != "all")
        for (const auto& d : split_commas(v)) devices.insert(std::atoi(d.c_str()));
    } else if (a == "-c" || a == "--launch-count") launch_count = std::atol(value(a).c_str());
    else if (a == "-s" || a == "--launch-skip") launch_skip = std::atol(value(a).c_str());
    else if (a == "-k" || a == "--kernel-name") kernel_filter = value(a);
    else if (a == "--log-file") log_file = value(a);
    else if (a == "--set" || a == "--section" || a == "--page" || a == "--target-processes" ||
             a == "--replay-mode" || a == "--cache-control" || a == "--clock-control" ||
             a == "--kernel-name-base" || a == "--print-units" || a == "--nvtx-include")
      (void)value(a);   // accepted; see usage for what each means here
    else if (a == "--nvtx" || a == "--print-summary" || a == "-f" || a == "--force-overwrite")
      ;
    else if (a == "-o" || a == "--export") {
      std::fprintf(stderr,
                   "==ERROR== --export writes an .ncu-rep, which is Nsight Compute's own report "
                   "format and is not produced here. Use --csv.\n");
      return 1;
    } else {
      std::fprintf(stderr, "==ERROR== unsupported option '%s'\n", a.c_str());
      usage(stderr);
      return 1;
    }
  }
  if (query) return query_metrics(query_mode, devices);
  if (i >= args.size()) {
    usage(stderr);
    return 1;
  }

  // Which metrics: the ones asked for that are served, or every one.
  std::vector<const Metric*> report;
  std::vector<std::string> not_served;
  if (wanted.empty()) {
    for (const Metric& m : metrics()) report.push_back(&m);
  } else {
    for (const std::string& w : wanted)
      if (const Metric* m = find_metric(w)) report.push_back(m);
      else not_served.push_back(w);
  }
  if (!not_served.empty()) {
    std::fprintf(stderr, "==WARNING== %zu metric(s) are not reported here, because counting them "
                         "would need a timing, cache or DRAM model (see `vgpu counters`):",
                 not_served.size());
    for (const auto& n : not_served) std::fprintf(stderr, " %s", n.c_str());
    std::fprintf(stderr, "\n");
  }

  // Run the application with the per-launch record switched on.
  char tmpl[] = "/tmp/vgpu-ncu-XXXXXX";
  const int fd = mkstemp(tmpl);
  if (fd < 0) {
    std::perror("==ERROR== ncu: temporary file");
    return 1;
  }
  close(fd);
  const std::string record = tmpl;
  std::fflush(nullptr);
  const pid_t child = fork();
  if (child < 0) {
    std::perror("==ERROR== ncu: fork");
    return 1;
  }
  if (child == 0) {
    setenv("VGPU_COUNTERS_FILE", record.c_str(), 1);
    std::vector<char*> argv;
    for (size_t k = i; k < args.size(); ++k) argv.push_back(const_cast<char*>(args[k].c_str()));
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    std::fprintf(stderr, "==ERROR== ncu: cannot run %s: %s\n", argv[0], std::strerror(errno));
    _exit(127);
  }
  std::printf("==PROF== Connected to process %d (%s)\n", static_cast<int>(child), args[i].c_str());
  std::fflush(stdout);
  int status = 0;
  while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
  const int rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);

  std::vector<Launch> launches;
  {
    std::ifstream in(record);
    std::string line;
    std::regex re;
    const bool is_regex = kernel_filter.rfind("regex:", 0) == 0;
    if (is_regex) re = std::regex(kernel_filter.substr(6));
    long seen = 0;
    while (std::getline(in, line)) {
      Launch l;
      if (!parse_launch(line, &l)) continue;
      if (!devices.empty() && !devices.count(l.device)) continue;
      if (!kernel_filter.empty()) {
        const std::string pretty = demangle(l.kernel);
        const std::string base = pretty.substr(0, pretty.find('('));
        const bool match = is_regex ? std::regex_search(pretty, re)
                                    : (l.kernel == kernel_filter || base == kernel_filter);
        if (!match) continue;
      }
      if (seen++ < launch_skip) continue;
      if (launch_count >= 0 && static_cast<long>(launches.size()) >= launch_count) break;
      launches.push_back(std::move(l));
    }
  }
  std::remove(record.c_str());
  std::printf("==PROF== Disconnected from process %d\n", static_cast<int>(child));

  vgpu::telemetry::Shared snap{};
  const bool have_machine = vgpu::read_machine(&snap);
  std::FILE* out = stdout;
  if (!log_file.empty() && !(out = std::fopen(log_file.c_str(), "w"))) {
    std::perror("==ERROR== ncu: --log-file");
    return 1;
  }
  char host[256] = "localhost";
  gethostname(host, sizeof host - 1);
  const std::string process = std::filesystem::path(args[i]).filename().string();
  if (launches.empty())
    std::fprintf(stderr, "==WARNING== No kernels were profiled.\n");
  if (csv_out)
    std::fprintf(out,
                 "\"ID\",\"Process ID\",\"Process Name\",\"Host Name\",\"Kernel Name\",\"Context\","
                 "\"Stream\",\"Block Size\",\"Grid Size\",\"Device\",\"CC\",\"Section Name\","
                 "\"Metric Name\",\"Metric Unit\",\"Metric Value\"\n");
  for (size_t id = 0; id < launches.size(); ++id) {
    const Launch& l = launches[id];
    char blk[64], grd[64], cc[16] = "N/A";
    std::snprintf(blk, sizeof blk, "(%u, %u, %u)", l.block[0], l.block[1], l.block[2]);
    std::snprintf(grd, sizeof grd, "(%u, %u, %u)", l.grid[0], l.grid[1], l.grid[2]);
    if (have_machine && l.device >= 0 && static_cast<uint32_t>(l.device) < snap.device_count)
      std::snprintf(cc, sizeof cc, "%d.%d", snap.devices[l.device].cc_major,
                    snap.devices[l.device].cc_minor);
    const std::string kernel = demangle(l.kernel);
    if (!csv_out)
      std::fprintf(out, "  %s, %s, Context 1, Device %d, CC %s\n    Command line profiler metrics\n",
                   kernel.c_str(), grd, l.device, cc);
    for (const Metric* m : report) {
      const std::string v = format_value(m->value(l.values), m->unit);
      if (csv_out)
        std::fprintf(out, "%s,%s,%s,%s,%s,\"1\",\"N/A\",%s,%s,\"%d\",%s,%s,%s,%s,%s\n",
                     csv(std::to_string(id)).c_str(), csv(std::to_string(l.pid)).c_str(),
                     csv(process).c_str(), csv(host).c_str(), csv(kernel).c_str(),
                     csv(blk).c_str(), csv(grd).c_str(), l.device, csv(cc).c_str(),
                     csv("Command line profiler metrics").c_str(), csv(m->name).c_str(),
                     csv(m->unit).c_str(), csv(v).c_str());
      else
        std::fprintf(out, "    %-66s %-8s %s\n", m->name, m->unit, v.c_str());
    }
  }
  if (out != stdout) std::fclose(out);
  return rc;
}
