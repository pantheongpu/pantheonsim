// `vgpu shell` — an interactive GPU machine simulator.
//
// Asks what machine you want (GPU model, how many, CUDA/ROCm and driver
// versions, OS and kernel), then drops you into a normal shell where the usual
// tools behave as if that hardware were present: nvidia-smi, rocm-smi,
// rocm_agent_enumerator, lspci, dmesg, uname, /proc/driver/nvidia/version, and
// any CUDA program you run.
//
// Two layers of realism:
//   * Always: a session bin/ directory first on PATH supplies the tools, the
//     shim directory on LD_LIBRARY_PATH supplies libcuda/libcudart/NVML/NVENC,
//     and the session holds the virtual devices open so telemetry is live.
//   * When user namespaces are available (default, --no-isolate to skip):
//     re-executes inside `unshare -r -m` and bind-mounts the synthesized
//     /proc/driver/nvidia, /sys/class/drm and /etc/os-release over the real
//     ones, so even programs that read those absolute paths see the simulated
//     machine.
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"

namespace {

struct OsChoice {
  const char* label;
  const char* id;
  const char* version_id;
  const char* pretty;
  const char* kernel;
};

// A small menu of common CI/base images. Anything else can be typed in.
const OsChoice kOsChoices[] = {
    {"Ubuntu 22.04 LTS", "ubuntu", "22.04", "Ubuntu 22.04.4 LTS", "5.15.0-91-generic"},
    {"Ubuntu 24.04 LTS", "ubuntu", "24.04", "Ubuntu 24.04.1 LTS", "6.8.0-45-generic"},
    {"Rocky Linux 9", "rocky", "9.3", "Rocky Linux 9.3 (Blue Onyx)", "5.14.0-362.el9.x86_64"},
    {"Debian 12", "debian", "12", "Debian GNU/Linux 12 (bookworm)", "6.1.0-18-amd64"},
};

struct Config {
  std::string gpu = "nvidia/h100";
  std::string command;   // -c: run this instead of an interactive shell
  int count = 1;
  long long vram_mb = 0;  // 0 = use the profile's own capacity
  std::string cuda = "12.4";
  std::string rocm = "6.2.0";
  std::string driver = "550.54.15";
  OsChoice os = kOsChoices[0];
  std::string kernel;
  std::string hostname = "vgpu-sim";
  double load = 0.0;
  bool isolate = true;
};

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Reads one answer directly from fd 0, one byte at a time.
//
// Deliberately not std::getline: iostreams buffer ahead, which would swallow
// input intended for the shell we hand stdin to afterwards. Reading exactly
// through the newline leaves the descriptor positioned correctly, so
// `printf 'answers...\ncommands...\n' | vgpu shell` works.
std::string ask(const std::string& prompt, const std::string& def) {
  std::cout << "  " << prompt << " [" << def << "]: " << std::flush;
  std::string line;
  char ch;
  while (true) {
    ssize_t n = ::read(0, &ch, 1);
    if (n <= 0) return def;  // EOF: take the default
    if (ch == '\n') break;
    line += ch;
  }
  line = trim(line);
  return line.empty() ? def : line;
}

void write_file(const std::string& path, const std::string& contents, bool executable = false) {
  std::ofstream f(path);
  f << contents;
  f.close();
  if (executable) ::chmod(path.c_str(), 0755);
}

void make_dirs(const std::string& path) {
  std::string cur;
  for (size_t i = 1; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      cur = path.substr(0, i);
      ::mkdir(cur.c_str(), 0755);
    }
  }
}

std::string exe_path() {
  char buf[4096];
  ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return "vgpu";
  buf[n] = '\0';
  return buf;
}

// The directory holding libcuda.so.1 / libcudart.so.13 / libnvidia-ml.so.1,
// built alongside the vgpu binary.
std::string shim_dir() {
  std::string exe = exe_path();
  size_t slash = exe.find_last_of('/');
  std::string dir = slash == std::string::npos ? "." : exe.substr(0, slash);
  // build/bin/vgpu -> build/shim ; build/vgpu -> build/shim
  if (dir.size() > 4 && dir.compare(dir.size() - 4, 4, "/bin") == 0)
    dir = dir.substr(0, dir.size() - 4);
  return dir + "/shim";
}

// ---------------------------------------------------------------------------
// Synthetic system files
// ---------------------------------------------------------------------------

std::string nvidia_proc_version(const Config& c) {
  return "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  " + c.driver +
         "  Release Build  (VirtualGPU)\n"
         "GCC version:  gcc version 12.3.0\n";
}

std::string os_release(const Config& c) {
  std::string id = c.os.id;
  std::string s;
  s += std::string("PRETTY_NAME=\"") + c.os.pretty + "\"\n";
  s += std::string("NAME=\"") + (id == "ubuntu" ? "Ubuntu" : id == "debian" ? "Debian GNU/Linux"
                                                                            : "Rocky Linux") +
       "\"\n";
  s += std::string("VERSION_ID=\"") + c.os.version_id + "\"\n";
  s += std::string("ID=") + id + "\n";
  if (id == "ubuntu" || id == "debian") s += "ID_LIKE=debian\n";
  if (id == "rocky") s += "ID_LIKE=\"rhel centos fedora\"\n";
  s += "HOME_URL=\"https://example.invalid/\"\n";
  return s;
}

// A plausible kernel ring buffer: boot, then the GPU driver bringing up each
// device. Timestamps are deterministic so runs are reproducible.
std::string dmesg_log(const Config& c, const vgpu::DeviceProfile& p) {
  char buf[512];
  std::string s;
  auto line = [&](double t, const std::string& msg) {
    std::snprintf(buf, sizeof buf, "[%12.6f] %s\n", t, msg.c_str());
    s += buf;
  };
  bool amd = p.vendor == "amd";
  line(0.000000, "Linux version " + c.kernel + " (build@vgpu) (gcc 12.3.0) #1 SMP");
  line(0.000000, "Command line: BOOT_IMAGE=/boot/vmlinuz-" + c.kernel + " root=/dev/sda1 ro");
  line(0.412000, "PCI: Using configuration type 1 for base access");
  line(1.004512, "ACPI: PCI Root Bridge [PCI0] (domain 0000 [bus 00-ff])");

  double t = 2.0;
  for (int i = 0; i < c.count; ++i) {
    char bdf[32];
    std::snprintf(bdf, sizeof bdf, "0000:%02x:00.0", i + 1);
    std::snprintf(buf, sizeof buf, "pci %s: [%04x:%04x] type 00 class 0x030200", bdf,
                  p.telemetry.pci_vendor_id, p.telemetry.pci_device_id);
    line(t += 0.001, buf);
    std::snprintf(buf, sizeof buf, "pci %s: reg 0x10: [mem 0x%08x-0x%08x 64bit pref]", bdf,
                  0xf0000000 + i * 0x1000000, 0xf0ffffff + i * 0x1000000);
    line(t += 0.001, buf);
  }
  if (amd) {
    line(t += 0.30, "[drm] amdgpu kernel modesetting enabled.");
    line(t += 0.01, "amdgpu: Virtual CRAT table created for CPU");
    for (int i = 0; i < c.count; ++i) {
      char bdf[32];
      std::snprintf(bdf, sizeof bdf, "0000:%02x:00.0", i + 1);
      line(t += 0.05, std::string("amdgpu ") + bdf + ": enabling device (0006 -> 0007)");
      line(t += 0.01, std::string("amdgpu ") + bdf + ": amdgpu: Fetched VBIOS from VFCT");
      line(t += 0.02, std::string("amdgpu ") + bdf + ": amdgpu: " +
                          std::to_string(p.vram_bytes / (1024 * 1024)) + "M of VRAM memory ready");
      line(t += 0.01, std::string("amdgpu ") + bdf + ": amdgpu: ASIC is " + p.model);
      line(t += 0.01, "kfd kfd: amdgpu: added device " +
                          std::to_string(p.telemetry.pci_vendor_id) + ":" +
                          std::to_string(p.telemetry.pci_device_id));
    }
    line(t += 0.10, "amdgpu: HMM registered " +
                        std::to_string(p.vram_bytes / (1024 * 1024)) + "MB device memory");
  } else {
    line(t += 0.30, "nvidia: loading out-of-tree module taints kernel.");
    line(t += 0.01, "nvidia-nvlink: Nvlink Core is being initialized, major device number 234");
    for (int i = 0; i < c.count; ++i) {
      char bdf[32];
      std::snprintf(bdf, sizeof bdf, "0000:%02x:00.0", i + 1);
      line(t += 0.05, std::string("nvidia ") + bdf + ": enabling device (0000 -> 0003)");
      line(t += 0.01, std::string("nvidia ") + bdf + ": vgaarb: VGA decodes changed");
    }
    line(t += 0.10, "NVRM: loading NVIDIA UNIX Open Kernel Module for x86_64  " + c.driver);
    line(t += 0.02, "nvidia_uvm: module uses symbols from proprietary module nvidia, inheriting taint.");
    line(t += 0.01, "nvidia-uvm: Loaded the UVM driver, major device number 235.");
    line(t += 0.05, "nvidia-modeset: Loading NVIDIA Kernel Mode Setting Driver for UNIX platforms " +
                        c.driver);
  }
  line(t += 0.50, "VirtualGPU: " + std::to_string(c.count) + " x " + p.model +
                      " presented by pantheonsim (simulated; no physical device)");
  return s;
}

// ---------------------------------------------------------------------------
// Session materialization
// ---------------------------------------------------------------------------

struct Session {
  std::string dir, bin, root, telemetry, dmesg_path, pci_dump;
};

Session build_session(const Config& c, const vgpu::DeviceProfile& p) {
  Session s;
  s.dir = "/tmp/vgpu-session-" + std::to_string(::getpid());
  s.bin = s.dir + "/bin";
  s.root = s.dir + "/root";
  s.telemetry = s.dir + "/telemetry";
  s.dmesg_path = s.dir + "/dmesg.log";
  s.pci_dump = s.dir + "/pci.dump";
  make_dirs(s.bin);
  make_dirs(s.root + "/proc/driver/nvidia/gpus");
  make_dirs(s.root + "/sys/class/drm");
  make_dirs(s.root + "/etc");

  const std::string vgpu = exe_path();
  const std::string shim = shim_dir();

  write_file(s.root + "/proc/driver/nvidia/version", nvidia_proc_version(c));
  write_file(s.root + "/etc/os-release", os_release(c));
  write_file(s.dmesg_path, dmesg_log(c, p));

  // Per-device sysfs-ish entries, enough for scripts that count cards.
  for (int i = 0; i < c.count; ++i) {
    char bdf[32];
    std::snprintf(bdf, sizeof bdf, "0000:%02x:00.0", i + 1);
    std::string gpudir = s.root + "/proc/driver/nvidia/gpus/" + bdf;
    make_dirs(gpudir);
    char info[512];
    std::snprintf(info, sizeof info,
                  "Model: \t\t %s\nIRQ:   \t\t %d\nGPU UUID: \t GPU-simulated-%d\n"
                  "Video BIOS: \t 96.00.00.00.00\nBus Type: \t PCIe\n"
                  "DMA Size: \t 47 bits\nDMA Mask: \t 0x7fffffffffff\n"
                  "Bus Location: \t %s\nDevice Minor: \t %d\n",
                  p.model.c_str(), 128 + i, i, bdf, i);
    write_file(gpudir + "/information", info);
    std::string card = s.root + "/sys/class/drm/card" + std::to_string(i);
    make_dirs(card + "/device");
    char id[16];
    std::snprintf(id, sizeof id, "0x%04x\n", p.telemetry.pci_vendor_id);
    write_file(card + "/device/vendor", id);
    std::snprintf(id, sizeof id, "0x%04x\n", p.telemetry.pci_device_id);
    write_file(card + "/device/device", id);
  }

  // Resolves a program to an absolute path using PATH as it stands now, which
  // is before the session binds its own tools over the real ones.
  auto find_program = [](const char* name) -> std::string {
    const char* path = std::getenv("PATH");
    if (!path) return "";
    const std::string p(path);
    size_t at = 0;
    while (at <= p.size()) {
      const size_t colon = p.find(':', at);
      const std::string dir =
          p.substr(at, colon == std::string::npos ? std::string::npos : colon - at);
      if (!dir.empty()) {
        const std::string cand = dir + "/" + name;
        if (::access(cand.c_str(), X_OK) == 0) {
          char resolved[4096];
          if (::realpath(cand.c_str(), resolved)) return resolved;
          return cand;
        }
      }
      if (colon == std::string::npos) break;
      at = colon + 1;
    }
    return "";
  };

  // --- session tools, first on PATH ---
  auto tool = [&](const std::string& name, const std::string& body) {
    write_file(s.bin + "/" + name, "#!/usr/bin/env bash\n" + body, true);
  };
  // --query-gpu, --format and -q go through untouched: callers ask for
  // particular fields in a particular shape and then parse what comes back,
  // so collapsing them all into one fixed CSV answers a different question.
  tool("nvidia-smi",
       "# VirtualGPU session tool.\n"
       "case \"${1:-}\" in\n"
       "  --version) echo \"NVIDIA-SMI version  : VirtualGPU (simulated)\";\n"
       "             echo \"DRIVER version      : " + c.driver + "\";\n"
       "             echo \"CUDA Version        : " + c.cuda + "\"; exit 0 ;;\n"
       "  -L|--list-gpus) exec \"" + vgpu + "\" smi --list ;;\n"
       "esac\nexec \"" + vgpu + "\" smi \"$@\"\n");
  tool("rocm-smi", "exec \"" + vgpu + "\" smi --rocm \"$@\"\n");
  tool("rocm_agent_enumerator", "exec \"" + vgpu + "\" smi --agents\n");
  tool("dmesg",
       "# Replays this session's synthetic kernel ring buffer.\n"
       "case \" $* \" in\n"
       "  *\" -w \"*|*\" --follow \"*) cat \"" + s.dmesg_path + "\"; tail -f /dev/null ;;\n"
       "esac\n"
       "cat \"" + s.dmesg_path + "\"\n");
  tool("uname",
       "# Reports the simulated kernel; everything else defers to the real uname.\n"
       "K=\"" + c.kernel + "\"\n"
       "case \"${1:-}\" in\n"
       "  -r) echo \"$K\" ;;\n"
       "  -a) echo \"Linux " + c.hostname + " $K #1 SMP x86_64 x86_64 x86_64 GNU/Linux\" ;;\n"
       "  -s|\"\") echo Linux ;;\n"
       "  *) exec /usr/bin/uname \"$@\" ;;\n"
       "esac\n");
  tool("lspci",
       "# Renders the simulated GPUs through the real lspci, which resolves their\n"
       "# names from the host pci.ids like any other device.\n"
       "\"" + vgpu + "\" smi --lspci-dump > \"" + s.pci_dump + "\" 2>/dev/null\n"
       "exec /usr/bin/lspci -F \"" + s.pci_dump + "\" \"$@\"\n");
  // The session compiler: the real one with -cudart shared added.
  //
  // nvcc links the CUDA runtime statically by default, and a static cudart is
  // NVIDIA's own runtime inside the binary, reaching libcuda through an
  // undocumented internal table rather than the documented driver API. Against
  // a simulated driver it probes several hundred entry points and then refuses
  // with "integrity checks failed" on the program's first CUDA call. Linking
  // the runtime shared changes nothing about the program and lets VirtualGPU's
  // libcudart answer, so build systems work unmodified.
  //
  // The real compiler's path is resolved here, before the session binds this
  // script over it: a PATH search at run time would find only this script and
  // exec itself forever.
  const std::string real_nvcc = find_program("nvcc");
  tool("nvcc",
       "# The session compiler. VGPU_NVCC_PASSTHROUGH=1 removes the -cudart flag.\n"
       "if [ \"${1:-}\" = \"--version\" ]; then\n"
       "  echo 'nvcc: NVIDIA (R) Cuda compiler driver'\n"
       "  echo 'Cuda compilation tools, release " + c.cuda + " (VirtualGPU session)'\n"
       "  exit 0\nfi\n"
       "REAL='" + real_nvcc + "'\n"
       "[ -x \"$REAL\" ] || { echo 'nvcc: no CUDA toolkit in this session' >&2; exit 127; }\n"
       "[ \"${VGPU_NVCC_PASSTHROUGH:-0}\" = 1 ] && exec \"$REAL\" \"$@\"\n"
       "for a in \"$@\"; do\n"
       "  case \"$a\" in\n"
       "    -cudart|-cudart=*|--cudart|--cudart=*|--help|-h) exec \"$REAL\" \"$@\" ;;\n"
       "  esac\n"
       "done\n"
       "exec \"$REAL\" -cudart shared \"$@\"\n");
  tool("vgpu", "exec \"" + vgpu + "\" \"$@\"\n");
  return s;
}

// The CUDA bin directories a build system is likely to put in front of PATH,
// in the order the common tools list them.
std::vector<std::string> cuda_bin_directories() {
  std::vector<std::string> out;
  auto add = [&](const std::string& d) {
    struct stat st{};
    if (::stat(d.c_str(), &st) == 0 && S_ISDIR(st.st_mode) &&
        std::find(out.begin(), out.end(), d) == out.end())
      out.push_back(d);
  };
  for (const char* var : {"CUDA_HOME", "CUDA_PATH"})
    if (const char* r = std::getenv(var); r && *r) add(std::string(r) + "/bin");
  add("/usr/local/cuda/bin");
  // Versioned toolkits, newest first, which is the order these tools use.
  if (DIR* d = ::opendir("/usr/local")) {
    std::vector<std::string> versioned;
    while (struct dirent* e = ::readdir(d))
      if (std::strncmp(e->d_name, "cuda-", 5) == 0)
        versioned.push_back(std::string("/usr/local/") + e->d_name + "/bin");
    ::closedir(d);
    std::sort(versioned.rbegin(), versioned.rend());
    for (const auto& v : versioned) add(v);
  }
  return out;
}

std::string human_vram(uint64_t bytes) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.0f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
  return buf;
}

}  // namespace

int cmd_shell(const std::vector<std::string>& args) {
  Config c;
  bool prompt = true;
  std::string stage2_session;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto next = [&]() { return i + 1 < args.size() ? args[++i] : std::string(); };
    if (a == "--gpu") c.gpu = next();
    else if (a == "--count") c.count = std::atoi(next().c_str());
    else if (a == "--vram-mb") c.vram_mb = std::atoll(next().c_str());
    else if (a == "--cuda") c.cuda = next();
    else if (a == "--rocm") c.rocm = next();
    else if (a == "--driver") c.driver = next();
    else if (a == "--kernel") c.kernel = next();
    else if (a == "--hostname") c.hostname = next();
    else if (a == "--load") c.load = std::atof(next().c_str());
    else if (a == "--no-isolate") c.isolate = false;
    else if (a == "-c" || a == "--command") c.command = next();
    else if (a == "--no-prompt" || a == "-y") prompt = false;
    else if (a == "--os") {
      // Accepts "rocky", "rocky:9" and "rocky:9.3". Rocky's version_id is
      // "9.3", so requiring the exact string made "--os rocky:9" match nothing
      // -- and an unmatched --os used to leave the default in place silently,
      // which handed back a different machine than the one asked for. An
      // argument that names no OS is an error now.
      const std::string want = next();
      const OsChoice* found = nullptr;
      for (const auto& o : kOsChoices) {
        const std::string id(o.id), ver(o.version_id);
        if (want == id || want == id + ":" + ver) { found = &o; break; }
        // "rocky:9" matches "rocky:9.3", but "ubuntu:2" must not match
        // "ubuntu:22.04" -- so the split has to land on a version boundary.
        if (want.rfind(id + ":", 0) == 0) {
          const std::string part = want.substr(id.size() + 1);
          if (ver.rfind(part, 0) == 0 && (ver.size() == part.size() || ver[part.size()] == '.')) {
            found = &o;
            break;
          }
        }
      }
      if (!found) {
        std::fprintf(stderr, "vgpu shell: unknown OS '%s'. Available:\n", want.c_str());
        for (const auto& o : kOsChoices)
          std::fprintf(stderr, "  %s:%s  (%s)\n", o.id, o.version_id, o.label);
        return 2;
      }
      c.os = *found;
    } else if (a == "--stage2") {
      stage2_session = next();
      prompt = false;
    } else {
      std::fprintf(stderr, "vgpu shell: unknown argument '%s'\n", a.c_str());
      return 2;
    }
  }

  // Prompting only makes sense when there is someone to answer. With -c the
  // session is a one-shot command, and with stdin closed or piped there is no
  // one at all -- and an unanswered prompt does not fail, it silently takes
  // the default. That is how `--os rocky:9` became Ubuntu 22.04: the flag was
  // parsed, the prompt ran anyway, read EOF, and overwrote it. A machine that
  // quietly differs from the one that was asked for is worse than an error.
  if (!c.command.empty() || !::isatty(STDIN_FILENO)) prompt = false;

  if (prompt) {
    std::cout << "\n  VirtualGPU machine simulator\n"
              << "  Configure the machine to simulate; press Enter to accept a default.\n\n";
    std::cout << "  Available GPUs:\n";
    auto ids = vgpu::available_gpus();
    for (size_t i = 0; i < ids.size(); ++i) {
      vgpu::DeviceProfile p = vgpu::load_gpu(ids[i]);
      std::printf("    %-14s %-28s %s%s\n", ids[i].c_str(), p.model.c_str(),
                  human_vram(p.vram_bytes).c_str(),
                  p.vendor == "amd" ? "   (discovery only: AMD execution unimplemented)" : "");
    }
    std::cout << "\n";
    c.gpu = ask("GPU model", c.gpu);
    c.count = std::atoi(ask("How many GPUs", std::to_string(c.count)).c_str());
    vgpu::DeviceProfile probe = vgpu::load_gpu(c.gpu);
    bool amd = probe.vendor == "amd";
    c.vram_mb = std::atoll(
        ask("VRAM per GPU (MiB)", std::to_string(probe.vram_bytes / (1024 * 1024))).c_str());
    if (amd)
      c.rocm = ask("ROCm version", c.rocm);
    else
      c.cuda = ask("CUDA version", c.cuda);
    c.driver = ask("Driver version", c.driver);
    std::cout << "\n  Operating systems:\n";
    for (size_t i = 0; i < sizeof kOsChoices / sizeof kOsChoices[0]; ++i)
      std::printf("    %zu) %-18s kernel %s\n", i + 1, kOsChoices[i].label, kOsChoices[i].kernel);
    std::cout << "\n";
    int pick = std::atoi(ask("OS (number)", "1").c_str());
    if (pick >= 1 && pick <= static_cast<int>(sizeof kOsChoices / sizeof kOsChoices[0]))
      c.os = kOsChoices[pick - 1];
    c.kernel = ask("Kernel version", c.os.kernel);
    c.hostname = ask("Hostname", c.hostname);
    c.load = std::atof(ask("Simulated GPU load 0..1 (0 = idle)", "0").c_str());
  }
  if (c.kernel.empty()) c.kernel = c.os.kernel;
  if (c.count < 1) c.count = 1;

  vgpu::DeviceProfile profile = vgpu::load_gpu(c.gpu);
  if (c.vram_mb > 0) profile.vram_bytes = static_cast<uint64_t>(c.vram_mb) * 1024 * 1024;

  // ---- isolation: re-exec inside a user+mount namespace once ----
  if (c.isolate && stage2_session.empty()) {
    Session s = build_session(c, profile);
    std::vector<std::string> argv = {"unshare", "-r", "-m", exe_path(), "shell", "--stage2", s.dir,
                                     "--gpu", c.gpu, "--count", std::to_string(c.count),
                                     "--cuda", c.cuda, "--rocm", c.rocm, "--driver", c.driver,
                                     "--kernel", c.kernel, "--hostname", c.hostname,
                                     "--load", std::to_string(c.load),
                                     // Carry the OS choice across, or the
                                     // isolated stage would report the default
                                     // while the generated files say otherwise.
                                     "--os", std::string(c.os.id) + ":" + c.os.version_id,
                                     "--vram-mb", std::to_string(profile.vram_bytes / (1024 * 1024))};
    // -c has to survive the re-exec, or an isolated scripted session silently
    // becomes an interactive one and exits on the first EOF.
    if (!c.command.empty()) {
      argv.push_back("-c");
      argv.push_back(c.command);
    }
    std::vector<char*> cargv;
    for (auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);
    ::execvp("unshare", cargv.data());
    // unshare unavailable: fall through and run without isolation -- and
    // actually without it. This used to print that message and then carry on
    // with c.isolate still set, so the bind mounts below ran in the *host's*
    // mount namespace: `mount --make-rprivate /`, then /etc/os-release,
    // /proc/driver and /sys/class/drm overlaid for every process on the
    // machine. As a normal user those fail and nobody notices; as root, on a
    // box without unshare, they succeed.
    std::fprintf(stderr, "[vgpu] namespaces unavailable; continuing without /proc isolation\n");
    c.isolate = false;
    stage2_session = s.dir;
  }

  Session s;
  if (!stage2_session.empty()) {
    s.dir = stage2_session;
    s.bin = s.dir + "/bin";
    s.root = s.dir + "/root";
    s.telemetry = s.dir + "/telemetry";
    s.dmesg_path = s.dir + "/dmesg.log";
    s.pci_dump = s.dir + "/pci.dump";
  } else {
    s = build_session(c, profile);
  }

  bool isolated = false;
  if (c.isolate) {
    // Overlay the synthesized system files onto the real paths. Best-effort:
    // a failure just means those files stay as the host's.
    auto bind = [&](const std::string& src, const std::string& dst) {
      std::string cmd = "mount --bind '" + src + "' '" + dst + "' 2>/dev/null";
      return std::system(cmd.c_str()) == 0;
    };
    // Private first, and only then bind. This is what keeps the overlays
    // inside the session's own namespace; on a host whose / is a shared mount,
    // a bind made without it propagates out. If it cannot be done, the binds
    // are not attempted at all -- the host's files showing through is the
    // documented fallback, and overlaying the host's is not.
    bool ok = false;
    if (std::system("mount --make-rprivate / 2>/dev/null") == 0) {
      ok = bind(s.root + "/etc/os-release", "/etc/os-release");
    // procfs will not accept new entries, so overlay the whole /proc/driver
    // directory. That makes /proc/driver/nvidia/version -- which plenty of
    // tools and install scripts check -- appear for the simulated driver.
    ok = bind(s.root + "/proc/driver", "/proc/driver") || ok;
      ok = bind(s.root + "/sys/class/drm", "/sys/class/drm") || ok;
    }
    isolated = ok;
  }

  setenv("VGPU_TELEMETRY_PATH", s.telemetry.c_str(), 1);
  setenv("VGPU_GPU", c.gpu.c_str(), 1);
  setenv("VGPU_DEVICE_COUNT", std::to_string(c.count).c_str(), 1);
  setenv("VGPU_VRAM_MB", std::to_string(profile.vram_bytes / (1024 * 1024)).c_str(), 1);
  setenv("VGPU_DRIVER_VERSION", c.driver.c_str(), 1);
  setenv("VGPU_CUDA_VERSION", c.cuda.c_str(), 1);
  setenv("VGPU_SESSION", s.dir.c_str(), 1);

  // Hold the devices open for the whole session: this is what publishes
  // telemetry that nvidia-smi / rocm-smi read.
  vgpu::runtime::Runtime rt(profile, c.count);
  std::atomic<bool> stop{false};
  std::thread pump;
  if (c.load > 0) {
    pump = std::thread([&] {
      const auto slice = std::chrono::milliseconds(100);
      while (!stop) {
        for (int i = 0; i < c.count; ++i)
          rt.device(i).note_busy(std::chrono::duration<double>(slice).count() * c.load);
        std::this_thread::sleep_for(slice);
      }
    });
  }

  // The session's tools go first, and every CUDA bin directory a build system
  // might reach for goes right behind them. Tools routinely prepend the
  // toolkit's directory to whatever PATH they were given -- pantheon.py does
  // exactly that -- but only when it is not already there, so listing them all
  // here is what keeps the session's nvcc in front of the real one.
  std::string path = s.bin;
  for (const std::string& dir : cuda_bin_directories())
    if (path.find(dir) == std::string::npos) path += ":" + dir;
  path += ":" + std::string(std::getenv("PATH") ? std::getenv("PATH") : "/usr/bin:/bin");
  setenv("PATH", path.c_str(), 1);
  std::string shim = shim_dir();
  const char* old_ld = std::getenv("LD_LIBRARY_PATH");
  setenv("LD_LIBRARY_PATH", (shim + (old_ld ? std::string(":") + old_ld : "")).c_str(), 1);

  if (c.command.empty()) {
  std::printf("\n");
  std::printf("  Simulated machine ready\n");
  std::printf("    GPUs     : %d x %s (%s each)\n", c.count, profile.model.c_str(),
              human_vram(profile.vram_bytes).c_str());
  std::printf("    %-9s: %s\n", profile.vendor == "amd" ? "ROCm" : "CUDA",
              profile.vendor == "amd" ? c.rocm.c_str() : c.cuda.c_str());
  std::printf("    Driver   : %s\n", c.driver.c_str());
  std::printf("    OS       : %s (kernel %s)\n", c.os.pretty, c.kernel.c_str());
  std::printf("    Isolation: %s\n",
              isolated ? "on (/proc/driver/nvidia and /etc/os-release overlaid)"
                       : "off (session tools only)");
  std::printf("\n  Try: nvidia-smi | rocm-smi | rocm_agent_enumerator | lspci | dmesg | uname -a\n");
  std::printf("       vgpu smi --explain   (what is measured vs modelled)\n");
  std::printf("  Type 'exit' to end the session.\n\n");
  }
  std::fflush(stdout);

  const char* shell = std::getenv("SHELL");
  if (!shell || !*shell) shell = "/bin/bash";
  std::string label = std::to_string(c.count) + "x" + profile.model;
  std::string ps1 = "\\[\\e[36m\\](vgpu " + label + ")\\[\\e[0m\\] \\w\\$ ";
  setenv("VGPU_PS1", ps1.c_str(), 1);
  std::string rcfile = s.dir + "/bashrc";
  // The user's rc files often prepend a real CUDA lib directory, which would
  // shadow the shim and send programs to NVIDIA's cudart. Re-assert both
  // search paths *after* sourcing them so the session's libraries win.
  write_file(rcfile,
             "[ -f /etc/bash.bashrc ] && . /etc/bash.bashrc\n"
             "[ -f \"$HOME/.bashrc\" ] && . \"$HOME/.bashrc\"\n"
             "PS1=\"$VGPU_PS1\"\n"
             "export PATH=\"" + s.bin + ":$PATH\"\n"
             "export LD_LIBRARY_PATH=\"" + shim + "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"\n");

  pid_t pid = ::fork();
  if (pid == 0) {
    if (!c.command.empty()) {
      // Non-interactive: run one command in the same simulated machine and
      // exit with its status, so a build or a test suite can be driven from a
      // script rather than typed at the prompt.
      ::execl(shell, shell, "--rcfile", rcfile.c_str(), "-c", c.command.c_str(), nullptr);
      ::execl("/bin/sh", "sh", "-c", c.command.c_str(), nullptr);
      _exit(127);
    }
    ::execl(shell, shell, "--rcfile", rcfile.c_str(), "-i", nullptr);
    ::execl("/bin/sh", "sh", "-i", nullptr);
    _exit(127);
  }
  int status = 0;
  ::waitpid(pid, &status, 0);
  stop = true;
  if (pump.joinable()) pump.join();
  // Remove the session directory; otherwise every run leaves its generated
  // tools and system files behind in /tmp.
  // No shell: the path is ours and the prefix check stays, but removing a
  // directory does not need /bin/sh, quoting, or a return value to ignore.
  if (s.dir.rfind("/tmp/vgpu-session-", 0) == 0) {
    std::error_code ec;
    std::filesystem::remove_all(s.dir, ec);
  }
  if (c.command.empty())
    std::printf("\n  Session ended. Simulated %d x %s.\n", c.count, profile.model.c_str());
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
