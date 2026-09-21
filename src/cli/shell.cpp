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
//     re-executes inside `unshare -r -m -u`, bind-mounts the synthesized
//     /proc/driver/nvidia, /sys/class/drm and /etc/os-release over the real
//     ones and sets the hostname in the session's own UTS namespace, so even
//     programs that read those absolute paths or call gethostname() see the
//     simulated machine.
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

#include "args.hpp"
#include "vgpu/driver_version.hpp"
#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/telemetry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vgpu/telemetry.hpp"

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
  bool has_command = false;  // -c was given, even as '' (which runs nothing)
  int count = 1;
  long long vram_mb = 0;  // 0 = use the profile's own capacity
  // The versions the build's own runtime can run under, from the same source
  // nvidia-smi uses outside a session. These were fixed at CUDA 12.4 and driver
  // 550, which on a CUDA 13 build made the session's driver older than the
  // runtime shim it loads -- every framework checks exactly that and refuses to
  // start -- and made the session's nvcc claim release 12.4.
  std::string cuda = vgpu::cuda_version_string(vgpu::kDefaultDriverVersion);
  std::string rocm = "6.2.0";
  std::string driver = vgpu::kDefaultDriverRelease;
  OsChoice os = kOsChoices[0];
  std::string kernel;
  std::string hostname = "vgpu-sim";
  double load = 0.0;
  bool isolate = true;
};

// What each value must look like, checked the same way for a flag and for an
// answer at the prompt.
constexpr const char* kCountHint = "a whole number from 1 to 16";
constexpr const char* kVramHint = "a positive whole number of MiB";
constexpr const char* kCudaHint = "a CUDA version such as 13.0";
constexpr const char* kVersionHint = "a dotted numeric version such as 580.65.06";
constexpr const char* kLoadHint = "a number from 0 to 1";
constexpr const char* kHostnameHint = "a hostname of letters, digits, '-' and '.' (at most 64)";
constexpr const char* kKernelHint = "a kernel release with no spaces or quotes";
constexpr long long kMaxVramMb = 1ll << 30;

// Exactly "major.minor", the only form the driver accepts (vgpu/driver_version.hpp).
// Anything else used to be exported as given and then silently replaced by the
// default in every CUDA call, while the session banner showed what was typed.
bool valid_cuda(const std::string& v) {
  const size_t dot = v.find('.');
  long long major = 0, minor = 0;
  return dot != std::string::npos && v.find('.', dot + 1) == std::string::npos &&
         vgpu::cli::parse_int(v.substr(0, dot), 1, 99, &major) &&
         v.substr(dot + 1).find('-') == std::string::npos &&
         vgpu::cli::parse_int(v.substr(dot + 1), 0, 99, &minor);
}
bool valid_version(const std::string& v) { return vgpu::cli::is_dotted_version(v, 2); }
// sethostname() takes at most 64 bytes, and the name is written into the
// session's generated scripts, so it is held to the characters a hostname has.
bool valid_hostname(const std::string& v) {
  return !v.empty() && v.size() <= 64 && v[0] != '-' && v[0] != '.' &&
         v.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.") ==
             std::string::npos;
}
bool valid_kernel(const std::string& v) {
  return !v.empty() && v.size() <= 64 && v.find_first_of(" \t\r\n'\"\\$`") == std::string::npos;
}

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

// Asks until the answer passes `ok`. An answer that does not parse used to be
// taken as whatever atoi made of it -- "eight" GPUs was one -- so it is asked
// again instead. EOF answers with the default, which is always valid, so a
// closed stdin cannot loop here.
template <class Ok>
std::string ask_valid(const std::string& prompt, const std::string& def, const char* hint, Ok ok) {
  while (true) {
    std::string v = ask(prompt, def);
    if (ok(v)) return v;
    std::cout << "    '" << v << "' is not " << hint << "\n";
  }
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
    // No BAR assignment line: config space reports BAR0 as unassigned until
    // BARs are modelled, and a log claiming a mapping that config space denies
    // is a contradiction a bring-up script would trip on.
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
    // The UUID nvidia-smi and NVML report for this device, and no VBIOS, which
    // is also NVML's answer: this file used to carry its own version of both.
    vgpu::telemetry::DeviceSample ds{};
    vgpu::telemetry::describe_device(p, i, &ds);
    char info[512];
    std::snprintf(info, sizeof info,
                  "Model: \t\t %s\nIRQ:   \t\t %d\nGPU UUID: \t %s\n"
                  "Video BIOS: \t N/A\nBus Type: \t PCIe\n"
                  "DMA Size: \t 47 bits\nDMA Mask: \t 0x7fffffffffff\n"
                  "Bus Location: \t %s\nDevice Minor: \t %d\n",
                  p.model.c_str(), 128 + i, ds.uuid, bdf, i);
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
  // --version too: `vgpu smi` prints it from VGPU_DRIVER_VERSION and
  // VGPU_CUDA_VERSION, which the session exports, so this and
  // nvidia/tools/nvidia-smi cannot drift apart again.
  tool("nvidia-smi", "# VirtualGPU session tool.\nexec \"" + vgpu + "\" smi \"$@\"\n");
  tool("rocm-smi", "exec \"" + vgpu + "\" smi --rocm \"$@\"\n");
  tool("rocm_agent_enumerator", "exec \"" + vgpu + "\" smi --agents\n");
  tool("dmesg",
       "# Replays this session's synthetic kernel ring buffer.\n"
       "case \" $* \" in\n"
       "  *\" -w \"*|*\" --follow \"*) cat \"" + s.dmesg_path + "\"; tail -f /dev/null ;;\n"
       "esac\n"
       "cat \"" + s.dmesg_path + "\"\n");
  // uname answers every field itself, in the order and combinations coreutils
  // accepts. It used to look only at $1, so `uname -n` printed the host's name,
  // `uname -sr` the host's kernel, and `uname -r -s` just the release -- the
  // simulated identity held only for the exact spelling `uname -a`. The kernel
  // and hostname were validated when parsed, so single quotes hold them safely.
  const std::string real_uname = find_program("uname");
  tool("uname",
       "# Reports the simulated kernel and hostname. --help and --version go to the\n"
       "# real uname; every other flag is answered here, alone or combined (-sr).\n"
       "K='" + c.kernel + "'\nN='" + c.hostname + "'\nREAL='" + real_uname + "'\n" +
       R"SH(s= n= r= v= m= p= i= o=
bad() { echo "uname: $1" >&2; echo "Try 'uname --help' for more information." >&2; exit 1; }
for a in "$@"; do
  case "$a" in
    --help|--version) [ -x "$REAL" ] && exec "$REAL" "$a"; bad "no real uname for $a" ;;
    --all) s=1 n=1 r=1 v=1 m=1 p=1 i=1 o=1 ;;
    --kernel-name) s=1 ;;
    --nodename) n=1 ;;
    --kernel-release) r=1 ;;
    --kernel-version) v=1 ;;
    --machine) m=1 ;;
    --processor) p=1 ;;
    --hardware-platform) i=1 ;;
    --operating-system) o=1 ;;
    --*) bad "unrecognized option '$a'" ;;
    -?*)
      f="${a#-}"
      while [ -n "$f" ]; do
        case "${f:0:1}" in
          a) s=1 n=1 r=1 v=1 m=1 p=1 i=1 o=1 ;;
          s) s=1 ;; n) n=1 ;; r) r=1 ;; v) v=1 ;; m) m=1 ;; p) p=1 ;; i) i=1 ;; o) o=1 ;;
          *) bad "invalid option -- '${f:0:1}'" ;;
        esac
        f="${f:1}"
      done ;;
    *) bad "extra operand '$a'" ;;
  esac
done
[ -z "$s$n$r$v$m$p$i$o" ] && s=1
out=()
[ -n "$s" ] && out+=(Linux)
[ -n "$n" ] && out+=("$N")
[ -n "$r" ] && out+=("$K")
[ -n "$v" ] && out+=("#1 SMP")
[ -n "$m" ] && out+=(x86_64)
[ -n "$p" ] && out+=(x86_64)
[ -n "$i" ] && out+=(x86_64)
[ -n "$o" ] && out+=(GNU/Linux)
echo "${out[*]}"
)SH");
  // An isolated session sets the name in its own UTS namespace, so the kernel
  // already agrees; with --no-isolate (or no unshare) only this script does.
  // Setting a name, or asking for addresses, goes to the real command.
  const std::string real_hostname = find_program("hostname");
  tool("hostname",
       "# Prints the session's configured hostname.\n"
       "N='" + c.hostname + "'\nREAL='" + real_hostname + "'\n" +
       R"SH(if [ $# -le 1 ]; then
  case "${1:-}" in
    "") echo "$N"; exit 0 ;;
    -s|--short) echo "${N%%.*}"; exit 0 ;;
    -f|--fqdn|--long) echo "$N"; exit 0 ;;
  esac
fi
[ -x "$REAL" ] && exec "$REAL" "$@"
echo "hostname: no real hostname command to run: hostname $*" >&2
exit 1
)SH");
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
  // -arch=native, resolved to the card this session has. The real nvcc asks
  // the driver and gets that answer too -- then builds for it as machine code
  // alone (code=sm_75, no PTX), which a GPU runs and this simulator cannot, so
  // the most common line in current tutorials built a binary whose every
  // launch was refused. -arch=sm_XX is the same target with PTX embedded.
  const std::string native_arch =
      p.vendor == "nvidia" && p.cc_major > 0
          ? "sm_" + std::to_string(p.cc_major) + std::to_string(p.cc_minor)
          : "";
  const std::string native_rewrite =
      native_arch.empty()
          ? ""
          : "prev=\n"
            "for a do\n"
            "  shift\n"
            "  case \"$a\" in\n"
            "    -arch=native|--gpu-architecture=native) a=\"${a%%=*}=" + native_arch + "\" ;;\n"
            "    native) case \"$prev\" in -arch|--gpu-architecture) a=" + native_arch + " ;; esac ;;\n"
            "  esac\n"
            "  set -- \"$@\" \"$a\"; prev=\"$a\"\n"
            "done\n";
  tool("nvcc",
       "# The session compiler. VGPU_NVCC_PASSTHROUGH=1 removes the -cudart flag.\n"
       "if [ \"${1:-}\" = \"--version\" ]; then\n"
       "  echo 'nvcc: NVIDIA (R) Cuda compiler driver'\n"
       "  echo 'Cuda compilation tools, release " + c.cuda + " (VirtualGPU session)'\n"
       "  exit 0\nfi\n"
       "REAL='" + real_nvcc + "'\n"
       "[ -x \"$REAL\" ] || { echo 'nvcc: no CUDA toolkit in this session' >&2; exit 127; }\n"
       "[ \"${VGPU_NVCC_PASSTHROUGH:-0}\" = 1 ] && exec \"$REAL\" \"$@\"\n" +
       native_rewrite +
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
    // A flag with nothing after it used to read as "": `-c` at the end became
    // an interactive shell reading stdin -- a CI job hung instead of failing --
    // and `--gpu` at the end asked for a profile named ''.
    auto next = [&]() -> std::string {
      if (i + 1 >= args.size()) {
        std::fprintf(stderr, "vgpu shell: %s needs a value\n", a.c_str());
        std::exit(2);
      }
      return args[++i];
    };
    // Numbers went through atoi/atof: `--count 0` and `--count abc` both gave
    // one GPU, `--vram-mb 1x` a 1 MiB card, `--vram-mb -1` the full one.
    auto checked = [&](bool (*ok)(const std::string&), const char* hint) -> std::string {
      std::string v = next();
      if (!ok(v)) {
        std::fprintf(stderr, "vgpu shell: %s needs %s, got '%s'\n", a.c_str(), hint, v.c_str());
        std::exit(2);
      }
      return v;
    };
    long long n = 0;
    if (a == "--gpu") c.gpu = next();
    else if (a == "--count") {
      const std::string v = next();
      if (!vgpu::cli::parse_int(v, 1, vgpu::telemetry::kMaxDevices, &n)) {
        std::fprintf(stderr, "vgpu shell: --count needs %s, got '%s'\n", kCountHint, v.c_str());
        return 2;
      }
      c.count = static_cast<int>(n);
    } else if (a == "--vram-mb") {
      const std::string v = next();
      if (!vgpu::cli::parse_int(v, 1, kMaxVramMb, &c.vram_mb)) {
        std::fprintf(stderr, "vgpu shell: --vram-mb needs %s, got '%s'\n", kVramHint, v.c_str());
        return 2;
      }
    } else if (a == "--cuda") c.cuda = checked(valid_cuda, kCudaHint);
    else if (a == "--rocm") c.rocm = checked(valid_version, kVersionHint);
    else if (a == "--driver") c.driver = checked(valid_version, kVersionHint);
    else if (a == "--kernel") c.kernel = checked(valid_kernel, kKernelHint);
    else if (a == "--hostname") c.hostname = checked(valid_hostname, kHostnameHint);
    else if (a == "--load") {
      const std::string v = next();
      if (!vgpu::cli::parse_double(v, 0.0, 1.0, &c.load)) {
        std::fprintf(stderr, "vgpu shell: --load needs %s, got '%s'\n", kLoadHint, v.c_str());
        return 2;
      }
    }
    else if (a == "--no-isolate") c.isolate = false;
    else if (a == "-c" || a == "--command") {
      c.command = next();
      c.has_command = true;
    }
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
  if (c.has_command || !::isatty(STDIN_FILENO)) prompt = false;

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
    long long n = 0;
    ask_valid("How many GPUs", std::to_string(c.count), kCountHint, [&](const std::string& v) {
      return vgpu::cli::parse_int(v, 1, vgpu::telemetry::kMaxDevices, &n);
    });
    c.count = static_cast<int>(n);
    vgpu::DeviceProfile probe = vgpu::load_gpu(c.gpu);
    bool amd = probe.vendor == "amd";
    ask_valid("VRAM per GPU (MiB)", std::to_string(probe.vram_bytes / (1024 * 1024)), kVramHint,
              [&](const std::string& v) { return vgpu::cli::parse_int(v, 1, kMaxVramMb, &c.vram_mb); });
    if (amd)
      c.rocm = ask_valid("ROCm version", c.rocm, kVersionHint, valid_version);
    else
      c.cuda = ask_valid("CUDA version", c.cuda, kCudaHint, valid_cuda);
    c.driver = ask_valid("Driver version", c.driver, kVersionHint, valid_version);
    std::cout << "\n  Operating systems:\n";
    const long long os_count = static_cast<long long>(sizeof kOsChoices / sizeof kOsChoices[0]);
    for (long long i = 0; i < os_count; ++i)
      std::printf("    %lld) %-18s kernel %s\n", i + 1, kOsChoices[i].label, kOsChoices[i].kernel);
    std::cout << "\n";
    long long pick = 1;
    ask_valid("OS (number)", "1", "one of the numbers listed", [&](const std::string& v) {
      return vgpu::cli::parse_int(v, 1, os_count, &pick);
    });
    c.os = kOsChoices[pick - 1];
    c.kernel = ask_valid("Kernel version", c.os.kernel, kKernelHint, valid_kernel);
    c.hostname = ask_valid("Hostname", c.hostname, kHostnameHint, valid_hostname);
    ask_valid("Simulated GPU load 0..1 (0 = idle)", "0", kLoadHint,
              [&](const std::string& v) { return vgpu::cli::parse_double(v, 0.0, 1.0, &c.load); });
  }
  if (c.kernel.empty()) c.kernel = c.os.kernel;

  vgpu::DeviceProfile profile = vgpu::load_gpu(c.gpu);
  if (c.vram_mb > 0) profile.vram_bytes = static_cast<uint64_t>(c.vram_mb) * 1024 * 1024;

  // ---- isolation: re-exec inside a user+mount namespace once ----
  if (c.isolate && stage2_session.empty()) {
    Session s = build_session(c, profile);
    // -u gives the session its own UTS namespace, so the hostname can be set
    // below without touching the host's. Without it --hostname reached only the
    // generated uname script; `hostname`, `uname -n` and every program calling
    // gethostname() still named the host.
    std::vector<std::string> argv = {"unshare", "-r", "-m", "-u", exe_path(), "shell", "--stage2", s.dir,
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
    if (c.has_command) {
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

  bool isolated = false, renamed = false;
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
    // The UTS namespace from `unshare -u` belongs to this session, so the
    // kernel's own answer -- hostname, uname -n, gethostname() in any program
    // -- becomes the configured name. Best-effort like the mounts: on failure
    // the session tools still print it.
    renamed = ::sethostname(c.hostname.c_str(), c.hostname.size()) == 0;
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

  if (!c.has_command) {
  std::printf("\n");
  std::printf("  Simulated machine ready\n");
  std::printf("    GPUs     : %d x %s (%s each)\n", c.count, profile.model.c_str(),
              human_vram(profile.vram_bytes).c_str());
  std::printf("    %-9s: %s\n", profile.vendor == "amd" ? "ROCm" : "CUDA",
              profile.vendor == "amd" ? c.rocm.c_str() : c.cuda.c_str());
  std::printf("    Driver   : %s\n", c.driver.c_str());
  std::printf("    OS       : %s (kernel %s)\n", c.os.pretty, c.kernel.c_str());
  std::printf("    Hostname : %s\n", c.hostname.c_str());
  std::printf("    Isolation: %s\n",
              isolated ? (renamed ? "on (/proc/driver/nvidia and /etc/os-release overlaid, hostname set)"
                                  : "on (/proc/driver/nvidia and /etc/os-release overlaid)")
                       : renamed ? "partial (hostname set; /proc and /etc left as the host's)"
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

  // --rcfile is a bash option. Handed to dash it is "Illegal option --", so
  // with SHELL=/bin/sh -- the default in many CI images -- not even `-c` ran.
  // Other shells go without it: they read their own startup files, and PATH
  // and LD_LIBRARY_PATH are already in the environment they inherit.
  const std::string shell_path(shell);
  const bool is_bash = shell_path.substr(shell_path.find_last_of('/') + 1) == "bash";

  pid_t pid = ::fork();
  if (pid == 0) {
    if (c.has_command) {
      // Non-interactive: run one command in the same simulated machine and
      // exit with its status, so a build or a test suite can be driven from a
      // script rather than typed at the prompt.
      if (is_bash)
        ::execl(shell, shell, "--rcfile", rcfile.c_str(), "-c", c.command.c_str(), nullptr);
      else
        ::execl(shell, shell, "-c", c.command.c_str(), nullptr);
      ::execl("/bin/sh", "sh", "-c", c.command.c_str(), nullptr);
      _exit(127);
    }
    if (is_bash)
      ::execl(shell, shell, "--rcfile", rcfile.c_str(), "-i", nullptr);
    else
      ::execl(shell, shell, "-i", nullptr);
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
  if (!c.has_command)
    std::printf("\n  Session ended. Simulated %d x %s.\n", c.count, profile.model.c_str());
  // A shell killed by a signal reports 128+N, as every shell and `vgpu run`
  // do. This returned 1, so `kill -9` looked like an ordinary failure.
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return 1;
}
