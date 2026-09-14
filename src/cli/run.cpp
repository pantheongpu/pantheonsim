// `vgpu run` — run an unmodified program against VirtualGPU.
//
// Everything this does can be done by hand:
//
//   VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH=/path/to/build/shim ./my_program
//
// The value here is not saving that line, it is the checks in front of it. The
// two ways a program fails to reach the simulator both look like a driver bug
// from the outside and neither produces a useful message:
//
//   * it needs a CUDA soname the shim does not carry (libcudart.so.12 against a
//     shim built for CUDA 13), so the loader finds NVIDIA's copy instead --
//     or nothing at all;
//   * it carries a DT_RPATH naming a directory that holds the real CUDA
//     libraries. DT_RPATH is the one search path the loader consults *before*
//     LD_LIBRARY_PATH, so the simulator is never reached however the
//     environment is set. (DT_RUNPATH, which is what a modern toolchain emits,
//     comes after LD_LIBRARY_PATH and is harmless -- the distinction is the
//     whole point of the newer tag.)
//
// Both are read straight out of the binary's dynamic section here, before
// anything is launched, and both come with the fix. Programs that pass go on to
// exec() in place, so exit codes and signals are the program's own.
#include <elf.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "args.hpp"
#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/telemetry.hpp"

namespace {

int fail(const char* fmt, const std::string& a) {
  std::fprintf(stderr, "vgpu run: ");
  std::fprintf(stderr, fmt, a.c_str());
  std::fprintf(stderr, "\n");
  return 2;
}

// ---------------------------------------------------------------------------
// What the binary asks the loader for
// ---------------------------------------------------------------------------

struct DynInfo {
  bool is_elf = false;         // false for scripts, wrappers, anything unreadable
  std::vector<std::string> needed;  // DT_NEEDED entries
  std::string runpath;              // DT_RUNPATH or DT_RPATH, whichever is set
  bool runpath_is_rpath = false;    // DT_RPATH is the older, stronger form
  // nvcc puts the embedded device code in a section of its own. Its presence is
  // what separates "a program with GPU kernels in it" from every other binary
  // on the system, which matters because the advice below is only for the
  // former -- /bin/sh also links no CUDA library, and does not want to hear
  // about it.
  bool has_gpu_code = false;
};

// Reads DT_NEEDED and DT_RUNPATH/DT_RPATH out of a 64-bit ELF. Deliberately
// in-process rather than shelling out to ldd, which *runs* the program to
// resolve its libraries -- the very thing this is checking whether to do.
DynInfo read_dynamic(const std::string& path) {
  DynInfo info;
  std::ifstream f(path, std::ios::binary);
  if (!f) return info;
  std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (blob.size() < sizeof(Elf64_Ehdr)) return info;
  const auto* eh = reinterpret_cast<const Elf64_Ehdr*>(blob.data());
  if (std::memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0) return info;
  if (eh->e_ident[EI_CLASS] != ELFCLASS64) return info;  // 32-bit CUDA is long gone
  info.is_elf = true;

  // Find PT_DYNAMIC, and the mapping from virtual address to file offset that
  // lets the string table be located.
  const auto* ph = reinterpret_cast<const Elf64_Phdr*>(blob.data() + eh->e_phoff);
  if (eh->e_phoff + static_cast<uint64_t>(eh->e_phnum) * sizeof(Elf64_Phdr) > blob.size())
    return info;

  // Section names, for the embedded-device-code check.
  if (eh->e_shoff && eh->e_shstrndx != SHN_UNDEF &&
      eh->e_shoff + static_cast<uint64_t>(eh->e_shnum) * sizeof(Elf64_Shdr) <= blob.size()) {
    const auto* sh = reinterpret_cast<const Elf64_Shdr*>(blob.data() + eh->e_shoff);
    const uint64_t names = sh[eh->e_shstrndx].sh_offset;
    for (uint16_t i = 0; i < eh->e_shnum && names < blob.size(); ++i) {
      const uint64_t at = names + sh[i].sh_name;
      if (at >= blob.size()) continue;
      const char* name = blob.data() + at;
      if (std::strncmp(name, ".nv_fatbin", 10) == 0 ||
          std::strncmp(name, ".nvFatBinSegment", 16) == 0)
        info.has_gpu_code = true;
    }
  }
  const Elf64_Phdr* dyn_ph = nullptr;
  for (uint16_t i = 0; i < eh->e_phnum; ++i)
    if (ph[i].p_type == PT_DYNAMIC) dyn_ph = &ph[i];
  if (!dyn_ph) return info;  // statically linked: nothing to intercept

  auto vaddr_to_off = [&](uint64_t va) -> uint64_t {
    for (uint16_t i = 0; i < eh->e_phnum; ++i)
      if (ph[i].p_type == PT_LOAD && va >= ph[i].p_vaddr && va < ph[i].p_vaddr + ph[i].p_filesz)
        return ph[i].p_offset + (va - ph[i].p_vaddr);
    return UINT64_MAX;
  };

  if (dyn_ph->p_offset + dyn_ph->p_filesz > blob.size()) return info;
  const auto* dyn = reinterpret_cast<const Elf64_Dyn*>(blob.data() + dyn_ph->p_offset);
  const size_t dyn_n = dyn_ph->p_filesz / sizeof(Elf64_Dyn);

  uint64_t strtab_off = UINT64_MAX;
  for (size_t i = 0; i < dyn_n && dyn[i].d_tag != DT_NULL; ++i)
    if (dyn[i].d_tag == DT_STRTAB) strtab_off = vaddr_to_off(dyn[i].d_un.d_ptr);
  if (strtab_off == UINT64_MAX || strtab_off >= blob.size()) return info;

  auto str_at = [&](uint64_t idx) -> std::string {
    const uint64_t at = strtab_off + idx;
    if (at >= blob.size()) return "";
    const size_t end = blob.find('\0', at);
    return blob.substr(at, (end == std::string::npos ? blob.size() : end) - at);
  };

  for (size_t i = 0; i < dyn_n && dyn[i].d_tag != DT_NULL; ++i) {
    if (dyn[i].d_tag == DT_NEEDED) info.needed.push_back(str_at(dyn[i].d_un.d_val));
    else if (dyn[i].d_tag == DT_RUNPATH) info.runpath = str_at(dyn[i].d_un.d_val);
    else if (dyn[i].d_tag == DT_RPATH && info.runpath.empty()) {
      info.runpath = str_at(dyn[i].d_un.d_val);
      info.runpath_is_rpath = true;
    }
  }
  return info;
}

// The CUDA libraries the shim stands in for. A DT_NEEDED entry that starts with
// one of these is something the simulator has to answer.
bool is_cuda_soname(const std::string& soname) {
  static const char* kPrefixes[] = {
      "libcuda.so",   "libcudart.so",  "libcublas.so", "libcublasLt.so", "libcudnn.so",
      "libcufft.so",  "libcurand.so",  "libcusparse.so", "libcusolver.so", "libnccl.so",
      "libnvrtc.so",  "libcupti.so",   "libnvidia-ml.so", "libnpp", "libnvjpeg.so",
      "libnvidia-encode.so"};
  for (const char* p : kPrefixes)
    if (soname.rfind(p, 0) == 0) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Locating things
// ---------------------------------------------------------------------------

std::string exe_dir() {
  char buf[4096];
  ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return ".";
  buf[n] = '\0';
  std::string exe(buf);
  size_t slash = exe.find_last_of('/');
  std::string dir = slash == std::string::npos ? std::string(".") : exe.substr(0, slash);
  // build/bin/vgpu -> build ; build/vgpu -> build
  if (dir.size() > 4 && dir.compare(dir.size() - 4, 4, "/bin") == 0)
    dir = dir.substr(0, dir.size() - 4);
  return dir;
}

bool is_file(const std::string& p) { return ::access(p.c_str(), F_OK) == 0; }

// An RPATH is a colon-separated list, like PATH.
std::vector<std::string> split_paths(const std::string& v) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= v.size()) {
    size_t colon = v.find(':', start);
    out.push_back(v.substr(start, colon == std::string::npos ? std::string::npos : colon - start));
    if (colon == std::string::npos) break;
    start = colon + 1;
  }
  return out;
}

// $ORIGIN is the directory of the binary itself, and relocatable builds use it
// constantly. Left unexpanded, every such RPATH would look like it points
// nowhere.
std::string expand_origin(const std::string& rpath, const std::string& program) {
  size_t slash = program.find_last_of('/');
  std::string dir = slash == std::string::npos ? std::string(".") : program.substr(0, slash);
  std::string out;
  for (size_t i = 0; i < rpath.size();) {
    if (rpath.compare(i, 7, "$ORIGIN") == 0) { out += dir; i += 7; }
    else if (rpath.compare(i, 9, "${ORIGIN}") == 0) { out += dir; i += 9; }
    else out += rpath[i++];
  }
  return out;
}

// Resolves a program name the way execvp will, so the checks below inspect the
// same file that is about to run.
std::string resolve_program(const std::string& name) {
  if (name.find('/') != std::string::npos) return name;
  const char* path = std::getenv("PATH");
  std::string p = path ? path : "/usr/bin:/bin";
  size_t start = 0;
  while (start <= p.size()) {
    size_t colon = p.find(':', start);
    std::string dir = p.substr(start, colon == std::string::npos ? std::string::npos : colon - start);
    if (dir.empty()) dir = ".";
    std::string cand = dir + "/" + name;
    if (::access(cand.c_str(), X_OK) == 0) return cand;
    if (colon == std::string::npos) break;
    start = colon + 1;
  }
  return name;  // let execvp produce the error
}

struct Options {
  std::string gpu;
  std::string count, vram_mb, threads, max_steps;
  bool strict = false, race = false, counters = false, trace = false, quiet = false;
  bool preload = false, print_env = false;
};

int usage(FILE* to) {
  std::fprintf(to,
               "Usage: vgpu run [options] <program> [args...]\n"
               "\n"
               "Runs a program against VirtualGPU: the simulator's CUDA libraries are put\n"
               "in front of the real ones and the program is exec'd in place, so its exit\n"
               "code and signals are its own. Everything after the program name belongs to\n"
               "the program; use -- to end the options early.\n"
               "\n"
               "Device:\n"
               "  --gpu <vendor/model>  Device to present (default: $VGPU_GPU, else nvidia/h100)\n"
               "  --count <n>           How many of them\n"
               "  --vram-mb <n>         VRAM per device, in MiB\n"
               "\n"
               "Execution:\n"
               "  --threads <n>         Interpreter worker threads, at least 1 (default: one per core)\n"
               "  --max-steps <n>       Per-thread instruction budget (0 disables the guard)\n"
               "  --strict              Refuse undefined behaviour instead of continuing\n"
               "  --race                Detect shared-memory data races\n"
               "  --counters            Collect per-kernel instruction and memory counters\n"
               "\n"
               "Output:\n"
               "  --trace               Log driver-level activity\n"
               "  --quiet               Silence every VirtualGPU diagnostic on stderr, errors and\n"
               "                        warnings included (sets VGPU_QUIET=1). The program's own\n"
               "                        output and exit code are unaffected.\n"
               "\n"
               "Loading:\n"
               "  --preload             Also LD_PRELOAD the shim. Needed when the program\n"
               "                        carries a RUNPATH, which the loader consults before\n"
               "                        LD_LIBRARY_PATH. 'vgpu run' says so when it applies.\n"
               "  --print-env           Print the environment and command, then exit\n");
  return to == stdout ? 0 : 2;
}

}  // namespace

int cmd_run(const std::vector<std::string>& args) {
  Options o;
  size_t i = 0;
  auto value = [&](const char* what) -> std::string {
    if (i + 1 >= args.size()) {
      std::fprintf(stderr, "vgpu run: %s needs a value\n", what);
      std::exit(2);
    }
    return args[++i];
  };
  for (; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--help" || a == "-h") return usage(stdout);
    if (a == "--") { ++i; break; }
    if (a.empty() || a[0] != '-') break;  // the program
    if (a == "--gpu") o.gpu = value("--gpu");
    else if (a == "--count") o.count = value("--count");
    else if (a == "--vram-mb") o.vram_mb = value("--vram-mb");
    else if (a == "--threads") o.threads = value("--threads");
    else if (a == "--max-steps") o.max_steps = value("--max-steps");
    else if (a == "--strict") o.strict = true;
    else if (a == "--race") o.race = true;
    else if (a == "--counters") o.counters = true;
    else if (a == "--trace") o.trace = true;
    else if (a == "--quiet") o.quiet = true;
    else if (a == "--preload") o.preload = true;
    else if (a == "--print-env") o.print_env = true;
    else {
      std::fprintf(stderr, "vgpu run: unknown option '%s'\n\n", a.c_str());
      return usage(stderr);
    }
  }
  // The values go into the environment as strings, and the libraries that read
  // them back are lenient on purpose (a bad VGPU_THREADS becomes 1, a bad
  // VGPU_VRAM_MB is ignored). That is right for an environment variable nobody
  // may be watching, and wrong for a flag someone just typed: `--vram-mb 1x`
  // ran with the full card and `--count 0` with one device, and neither said so.
  {
    long long n = 0;
    auto check = [&](const std::string& v, const char* flag, long long lo, long long hi,
                     const char* what) {
      if (v.empty() || vgpu::cli::parse_int(v, lo, hi, &n)) return;
      std::fprintf(stderr, "vgpu run: %s needs %s, got '%s'\n", flag, what, v.c_str());
      std::exit(2);
    };
    check(o.count, "--count", 1, vgpu::telemetry::kMaxDevices, "a whole number from 1 to 16");
    check(o.vram_mb, "--vram-mb", 1, 1ll << 30, "a positive whole number of MiB");
    check(o.threads, "--threads", 1, 1 << 16, "a whole number of at least 1");
    check(o.max_steps, "--max-steps", 0, LLONG_MAX, "a whole number of at least 0");
  }
  if (i >= args.size()) {
    std::fprintf(stderr, "vgpu run: no program given\n\n");
    return usage(stderr);
  }

  const std::string shim = exe_dir() + "/shim";
  if (!is_file(shim + "/libcuda.so.1"))
    return fail("no simulator libraries at %s. Build them first (cmake --build <dir>), "
                "and run the vgpu binary from that same build.",
                shim);

  // Validate the device model now rather than letting the program discover it,
  // so a typo is one clear line instead of a failure inside CUDA init.
  std::string gpu = o.gpu;
  if (gpu.empty()) {
    const char* e = std::getenv("VGPU_GPU");
    gpu = (e && e[0]) ? e : "nvidia/h100";
  }
  vgpu::load_gpu(gpu);  // throws with the list of known models

  const std::string program = resolve_program(args[i]);
  std::vector<std::string> argv_str(args.begin() + i, args.end());

  // ---- pre-flight: will the loader actually reach us? ----
  const DynInfo dyn = read_dynamic(program);
  std::vector<std::string> cuda_needs;
  for (const auto& n : dyn.needed)
    if (is_cuda_soname(n)) cuda_needs.push_back(n);

  bool advise_preload = false;
  for (const auto& n : cuda_needs) {
    if (is_file(shim + "/" + n)) continue;
    // A soname the shim does not carry. Almost always a major-version mismatch:
    // the program was built against CUDA 12 and the shim against 13. Silence
    // here would send the program to NVIDIA's library, or to no library at all.
    // Name the major the shim does carry, when there is one. "needs
    // libcudart.so.12, this has libcudart.so.13" says what the problem is;
    // "not provided" leaves the reader to guess.
    const std::string stem = n.substr(0, n.find(".so") + 3);
    std::string have;
    for (int major = 30; major >= 1; --major) {
      const std::string cand = stem + "." + std::to_string(major);
      if (is_file(shim + "/" + cand)) { have = cand; break; }
    }
    std::string detail = have.empty() ? std::string("this build of the simulator has no ") + stem
                                      : "this build of the simulator has " + have;
    std::fprintf(stderr,
                 "vgpu run: %s needs %s, but\n"
                 "          %s.\n"
                 "          Rebuild the program against the CUDA version the shim was built for,\n"
                 "          or rebuild the shim against the program's toolkit.\n",
                 program.c_str(), n.c_str(), detail.c_str());
    return 1;
  }

  // Only DT_RPATH outranks LD_LIBRARY_PATH; DT_RUNPATH is searched after it and
  // changes nothing here. And even an RPATH only matters if the directory it
  // names actually holds one of the libraries the program needs -- pointing at
  // an empty or unrelated directory is common and harmless, so the check is for
  // a real conflicting file rather than for the tag.
  if (dyn.runpath_is_rpath && !cuda_needs.empty()) {
    std::string shadowed;
    for (const auto& dir : split_paths(expand_origin(dyn.runpath, program))) {
      for (const auto& n : cuda_needs)
        if (is_file(dir + "/" + n)) { shadowed = dir + "/" + n; break; }
      if (!shadowed.empty()) break;
    }
    if (!shadowed.empty()) {
      advise_preload = true;
      if (!o.preload)
        std::fprintf(stderr,
                     "vgpu run: %s has a DT_RPATH holding\n"
                     "          %s\n"
                     "          DT_RPATH is the one search path the loader consults before\n"
                     "          LD_LIBRARY_PATH, so the real library wins and the simulator is never\n"
                     "          reached. Re-run with --preload, which is loaded ahead of every\n"
                     "          search.\n",
                     program.c_str(), shadowed.c_str());
    }
  }

  if (dyn.has_gpu_code && cuda_needs.empty()) {
    // The binary carries device code but names no CUDA library, which leaves
    // only one explanation: it links the runtime statically. That is NVIDIA's
    // own runtime inside the program, reaching the driver through undocumented
    // tables the simulator does not implement. Worth saying here, because the
    // failure it produces later ("integrity checks failed") points nowhere near
    // the cause.
    std::fprintf(stderr,
                 "vgpu run: %s embeds GPU code but links no CUDA library, so it was built with\n"
                 "          the static runtime -- which is NVIDIA's own runtime inside the binary\n"
                 "          and cannot run against a simulated driver. Rebuild with\n"
                 "          'nvcc -cudart shared', or build it inside 'vgpu shell', whose nvcc\n"
                 "          adds that flag for you.\n",
                 program.c_str());
  }

  // ---- environment ----
  std::vector<std::pair<std::string, std::string>> env;
  env.emplace_back("VGPU_GPU", gpu);
  if (!o.count.empty()) env.emplace_back("VGPU_DEVICE_COUNT", o.count);
  if (!o.vram_mb.empty()) env.emplace_back("VGPU_VRAM_MB", o.vram_mb);
  if (!o.threads.empty()) env.emplace_back("VGPU_THREADS", o.threads);
  if (!o.max_steps.empty()) env.emplace_back("VGPU_MAX_STEPS", o.max_steps);
  if (o.strict) env.emplace_back("VGPU_STRICT", "1");
  if (o.race) env.emplace_back("VGPU_RACE", "1");
  if (o.counters) env.emplace_back("VGPU_COUNTERS", "1");
  if (o.trace) env.emplace_back("VGPU_TRACE", "1");
  if (o.quiet) env.emplace_back("VGPU_QUIET", "1");

  const char* old_ld = std::getenv("LD_LIBRARY_PATH");
  env.emplace_back("LD_LIBRARY_PATH", shim + (old_ld && old_ld[0] ? ":" + std::string(old_ld) : ""));

  if (o.preload) {
    // Exactly the libraries the program already asked for, and no more. Two
    // reasons to be narrow: preloading the whole shim would pull cuDNN and NCCL
    // into every child process, and -- more importantly -- each shim library
    // carries its own copy of the simulator core rather than sharing one
    // through libcuda, so preloading a library the program never names would
    // stand up a second, disjoint set of virtual devices beside the real one.
    std::string pre;
    for (const auto& n : cuda_needs)
      if (is_file(shim + "/" + n)) pre += (pre.empty() ? "" : " ") + shim + "/" + n;
    // Nothing named statically: the program must be dlopen-ing the driver,
    // which is the one case where there is no DT_NEEDED to go by.
    if (pre.empty()) pre = shim + "/libcuda.so.1";
    const char* old_pre = std::getenv("LD_PRELOAD");
    env.emplace_back("LD_PRELOAD", pre + (old_pre && old_pre[0] ? " " + std::string(old_pre) : ""));
  }

  if (o.print_env) {
    for (const auto& [k, v] : env) std::printf("%s=%s\n", k.c_str(), v.c_str());
    std::printf("exec:");
    for (const auto& a : argv_str) std::printf(" %s", a.c_str());
    std::printf("\n");
    if (advise_preload && !o.preload)
      std::printf("note: the program has a RUNPATH; --preload may be required\n");
    return 0;
  }

  for (const auto& [k, v] : env) ::setenv(k.c_str(), v.c_str(), 1);

  std::vector<char*> argv;
  argv.reserve(argv_str.size() + 1);
  for (auto& a : argv_str) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  ::execvp(program.c_str(), argv.data());
  // Only reached when exec fails.
  std::fprintf(stderr, "vgpu run: cannot execute '%s': %s\n", program.c_str(), std::strerror(errno));
  return 127;
}
