// `vgpu test --matrix` — run one program against every device profile and
// compare what came back.
//
// The question it answers is the one a simulator is uniquely able to answer:
// does this program behave the same on an A100 as on a T4, and if not, which
// device made the difference? Renting nine GPUs to find that out costs more
// than the answer is usually worth, and half of them are hard to get hold of
// at all.
//
// Differences are the interesting output, not the failures. A kernel that
// launches on an H100 and is rejected on a T4 has found a real portability
// limit -- a block size the older part cannot hold, or a shared-memory request
// above what it has -- and it found it without the T4.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"

namespace {

struct Result {
  std::string profile;
  int exit_code = 0;
  std::string output;
  bool ran = false;
};

// Runs `argv` with VGPU_GPU set, capturing stdout and stderr together. Output
// is what gets compared, so both streams matter: a program that prints its
// answer on stdout and its complaint on stderr has told you two things.
Result run_one(const std::string& profile, const std::vector<std::string>& argv,
               const std::string& shim) {
  Result r;
  r.profile = profile;
  int fds[2];
  if (pipe(fds) != 0) return r;

  const pid_t pid = fork();
  if (pid < 0) {
    close(fds[0]);
    close(fds[1]);
    return r;
  }
  if (pid == 0) {
    close(fds[0]);
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);
    setenv("VGPU_GPU", profile.c_str(), 1);
    setenv("VGPU_QUIET", "1", 1);  // the banner names the device and would differ every time
    if (!shim.empty()) {
      const char* old = getenv("LD_LIBRARY_PATH");
      const std::string v = shim + (old && old[0] ? ":" + std::string(old) : "");
      setenv("LD_LIBRARY_PATH", v.c_str(), 1);
    }
    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);
    execvp(cargv[0], cargv.data());
    _exit(127);
  }

  close(fds[1]);
  char buf[4096];
  ssize_t n;
  while ((n = ::read(fds[0], buf, sizeof buf)) > 0) r.output.append(buf, static_cast<size_t>(n));
  close(fds[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  r.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  r.ran = true;
  return r;
}

std::string exe_dir_shim() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return "";
  buf[n] = '\0';
  std::string exe(buf);
  const size_t slash = exe.find_last_of('/');
  std::string dir = slash == std::string::npos ? std::string(".") : exe.substr(0, slash);
  if (dir.size() > 4 && dir.compare(dir.size() - 4, 4, "/bin") == 0)
    dir = dir.substr(0, dir.size() - 4);
  return dir + "/shim";
}

std::vector<std::string> split_commas(const std::string& v) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= v.size()) {
    const size_t c = v.find(',', start);
    std::string item = v.substr(start, c == std::string::npos ? std::string::npos : c - start);
    if (!item.empty()) out.push_back(item);
    if (c == std::string::npos) break;
    start = c + 1;
  }
  return out;
}

int usage(FILE* to) {
  std::fprintf(to,
               "Usage: vgpu test --matrix [options] <program> [args...]\n"
               "\n"
               "Runs the program once per device profile and compares the output. The\n"
               "first profile is the baseline; the rest are reported as identical to it\n"
               "or different from it.\n"
               "\n"
               "  --gpus <a,b,c>   Only these profiles (default: every verified NVIDIA one)\n"
               "  --all            Include unverified profiles too\n"
               "  --show-diff      Print the output of the first profile that differs\n");
  return to == stdout ? 0 : 2;
}

}  // namespace

int cmd_test(const std::vector<std::string>& args) {
  bool matrix = false, all = false, show_diff = false;
  std::vector<std::string> gpus;
  size_t i = 0;
  for (; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--help" || a == "-h") return usage(stdout);
    if (a == "--matrix") { matrix = true; continue; }
    if (a == "--all") { all = true; continue; }
    if (a == "--show-diff") { show_diff = true; continue; }
    if (a == "--gpus" && i + 1 < args.size()) { gpus = split_commas(args[++i]); continue; }
    if (a == "--") { ++i; break; }
    if (a.empty() || a[0] != '-') break;
    std::fprintf(stderr, "vgpu test: unknown option '%s'\n\n", a.c_str());
    return usage(stderr);
  }
  if (!matrix) {
    std::fprintf(stderr, "vgpu test: only --matrix is implemented\n\n");
    return usage(stderr);
  }
  if (i >= args.size()) {
    std::fprintf(stderr, "vgpu test: no program given\n\n");
    return usage(stderr);
  }
  const std::vector<std::string> program(args.begin() + i, args.end());

  if (gpus.empty()) {
    for (const auto& id : vgpu::available_gpus()) {
      const vgpu::DeviceProfile p = vgpu::load_gpu(id);
      // AMD profiles are excluded even though the interpreter executes 64-lane
      // wavefronts now. The reason changed: it is no longer that the engine
      // cannot run them, it is that the programs in this matrix are CUDA. What
      // a CUDA binary does on a 64-lane wavefront is not what an MI300X does
      // with it, so including the column would compare against a machine that
      // does not exist. Verified-only by default for the same reason a
      // placeholder is not evidence.
      if (p.vendor != "nvidia") continue;
      if (!all && !p.verified) continue;
      gpus.push_back(id);
    }
  }
  if (gpus.empty()) {
    std::fprintf(stderr, "vgpu test: no profiles selected\n");
    return 1;
  }

  const std::string shim = exe_dir_shim();
  std::vector<Result> results;
  results.reserve(gpus.size());
  for (const auto& g : gpus) {
    vgpu::load_gpu(g);  // fail on a bad name before running anything
    results.push_back(run_one(g, program, shim));
  }

  const Result& base = results.front();
  size_t same = 0, differ = 0;
  std::printf("%-24s %6s  %s\n", "profile", "exit", "output");
  for (const Result& r : results) {
    const bool identical = r.output == base.output && r.exit_code == base.exit_code;
    const char* verdict = &r == &base ? "baseline"
                          : identical ? "identical"
                                      : "DIFFERS";
    if (&r != &base) (identical ? same : differ)++;
    std::printf("%-24s %6d  %s\n", r.profile.c_str(), r.exit_code, verdict);
  }
  std::printf("\n%zu profiles: %zu identical to %s, %zu different\n", results.size(), same,
              base.profile.c_str(), differ);

  if (show_diff && differ) {
    for (const Result& r : results) {
      if (&r == &base) continue;
      if (r.output == base.output && r.exit_code == base.exit_code) continue;
      std::printf("\n--- %s (baseline) ---\n%s", base.profile.c_str(), base.output.c_str());
      std::printf("\n--- %s ---\n%s", r.profile.c_str(), r.output.c_str());
      break;
    }
  }
  // A difference is the finding, not an error: exit non-zero so a CI job can
  // gate on it, and say so plainly rather than calling it a failure.
  return differ ? 1 : 0;
}
