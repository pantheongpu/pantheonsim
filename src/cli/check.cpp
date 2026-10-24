// `vgpu check` — run a program on the real GPU of this machine and on the simulator, and compare what it printed.
//
// The simulator's claim is that it computes what the card computes. This is the command that tests the claim on one program: the same
// binary, once with NVIDIA's own libraries (the card) and once with the simulator's libraries in front, with the simulated profile
// equal to the card. It changes nothing in the simulator; it only runs it. See docs/perf-design/08-host-gpu-opt-in-scope.md (Phase 0).
//
// A comparison is only worth reading if the card agrees with itself, so the program is run on the card twice and lines that differ
// between those two runs are set aside before the simulator is compared against either. What a comparison drops (timing lines) and
// sets aside (lines that name the machine) is counted and can be listed with --show-ignored.
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "check_compare.hpp"
#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"

extern char** environ;

namespace {

using namespace vgpu::check;

// Documented in usage(). 3 is a finding; 4 and 5 and 6 say why there is no finding.
constexpr int kExitDiffer = 3;
constexpr int kExitNativeDidNotRun = 4;
constexpr int kExitInconclusive = 5;
constexpr int kExitNoCard = 6;

struct Run {
  bool started = false;
  int exec_errno = 0;
  int signal = 0;
  bool timed_out = false;
  int exit_code = 0;
  std::string output;
  double seconds = 0;
  std::vector<std::pair<std::string, std::string>> files;  // --file path, contents ("" if absent)
};

std::string exe_path() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
  return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

std::string dir_of(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? "." : p.substr(0, s);
}

std::string read_file(const std::string& path, bool* exists = nullptr) {
  std::ifstream f(path, std::ios::binary);
  if (exists) *exists = static_cast<bool>(f);
  if (!f) return {};
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// The environment for the native side: the real libraries, nothing of the simulator. A VGPU_* variable would reach the shim if
// the program loaded one, and the simulator's library directory in LD_LIBRARY_PATH would put it in front of NVIDIA's.
std::vector<std::string> native_env(const std::string& shim_dir) {
  std::vector<std::string> env;
  for (char** e = environ; e && *e; ++e) {
    std::string kv = *e;
    if (kv.rfind("VGPU_", 0) == 0) continue;
    if (kv.rfind("LD_LIBRARY_PATH=", 0) == 0) {
      std::string out = "LD_LIBRARY_PATH=";
      std::string rest = kv.substr(out.size());
      size_t s = 0;
      bool first = true;
      while (s <= rest.size()) {
        const size_t c = rest.find(':', s);
        const std::string item = rest.substr(s, c == std::string::npos ? std::string::npos : c - s);
        if (!item.empty() && item != shim_dir) {
          if (!first) out += ':';
          out += item;
          first = false;
        }
        if (c == std::string::npos) break;
        s = c + 1;
      }
      if (first) continue;  // nothing left
      kv = out;
    }
    env.push_back(kv);
  }
  return env;
}

// Runs `argv` (with `env` replacing the environment when given), stdout and stderr together, killing the process group at the timeout.
Run run_capture(const std::vector<std::string>& argv, const std::vector<std::string>* env, int timeout_s,
                const std::vector<std::string>& files) {
  Run r;
  for (const auto& f : files) {
    ::unlink(f.c_str());
    r.files.push_back({f, ""});
  }
  int fds[2];
  if (pipe(fds) != 0) return r;
  int errfds[2];
  if (pipe2(errfds, O_CLOEXEC) != 0) {
    close(fds[0]);
    close(fds[1]);
    return r;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const pid_t pid = fork();
  if (pid < 0) {
    close(fds[0]); close(fds[1]); close(errfds[0]); close(errfds[1]);
    return r;
  }
  if (pid == 0) {
    setsid();  // so a timeout can kill the program's children too
    close(fds[0]);
    close(errfds[0]);
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);
    std::vector<char*> cargv;
    for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);
    if (env) {
      std::vector<char*> cenv;
      for (const auto& e : *env) cenv.push_back(const_cast<char*>(e.c_str()));
      cenv.push_back(nullptr);
      execvpe(cargv[0], cargv.data(), cenv.data());
    } else {
      execvp(cargv[0], cargv.data());
    }
    const int e = errno;
    if (::write(errfds[1], &e, sizeof e) < 0) {}
    _exit(127);
  }
  close(fds[1]);
  close(errfds[1]);
  r.started = true;
  char buf[4096];
  for (;;) {
    const auto left = std::chrono::seconds(timeout_s) - (std::chrono::steady_clock::now() - t0);
    if (left <= std::chrono::seconds(0)) {
      r.timed_out = true;
      kill(-pid, SIGKILL);
      break;
    }
    pollfd p{fds[0], POLLIN, 0};
    const int ms = static_cast<int>(std::min<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(left).count(), 1000));
    const int pr = poll(&p, 1, ms);
    if (pr < 0 && errno == EINTR) continue;
    if (pr <= 0) continue;
    const ssize_t n = ::read(fds[0], buf, sizeof buf);
    if (n > 0) { r.output.append(buf, static_cast<size_t>(n)); continue; }
    if (n == 0 || (n < 0 && errno != EINTR)) break;
  }
  close(fds[0]);
  int e = 0;
  if (::read(errfds[0], &e, sizeof e) == static_cast<ssize_t>(sizeof e)) r.exec_errno = e;
  close(errfds[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (WIFSIGNALED(status)) r.signal = WTERMSIG(status);
  r.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  for (auto& f : r.files) f.second = read_file(f.first);
  return r;
}

// Why a run says nothing about the program, or "". As in `vgpu test --matrix`: 126 and 127 are also what a wrapper returns.
std::string did_not_run(const Run& r) {
  if (!r.started) return "could not be started";
  if (r.exec_errno) return std::string("could not be executed: ") + std::strerror(r.exec_errno);
  if (r.timed_out) return "timed out";
  if (r.signal) return std::string("was killed by signal ") + std::to_string(r.signal) + " (" + strsignal(r.signal) + ")";
  if (r.exit_code == 127) return "exited 127 (command not found)";
  if (r.exit_code == 126) return "exited 126 (found but not executable)";
  return "";
}

struct Card {
  std::string name;  // as nvidia-smi says
  int cc_major = 0, cc_minor = 0;
};

std::string normalize(std::string s) {
  s = vgpu::check::detail::lower(s);
  for (const char* w : {"nvidia ", "geforce ", "tesla "}) {
    const size_t p = s.find(w);
    if (p == 0) s.erase(0, std::strlen(w));
  }
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
  return s;
}

// nvidia-smi from the real driver: not the vgpu-provided one that may sit in this build's bin directory.
std::string find_real_nvidia_smi(const std::string& self_dir) {
  std::vector<std::string> dirs = {"/usr/bin", "/usr/local/bin", "/usr/local/nvidia/bin", "/usr/lib/wsl/lib"};
  if (const char* path = getenv("PATH")) {
    std::stringstream ss(path);
    std::string d;
    while (std::getline(ss, d, ':'))
      if (!d.empty() && d != self_dir && d != self_dir + "/bin") dirs.push_back(d);
  }
  for (const auto& d : dirs) {
    const std::string p = d + "/nvidia-smi";
    if (::access(p.c_str(), X_OK) == 0) return p;
  }
  return "";
}

std::optional<Card> detect_card(const std::string& self_dir, const std::string& override_name) {
  Card c;
  if (!override_name.empty()) { c.name = override_name; return c; }
  const std::string smi = find_real_nvidia_smi(self_dir);
  if (smi.empty()) return std::nullopt;
  const std::vector<std::string> env = native_env(self_dir + "/shim");
  const Run r = run_capture({smi, "--query-gpu=name,compute_cap", "--format=csv,noheader,nounits", "-i", "0"}, &env, 30, {});
  if (!r.started || r.exit_code != 0 || r.output.empty()) return std::nullopt;
  std::string line = r.output.substr(0, r.output.find('\n'));
  const size_t comma = line.rfind(',');
  if (comma == std::string::npos) return std::nullopt;
  c.name = line.substr(0, comma);
  while (!c.name.empty() && c.name.back() == ' ') c.name.pop_back();
  while (!c.name.empty() && c.name.front() == ' ') c.name.erase(0, 1);
  const std::string cc = line.substr(comma + 1);
  std::sscanf(cc.c_str(), " %d.%d", &c.cc_major, &c.cc_minor);
  return c;
}

int usage(FILE* to) {
  std::fprintf(to,
               "Usage: vgpu check [options] <program> [args...]\n"
               "\n"
               "Runs the program on this machine's real NVIDIA GPU (NVIDIA's own libraries) and on the\n"
               "simulator with the profile of that same card, and compares what it printed. The program\n"
               "must be built with the shared runtime (nvcc -cudart shared) so that `vgpu run` can put the\n"
               "simulator's libcudart in front of the real one. It runs on the card twice first: lines\n"
               "that differ between those two runs are the card's own noise and are not held against the\n"
               "simulator. Timing lines are dropped; lines naming the machine are compared separately.\n"
               "\n"
               "  --gpu <vendor/model>  The profile to simulate. Default: the one matching the card\n"
               "  --allow-cross         Compare even when the profile is not the card (two different machines)\n"
               "  --card <name>         The card's name when nvidia-smi is not usable\n"
               "  --native-runs <n>     Runs on the card (default 2; 1 skips the noise check)\n"
               "  --sim-runs <n>        Runs on the simulator (default 1; 2 also checks it repeats)\n"
               "  --timeout <seconds>   Per run (default 600)\n"
               "  --tolerance <rel>     Accept numbers within this relative difference (default 0: exact)\n"
               "  --file <path>         Also compare this file the program writes (repeatable; bytes)\n"
               "  --show-diff           Print the first differing pair of lines\n"
               "  --show-ignored        List the lines dropped as timing and the lines set aside as identity\n"
               "\n"
               "Exit status:\n"
               "  0  the simulator agrees with the card (identical, or only timing and identity lines differ,\n"
               "     or numbers differ within --tolerance)\n"
               "  1  vgpu itself failed\n"
               "  2  usage error\n"
               "  3  the outputs, exit codes or files differ\n"
               "  4  the program did not run on the card (not found, not executable, exit 126 or 127,\n"
               "     killed, or timed out), so there is nothing to compare\n"
               "  5  inconclusive: the card's two runs cannot be lined up (different number of lines)\n"
               "  6  no usable card: none found, or the profile is not that card (see --allow-cross)\n");
  return to == stdout ? 0 : 2;
}

bool parse_int(const std::string& s, int* out) {
  char* end = nullptr;
  const long v = std::strtol(s.c_str(), &end, 10);
  if (s.empty() || *end || v < 1 || v > 100000) return false;
  *out = static_cast<int>(v);
  return true;
}

}  // namespace

int cmd_check(const std::vector<std::string>& args) {
  std::string gpu, card_override;
  bool allow_cross = false, show_diff = false, show_ignored = false;
  int native_runs = 2, sim_runs = 1, timeout_s = 600;
  double tolerance = 0;
  std::vector<std::string> files;
  size_t i = 0;
  auto need = [&](const char* opt) -> const std::string* {
    if (i + 1 >= args.size()) {
      std::fprintf(stderr, "vgpu check: %s needs a value\n\n", opt);
      return nullptr;
    }
    return &args[++i];
  };
  for (; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--help" || a == "-h") return usage(stdout);
    if (a == "--allow-cross") { allow_cross = true; continue; }
    if (a == "--show-diff") { show_diff = true; continue; }
    if (a == "--show-ignored") { show_ignored = true; continue; }
    if (a == "--gpu" || a == "--card" || a == "--file" || a == "--native-runs" || a == "--sim-runs" || a == "--timeout" ||
        a == "--tolerance") {
      const std::string* v = need(a.c_str());
      if (!v) return usage(stderr);
      if (a == "--gpu") gpu = *v;
      else if (a == "--card") card_override = *v;
      else if (a == "--file") files.push_back(*v);
      else if (a == "--tolerance") {
        char* end = nullptr;
        tolerance = std::strtod(v->c_str(), &end);
        if (v->empty() || *end || !(tolerance >= 0)) {
          std::fprintf(stderr, "vgpu check: --tolerance needs a number >= 0\n\n");
          return usage(stderr);
        }
      } else {
        int n = 0;
        if (!parse_int(*v, &n)) {
          std::fprintf(stderr, "vgpu check: %s needs a whole number >= 1\n\n", a.c_str());
          return usage(stderr);
        }
        (a == "--native-runs" ? native_runs : a == "--sim-runs" ? sim_runs : timeout_s) = n;
      }
      continue;
    }
    if (a == "--") { ++i; break; }
    if (a.empty() || a[0] != '-') break;
    std::fprintf(stderr, "vgpu check: unknown option '%s'\n\n", a.c_str());
    return usage(stderr);
  }
  if (i >= args.size()) {
    std::fprintf(stderr, "vgpu check: no program given\n\n");
    return usage(stderr);
  }
  const std::vector<std::string> program(args.begin() + i, args.end());

  const std::string self = exe_path();
  if (self.empty()) {
    std::fprintf(stderr, "vgpu check: cannot find the vgpu executable\n");
    return 1;
  }
  std::string self_dir = dir_of(self);
  if (self_dir.size() > 4 && self_dir.compare(self_dir.size() - 4, 4, "/bin") == 0) self_dir.resize(self_dir.size() - 4);

  // The card, and the profile that is that card. A profile that is not the card compares two different machines, and what differs
  // is then the machines, not the simulator.
  const std::optional<Card> card = detect_card(self_dir, card_override);
  if (!card) {
    std::fprintf(stderr, "vgpu check: no NVIDIA card found (nvidia-smi is missing or failed). Use --card <name> if it is there.\n");
    return kExitNoCard;
  }
  std::string profile;
  vgpu::DeviceProfile prof;
  if (!gpu.empty()) {
    prof = vgpu::load_gpu(gpu);
    profile = gpu;
  } else {
    for (const auto& id : vgpu::available_gpus()) {
      const vgpu::DeviceProfile p = vgpu::load_gpu(id);
      if (p.vendor != "nvidia") continue;
      if (normalize(p.model) == normalize(card->name)) { prof = p; profile = id; break; }
    }
    if (profile.empty()) {
      std::fprintf(stderr, "vgpu check: the card is '%s' and no profile has that name; pass --gpu <profile> (see vgpu list-gpus)\n",
                   card->name.c_str());
      return kExitNoCard;
    }
  }
  const bool same_card = normalize(prof.model) == normalize(card->name) &&
                         (card->cc_major == 0 || (prof.cc_major == card->cc_major && prof.cc_minor == card->cc_minor));
  if (!same_card && !allow_cross) {
    std::fprintf(stderr,
                 "vgpu check: profile %s (%s, sm_%d%d) is not the card (%s%s). The two would be different machines; use "
                 "--allow-cross to compare anyway.\n",
                 profile.c_str(), prof.model.c_str(), prof.cc_major, prof.cc_minor, card->name.c_str(),
                 card->cc_major ? (", sm_" + std::to_string(card->cc_major) + std::to_string(card->cc_minor)).c_str() : "");
    return kExitNoCard;
  }

  const std::vector<std::string> nenv = native_env(self_dir + "/shim");
  std::vector<std::string> sim_argv = {self, "run", "--quiet", "--gpu", profile, "--"};
  sim_argv.insert(sim_argv.end(), program.begin(), program.end());

  std::vector<Run> native, sim;
  for (int k = 0; k < native_runs; ++k) {
    native.push_back(run_capture(program, &nenv, timeout_s, files));
    if (k == 0 && !did_not_run(native.front()).empty()) break;
  }
  if (const std::string why = did_not_run(native.front()); !why.empty()) {
    std::fprintf(stderr, "vgpu check: '%s' %s on the card, so there is nothing to compare\n", program.front().c_str(), why.c_str());
    if (!native.front().output.empty()) std::fprintf(stderr, "--- its output ---\n%s", native.front().output.c_str());
    return kExitNativeDidNotRun;
  }
  for (int k = 0; k < sim_runs; ++k) sim.push_back(run_capture(sim_argv, nullptr, timeout_s, files));

  std::vector<std::string> identity_words = {prof.model, card->name};
  if (card->cc_major) identity_words.push_back("sm_" + std::to_string(card->cc_major) + std::to_string(card->cc_minor));
  auto filt = [&](const Run& r) { return filter(r.output, identity_words); };
  auto files_equal = [&](const Run& a, const Run& b) { return a.files == b.files; };

  std::printf("vgpu check: %s\n", program.front().c_str());
  std::printf("  card:       %s%s, simulated as %s\n", card->name.c_str(),
              card->cc_major ? (" (sm_" + std::to_string(card->cc_major) + std::to_string(card->cc_minor) + ")").c_str() : "",
              profile.c_str());
  if (!same_card) std::printf("  WARNING:    the profile is not the card (--allow-cross): the two are different machines\n");

  // The card against itself.
  std::vector<bool> mask;
  bool use_mask = false;
  int noisy = 0;
  if (native.size() >= 2) {
    bool inconclusive = false;
    for (size_t k = 1; k < native.size(); ++k) {
      const Filtered fa = filt(native[0]), fb = filt(native[k]);
      const Compared c = compare(fa, fb);
      const bool exit_same = native[0].exit_code == native[k].exit_code && !did_not_run(native[k]).size();
      const bool stable = (c.verdict == Verdict::Identical || c.verdict == Verdict::TimingOnly || c.verdict == Verdict::IdentityOnly) &&
                          exit_same && files_equal(native[0], native[k]);
      if (stable) continue;
      bool aligned = false;
      std::vector<bool> m = noisy_lines(fa, fb, &aligned);
      if (!aligned) { inconclusive = true; break; }
      if (mask.size() < m.size()) mask.resize(m.size());
      for (size_t j = 0; j < m.size(); ++j) if (m[j]) mask[j] = true;
      use_mask = true;
    }
    if (inconclusive) {
      std::printf("  card x%zu:   NOT REPRODUCIBLE and the runs cannot be lined up (different number of lines)\n", native.size());
      std::printf("result: inconclusive: the card disagrees with itself\n");
      return kExitInconclusive;
    }
    for (bool b : mask) noisy += b ? 1 : 0;
    if (use_mask && noisy)
      std::printf("  card x%zu:   not reproducible: %d line(s) differ between runs and are left out of the comparison\n", native.size(), noisy);
    else if (use_mask)
      std::printf("  card x%zu:   exit code or a file differs between runs (the program is not reproducible on the card)\n", native.size());
    else
      std::printf("  card x%zu:   reproducible\n", native.size());
  } else {
    std::printf("  card x1:    reproducibility not checked (--native-runs 1)\n");
  }

  // The simulator against itself.
  if (sim.size() >= 2) {
    bool same = true;
    for (size_t k = 1; k < sim.size(); ++k)
      same = same && sim[0].output == sim[k].output && sim[0].exit_code == sim[k].exit_code && files_equal(sim[0], sim[k]);
    std::printf("  simulator x%zu: %s\n", sim.size(), same ? "reproducible (identical bytes)" : "DIFFERS between runs (a finding in itself)");
  }

  // The simulator against the card.
  const Run& n0 = native.front();
  const Run& s0 = sim.front();
  const Filtered fn = filt(n0), fs = filt(s0);
  const Compared c = compare(fn, fs, use_mask ? &mask : nullptr);
  const bool exit_same = n0.exit_code == s0.exit_code;
  const bool files_same = files_equal(n0, s0);
  std::printf("  wall time:  card %.2f s, simulator %.2f s (process wall clock, not a result)\n", n0.seconds, s0.seconds);
  std::printf("  exit code:  card %d, simulator %d%s\n", n0.exit_code, s0.exit_code, exit_same ? "" : "   <-- DIFFERS");
  for (size_t f = 0; f < n0.files.size(); ++f) {
    const bool same = n0.files[f].second == s0.files[f].second;
    std::printf("  file:       %s: card %zu bytes, simulator %zu bytes%s\n", n0.files[f].first.c_str(), n0.files[f].second.size(),
                s0.files[f].second.size(), same ? ", identical" : "   <-- DIFFERS");
  }
  std::printf("  ignored:    %zu timing/blank line(s) on the card, %zu on the simulator; %zu/%zu line(s) naming the machine compared apart\n",
              fn.dropped, fs.dropped, fn.identity.size(), fs.identity.size());

  const bool numeric_ok = c.verdict == Verdict::NumericClose && c.max_rel <= tolerance;
  const bool text_ok = c.verdict == Verdict::Identical || c.verdict == Verdict::TimingOnly || c.verdict == Verdict::IdentityOnly || numeric_ok;
  const bool match = text_ok && exit_same && files_same && !did_not_run(s0).size();

  if (c.verdict == Verdict::NumericClose)
    std::printf("  numbers:    %zu differ, largest relative difference %.3g%s\n", c.numbers_differ, c.max_rel,
                numeric_ok ? " (within --tolerance)" : "");
  if (c.verdict == Verdict::IdentityOnly) {
    std::printf("  identity:   the lines naming the machine differ (driver and runtime strings are expected to):\n");
    for (const auto& l : fn.identity) std::printf("    card:      %s\n", l.c_str());
    for (const auto& l : fs.identity) std::printf("    simulator: %s\n", l.c_str());
  }
  if (show_ignored) {
    for (const auto& l : fn.dropped_lines) std::printf("  dropped (card):      %s\n", l.c_str());
    for (const auto& l : fs.dropped_lines) std::printf("  dropped (simulator): %s\n", l.c_str());
    for (const auto& l : fn.masked_lines) std::printf("  time masked (card):      %s\n", l.c_str());
    for (const auto& l : fs.masked_lines) std::printf("  time masked (simulator): %s\n", l.c_str());
    for (const auto& l : fn.identity) std::printf("  identity (card):      %s\n", l.c_str());
    for (const auto& l : fs.identity) std::printf("  identity (simulator): %s\n", l.c_str());
  }
  if (!match && show_diff && !c.first_a.empty())
    std::printf("  first difference:\n    card:      %s\n    simulator: %s\n", c.first_a.c_str(), c.first_b.c_str());

  const std::string sim_why = did_not_run(s0);
  std::printf("result: %s%s\n", match ? "the simulator agrees with the card: " : "DIFFERENT: ",
              !sim_why.empty() ? ("the program " + sim_why + " on the simulator").c_str()
              : !exit_same     ? "exit codes differ"
              : !files_same    ? "a file differs"
                               : verdict_name(c.verdict));
  return match ? 0 : kExitDiffer;
}
