// Turbo mode's program history (docs/perf-design/12-host-gpu-turbo.md section 17). No simulator state and no CUDA: unit tested on any machine
// (tests/unit/test_host_gpu_history.cpp).
//
// What it is for. With VGPU_HOST_GPU_FIT=1 the card's driver starts in every process (0.1 to 0.25 s, measured on an A10G) while the program sets
// up, whatever the program is. A program whose whole simulation takes less than that (Rodinia bfs, HeCBench atan2) is slower under turbo than in the
// simulator, and a program that faults on the card (Rodinia srad_v2: error 700) pays the start for nothing. A repeated run of the same program can
// know both. The history is a small file per (program, simulated profile, card model, mode) key that a process reads when its simulated device is
// created and writes when it ends.
//
// What it must never do: change a result. It only decides WHERE a launch runs (simulator or card) and whether the driver starts early; turbo mode
// is not bit-accurate by the owner's decision and its outputs are unchanged by this file existing, missing, stale, corrupt or shared by concurrent
// processes. Every failure of this header is "no history": the caller then does what it did before. Exact mode and FIT=0 never read it.
//
// Opt-in (VGPU_HOST_GPU_HISTORY=<directory>, or 1 for the default directory). A simulator that writes files nobody asked for, on a machine that
// may be an ephemeral CI runner or a read-only home, is a surprise; the cost of leaving it off is the one-off start of the driver of each short
// program that the history would have saved.
#pragma once

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace vgpu::hostgpu::history {

// Bumped whenever the meaning of a field changes: an older or newer file is a miss and is overwritten by the next write.
constexpr int kVersion = 1;
constexpr size_t kMaxFileBytes = 16384;   // a larger file is read as a miss, and no record is written larger than this
constexpr size_t kMaxKernels = 48;        // kernels remembered per program (the rest are not: the program keeps the behaviour it had)
constexpr size_t kMaxNameBytes = 160;
constexpr size_t kMaxFiles = 256;         // history files in the directory; the oldest by modification time go first when a write finds more
constexpr size_t kMaxImageBytes = 256u << 20;   // a fatbin larger than this is not hashed: the program has no key

// What a program looks like to the next run of it. Tiny: a run that stayed in the simulator observed less simulator time than `tiny_s`. Candidate: a
// run on the card spent so little on it that the simulator might do the whole program in less than the start of the driver; the next run tries.
// Long: a run found out the simulator needs more than that. Unknown: nothing that says either.
enum class Verdict : uint8_t { Unknown = 0, Candidate = 1, Tiny = 2, Long = 3 };

struct Kernel {
  std::string name;
  uint32_t launches = 0;   // in the last run that saw the kernel
  uint64_t max_threads = 0;   // threads of its largest launch in that run (a launch of more than twice that ends a hold before it runs)
  double card_s = 0;       // seconds of its launches on the card path (the last run that sent it)
  double sim_s = 0;        // seconds its launches took in the simulator (the last run that kept it)
  bool card_ok = false;    // a launch of it came back from the card
  bool faulted = false;    // a launch of it failed on the card (error 700, 719, ...): it is kept in the simulator from the first launch, in every later run
  bool loop = false;       // its code has a loop (the simulator's time then is not a function of the code and the grid: aobench 0.7 s on 65,536 threads, bfs 5 ms on 106,496)
};

struct Record {
  uint64_t runs = 0;
  Verdict verdict = Verdict::Unknown;
  double sim_total_s = 0;    // observed simulator seconds of the last run that stayed in the simulator (0 when none did)
  double card_total_s = 0;   // seconds of the card path of the last run that used the card
  uint32_t launches = 0;     // launches of the last run
  uint64_t threads_total = 0;   // threads of all its launches
  double loopfree_work = 0;
  std::vector<Kernel> kernels;
};

// What one process saw; folded into the record by `update` when it ends.
struct Observed {
  bool held = false;       // the process stayed in the simulator on the history's word
  bool released = false;   // ... and gave up on it (the simulator spent its budget): the driver was started
  double sim_s = 0;        // simulator seconds of the launches the simulator ran
  double card_s = 0;       // seconds of the launches the card ran (card path, copies included)
  uint32_t launches = 0;   // all launches
  uint64_t threads = 0;    // threads of all launches (grid x block)
  double loopfree_work = 0;   // threads x static instructions, over the launches whose kernel has no loop
  std::vector<Kernel> kernels;   // this process's: launches, card_s, sim_s, card_ok, faulted
};

struct Limits {
  double tiny_s = 0.125;      // a program is tiny when its simulation took less than this: half the driver's start (VGPU_HOST_GPU_START_COST, 0.25 s)
  double budget_s = 0.25;     // a process held in the simulator gives up after this much simulator time: a wrong history costs at most one start
  double candidate_card_s = 0.005;   // a card run is a candidate when the card path took less than this in all ...
  uint32_t candidate_launches = 512;   // ... in at most this many launches ...
  double candidate_loopfree_work = 4e8;   // ... and the threads x static instructions of the kernels without a loop at most this much (atan2 6e7 took 19 ms in the simulator, blockAccess 7e7 13 ms) ...
  uint64_t candidate_loop_threads = 8192;   // ... and no launch of a kernel with a loop larger than this (bfs: 4,096; aobench 65,536 and bilateral 32,768 are out). Not a bound on the simulator's time (a
                                            // loop of long trip count in a few threads passes), only on what the measured programs show (docs section 17.1)
};

inline Limits limits_for_start_cost(double start_cost_s, double candidate_card_s = 0.005) {
  Limits l;
  l.tiny_s = start_cost_s * 0.5;
  l.budget_s = start_cost_s;
  l.candidate_card_s = candidate_card_s;
  return l;
}

inline Kernel* find_kernel(std::vector<Kernel>& v, const std::string& name) {
  for (Kernel& k : v)
    if (k.name == name) return &k;
  return nullptr;
}
inline const Kernel* find_kernel(const std::vector<Kernel>& v, const std::string& name) {
  for (const Kernel& k : v)
    if (k.name == name) return &k;
  return nullptr;
}

// Folds one run into the record. A fault is remembered for good (it names the kernel, and a kernel's bad address does not depend on the run); the
// verdict is the last observation's: a held run says Tiny or Long by what the simulator took, a card run says Candidate when the card path was
// short and nothing faulted, and leaves a Tiny or Long verdict as it was (it knows nothing about the simulator).
inline void update(Record* rec, const Observed& o, const Limits& lim) {
  ++rec->runs;
  rec->launches = o.launches;
  rec->threads_total = o.threads;
  rec->loopfree_work = o.loopfree_work;
  bool any_fault = false, loops_too_wide = false;
  for (const Kernel& k : o.kernels) {
    any_fault = any_fault || k.faulted;
    Kernel* m = find_kernel(rec->kernels, k.name);
    if (!m) {
      if (rec->kernels.size() >= kMaxKernels || k.name.empty() || k.name.size() > kMaxNameBytes) continue;
      rec->kernels.push_back(Kernel{k.name, 0, 0, 0, 0, false, false, false});
      m = &rec->kernels.back();
    }
    m->launches = k.launches;
    m->max_threads = k.max_threads;
    m->loop = k.loop;
    loops_too_wide = loops_too_wide || (k.loop && k.max_threads > lim.candidate_loop_threads);
    if (k.card_ok || k.card_s > 0) m->card_s = k.card_s;
    if (k.sim_s > 0) m->sim_s = k.sim_s;
    m->card_ok = m->card_ok || k.card_ok;
    m->faulted = m->faulted || k.faulted;
  }
  if (o.held) {
    rec->sim_total_s = o.sim_s;
    rec->verdict = !o.released && o.sim_s < lim.tiny_s ? Verdict::Tiny : Verdict::Long;
    return;
  }
  rec->card_total_s = o.card_s;
  if (rec->verdict == Verdict::Tiny || rec->verdict == Verdict::Long) return;
  rec->verdict = !any_fault && o.card_s < lim.candidate_card_s && o.launches > 0 && o.launches <= lim.candidate_launches && o.loopfree_work <= lim.candidate_loopfree_work && !loops_too_wide ? Verdict::Candidate : Verdict::Unknown;
}

// A kernel's failure on the card, noted at once (the run is counted at its end by `update`).
inline void note_fault(Record* rec, const std::string& kernel) {
  if (kernel.empty() || kernel.size() > kMaxNameBytes) return;
  Kernel* m = find_kernel(rec->kernels, kernel);
  if (!m) {
    if (rec->kernels.size() >= kMaxKernels) return;
    rec->kernels.push_back(Kernel{kernel, 0, 0, 0, 0, false, false, false});
    m = &rec->kernels.back();
  }
  m->faulted = true;
}

struct Decision {
  bool start_driver = true;   // the driver starts when the simulated device is created, as without a history
  bool hold = false;          // launches stay in the simulator until the simulator has spent `budget_s`
  const char* why = "no history";
};

// What a process does at the creation of its simulated device. `rec` is null for a miss.
inline Decision decide_start(const Record* rec) {
  Decision d;
  if (!rec) return d;
  if (rec->verdict == Verdict::Tiny) return {false, true, "the program's last run in the simulator took less than the driver's start"};
  if (rec->verdict == Verdict::Candidate) return {false, true, "the program's last run on the card was short: trying it in the simulator"};
  bool faulted = false, ok = false;
  for (const Kernel& k : rec->kernels) {
    faulted = faulted || k.faulted;
    ok = ok || k.card_ok;
  }
  // Every kernel it launched failed on the card or none came back from it: the driver is not needed for what is known; a kernel the history has not
  // seen starts it when it is launched.
  if (faulted && !ok) return {false, false, "every kernel the program ran on the card failed there"};
  return d;
}

inline bool known_faulted(const Record* rec, const std::string& kernel) {
  if (!rec) return false;
  const Kernel* k = find_kernel(rec->kernels, kernel);
  return k && k->faulted;
}

// A kernel the history knows to take as long in the simulator as the budget has left, or a launch of more than twice the threads the largest one of the record had (another
// input), ends the hold before it is launched (a launch in the simulator cannot be stopped).
inline bool kernel_outlasts(const Record* rec, const std::string& kernel, double left_s, uint64_t threads = 0) {
  if (!rec) return false;
  const Kernel* k = find_kernel(rec->kernels, kernel);
  if (!k) return false;
  if (k->max_threads > 0 && threads > 2 * k->max_threads) return true;
  return k->launches > 0 && k->sim_s / k->launches >= left_s;
}

// ---- the 64-bit hash of the key (advisory: a collision can only send a program the wrong way, never change what it prints) ----
struct Hasher {
  uint64_t h = 0x9E3779B97F4A7C15ull;
  static uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
  }
  void word(uint64_t w) { h = mix(h ^ (w + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2))); }
  void bytes(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      uint64_t w;
      std::memcpy(&w, b + i, 8);
      word(w);
    }
    uint64_t tail = 0;
    for (size_t j = 0; i + j < n; ++j) tail |= uint64_t{b[i + j]} << (8 * j);
    word(tail ^ (uint64_t{n} << 56));
  }
  void text(const std::string& s) { bytes(s.data(), s.size()); }
};

inline std::string hex16(uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

// ---- the file: text, one record per file, a checksum over the body ----
// vgpu-host-gpu-history <version>
// key <16 hex>
// runs <n>
// verdict <0..3>
// totals <sim_total_s> <card_total_s> <launches> <threads> <loopfree_work>
// kernel <launches> <max_threads> <card_s> <sim_s> <card_ok> <faulted> <loop> <name>     (up to kMaxKernels)
// sum <16 hex>                                                         (of every byte before this line)
inline std::string serialize(const std::string& key, const Record& r) {
  std::string body = "vgpu-host-gpu-history " + std::to_string(kVersion) + "\nkey " + key + "\nruns " + std::to_string(r.runs) + "\nverdict " +
                     std::to_string(static_cast<int>(r.verdict)) + "\n";
  char buf[320];
  std::snprintf(buf, sizeof buf, "totals %.9g %.9g %u %llu %.9g\n", r.sim_total_s, r.card_total_s, r.launches, static_cast<unsigned long long>(r.threads_total), r.loopfree_work);
  body += buf;
  size_t n = 0;
  for (const Kernel& k : r.kernels) {
    if (n++ >= kMaxKernels || k.name.empty() || k.name.size() > kMaxNameBytes || k.name.find('\n') != std::string::npos) continue;
    std::snprintf(buf, sizeof buf, "kernel %u %llu %.9g %.9g %d %d %d ", k.launches, static_cast<unsigned long long>(k.max_threads), k.card_s, k.sim_s, k.card_ok ? 1 : 0, k.faulted ? 1 : 0, k.loop ? 1 : 0);
    body += buf + k.name + "\n";
  }
  Hasher h;
  h.text(body);
  return body + "sum " + hex16(h.h) + "\n";
}

// False on anything unexpected: a wrong version, a wrong key, a bad checksum, a malformed number, a truncated file.
inline bool parse(const std::string& text, const std::string& key, Record* out) {
  if (text.empty() || text.size() > kMaxFileBytes) return false;
  const size_t sum_at = text.rfind("sum ");
  if (sum_at == std::string::npos || (sum_at != 0 && text[sum_at - 1] != '\n')) return false;
  Hasher h;
  h.text(text.substr(0, sum_at));
  if (text.compare(sum_at, std::string::npos, "sum " + hex16(h.h) + "\n") != 0) return false;
  Record r;
  bool saw_version = false, saw_key = false, saw_runs = false, saw_verdict = false, saw_totals = false;
  size_t at = 0;
  while (at < sum_at) {
    size_t eol = text.find('\n', at);
    if (eol == std::string::npos || eol > sum_at) return false;
    const std::string line = text.substr(at, eol - at);
    at = eol + 1;
    const auto starts = [&](const char* p) { return line.compare(0, std::strlen(p), p) == 0; };
    if (starts("vgpu-host-gpu-history ")) {
      if (saw_version || line != "vgpu-host-gpu-history " + std::to_string(kVersion)) return false;
      saw_version = true;
    } else if (starts("key ")) {
      if (!saw_version || saw_key || line != "key " + key) return false;
      saw_key = true;
    } else if (starts("runs ")) {
      char* end = nullptr;
      errno = 0;
      const unsigned long long v = std::strtoull(line.c_str() + 5, &end, 10);
      if (errno || *end || end == line.c_str() + 5) return false;
      r.runs = v;
      saw_runs = true;
    } else if (starts("verdict ")) {
      if (line.size() != 9 || line[8] < '0' || line[8] > '3') return false;
      r.verdict = static_cast<Verdict>(line[8] - '0');
      saw_verdict = true;
    } else if (starts("totals ")) {
      unsigned l = 0;
      unsigned long long th = 0;
      double a = 0, b = 0, w = 0;
      char extra = 0;
      if (std::sscanf(line.c_str() + 7, "%lf %lf %u %llu %lf%c", &a, &b, &l, &th, &w, &extra) != 5 || !(a >= 0) || !(b >= 0) || !(w >= 0)) return false;
      r.loopfree_work = w;
      r.sim_total_s = a;
      r.card_total_s = b;
      r.launches = l;
      r.threads_total = th;
      saw_totals = true;
    } else if (starts("kernel ")) {
      Kernel k;
      unsigned l = 0;
      unsigned long long mt = 0;
      double cs = 0, ss = 0;
      int ok = 0, f = 0, lp = 0, used = 0;
      if (std::sscanf(line.c_str() + 7, "%u %llu %lf %lf %d %d %d %n", &l, &mt, &cs, &ss, &ok, &f, &lp, &used) != 7 || used <= 0) return false;
      const std::string name = line.substr(7 + used);
      if (name.empty() || name.size() > kMaxNameBytes || !(cs >= 0) || !(ss >= 0) || (ok != 0 && ok != 1) || (f != 0 && f != 1) || (lp != 0 && lp != 1) || r.kernels.size() >= kMaxKernels) return false;
      k.name = name;
      k.launches = l;
      k.max_threads = mt;
      k.card_s = cs;
      k.sim_s = ss;
      k.card_ok = ok == 1;
      k.faulted = f == 1;
      k.loop = lp == 1;
      r.kernels.push_back(std::move(k));
    } else {
      return false;
    }
  }
  if (!(saw_version && saw_key && saw_runs && saw_verdict && saw_totals)) return false;
  *out = std::move(r);
  return true;
}

// ---- the directory ----
inline std::string file_of(const std::string& dir, const std::string& key) { return dir + "/" + key + ".hist"; }

// Reads and parses one key's file. Any failure (missing, unreadable, too large, corrupt, another version) is a miss.
inline bool load(const std::string& dir, const std::string& key, Record* out) {
  const int fd = ::open(file_of(dir, key).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return false;
  struct stat st {};
  std::string text;
  bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0 && static_cast<size_t>(st.st_size) <= kMaxFileBytes;
  if (ok) {
    text.resize(static_cast<size_t>(st.st_size));
    size_t got = 0;
    while (got < text.size()) {
      const ssize_t n = ::read(fd, &text[got], text.size() - got);
      if (n <= 0) break;
      got += static_cast<size_t>(n);
    }
    ok = got == text.size();
  }
  ::close(fd);
  return ok && parse(text, key, out);
}

inline bool make_dirs(const std::string& dir) {
  if (dir.empty()) return false;
  std::string cur;
  for (size_t i = 0; i <= dir.size(); ++i) {
    if (i == dir.size() || dir[i] == '/') {
      if (!cur.empty() && ::mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST) return false;
    }
    if (i < dir.size()) cur += dir[i];
  }
  struct stat st {};
  return ::stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Keeps the directory bounded: more than kMaxFiles history files, the oldest by modification time are removed (best effort, only names this header writes).
inline void trim(const std::string& dir) {
  DIR* d = ::opendir(dir.c_str());
  if (!d) return;
  std::vector<std::pair<long long, std::string>> files;
  const time_t now = ::time(nullptr);
  while (const dirent* e = ::readdir(d)) {
    const std::string n = e->d_name;
    const bool hist = n.size() == 21 && n.compare(16, 5, ".hist") == 0;
    const bool tmp = n.size() > 21 && n.compare(16, 10, ".hist.tmp.") == 0;
    if (!hist && !tmp) continue;
    struct stat st {};
    if (::stat((dir + "/" + n).c_str(), &st) != 0) continue;
    if (tmp) {   // a writer that died between its write and its rename
      if (now - st.st_mtime > 3600) ::unlink((dir + "/" + n).c_str());
      continue;
    }
    files.emplace_back(static_cast<long long>(st.st_mtime), n);
  }
  ::closedir(d);
  if (files.size() <= kMaxFiles) return;
  std::sort(files.begin(), files.end());
  for (size_t i = 0; i + kMaxFiles * 3 / 4 < files.size(); ++i) ::unlink((dir + "/" + files[i].second).c_str());
}

// Writes the record to a file of its own name and renames it over the key's file: a reader sees the old file or the new, never half of one, and two
// writers leave one of their records (the later rename), not a mixture.
inline bool store(const std::string& dir, const std::string& key, const Record& r) {
  if (!make_dirs(dir)) return false;
  const std::string text = serialize(key, r);
  if (text.size() > kMaxFileBytes) return false;
  static std::atomic<unsigned> counter{0};
  const std::string tmp = file_of(dir, key) + ".tmp." + std::to_string(::getpid()) + "." + std::to_string(counter++);
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0644);
  if (fd < 0) return false;
  size_t put = 0;
  while (put < text.size()) {
    const ssize_t n = ::write(fd, text.data() + put, text.size() - put);
    if (n <= 0) break;
    put += static_cast<size_t>(n);
  }
  const bool whole = put == text.size();
  ::close(fd);
  if (!whole || ::rename(tmp.c_str(), file_of(dir, key).c_str()) != 0) {
    ::unlink(tmp.c_str());
    return false;
  }
  trim(dir);
  return true;
}

// The directory VGPU_HOST_GPU_HISTORY names: empty when the history is off (unset, 0, empty). `1` is the default directory under the user's cache.
inline std::string directory_from(const char* value, const char* xdg_cache, const char* home) {
  if (!value || !*value || std::string(value) == "0") return "";
  if (std::string(value) != "1") return value;
  if (xdg_cache && *xdg_cache) return std::string(xdg_cache) + "/pantheonsim/host-gpu-history";
  if (home && *home) return std::string(home) + "/.cache/pantheonsim/host-gpu-history";
  return "";
}

}  // namespace vgpu::hostgpu::history
