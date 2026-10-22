// Tier 2 of the SASS executor, milestone 1: see jit.hpp.
#include "jit.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <set>
#include <thread>

#include "tier.hpp"
#include "vgpu/sass/sass.hpp"

namespace vgpu::sass::jit {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
constexpr bool kSanitized = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
constexpr bool kSanitized = true;
#else
constexpr bool kSanitized = false;
#endif
#else
constexpr bool kSanitized = false;
#endif

// Two independent 64-bit hashes of the bytes, as 32 hex digits (the key and the object's checksum). Not cryptographic: a key collision is
// caught by the metadata check, a damaged object by the checksum.
void hash128(const void* data, size_t n, uint64_t& a, uint64_t& b) {
  const unsigned char* d = static_cast<const unsigned char*>(data);
  a = 14695981039346656037ull;
  b = 0x9e3779b97f4a7c15ull ^ (n * 0xff51afd7ed558ccdull);
  for (size_t i = 0; i < n; ++i) {
    a = (a ^ d[i]) * 1099511628211ull;
    b = (b + d[i] + 1) * 0x9fb21c651e98df25ull;
    b ^= b >> 29;
  }
}
std::string hex128(uint64_t a, uint64_t b) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%016llx%016llx", static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
  return buf;
}
std::string hash_text(const std::string& s) {
  uint64_t a, b;
  hash128(s.data(), s.size(), a, b);
  return hex128(a, b);
}

std::string detect_isa() {
#if defined(__x86_64__) && defined(__GNUC__)
  __builtin_cpu_init();
  const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  if (avx2 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512vl"))
    return "avx512";
  return avx2 ? "avx2" : "x86-64";
#else
  return "generic";
#endif
}
std::string isa_flags(const std::string& isa) {
  if (isa == "avx512") return "-mavx2 -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl";
  if (isa == "avx2") return "-mavx2 -mfma";
  return "";
}

std::vector<std::string> split_words(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) ++i;
    size_t j = i;
    while (j < s.size() && !(s[j] == ' ' || s[j] == '\t' || s[j] == '\n')) ++j;
    if (j > i) out.push_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

bool read_file(const std::string& path, std::string& out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  out.clear();
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
  const bool ok = !std::ferror(f);
  std::fclose(f);
  return ok;
}

bool write_file(const std::string& path, const std::string& data) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  size_t off = 0;
  while (off < data.size()) {
    const ssize_t w = ::write(fd, data.data() + off, data.size() - off);
    if (w < 0) {
      if (errno == EINTR) continue;
      ::close(fd);
      return false;
    }
    off += static_cast<size_t>(w);
  }
  return ::close(fd) == 0;
}

bool executable_file(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && ::access(p.c_str(), X_OK) == 0;
}

// The first of `names` (or `name` itself, if it has a slash) that is an executable file on `path` ("" if none).
std::string find_exe(const std::string& name, const std::string& path) {
  if (name.find('/') != std::string::npos) return executable_file(name) ? name : std::string();
  size_t i = 0;
  while (i <= path.size()) {
    size_t j = path.find(':', i);
    if (j == std::string::npos) j = path.size();
    const std::string dir = path.substr(i, j - i);
    const std::string cand = (dir.empty() ? std::string(".") : dir) + "/" + name;
    if (executable_file(cand)) return cand;
    i = j + 1;
  }
  return {};
}

struct Proc {
  int rc = -1;               // exit status, or 128 + signal
  bool timed_out = false;
  bool cancelled = false;
  bool spawn_failed = false;
  std::string out;           // stdout and stderr, the first 64 KB
};

// Runs argv with its output captured, under a wall limit and an address-space limit (mem_mb 0: none), in a process group of its own so that
// a timeout or a cancel kills the compiler's children too. The child does only async-signal-safe work between fork and exec.
Proc run_process(const std::vector<std::string>& args, unsigned timeout_ms, unsigned mem_mb, const std::atomic<bool>* cancel) {
  Proc r;
  std::vector<char*> argv;
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  int pfd[2];
  if (::pipe2(pfd, O_CLOEXEC) != 0) {
    r.spawn_failed = true;
    return r;
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(pfd[0]);
    ::close(pfd[1]);
    r.spawn_failed = true;
    return r;
  }
  if (pid == 0) {
    ::setpgid(0, 0);
    const int nul = ::open("/dev/null", O_RDONLY);
    if (nul >= 0) ::dup2(nul, 0);
    ::dup2(pfd[1], 1);
    ::dup2(pfd[1], 2);
    if (mem_mb) {
      rlimit rl;
      rl.rlim_cur = rl.rlim_max = static_cast<rlim_t>(mem_mb) << 20;
      ::setrlimit(RLIMIT_AS, &rl);
    }
    ::setpriority(PRIO_PROCESS, 0, 10);
    ::execv(argv[0], argv.data());
    ::_exit(127);
  }
  ::close(pfd[1]);
  ::setpgid(pid, pid);   // (also done by the child: whichever runs first wins, the other is harmless)
  const int fl = ::fcntl(pfd[0], F_GETFL);
  ::fcntl(pfd[0], F_SETFL, fl | O_NONBLOCK);
  const auto t0 = Clock::now();
  bool reaped = false;
  int status = 0;
  const auto drain = [&] {
    char buf[4096];
    for (;;) {
      const ssize_t n = ::read(pfd[0], buf, sizeof buf);
      if (n > 0) {
        if (r.out.size() < (1u << 16)) r.out.append(buf, static_cast<size_t>(n));
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      return n == 0;   // eof; else EAGAIN
    }
  };
  while (!reaped) {
    pollfd p{pfd[0], POLLIN, 0};
    ::poll(&p, 1, 10);
    const bool eof = drain();
    const pid_t w = ::waitpid(pid, &status, WNOHANG);
    if (w == pid) {
      reaped = true;
      break;
    }
    (void)eof;
    const bool late = ms_since(t0) > timeout_ms;
    const bool stop = cancel && cancel->load();
    if (late || stop) {
      r.timed_out = late;
      r.cancelled = stop && !late;
      ::kill(-pid, SIGKILL);
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      reaped = true;
    }
  }
  drain();
  ::close(pfd[0]);
  r.rc = WIFEXITED(status) ? WEXITSTATUS(status) : WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1;
  if (r.rc == 127 && r.out.empty() && !r.timed_out) r.spawn_failed = true;
  ::kill(-pid, SIGKILL);   // (anything the compiler left behind in its group)
  return r;
}

constexpr uint32_t kMetaMagic = 0x54494a56u;   // "VJIT"

}  // namespace

const char* status_name(Status s) {
  switch (s) {
    case Status::Ok: return "ok";
    case Status::NoCompiler: return "no-compiler";
    case Status::CompileFailed: return "compile-failed";
    case Status::CompileTimeout: return "compile-timeout";
    case Status::CacheUnusable: return "cache-unusable";
    case Status::Mismatch: return "mismatch";
  }
  return "?";
}

std::string checksum(const std::string& bytes) { return hash_text(bytes); }

Config Config::from_env() {
  Config c;
  const auto env = [](const char* n) -> std::string {
    const char* v = std::getenv(n);
    return v ? v : "";
  };
  c.cache_dir = env("VGPU_JIT_CACHE");
  c.cxx = env("VGPU_JIT_CXX");
  c.extra_flags = env("VGPU_JIT_FLAGS");
  if (const std::string t = env("VGPU_JIT_TIMEOUT"); !t.empty()) c.timeout_s = static_cast<unsigned>(std::strtoul(t.c_str(), nullptr, 10));
  if (const std::string m = env("VGPU_JIT_MEM_MB"); !m.empty()) c.mem_mb = static_cast<unsigned>(std::strtoul(m.c_str(), nullptr, 10));
  return c;
}

// ---- the generator -----------------------------------------------------------------------------------------------------------------

std::vector<uint32_t> Engine::meta(const UnitSpec& spec) {
  std::vector<uint32_t> m{kMetaMagic, kAbiVersion, kGeneratorVersion, static_cast<uint32_t>(spec.sm), static_cast<uint32_t>(spec.regions.size())};
  for (const RegionSpec& r : spec.regions) {
    m.push_back(static_cast<uint32_t>(r.ops.size()));
    for (const OpFacts& o : r.ops) {
      m.push_back(static_cast<uint32_t>(o.lo));
      m.push_back(static_cast<uint32_t>(o.lo >> 32));
      m.push_back(static_cast<uint32_t>(o.hi));
      m.push_back(static_cast<uint32_t>(o.hi >> 32));
      m.push_back(o.cls | uint32_t{o.guard} << 8 | uint32_t{o.guard_not} << 16 | uint32_t{o.guard_uniform} << 17 | uint32_t{o.unguarded} << 18);
    }
  }
  return m;
}

std::string Engine::source(const UnitSpec& spec) {
  std::string s;
  const auto put = [&](const char* fmt, auto... a) {
    char buf[512];
    std::snprintf(buf, sizeof buf, fmt, a...);
    s += buf;
  };
  put("// Generated by the pantheonsim SASS JIT (tier 2), generator %u, ABI %u. Do not edit; this is a cache file.\n", kGeneratorVersion, kAbiVersion);
  s += "#include <cstddef>\n#include <cstdint>\n\n";
  s += "struct Ctx {\n"
       "  uint32_t abi;\n  uint32_t flags;\n  const bool* up;\n  const uint32_t* p;\n  uint64_t* steps;\n  uint64_t* instructions;\n"
       "  uint64_t* thread_instructions;\n  uint64_t* by_class;\n  void (*escape)(Ctx*, uint32_t op, uint32_t ex);\n  void* host;\n"
       "  uint32_t* R;\n  const uint8_t* b0;\n  uint64_t b0n;\n  const void* pool;\n};\n";
  put("static_assert(sizeof(Ctx) == %zu, \"Ctx layout\");\n", sizeof(Ctx));
#define VJ_FIELD(f) put("static_assert(offsetof(Ctx, " #f ") == %zu, \"Ctx::" #f "\");\n", offsetof(Ctx, f))
  VJ_FIELD(abi); VJ_FIELD(flags); VJ_FIELD(up); VJ_FIELD(p); VJ_FIELD(steps); VJ_FIELD(instructions); VJ_FIELD(thread_instructions);
  VJ_FIELD(by_class); VJ_FIELD(escape); VJ_FIELD(host); VJ_FIELD(R); VJ_FIELD(b0); VJ_FIELD(b0n); VJ_FIELD(pool);
#undef VJ_FIELD
  s += "\n#define VJ_EXPORT __attribute__((visibility(\"default\")))\n\n";
  const std::vector<uint32_t> m = meta(spec);
  s += "extern \"C\" {\n";
  put("VJ_EXPORT extern const uint32_t vgpu_jit_meta_len = %zu;\n", m.size());
  s += "VJ_EXPORT extern const uint32_t vgpu_jit_meta[] = {";
  for (size_t i = 0; i < m.size(); ++i) {
    if (i % 10 == 0) s += "\n  ";
    put("0x%08xu,", m[i]);
  }
  s += "\n};\n}\n\n";
  for (size_t ri = 0; ri < spec.regions.size(); ++ri) {
    const RegionSpec& r = spec.regions[ri];
    put("// region %zu: %zu ops\n", ri, r.ops.size());
    put("extern \"C\" VJ_EXPORT void vgpu_jit_r%zu(Ctx* c, uint32_t group, uint32_t glanes) {\n", ri);
    for (size_t i = 0; i < r.ops.size(); ++i) {
      const OpFacts& o = r.ops[i];
      put("  {  // op %zu: %016llx%016llx, class %u", i, static_cast<unsigned long long>(o.hi), static_cast<unsigned long long>(o.lo), unsigned{o.cls});
      if (!o.unguarded) put(", guard %s%s%u", o.guard_not ? "!" : "", o.guard_uniform ? "UP" : "P", unsigned{o.guard});
      s += "\n";
      if (o.unguarded) {
        s += "    const uint32_t ex = group, lanes = glanes;\n";
      } else {
        // tier_guard: the guard predicate as a lane mask (PT: all lanes), negated for @!P, narrowed to the group.
        if (o.guard_uniform) put("    uint32_t g = (%s) ? 0xffffffffu : 0u;\n", o.guard == 7 ? "true" : (std::string("c->up[") + std::to_string(o.guard) + "]").c_str());
        else if (o.guard == 7) s += "    uint32_t g = 0xffffffffu;\n";
        else put("    uint32_t g = c->p[%u];\n", unsigned{o.guard});
        if (o.guard_not) s += "    g = ~g;\n";
        s += "    const uint32_t ex = group & g;\n"
             "    const uint32_t lanes = ex == group ? glanes : static_cast<uint32_t>(__builtin_popcount(ex));\n";
      }
      s += "    ++*c->steps;\n    ++*c->instructions;\n    *c->thread_instructions += lanes;\n";
      put("    c->by_class[%u] += lanes;\n", unsigned{o.cls});
      put("    if (ex | (c->flags & 1u)) c->escape(c, %zuu, ex);\n", i);
      s += "  }\n";
    }
    s += "}\n\n";
  }
  return s;
}

// ---- the engine -----------------------------------------------------------------------------------------------------------------------

struct Engine::Impl {
  std::mutex mu;
  bool cxx_resolved = false;
  std::string cxx_path, cxx_id, cxx_why;
  bool dir_checked = false;
  std::string dir, dir_err;
  std::atomic<uint64_t> seq{0};
  std::atomic<bool> cancel{false};
};

Engine::Engine(Config cfg) : cfg_(std::move(cfg)), p_(new Impl) {}
Engine::~Engine() = default;
void Engine::cancel() { p_->cancel.store(true); }

std::string Engine::compiler_id(std::string* why) {
  std::lock_guard<std::mutex> g(p_->mu);
  if (!p_->cxx_resolved) {
    p_->cxx_resolved = true;
    const char* envp = std::getenv("PATH");
    const std::string search = !cfg_.path.empty() ? cfg_.path : (envp ? envp : "/usr/local/bin:/usr/bin:/bin");
    std::string found;
    if (!cfg_.cxx.empty()) {
      found = find_exe(cfg_.cxx, search);
      if (found.empty()) p_->cxx_why = "the compiler '" + cfg_.cxx + "' (VGPU_JIT_CXX) is not an executable file";
    } else {
      for (const char* n : {"c++", "g++", "clang++"}) {
        found = find_exe(n, search);
        if (!found.empty()) break;
      }
      if (found.empty()) p_->cxx_why = "no C++ compiler (c++, g++, clang++) on PATH; set VGPU_JIT_CXX";
    }
    if (!found.empty()) {
      const Proc v = run_process({found, "--version"}, 20000, 0, &p_->cancel);
      if (v.rc != 0 || v.out.empty()) {
        p_->cxx_why = "cannot run '" + found + " --version'";
      } else {
        const Proc m = run_process({found, "-dumpmachine"}, 20000, 0, &p_->cancel);
        std::string first = v.out.substr(0, v.out.find('\n'));
        std::string mach = m.rc == 0 ? m.out.substr(0, m.out.find('\n')) : std::string();
        p_->cxx_id = first + " | " + mach;
        p_->cxx_path = found;
      }
    }
  }
  if (p_->cxx_path.empty()) {
    if (why) *why = p_->cxx_why;
    return {};
  }
  return p_->cxx_id;
}

std::string Engine::cache_path() const {
  if (!cfg_.cache_dir.empty()) return cfg_.cache_dir;
  std::string root;
  if (const char* e = std::getenv("VGPU_JIT_CACHE"); e && *e) return e;
  if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) root = x;
  else if (const char* h = std::getenv("HOME"); h && *h) root = std::string(h) + "/.cache";
  else root = "/tmp";
  return root + "/pantheonsim/jit/" + hash_text(std::string(__DATE__ " " __TIME__ " abi") + std::to_string(kAbiVersion)).substr(0, 12);
}

std::string Engine::object_path(const std::string& key) const { return cache_path() + "/" + key + ".so"; }

namespace {
// mkdir -p with mode 0700, then the checks of the leaf: a directory, ours, and writable by no one else.
std::string ensure_dir(const std::string& dir) {
  if (dir.empty()) return "no cache directory";
  for (size_t i = 1; i <= dir.size(); ++i) {
    if (i != dir.size() && dir[i] != '/') continue;
    const std::string pre = dir.substr(0, i);
    if (::mkdir(pre.c_str(), 0700) != 0 && errno != EEXIST) return "cannot create " + pre + ": " + std::strerror(errno);
  }
  struct stat st;
  if (::stat(dir.c_str(), &st) != 0) return "cannot stat " + dir + ": " + std::strerror(errno);
  if (!S_ISDIR(st.st_mode)) return dir + " is not a directory";
  if (st.st_uid != ::geteuid()) return dir + " is not owned by the user running the simulator";
  if (st.st_mode & 022) return dir + " is writable by others";
  return {};
}
}  // namespace

std::string Engine::key(const UnitSpec& spec, std::string* why) {
  const std::string cid = compiler_id(why);
  if (cid.empty()) return {};
  const std::string isa = cfg_.isa.empty() ? detect_isa() : cfg_.isa;
  std::string text = source(spec);
  text += "\n//key: ";
  text += cid + " | isa " + isa + " | flags " + isa_flags(isa) + " | " + cfg_.extra_flags + " | abi " + std::to_string(kAbiVersion) + " | gen " +
          std::to_string(kGeneratorVersion);
  return hash_text(text);
}

Result Engine::load_(const UnitSpec& spec, const std::string& key, bool fresh) {
  Result r;
  r.key = key;
  const std::string dir = cache_path();
  const std::string so = dir + "/" + key + ".so", sum = dir + "/" + key + ".sum";
  std::string sumtext, bytes;
  if (!read_file(sum, sumtext) || !read_file(so, bytes)) return r;   // absent (or half written by a compile that has not finished): a miss
  {
    unsigned long long size = 0;
    char h[64] = {};
    if (std::sscanf(sumtext.c_str(), "%63s %llu", h, &size) != 2 || size != bytes.size() || hash_text(bytes) != h) {
      r.message = "the cached object " + key + " fails its checksum";
      r.replaced_corrupt = true;   // (the caller removes it, under the lock)
      return r;
    }
  }
  std::string err;
  void* h = cfg_.dlopen_fn ? cfg_.dlopen_fn(so.c_str(), &err) : ::dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    if (!cfg_.dlopen_fn) {
      const char* e = ::dlerror();
      err = e ? e : "dlopen failed";
    }
    if (fresh) {
      r.status = Status::CacheUnusable;
      r.message = "cannot load an object just built in " + dir + " (is it mounted noexec?): " + err;
    } else {
      r.message = "the cached object " + key + " does not load: " + err;
      r.replaced_corrupt = true;
    }
    return r;
  }
  const auto bad = [&](const std::string& what) {
    if (!cfg_.dlopen_fn) ::dlclose(h);
    r.message = "the cached object " + key + ": " + what;
    if (fresh) r.status = Status::Mismatch;
    else r.replaced_corrupt = true;
    return r;
  };
  const uint32_t* mlen = static_cast<const uint32_t*>(::dlsym(h, "vgpu_jit_meta_len"));
  const uint32_t* mdat = static_cast<const uint32_t*>(::dlsym(h, "vgpu_jit_meta"));
  if (!mlen || !mdat) return bad("has no metadata");
  const std::vector<uint32_t> want = meta(spec);
  if (*mlen != want.size() || std::memcmp(mdat, want.data(), want.size() * sizeof(uint32_t)) != 0)
    return bad("was made for other instruction words, architecture or generator version");
  auto lib = std::make_shared<Loaded>();
  lib->handle = h;
  for (size_t i = 0; i < spec.regions.size(); ++i) {
    void* f = ::dlsym(h, ("vgpu_jit_r" + std::to_string(i)).c_str());
    if (!f) return bad("lacks region " + std::to_string(i));
    lib->fns.push_back(reinterpret_cast<RegionFn>(f));
  }
  r.lib = lib;
  r.found = true;
  r.hit = true;
  return r;
}

Result Engine::lookup(const UnitSpec& spec) {
  const auto t0 = Clock::now();
  Result r;
  std::string why;
  const std::string key = this->key(spec, &why);
  if (key.empty()) {
    r.status = Status::NoCompiler;
    r.message = why;
    return r;
  }
  {
    std::lock_guard<std::mutex> g(p_->mu);
    if (!p_->dir_checked) {
      p_->dir = cache_path();
      p_->dir_err = ensure_dir(p_->dir);
      p_->dir_checked = true;
    }
    if (!p_->dir_err.empty()) {
      r.status = Status::CacheUnusable;
      r.message = p_->dir_err;
      return r;
    }
  }
  r = load_(spec, key, false);
  r.lookup_ms = ms_since(t0);
  return r;
}

Result Engine::obtain(const UnitSpec& spec) {
  Result r = lookup(spec);
  if (r.found || r.status != Status::Ok) return r;
  const auto t0 = Clock::now();
  const std::string key = r.key;
  const std::string dir = cache_path();
  const bool damaged = r.replaced_corrupt;
  const int lockfd = ::open((dir + "/" + key + ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
  if (lockfd < 0) {
    r.status = Status::CacheUnusable;
    r.message = "cannot create a lock file in " + dir + ": " + std::strerror(errno);
    return r;
  }
  ::flock(lockfd, LOCK_EX);
  struct Unlock {
    int fd;
    ~Unlock() {
      ::flock(fd, LOCK_UN);
      ::close(fd);
    }
  } unlock{lockfd};
  // Another thread or process may have built it while this one waited.
  Result again = load_(spec, key, false);
  if (again.found) {
    again.replaced_corrupt = damaged;
    return again;
  }
  const std::string so = dir + "/" + key + ".so", sum = dir + "/" + key + ".sum";
  ::unlink(sum.c_str());
  ::unlink(so.c_str());   // (damaged, or half of a pair)
  Result out;
  out.key = key;
  out.replaced_corrupt = damaged || again.replaced_corrupt;
  const std::string stem = dir + "/" + key + "." + std::to_string(::getpid()) + "." + std::to_string(p_->seq.fetch_add(1));
  const std::string tmp_cpp = stem + ".cpp", tmp_so = stem + ".so";
  if (!write_file(tmp_cpp, source(spec))) {
    out.status = Status::CacheUnusable;
    out.message = "cannot write to " + dir + ": " + std::strerror(errno);
    return out;
  }
  const std::string isa = cfg_.isa.empty() ? detect_isa() : cfg_.isa;
  std::vector<std::string> cmd{p_->cxx_path, "-std=c++20", "-O1", "-fPIC", "-shared", "-fexceptions", "-ffp-contract=off", "-fvisibility=hidden", "-w"};
  for (const std::string& f : split_words(isa_flags(isa))) cmd.push_back(f);
  for (const std::string& f : split_words(cfg_.extra_flags)) cmd.push_back(f);
  cmd.push_back("-o");
  cmd.push_back(tmp_so);
  cmd.push_back(tmp_cpp);
  const Proc pr = run_process(cmd, cfg_.timeout_s * 1000u, cfg_.mem_mb, &p_->cancel);
  ::unlink(tmp_cpp.c_str());
  out.compile_ms = ms_since(t0);
  compiles_.fetch_add(1);
  out.compiled = true;
  if (pr.timed_out || pr.cancelled || pr.rc != 0) {
    ::unlink(tmp_so.c_str());
    out.status = pr.timed_out ? Status::CompileTimeout : Status::CompileFailed;
    out.message = pr.timed_out ? "the compile took more than " + std::to_string(cfg_.timeout_s) + " s and was killed"
                  : pr.cancelled ? "the compile was cancelled"
                                 : "the compiler failed (status " + std::to_string(pr.rc) + "): " + pr.out.substr(0, 600);
    return out;
  }
  std::string bytes;
  if (!read_file(tmp_so, bytes) || bytes.empty()) {
    ::unlink(tmp_so.c_str());
    out.status = Status::CompileFailed;
    out.message = "the compiler produced no object";
    return out;
  }
  const std::string sumtext = hash_text(bytes) + " " + std::to_string(bytes.size()) + "\n";
  // The object first, then its checksum: a reader that sees an object without a checksum takes it for a miss.
  if (::rename(tmp_so.c_str(), so.c_str()) != 0 || !write_file(stem + ".sum", sumtext) || ::rename((stem + ".sum").c_str(), sum.c_str()) != 0) {
    ::unlink(tmp_so.c_str());
    ::unlink((stem + ".sum").c_str());
    out.status = Status::CacheUnusable;
    out.message = "cannot install the object in " + dir + ": " + std::strerror(errno);
    return out;
  }
  Result fresh = load_(spec, key, true);
  fresh.compiled = true;
  fresh.hit = false;
  fresh.compile_ms = out.compile_ms;
  fresh.replaced_corrupt = out.replaced_corrupt;
  if (!fresh.found) {
    ::unlink(sum.c_str());
    ::unlink(so.c_str());
    if (fresh.status == Status::Ok) fresh.status = Status::CacheUnusable;
  }
  return fresh;
}

// ---- the manager: statistics, the worker, the notices ------------------------------------------------------------------------------------

Stats& stats() {
  static Stats s;
  return s;
}

namespace {

enum class Job { Lookup, Compile };

struct Manager {
  Engine engine;
  uint64_t threshold;
  bool sync;
  bool quiet;
  unsigned max_region;
  std::atomic<bool> disabled{false};
  std::mutex mu;
  std::condition_variable cv, idle_cv;
  std::deque<std::pair<Job, std::shared_ptr<Unit>>> q;
  std::thread worker;
  bool stop = false, busy = false;
  std::set<std::string> told;

  static Config config() {
    Config c = Config::from_env();
    return c;
  }
  Manager() : engine(config()) {
    (void)stats();   // (constructed first, so destroyed last)
    const auto num = [](const char* n, uint64_t d) {
      const char* v = std::getenv(n);
      return v && *v ? static_cast<uint64_t>(std::strtoull(v, nullptr, 10)) : d;
    };
    threshold = num("VGPU_JIT_THRESHOLD", 10000000ull);
    const char* sy = std::getenv("VGPU_JIT_SYNC");
    sync = sy && sy[0] == '1';
    const char* q_ = std::getenv("VGPU_QUIET");
    quiet = q_ && *q_ && !(q_[0] == '0' && !q_[1]);
    max_region = static_cast<unsigned>(std::clamp<uint64_t>(num("VGPU_JIT_MAX_REGION", 96), kMinRegion, 4096));
  }
  ~Manager() {
    {
      std::lock_guard<std::mutex> g(mu);
      stop = true;
    }
    engine.cancel();
    cv.notify_all();
    if (worker.joinable()) worker.join();
    report();
  }
  void report();

  void notice(const std::string& kind, const std::string& msg) {
    {
      std::lock_guard<std::mutex> g(mu);
      if (!told.insert(kind).second) return;
    }
    if (!quiet) std::fprintf(stderr, "pantheonsim: JIT: %s; those regions stay on the tier\n", msg.c_str());
  }

  void publish(Unit& u, Result& r) {
    u.lib = r.lib;
    u.state.store(static_cast<int>(State::Ready), std::memory_order_release);
    stats().units_ready.fetch_add(1);
  }
  void fail(Unit& u, const Result& r) {
    u.state.store(static_cast<int>(State::Failed), std::memory_order_release);
    stats().failures.fetch_add(1);
    notice(status_name(r.status), r.message);
    if (r.status == Status::CacheUnusable || r.status == Status::NoCompiler) disabled.store(true);   // nothing will work: stop trying
  }

  void run(Job job, const std::shared_ptr<Unit>& u) {
    if (disabled.load()) {
      u->state.store(static_cast<int>(State::Failed), std::memory_order_release);
      return;
    }
    if (job == Job::Lookup) {
      Result r = engine.lookup(u->spec);
      stats().lookup_us.fetch_add(static_cast<uint64_t>(r.lookup_ms * 1000));
      if (r.found) {
        stats().cache_hits.fetch_add(1);
        publish(*u, r);
      } else if (r.status != Status::Ok) {
        fail(*u, r);
      } else {
        int exp = static_cast<int>(State::Looking);
        u->state.compare_exchange_strong(exp, static_cast<int>(State::Counting));
      }
    } else {
      Result r = engine.obtain(u->spec);
      if (r.compiled) {
        stats().compiles.fetch_add(1);
        stats().compile_us.fetch_add(static_cast<uint64_t>(r.compile_ms * 1000));
      }
      if (r.replaced_corrupt) stats().corrupt.fetch_add(1);
      if (r.found) {
        if (r.hit) stats().cache_hits.fetch_add(1);
        publish(*u, r);
      } else {
        fail(*u, r);
      }
    }
  }

  void submit(Job job, std::shared_ptr<Unit> u) {
    if (sync) {
      run(job, u);
      return;
    }
    std::unique_lock<std::mutex> g(mu);
    if (stop) return;
    if (!worker.joinable()) worker = std::thread([this] { loop(); });
    q.emplace_back(job, std::move(u));
    cv.notify_one();
  }
  void loop() {
    std::unique_lock<std::mutex> g(mu);
    for (;;) {
      cv.wait(g, [&] { return stop || !q.empty(); });
      if (stop) return;
      auto [job, u] = std::move(q.front());
      q.pop_front();
      busy = true;
      g.unlock();
      run(job, u);
      g.lock();
      busy = false;
      idle_cv.notify_all();
    }
  }
  void drain() {
    std::unique_lock<std::mutex> g(mu);
    idle_cv.wait(g, [&] { return q.empty() && !busy; });
  }
};

Manager& manager() {
  static Manager m;
  return m;
}

}  // namespace


void Unit::note(uint32_t k) {
  if (state.load(std::memory_order_relaxed) != static_cast<int>(State::Counting)) return;
  const uint64_t total = executed.fetch_add(k, std::memory_order_relaxed) + k;
  if (total < manager().threshold) return;
  int exp = static_cast<int>(State::Counting);
  if (state.compare_exchange_strong(exp, static_cast<int>(State::Queued))) manager().submit(Job::Compile, shared_from_this());
}

void drain() { manager().drain(); }

void Manager::report() {
  const char* v = std::getenv("VGPU_JIT_STATS");
  if (!(v && *v && v[0] != '0')) return;
  std::FILE* out = stderr;
  if (const char* f = std::getenv("VGPU_JIT_STATS_FILE"); f && *f)
    if (std::FILE* g = std::fopen(f, "a")) out = g;
  Stats& s = stats();
  std::fprintf(out,
               "jit: %llu units (%llu regions), %llu ready, %llu cache hits, %llu compiles (%.1f ms), %llu failures, %llu damaged objects replaced; "
               "%llu region calls ran %llu warp instructions, %llu exceptions unwound through them\n",
               (unsigned long long)s.units.load(), (unsigned long long)s.regions.load(), (unsigned long long)s.units_ready.load(),
               (unsigned long long)s.cache_hits.load(), (unsigned long long)s.compiles.load(), double(s.compile_us.load()) / 1000.0,
               (unsigned long long)s.failures.load(), (unsigned long long)s.corrupt.load(), (unsigned long long)s.regions_run.load(),
               (unsigned long long)s.instructions_run.load(), (unsigned long long)s.region_throws.load());
  if (out != stderr) std::fclose(out);
}

bool requested() {
  static const bool on = [] {
    if (kSanitized) return false;
    const char* v = std::getenv("VGPU_JIT");
    return v && v[0] == '1' && !v[1] && tier::requested();
  }();
  return on;
}

std::shared_ptr<Unit> make_unit(const tier::Program& prog, int sm) {
  if (!requested()) return nullptr;
  Manager& mgr = manager();
  if (mgr.disabled.load()) return nullptr;
  const size_t count = prog.ops.size();
  // Where a branch lands starts a region, so that a warp that jumps to a loop's top finds one there.
  std::vector<uint8_t> leader(count + 1, 0);
  for (const tier::MOp& op : prog.ops)
    if (op.form == tier::Form::Bra && op.imm >= 0 && (op.imm & 15) == 0 && static_cast<uint64_t>(op.imm / 16) < count) leader[static_cast<size_t>(op.imm / 16)] = 1;
  const auto eligible = [&](size_t i) {
    const tier::MOp& op = prog.ops[i];
    // The run loop's own condition for executing an op inside a run (n < 256 && idx + 1 < count && straight, a handler or @!PT), and not a
    // store to global memory: the store gate, which ends a warp's turn at its first one, is checked by the loop before every op and a
    // region never contains the op that could trip it.
    if (i + 1 >= count || !op.straight) return false;
    if (!(op.fn || op.never)) return false;
    if (op.form == tier::Form::Stg || op.ins->op == Op::STG || op.ins->op == Op::ST) return false;
    return true;
  };
  auto u = std::make_shared<Unit>();
  u->spec.sm = sm;
  u->region_at.assign(count, 0);
  size_t i = 0;
  while (i < count) {
    if (!eligible(i)) {
      ++i;
      continue;
    }
    size_t j = i + 1;
    while (j < count && j - i < mgr.max_region && eligible(j) && !leader[j]) ++j;
    if (j - i >= kMinRegion) {
      Region rg;
      rg.first = static_cast<uint32_t>(i);
      rg.k = static_cast<uint32_t>(j - i);
      RegionSpec rs;
      for (size_t x = i; x < j; ++x) {
        const tier::MOp& op = prog.ops[x];
        if (op.fn) rg.maxreg = std::max<uint32_t>(rg.maxreg, op.maxreg);
        OpFacts f;
        f.lo = op.ins->w.lo;
        f.hi = op.ins->w.hi;
        f.cls = op.cls;
        f.guard = op.ins->guard;
        f.guard_not = op.ins->guard_not;
        f.guard_uniform = op.ins->guard_uniform;
        f.unguarded = op.unguarded;
        rs.ops.push_back(f);
      }
      u->region_at[i] = static_cast<uint32_t>(u->regions.size()) + 1;
      u->regions.push_back(rg);
      u->spec.regions.push_back(std::move(rs));
    }
    i = j;
  }
  if (u->regions.empty()) return nullptr;
  stats().units.fetch_add(1);
  stats().regions.fetch_add(u->regions.size());
  mgr.submit(Job::Lookup, u);
  return u;
}

}  // namespace vgpu::sass::jit
