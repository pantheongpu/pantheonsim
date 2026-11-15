// libhiprtc for a simulated machine: ROCm's own libhiprtc behind a cache of its compile results.
//
// A HIP program that compiles kernels at run time (VkFFT's plans, CuPy's and PyTorch's kernels) calls hiprtc, which is not part of the
// HIP runtime the simulator stands in for: it is ROCm's compiler front (libhiprtc.so) over AMD's compiler library (libamd_comgr.so),
// native code that takes about 150 ms a kernel whatever GPU the program then runs it on. The short tests of a suite spend most of
// their time there (inq's FFT tests: 77 to 93% of the wall time, Round sp69), and compile the same kernels in every process.
//
// This library takes the real one's place on the library path (build/shim/libhiprtc.so.6 and .so.7), loads the real one, and
// forwards every entry point to it. What it adds: a program whose compile it can name completely is compiled once. The code, the
// log and the lowered names the real library returned are kept, in memory for the process and in a file for the next, and the
// next compile of the same thing is answered from them: the bytes are the ones the real library returned, never bytes of the
// simulator's making.
//
// "The same thing" is the key (Key below): the source, the program's name, every header given with it, the options, the name
// expressions, the architecture of the current device (which the real library asks the HIP runtime for), the environment
// variables the compiler reads, and the toolchain -- the real libhiprtc (which holds the built-in header) and the compiler library by
// content or build id, and the files the compile reads from the installation (clang's configuration files, the device bitcode
// libraries, the version file) by name, size and time. An entry holds its whole key, and a hit is an equal key, not an equal hash.
//
// Whatever cannot be named completely is compiled by the real library every time, as before (not_cacheable below): a source that
// includes a file the caller did not give (except <hip/hip_runtime.h>, which the built-in header turns into nothing: see
// runtime_header_is_inert), `__DATE__`, `__TIME__` or `__FILE__`, an option outside the list of options that only change the code
// (`-g` puts the compile's temporary directory in the output; `-include` reads a file), a program made with odd arguments. So
// is every call the cache has no answer for: the bitcode, a second compile, a name expression after the compile, the link API
// (hiprtcLink*), which is forwarded and never cached. A failed compile is never kept.
//
// VGPU_HIPRTC_CACHE: `0` (or `off`) forwards everything, `mem` keeps results for the process only, anything else (the default)
// also keeps them in VGPU_HIPRTC_CACHE_DIR, else $XDG_CACHE_HOME/pantheonsim/hiprtc-v1, else ~/.cache/pantheonsim/hiprtc-v1 -- a
// directory that is this user's alone, one file an entry, written under another name and renamed, so a reader sees a whole
// entry or none; VGPU_HIPRTC_CACHE_MAX_MB (default 256, 0 for no limit) removes the least recently used past it.
// VGPU_HIPRTC_CACHE_STATS=1 prints what the cache did when the process ends. VGPU_HIPRTC_REAL names the real library where the
// search does not find it. docs/cli.md has the table; docs/perf-design/rounds (Round sp71) the measurements.
//
// Standalone on purpose: no simulator code is linked in, only libc, libdl and the C++ library.
#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

extern char** environ;

#define VGPU_API extern "C" __attribute__((visibility("default")))

namespace {

// hiprtcResult's values (hip/hiprtc.h).
constexpr int kSuccess = 0, kInvalidProgram = 4, kInternalError = 11;

// ---- small tools

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const unsigned char* b = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}
// A second, independent 64 bits (the file's name is both; equality is the key itself, compared whole).
uint64_t mix(const void* p, size_t n) {
  const unsigned char* b = static_cast<const unsigned char*>(p);
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (n * 0xC2B2AE3D27D4EB4Full);
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 29;
  }
  return h ^ (h >> 32);
}
std::string hex16(uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}
bool read_file(const std::string& path, std::string* out, size_t limit = size_t{1} << 30) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st;
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || static_cast<size_t>(st.st_size) > limit) {
    ::close(fd);
    return false;
  }
  out->resize(static_cast<size_t>(st.st_size));
  size_t got = 0;
  while (got < out->size()) {
    const ssize_t r = ::read(fd, out->data() + got, out->size() - got);
    if (r <= 0) break;
    got += static_cast<size_t>(r);
  }
  ::close(fd);
  return got == out->size();
}
std::string real_path(const std::string& p) {
  char buf[PATH_MAX];
  return ::realpath(p.c_str(), buf) ? std::string(buf) : std::string();
}
std::string dir_of(const std::string& p) {
  const size_t s = p.rfind('/');
  return s == std::string::npos ? std::string(".") : s == 0 ? std::string("/") : p.substr(0, s);
}
// "name size mtime" of a file, or "" when it is not there.
std::string stat_line(const std::string& path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return {};
  return path + " " + std::to_string(static_cast<long long>(st.st_size)) + " " + std::to_string(static_cast<long long>(st.st_mtim.tv_sec)) + "." +
         std::to_string(static_cast<long long>(st.st_mtim.tv_nsec));
}
bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

// The GNU build id of an ELF file's bytes ("" when it has none or the bytes are not a 64-bit ELF file).
std::string build_id_of(const std::string& bytes) {
  if (bytes.size() < sizeof(Elf64_Ehdr) || std::memcmp(bytes.data(), ELFMAG, SELFMAG) != 0 || bytes[EI_CLASS] != ELFCLASS64) return {};
  Elf64_Ehdr eh;
  std::memcpy(&eh, bytes.data(), sizeof eh);
  for (unsigned i = 0; i < eh.e_phnum; ++i) {
    const size_t at = eh.e_phoff + size_t{i} * eh.e_phentsize;
    if (at + sizeof(Elf64_Phdr) > bytes.size()) return {};
    Elf64_Phdr ph;
    std::memcpy(&ph, bytes.data() + at, sizeof ph);
    if (ph.p_type != PT_NOTE || ph.p_offset + ph.p_filesz > bytes.size()) continue;
    size_t p = ph.p_offset;
    const size_t end = ph.p_offset + ph.p_filesz;
    while (p + sizeof(Elf64_Nhdr) <= end) {
      Elf64_Nhdr nh;
      std::memcpy(&nh, bytes.data() + p, sizeof nh);
      const size_t name = p + sizeof nh, desc = name + ((nh.n_namesz + 3) & ~size_t{3});
      if (desc + nh.n_descsz > end) break;
      if (nh.n_type == NT_GNU_BUILD_ID && nh.n_namesz == 4 && std::memcmp(bytes.data() + name, "GNU", 4) == 0) {
        std::string id;
        for (size_t k = 0; k < nh.n_descsz; ++k) {
          char b[3];
          std::snprintf(b, sizeof b, "%02x", static_cast<unsigned char>(bytes[desc + k]));
          id += b;
        }
        return id;
      }
      p = desc + ((nh.n_descsz + 3) & ~size_t{3});
    }
  }
  return {};
}

// ---- the real library

using CreateFn = int (*)(void**, const char*, const char*, int, const char**, const char**);
using CompileFn = int (*)(void*, int, const char**);
using ProgStrFn = int (*)(void*, const char*);
using ProgBufFn = int (*)(void*, char*);
using ProgSizeFn = int (*)(void*, size_t*);
using LoweredFn = int (*)(void*, const char*, const char**);
using DestroyFn = int (*)(void**);
using ErrStrFn = const char* (*)(int);
using VersionFn = int (*)(int*, int*);
using LinkCreateFn = int (*)(unsigned, void*, void**, void**);
using LinkAddFileFn = int (*)(void*, int, const char*, unsigned, void*, void**);
using LinkAddDataFn = int (*)(void*, int, void*, size_t, const char*, unsigned, void*, void**);
using LinkCompleteFn = int (*)(void*, void**, size_t*);
using LinkDestroyFn = int (*)(void*);

struct Real {
  void* handle = nullptr;
  std::string path;   // the file loaded (as found, not resolved)
  CreateFn create = nullptr;
  CompileFn compile = nullptr;
  ProgStrFn add_name = nullptr;
  ProgBufFn get_code = nullptr, get_log = nullptr, get_bitcode = nullptr;
  ProgSizeFn code_size = nullptr, log_size = nullptr, bitcode_size = nullptr;
  LoweredFn lowered = nullptr;
  DestroyFn destroy = nullptr;
  ErrStrFn error_string = nullptr;
  VersionFn version = nullptr;
  LinkCreateFn link_create = nullptr;
  LinkAddFileFn link_add_file = nullptr;
  LinkAddDataFn link_add_data = nullptr;
  LinkCompleteFn link_complete = nullptr;
  LinkDestroyFn link_destroy = nullptr;
  bool ok() const { return handle != nullptr; }
};

// This library's own file, and the name it was loaded under.
std::string self_path() {
  Dl_info info;
  if (::dladdr(reinterpret_cast<void*>(&self_path), &info) && info.dli_fname) return info.dli_fname;
  return {};
}
bool same_file(const std::string& a, const std::string& b) {
  struct stat sa, sb;
  return ::stat(a.c_str(), &sa) == 0 && ::stat(b.c_str(), &sb) == 0 && sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}
// The directories the loader searches for the libraries of `handle`'s object, in its order.
std::vector<std::string> search_dirs(void* handle) {
  std::vector<std::string> dirs;
  Dl_serinfo size;
  if (!handle || ::dlinfo(handle, RTLD_DI_SERINFOSIZE, &size) != 0) return dirs;
  std::vector<char> buf(size.dls_size);
  Dl_serinfo* info = reinterpret_cast<Dl_serinfo*>(buf.data());
  if (::dlinfo(handle, RTLD_DI_SERINFOSIZE, info) != 0 || ::dlinfo(handle, RTLD_DI_SERINFO, info) != 0) return dirs;
  for (unsigned i = 0; i < info->dls_cnt; ++i)
    if (info->dls_serpath[i].dls_name && info->dls_serpath[i].dls_name[0]) dirs.push_back(info->dls_serpath[i].dls_name);
  return dirs;
}
bool system_dir(const std::string& d) { return d.rfind("/lib", 0) == 0 || d.rfind("/usr/lib", 0) == 0; }

// The real libhiprtc: VGPU_HIPRTC_REAL, else the first libhiprtc on the program's library search path that is not this file
// (under the name this one was loaded as, then the other sonames), else ROCm's usual places.
const Real& real() {
  static const Real r = [] {
    Real out;
    const std::string self = self_path();
    std::vector<std::string> candidates;
    if (const char* e = std::getenv("VGPU_HIPRTC_REAL"); e && *e) {
      candidates.push_back(e);
    } else {
      std::vector<std::string> names;
      const size_t slash = self.rfind('/');
      const std::string loaded_as = slash == std::string::npos ? self : self.substr(slash + 1);
      if (loaded_as.rfind("libhiprtc.so", 0) == 0) names.push_back(loaded_as);
      for (const char* n : {"libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so"})
        if (std::find(names.begin(), names.end(), n) == names.end()) names.push_back(n);
      std::vector<std::string> dirs = search_dirs(::dlopen(nullptr, RTLD_LAZY));
      for (const char* v : {"VGPU_ROCM_PATH", "ROCM_PATH", "HIP_PATH"})
        if (const char* e = std::getenv(v); e && *e) dirs.push_back(std::string(e) + "/lib");
      dirs.push_back("/opt/rocm/lib");
      for (const std::string& d : dirs)
        for (const std::string& n : names) candidates.push_back(d + "/" + n);
    }
    for (const std::string& c : candidates) {
      struct stat st;
      if (::stat(c.c_str(), &st) != 0 || (!self.empty() && same_file(c, self))) continue;
      void* h = ::dlopen(c.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!h) continue;
      const auto sym = [&](const char* name) { return ::dlsym(h, name); };
      Real t;
      t.handle = h, t.path = c;
      t.create = reinterpret_cast<CreateFn>(sym("hiprtcCreateProgram"));
      t.compile = reinterpret_cast<CompileFn>(sym("hiprtcCompileProgram"));
      t.add_name = reinterpret_cast<ProgStrFn>(sym("hiprtcAddNameExpression"));
      t.get_code = reinterpret_cast<ProgBufFn>(sym("hiprtcGetCode"));
      t.get_log = reinterpret_cast<ProgBufFn>(sym("hiprtcGetProgramLog"));
      t.get_bitcode = reinterpret_cast<ProgBufFn>(sym("hiprtcGetBitcode"));
      t.code_size = reinterpret_cast<ProgSizeFn>(sym("hiprtcGetCodeSize"));
      t.log_size = reinterpret_cast<ProgSizeFn>(sym("hiprtcGetProgramLogSize"));
      t.bitcode_size = reinterpret_cast<ProgSizeFn>(sym("hiprtcGetBitcodeSize"));
      t.lowered = reinterpret_cast<LoweredFn>(sym("hiprtcGetLoweredName"));
      t.destroy = reinterpret_cast<DestroyFn>(sym("hiprtcDestroyProgram"));
      t.error_string = reinterpret_cast<ErrStrFn>(sym("hiprtcGetErrorString"));
      t.version = reinterpret_cast<VersionFn>(sym("hiprtcVersion"));
      t.link_create = reinterpret_cast<LinkCreateFn>(sym("hiprtcLinkCreate"));
      t.link_add_file = reinterpret_cast<LinkAddFileFn>(sym("hiprtcLinkAddFile"));
      t.link_add_data = reinterpret_cast<LinkAddDataFn>(sym("hiprtcLinkAddData"));
      t.link_complete = reinterpret_cast<LinkCompleteFn>(sym("hiprtcLinkComplete"));
      t.link_destroy = reinterpret_cast<LinkDestroyFn>(sym("hiprtcLinkDestroy"));
      // (A library without the program functions is not a libhiprtc this one can stand in front of.)
      if (!t.create || !t.compile || !t.destroy || !t.get_code || !t.code_size) {
        ::dlclose(h);
        continue;
      }
      out = t;
      break;
    }
    if (!out.ok() && !quiet())
      std::fprintf(stderr, "VirtualGPU hiprtc: ROCm's libhiprtc was not found (the library path, ROCM_PATH/lib, /opt/rocm/lib; VGPU_HIPRTC_REAL names it). "
                           "Run-time compilation needs ROCm's compiler: every hiprtc call fails.\n");
    return out;
  }();
  return r;
}

// The real library may call its own entry points through the names this library defines: such a call is passed straight on.
thread_local int t_inside = 0;
struct Inside {
  Inside() { ++t_inside; }
  ~Inside() { --t_inside; }
};

// ---- settings

enum class Mode { Off, Memory, Disk };
Mode mode() {
  static const Mode m = [] {
    const char* e = std::getenv("VGPU_HIPRTC_CACHE");
    if (!e || !*e) return Mode::Disk;
    const std::string v = e;
    if (v == "0" || v == "off" || v == "false") return Mode::Off;
    if (v == "mem" || v == "memory") return Mode::Memory;
    return Mode::Disk;
  }();
  return m;
}
uint64_t max_bytes() {
  static const uint64_t n = [] {
    const char* e = std::getenv("VGPU_HIPRTC_CACHE_MAX_MB");
    return (e && *e ? std::strtoull(e, nullptr, 10) : 256ull) << 20;
  }();
  return n;
}
bool ours_alone(const struct stat& st, mode_t kind) { return (st.st_mode & S_IFMT) == kind && st.st_uid == ::geteuid() && (st.st_mode & 022) == 0; }

// The cache's directory, made if it can be, or "" (no home, a relative path, or a directory that is not this user's alone).
const std::string& cache_dir() {
  static const std::string& dir = *new std::string([] {   // (never destroyed: read when the process ends)
    std::string d;
    bool own = false;   // (given by name: only the directory itself is made, not its parents)
    if (const char* e = std::getenv("VGPU_HIPRTC_CACHE_DIR"); e && *e) d = e, own = true;
    else if (const char* x = std::getenv("XDG_CACHE_HOME"); x && x[0] == '/') d = std::string(x) + "/pantheonsim/hiprtc-v1";
    else if (const char* h = std::getenv("HOME"); h && h[0] == '/') d = std::string(h) + "/.cache/pantheonsim/hiprtc-v1";
    if (d.empty() || d[0] != '/') return std::string();
    struct stat st;
    if (::lstat(d.c_str(), &st) != 0) {
      if (!own) {
        const std::string parent = dir_of(d);
        ::mkdir(dir_of(parent).c_str(), 0700);
        ::mkdir(parent.c_str(), 0700);
      }
      ::mkdir(d.c_str(), 0700);
      if (::lstat(d.c_str(), &st) != 0) return std::string();
    }
    return ours_alone(st, S_IFDIR) ? d : std::string();
  }());
  return dir;
}

// ---- statistics

struct Stats {
  std::atomic<uint64_t> compiles{0}, memory_hits{0}, disk_hits{0}, real_compiles{0}, stored{0}, not_cacheable{0}, bad_entries{0}, evicted{0};
  std::atomic<uint64_t> real_ns{0}, saved_ns{0};
  std::mutex mu;
  std::map<std::string, uint64_t> reasons;
};
Stats& stats() {
  static Stats* s = new Stats;   // (never destroyed: programs are compiled from static destructors too)
  return *s;
}
void print_stats() {
  Stats& s = stats();
  std::string why;
  {
    std::lock_guard<std::mutex> lock(s.mu);
    for (const auto& [r, n] : s.reasons) why += (why.empty() ? "" : "; ") + r + " x" + std::to_string(n);
  }
  std::fprintf(stderr,
               "VirtualGPU hiprtc-cache: %llu compiles: %llu from memory, %llu from %s, %llu by the real library (%.2f s; %llu not cacheable%s%s), "
               "%llu stored, %llu bad entries, %llu removed by the size cap; about %.2f s of compiling saved\n",
               static_cast<unsigned long long>(s.compiles.load()), static_cast<unsigned long long>(s.memory_hits.load()),
               static_cast<unsigned long long>(s.disk_hits.load()), mode() != Mode::Disk || cache_dir().empty() ? "disk (off)" : cache_dir().c_str(),
               static_cast<unsigned long long>(s.real_compiles.load()), static_cast<double>(s.real_ns.load()) * 1e-9,
               static_cast<unsigned long long>(s.not_cacheable.load()), why.empty() ? "" : ": ", why.c_str(),
               static_cast<unsigned long long>(s.stored.load()), static_cast<unsigned long long>(s.bad_entries.load()),
               static_cast<unsigned long long>(s.evicted.load()), static_cast<double>(s.saved_ns.load()) * 1e-9);
}
void stats_at_exit() {
  static const bool on = [] {
    const char* e = std::getenv("VGPU_HIPRTC_CACHE_STATS");
    if (!(e && *e && *e != '0')) return false;
    std::atexit(print_stats);
    return true;
  }();
  (void)on;
}
void note_reason(const std::string& r) {
  Stats& s = stats();
  s.not_cacheable.fetch_add(1);
  std::lock_guard<std::mutex> lock(s.mu);
  ++s.reasons[r];
}

// ---- the toolchain: what the real compile is made of, besides its inputs

// Loaded objects whose file name contains `part`: their paths as the loader has them.
struct Loaded {
  const char* part;
  std::vector<std::string> paths;
};
int find_loaded(struct dl_phdr_info* info, size_t, void* data) {
  Loaded* l = static_cast<Loaded*>(data);
  if (info->dlpi_name && std::strstr(info->dlpi_name, l->part)) l->paths.push_back(info->dlpi_name);
  return 0;
}
std::string loaded_object(const char* part) {
  Loaded l{part, {}};
  ::dl_iterate_phdr(find_loaded, &l);
  return l.paths.size() == 1 ? real_path(l.paths[0]) : std::string();
}
// The library names starting with `prefix` that the bytes of a library mention (what it will dlopen).
std::vector<std::string> names_in(const std::string& bytes, const char* prefix) {
  std::vector<std::string> out;
  const size_t n = std::strlen(prefix);
  for (size_t at = bytes.find(prefix); at != std::string::npos; at = bytes.find(prefix, at + 1)) {
    size_t end = at + n;
    while (end < bytes.size() && bytes[end] > ' ' && bytes[end] < 127 && end - at < 64) ++end;
    const std::string name = bytes.substr(at, end - at);
    if (std::find(out.begin(), out.end(), name) == out.end()) out.push_back(name);
  }
  return out;
}

struct Toolchain {
  bool known = false;         // every part below was found: results may be kept in files
  std::string text;           // goes into the key
  std::string why;            // what was missing, when not known
  std::string comgr, builtins;   // resolved paths, to hold against what the real compile then loads ("": no such library)
  std::string header_bytes;      // the file the built-in header is in (runtime_header_is_inert reads it)
  std::vector<std::string> roots;   // the installation's roots, for the headers the built-in one covers
};

// A library the real one will dlopen by name: the one already loaded, else the first on the real library's search path outside
// the system directories (there the loader's cache, which this does not read, has the say: such a library is known only once the
// real compile has loaded it).
std::string locate_dependency(const std::vector<std::string>& names, const char* part) {
  const std::string loaded = loaded_object(part);
  if (!loaded.empty()) return loaded;
  for (const std::string& d : search_dirs(real().handle)) {
    if (system_dir(d)) continue;
    for (const std::string& n : names) {
      const std::string p = real_path(d + "/" + n);
      if (!p.empty()) return p;
    }
  }
  return {};
}
void add_tree(const std::string& dir, const char* suffix, std::vector<std::string>* lines) {
  DIR* d = ::opendir(dir.c_str());
  if (!d) return;
  std::vector<std::string> names;
  while (const dirent* e = ::readdir(d)) names.push_back(e->d_name);
  ::closedir(d);
  std::sort(names.begin(), names.end());
  const size_t ns = std::strlen(suffix);
  for (const std::string& n : names) {
    if (n.size() < ns || n.compare(n.size() - ns, ns, suffix) != 0) continue;
    const std::string line = stat_line(dir + "/" + n);
    if (!line.empty()) lines->push_back(line);
  }
}

Toolchain make_toolchain() {
  Toolchain t;
  const Real& r = real();
  if (!r.ok()) return t.why = "no real library", t;
  std::string text;
  // The real libhiprtc, by content.
  const std::string rtc_path = real_path(r.path);
  std::string rtc;
  if (rtc_path.empty() || !read_file(rtc_path, &rtc)) return t.why = "the real libhiprtc cannot be read", t;
  text += "hiprtc " + rtc_path + " " + build_id_of(rtc) + " " + hex16(fnv(rtc.data(), rtc.size())) + hex16(mix(rtc.data(), rtc.size())) + "\n";
  // The header every program is compiled with is inside the real libhiprtc (ROCm 6.4 and 7.1; their libhiprtc-builtins.so is
  // not what libhiprtc reads). A libhiprtc that names a built-in header library, or has one loaded, reads it from there: then
  // that library is part of the toolchain, by content.
  t.header_bytes = rtc;
  const std::vector<std::string> bnames = names_in(rtc, "libhiprtc-builtins.so");
  if (!bnames.empty() || !loaded_object("libhiprtc-builtins").empty()) {
    t.builtins = locate_dependency(bnames.empty() ? std::vector<std::string>{"libhiprtc-builtins.so"} : bnames, "libhiprtc-builtins");
    if (t.builtins.empty() || !read_file(t.builtins, &t.header_bytes)) return t.why = "the built-in header library the real libhiprtc names was not found", t;
    text += "builtins " + t.builtins + " " + build_id_of(t.header_bytes) + " " + hex16(fnv(t.header_bytes.data(), t.header_bytes.size())) +
            hex16(mix(t.header_bytes.data(), t.header_bytes.size())) + "\n";
  }
  // The compiler library: its build id (it is some 150 MB: not read whole), size and time.
  std::vector<std::string> cnames = names_in(rtc, "libamd_comgr.so");
  if (cnames.empty()) return t.why = "the real libhiprtc names no compiler library", t;
  t.comgr = locate_dependency(cnames, "libamd_comgr");
  if (t.comgr.empty()) return t.why = "libamd_comgr was not found on the real library's path", t;
  {
    std::string head;
    const int fd = ::open(t.comgr.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return t.why = "libamd_comgr cannot be read", t;
    head.resize(1 << 16);
    const ssize_t got = ::read(fd, head.data(), head.size());
    ::close(fd);
    head.resize(got > 0 ? static_cast<size_t>(got) : 0);
    text += "comgr " + stat_line(t.comgr) + " " + build_id_of(head) + "\n";
  }
  // What the compile reads from the installation: clang's configuration files, the device bitcode libraries, the version files.
  // The roots: the compiler library's (lib/..), and the ones the environment names.
  std::vector<std::string> roots = {real_path(dir_of(t.comgr) + "/..")};
  for (const char* v : {"ROCM_PATH", "HIP_PATH"})
    if (const char* e = std::getenv(v); e && *e)
      if (const std::string p = real_path(e); !p.empty() && std::find(roots.begin(), roots.end(), p) == roots.end()) roots.push_back(p);
  std::vector<std::string> llvms;
  for (const std::string& root : roots)
    for (const char* sub : {"/llvm", "/lib/llvm"})
      if (const std::string p = real_path(root + sub); !p.empty() && std::find(llvms.begin(), llvms.end(), p) == llvms.end()) llvms.push_back(p);
  if (const char* e = std::getenv("LLVM_PATH"); e && *e)
    if (const std::string p = real_path(e); !p.empty() && std::find(llvms.begin(), llvms.end(), p) == llvms.end()) llvms.push_back(p);
  std::vector<std::string> lines;
  size_t bitcode = 0;
  for (const std::string& llvm : llvms) {
    add_tree(llvm + "/bin", ".cfg", &lines);
    if (DIR* d = ::opendir((llvm + "/lib/clang").c_str())) {
      std::vector<std::string> versions;
      while (const dirent* e = ::readdir(d))
        if (e->d_name[0] != '.') versions.push_back(e->d_name);
      ::closedir(d);
      std::sort(versions.begin(), versions.end());
      for (const std::string& v : versions) {
        const size_t before = lines.size();
        add_tree(llvm + "/lib/clang/" + v + "/lib/amdgcn/bitcode", ".bc", &lines);
        bitcode += lines.size() - before;
      }
    }
  }
  for (const std::string& root : roots)
    for (const char* f : {"/share/hip/version", "/bin/.hipVersion", "/lib/.hipVersion", "/.info/version"})
      if (const std::string line = stat_line(root + f); !line.empty()) lines.push_back(line);
  if (!bitcode) return t.why = "the device bitcode libraries were not found under the compiler library's installation", t;
  for (const std::string& l : lines) text += l + "\n";
  t.roots = roots;
  t.text = text;
  t.known = true;
  return t;
}
struct ToolchainSlot {
  std::mutex mu;
  std::shared_ptr<const Toolchain> now;
};
ToolchainSlot& toolchain_slot() {
  static ToolchainSlot* s = new ToolchainSlot;   // (never destroyed)
  return *s;
}
std::shared_ptr<const Toolchain> toolchain() {
  ToolchainSlot& s = toolchain_slot();
  std::lock_guard<std::mutex> lock(s.mu);
  if (!s.now) s.now = std::make_shared<Toolchain>(make_toolchain());
  return s.now;
}
// After a real compile: the libraries it loaded must be the ones the key named. Where they are not (or were not known before),
// the toolchain is worked out again now that they are loaded; the result of that compile is kept only if they agreed.
bool toolchain_held(const std::shared_ptr<const Toolchain>& used) {
  const std::string comgr = loaded_object("libamd_comgr"), builtins = loaded_object("libhiprtc-builtins");
  const bool held = used->known && comgr == used->comgr && builtins == used->builtins && !comgr.empty();
  if (!held) {
    ToolchainSlot& s = toolchain_slot();
    std::lock_guard<std::mutex> lock(s.mu);
    s.now = std::make_shared<Toolchain>(make_toolchain());
  }
  return held;
}

// The architecture the real library compiles for when no option says: the current device's, asked of the HIP runtime in the
// process as the real library asks (hipGetDevice, hipGetDevicePropertiesR0600). "" when there is no runtime to ask.
std::string device_arch() {
  using GetDevice = int (*)(int*);
  using GetProps = int (*)(void*, int);
  auto get_device = reinterpret_cast<GetDevice>(::dlsym(RTLD_DEFAULT, "hipGetDevice"));
  auto get_props = reinterpret_cast<GetProps>(::dlsym(RTLD_DEFAULT, "hipGetDevicePropertiesR0600"));
  if (!get_device || !get_props) {
    // A program that links hiprtc and not the runtime: the real library then loads libamdhip64.so by that name, and so does this.
    static void* const runtime = ::dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
    if (!runtime) return {};
    get_device = reinterpret_cast<GetDevice>(::dlsym(runtime, "hipGetDevice"));
    get_props = reinterpret_cast<GetProps>(::dlsym(runtime, "hipGetDevicePropertiesR0600"));
    if (!get_device || !get_props) return {};
  }
  int id = -1;
  if (get_device(&id) != 0) return {};
  // hipDeviceProp_tR0600: name[256], uuid[16], luid[8], luidDeviceNodeMask (4), totalGlobalMem (8) ... and gcnArchName[256] at the
  // end of the fields the structure began with. Rather than its layout: the structure is filled into a buffer larger than any
  // release's, and the architecture is the text in it that names a gfx target.
  std::vector<char> props(16384, 0);
  if (get_props(props.data(), id) != 0) return {};
  for (size_t at = 256; at + 3 < props.size(); ++at)   // (past name[256])
    if (std::memcmp(&props[at], "gfx", 3) == 0 && (at == 0 || props[at - 1] == 0 || props[at - 1] == '-')) {
      size_t begin = at;
      while (begin > 256 && props[begin - 1] != 0) --begin;
      return std::string(&props[begin]);
    }
  return {};
}

// The environment the compile can read: every variable of the families below, sorted.
std::string compile_environment() {
  static const char* const kPrefixes[] = {"HIPRTC_", "HIP_", "AMD_", "ROCM_", "LLVM_", "CLANG_", "CCC_", "RC_DEBUG_", "GPU_"};
  static const char* const kNames[] = {"CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "OBJCPLUS_INCLUDE_PATH",
                                       "COMPILER_PATH", "LIBRARY_PATH", "SOURCE_DATE_EPOCH", "SDKROOT", "TMPDIR"};
  std::vector<std::string> vars;
  for (char** e = environ; e && *e; ++e) {
    const char* eq = std::strchr(*e, '=');
    if (!eq) continue;
    const std::string name(*e, static_cast<size_t>(eq - *e));
    bool in = false;
    for (const char* p : kPrefixes) in = in || name.rfind(p, 0) == 0;
    for (const char* n : kNames) in = in || name == n;
    if (in && name != "TMPDIR") vars.push_back(*e);   // (TMPDIR: where the temporary files go, not what is in them)
  }
  std::sort(vars.begin(), vars.end());
  std::string out;
  for (const std::string& v : vars) out += v + "\n";
  return out;
}
bool include_environment_set() {
  for (const char* n : {"CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"})
    if (const char* e = std::getenv(n); e && *e) return true;
  return false;
}

// ---- which programs can be named completely

// Text without its comments and with its continued lines joined (what the preprocessor's directives are read from).
std::string plain_text(const std::string& s) {
  std::string joined;
  joined.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == '\n') { ++i; continue; }
    if (s[i] == '\\' && i + 2 < s.size() && s[i + 1] == '\r' && s[i + 2] == '\n') { i += 2; continue; }
    joined += s[i];
  }
  std::string out;
  out.reserve(joined.size());
  for (size_t i = 0; i < joined.size(); ++i) {
    const char c = joined[i];
    if (c == '/' && i + 1 < joined.size() && joined[i + 1] == '/') {
      while (i < joined.size() && joined[i] != '\n') ++i;
      out += '\n';
    } else if (c == '/' && i + 1 < joined.size() && joined[i + 1] == '*') {
      const size_t end = joined.find("*/", i + 2);
      for (size_t k = i; k < (end == std::string::npos ? joined.size() : end + 2); ++k)
        if (joined[k] == '\n') out += '\n';
      out += ' ';
      if (end == std::string::npos) break;
      i = end + 1;
    } else if (c == '"' || c == '\'') {   // (a literal: copied whole, so that "//" inside one is not a comment)
      out += c;
      for (++i; i < joined.size() && joined[i] != c && joined[i] != '\n'; ++i) {
        out += joined[i];
        if (joined[i] == '\\' && i + 1 < joined.size()) out += joined[++i];
      }
      if (i < joined.size()) out += joined[i];
    } else {
      out += c;
    }
  }
  return out;
}

// Whether a file is hip_runtime.h as ROCm ships it: everything in it inside `#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_H`, so that
// it is empty to a compile that has the macro already.
constexpr const char* kGuard = "HIP_INCLUDE_HIP_HIP_RUNTIME_H";
bool wholly_guarded(const std::string& bytes) {
  const std::string text = plain_text(bytes);
  int depth = 0;
  bool opened = false, closed = false;
  size_t at = 0;
  while (at < text.size()) {
    size_t end = text.find('\n', at);
    if (end == std::string::npos) end = text.size();
    size_t p = at;
    while (p < end && (text[p] == ' ' || text[p] == '\t' || text[p] == '\r')) ++p;
    if (p < end) {
      if (closed) return false;   // something after the guard's #endif
      if (text[p] != '#') {
        if (!opened) return false;   // something before the guard
      } else {
        ++p;
        while (p < end && (text[p] == ' ' || text[p] == '\t')) ++p;
        size_t w = p;
        while (w < end && (std::isalnum(static_cast<unsigned char>(text[w])) || text[w] == '_')) ++w;
        const std::string word = text.substr(p, w - p);
        if (!opened) {
          size_t a = w;
          while (a < end && (text[a] == ' ' || text[a] == '\t')) ++a;
          size_t b = a;
          while (b < end && (std::isalnum(static_cast<unsigned char>(text[b])) || text[b] == '_')) ++b;
          if (word != "ifndef" || text.substr(a, b - a) != kGuard) return false;
          opened = true, depth = 1;
        } else if (word == "if" || word == "ifdef" || word == "ifndef") {
          ++depth;
        } else if (word == "endif") {
          if (--depth == 0) closed = true;
        }
      }
    }
    at = end + 1;
  }
  return opened && closed;
}

// `#include <hip/hip_runtime.h>` in a program does nothing where the built-in header has defined the file's guard and every
// hip_runtime.h the compile could find is ROCm's, guarded whole. `key` gets the files (name, size, time, content).
bool runtime_header_is_inert(const Toolchain& t, std::string* key) {
  if (t.header_bytes.find(std::string("#define ") + kGuard) == std::string::npos) return false;
  std::vector<std::string> dirs;
  for (const std::string& r : t.roots) dirs.push_back(r + "/include");
  dirs.push_back("/usr/include");
  dirs.push_back("/usr/local/include");
  size_t found = 0;
  for (const std::string& d : dirs) {
    const std::string p = d + "/hip/hip_runtime.h";
    std::string bytes;
    struct stat st;
    if (::stat(p.c_str(), &st) != 0) continue;
    if (!read_file(p, &bytes, 1 << 22) || !wholly_guarded(bytes)) return false;
    *key += "header " + stat_line(p) + " " + hex16(fnv(bytes.data(), bytes.size())) + "\n";
    ++found;
  }
  return found > 0;
}

// The options that change only the code that comes out (and its diagnostics): anything else is the real library's every time.
// A macro's name must be an ordinary one: the reserved ones (_X, __x, HIP...) steer the built-in header.
bool pure_option(const std::string& o, bool* include_path) {
  static const char* const kExact[] = {"-w", "-ffast-math", "-fno-fast-math", "-ffinite-math-only", "-fno-finite-math-only", "-funsafe-math-optimizations",
                                       "-fno-unsafe-math-optimizations", "-fno-math-errno", "-fmath-errno", "-freciprocal-math", "-fno-reciprocal-math",
                                       "-fassociative-math", "-fno-associative-math", "-fno-signed-zeros", "-fsigned-zeros", "-ftrapping-math",
                                       "-fno-trapping-math", "-fgpu-rdc", "-fno-gpu-rdc", "-fno-exceptions", "-fexceptions", "-fno-rtti", "-frtti",
                                       "-funroll-loops", "-fno-unroll-loops", "-finline-functions", "-fno-inline-functions", "-fno-inline", "-g0",
                                       "-fgpu-flush-denormals-to-zero", "-fno-gpu-flush-denormals-to-zero", "-fcuda-flush-denormals-to-zero",
                                       "-fgpu-approx-transcendentals", "-fno-gpu-approx-transcendentals", "-fapprox-func", "-fno-approx-func",
                                       "-O", "-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Ofast"};
  static const char* const kPrefix[] = {"-std=", "-W", "--offload-arch=", "-mcpu=", "-ffp-contract=", "-ftemplate-depth=", "-fconstexpr-steps=",
                                        "-fconstexpr-depth=", "-ferror-limit=", "-fvisibility=", "-ffp-model=", "-fdenormal-fp-math="};
  for (const char* e : kExact)
    if (o == e) return true;
  if (o.rfind("-Wl,", 0) == 0 || o.rfind("-Wa,", 0) == 0 || o.rfind("-Wp,", 0) == 0) return false;   // (handed to another tool)
  for (const char* p : kPrefix)
    if (o.rfind(p, 0) == 0 && o.find('@') == std::string::npos) return true;
  if ((o.rfind("-D", 0) == 0 || o.rfind("-U", 0) == 0) && o.size() > 2) {
    const std::string name = o.substr(2, o.find('=') == std::string::npos ? std::string::npos : o.find('=') - 2);
    if (name.empty() || name[0] == '_' || name.rfind("HIP", 0) == 0 || name.rfind("CUDA", 0) == 0) return false;
    for (const char c : name)
      if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    return true;
  }
  if ((o.rfind("-I", 0) == 0 && o.size() > 2) || (o.rfind("-isystem", 0) == 0 && o.size() > 8)) return *include_path = true;
  return false;
}

struct Entry {
  std::string key;
  std::string code;
  uint64_t log_size = 0;   // what hiprtcGetProgramLogSize said
  std::string log;         // what hiprtcGetProgramLog wrote (log_size bytes, or more where it wrote more)
  std::vector<std::string> lowered;   // one for each name expression, in order
};

struct Prog {
  enum class State { Fresh, Cached, Real };
  State state = State::Fresh;
  std::string src, name;
  std::vector<std::string> headers, include_names, name_exprs, options;
  void* real = nullptr;
  std::shared_ptr<const Entry> entry;
};

// "" when the compile of `p` with `options` can be named completely, else why not. Adds what it relied on to `extra`.
std::string not_cacheable(const Prog& p, const std::vector<std::string>& options, const Toolchain& t, std::string* extra) {
  bool include_path = false;
  for (const std::string& o : options)
    if (!pure_option(o, &include_path)) return "an option outside the list (" + o.substr(0, 24) + ")";
  bool external = false, runtime_header = false;
  const auto scan = [&](const std::string& raw) -> const char* {
    for (const char* word : {"__DATE__", "__TIME__", "__TIMESTAMP__", "__FILE__", "__BASE_FILE__", "__has_include", "__has_embed", "?" "?=", "%:"})
      if (raw.find(word) != std::string::npos) return word;
    const std::string text = plain_text(raw);
    for (size_t at = text.find('#'); at != std::string::npos; at = text.find('#', at + 1)) {
      size_t w = at + 1;
      while (w < text.size() && (text[w] == ' ' || text[w] == '\t')) ++w;
      size_t e = w;
      while (e < text.size() && (std::isalnum(static_cast<unsigned char>(text[e])) || text[e] == '_')) ++e;
      const std::string word = text.substr(w, e - w);
      if (word == "include_next" || word == "import" || word == "embed" || word == "include_macros") return "#include_next, #import or #embed";
      if (word != "include") continue;
      while (e < text.size() && (text[e] == ' ' || text[e] == '\t')) ++e;
      if (e >= text.size() || (text[e] != '<' && text[e] != '"')) return "an #include that is not a file name";
      const char close = text[e] == '<' ? '>' : '"';
      const size_t end = text.find_first_of(std::string(1, close) + "\n", e + 1);
      if (end == std::string::npos || text[end] != close) return "an #include that is not a file name";
      const std::string file = text.substr(e + 1, end - e - 1);
      if (std::find(p.include_names.begin(), p.include_names.end(), file) != p.include_names.end()) continue;
      if (file == "hip/hip_runtime.h") runtime_header = true;
      else external = true;
    }
    return nullptr;
  };
  if (const char* why = scan(p.src)) return std::string("the source has ") + why;
  for (const std::string& h : p.headers)
    if (const char* why = scan(h)) return std::string("a header has ") + why;
  if (external) return "an #include of a file that was not given";
  if (runtime_header) {
    if (include_path || include_environment_set()) return "<hip/hip_runtime.h> with an include path";
    if (!runtime_header_is_inert(t, extra)) return "<hip/hip_runtime.h> is not the guarded header";
  }
  // (The built-in header is included by name, and the compile looks in the working directory first.)
  struct stat st;
  if (::stat("hiprtc_runtime.h", &st) == 0) return "hiprtc_runtime.h in the working directory";
  return {};
}

void put(std::string* out, const std::string& s) {
  const uint64_t n = s.size();
  out->append(reinterpret_cast<const char*>(&n), sizeof n);
  out->append(s);
}
void put(std::string* out, const std::vector<std::string>& v) {
  const uint64_t n = v.size();
  out->append(reinterpret_cast<const char*>(&n), sizeof n);
  for (const std::string& s : v) put(out, s);
}
bool get(const std::string& in, size_t* at, std::string* s) {
  uint64_t n;
  if (*at + sizeof n > in.size()) return false;
  std::memcpy(&n, in.data() + *at, sizeof n);
  *at += sizeof n;
  if (n > in.size() - *at) return false;
  s->assign(in, *at, static_cast<size_t>(n));
  *at += static_cast<size_t>(n);
  return true;
}
bool get(const std::string& in, size_t* at, std::vector<std::string>* v) {
  uint64_t n;
  if (*at + sizeof n > in.size()) return false;
  std::memcpy(&n, in.data() + *at, sizeof n);
  *at += sizeof n;
  if (n > in.size()) return false;
  v->resize(static_cast<size_t>(n));
  for (std::string& s : *v)
    if (!get(in, at, &s)) return false;
  return true;
}

constexpr const char kKeyTag[] = "vgpu-hiprtc-key-1";
std::string make_key(const Prog& p, const std::vector<std::string>& options, const std::string& toolchain_text, const std::string& arch,
                     const std::string& extra) {
  std::string k;
  put(&k, std::string(kKeyTag));
  put(&k, toolchain_text);
  put(&k, arch);
  put(&k, compile_environment());
  put(&k, extra);
  put(&k, p.name);
  put(&k, p.src);
  put(&k, p.include_names);
  put(&k, p.headers);
  put(&k, options);
  put(&k, p.name_exprs);
  return k;
}

// ---- the store: memory, then files

struct Memory {
  std::mutex mu;
  std::unordered_map<std::string, std::shared_ptr<const Entry>> by_key;
  uint64_t bytes = 0;
};
Memory& memory() {
  static Memory* m = new Memory;
  return *m;
}
constexpr uint64_t kMemoryCap = uint64_t{512} << 20;

// An entry's file: "VGPUHRTC", the format's number, the lengths of the key and of the rest, their two checks, the key, the rest.
constexpr char kMagic[8] = {'V', 'G', 'P', 'U', 'H', 'R', 'T', 'C'};
constexpr uint32_t kFormat = 1;
struct FileHead {
  char magic[8];
  uint32_t format;
  uint32_t reserved;
  uint64_t key_size, body_size, key_check, body_check;
};
std::string file_of(const std::string& key) { return cache_dir() + "/" + hex16(fnv(key.data(), key.size())) + hex16(mix(key.data(), key.size())) + ".rtc"; }

std::shared_ptr<const Entry> read_entry(const std::string& key) {
  if (cache_dir().empty()) return nullptr;
  const std::string path = file_of(key);
  const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return nullptr;
  const auto bad = [&]() -> std::shared_ptr<const Entry> {
    stats().bad_entries.fetch_add(1);
    ::close(fd);
    return nullptr;
  };
  struct stat st;
  if (::fstat(fd, &st) != 0 || !ours_alone(st, S_IFREG)) return bad();
  std::string all(static_cast<size_t>(st.st_size), '\0');
  size_t got = 0;
  while (got < all.size()) {
    const ssize_t r = ::read(fd, all.data() + got, all.size() - got);
    if (r <= 0) break;
    got += static_cast<size_t>(r);
  }
  FileHead h;
  if (got != all.size() || all.size() < sizeof h) return bad();
  std::memcpy(&h, all.data(), sizeof h);
  if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || h.format != kFormat || h.key_size > all.size() || h.body_size > all.size() ||
      sizeof h + h.key_size + h.body_size != all.size())
    return bad();
  const char* k = all.data() + sizeof h;
  const char* b = k + h.key_size;
  if (fnv(k, h.key_size) != h.key_check || fnv(b, h.body_size) != h.body_check) return bad();
  if (h.key_size != key.size() || std::memcmp(k, key.data(), key.size()) != 0) {   // (another key under the same name: not this entry)
    ::close(fd);
    return nullptr;
  }
  auto e = std::make_shared<Entry>();
  e->key = key;
  const std::string body(b, h.body_size);
  size_t at = 0;
  std::string log_size;
  if (!get(body, &at, &e->code) || !get(body, &at, &log_size) || log_size.size() != 8 || !get(body, &at, &e->log) || !get(body, &at, &e->lowered) ||
      at != body.size())
    return bad();
  std::memcpy(&e->log_size, log_size.data(), 8);
  ::futimens(fd, nullptr);   // (use renews the file's time: the size cap removes the least recently used first)
  ::close(fd);
  return e;
}

// Removes the least recently used entries past the cap (down to nine tenths of it), and temporary files an hour old.
void trim() {
  const uint64_t cap = max_bytes();
  DIR* d = ::opendir(cache_dir().c_str());
  if (!d) return;
  struct Item {
    std::string path;
    uint64_t size;
    struct timespec when;
  };
  std::vector<Item> items;
  uint64_t total = 0;
  const time_t now = ::time(nullptr);
  while (const dirent* e = ::readdir(d)) {
    const std::string n = e->d_name;
    const std::string p = cache_dir() + "/" + n;
    struct stat st;
    if (n == "." || n == ".." || ::lstat(p.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
    if (n.rfind(".tmp.", 0) == 0) {
      if (st.st_mtim.tv_sec + 3600 < now) ::unlink(p.c_str());
      continue;
    }
    if (n.size() < 4 || n.compare(n.size() - 4, 4, ".rtc") != 0) continue;
    items.push_back({p, static_cast<uint64_t>(st.st_size), st.st_mtim});
    total += static_cast<uint64_t>(st.st_size);
  }
  ::closedir(d);
  if (!cap || total <= cap) return;
  std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
    return a.when.tv_sec != b.when.tv_sec ? a.when.tv_sec < b.when.tv_sec : a.when.tv_nsec < b.when.tv_nsec;
  });
  for (const Item& it : items) {
    if (total <= cap / 10 * 9) break;
    if (::unlink(it.path.c_str()) == 0) {
      total -= it.size;
      stats().evicted.fetch_add(1);
    }
  }
}

void write_entry(const Entry& e) {
  if (cache_dir().empty()) return;
  std::string body;
  put(&body, e.code);
  put(&body, std::string(reinterpret_cast<const char*>(&e.log_size), 8));
  put(&body, e.log);
  put(&body, e.lowered);
  FileHead h{};
  std::memcpy(h.magic, kMagic, sizeof kMagic);
  h.format = kFormat;
  h.key_size = e.key.size(), h.body_size = body.size();
  h.key_check = fnv(e.key.data(), e.key.size()), h.body_check = fnv(body.data(), body.size());
  static std::atomic<uint64_t> serial{0};
  const std::string tmp = cache_dir() + "/.tmp." + std::to_string(static_cast<long long>(::getpid())) + "." + std::to_string(serial.fetch_add(1)) + "." +
                          hex16(static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) return;
  const auto write_all = [&](const char* p, size_t n) {
    while (n) {
      const ssize_t w = ::write(fd, p, n);
      if (w <= 0) return false;
      p += w, n -= static_cast<size_t>(w);
    }
    return true;
  };
  const bool ok = write_all(reinterpret_cast<const char*>(&h), sizeof h) && write_all(e.key.data(), e.key.size()) && write_all(body.data(), body.size());
  if (::close(fd) != 0 || !ok || ::rename(tmp.c_str(), file_of(e.key).c_str()) != 0) {
    ::unlink(tmp.c_str());
    return;
  }
  stats().stored.fetch_add(1);
  static std::atomic<uint64_t> writes{0};
  if (writes.fetch_add(1) % 32 == 0) trim();   // (the first write of a process, and every 32nd after)
}

// ---- programs

struct Live {
  std::mutex mu;
  std::set<Prog*> programs;
};
Live& live_set() {
  static Live* l = new Live;   // (never destroyed)
  return *l;
}
Prog* live(void* handle) {
  Live& l = live_set();
  std::lock_guard<std::mutex> lock(l.mu);
  Prog* p = static_cast<Prog*>(handle);
  return l.programs.count(p) ? p : nullptr;
}

std::vector<const char*> pointers(const std::vector<std::string>& v) {
  std::vector<const char*> out;
  for (const std::string& s : v) out.push_back(s.c_str());
  return out;
}

// Makes the real program behind `p` (and, where the cache answered its compile, compiles it): from here on every call is the real
// library's. Returns the real library's code for whatever failed.
int make_real(Prog* p) {
  const Real& r = real();
  if (!r.ok()) return kInternalError;
  Inside in;
  if (!p->real) {
    std::vector<const char*> headers = pointers(p->headers), names = pointers(p->include_names);
    const int c = r.create(&p->real, p->src.c_str(), p->name.c_str(), static_cast<int>(headers.size()), headers.empty() ? nullptr : headers.data(),
                           names.empty() ? nullptr : names.data());
    if (c != kSuccess) return p->real = nullptr, c;
    for (const std::string& e : p->name_exprs)
      if (r.add_name) r.add_name(p->real, e.c_str());
  }
  if (p->state == Prog::State::Cached) {
    std::vector<const char*> options = pointers(p->options);
    const auto t0 = std::chrono::steady_clock::now();
    const int c = r.compile(p->real, static_cast<int>(options.size()), options.empty() ? nullptr : options.data());
    stats().real_compiles.fetch_add(1);
    stats().real_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
    p->state = Prog::State::Real;
    p->entry.reset();
    return c;
  }
  p->state = Prog::State::Real;
  return kSuccess;
}

// What the real program holds after a compile that succeeded, read through the real library's own calls. The buffers are
// filled twice over different backgrounds, so that what the library wrote is told from what was there (a log's size does not
// say whether its end is counted).
bool capture(Prog* p, Entry* e) {
  const Real& r = real();
  size_t code_size = 0;
  if (r.code_size(p->real, &code_size) != kSuccess) return false;
  const auto filled = [&](ProgBufFn fn, size_t size, std::string* out) {
    std::string a(size + 64, '\xA5'), b(size + 64, '\x5A');
    if (fn(p->real, a.data()) != kSuccess || fn(p->real, b.data()) != kSuccess) return false;
    size_t n = a.size();
    while (n > 0 && a[n - 1] == '\xA5' && b[n - 1] == '\x5A') --n;
    if (n > size + 1) return false;   // (it wrote further than its size and one more: not a buffer this can copy)
    for (size_t i = 0; i < n; ++i)
      if (a[i] != b[i]) return false;   // (a byte it did not write, inside what it wrote)
    out->assign(a, 0, n);   // (exactly the bytes it wrote: a hit writes those and no others)
    return true;
  };
  if (!filled(r.get_code, code_size, &e->code) || e->code.size() != code_size) return false;
  size_t log_size = 0;
  if (!r.log_size || !r.get_log || r.log_size(p->real, &log_size) != kSuccess) return false;
  e->log_size = log_size;
  if (!filled(r.get_log, log_size, &e->log)) return false;
  for (const std::string& n : p->name_exprs) {
    const char* lowered = nullptr;
    if (!r.lowered || r.lowered(p->real, n.c_str(), &lowered) != kSuccess || !lowered) return false;
    e->lowered.push_back(lowered);
  }
  return true;
}

int compile(Prog* p, int num_options, const char** options) {
  const Real& r = real();
  stats_at_exit();
  stats().compiles.fetch_add(1);
  const auto forward = [&] {
    const int made = make_real(p);
    if (made != kSuccess) return made;
    Inside in;
    const auto t0 = std::chrono::steady_clock::now();
    const int c = r.compile(p->real, num_options, options);
    stats().real_compiles.fetch_add(1);
    stats().real_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()));
    return c;
  };
  if (p->state != Prog::State::Fresh) return forward();   // (the cache off: the program was made real when it was created)
  if (num_options < 0 || (num_options > 0 && !options)) return forward();
  std::vector<std::string> opts;
  for (int i = 0; i < num_options; ++i) {
    if (!options[i]) return forward();
    opts.push_back(options[i]);
  }
  if (!r.ok()) return kInternalError;
  const std::shared_ptr<const Toolchain> t = toolchain();
  const std::string arch = device_arch();
  std::string extra;
  std::string why = arch.empty() ? std::string("no HIP runtime to ask for the device's architecture") : not_cacheable(*p, opts, *t, &extra);
  if (!why.empty()) {
    note_reason(why);
    return forward();
  }
  const bool files = mode() == Mode::Disk && t->known && !cache_dir().empty();
  const std::string key = make_key(*p, opts, t->known ? t->text : std::string("(this process)"), arch, extra);
  std::shared_ptr<const Entry> hit;
  {
    Memory& m = memory();
    std::lock_guard<std::mutex> lock(m.mu);
    if (const auto it = m.by_key.find(key); it != m.by_key.end()) hit = it->second;
  }
  bool from_file = false;
  if (!hit && files) {
    hit = read_entry(key);
    from_file = hit != nullptr;
  }
  if (hit && hit->lowered.size() == p->name_exprs.size()) {
    if (from_file) {
      Memory& m = memory();
      std::lock_guard<std::mutex> lock(m.mu);
      if (m.bytes + key.size() + hit->code.size() <= kMemoryCap && m.by_key.emplace(key, hit).second) m.bytes += key.size() + hit->code.size();
    }
    (from_file ? stats().disk_hits : stats().memory_hits).fetch_add(1);
    // (what one compile costs here, for the line of statistics: the mean of the real ones so far, else 150 ms)
    const uint64_t n = stats().real_compiles.load();
    stats().saved_ns.fetch_add(n ? stats().real_ns.load() / n : 150000000ull);
    p->options = opts;
    p->entry = hit;
    p->state = Prog::State::Cached;
    return kSuccess;
  }
  const int c = forward();
  if (c != kSuccess || p->state != Prog::State::Real) return c;
  auto e = std::make_shared<Entry>();
  e->key = key;
  bool captured;
  {
    Inside in;
    captured = capture(p, e.get());
  }
  if (!captured) return c;
  // (the libraries the compile loaded are the ones the key named, or the result stays out of the files)
  const bool held = toolchain_held(t);
  {
    Memory& m = memory();
    std::lock_guard<std::mutex> lock(m.mu);
    if (m.bytes + key.size() + e->code.size() <= kMemoryCap && m.by_key.emplace(key, e).second) m.bytes += key.size() + e->code.size();
  }
  if (files && held) write_entry(*e);
  return c;
}

}  // namespace

// ---- hiprtc's entry points

VGPU_API const char* hiprtcGetErrorString(int result) {
  const Real& r = real();
  if (r.ok() && r.error_string) {
    Inside in;
    return r.error_string(result);
  }
  switch (result) {
    case 0: return "HIPRTC_SUCCESS";
    case 1: return "HIPRTC_ERROR_OUT_OF_MEMORY";
    case 2: return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case 3: return "HIPRTC_ERROR_INVALID_INPUT";
    case 4: return "HIPRTC_ERROR_INVALID_PROGRAM";
    case 5: return "HIPRTC_ERROR_INVALID_OPTION";
    case 6: return "HIPRTC_ERROR_COMPILATION";
    case 7: return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case 8: return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case 9: return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case 10: return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case 11: return "HIPRTC_ERROR_INTERNAL_ERROR";
    case 100: return "HIPRTC_ERROR_LINKING";
    default: return "Invalid HIPRTC error code";
  }
}

VGPU_API int hiprtcVersion(int* major, int* minor) {
  const Real& r = real();
  if (!r.ok() || !r.version) return kInternalError;
  Inside in;
  return r.version(major, minor);
}

VGPU_API int hiprtcCreateProgram(void** prog, const char* src, const char* name, int num_headers, const char** headers, const char** include_names) {
  const Real& r = real();
  if (t_inside) return r.create(prog, src, name, num_headers, headers, include_names);
  bool plain = mode() != Mode::Off && prog && src && name && num_headers >= 0 && (num_headers == 0 || (headers && include_names));
  for (int i = 0; plain && i < num_headers; ++i) plain = headers[i] && include_names[i];
  auto p = std::make_unique<Prog>();
  if (plain) {
    p->src = src, p->name = name;
    for (int i = 0; i < num_headers; ++i) p->headers.push_back(headers[i]), p->include_names.push_back(include_names[i]);
  } else {
    // Arguments the real library has its own answer for, or the cache off: the real program, at once.
    if (!r.ok()) return kInternalError;
    Inside in;
    void* made = nullptr;
    const int c = r.create(prog ? &made : nullptr, src, name, num_headers, headers, include_names);
    if (c != kSuccess || !prog) return c;
    p->real = made;
    p->state = Prog::State::Real;
  }
  Prog* raw = p.release();
  {
    Live& l = live_set();
    std::lock_guard<std::mutex> lock(l.mu);
    l.programs.insert(raw);
  }
  *prog = raw;
  return kSuccess;
}

VGPU_API int hiprtcDestroyProgram(void** prog) {
  const Real& r = real();
  if (t_inside) return r.destroy(prog);
  Prog* p = prog ? live(*prog) : nullptr;
  if (!p) {   // (nothing of this library's: the real one says what that is)
    if (!r.ok()) return prog && *prog ? kInvalidProgram : kInternalError;
    Inside in;
    void* none = nullptr;
    return r.destroy(prog && *prog ? &none : prog);
  }
  int c = kSuccess;
  if (p->real) {
    Inside in;
    c = r.destroy(&p->real);
  }
  {
    Live& l = live_set();
    std::lock_guard<std::mutex> lock(l.mu);
    l.programs.erase(p);
  }
  delete p;   // (the caller's handle is left as it was: ROCm's library does not clear it)
  return c;
}

VGPU_API int hiprtcAddNameExpression(void* prog, const char* name_expression) {
  const Real& r = real();
  if (t_inside) return r.add_name(prog, name_expression);
  Prog* p = live(prog);
  if (p && p->state == Prog::State::Fresh && name_expression) return p->name_exprs.push_back(name_expression), kSuccess;
  if (!r.ok() || !r.add_name) return kInternalError;
  if (p)
    if (const int made = make_real(p); made != kSuccess) return made;
  Inside in;
  return r.add_name(p ? p->real : nullptr, name_expression);
}

VGPU_API int hiprtcCompileProgram(void* prog, int num_options, const char** options) {
  const Real& r = real();
  if (t_inside) return r.compile(prog, num_options, options);
  Prog* p = live(prog);
  if (!p) {
    if (!r.ok()) return kInternalError;
    Inside in;
    return r.compile(nullptr, num_options, options);
  }
  return compile(p, num_options, options);
}

namespace {
// A call the cache answers for a program in State::Cached with `answer`, and the real library otherwise.
template <typename Answer, typename Forward>
int answer_or_forward(void* prog, bool cacheable_args, Answer&& answer, Forward&& fwd) {
  const Real& r = real();
  Prog* p = live(prog);
  if (p && p->state == Prog::State::Cached && cacheable_args)
    if (answer(*p)) return kSuccess;
  if (!r.ok()) return kInternalError;
  if (p)
    if (const int made = make_real(p); made != kSuccess) return made;
  Inside in;
  return fwd(p ? p->real : nullptr);
}
}  // namespace

VGPU_API int hiprtcGetCodeSize(void* prog, size_t* size) {
  if (t_inside) return real().code_size(prog, size);
  return answer_or_forward(
      prog, size != nullptr, [&](const Prog& p) { return *size = p.entry->code.size(), true; }, [&](void* rp) { return real().code_size(rp, size); });
}
VGPU_API int hiprtcGetCode(void* prog, char* code) {
  if (t_inside) return real().get_code(prog, code);
  return answer_or_forward(
      prog, code != nullptr, [&](const Prog& p) { return std::memcpy(code, p.entry->code.data(), p.entry->code.size()), true; },
      [&](void* rp) { return real().get_code(rp, code); });
}
VGPU_API int hiprtcGetProgramLogSize(void* prog, size_t* size) {
  const Real& r = real();
  if (t_inside) return r.log_size(prog, size);
  return answer_or_forward(
      prog, size != nullptr, [&](const Prog& p) { return *size = static_cast<size_t>(p.entry->log_size), true; },
      [&](void* rp) { return r.log_size ? r.log_size(rp, size) : kInternalError; });
}
VGPU_API int hiprtcGetProgramLog(void* prog, char* log) {
  const Real& r = real();
  if (t_inside) return r.get_log(prog, log);
  return answer_or_forward(
      prog, log != nullptr, [&](const Prog& p) { return std::memcpy(log, p.entry->log.data(), p.entry->log.size()), true; },
      [&](void* rp) { return r.get_log ? r.get_log(rp, log) : kInternalError; });
}
VGPU_API int hiprtcGetLoweredName(void* prog, const char* name_expression, const char** lowered_name) {
  const Real& r = real();
  if (t_inside) return r.lowered(prog, name_expression, lowered_name);
  return answer_or_forward(
      prog, name_expression && lowered_name,
      [&](const Prog& p) {
        for (size_t i = 0; i < p.name_exprs.size() && i < p.entry->lowered.size(); ++i)
          if (p.name_exprs[i] == name_expression) return *lowered_name = p.entry->lowered[i].c_str(), true;
        return false;   // (a name that was not added: the real library's answer)
      },
      [&](void* rp) { return r.lowered ? r.lowered(rp, name_expression, lowered_name) : kInternalError; });
}
// The bitcode is not kept: asking for it makes the real program (and compiles it, where the cache had answered).
VGPU_API int hiprtcGetBitcodeSize(void* prog, size_t* size) {
  const Real& r = real();
  if (t_inside) return r.bitcode_size(prog, size);
  return answer_or_forward(
      prog, false, [](const Prog&) { return false; }, [&](void* rp) { return r.bitcode_size ? r.bitcode_size(rp, size) : kInternalError; });
}
VGPU_API int hiprtcGetBitcode(void* prog, char* bitcode) {
  const Real& r = real();
  if (t_inside) return r.get_bitcode(prog, bitcode);
  return answer_or_forward(
      prog, false, [](const Prog&) { return false; }, [&](void* rp) { return r.get_bitcode ? r.get_bitcode(rp, bitcode) : kInternalError; });
}

// The link API works on bitcode the caller holds, not on programs: forwarded as it is, never cached.
VGPU_API int hiprtcLinkCreate(unsigned num_options, void* option_ptr, void** option_vals, void** link_state) {
  const Real& r = real();
  if (!r.ok() || !r.link_create) return kInternalError;
  Inside in;
  return r.link_create(num_options, option_ptr, option_vals, link_state);
}
VGPU_API int hiprtcLinkAddFile(void* link_state, int input_type, const char* file_path, unsigned num_options, void* options, void** option_values) {
  const Real& r = real();
  if (!r.ok() || !r.link_add_file) return kInternalError;
  Inside in;
  return r.link_add_file(link_state, input_type, file_path, num_options, options, option_values);
}
VGPU_API int hiprtcLinkAddData(void* link_state, int input_type, void* image, size_t image_size, const char* name, unsigned num_options, void* options,
                               void** option_values) {
  const Real& r = real();
  if (!r.ok() || !r.link_add_data) return kInternalError;
  Inside in;
  return r.link_add_data(link_state, input_type, image, image_size, name, num_options, options, option_values);
}
VGPU_API int hiprtcLinkComplete(void* link_state, void** bin_out, size_t* size_out) {
  const Real& r = real();
  if (!r.ok() || !r.link_complete) return kInternalError;
  Inside in;
  return r.link_complete(link_state, bin_out, size_out);
}
VGPU_API int hiprtcLinkDestroy(void* link_state) {
  const Real& r = real();
  if (!r.ok() || !r.link_destroy) return kInternalError;
  Inside in;
  return r.link_destroy(link_state);
}
