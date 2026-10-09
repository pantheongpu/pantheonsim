// libvgpunvrtc -- VirtualGPU's NVRTC, presented as libnvrtc.so.13.
//
// NVRTC's job is to turn a string of CUDA C++ into PTX, a cubin, or LTO-IR at
// run time. That is a C++ compiler with a PTX assembler inside, and there is no
// honest way to fake either -- so this shim does not try. It is a host-side
// library that needs no GPU, and the toolkit's own is the compiler:
//
// - Where the toolkit's libnvrtc is installed, every call goes to it, and what
//   it answers is what comes back: PTX, the cubin ptxas makes for an sm_
//   target, LTO-IR (-dlto), OptiX-IR (--optix-ir), the program log, lowered
//   names, precompiled headers (--pch, --create-pch, --use-pch, the heap-size
//   calls and the creation status), time traces (--time, --fdevice-time-trace),
//   the flow callback, the supported architectures. Its builtin headers are
//   the ones in use, and nothing of the host's. Nothing is emulated, so the
//   results are the ones an RTX 3060's NVRTC gives (nvidia/tests/e2e/
//   nvrtc_paths.cpp passes unchanged against it and against this).
// - Otherwise the program is written out and compiled by `nvcc --ptx` (and
//   `nvcc --cubin` for the cubin of an sm_ target, when asked for). PTX, the
//   cubin and the compiler's diagnostics come back; LTO-IR, OptiX-IR, Tile IR
//   and the precompiled-header and time-trace options need NVRTC itself and are
//   refused by name. If neither is found the call fails with a log saying so,
//   rather than a mystery NVRTC_ERROR_COMPILATION.
//
// The cubin of an sm_ target is the real one -- an ELF image of SASS, which
// VirtualGPU's driver runs on its SASS engine (nvidia/docs/sass.md) -- unless
// the PTX engine is in use (VGPU_SASS=0, or VGPU_NVRTC_CUBIN=ptx): then the
// program is compiled for the matching compute_ architecture and the "cubin" is
// its PTX, which that engine's driver loads from wherever a cubin goes (the
// choice PyTorch's jiterator ran on before the SASS engine existed). A real
// cubin comes with its PTX noted aside (vgpu_nvrtc_ptx_for_cubin), which the
// driver shim runs instead when the SASS engine cannot yet run some
// instruction in it -- the fallback a fatbin's PTX gives.
//
// The two compilers differ where it matters: nvcc pre-includes cuda_runtime.h,
// and with it the host's <stdint.h> and <math.h>, so a program that defines
// int64_t or INFINITY itself -- PyTorch's jiterator does both -- compiles under
// NVRTC and fails under nvcc. VGPU_NVRTC=nvcc forces the nvcc path;
// VGPU_NVRTC_LIB names the library to use.
//
// This is what makes the JIT frameworks reachable: CuPy, Numba, Triton and
// PyTorch's inductor all compile kernels through NVRTC and then load the result
// through the driver API -- which VirtualGPU already interprets.
#include <nvrtc.h>

#include <cuda.h>  // CUDA_VERSION: the toolkit this shim was built against

#include <dirent.h>
#include <dlfcn.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

bool trace() { const char* t = std::getenv("VGPU_TRACE"); return t && t[0] == '1'; }

// The PTX engine (VGPU_SASS=0) has no use for SASS, so a cubin is its PTX.
bool ptx_cubins() {
  const char* s = std::getenv("VGPU_SASS");
  if (s && s[0] == '0') return true;
  const char* c = std::getenv("VGPU_NVRTC_CUBIN");
  return c && std::strcmp(c, "ptx") == 0;
}

// NVRTC's result codes past CUDA 12.0's header, by number (13.0's nvrtc.h).
constexpr int kTimeFileWriteFailed = 12, kNoPchCreateAttempted = 13, kPchCreateHeapExhausted = 14,
              kPchCreate = 15, kCancelled = 16, kTimeTraceFileWriteFailed = 17;
nvrtcResult code(int n) { return static_cast<nvrtcResult>(n); }

struct Program {
  // The toolkit's program, when its NVRTC is the compiler: every query goes to it.
  nvrtcProgram real = nullptr;
  bool ptx_stand_in = false;   // compiled for compute_, so the "cubin" is the PTX
  // The nvcc path.
  std::string source;
  std::string name = "default_program";
  std::vector<std::pair<std::string, std::string>> headers;  // include name -> contents
  std::vector<std::string> name_expressions;
  std::vector<std::string> lowered;   // parallel to name_expressions, after compile
  std::string ptx;
  std::string log;
  std::string cubin;
  bool cubin_made = false;
  std::string scratch;      // the nvcc path's directory, kept while the cubin can still be asked for
  std::string cubin_cmd;    // the nvcc command that makes it
  bool compiled = false;
  bool real_arch = false;   // compiled for an sm_ target, so the caller will ask for CUBIN
  int (*flow)(void*, void*) = nullptr;
  void* flow_payload = nullptr;
  ~Program();
};

std::mutex g_mu;
std::set<const void*> g_live;

Program* get(nvrtcProgram p) {
  std::lock_guard<std::mutex> l(g_mu);
  return g_live.count(p) ? reinterpret_cast<Program*>(p) : nullptr;
}

// The program name and the include names come from the caller and become
// filenames, so keep them to something that cannot escape the scratch
// directory.
std::string safe_name(const std::string& in, const char* fallback) {
  std::string out;
  for (char c : in)
    if (isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-') out += c;
  if (out.empty() || out == "." || out == "..") return fallback;
  return out;
}

bool safe_include(const std::string& in) {
  if (in.empty() || in[0] == '/') return false;
  return in.find("..") == std::string::npos;
}

std::string tempdir() {
  const char* t = std::getenv("TMPDIR");
  std::string base = std::string(t && *t ? t : "/tmp") + "/vgpu-nvrtc-" + std::to_string(getpid());
  ::mkdir(base.c_str(), 0700);
  static std::atomic_int seq{0};
  const std::string d = base + "/" + std::to_string(seq.fetch_add(1));
  ::mkdir(d.c_str(), 0700);
  return d;
}

bool write_file(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write(text.data(), (std::streamsize)text.size());
  return f.good();
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool have_nvcc() { return std::system("command -v nvcc >/dev/null 2>&1") == 0; }

// Quote for /bin/sh. Paths here are ours, but option strings come from the
// caller and can contain anything.
std::string shq(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  return out + "'";
}

// Only ever called on a directory this file created under TMPDIR.
void remove_tree(const std::string& dir) {
  const int rc = std::system(("rm -rf " + shq(dir)).c_str());
  (void)rc;
}

Program::~Program() {
  if (!scratch.empty() && !trace()) remove_tree(scratch);
}

// ---- the toolkit's NVRTC ----

struct RealNvrtc {
  void* lib = nullptr;
  nvrtcResult (*create)(nvrtcProgram*, const char*, const char*, int, const char* const*,
                        const char* const*) = nullptr;
  nvrtcResult (*destroy)(nvrtcProgram*) = nullptr;
  nvrtcResult (*add_name)(nvrtcProgram, const char*) = nullptr;
  nvrtcResult (*compile)(nvrtcProgram, int, const char* const*) = nullptr;
  nvrtcResult (*ptx_size)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*ptx)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*log_size)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*log)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*lowered)(nvrtcProgram, const char*, const char**) = nullptr;
  // Optional: older toolkits lack some.
  nvrtcResult (*cubin_size)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*cubin)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*ltoir_size)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*ltoir)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*optix_size)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*optix)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*tile_size)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*tile)(nvrtcProgram, char*) = nullptr;
  nvrtcResult (*pch_heap_get)(size_t*) = nullptr;
  nvrtcResult (*pch_heap_set)(size_t) = nullptr;
  nvrtcResult (*pch_status)(nvrtcProgram) = nullptr;
  nvrtcResult (*pch_required)(nvrtcProgram, size_t*) = nullptr;
  nvrtcResult (*flow)(nvrtcProgram, int (*)(void*, void*), void*) = nullptr;
  nvrtcResult (*version)(int*, int*) = nullptr;
  nvrtcResult (*num_archs)(int*) = nullptr;
  nvrtcResult (*archs)(int*) = nullptr;
  std::string path;
};

std::string real_path(const std::string& p) {
  char buf[PATH_MAX];
  return ::realpath(p.c_str(), buf) ? std::string(buf) : std::string();
}

// The library this shim is, so the search below never finds itself.
std::string own_path() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&own_path), &info) || !info.dli_fname) return {};
  return real_path(info.dli_fname);
}

// The candidates, most specific first: an explicit path, then the toolkit
// that owns the nvcc on PATH, then the usual install locations. The major is
// the one this shim was built for, since that is the ABI it presents.
std::vector<std::string> real_nvrtc_candidates() {
  const std::string so = "libnvrtc.so." + std::to_string(CUDA_VERSION / 1000);
  std::vector<std::string> out;
  if (const char* e = std::getenv("VGPU_NVRTC_LIB"); e && *e) out.push_back(e);
  if (FILE* f = popen("command -v nvcc 2>/dev/null", "r")) {
    char buf[PATH_MAX] = {0};
    if (fgets(buf, sizeof buf, f)) {
      std::string nvcc = real_path(std::string(buf).substr(0, std::strcspn(buf, "\n")));
      const size_t slash = nvcc.find_last_of('/');
      if (slash != std::string::npos) {
        const std::string root = nvcc.substr(0, slash) + "/..";
        out.push_back(root + "/lib64/" + so);
        out.push_back(root + "/targets/x86_64-linux/lib/" + so);
        out.push_back(root + "/targets/sbsa-linux/lib/" + so);
      }
    }
    pclose(f);
  }
  for (const char* dir : {"/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"})
    out.push_back(std::string(dir) + "/" + so);
  return out;
}

// libnvrtc.so finds libnvrtc-builtins.so.<major>.<minor> with a plain dlopen by soname when the first
// compile runs. NVIDIA's pip wheels (nvidia/cu13/lib) keep both in one directory that is on no search path,
// and the real library is opened RTLD_LOCAL here, so that lookup fails ("failed to open
// libnvrtc-builtins.so.13.0") and every run-time compile with it fails too. Opening the sibling by full path
// first makes the later by-name lookup resolve to the already loaded object (same soname).
void preload_builtins(const std::string& nvrtc_path) {
  const size_t slash = nvrtc_path.find_last_of('/');
  if (slash == std::string::npos) return;
  const std::string dir = nvrtc_path.substr(0, slash);
  DIR* d = ::opendir(dir.c_str());
  if (!d) return;
  std::vector<std::string> names;
  while (const dirent* e = ::readdir(d)) {
    const std::string n = e->d_name;
    // the plain builtins, not the ".alt" variant that goes with libnvrtc.alt.so
    if (n.rfind("libnvrtc-builtins.so.", 0) == 0) names.push_back(n);
  }
  ::closedir(d);
  for (const std::string& n : names) {
    void* h = ::dlopen((dir + "/" + n).c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (trace()) std::fprintf(stderr, "[vgpu] nvrtc: preload %s/%s -> %s\n", dir.c_str(), n.c_str(), h ? "ok" : "failed");
  }
}

const RealNvrtc* real_nvrtc() {
  static const RealNvrtc* found = [] () -> const RealNvrtc* {
    if (const char* m = std::getenv("VGPU_NVRTC"); m && std::strcmp(m, "nvcc") == 0) return nullptr;
    const std::string self = own_path();
    // RTLD_DEEPBIND keeps the real library's calls to its own nvrtc* entry
    // points inside it, rather than resolving them to this shim's. A sanitizer
    // runtime refuses the flag; there, the library's own internal binding has
    // to do.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    const int flags = RTLD_NOW | RTLD_LOCAL;
#else
    const int flags = RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND;
#endif
    for (const std::string& c : real_nvrtc_candidates()) {
      const std::string rp = real_path(c);
      if (rp.empty() || rp == self) continue;
      void* lib = dlopen(rp.c_str(), flags);
      if (!lib) continue;
      auto* r = new RealNvrtc();
      r->lib = lib;
      r->path = rp;
      bool ok = true;
      auto sym = [&](auto& fn, const char* name, bool required = true) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
        ok = ok && (fn || !required);
      };
      sym(r->create, "nvrtcCreateProgram");
      sym(r->destroy, "nvrtcDestroyProgram");
      sym(r->add_name, "nvrtcAddNameExpression");
      sym(r->compile, "nvrtcCompileProgram");
      sym(r->ptx_size, "nvrtcGetPTXSize");
      sym(r->ptx, "nvrtcGetPTX");
      sym(r->log_size, "nvrtcGetProgramLogSize");
      sym(r->log, "nvrtcGetProgramLog");
      sym(r->lowered, "nvrtcGetLoweredName");
      sym(r->cubin_size, "nvrtcGetCUBINSize", false);
      sym(r->cubin, "nvrtcGetCUBIN", false);
      sym(r->ltoir_size, "nvrtcGetLTOIRSize", false);
      sym(r->ltoir, "nvrtcGetLTOIR", false);
      sym(r->optix_size, "nvrtcGetOptiXIRSize", false);
      sym(r->optix, "nvrtcGetOptiXIR", false);
      sym(r->tile_size, "nvrtcGetTileIRSize", false);
      sym(r->tile, "nvrtcGetTileIR", false);
      sym(r->pch_heap_get, "nvrtcGetPCHHeapSize", false);
      sym(r->pch_heap_set, "nvrtcSetPCHHeapSize", false);
      sym(r->pch_status, "nvrtcGetPCHCreateStatus", false);
      sym(r->pch_required, "nvrtcGetPCHHeapSizeRequired", false);
      sym(r->flow, "nvrtcSetFlowCallback", false);
      sym(r->version, "nvrtcVersion", false);
      sym(r->num_archs, "nvrtcGetNumSupportedArchs", false);
      sym(r->archs, "nvrtcGetSupportedArchs", false);
      if (ok) {
        preload_builtins(rp);
        if (trace()) std::fprintf(stderr, "[vgpu] nvrtc: compiling with %s\n", rp.c_str());
        return r;
      }
      delete r;
      dlclose(lib);
    }
    return nullptr;
  }();
  return found;
}

// For the PTX engine: a real architecture becomes its virtual one, since
// NVRTC emits PTX alone for a compute_ target and a cubin is of no use to that
// engine. The link-time-optimisation flags would ask for LTO-IR in its place,
// so they go.
void ptx_engine_option(const std::string& opt, std::vector<std::string>* out) {
  auto starts = [&](const char* p) { return opt.rfind(p, 0) == 0; };
  if (starts("--gpu-architecture=") || starts("-arch=")) {
    std::string arch = opt.substr(opt.find('=') + 1);
    if (arch.rfind("sm_", 0) == 0) arch = "compute_" + arch.substr(3);
    out->push_back("--gpu-architecture=" + arch);
    return;
  }
  if (starts("-dlto") || starts("--dlink-time-opt")) return;
  out->push_back(opt);
}

// Whether an option asks for a real (sm_) architecture.
bool names_real_arch(const std::string& o) {
  const size_t eq = o.find('=');
  return (o.rfind("--gpu-architecture=", 0) == 0 || o.rfind("-arch=", 0) == 0) && eq != std::string::npos &&
         o.compare(eq + 1, 3, "sm_") == 0;
}

// ---- a cubin and the PTX it came from ----
//
// A cubin NVRTC makes for an sm_ target is handed out with the PTX of the same
// program noted aside, keyed by the cubin's content. VirtualGPU's driver shim
// finds this through vgpu_nvrtc_ptx_for_cubin (a dlsym away: it need not link
// this library) and runs the PTX when the SASS engine cannot yet run some
// instruction of the cubin, as it does for a fatbin that carries both.
struct CubinKey {
  uint64_t hash;
  size_t size;
  bool operator<(const CubinKey& o) const { return hash != o.hash ? hash < o.hash : size < o.size; }
};
CubinKey key_of(const void* data, size_t size) {
  uint64_t h = 1469598103934665603ull;   // FNV-1a
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return {h, size};
}
// How much of a cubin the driver shim reads when it is handed the bare pointer:
// up to the end of the section header table (the program headers NVRTC's cubins
// carry after it are not part of that). Both sides key on this prefix.
size_t elf_extent(const void* data, size_t size) {
  if (size < 0x40) return size;
  const auto* p = static_cast<const unsigned char*>(data);
  uint64_t shoff;
  uint16_t shentsize, shnum;
  std::memcpy(&shoff, p + 0x28, 8);
  std::memcpy(&shentsize, p + 0x3a, 2);
  std::memcpy(&shnum, p + 0x3c, 2);
  const uint64_t end = shoff + static_cast<uint64_t>(shentsize) * shnum;
  return end <= size ? static_cast<size_t>(end) : size;
}
std::mutex g_aside_mu;
std::map<CubinKey, std::string> g_aside;   // bounded: the oldest key goes first
std::vector<CubinKey> g_aside_order;
void note_aside(const void* cubin, size_t size, std::string ptx) {
  if (ptx.empty()) return;
  std::lock_guard<std::mutex> l(g_aside_mu);
  const CubinKey k = key_of(cubin, elf_extent(cubin, size));
  if (g_aside.emplace(k, std::move(ptx)).second) g_aside_order.push_back(k);
  while (g_aside_order.size() > 512) {
    g_aside.erase(g_aside_order.front());
    g_aside_order.erase(g_aside_order.begin());
  }
}

// The toolkit's library lacks this entry (an older NVRTC than the call needs).
nvrtcResult missing(const char* what) {
  static std::atomic_bool said{false};
  if (!said.exchange(true))
    std::fprintf(stderr, "[vgpu] nvrtc: the toolkit's libnvrtc has no %s (a newer toolkit's NVRTC is needed)\n", what);
  return NVRTC_ERROR_INTERNAL_ERROR;
}

// Translate the NVRTC option spelling to the nvcc one. Most options are the
// same; the ones that differ are the architecture and a handful of NVRTC-only
// flags that have no nvcc equivalent and are dropped.
bool translate_option(const std::string& opt, std::vector<std::string>* out, std::string* log) {
  auto starts = [&](const char* p) { return opt.rfind(p, 0) == 0; };
  if (starts("--gpu-architecture=") || starts("-arch=")) {
    const std::string arch = opt.substr(opt.find('=') + 1);
    out->push_back("-arch=" + arch);
    return true;
  }
  if (starts("-D") || starts("-U") || starts("-I") || starts("--include-path=") ||
      starts("--define-macro=") || starts("--undefine-macro=") || starts("-std=") ||
      starts("--std=") || starts("--use_fast_math") || starts("--fmad=") ||
      starts("--extra-device-vectorization") || starts("--restrict") ||
      starts("--device-debug") || starts("--generate-line-info") || starts("-lineinfo") ||
      starts("-G") || starts("--relocatable-device-code=") || starts("-rdc=") ||
      starts("--device-int128") || starts("--optimization-info=") ||
      starts("--diag-suppress=") || starts("-w") || starts("--Werror") ||
      starts("--expt-relaxed-constexpr") || starts("--expt-extended-lambda") ||
      starts("--pre-include=")) {
    out->push_back(opt);
    return true;
  }
  if (starts("--device-as-default-execution-space") || starts("-default-device") ||
      starts("--dopt=") || starts("--split-compile=") || starts("--minimal") ||
      starts("--builtin-move-forward=") || starts("--builtin-initializer-list=")) {
    // Accepted by NVRTC, meaningless or unavailable here. Say so under trace
    // rather than failing the compile over a flag that changes nothing.
    if (trace()) std::fprintf(stderr, "[vgpu] nvrtc: ignoring option %s\n", opt.c_str());
    return true;
  }
  if (starts("--dlink-time-opt") || starts("-dlto") || starts("--optix-ir") || starts("-optix-ir") ||
      starts("--gen-opt-lto") || starts("-gen-opt-lto")) {
    *log += "vgpu nvrtc: option '" + opt +
            "' asks for LTO-IR or OptiX-IR, which only NVIDIA's NVRTC makes (it is NVVM bitcode); install the "
            "toolkit's libnvrtc, which this shim uses when it is found\n";
    return false;
  }
  if (starts("--pch") || starts("-pch") || starts("--create-pch") || starts("-create-pch") ||
      starts("--use-pch") || starts("-use-pch") || starts("--instantiate-templates-in-pch") ||
      starts("-instantiate-templates-in-pch")) {
    *log += "vgpu nvrtc: option '" + opt +
            "' asks for a precompiled header, which only NVIDIA's NVRTC makes; install the toolkit's "
            "libnvrtc, which this shim uses when it is found\n";
    return false;
  }
  if (starts("--time=") || starts("-time=") || starts("--fdevice-time-trace=") || starts("-fdevice-time-trace=")) {
    *log += "vgpu nvrtc: option '" + opt +
            "' asks for a time trace of NVRTC's own compiler, which nvcc does not write; install the "
            "toolkit's libnvrtc, which this shim uses when it is found\n";
    return false;
  }
  *log += "vgpu nvrtc: unrecognised option '" + opt + "'\n";
  return false;
}

// The PCH heap, where the toolkit's NVRTC is not the one answering: 13.0's
// default, and a size rounded up to a multiple of 4096 (12345 reads back as
// 16384, 1 as 4096, 0 as 0: measured).
std::atomic<size_t> g_pch_heap{268435456};

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

VGPU_EXPORT const char* nvrtcGetErrorString(nvrtcResult r) {
  switch (static_cast<int>(r)) {
    case 0: return "NVRTC_SUCCESS";
    case 1: return "NVRTC_ERROR_OUT_OF_MEMORY";
    case 2: return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case 3: return "NVRTC_ERROR_INVALID_INPUT";
    case 4: return "NVRTC_ERROR_INVALID_PROGRAM";
    case 5: return "NVRTC_ERROR_INVALID_OPTION";
    case 6: return "NVRTC_ERROR_COMPILATION";
    case 7: return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case 8: return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case 9: return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case 10: return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case 11: return "NVRTC_ERROR_INTERNAL_ERROR";
    case kTimeFileWriteFailed: return "NVRTC_ERROR_TIME_FILE_WRITE_FAILED";
    case kNoPchCreateAttempted: return "NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED";
    case kPchCreateHeapExhausted: return "NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED";
    case kPchCreate: return "NVRTC_ERROR_PCH_CREATE";
    case kCancelled: return "NVRTC_ERROR_CANCELLED";
    case kTimeTraceFileWriteFailed: return "NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED";
    default: return "NVRTC_ERROR unknown";   // as 13.0's answers for 18 and up, and for a negative value
  }
}

VGPU_EXPORT nvrtcResult nvrtcVersion(int* major, int* minor) {
  if (!major || !minor) return NVRTC_ERROR_INVALID_INPUT;
  // The compiler that will actually run is the toolkit's NVRTC, or else
  // whichever nvcc is on PATH, so report its version rather than the header's;
  // fall back to the build-time toolkit when neither can be asked.
  if (const RealNvrtc* r = real_nvrtc(); r && r->version) return r->version(major, minor);
  static int cached_major = 0, cached_minor = 0;
  static std::once_flag once;
  std::call_once(once, [] {
    cached_major = CUDA_VERSION / 1000;
    cached_minor = (CUDA_VERSION % 1000) / 10;
    if (FILE* f = popen("nvcc --version 2>/dev/null", "r")) {
      char line[512];
      while (std::fgets(line, sizeof(line), f)) {
        const char* r = std::strstr(line, "release ");
        int a = 0, b = 0;
        if (r && std::sscanf(r + 8, "%d.%d", &a, &b) == 2) { cached_major = a; cached_minor = b; }
      }
      pclose(f);
    }
  });
  *major = cached_major;
  *minor = cached_minor;
  return NVRTC_SUCCESS;
}

// Without the toolkit's NVRTC, the architectures VirtualGPU's profiles cover,
// ascending as the API requires. With it, the ones it compiles for.
static const int kArchs[] = {70, 75, 80, 86, 89, 90, 100, 120};

VGPU_EXPORT nvrtcResult nvrtcGetNumSupportedArchs(int* n) {
  if (const RealNvrtc* r = real_nvrtc(); r && r->num_archs && r->archs) return r->num_archs(n);
  if (!n) return NVRTC_ERROR_INVALID_INPUT;
  *n = (int)(sizeof(kArchs) / sizeof(kArchs[0]));
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetSupportedArchs(int* out) {
  if (const RealNvrtc* r = real_nvrtc(); r && r->num_archs && r->archs) return r->archs(out);
  if (!out) return NVRTC_ERROR_INVALID_INPUT;
  std::memcpy(out, kArchs, sizeof(kArchs));
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name,
                                           int num_headers, const char* const* headers,
                                           const char* const* include_names) {
  if (const RealNvrtc* r = real_nvrtc()) {
    nvrtcProgram made = nullptr;
    const nvrtcResult rc = r->create(prog ? &made : nullptr, src, name, num_headers, headers, include_names);
    if (rc != NVRTC_SUCCESS) return rc;
    auto* p = new Program();
    p->real = made;
    {
      std::lock_guard<std::mutex> l(g_mu);
      g_live.insert(p);
    }
    *prog = reinterpret_cast<nvrtcProgram>(p);
    return NVRTC_SUCCESS;
  }
  if (!prog) return NVRTC_ERROR_INVALID_PROGRAM;
  if (!src) return NVRTC_ERROR_INVALID_INPUT;
  if (num_headers < 0 || (num_headers > 0 && (!headers || !include_names)))
    return NVRTC_ERROR_INVALID_INPUT;
  auto* p = new Program();
  p->source = src;
  if (name && *name) p->name = name;
  for (int i = 0; i < num_headers; ++i) {
    if (!headers[i] || !include_names[i]) { delete p; return NVRTC_ERROR_INVALID_INPUT; }
    p->headers.emplace_back(include_names[i], headers[i]);
  }
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.insert(p);
  }
  *prog = reinterpret_cast<nvrtcProgram>(p);
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) {
  // A null handle is refused, as NVIDIA's refuses it (INVALID_PROGRAM, measured).
  if (!prog || !*prog) return NVRTC_ERROR_INVALID_PROGRAM;
  Program* p = get(*prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.erase(p);
  }
  if (p->real) real_nvrtc()->destroy(&p->real);
  delete p;
  *prog = nullptr;
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* expr) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) return real_nvrtc()->add_name(p->real, expr);
  if (!expr) return NVRTC_ERROR_INVALID_INPUT;
  if (p->compiled) return NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
  p->name_expressions.emplace_back(expr);
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int num_options,
                                            const char* const* options) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  p->compiled = true;   // "compilation was attempted", which is what the API means
  p->real_arch = false;
  p->ptx_stand_in = false;
  for (int i = 0; options && i < num_options; ++i)
    if (options[i] && names_real_arch(options[i])) p->real_arch = true;

  if (p->real) {
    const RealNvrtc& r = *real_nvrtc();
    if (!ptx_cubins() || !p->real_arch) return r.compile(p->real, num_options, options);
    // The PTX engine: compile for the virtual architecture, whose PTX is the
    // "cubin".
    std::vector<std::string> opts;
    for (int i = 0; i < num_options; ++i) {
      if (!options || !options[i]) return NVRTC_ERROR_INVALID_OPTION;
      ptx_engine_option(options[i], &opts);
    }
    std::vector<const char*> argv;
    for (const auto& o : opts) argv.push_back(o.c_str());
    p->ptx_stand_in = true;
    return r.compile(p->real, static_cast<int>(argv.size()), argv.data());
  }

  p->log.clear();
  p->ptx.clear();
  p->lowered.clear();
  p->cubin.clear();
  p->cubin_made = false;
  if (!have_nvcc()) {
    p->log =
        "vgpu nvrtc: nvcc was not found on PATH.\n"
        "VirtualGPU implements NVRTC with the CUDA toolkit's compiler, which runs on the "
        "host and needs no GPU. Install the toolkit, or put nvcc on PATH, to compile at run "
        "time.\n";
    return NVRTC_ERROR_COMPILATION;
  }

  std::vector<std::string> nvcc_opts;
  for (int i = 0; i < num_options; ++i) {
    if (!options || !options[i]) return NVRTC_ERROR_INVALID_OPTION;
    if (!translate_option(options[i], &nvcc_opts, &p->log)) return NVRTC_ERROR_INVALID_OPTION;
  }

  if (!p->scratch.empty()) remove_tree(p->scratch);
  const std::string dir = tempdir();
  p->scratch = dir;   // kept until the program goes: the cubin is made when asked for

  // Headers are supplied by string, so materialise them and put the directory
  // on the include path -- that is what #include in the source will look for.
  for (const auto& [inc, text] : p->headers) {
    if (!safe_include(inc)) {
      p->log += "vgpu nvrtc: refusing header name '" + inc + "': it escapes the include tree\n";
      return NVRTC_ERROR_INVALID_INPUT;
    }
    std::string path = dir + "/" + inc;
    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos && slash > dir.size()) {
      // Create any subdirectory the include name implies, one level at a time.
      for (size_t i = dir.size() + 1; i <= slash; ++i)
        if (path[i] == '/') { std::string sub = path.substr(0, i); ::mkdir(sub.c_str(), 0700); }
    }
    if (!write_file(path, text)) {
      p->log += "vgpu nvrtc: cannot write header " + inc + "\n";
      return NVRTC_ERROR_COMPILATION;
    }
  }

  std::string source = p->source;
  // Name expressions: NVRTC's contract is to report the mangled symbol for a
  // C++ expression such as "kernel<float>". Getting that right needs the
  // compiler's own mangling, so emit a device variable initialised with the
  // expression's address and read the symbol back out of the generated PTX.
  for (size_t i = 0; i < p->name_expressions.size(); ++i)
    source += "\n__device__ const void* vgpu_nvrtc_name_" + std::to_string(i) + " = (const void*)&(" +
              p->name_expressions[i] + ");\n";

  const std::string base = safe_name(p->name, "program.cu");
  const std::string cu = dir + "/" + base + (base.find(".cu") == std::string::npos ? ".cu" : "");
  const std::string ptx = dir + "/out.ptx";
  const std::string err = dir + "/stderr.txt";
  if (!write_file(cu, source)) {
    p->log += "vgpu nvrtc: cannot write program source\n";
    return NVRTC_ERROR_COMPILATION;
  }

  std::string common = " -Wno-deprecated-gpu-targets -I " + shq(dir);
  for (const auto& o : nvcc_opts) common += " " + shq(o);
  std::string cmd = "nvcc --ptx" + common + " -o " + shq(ptx) + " " + shq(cu) + " 2> " + shq(err);
  p->cubin_cmd = "nvcc --cubin" + common + " -o " + shq(dir + "/out.cubin") + " " + shq(cu) + " 2> " +
                 shq(dir + "/cubin-stderr.txt");
  if (trace()) std::fprintf(stderr, "[vgpu] nvrtc: %s\n", cmd.c_str());
  // The flow callback gets its two chances: before the compiler runs, and after.
  if (p->flow && p->flow(p->flow_payload, nullptr) == 1) return code(kCancelled);
  const int rc = std::system(cmd.c_str());
  if (p->flow && p->flow(p->flow_payload, nullptr) == 1) return code(kCancelled);

  p->log += read_file(err);
  if (rc != 0) return NVRTC_ERROR_COMPILATION;
  p->ptx = read_file(ptx);
  if (p->ptx.empty()) {
    p->log += "vgpu nvrtc: nvcc reported success but produced no PTX\n";
    return NVRTC_ERROR_COMPILATION;
  }

  // Pull each lowered name back out: the marker variable's initialiser names the
  // mangled symbol, e.g ".global .align 8 .u64 vgpu_nvrtc_name_0 = _Z6kernelIfEvPf;"
  for (size_t i = 0; i < p->name_expressions.size(); ++i) {
    const std::string marker = "vgpu_nvrtc_name_" + std::to_string(i);
    std::string lowered;
    size_t at = p->ptx.find(marker);
    while (at != std::string::npos) {
      const size_t eq = p->ptx.find('=', at);
      const size_t eol = p->ptx.find('\n', at);
      if (eq != std::string::npos && (eol == std::string::npos || eq < eol)) {
        size_t b = eq + 1;
        while (b < p->ptx.size() && (p->ptx[b] == ' ' || p->ptx[b] == '\t')) ++b;
        size_t e = b;
        while (e < p->ptx.size() && (isalnum((unsigned char)p->ptx[e]) || p->ptx[e] == '_' ||
                                     p->ptx[e] == '$'))
          ++e;
        if (e > b) { lowered = p->ptx.substr(b, e - b); break; }
      }
      at = p->ptx.find(marker, at + marker.size());
    }
    // A plain extern "C" kernel has no mangling to recover; its own name is the
    // lowered name.
    if (lowered.empty()) lowered = p->name_expressions[i];
    p->lowered.push_back(lowered);
  }
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char* expr,
                                            const char** lowered) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) return real_nvrtc()->lowered(p->real, expr, lowered);
  if (!expr || !lowered) return NVRTC_ERROR_INVALID_INPUT;
  if (!p->compiled) return NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;
  for (size_t i = 0; i < p->name_expressions.size(); ++i)
    if (p->name_expressions[i] == expr) {
      if (i >= p->lowered.size()) return NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
      *lowered = p->lowered[i].c_str();
      return NVRTC_SUCCESS;
    }
  return NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
}

VGPU_EXPORT nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* size) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) return real_nvrtc()->ptx_size(p->real, size);
  if (!size) return NVRTC_ERROR_INVALID_INPUT;
  *size = p->ptx.size() + 1;   // the API counts the terminating NUL
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* out) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) return real_nvrtc()->ptx(p->real, out);
  if (!out) return NVRTC_ERROR_INVALID_INPUT;
  std::memcpy(out, p->ptx.c_str(), p->ptx.size() + 1);
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* size) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) return real_nvrtc()->log_size(p->real, size);
  if (!size) return NVRTC_ERROR_INVALID_INPUT;
  *size = p->log.size() + 1;
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* out) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) return real_nvrtc()->log(p->real, out);
  if (!out) return NVRTC_ERROR_INVALID_INPUT;
  std::memcpy(out, p->log.c_str(), p->log.size() + 1);
  return NVRTC_SUCCESS;
}

/* ---- the other outputs ---- */

// The cubin of an sm_ target. With the toolkit's NVRTC it is the one NVRTC
// made (ptxas inside it), and a program compiled for a virtual architecture has
// none, size 0, as documented. On the nvcc path, `nvcc --cubin` makes it when
// first asked for. For the PTX engine it is the PTX, NUL included: a caller
// hands CUBIN to cuModuleLoadData, and that engine's driver loads PTX from
// there as it loads it from anywhere.
static bool make_cubin_nvcc(Program* p) {
  if (p->cubin_made) return !p->cubin.empty();
  p->cubin_made = true;
  if (p->cubin_cmd.empty()) return false;
  if (trace()) std::fprintf(stderr, "[vgpu] nvrtc: %s\n", p->cubin_cmd.c_str());
  const int rc = std::system(p->cubin_cmd.c_str());
  if (rc == 0) p->cubin = read_file(p->scratch + "/out.cubin");
  if (p->cubin.empty()) p->log += read_file(p->scratch + "/cubin-stderr.txt");
  return !p->cubin.empty();
}

VGPU_EXPORT nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* size) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real && !p->ptx_stand_in) {
    const RealNvrtc& r = *real_nvrtc();
    return r.cubin_size ? r.cubin_size(p->real, size) : missing("nvrtcGetCUBINSize");
  }
  if (!size) return NVRTC_ERROR_INVALID_INPUT;
  if (p->real) {   // the PTX engine's stand-in
    size_t n = 0;
    *size = real_nvrtc()->ptx_size(p->real, &n) == NVRTC_SUCCESS && n > 1 ? n : 0;
    return NVRTC_SUCCESS;
  }
  if (!p->real_arch || p->ptx.empty()) {
    *size = 0;
    return NVRTC_SUCCESS;
  }
  if (ptx_cubins()) {
    *size = p->ptx.size() + 1;
    return NVRTC_SUCCESS;
  }
  *size = make_cubin_nvcc(p) ? p->cubin.size() : 0;
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* out) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real && !p->ptx_stand_in) {
    const RealNvrtc& r = *real_nvrtc();
    if (!r.cubin) return missing("nvrtcGetCUBIN");
    const nvrtcResult rc = r.cubin(p->real, out);
    if (rc == NVRTC_SUCCESS && out) {
      // Note the PTX aside, for the driver shim's fallback.
      size_t n = 0, m = 0;
      if (r.cubin_size(p->real, &n) == NVRTC_SUCCESS && n > 0 && r.ptx_size(p->real, &m) == NVRTC_SUCCESS && m > 1) {
        std::string text(m, '\0');
        if (r.ptx(p->real, &text[0]) == NVRTC_SUCCESS) {
          text.resize(m - 1);
          note_aside(out, n, std::move(text));
        }
      }
    }
    return rc;
  }
  if (p->real) {   // the PTX engine's stand-in
    size_t n = 0;
    if (real_nvrtc()->ptx_size(p->real, &n) != NVRTC_SUCCESS || n <= 1) return NVRTC_ERROR_INVALID_PROGRAM;
    if (!out) return NVRTC_ERROR_INVALID_INPUT;
    return real_nvrtc()->ptx(p->real, out);
  }
  if (!out) return NVRTC_ERROR_INVALID_INPUT;
  if (!p->real_arch || p->ptx.empty()) return NVRTC_SUCCESS;   // no cubin: its size is 0, there is nothing to copy
  if (ptx_cubins()) {
    std::memcpy(out, p->ptx.c_str(), p->ptx.size() + 1);
    return NVRTC_SUCCESS;
  }
  if (!make_cubin_nvcc(p)) return NVRTC_ERROR_COMPILATION;
  std::memcpy(out, p->cubin.data(), p->cubin.size());
  note_aside(p->cubin.data(), p->cubin.size(), p->ptx);
  return NVRTC_SUCCESS;
}

// The nvcc path cannot make these (the options that ask for them are refused),
// so their size is 0 -- what NVIDIA's reports for a program not compiled to
// that output -- and copying them is a no-op.
#define VGPU_NVRTC_OUTPUT(Name, field, label)                                                         \
  VGPU_EXPORT nvrtcResult nvrtcGet##Name##Size(nvrtcProgram prog, size_t* size) {                     \
    Program* p = get(prog);                                                                           \
    if (!p) return NVRTC_ERROR_INVALID_PROGRAM;                                                       \
    if (p->real) {                                                                                    \
      const RealNvrtc& r = *real_nvrtc();                                                             \
      return r.field##_size ? r.field##_size(p->real, size) : missing("nvrtcGet" #Name "Size");       \
    }                                                                                                 \
    if (!size) return NVRTC_ERROR_INVALID_INPUT;                                                      \
    *size = 0;                                                                                        \
    return NVRTC_SUCCESS;                                                                             \
  }                                                                                                   \
  VGPU_EXPORT nvrtcResult nvrtcGet##Name(nvrtcProgram prog, char* out) {                              \
    Program* p = get(prog);                                                                           \
    if (!p) return NVRTC_ERROR_INVALID_PROGRAM;                                                       \
    if (p->real) {                                                                                    \
      const RealNvrtc& r = *real_nvrtc();                                                             \
      return r.field ? r.field(p->real, out) : missing("nvrtcGet" #Name);                             \
    }                                                                                                 \
    (void)label;                                                                                      \
    return out ? NVRTC_SUCCESS : NVRTC_ERROR_INVALID_INPUT;   /* never made: size 0, nothing to copy */ \
  }
VGPU_NVRTC_OUTPUT(LTOIR, ltoir, "LTO-IR")
VGPU_NVRTC_OUTPUT(OptiXIR, optix, "OptiX-IR")
VGPU_NVRTC_OUTPUT(TileIR, tile, "Tile IR")

/* ---- precompiled headers, the flow callback ---- */

VGPU_EXPORT nvrtcResult nvrtcGetPCHHeapSize(size_t* ret) {
  if (const RealNvrtc* r = real_nvrtc()) return r->pch_heap_get ? r->pch_heap_get(ret) : missing("nvrtcGetPCHHeapSize");
  if (!ret) return NVRTC_ERROR_INVALID_INPUT;
  *ret = g_pch_heap.load();
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcSetPCHHeapSize(size_t size) {
  if (const RealNvrtc* r = real_nvrtc()) return r->pch_heap_set ? r->pch_heap_set(size) : missing("nvrtcSetPCHHeapSize");
  g_pch_heap.store(size > SIZE_MAX - 4095 ? size : (size + 4095) / 4096 * 4096);
  return NVRTC_SUCCESS;
}
// Without the toolkit's NVRTC no precompiled header is ever made (the options
// that ask for one are refused), so none was attempted.
VGPU_EXPORT nvrtcResult nvrtcGetPCHCreateStatus(nvrtcProgram prog) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) {
    const RealNvrtc& r = *real_nvrtc();
    return r.pch_status ? r.pch_status(p->real) : missing("nvrtcGetPCHCreateStatus");
  }
  return code(kNoPchCreateAttempted);
}
VGPU_EXPORT nvrtcResult nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) {
    const RealNvrtc& r = *real_nvrtc();
    return r.pch_required ? r.pch_required(p->real, size) : missing("nvrtcGetPCHHeapSizeRequired");
  }
  if (!size) return NVRTC_ERROR_INVALID_INPUT;
  *size = 0;
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcSetFlowCallback(nvrtcProgram prog, int (*callback)(void*, void*), void* payload) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (p->real) {
    const RealNvrtc& r = *real_nvrtc();
    return r.flow ? r.flow(p->real, callback, payload) : missing("nvrtcSetFlowCallback");
  }
  if (!callback) return NVRTC_ERROR_INVALID_INPUT;
  p->flow = callback;
  p->flow_payload = payload;
  return NVRTC_SUCCESS;
}

// For VirtualGPU's driver shim, found by dlsym when it holds a cubin it cannot
// run in full: the PTX of the program NVRTC made that cubin from, as a
// malloc'd copy the caller frees, or null.
extern "C" __attribute__((visibility("default"))) char* vgpu_nvrtc_ptx_for_cubin(const void* cubin, size_t size) {
  std::lock_guard<std::mutex> l(g_aside_mu);
  const auto it = g_aside.find(key_of(cubin, size));
  if (it == g_aside.end()) return nullptr;
  char* copy = static_cast<char*>(std::malloc(it->second.size() + 1));
  if (copy) std::memcpy(copy, it->second.c_str(), it->second.size() + 1);
  return copy;
}
