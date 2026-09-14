// libvgpunvrtc -- VirtualGPU's NVRTC, presented as libnvrtc.so.13.
//
// NVRTC's job is to turn a string of CUDA C++ into PTX at run time. That is a
// C++ compiler, and there is no honest way to fake one -- so this shim does not
// try. It writes the program out and invokes `nvcc --ptx`, then hands back the
// PTX and the compiler's diagnostics as the program log.
//
// The toolkit is a host-side dependency and needs no GPU, so this works on the
// same CPU-only box everything else here runs on. If nvcc is not on PATH the
// call fails with a log saying exactly that, rather than a mystery
// NVRTC_ERROR_COMPILATION.
//
// This is what makes the JIT frameworks reachable: CuPy, Numba, Triton and
// PyTorch's inductor all compile kernels through NVRTC and then load the PTX
// through the driver API -- which VirtualGPU already interprets.
//
// Not implemented: CUBIN, LTO-IR and OptiX-IR output (all of which are SASS or
// vendor bitcode, neither of which VirtualGPU can execute), precompiled
// headers, and the time-trace files. Those return a clear status.
#include <nvrtc.h>

#include <cuda.h>  // CUDA_VERSION: the toolkit this shim was built against

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool trace() { const char* t = std::getenv("VGPU_TRACE"); return t && t[0] == '1'; }

struct Program {
  std::string source;
  std::string name = "default_program";
  std::vector<std::pair<std::string, std::string>> headers;  // include name -> contents
  std::vector<std::string> name_expressions;
  std::vector<std::string> lowered;   // parallel to name_expressions, after compile
  std::string ptx;
  std::string log;
  bool compiled = false;
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
      starts("--dopt=") || starts("--dlink-time-opt") || starts("-dlto") ||
      starts("--split-compile=") || starts("--minimal") || starts("--time=") ||
      starts("--builtin-move-forward=") || starts("--builtin-initializer-list=")) {
    // Accepted by NVRTC, meaningless or unavailable here. Say so under trace
    // rather than failing the compile over a flag that changes nothing.
    if (trace()) std::fprintf(stderr, "[vgpu] nvrtc: ignoring option %s\n", opt.c_str());
    return true;
  }
  *log += "vgpu nvrtc: unrecognised option '" + opt + "'\n";
  return false;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

VGPU_EXPORT const char* nvrtcGetErrorString(nvrtcResult r) {
  switch (r) {
    case NVRTC_SUCCESS: return "NVRTC_SUCCESS";
    case NVRTC_ERROR_OUT_OF_MEMORY: return "NVRTC_ERROR_OUT_OF_MEMORY";
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE: return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case NVRTC_ERROR_INVALID_INPUT: return "NVRTC_ERROR_INVALID_INPUT";
    case NVRTC_ERROR_INVALID_PROGRAM: return "NVRTC_ERROR_INVALID_PROGRAM";
    case NVRTC_ERROR_INVALID_OPTION: return "NVRTC_ERROR_INVALID_OPTION";
    case NVRTC_ERROR_COMPILATION: return "NVRTC_ERROR_COMPILATION";
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE: return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID: return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    default: return "NVRTC_ERROR_INTERNAL_ERROR";
  }
}

VGPU_EXPORT nvrtcResult nvrtcVersion(int* major, int* minor) {
  if (!major || !minor) return NVRTC_ERROR_INVALID_INPUT;
  // The compiler that will actually run is whichever nvcc is on PATH, so report
  // its version rather than the header's; fall back to the build-time toolkit
  // when nvcc cannot be asked.
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

// The architectures VirtualGPU's profiles cover, ascending as the API requires.
static const int kArchs[] = {70, 75, 80, 86, 89, 90, 100, 120};

VGPU_EXPORT nvrtcResult nvrtcGetNumSupportedArchs(int* n) {
  if (!n) return NVRTC_ERROR_INVALID_INPUT;
  *n = (int)(sizeof(kArchs) / sizeof(kArchs[0]));
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetSupportedArchs(int* out) {
  if (!out) return NVRTC_ERROR_INVALID_INPUT;
  std::memcpy(out, kArchs, sizeof(kArchs));
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name,
                                           int num_headers, const char* const* headers,
                                           const char* const* include_names) {
  if (!prog || !src) return NVRTC_ERROR_INVALID_INPUT;
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
  if (!prog || !*prog) return NVRTC_SUCCESS;
  Program* p = get(*prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.erase(p);
  }
  delete p;
  *prog = nullptr;
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* expr) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (!expr) return NVRTC_ERROR_INVALID_INPUT;
  if (p->compiled) return NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
  p->name_expressions.emplace_back(expr);
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int num_options,
                                            const char* const* options) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  p->log.clear();
  p->ptx.clear();
  p->lowered.clear();
  p->compiled = true;   // "compilation was attempted", which is what the API means

  if (!have_nvcc()) {
    p->log =
        "vgpu nvrtc: nvcc was not found on PATH.\n"
        "VirtualGPU implements NVRTC by invoking the CUDA toolkit's compiler, which runs on the "
        "host and needs no GPU. Install the toolkit, or put nvcc on PATH, to compile at run "
        "time.\n";
    return NVRTC_ERROR_COMPILATION;
  }

  std::vector<std::string> nvcc_opts;
  for (int i = 0; i < num_options; ++i) {
    if (!options || !options[i]) return NVRTC_ERROR_INVALID_INPUT;
    if (!translate_option(options[i], &nvcc_opts, &p->log)) return NVRTC_ERROR_INVALID_OPTION;
  }

  const std::string dir = tempdir();
  // Keep the scratch tree when tracing, so a failed compile can be inspected.
  struct Cleanup {
    std::string dir;
    ~Cleanup() { if (!trace()) remove_tree(dir); }
  } cleanup{dir};

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

  std::string cmd = "nvcc --ptx -Wno-deprecated-gpu-targets -I " + shq(dir);
  for (const auto& o : nvcc_opts) cmd += " " + shq(o);
  cmd += " -o " + shq(ptx) + " " + shq(cu) + " 2> " + shq(err);
  if (trace()) std::fprintf(stderr, "[vgpu] nvrtc: %s\n", cmd.c_str());
  const int rc = std::system(cmd.c_str());

  p->log += read_file(err);
  if (rc != 0) return NVRTC_ERROR_COMPILATION;
  p->ptx = read_file(ptx);
  if (p->ptx.empty()) {
    p->log += "vgpu nvrtc: nvcc reported success but produced no PTX\n";
    return NVRTC_ERROR_COMPILATION;
  }

  // Pull each lowered name back out: the marker variable's initialiser names the
  // mangled symbol, e.g. ".global .align 8 .u64 vgpu_nvrtc_name_0 = _Z6kernelIfEvPf;"
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
  if (!size) return NVRTC_ERROR_INVALID_INPUT;
  *size = p->ptx.size() + 1;   // the API counts the terminating NUL
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* out) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (!out) return NVRTC_ERROR_INVALID_INPUT;
  std::memcpy(out, p->ptx.c_str(), p->ptx.size() + 1);
  return NVRTC_SUCCESS;
}

VGPU_EXPORT nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* size) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (!size) return NVRTC_ERROR_INVALID_INPUT;
  *size = p->log.size() + 1;
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* out) {
  Program* p = get(prog);
  if (!p) return NVRTC_ERROR_INVALID_PROGRAM;
  if (!out) return NVRTC_ERROR_INVALID_INPUT;
  std::memcpy(out, p->log.c_str(), p->log.size() + 1);
  return NVRTC_SUCCESS;
}

/* ---- output formats VirtualGPU cannot execute ---- */

static nvrtcResult unsupported_output(const char* what) {
  std::fprintf(stderr,
               "[vgpu] nvrtc: %s output is not available. VirtualGPU executes PTX, not SASS or "
               "vendor bitcode; ask for PTX instead.\n",
               what);
  return NVRTC_ERROR_INVALID_PROGRAM;
}
VGPU_EXPORT nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* size) {
  if (!get(prog)) return NVRTC_ERROR_INVALID_PROGRAM;
  if (size) *size = 0;   // documented: zero when -arch names a virtual architecture
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetCUBIN(nvrtcProgram, char*) { return unsupported_output("CUBIN"); }
VGPU_EXPORT nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* size) {
  if (!get(prog)) return NVRTC_ERROR_INVALID_PROGRAM;
  if (size) *size = 0;
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetLTOIR(nvrtcProgram, char*) { return unsupported_output("LTO-IR"); }
VGPU_EXPORT nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t* size) {
  if (!get(prog)) return NVRTC_ERROR_INVALID_PROGRAM;
  if (size) *size = 0;
  return NVRTC_SUCCESS;
}
VGPU_EXPORT nvrtcResult nvrtcGetOptiXIR(nvrtcProgram, char*) {
  return unsupported_output("OptiX-IR");
}
