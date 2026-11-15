// A stand-in for ROCm's libhiprtc, for tests/hiprtc/test_hiprtc_cache.cpp: the entry points of the real library with a "compiler"
// whose output is a function of everything a real compile depends on, so that the test can tell a result that was kept for the
// wrong inputs from the right one, without ROCm.
//
// A compile appends a line to the file $FAKE_HIPRTC_LOG (the test counts real compiles there, across processes) and loads, as
// the real library does, the compiler library and the built-in header library by name from beside itself (libamd_comgr.so.3,
// libhiprtc-builtins.so). The code it "generates" is text made of: the compiler library's version, the built-in header, the
// device bitcode file <root>/llvm/lib/clang/19/lib/amdgcn/bitcode/hip.bc and the configuration file <root>/llvm/bin/rocm.cfg
// (root: the directory above the one this library is in), the architecture of the current device (asked of hipGetDevice and
// hipGetDevicePropertiesR0600 in the process), $HIP_FAKE_FLAVOUR, the program's name, source, headers, options and name
// expressions, and for `#include <hip/hip_runtime.h>` the file <root>/include/hip/hip_runtime.h. A source with "FAIL" in it does
// not compile; "WARN" leaves a warning in the log; "SIZE=<n>" pads the code to n bytes; "__DATE__" puts the time in the code.
#include <dlfcn.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define API extern "C" __attribute__((visibility("default")))

namespace {
const char* const kDependencies[] = {"libamd_comgr.so.3", "libhiprtc-builtins.so"};   // (the names the cache reads out of this file)

// A dependency by its name, which the loader finds through this library's RUNPATH ($ORIGIN). Under ThreadSanitizer dlopen is
// libtsan's interceptor, so the loader takes libtsan for the caller and never looks at that RUNPATH: then the same file is opened by
// its path, beside this library.
void* open_dependency(const char* name) {
  if (void* h = dlopen(name, RTLD_NOW | RTLD_LOCAL)) return h;
  Dl_info me{};
  if (!dladdr(reinterpret_cast<void*>(&open_dependency), &me) || !me.dli_fname) return nullptr;
  const std::string self = me.dli_fname;
  const size_t slash = self.rfind('/');
  if (slash == std::string::npos) return nullptr;
  return dlopen((self.substr(0, slash + 1) + name).c_str(), RTLD_NOW | RTLD_LOCAL);
}

struct Program {
  std::string src, name;
  std::vector<std::string> headers, include_names, exprs, lowered;
  bool compiled = false;
  std::string code, log, bitcode;
};
struct LinkState {
  std::string all;
};

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}
std::string root() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&root), &info) || !info.dli_fname) return ".";
  std::string lib = info.dli_fname;
  lib = lib.substr(0, lib.rfind('/'));
  return lib + "/..";
}
std::string arch() {
  using GetDevice = int (*)(int*);
  using GetProps = int (*)(void*, int);
  const auto get_device = reinterpret_cast<GetDevice>(dlsym(RTLD_DEFAULT, "hipGetDevice"));
  const auto get_props = reinterpret_cast<GetProps>(dlsym(RTLD_DEFAULT, "hipGetDevicePropertiesR0600"));
  int id = 0;
  std::vector<char> props(16384, 0);
  if (!get_device || !get_props || get_device(&id) != 0 || get_props(props.data(), id) != 0) return "no-device";
  return std::string(&props[1024]);   // (where the test's stand-in for the HIP runtime puts gcnArchName)
}
}  // namespace

API const char* hiprtcGetErrorString(int r) { return r == 0 ? "FAKE_SUCCESS" : r == 6 ? "FAKE_ERROR_COMPILATION" : "FAKE_ERROR"; }
API int hiprtcVersion(int* major, int* minor) {
  if (!major || !minor) return 3;
  *major = 9, *minor = 7;
  return 0;
}
API int hiprtcCreateProgram(void** prog, const char* src, const char* name, int n, const char** headers, const char** include_names) {
  if (!prog || !src || n < 0) return 3;
  auto* p = new Program;
  p->src = src, p->name = name ? name : "(null)";
  for (int i = 0; i < n; ++i) p->headers.push_back(headers[i]), p->include_names.push_back(include_names[i]);
  *prog = p;
  return 0;
}
API int hiprtcDestroyProgram(void** prog) {
  if (!prog || !*prog) return 3;
  delete static_cast<Program*>(*prog);   // (the handle is left as it was, as ROCm's library leaves it)
  return 0;
}
API int hiprtcAddNameExpression(void* prog, const char* e) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (!e) return 3;
  if (p->compiled) return 8;
  p->exprs.push_back(e);
  return 0;
}
API int hiprtcCompileProgram(void* prog, int n, const char** options) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (n < 0 || (n > 0 && !options)) return 3;
  if (const char* log = std::getenv("FAKE_HIPRTC_LOG")) {
    std::string line = "compile " + p->name + " pid " + std::to_string(getpid()) + "\n";
    if (FILE* f = std::fopen(log, "a")) {
      std::fwrite(line.data(), 1, line.size(), f);
      std::fclose(f);
    }
  }
  if (const char* ms = std::getenv("FAKE_HIPRTC_SLEEP_MS")) usleep(static_cast<useconds_t>(std::atoi(ms)) * 1000);
  // The compiler library and the built-in header, loaded by name as the real library loads them.
  void* comgr = open_dependency(kDependencies[0]);
  void* builtins = open_dependency(kDependencies[1]);
  using VersionFn = const char* (*)();
  const auto version = comgr ? reinterpret_cast<VersionFn>(dlsym(comgr, "fake_comgr_version")) : nullptr;
  const char* header = builtins ? static_cast<const char*>(dlsym(builtins, "__hipRTC_header")) : nullptr;
  if (!version || !header) return p->log = "fake: no compiler library beside the fake libhiprtc", 11;
  p->compiled = true;
  if (p->src.find("FAIL") != std::string::npos) {
    p->log = "k.cu:1:1: error: the fake compiler does not compile FAIL\n";
    p->code.clear();
    return 6;
  }
  std::string code = "FAKE CODE OBJECT\n";
  code += std::string("comgr: ") + version() + "\n";
  code += "builtins: " + std::to_string(std::strlen(header)) + " " + std::string(header).substr(0, 60) + "\n";
  code += "bitcode: " + slurp(root() + "/llvm/lib/clang/19/lib/amdgcn/bitcode/hip.bc") + "\n";
  code += "cfg: " + slurp(root() + "/llvm/bin/rocm.cfg") + "\n";
  code += "arch: " + arch() + "\n";
  code += std::string("flavour: ") + (std::getenv("HIP_FAKE_FLAVOUR") ? std::getenv("HIP_FAKE_FLAVOUR") : "") + "\n";
  code += "name: " + p->name + "\n";
  for (int i = 0; i < n; ++i) code += std::string("option: ") + (options[i] ? options[i] : "(null)") + "\n";
  for (size_t i = 0; i < p->headers.size(); ++i) code += "header " + p->include_names[i] + ": " + p->headers[i] + "\n";
  if (p->src.find("#include <hip/hip_runtime.h>") != std::string::npos) code += "hip_runtime.h: " + slurp(root() + "/include/hip/hip_runtime.h") + "\n";
  if (p->src.find("#include \"local.h\"") != std::string::npos) code += "local.h: " + slurp("local.h") + "\n";
  if (p->src.find("__DATE__") != std::string::npos)
    code += "date: " + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "\n";
  for (int i = 0; i < n; ++i)
    if (options[i] && !std::strcmp(options[i], "-g")) code += "debug: /tmp/comgr-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "\n";
  code += "source: " + p->src + "\n";
  p->lowered.clear();
  for (const std::string& e : p->exprs) {
    code += "expr: " + e + "\n";
    p->lowered.push_back("_ZN4fake" + std::to_string(e.size()) + e + "E");
  }
  code.push_back('\0');   // (a code object is not text: bytes of every kind follow)
  for (int i = 0; i < 300; ++i) code.push_back(static_cast<char>(i * 7 + 3));
  if (const size_t at = p->src.find("SIZE="); at != std::string::npos) {
    const size_t want = std::strtoull(p->src.c_str() + at + 5, nullptr, 10);
    while (code.size() < want) code.push_back(static_cast<char>(code.size() * 31));
  }
  p->code = code;
  p->bitcode = "BC\xC0\xDE" + p->src;
  p->log = p->src.find("WARN") != std::string::npos ? "k.cu:2:3: warning: the fake compiler warns about WARN\n" : "";
  return 0;
}
API int hiprtcGetCodeSize(void* prog, size_t* size) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (!size) return 3;
  if (!p->compiled) return 4;
  *size = p->code.size();
  return 0;
}
API int hiprtcGetCode(void* prog, char* code) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (!code) return 3;
  if (!p->compiled) return 4;
  std::memcpy(code, p->code.data(), p->code.size());
  return 0;
}
// (As the real library: the size counts the terminating zero, and the log is copied with it.)
API int hiprtcGetProgramLogSize(void* prog, size_t* size) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (!size) return 3;
  *size = p->log.empty() ? 0 : p->log.size() + 1;
  return 0;
}
API int hiprtcGetProgramLog(void* prog, char* log) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (!log) return 3;
  if (!p->log.empty()) std::memcpy(log, p->log.c_str(), p->log.size() + 1);
  return 0;
}
API int hiprtcGetLoweredName(void* prog, const char* e, const char** out) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (!e || !out) return 3;
  if (!p->compiled) return 9;
  for (size_t i = 0; i < p->exprs.size(); ++i)
    if (p->exprs[i] == e) return *out = p->lowered[i].c_str(), 0;
  return 10;
}
API int hiprtcGetBitcodeSize(void* prog, size_t* size) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (!size) return 3;
  if (!p->compiled) return 4;
  *size = p->bitcode.size();
  return 0;
}
API int hiprtcGetBitcode(void* prog, char* out) {
  auto* p = static_cast<Program*>(prog);
  if (!p) return 4;
  if (!out) return 3;
  if (!p->compiled) return 4;
  std::memcpy(out, p->bitcode.data(), p->bitcode.size());
  return 0;
}
API int hiprtcLinkCreate(unsigned, void*, void**, void** state) {
  if (!state) return 3;
  *state = new LinkState;
  return 0;
}
API int hiprtcLinkAddFile(void* state, int, const char* path, unsigned, void*, void**) {
  if (!state || !path) return 3;
  static_cast<LinkState*>(state)->all += "file " + std::string(path) + ";";
  return 0;
}
API int hiprtcLinkAddData(void* state, int, void* image, size_t size, const char* name, unsigned, void*, void**) {
  if (!state || !image) return 3;
  static_cast<LinkState*>(state)->all += "data " + std::string(name ? name : "") + " " + std::string(static_cast<const char*>(image), size) + ";";
  return 0;
}
API int hiprtcLinkComplete(void* state, void** bin, size_t* size) {
  if (!state || !bin || !size) return 3;
  auto* s = static_cast<LinkState*>(state);
  *bin = s->all.data(), *size = s->all.size();
  return 0;
}
API int hiprtcLinkDestroy(void* state) {
  if (!state) return 3;
  delete static_cast<LinkState*>(state);
  return 0;
}
