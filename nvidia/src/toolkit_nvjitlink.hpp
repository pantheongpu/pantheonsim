// Turning LTO-IR into machine code with the CUDA toolkit's own libnvJitLink.
//
// LTO-IR is NVVM bitcode, which only NVIDIA's compiler reads (nvcc -dlto, and NVRTC's
// -dlto, produce it; cuFFT's LTO callbacks and cuSPARSE's SpMM operators are given as it).
// VirtualGPU has no such compiler. What it can do is what its NVRTC does: where the
// toolkit is installed on the host -- a host library that needs no GPU -- use its
// libnvJitLink to link the LTO-IR (with any PTX that goes with it) into a cubin for the
// simulated device's architecture, and run that. Without the toolkit the call reports
// that, and the caller fails as it did before.
//
// The library is opened by path, never by soname: this shim's own libnvJitLink would
// otherwise be found first on the library path.
#pragma once

#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace vgpu::cuda {

struct ToolkitLinkInput {
  enum Kind { Ptx, Fatbin, LtoIr } kind = Ptx;  // what the bytes are (nvJitLink's input kinds)
  std::string name;
  std::vector<char> bytes;  // PTX text needs its terminating NUL
};

struct ToolkitLinkResult {
  bool ok = false;
  std::string cubin;  // the linked image for sm_<arch>
  std::string log;    // errors, or why the toolkit's library could not be used
};

namespace detail {

struct NvJitLinkApi {
  void* lib = nullptr;
  std::string path;
  int (*create)(void**, unsigned, const char**) = nullptr;
  int (*destroy)(void**) = nullptr;
  int (*add_data)(void*, int, const void*, size_t, const char*) = nullptr;
  int (*complete)(void*) = nullptr;
  int (*cubin_size)(void*, size_t*) = nullptr;
  int (*cubin)(void*, void*) = nullptr;
  int (*error_log_size)(void*, size_t*) = nullptr;
  int (*error_log)(void*, char*) = nullptr;
};

inline std::string real_path_of(const std::string& p) {
  char buf[PATH_MAX];
  return ::realpath(p.c_str(), buf) ? std::string(buf) : std::string();
}

inline std::vector<std::string> nvjitlink_candidates() {
  std::vector<std::string> out;
  if (const char* e = std::getenv("VGPU_NVJITLINK_LIB"); e && *e) out.push_back(e);
  std::vector<std::string> roots;
  if (FILE* f = popen("command -v nvcc 2>/dev/null", "r")) {
    char buf[PATH_MAX] = {0};
    if (fgets(buf, sizeof buf, f)) {
      const std::string nvcc = real_path_of(std::string(buf).substr(0, std::strcspn(buf, "\n")));
      const size_t slash = nvcc.find_last_of('/');
      if (slash != std::string::npos) roots.push_back(nvcc.substr(0, slash) + "/..");
    }
    pclose(f);
  }
  for (const char* r : {"/usr/local/cuda"}) roots.push_back(r);
  for (const std::string& r : roots)
    for (const char* major : {"13", "12"})
      for (const char* sub : {"/lib64/", "/targets/x86_64-linux/lib/", "/targets/sbsa-linux/lib/"})
        out.push_back(r + sub + "libnvJitLink.so." + major);
  for (const char* dir : {"/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"})
    for (const char* major : {"13", "12"}) out.push_back(std::string(dir) + "/libnvJitLink.so." + major);
  return out;
}

inline const NvJitLinkApi* nvjitlink_api(std::string* why) {
  static std::string reason;
  static const NvJitLinkApi* found = [] () -> const NvJitLinkApi* {
    Dl_info self{};
    std::string own;
    if (dladdr(reinterpret_cast<void*>(&nvjitlink_api), &self) && self.dli_fname) own = real_path_of(self.dli_fname);
    // A sanitizer runtime refuses RTLD_DEEPBIND; there the library's own internal binding has to do.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    const int flags = RTLD_NOW | RTLD_LOCAL;
#else
    const int flags = RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND;
#endif
    for (const std::string& c : nvjitlink_candidates()) {
      const std::string rp = real_path_of(c);
      if (rp.empty() || rp == own) continue;
      void* lib = dlopen(rp.c_str(), flags);
      if (!lib) continue;
      auto* api = new NvJitLinkApi();
      api->lib = lib;
      api->path = rp;
      bool ok = true;
      auto sym = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
        ok = ok && fn;
      };
      sym(api->create, "nvJitLinkCreate");
      sym(api->destroy, "nvJitLinkDestroy");
      sym(api->add_data, "nvJitLinkAddData");
      sym(api->complete, "nvJitLinkComplete");
      sym(api->cubin_size, "nvJitLinkGetLinkedCubinSize");
      sym(api->cubin, "nvJitLinkGetLinkedCubin");
      sym(api->error_log_size, "nvJitLinkGetErrorLogSize");
      sym(api->error_log, "nvJitLinkGetErrorLog");
      if (ok) return api;
      delete api;
      dlclose(lib);
    }
    reason = "LTO-IR needs the CUDA toolkit's libnvJitLink on the host to turn it into machine code, and none was found "
             "(VGPU_NVJITLINK_LIB names one)";
    return nullptr;
  }();
  if (!found && why) *why = reason;
  return found;
}

}  // namespace detail

// Links `inputs` (PTX, fatbins and LTO-IR) into a cubin for sm_<arch> with link-time
// optimisation, using the toolkit's libnvJitLink.
inline ToolkitLinkResult toolkit_lto_link(const std::vector<ToolkitLinkInput>& inputs, int arch) {
  ToolkitLinkResult r;
  const detail::NvJitLinkApi* api = detail::nvjitlink_api(&r.log);
  if (!api) return r;
  const std::string archopt = "-arch=sm_" + std::to_string(arch);
  const char* opts[] = {"-lto", archopt.c_str()};
  void* h = nullptr;
  if (api->create(&h, 2, opts) != 0) {
    r.log = "nvJitLinkCreate failed";
    return r;
  }
  auto fail_with_log = [&](const char* what) {
    size_t n = 0;
    r.log = what;
    if (api->error_log_size(h, &n) == 0 && n > 1) {
      std::string e(n, '\0');
      if (api->error_log(h, &e[0]) == 0) r.log += ":\n" + e;
    }
    api->destroy(&h);
    return r;
  };
  for (const ToolkitLinkInput& in : inputs) {
    // nvJitLink's kinds: PTX 2, LTO-IR 3, fatbin 4.
    const int kind = in.kind == ToolkitLinkInput::Ptx ? 2 : in.kind == ToolkitLinkInput::LtoIr ? 3 : 4;
    if (api->add_data(h, kind, in.bytes.data(), in.bytes.size(), in.name.c_str()) != 0)
      return fail_with_log(("nvJitLink refused " + in.name).c_str());
  }
  if (api->complete(h) != 0) return fail_with_log("nvJitLink could not link");
  size_t n = 0;
  if (api->cubin_size(h, &n) != 0 || n == 0) return fail_with_log("nvJitLink produced no cubin");
  r.cubin.assign(n, '\0');
  if (api->cubin(h, &r.cubin[0]) != 0) return fail_with_log("nvJitLink could not return the cubin");
  api->destroy(&h);
  r.ok = true;
  return r;
}

}  // namespace vgpu::cuda
