// libvgpunvjitlink -- VirtualGPU's nvJitLink, presented as libnvJitLink.so.13
// (.so.12 when built against a CUDA 12 toolkit).
//
// nvJitLink is the device linker as a library: it takes PTX, cubins, fatbins,
// host objects and libraries carrying device code, and LTO-IR, and links them
// into one cubin a program then loads with cuModuleLoadData. NVIDIA's compiles
// the PTX to SASS on the way. VirtualGPU executes PTX, so here linking stops
// at PTX: the inputs' PTX is linked into one module (ptx_link.cpp -- the
// resolution rules are the device linker's, measured on an RTX 3060), and that
// module is the "cubin". It is the same choice NVRTC's shim makes for
// nvrtcGetCUBIN: the caller hands the image to cuModuleLoadData, or
// cuLibraryLoadData, or writes it to a file for cuModuleLoad, and VirtualGPU's
// driver loads PTX from every one of those. nvJitLinkGetLinkedPtx hands out the
// same module, without the -lto -ptx NVIDIA's requires, since it is what this
// linker produces either way.
//
// What can be linked: PTX; a fatbin's PTX (the image the driver would JIT for
// -arch, as pick_ptx chooses it); a host object's or a static library's
// device code, through the fatbins nvcc puts in their .nv_fatbin and
// __nv_relfatbin sections; and this library's own output, or NVRTC's shim's
// "CUBIN", which are PTX, given back as NVJITLINK_INPUT_CUBIN. A linked cubin
// (SASS, nvcc -cubin) is accepted and adds nothing, as NVIDIA's treats one.
//
// Relocatable SASS -- -rdc / -dc cubins, and fatbins, objects and libraries
// carrying them -- links too, as machine code (sass_link.cpp): when every
// input has SASS for -arch and some input has no PTX, the link is SASS, and
// the cubin is a real one, which VirtualGPU's driver runs as SASS. When every
// input has PTX the link stays PTX, as before: it is the form the PTX engine
// and every older path take. What cannot be linked is SASS with PTX that has
// no SASS beside it -- NVIDIA's compiles the PTX for -arch; there is no
// compiler here -- which is refused by name when the second kind arrives, and
// LTO-IR (NVVM bitcode, which only NVIDIA's compiler reads).
//
// What it cannot do itself -- LTO-IR, which only NVIDIA's compiler reads, index
// files, and SASS beside PTX that has no SASS, which needs a PTX compiler -- is
// done by the toolkit's own libnvJitLink where that is installed (it is a host
// library and needs no GPU), the way libvgpunvrtc uses the toolkit's NVRTC: the
// inputs are handed to it as given and what it links, a cubin of SASS or the
// PTX of `-lto -ptx`, comes back. Without it these inputs are refused by name,
// as before. (Nothing else goes that way: everything this library links itself
// stays its own.)
//
// Results, error-log text and the order checks happen in follow NVIDIA's
// library as measured on CUDA 13.0's: a failed nvJitLinkCreate still returns
// a handle whose log says why; an undefined reference fails the link with
// NVJITLINK_ERROR_INTERNAL; a second definition of a symbol is logged and
// dropped while the link succeeds.
#include "../include/vgpu_nvjitlink.h"

#include <dlfcn.h>
#include <limits.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "fatbin.hpp"
#include "ptx_link.hpp"
#include "sass_link.hpp"
#include "vgpu/error.hpp"

// CUDA_VERSION of the toolkit this library stands in for (CMake passes it).
#ifndef VGPU_NVJITLINK_VERSION
#define VGPU_NVJITLINK_VERSION 13000
#endif

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

// ---- the toolkit's nvJitLink, for what this library cannot link ----

struct RealJitLink {
  void* lib = nullptr;
  nvJitLinkResult (*create)(nvJitLinkHandle*, uint32_t, const char**) = nullptr;
  nvJitLinkResult (*destroy)(nvJitLinkHandle*) = nullptr;
  nvJitLinkResult (*add_data)(nvJitLinkHandle, int, const void*, size_t, const char*) = nullptr;   // (the input type is an int on the wire)
  nvJitLinkResult (*complete)(nvJitLinkHandle) = nullptr;
  nvJitLinkResult (*cubin_size)(nvJitLinkHandle, size_t*) = nullptr;
  nvJitLinkResult (*cubin)(nvJitLinkHandle, void*) = nullptr;
  nvJitLinkResult (*ptx_size)(nvJitLinkHandle, size_t*) = nullptr;
  nvJitLinkResult (*ptx)(nvJitLinkHandle, char*) = nullptr;
  nvJitLinkResult (*error_log_size)(nvJitLinkHandle, size_t*) = nullptr;
  nvJitLinkResult (*error_log)(nvJitLinkHandle, char*) = nullptr;
  nvJitLinkResult (*info_log_size)(nvJitLinkHandle, size_t*) = nullptr;
  nvJitLinkResult (*info_log)(nvJitLinkHandle, char*) = nullptr;
  std::string path;
};

std::string real_path_of(const std::string& p) {
  char buf[PATH_MAX];
  return ::realpath(p.c_str(), buf) ? std::string(buf) : std::string();
}

const RealJitLink* toolkit_linker() {
  static const RealJitLink* found = []() -> const RealJitLink* {
    if (const char* m = std::getenv("VGPU_NVJITLINK"); m && std::strcmp(m, "own") == 0) return nullptr;
    const std::string so = "libnvJitLink.so." + std::to_string(VGPU_NVJITLINK_VERSION / 1000);
    std::string self;
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&toolkit_linker), &info) && info.dli_fname) self = real_path_of(info.dli_fname);
    std::vector<std::string> candidates;
    if (const char* e = std::getenv("VGPU_NVJITLINK_LIB"); e && *e) candidates.push_back(e);
    if (FILE* f = popen("command -v nvcc 2>/dev/null", "r")) {
      char buf[PATH_MAX] = {0};
      if (fgets(buf, sizeof buf, f)) {
        const std::string nvcc = real_path_of(std::string(buf).substr(0, std::strcspn(buf, "\n")));
        const size_t slash = nvcc.find_last_of('/');
        if (slash != std::string::npos) {
          const std::string root = nvcc.substr(0, slash) + "/..";
          candidates.push_back(root + "/lib64/" + so);
          candidates.push_back(root + "/targets/x86_64-linux/lib/" + so);
          candidates.push_back(root + "/targets/sbsa-linux/lib/" + so);
        }
      }
      pclose(f);
    }
    for (const char* dir : {"/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"})
      candidates.push_back(std::string(dir) + "/" + so);
    // As for NVRTC: the real library keeps its own calls to itself, where a
    // sanitizer runtime allows the flag.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    const int flags = RTLD_NOW | RTLD_LOCAL;
#else
    const int flags = RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND;
#endif
    for (const std::string& c : candidates) {
      const std::string rp = real_path_of(c);
      if (rp.empty() || rp == self) continue;
      void* lib = dlopen(rp.c_str(), flags);
      if (!lib) continue;
      auto* r = new RealJitLink();
      r->lib = lib;
      r->path = rp;
      bool ok = true;
      auto sym = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
        ok = ok && fn;
      };
      sym(r->create, "nvJitLinkCreate");
      sym(r->destroy, "nvJitLinkDestroy");
      sym(r->add_data, "nvJitLinkAddData");
      sym(r->complete, "nvJitLinkComplete");
      sym(r->cubin_size, "nvJitLinkGetLinkedCubinSize");
      sym(r->cubin, "nvJitLinkGetLinkedCubin");
      sym(r->ptx_size, "nvJitLinkGetLinkedPtxSize");
      sym(r->ptx, "nvJitLinkGetLinkedPtx");
      sym(r->error_log_size, "nvJitLinkGetErrorLogSize");
      sym(r->error_log, "nvJitLinkGetErrorLog");
      sym(r->info_log_size, "nvJitLinkGetInfoLogSize");
      sym(r->info_log, "nvJitLinkGetInfoLog");
      if (ok) return r;
      delete r;
      dlclose(lib);
    }
    return nullptr;
  }();
  return found;
}

struct Link {
  std::string arch;          // "sm_86", as -arch named it
  uint32_t arch_number = 0;
  char arch_suffix = 0;
  bool lto = false;
  bool ptx_output = false;
  bool verbose = false;
  vgpu::cuda::SassLinkOptions sass_options;   // -g, -kernels-used
  bool failed_create = false;   // the handle exists only so its log can be read
  bool completed = false;
  bool linked_ok = false;
  std::vector<vgpu::cuda::PtxInput> inputs;
  // Each input, by what it brings: PTX (an index into `inputs`), relocatable
  // SASS for -arch (into `sass`), or both (a fatbin or object built for both).
  struct Piece {
    std::string label;
    int ptx = -1, sass = -1;
  };
  std::vector<Piece> pieces;
  std::vector<vgpu::cuda::SassLinkInput> sass;
  bool linked_sass = false;   // `linked` is a cubin (ELF), not PTX text
  // The toolkit's linker finishes the link when an input is one only it reads:
  // every input as given is kept for it, and what it makes comes back here.
  struct Raw {
    int type;   // the input type as the caller gave it (maybe past what this header's enum holds)
    std::string data, label;
  };
  std::vector<std::string> option_strings;
  std::vector<Raw> raw;
  bool delegate = false;
  // What the toolkit's linker made, as it made it, and what it answered to asking for each.
  std::string d_cubin, d_ptx;
  nvJitLinkResult d_cubin_rc = NVJITLINK_ERROR_INVALID_INPUT, d_ptx_rc = NVJITLINK_ERROR_INVALID_INPUT;
  size_t unnamed = 0;
  std::string linked;
  std::string error_log;
  std::string info_log;
};

std::mutex g_mu;
std::set<const void*> g_live;

Link* get(nvJitLinkHandle h) {
  std::lock_guard<std::mutex> l(g_mu);
  return g_live.count(h) ? reinterpret_cast<Link*>(h) : nullptr;
}

// The architectures nvJitLink knows. Those older than the toolkit supports
// are known but refused (UNSUPPORTED_ARCH): CUDA 13 dropped everything before
// Turing, and its nvJitLink says so for -arch=sm_70.
bool known_arch(uint32_t n) {
  static const uint32_t kKnown[] = {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87,
                                    88, 89, 90, 100, 101, 103, 110, 120, 121};
  for (uint32_t k : kKnown)
    if (k == n) return true;
  return false;
}
uint32_t oldest_supported_arch() { return VGPU_NVJITLINK_VERSION >= 13000 ? 75 : 50; }

nvJitLinkResult parse_options(Link& L, uint32_t n, const char* const* options) {
  bool virtual_arch = false;
  auto accepts_value = [](const std::string& o) {
    static const char* kWithValue[] = {
        "-maxrregcount=", "-ftz=", "-prec-div=", "-prec-sqrt=", "-fma=", "-kernels-used=",
        "-variables-used=", "-Xptxas=", "-Xnvvm=", "-split-compile=", "-split-compile-extended=",
        "-jump-table-density=", "-device-stack-protector="};
    for (const char* p : kWithValue)
      if (o.rfind(p, 0) == 0 && o.size() > std::strlen(p)) return true;
    return false;
  };
  for (uint32_t i = 0; i < n; ++i) {
    const std::string o = options[i];
    if (o.rfind("-arch=", 0) == 0) {
      const std::string a = o.substr(6);
      uint32_t num = 0;
      char suffix = 0;
      if (!vgpu::cuda::parse_arch(a, &num, &suffix)) {
        L.error_log += "ERROR NVJITLINK_ERROR_UNRECOGNIZED_OPTION: missing option value (" + o + ")\n";
        return NVJITLINK_ERROR_UNRECOGNIZED_OPTION;
      }
      const std::string compute = "compute_" + std::to_string(num);
      if (!known_arch(num)) {
        L.error_log += "ERROR NVJITLINK_ERROR_UNRECOGNIZED_ARCH: unrecognized architecture (" +
                       compute + ")\n";
        return NVJITLINK_ERROR_UNRECOGNIZED_ARCH;
      }
      if (num < oldest_supported_arch()) {
        L.error_log += "ERROR NVJITLINK_ERROR_UNSUPPORTED_ARCH: unsupported architecture (" +
                       compute + ")\n";
        return NVJITLINK_ERROR_UNSUPPORTED_ARCH;
      }
      // The last -arch wins, as it does in NVIDIA's.
      L.arch = "sm_" + std::to_string(num) + (suffix ? std::string(1, suffix) : std::string());
      L.arch_number = num;
      L.arch_suffix = suffix;
      virtual_arch = a.rfind("compute_", 0) == 0;
    } else if (o == "-lto") {
      L.lto = true;
    } else if (o == "-ptx") {
      L.ptx_output = true;
    } else if (o == "-verbose") {
      L.verbose = true;
    } else if (o == "-g") {
      L.sass_options.keep_unused = true;
    } else if (o.rfind("-kernels-used=", 0) == 0 && o.size() > 14) {
      L.sass_options.kernels_used.push_back(o.substr(14));
    } else if (o == "-time" || o == "-lineinfo" || o == "-no-cache" ||
               o == "-optimize-unused-variables" ||
               (o.size() == 3 && o[0] == '-' && o[1] == 'O' && std::isdigit(static_cast<unsigned char>(o[2]))) ||
               accepts_value(o)) {
      // Code-generation choices: there is no code generator behind this
      // linker for them to steer.
    } else {
      L.error_log += "unrecognized option: " + o + "\n";
      return NVJITLINK_ERROR_UNRECOGNIZED_OPTION;
    }
  }
  if (L.arch.empty()) {
    L.error_log += "ERROR: -arch=sm_NN is required\n";
    return NVJITLINK_ERROR_MISSING_ARCH;
  }
  if (virtual_arch && !L.ptx_output) {
    L.error_log += "ERROR: virtual arch requires -ptx\n";
    return NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS;
  }
  if (L.ptx_output && !L.lto) {
    L.error_log += "ERROR: -ptx requires -lto\n";
    return NVJITLINK_ERROR_INCOMPATIBLE_OPTIONS;
  }
  return NVJITLINK_SUCCESS;
}

nvJitLinkResult refuse_ltoir(Link& L, const std::string& label) {
  if (!L.lto) {
    L.error_log += "ERROR: LTO-IR input '" + label + "' cannot be used without LTO enabled (-lto)\n";
    return NVJITLINK_ERROR_LTO_NOT_ENABLED;
  }
  if (toolkit_linker()) {   // NVIDIA's compiles it when the link completes
    L.delegate = true;
    return NVJITLINK_SUCCESS;
  }
  L.error_log += "ERROR: LTO-IR input '" + label +
                 "' is NVVM bitcode, which only NVIDIA's compiler reads, and the toolkit's libnvJitLink "
                 "(which this library hands such inputs to) was not found. Install the toolkit, or build "
                 "the input as PTX (-gencode arch=compute_XX,code=compute_XX; NVRTC: without -dlto) and add that\n";
  return NVJITLINK_ERROR_NVVM_COMPILE;
}

// SASS and PTX without SASS cannot be linked together: NVIDIA's compiles the
// PTX for -arch first, and there is no compiler here. Refused when the second
// kind arrives, naming both.
nvJitLinkResult refuse_mixing(Link& L, const std::string& label, bool adding_sass) {
  if (toolkit_linker()) {   // NVIDIA's compiles the PTX when the link completes
    L.delegate = true;
    return NVJITLINK_SUCCESS;
  }
  std::string other;
  for (const Link::Piece& p : L.pieces)
    if (adding_sass ? p.sass < 0 : p.ptx < 0) {
      other = p.label;
      break;
    }
  if (adding_sass)
    L.error_log += "ERROR: '" + label + "' is relocatable SASS, and '" + other +
                   "' is PTX with no SASS for " + L.arch +
                   ": VirtualGPU links SASS with SASS and PTX with PTX, and compiling PTX to SASS is the "
                   "toolkit's libnvJitLink's, which was not found. Add the PTX this input was built from, or SASS for the PTX one\n";
  else
    L.error_log += "ERROR: '" + label + "' is PTX with no SASS for " + L.arch + ", and '" + other +
                   "' is relocatable SASS: VirtualGPU links SASS with SASS and PTX with PTX, and compiling "
                   "PTX to SASS is the toolkit's libnvJitLink's, which was not found\n";
  return NVJITLINK_ERROR_INVALID_INPUT;
}
bool has_sass_only(const Link& L) {
  for (const Link::Piece& p : L.pieces)
    if (p.ptx < 0) return true;
  return false;
}
bool has_ptx_only(const Link& L) {
  for (const Link::Piece& p : L.pieces)
    if (p.sass < 0) return true;
  return false;
}

// Whether SASS built for sm_<sass> runs on -arch: machine code carries over
// within a major version only, to the same or a later minor (sm_80's runs on
// sm_86 and sm_89, sm_100's on sm_103; none runs on another major, newer or
// not) -- unlike PTX, which any later architecture compiles.
bool sass_runs_on(uint32_t sass, uint32_t arch) { return sass / 10 == arch / 10 && sass <= arch; }

// Relocatable SASS for -arch, as a link input. A cubin for an architecture
// -arch cannot run is refused when it is added, as NVIDIA's refuses it
// (NVJITLINK_ERROR_INVALID_INPUT, "ERROR 4: bad input:<name>" on CUDA 13.0's).
nvJitLinkResult add_sass(Link& L, const void* data, size_t size, const std::string& label, std::string ptx = {}) {
  const uint32_t arch = vgpu::cuda::cubin_arch(data, size);
  if (!sass_runs_on(arch, L.arch_number)) {
    L.error_log += "ERROR 4: bad input:" + label + "\n";
    return NVJITLINK_ERROR_INVALID_INPUT;
  }
  const bool with_ptx = !ptx.empty();
  if (!with_ptx && has_ptx_only(L)) return refuse_mixing(L, label, true);
  Link::Piece p;
  p.label = label;
  p.sass = static_cast<int>(L.sass.size());
  const auto* b = static_cast<const uint8_t*>(data);
  L.sass.push_back({label, std::vector<uint8_t>(b, b + size)});
  if (with_ptx) {
    p.ptx = static_cast<int>(L.inputs.size());
    L.inputs.push_back({label, std::move(ptx)});
  }
  L.pieces.push_back(std::move(p));
  return NVJITLINK_SUCCESS;
}

nvJitLinkResult add_ptx(Link& L, std::string text, const std::string& label) {
  while (!text.empty() && text.back() == '\0') text.pop_back();
  uint32_t target = 0;
  char suffix = 0;
  if (vgpu::cuda::ptx_module_target(text, &target, &suffix) &&
      !vgpu::cuda::target_runs_on(target, suffix, L.arch_number, L.arch_suffix)) {
    // ptxas compiles each input for -arch as it is added, so this is where
    // NVIDIA's fails it too.
    L.error_log += "ptxas fatal   : '" + label + "' targets sm_" + std::to_string(target) +
                   (suffix ? std::string(1, suffix) : std::string()) +
                   ", which cannot be compiled for -arch=" + L.arch + "\n";
    L.error_log += "ERROR NVJITLINK_ERROR_PTX_COMPILE: JIT the PTX (" + label + ")\n";
    return NVJITLINK_ERROR_PTX_COMPILE;
  }
  if (has_sass_only(L)) return refuse_mixing(L, label, false);
  L.pieces.push_back({label, static_cast<int>(L.inputs.size()), -1});
  L.inputs.push_back({label, std::move(text)});
  return NVJITLINK_SUCCESS;
}

// A fatbin's contribution: the PTX the driver would JIT for -arch.
nvJitLinkResult add_fatbin(Link& L, const void* data, size_t size, const std::string& label) {
  std::vector<vgpu::cuda::FatbinImage> images;
  try {
    images = vgpu::cuda::extract_images(data, size);
  } catch (const vgpu::Error& e) {
    L.error_log += "ERROR: fatbin '" + label + "' cannot be read: " + e.message() + "\n";
    return NVJITLINK_ERROR_INVALID_INPUT;
  }
  std::vector<vgpu::cuda::FatbinPtx> ptx;
  bool ltoir = false;
  const vgpu::cuda::FatbinImage* sass = nullptr;   // the relocatable SASS for -arch, the newest that runs
  uint32_t other_sass = 0;
  for (auto& im : images) {
    if (im.is_ptx()) ptx.push_back({im.arch, im.data});
    else if (im.kind == vgpu::cuda::kFatbinLtoIr) ltoir = true;
    else if (im.kind == vgpu::cuda::kFatbinElf && !im.stored &&
             vgpu::cuda::cubin_relocatable(im.data.data(), im.data.size())) {
      const uint32_t a = vgpu::cuda::cubin_arch(im.data.data(), im.data.size());
      if (sass_runs_on(a, L.arch_number)) {
        if (!sass || a > vgpu::cuda::cubin_arch(sass->data.data(), sass->data.size())) sass = &im;
      } else {
        other_sass = a;
      }
    }
  }
  // Both, when it has both: the link is PTX unless some input has SASS only.
  if (sass)
    return add_sass(L, sass->data.data(), sass->data.size(), label,
                    ptx.empty() ? std::string() : ptx[vgpu::cuda::pick_ptx(ptx, L.arch_number)].text);
  if (!ptx.empty()) return add_ptx(L, std::move(ptx[vgpu::cuda::pick_ptx(ptx, L.arch_number)].text), label);
  if (ltoir) return refuse_ltoir(L, label);
  if (other_sass) {
    L.error_log += "ERROR 4: bad input:" + label + "\n";
    return NVJITLINK_ERROR_INVALID_INPUT;
  }
  // An empty fatbin, or one of linked cubins only, contributes nothing: NVIDIA's
  // nvJitLink links relocatable code, and passes over a linked cubin.
  return NVJITLINK_SUCCESS;
}

nvJitLinkResult add_object(Link& L, const void* data, size_t size, const std::string& label) {
  std::vector<std::string> fatbins;
  try {
    fatbins = vgpu::cuda::host_object_fatbins(data, size);
  } catch (const vgpu::Error& e) {
    L.error_log += "ERROR: object '" + label + "' cannot be read: " + e.message() + "\n";
    return NVJITLINK_ERROR_INVALID_INPUT;
  }
  // An object without device code is accepted and adds nothing, as it is in
  // NVIDIA's.
  for (const std::string& f : fatbins)
    if (nvJitLinkResult r = add_fatbin(L, f.data(), f.size(), label); r != NVJITLINK_SUCCESS) return r;
  return NVJITLINK_SUCCESS;
}

nvJitLinkResult add_input(Link& L, nvJitLinkInputType type, const void* data, size_t size,
                          const std::string& label) {
  using vgpu::cuda::BlobKind;
  const BlobKind kind = vgpu::cuda::classify_blob(data, size);
  auto wrong_type = [&](const char* type_name) {
    L.error_log += "ERROR: bad input: '" + label + "' does not match type " + type_name + "\n";
    return NVJITLINK_ERROR_INCORRECT_INPUT_TYPE;
  };
  switch (static_cast<int>(type)) {
    case NVJITLINK_INPUT_ANY:
      switch (kind) {
        case BlobKind::Ptx: return add_ptx(L, std::string(static_cast<const char*>(data), size), label);
        case BlobKind::Fatbin: return add_fatbin(L, data, size, label);
        case BlobKind::Cubin: return add_input(L, NVJITLINK_INPUT_CUBIN, data, size, label);
        case BlobKind::HostObject: return add_object(L, data, size, label);
        case BlobKind::Archive: return add_input(L, NVJITLINK_INPUT_LIBRARY, data, size, label);
        case BlobKind::LtoIr: return refuse_ltoir(L, label);
        case BlobKind::Unknown: break;
      }
      L.error_log += "unrecognized input type: " + label + "\n";
      return NVJITLINK_ERROR_UNRECOGNIZED_INPUT;
    case NVJITLINK_INPUT_PTX:
      if (kind != BlobKind::Ptx) return wrong_type("NVJITLINK_INPUT_PTX");
      return add_ptx(L, std::string(static_cast<const char*>(data), size), label);
    case NVJITLINK_INPUT_CUBIN:
      // VirtualGPU's cubins -- this library's output, NVRTC's shim's -- are
      // PTX, and link as PTX.
      if (kind == BlobKind::Ptx)
        return add_ptx(L, std::string(static_cast<const char*>(data), size), label);
      if (kind != BlobKind::Cubin) return wrong_type("NVJITLINK_INPUT_CUBIN");
      // A linked cubin (nvcc -cubin) NVIDIA's accepts and passes over -- it
      // adds nothing to the link, as an RTX 3060 run showed -- and so does
      // this one.
      if (vgpu::cuda::cubin_linked(data, size)) {
        if (L.verbose)
          L.info_log += "info    : '" + label + "' is a linked cubin, which adds nothing to a link\n";
        return NVJITLINK_SUCCESS;
      }
      return add_sass(L, data, size, label);
    case NVJITLINK_INPUT_FATBIN:
      if (kind != BlobKind::Fatbin) return wrong_type("NVJITLINK_INPUT_FATBIN");
      return add_fatbin(L, data, size, label);
    case NVJITLINK_INPUT_OBJECT:
      if (kind != BlobKind::HostObject) return wrong_type("NVJITLINK_INPUT_OBJECT");
      return add_object(L, data, size, label);
    case NVJITLINK_INPUT_LIBRARY: {
      if (kind != BlobKind::Archive) return wrong_type("NVJITLINK_INPUT_LIBRARY");
      std::vector<std::pair<std::string, std::string>> members;
      try {
        members = vgpu::cuda::archive_members(data, size);
      } catch (const vgpu::Error& e) {
        L.error_log += "ERROR: library '" + label + "' cannot be read: " + e.message() + "\n";
        return NVJITLINK_ERROR_INVALID_INPUT;
      }
      for (const auto& [name, bytes] : members) {
        if (vgpu::cuda::classify_blob(bytes.data(), bytes.size()) != BlobKind::HostObject) continue;
        const nvJitLinkResult r = add_object(L, bytes.data(), bytes.size(), label + "(" + name + ")");
        if (r != NVJITLINK_SUCCESS) return r;
      }
      return NVJITLINK_SUCCESS;
    }
    case NVJITLINK_INPUT_LTOIR:
      return refuse_ltoir(L, label);
    case NVJITLINK_INPUT_INDEX:
      if (toolkit_linker()) {   // an index of LTO-IR libraries: the toolkit's linker reads it
        L.delegate = true;
        return NVJITLINK_SUCCESS;
      }
      L.error_log += "ERROR: index file '" + label +
                     "' names LTO-IR libraries, which only the toolkit's libnvJitLink links, and it was not found\n";
      return NVJITLINK_ERROR_INVALID_INPUT;
    default:
      L.error_log += "unsupported input type: " + label + "\n";
      return NVJITLINK_ERROR_INCORRECT_INPUT_TYPE;
  }
}

nvJitLinkResult copy_size(const std::string& s, size_t* size) {
  if (!size) return NVJITLINK_ERROR_NULL_INPUT;
  *size = s.size() + 1;
  return NVJITLINK_SUCCESS;
}
nvJitLinkResult copy_out(const std::string& s, void* out) {
  if (!out) return NVJITLINK_ERROR_NULL_INPUT;
  std::memcpy(out, s.c_str(), s.size() + 1);
  return NVJITLINK_SUCCESS;
}
// A log's size counts its terminator, and an empty log is 0, not 1.
nvJitLinkResult log_size(const std::string& log, size_t* size) {
  if (!size) return NVJITLINK_ERROR_NULL_INPUT;
  *size = log.empty() ? 0 : log.size() + 1;
  return NVJITLINK_SUCCESS;
}

}  // namespace

VGPU_EXPORT nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle* handle, uint32_t numOptions,
                                            const char** options) {
  if (!handle || !options) return NVJITLINK_ERROR_NULL_INPUT;
  for (uint32_t i = 0; i < numOptions; ++i)
    if (!options[i]) return NVJITLINK_ERROR_NULL_INPUT;
  auto* L = new Link;
  for (uint32_t i = 0; i < numOptions; ++i) L->option_strings.push_back(options[i]);
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.insert(L);
  }
  *handle = reinterpret_cast<nvJitLinkHandle>(L);
  const nvJitLinkResult r = parse_options(*L, numOptions, options);
  if (r != NVJITLINK_SUCCESS) L->failed_create = true;
  return r;
}

VGPU_EXPORT nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle* handle) {
  if (!handle || !*handle) return NVJITLINK_ERROR_NULL_INPUT;
  Link* L = get(*handle);
  if (!L) return NVJITLINK_ERROR_INVALID_INPUT;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.erase(L);
  }
  delete L;
  *handle = nullptr;
  return NVJITLINK_SUCCESS;
}

VGPU_EXPORT nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType,
                                             const void* data, size_t size, const char* name) {
  if (!handle) return NVJITLINK_ERROR_NULL_INPUT;
  Link* L = get(handle);
  if (!L) return NVJITLINK_ERROR_INVALID_INPUT;
  if (L->failed_create || L->completed) return NVJITLINK_ERROR_INTERNAL;
  if (!data) return NVJITLINK_ERROR_NULL_INPUT;
  if (size == 0) return NVJITLINK_ERROR_INVALID_INPUT;
  const std::string label =
      name ? std::string(name) : "(unnamed input " + std::to_string(++L->unnamed) + ")";
  L->raw.push_back({static_cast<int>(inputType), std::string(static_cast<const char*>(data), size), label});
  return add_input(*L, inputType, data, size, label);
}

VGPU_EXPORT nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle, nvJitLinkInputType inputType,
                                             const char* fileName) {
  if (!handle || !fileName) return NVJITLINK_ERROR_NULL_INPUT;
  Link* L = get(handle);
  if (!L) return NVJITLINK_ERROR_INVALID_INPUT;
  if (L->failed_create || L->completed) return NVJITLINK_ERROR_INTERNAL;
  std::ifstream f(fileName, std::ios::binary);
  if (!f) {
    L->error_log += std::string("no such file: ") + fileName + "\n";
    return NVJITLINK_ERROR_INVALID_INPUT;
  }
  const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (bytes.empty()) return NVJITLINK_ERROR_INVALID_INPUT;
  L->raw.push_back({static_cast<int>(inputType), bytes, fileName});
  return add_input(*L, inputType, bytes.data(), bytes.size(), fileName);
}

// The link, done by the toolkit's libnvJitLink from the inputs as given.
static nvJitLinkResult complete_with_toolkit(Link& L) {
  const RealJitLink& r = *toolkit_linker();
  std::vector<const char*> opts;
  for (const std::string& o : L.option_strings) opts.push_back(o.c_str());
  nvJitLinkHandle h = nullptr;
  const auto take_logs = [&] {
    size_t n = 0;
    if (r.error_log_size(h, &n) == NVJITLINK_SUCCESS && n > 1) {
      std::string text(n, '\0');
      if (r.error_log(h, &text[0]) == NVJITLINK_SUCCESS) L.error_log += text.substr(0, n - 1);
    }
    n = 0;
    if (r.info_log_size(h, &n) == NVJITLINK_SUCCESS && n > 1) {
      std::string text(n, '\0');
      if (r.info_log(h, &text[0]) == NVJITLINK_SUCCESS) L.info_log += text.substr(0, n - 1);
    }
  };
  nvJitLinkResult rc = r.create(&h, static_cast<uint32_t>(opts.size()), opts.data());
  if (rc != NVJITLINK_SUCCESS) {
    if (h) {
      take_logs();
      r.destroy(&h);
    }
    return rc;
  }
  for (const Link::Raw& in : L.raw) {
    // A "cubin" this library handed out earlier is PTX text.
    int type = in.type;
    if (type == NVJITLINK_INPUT_CUBIN &&
        vgpu::cuda::classify_blob(in.data.data(), in.data.size()) == vgpu::cuda::BlobKind::Ptx)
      type = NVJITLINK_INPUT_PTX;
    rc = r.add_data(h, type, in.data.data(), in.data.size(), in.label.c_str());
    if (rc != NVJITLINK_SUCCESS) {
      take_logs();
      r.destroy(&h);
      return rc;
    }
  }
  rc = r.complete(h);
  take_logs();
  if (rc == NVJITLINK_SUCCESS) {
    size_t n = 0;
    L.d_cubin_rc = r.cubin_size(h, &n);
    if (L.d_cubin_rc == NVJITLINK_SUCCESS) {
      L.d_cubin.assign(n, '\0');
      if (n) L.d_cubin_rc = r.cubin(h, &L.d_cubin[0]);
    }
    n = 0;
    L.d_ptx_rc = r.ptx_size(h, &n);
    if (L.d_ptx_rc == NVJITLINK_SUCCESS) {
      L.d_ptx.assign(n, '\0');
      if (n) L.d_ptx_rc = r.ptx(h, &L.d_ptx[0]);
    }
    L.linked_ok = true;
  }
  r.destroy(&h);
  return rc;
}

VGPU_EXPORT nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) {
  if (!handle) return NVJITLINK_ERROR_NULL_INPUT;
  Link* L = get(handle);
  if (!L) return NVJITLINK_ERROR_INVALID_INPUT;
  if (L->failed_create || L->completed) return NVJITLINK_ERROR_INTERNAL;
  L->completed = true;
  if (L->delegate) return complete_with_toolkit(*L);
  if (L->ptx_output && !L->inputs.empty()) {
    // -lto -ptx is LTO-IR in, PTX out; every input here is PTX already.
    L->error_log += "ERROR: -ptx requires that all inputs have LTOIR\n";
    return NVJITLINK_ERROR_INCORRECT_INPUT_TYPE;
  }
  if (has_sass_only(*L)) {
    // Machine code: every piece has SASS for -arch (add_sass saw to that).
    std::vector<vgpu::cuda::SassLinkInput> in;
    for (const Link::Piece& p : L->pieces) in.push_back(L->sass[static_cast<size_t>(p.sass)]);
    vgpu::cuda::SassLinkResult r = vgpu::cuda::link_sass(in, L->arch_number, L->sass_options);
    L->error_log += r.errors;
    if (!r.ok) return NVJITLINK_ERROR_INTERNAL;
    L->linked.assign(r.cubin.begin(), r.cubin.end());
    L->linked_sass = true;
    L->linked_ok = true;
    if (L->verbose)
      L->info_log += "info    : linked " + std::to_string(in.size()) + " relocatable SASS module" +
                     (in.size() == 1 ? "" : "s") + " for " + L->arch + "\n";
    return NVJITLINK_SUCCESS;
  }
  vgpu::cuda::PtxLinkResult r = vgpu::cuda::link_ptx(L->inputs, L->arch);
  L->error_log += r.errors;
  if (!r.ok) return NVJITLINK_ERROR_INTERNAL;
  L->linked = std::move(r.ptx);
  L->linked_ok = true;
  if (L->verbose)
    L->info_log += "info    : linked " + std::to_string(L->inputs.size()) + " PTX module" +
                   (L->inputs.size() == 1 ? "" : "s") + " for " + L->arch +
                   " (VirtualGPU: the linked image is PTX, which its driver loads)\n";
  return NVJITLINK_SUCCESS;
}

VGPU_EXPORT nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t* size) {
  Link* L = get(handle);
  if (!L) return handle ? NVJITLINK_ERROR_INVALID_INPUT : NVJITLINK_ERROR_NULL_INPUT;
  if (!size) return NVJITLINK_ERROR_NULL_INPUT;
  if (!L->linked_ok) return NVJITLINK_ERROR_INTERNAL;
  if (L->delegate) {
    if (L->d_cubin_rc != NVJITLINK_SUCCESS) return L->d_cubin_rc;
    *size = L->d_cubin.size();
    return NVJITLINK_SUCCESS;
  }
  if (L->linked_sass) {   // a real cubin: its own size, no terminator
    *size = L->linked.size();
    return NVJITLINK_SUCCESS;
  }
  return copy_size(L->linked, size);
}
VGPU_EXPORT nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void* cubin) {
  Link* L = get(handle);
  if (!L) return handle ? NVJITLINK_ERROR_INVALID_INPUT : NVJITLINK_ERROR_NULL_INPUT;
  if (!cubin) return NVJITLINK_ERROR_NULL_INPUT;
  if (!L->linked_ok) return NVJITLINK_ERROR_INTERNAL;
  if (L->delegate) {
    if (L->d_cubin_rc != NVJITLINK_SUCCESS) return L->d_cubin_rc;
    std::memcpy(cubin, L->d_cubin.data(), L->d_cubin.size());
    return NVJITLINK_SUCCESS;
  }
  if (L->linked_sass) {
    std::memcpy(cubin, L->linked.data(), L->linked.size());
    return NVJITLINK_SUCCESS;
  }
  return copy_out(L->linked, cubin);
}
VGPU_EXPORT nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t* size) {
  Link* L = get(handle);
  if (!L) return handle ? NVJITLINK_ERROR_INVALID_INPUT : NVJITLINK_ERROR_NULL_INPUT;
  if (!size) return NVJITLINK_ERROR_NULL_INPUT;
  if (!L->linked_ok) return NVJITLINK_ERROR_INTERNAL;
  if (L->delegate) {
    if (L->d_ptx_rc != NVJITLINK_SUCCESS) return L->d_ptx_rc;
    *size = L->d_ptx.size();
    return NVJITLINK_SUCCESS;
  }
  // A SASS link has no PTX to give: NVIDIA's 13.0 returned
  // NVJITLINK_ERROR_INVALID_INPUT for it.
  if (L->linked_sass) return NVJITLINK_ERROR_INVALID_INPUT;
  return copy_size(L->linked, size);
}
VGPU_EXPORT nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char* ptx) {
  Link* L = get(handle);
  if (!L) return handle ? NVJITLINK_ERROR_INVALID_INPUT : NVJITLINK_ERROR_NULL_INPUT;
  if (!ptx) return NVJITLINK_ERROR_NULL_INPUT;
  if (!L->linked_ok) return NVJITLINK_ERROR_INTERNAL;
  if (L->delegate) {
    if (L->d_ptx_rc != NVJITLINK_SUCCESS) return L->d_ptx_rc;
    std::memcpy(ptx, L->d_ptx.data(), L->d_ptx.size());
    return NVJITLINK_SUCCESS;
  }
  if (L->linked_sass) return NVJITLINK_ERROR_INVALID_INPUT;
  return copy_out(L->linked, ptx);
}

VGPU_EXPORT nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t* size) {
  Link* L = get(handle);
  if (!L) return handle ? NVJITLINK_ERROR_INVALID_INPUT : NVJITLINK_ERROR_NULL_INPUT;
  return log_size(L->error_log, size);
}
VGPU_EXPORT nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char* log) {
  Link* L = get(handle);
  if (!L) return handle ? NVJITLINK_ERROR_INVALID_INPUT : NVJITLINK_ERROR_NULL_INPUT;
  return copy_out(L->error_log, log);
}
VGPU_EXPORT nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t* size) {
  Link* L = get(handle);
  if (!L) return handle ? NVJITLINK_ERROR_INVALID_INPUT : NVJITLINK_ERROR_NULL_INPUT;
  return log_size(L->info_log, size);
}
VGPU_EXPORT nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char* log) {
  Link* L = get(handle);
  if (!L) return handle ? NVJITLINK_ERROR_INVALID_INPUT : NVJITLINK_ERROR_NULL_INPUT;
  return copy_out(L->info_log, log);
}

// For tests: whether LTO-IR and PTX-beside-SASS links go to the toolkit's own linker.
extern "C" __attribute__((visibility("default"))) int vgpu_nvjitlink_has_toolkit_linker(void) {
  return toolkit_linker() != nullptr;
}

VGPU_EXPORT nvJitLinkResult nvJitLinkVersion(unsigned int* major, unsigned int* minor) {
  if (!major || !minor) return NVJITLINK_ERROR_NULL_INPUT;
  *major = VGPU_NVJITLINK_VERSION / 1000;
  *minor = VGPU_NVJITLINK_VERSION % 1000 / 10;
  return NVJITLINK_SUCCESS;
}

// The entry points a compiled program actually imports. NVIDIA's header makes
// each call above an inline wrapper around a name carrying the API's version
// -- __nvJitLinkCreate_12_0 under every CUDA 12 toolkit, _13_0 under CUDA 13
// -- so both families are defined, and a program built against either header
// finds its names here.
#define VGPU_NVJITLINK_VERSIONED(ver)                                                              \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkCreate_##ver(nvJitLinkHandle* h, uint32_t n,              \
                                                       const char* const* o) {                     \
    return nvJitLinkCreate(h, n, const_cast<const char**>(o));                                     \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkDestroy_##ver(nvJitLinkHandle* h) {                       \
    return nvJitLinkDestroy(h);                                                                    \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkAddData_##ver(nvJitLinkHandle h, nvJitLinkInputType t,    \
                                                        const void* d, size_t s, const char* n) {  \
    return nvJitLinkAddData(h, t, d, s, n);                                                        \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkAddFile_##ver(nvJitLinkHandle h, nvJitLinkInputType t,    \
                                                        const char* f) {                           \
    return nvJitLinkAddFile(h, t, f);                                                              \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkComplete_##ver(nvJitLinkHandle h) {                       \
    return nvJitLinkComplete(h);                                                                   \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkGetLinkedCubinSize_##ver(nvJitLinkHandle h, size_t* s) {  \
    return nvJitLinkGetLinkedCubinSize(h, s);                                                      \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkGetLinkedCubin_##ver(nvJitLinkHandle h, void* c) {        \
    return nvJitLinkGetLinkedCubin(h, c);                                                          \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkGetLinkedPtxSize_##ver(nvJitLinkHandle h, size_t* s) {    \
    return nvJitLinkGetLinkedPtxSize(h, s);                                                        \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkGetLinkedPtx_##ver(nvJitLinkHandle h, char* p) {          \
    return nvJitLinkGetLinkedPtx(h, p);                                                            \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkGetErrorLogSize_##ver(nvJitLinkHandle h, size_t* s) {     \
    return nvJitLinkGetErrorLogSize(h, s);                                                         \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkGetErrorLog_##ver(nvJitLinkHandle h, char* l) {           \
    return nvJitLinkGetErrorLog(h, l);                                                             \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkGetInfoLogSize_##ver(nvJitLinkHandle h, size_t* s) {      \
    return nvJitLinkGetInfoLogSize(h, s);                                                          \
  }                                                                                                \
  VGPU_EXPORT nvJitLinkResult __nvJitLinkGetInfoLog_##ver(nvJitLinkHandle h, char* l) {            \
    return nvJitLinkGetInfoLog(h, l);                                                              \
  }

VGPU_NVJITLINK_VERSIONED(12_0)
VGPU_NVJITLINK_VERSIONED(13_0)
