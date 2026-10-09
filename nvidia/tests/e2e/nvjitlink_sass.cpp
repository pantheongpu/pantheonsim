// nvJitLink linking machine code: relocatable SASS (nvcc -rdc / -dc, SASS
// only) linked into a cubin, loaded with cuModuleLoadData and run. The device
// code is sass_link_main.cu and sass_link_lib.cu, which between them make
// every kind of reference only a link resolves (see sass_link_main.cu); the
// answers are checked exactly.
//
//   nvjitlink_sass <arch> <main.cubin> <lib.cubin> [<other-arch.cubin> [<lib.o> <lib.a>]]
//
// run_jit_link_sass.sh builds the inputs. Also checked: an undefined reference
// fails the link and names the symbol; a second definition is reported and
// the link still succeeds; a cubin for an architecture -arch cannot run is
// refused when added; a SASS link has no PTX to hand out; a host object's and
// a static library's relocatable SASS link like a cubin. And what a link
// keeps: a function nothing reaches is left out unless -g asks for it,
// -kernels-used drops the kernels it does not name, and the frame table and
// the other debug sections of the modules come along.
//
// Every check passes against NVIDIA's libnvJitLink 13.0 and driver on an RTX
// 3060 (sm_86), and with VirtualGPU's libnvJitLink in its place there, whose
// cubin NVIDIA's driver then runs.
#include <cuda.h>
#ifdef VGPU_OWN_NVJITLINK_H   // a toolkit without nvJitLink (run_jit_link_sass.sh)
#include "../../include/vgpu_nvjitlink.h"
#else
#include <nvJitLink.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
static void is(long long got, long long want, const char* what) {
  check(got == want, what);
  if (got != want) std::printf("     got %lld, want %lld\n", got, want);
}

static std::vector<char> slurp(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static std::string error_log(nvJitLinkHandle h) {
  size_t n = 0;
  if (nvJitLinkGetErrorLogSize(h, &n) != NVJITLINK_SUCCESS || n == 0) return {};
  std::string s(n, '\0');
  nvJitLinkGetErrorLog(h, &s[0]);
  return s;
}

struct Input {
  nvJitLinkInputType type;
  const char* path;
};

// Links the inputs; returns the result of nvJitLinkComplete (or of the first
// add that failed), the cubin in *out and the error log in *log.
static int link(const std::string& arch, const std::vector<Input>& inputs, std::vector<char>* out, std::string* log,
                const std::vector<std::string>& extra = {}) {
  const std::string opt = "-arch=" + arch;
  std::vector<const char*> opts = {opt.c_str()};
  for (const std::string& e : extra) opts.push_back(e.c_str());
  nvJitLinkHandle h = nullptr;
  int r = nvJitLinkCreate(&h, static_cast<uint32_t>(opts.size()), opts.data());
  if (r != NVJITLINK_SUCCESS) return r;
  for (const Input& in : inputs) {
    const std::vector<char> bytes = slurp(in.path);
    r = nvJitLinkAddData(h, in.type, bytes.data(), bytes.size(), in.path);
    if (r != NVJITLINK_SUCCESS) {
      *log = error_log(h);
      nvJitLinkDestroy(&h);
      return r;
    }
  }
  r = nvJitLinkComplete(h);
  *log = error_log(h);
  if (r == NVJITLINK_SUCCESS && out) {
    size_t n = 0;
    nvJitLinkGetLinkedCubinSize(h, &n);
    out->assign(n, 0);
    nvJitLinkGetLinkedCubin(h, out->data());
    // A SASS link has no PTX to give.
    size_t pn = 0;
    is(nvJitLinkGetLinkedPtxSize(h, &pn), NVJITLINK_ERROR_INVALID_INPUT, "a SASS link has no linked PTX");
  }
  nvJitLinkDestroy(&h);
  return r;
}

// The names of a cubin's functions and of its sections (ELF64: the section table, the symbol table).
template <class T>
static T at(const std::vector<char>& b, size_t off) {
  T v{};
  if (off + sizeof v <= b.size()) std::memcpy(&v, b.data() + off, sizeof v);
  return v;
}
static void names_in(const std::vector<char>& elf, std::set<std::string>* functions, std::set<std::string>* sections,
                     std::map<std::string, size_t>* section_size = nullptr) {
  const size_t shoff = at<uint64_t>(elf, 0x28);
  const unsigned shentsize = at<uint16_t>(elf, 0x3a), shnum = at<uint16_t>(elf, 0x3c), shstrndx = at<uint16_t>(elf, 0x3e);
  const size_t str = at<uint64_t>(elf, shoff + size_t{shstrndx} * shentsize + 24);
  auto cstr = [&](size_t off) { return std::string(elf.data() + std::min(off, elf.size()), strnlen(elf.data() + std::min(off, elf.size()), elf.size() - std::min(off, elf.size()))); };
  for (unsigned i = 0; i < shnum; ++i) {
    const size_t h = shoff + size_t{i} * shentsize;
    const std::string name = cstr(str + at<uint32_t>(elf, h));
    if (sections) sections->insert(name);
    if (section_size) (*section_size)[name] = at<uint64_t>(elf, h + 32);
    if (at<uint32_t>(elf, h + 4) != 2 /* SHT_SYMTAB */ || !functions) continue;
    const size_t symoff = at<uint64_t>(elf, h + 24), symsize = at<uint64_t>(elf, h + 32);
    const unsigned link = at<uint32_t>(elf, h + 40);
    const size_t strtab = at<uint64_t>(elf, shoff + size_t{link} * shentsize + 24);
    for (size_t o = symoff; o + 24 <= symoff + symsize; o += 24)
      if ((at<uint8_t>(elf, o + 4) & 0xf) == 2 /* STT_FUNC */ && at<uint16_t>(elf, o + 6) != 0)
        functions->insert(cstr(strtab + at<uint32_t>(elf, o)));
  }
}

// Loads a linked cubin and runs both modules' kernels, checking every answer.
static void run(const std::vector<char>& cubin, const char* what) {
  std::printf("-- %s\n", what);
  check(cubin.size() > 64 && cubin[0] == 0x7f && cubin[1] == 'E', "the linked image is a cubin");
  CUmodule m = nullptr;
  if (cuModuleLoadData(&m, cubin.data()) != CUDA_SUCCESS) {
    check(false, "cuModuleLoadData takes it");
    return;
  }
  CUfunction run_all = nullptr, lib_kernel = nullptr;
  check(cuModuleGetFunction(&run_all, m, "_Z7run_allPii") == CUDA_SUCCESS, "the main module's kernel is there");
  check(cuModuleGetFunction(&lib_kernel, m, "_Z10lib_kernelPi") == CUDA_SUCCESS, "and the library's");
  if (!run_all || !lib_kernel) return;
  CUdeviceptr out;
  cuMemAlloc(&out, 64 * sizeof(int));
  cuMemsetD32(out, 0, 64);
  int sel = 5;
  void* args[] = {&out, &sel};
  check(cuLaunchKernel(run_all, 1, 1, 1, 64, 1, 1, 64 * sizeof(int), nullptr, args, nullptr) == CUDA_SUCCESS,
        "run_all launches, with 256 bytes of dynamic shared memory");
  check(cuCtxSynchronize() == CUDA_SUCCESS, "and finishes");
  int got[10] = {};
  cuMemcpyDtoH(got, out, sizeof got);
  is(got[0], 5 + 17 + 3, "a call into the library, which reads both modules' variables");
  is(got[1], 17, "the library's variable");
  is(got[2], 4 + 6, "the library's constants, direct and indexed");
  is(got[3], 30 + 60, "the main module's constants, direct and indexed");
  is(got[4], 1001 + 2001, "each module's file-scope function of the same name is its own");
  is(got[5], 42 - 4 - 1, "calls through pointers to functions in both modules");
  is(got[6], 11 + 11, "a template both modules instantiate");
  is(got[7], 300 + 6, "pointers initialised to other variables' addresses");
  is(got[8], 11, "malloc in linked code");
  is(got[9], 64 * 5, "no two shared arrays overlap (kernel, functions in both modules, dynamic)");
  CUdeviceptr counter;
  size_t bytes = 0;
  int count = 0;
  check(cuModuleGetGlobal(&counter, &bytes, m, "main_counter") == CUDA_SUCCESS && bytes == 4,
        "cuModuleGetGlobal finds a variable");
  cuMemcpyDtoH(&count, counter, 4);
  is(count, 64, "which every thread added to");
  check(cuModuleGetGlobal(&counter, &bytes, m, "lib_counter") == CUDA_SUCCESS, "and the library's");
  cuMemcpyDtoH(&count, counter, 4);
  is(count, 64, "which every thread added to");
  // New constants, through the module, read by the library's kernel.
  CUdeviceptr c;
  check(cuModuleGetGlobal(&c, &bytes, m, "main_const") == CUDA_SUCCESS && bytes == 32, "a __constant__ array");
  int consts[8];
  for (int i = 0; i < 8; ++i) consts[i] = 1000 + i;
  cuMemcpyHtoD(c, consts, sizeof consts);
  void* args2[] = {&out};
  check(cuLaunchKernel(lib_kernel, 1, 1, 1, 8, 1, 1, 0, nullptr, args2, nullptr) == CUDA_SUCCESS &&
            cuCtxSynchronize() == CUDA_SUCCESS,
        "the library's kernel runs");
  cuMemcpyDtoH(got, out, 8 * sizeof(int));
  bool right = true;
  for (int t = 0; t < 8; ++t) right = right && got[t] == t + 17 + 3 + 1000 + t;
  check(right, "and reads the other module's constants as the host set them");
  cuMemFree(out);
  cuModuleUnload(m);
}

// What a link keeps and leaves out.
static void dead_code_and_debug_info(const std::string& arch, const char* main_cubin, const char* lib_cubin) {
  const std::vector<Input> in = {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_CUBIN, lib_cubin}};
  std::vector<char> cubin;
  std::string log;
  std::set<std::string> fns, secs;
  std::map<std::string, size_t> sizes;
  is(link(arch, in, &cubin, &log), NVJITLINK_SUCCESS, "the two modules link (for what is kept)");
  names_in(cubin, &fns, &secs, &sizes);
  check(fns.count("_Z7run_allPii") && fns.count("_Z10lib_kernelPi") && fns.count("_Z7lib_addi"),
        "the kernels and the function they call stay");
  check(!fns.count("_Z10lib_unusedi") && !fns.count("_Z15lib_unused_leafi"),
        "a function nothing calls goes, and so does the one only it calls");
  check(fns.count("_Z10main_twicei") && fns.count("_Z10lib_negatei"),
        "while functions whose addresses a kept variable holds stay");
  check(secs.count(".debug_frame"), "the frame table of the modules is carried");
  std::map<std::string, size_t> main_sizes, lib_sizes;
  names_in(slurp(main_cubin), nullptr, nullptr, &main_sizes);
  names_in(slurp(lib_cubin), nullptr, nullptr, &lib_sizes);
  // (From sm_100 NVIDIA's link also takes the dropped functions' entries out of the table.)
  if (std::atoi(arch.c_str() + 3) < 100)
    check(sizes[".debug_frame"] == main_sizes[".debug_frame"] + lib_sizes[".debug_frame"],
          "and is the two modules' tables one after the other, the dropped functions' entries included");

  is(link(arch, in, &cubin, &log, {"-g"}), NVJITLINK_SUCCESS, "-g links");
  fns.clear();
  names_in(cubin, &fns, nullptr);
  check(fns.count("_Z10lib_unusedi") && fns.count("_Z15lib_unused_leafi"), "and keeps every function, called or not");
  run(cubin, "linked with -g");

  is(link(arch, in, &cubin, &log, {"-kernels-used=run_all"}), NVJITLINK_SUCCESS, "-kernels-used links");
  fns.clear();
  names_in(cubin, &fns, nullptr);
  check(fns.count("_Z7run_allPii") && !fns.count("_Z10lib_kernelPi"),
        "-kernels-used=run_all keeps that kernel (a substring of its mangled name) and drops the other");
  check(fns.count("_Z7lib_addi"), "and what the kernel still reaches");
  is(link(arch, in, &cubin, &log, {"-kernels-used=lib_k*l", "-kernels-used=nosuch"}), NVJITLINK_SUCCESS, "two patterns link");
  fns.clear();
  names_in(cubin, &fns, nullptr);
  check(fns.count("_Z10lib_kernelPi") && !fns.count("_Z7run_allPii"),
        "a * in a pattern stands for any run of characters");
  is(link(arch, in, &cubin, &log, {"-kernels-used=nosuch"}), NVJITLINK_SUCCESS, "a pattern nothing matches links");
  fns.clear();
  names_in(cubin, &fns, nullptr);
  check(!fns.count("_Z7run_allPii") && !fns.count("_Z10lib_kernelPi") && !fns.count("_Z7lib_addi"),
        "and leaves no kernel, nor any function only they reach");
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s <arch> <main.cubin> <lib.cubin> [other-arch.cubin [lib.o lib.a]]\n", argv[0]);
    return 2;
  }
  const std::string arch = argv[1];
  const char* main_cubin = argv[2];
  const char* lib_cubin = argv[3];
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS || cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    std::printf("FAIL: no CUDA device\n");
    return 1;
  }
  std::vector<char> cubin;
  std::string log;

  is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_CUBIN, lib_cubin}}, &cubin, &log),
     NVJITLINK_SUCCESS, "two relocatable cubins link");
  if (!log.empty()) std::printf("     log: %s", log.c_str());
  run(cubin, "the linked cubin");

  is(link(arch, {{NVJITLINK_INPUT_CUBIN, lib_cubin}, {NVJITLINK_INPUT_CUBIN, main_cubin}}, &cubin, &log),
     NVJITLINK_SUCCESS, "the other order links");
  run(cubin, "linked library first");

  dead_code_and_debug_info(arch, main_cubin, lib_cubin);

  // A reference nothing defines.
  is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}}, nullptr, &log), 6 /* NVJITLINK_ERROR_INTERNAL */,
     "an undefined reference fails the link");
  check(log.find("Undefined reference to 'lib_var'") != std::string::npos, "and names the symbol");

  // A second definition does not fail the link. (NVIDIA's 13.0 prints
  // "Multiple definition of ..." to stderr, and its driver then refuses the
  // image; VirtualGPU's puts the line in the error log and links the first.)
  is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_CUBIN, lib_cubin}, {NVJITLINK_INPUT_CUBIN, lib_cubin}},
          &cubin, &log),
     NVJITLINK_SUCCESS, "a module linked twice still links");

  if (argc > 4) {
    is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_CUBIN, argv[4]}}, nullptr, &log),
       NVJITLINK_ERROR_INVALID_INPUT, "a cubin for an architecture -arch cannot run is refused");
  }
  if (argc > 6) {
    is(link(arch, {{NVJITLINK_INPUT_CUBIN, main_cubin}, {NVJITLINK_INPUT_OBJECT, argv[5]}}, &cubin, &log),
       NVJITLINK_SUCCESS, "a host object's relocatable SASS links");
    run(cubin, "linked with a host object");
    is(link(arch, {{NVJITLINK_INPUT_LIBRARY, argv[6]}, {NVJITLINK_INPUT_CUBIN, main_cubin}}, &cubin, &log),
       NVJITLINK_SUCCESS, "and a static library's");
    run(cubin, "linked with a static library");
  }
  std::printf(failures ? "FAIL: %d checks\n" : "PASS\n", failures);
  return failures != 0;
}
