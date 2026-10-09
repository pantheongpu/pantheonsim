// nvFatbin as a program packaging its JIT output uses it: PTX for several
// architectures, a cubin from nvJitLink, LTO-IR and a host object's
// relocatable PTX in one fatbin, which cuModuleLoadData and
// cuModuleLoadFatBinary then load -- choosing the image for the device, as the
// driver does -- and nvJitLink takes as an input. Plus the result codes for
// each misuse.
//
//   nvfatbin_paths [object.o]
//
// The object, built by run_jit_link.sh from jitlink_lib.cu with nvcc -dc,
// feeds nvFatbinAddReloc; without it that check is skipped.
//
// Built against VirtualGPU's declarations (CUDA 12.0 has no nvFatbin.h),
// which follow NVIDIA's ABI: every check passes against NVIDIA's
// libnvfatbin 13.0 on an RTX 3060, and the fatbins it writes load there.
//
// Beyond the loading: the options' rules and what they put in an entry's flag
// word, the comments PTX loses on the way in, which entries are compressed and
// with what (zstd by default for PTX, LZ4 for -compress-mode=speed, cubins only
// when asked), the result strings, and an index file's refusal -- all as
// libnvfatbin 13.0 answers them.
#include <cuda.h>
#include <dlfcn.h>
#ifdef VGPU_OWN_NVJITLINK_H   // a toolkit without nvJitLink (run_jit_link.sh)
#include "../../include/vgpu_nvjitlink.h"
#else
#include <nvJitLink.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../../include/vgpu_nvfatbin.h"

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
static void is(int got, int want, const char* what) {
  check(got == want, what);
  if (got != want) std::printf("     returned %d\n", got);
}
#define IS(call, want) is((int)(call), (int)(want), #call " -> " #want)

#include "jitlink_ptx.inc"

// Two builds of one kernel, each writing which one ran: for sm_80, and for
// sm_90, which neither an A100 nor an RTX 3060 can run.
static std::string which(int arch, int value) {
  return ".version 7.8\n.target sm_" + std::to_string(arch) +
         "\n.address_size 64\n"
         ".visible .entry which(.param .u64 out)\n{\n"
         "\t.reg .b32 %r<2>;\n\t.reg .b64 %rd<3>;\n"
         "\tld.param.u64 %rd1, [out];\n\tcvta.to.global.u64 %rd2, %rd1;\n"
         "\tmov.u32 %r1, " + std::to_string(value) + ";\n\tst.global.u32 [%rd2], %r1;\n\tret;\n}\n";
}

static std::vector<char> image_of(nvFatbinHandle h) {
  size_t n = 0;
  if (nvFatbinSize(h, &n) != NVFATBIN_SUCCESS || n == 0) return {};
  std::vector<char> b(n);
  if (nvFatbinGet(h, b.data()) != NVFATBIN_SUCCESS) return {};
  return b;
}

// Runs `kernel` with one pointer argument over `threads` threads; the ints it
// wrote, or nothing when the image does not load or run.
static std::vector<int> run(CUmodule mod, const char* kernel, int threads) {
  CUfunction fn;
  if (cuModuleGetFunction(&fn, mod, kernel) != CUDA_SUCCESS) return {};
  CUdeviceptr d = 0;
  if (cuMemAlloc(&d, threads * sizeof(int)) != CUDA_SUCCESS) return {};
  void* args[] = {&d};
  std::vector<int> out(threads);
  const bool ok = cuLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS &&
                  cuCtxSynchronize() == CUDA_SUCCESS &&
                  cuMemcpyDtoH(out.data(), d, threads * sizeof(int)) == CUDA_SUCCESS;
  cuMemFree(d);
  return ok ? out : std::vector<int>{};
}

static bool computes_k(const std::vector<int>& out) {
  if (out.size() != 32) return false;
  for (int t = 0; t < 32; ++t)
    if (out[t] != expected(t)) return false;
  return true;
}

static std::vector<char> linked_cubin(std::vector<std::pair<nvJitLinkInputType, std::vector<char>>> in,
                                      const char* arch) {
  const char* opts[] = {arch};
  nvJitLinkHandle h;
  if (nvJitLinkCreate(&h, 1, opts) != NVJITLINK_SUCCESS) return {};
  bool ok = true;
  for (auto& [type, bytes] : in)
    ok = ok && nvJitLinkAddData(h, type, bytes.data(), bytes.size(), "input") == NVJITLINK_SUCCESS;
  size_t n = 0;
  ok = ok && nvJitLinkComplete(h) == NVJITLINK_SUCCESS && nvJitLinkGetLinkedCubinSize(h, &n) == NVJITLINK_SUCCESS;
  std::vector<char> cubin(ok ? n : 0);
  if (ok) nvJitLinkGetLinkedCubin(h, cubin.data());
  nvJitLinkDestroy(&h);
  return cubin;
}

// PTX as bytes, its terminator included: NVIDIA's nvJitLink reads PTX as a
// C string whatever size it is given, so without one it reads on into
// whatever follows (and fails now and then, as an RTX 3060 run showed).
static std::vector<char> bytes_of(const char* s) { return std::vector<char>(s, s + std::strlen(s) + 1); }

// The entries of a fatbin container: kind, the header's flag word, whether the payload was stored
// compressed, and the payload as stored.
struct Entry {
  uint16_t kind;
  uint64_t flags, usize;
  uint32_t psize;
  std::string ident, payload;
};
static std::vector<Entry> entries_of(const std::vector<char>& img) {
  std::vector<Entry> out;
  auto rd = [&](size_t off, size_t n) {
    uint64_t v = 0;
    if (off + n <= img.size()) std::memcpy(&v, img.data() + off, n);
    return v;
  };
  const uint64_t body = rd(8, 8);
  size_t off = 16;
  while (off + 64 <= 16 + body && off + 64 <= img.size()) {
    Entry e{};
    e.kind = static_cast<uint16_t>(rd(off, 2));
    const size_t hsz = rd(off + 4, 4), padded = rd(off + 8, 8);
    e.psize = static_cast<uint32_t>(rd(off + 16, 4));
    e.flags = rd(off + 40, 8);
    e.usize = rd(off + 56, 8);
    const size_t id_off = rd(off + 32, 4), id_len = rd(off + 36, 4);
    if (id_off + id_len <= hsz && off + id_off + id_len <= img.size()) e.ident.assign(&img[off + id_off], id_len);
    const size_t n = e.psize ? e.psize : padded;
    if (off + hsz + n <= img.size()) e.payload.assign(&img[off + hsz], n);
    out.push_back(std::move(e));
    off += hsz + padded;
  }
  return out;
}

static std::vector<char> fatbin_with(std::vector<const char*> options,
                                     const std::vector<std::pair<const char*, const char*>>& ptx_by_arch,
                                     const std::vector<char>* cubin = nullptr, const char* cubin_arch = "80") {
  nvFatbinHandle h = nullptr;
  if (nvFatbinCreate(&h, options.data(), options.size()) != NVFATBIN_SUCCESS) return {};
  for (const auto& [text, arch] : ptx_by_arch) nvFatbinAddPTX(h, text, std::strlen(text), arch, "p", "");
  if (cubin) nvFatbinAddCubin(h, cubin->data(), cubin->size(), cubin_arch, "c");
  std::vector<char> img = image_of(h);
  nvFatbinDestroy(&h);
  return img;
}

// The options' rules, the flag word, the stripped comments, compression.
static void options_flags_comments_compression(const std::vector<char>& cubin) {
  using Probe = int (*)();
  const auto zstd_probe = reinterpret_cast<Probe>(dlsym(RTLD_DEFAULT, "vgpu_nvfatbin_compresses_with_zstd"));
  const bool zstd = !zstd_probe || zstd_probe() != 0;   // NVIDIA's library always can

  // Which option strings make a handle.
  auto creates = [](std::vector<const char*> o) {
    nvFatbinHandle h = nullptr;
    const int r = nvFatbinCreate(&h, o.data(), o.size());
    if (h) nvFatbinDestroy(&h);
    return r;
  };
  is(creates({"-32", "-64"}), NVFATBIN_ERROR_UNRECOGNIZED_OPTION, "-32 and -64 exclude each other");
  is(creates({"-cuda", "-opencl"}), NVFATBIN_ERROR_UNRECOGNIZED_OPTION, "-cuda and -opencl exclude each other");
  is(creates({"-compress=maybe"}), NVFATBIN_ERROR_UNRECOGNIZED_OPTION, "-compress takes true or false");
  is(creates({"-compress=TRUE"}), NVFATBIN_ERROR_UNRECOGNIZED_OPTION, "in lower case");
  is(creates({"-compress-mode=fast"}), NVFATBIN_ERROR_UNRECOGNIZED_OPTION, "-compress-mode takes one of five words");
  is(creates({"-host=Linux"}), NVFATBIN_ERROR_UNRECOGNIZED_OPTION, "-host takes linux, windows or mac");
  is(creates({"-compress-all=true"}), NVFATBIN_ERROR_UNRECOGNIZED_OPTION, "-compress-all takes no value");
  is(creates({"-O3"}), NVFATBIN_ERROR_UNRECOGNIZED_OPTION, "and nothing else is an option");
  is(creates({"-c", "-compress-all", "-compress-mode=balance", "-compress-mode=speed", "-host=mac", "-opencl", "-32"}),
     NVFATBIN_SUCCESS, "while the rest combine");

  // The result strings: none for success, NVIDIA's words for the others.
  check(nvFatbinGetErrorString(NVFATBIN_SUCCESS) == nullptr, "success has no description");
  check(std::strcmp(nvFatbinGetErrorString(NVFATBIN_ERROR_INVALID_ARCH), "invalid architecture") == 0 &&
            std::strcmp(nvFatbinGetErrorString(NVFATBIN_ERROR_INVALID_INDEX), "invalid index input") == 0 &&
            std::strcmp(nvFatbinGetErrorString(NVFATBIN_ERROR_NULL_POINTER), "input contained null pointer") == 0,
        "the descriptions are NVIDIA's");
  check(std::strcmp(nvFatbinGetErrorString(static_cast<nvFatbinResult>(99)), "unknown error") == 0,
        "and an unknown code reads \"unknown error\"");

  // The architecture is read as a decimal number with an optional a or f.
  nvFatbinHandle h = nullptr;
  nvFatbinCreate(&h, nullptr, 0);
  const std::string ptx = which(80, 1);
  bool taken = true, refused = true;
  for (const char* a : {"", "+80", " 80", "99999", "8"})
    taken = taken && nvFatbinAddPTX(h, ptx.c_str(), ptx.size(), a, "x", "") == NVFATBIN_SUCCESS;
  for (const char* a : {"sm_80", "80 ", "-5", "0x50", "80af", "compute_80"})
    refused = refused && nvFatbinAddPTX(h, ptx.c_str(), ptx.size(), a, "x", "") == NVFATBIN_ERROR_INVALID_ARCH;
  check(taken, "an architecture is read like strtol reads a number: \"\", \"+80\", \" 80\" are taken");
  check(refused, "and \"sm_80\", \"80 \", \"-5\", \"0x50\", \"80af\" are refused");
  nvFatbinDestroy(&h);

  // The flag word: bit 0 64-bit, 1 debug, 2 CUDA, 3 OpenCL, 4 Linux, 5 Mac, 6 Windows host; an a or f
  // on the architecture adds 0x100000 or 0x200000.
  struct Case { std::vector<const char*> o; uint64_t want; const char* what; };
  for (const Case& c : {Case{{}, 0x11, "by default"}, Case{{"-32"}, 0x10, "-32 clears the 64-bit bit"},
                        Case{{"-host=windows"}, 0x41, "-host=windows"}, Case{{"-host=mac"}, 0x21, "-host=mac"},
                        Case{{"-opencl"}, 0x19, "-opencl"}, Case{{"-cuda"}, 0x15, "-cuda"}, Case{{"-g"}, 0x13, "-g"}}) {
    std::vector<const char*> o = {"-compress=false"};
    o.insert(o.end(), c.o.begin(), c.o.end());
    const auto es = entries_of(fatbin_with(o, {{ptx.c_str(), "80"}}));
    check(es.size() == 1 && es[0].flags == c.want, (std::string("an entry's flags ") + c.what).c_str());
  }
  {
    const auto es = entries_of(fatbin_with({"-compress=false"}, {{ptx.c_str(), "90a"}, {ptx.c_str(), "100f"}, {ptx.c_str(), "86"}}));
    check(es.size() == 3 && es[0].flags == 0x100011 && es[1].flags == 0x200011 && es[2].flags == 0x11,
          "an arch suffix a or f sets 0x100000 or 0x200000 in the entry's flags");
  }

  // Comments go; a string's contents stay.
  const std::string commented = ".version 7.8\n.target sm_80\n.address_size 64\n// whole line\n/* block\nspanning */\n"
                                ".file 1 \"/tmp//a.cu\" // trailing\n/* a */.visible /* b */.entry k() { ret; } /* unterminated";
  {
    const auto es = entries_of(fatbin_with({"-compress=false"}, {{commented.c_str(), "80"}}));
    const std::string want = ".version 7.8\n.target sm_80\n.address_size 64\n\n\n.file 1 \"/tmp//a.cu\" \n.visible .entry k() { ret; } ";
    check(es.size() == 1 && es[0].payload.compare(0, want.size(), want) == 0 && es[0].payload.size() == want.size() + 8 - want.size() % 8,
          "PTX is stored without its comments (// to the line's end, /* */ whole, quoted text kept)");
  }

  // Compression. PTX by default (zstd), LZ4 for speed, nothing for false / none; a cubin only when asked.
  const std::string big = [&] {
    std::string t = ".version 7.8\n.target sm_80\n.address_size 64\n";
    for (int i = 0; i < 200; ++i) t += ".visible .entry kk" + std::to_string(i) + "(.param .u64 p) { .reg .b32 %r<4>; ret; }\n";
    return t;
  }();
  const uint64_t zflag = 0x8000, lflag = 0x2000;
  auto ptx_entry = [&](std::vector<const char*> o) {
    const auto es = entries_of(fatbin_with(o, {{big.c_str(), "80"}}));
    return es.empty() ? Entry{} : es[0];
  };
  const Entry def = ptx_entry({}), none = ptx_entry({"-compress=false"}), spd = ptx_entry({"-compress-mode=speed"}),
              nm = ptx_entry({"-compress=true", "-compress-mode=none"}), sz = ptx_entry({"-compress-mode=size"});
  check(none.flags == 0x11 && none.psize == 0 && nm.flags == 0x11, "-compress=false and -compress-mode=none store PTX as it is");
  check(spd.flags == (0x11 | lflag) && spd.usize == big.size() + 1 && spd.psize < big.size() / 2,
        "-compress-mode=speed compresses PTX with LZ4, and the entry says how big it expands");
  if (zstd) {
    check(def.flags == (0x11 | zflag) && def.usize == big.size() + 1 && def.psize < big.size() / 4,
          "PTX is compressed with zstd by default");
    check(sz.flags == (0x11 | zflag) && sz.usize == big.size() + 1 && sz.psize < big.size() / 4,
          "so does -compress-mode=size");
  } else {
    std::printf("skip zstd checks: libzstd is not installed, so entries are stored as they are\n");
  }
  {  // a cubin: compressed only for -compress-all, -compress-mode=size and -g
    const auto plain = entries_of(fatbin_with({}, {}, &cubin));
    const auto all = entries_of(fatbin_with({"-compress-all"}, {}, &cubin));
    const auto gee = entries_of(fatbin_with({"-g"}, {}, &cubin));
    const auto fls = entries_of(fatbin_with({"-compress-all", "-compress=false"}, {}, &cubin));
    // (A cubin that is PTX -- VirtualGPU's -- is PTX here and compressed anyway.)
    if (!plain.empty() && plain[0].kind == 2) {
      check(plain[0].flags == 0x11, "a cubin is stored as it is by default");
      check(zstd ? (all[0].flags & zflag) && (gee[0].flags & zflag) : true, "but -compress-all and -g compress it");
      check(fls[0].flags == 0x11, "and -compress=false wins over -compress-all");
    }
  }
  // Every one of these loads, and runs.
  for (const std::vector<const char*>& o : {std::vector<const char*>{}, {"-compress-mode=speed"}, {"-compress-mode=size"},
                                            {"-compress-all"}, {"-compress=false"}, {"-g"}}) {
    const std::vector<char> img = fatbin_with(o, {{which(80, 7).c_str(), "80"}});
    CUmodule mod;
    bool ok = !img.empty() && cuModuleLoadData(&mod, img.data()) == CUDA_SUCCESS;
    const std::vector<int> out = ok ? run(mod, "which", 1) : std::vector<int>{};
    check(out.size() == 1 && out[0] == 7, ("a fatbin built with " + std::string(o.empty() ? "no options" : o[0]) + " loads and runs").c_str());
    if (ok) cuModuleUnload(mod);
  }
  // An index file: nothing offered is one.
  nvFatbinCreate(&h, nullptr, 0);
  const char text[] = "an index?";
  IS(nvFatbinAddIndex(h, text, sizeof text, "i"), NVFATBIN_ERROR_INVALID_INDEX);
  IS(nvFatbinAddIndex(h, text, 0, "i"), NVFATBIN_ERROR_EMPTY_INPUT);
  IS(nvFatbinAddIndex(h, nullptr, 4, "i"), NVFATBIN_ERROR_NULL_POINTER);
  nvFatbinDestroy(&h);
}

int main(int argc, char** argv) {
  if (cuInit(0) != CUDA_SUCCESS) {
    std::printf("FAIL: cuInit\n");
    return 1;
  }
  CUdevice dev;
  CUcontext ctx;
  cuDeviceGet(&dev, 0);
  cuDevicePrimaryCtxRetain(&ctx, dev);
  cuCtxSetCurrent(ctx);

  unsigned major = 0, minor = 0;
  IS(nvFatbinVersion(&major, &minor), NVFATBIN_SUCCESS);
  check(major >= 12, "nvFatbinVersion is a CUDA 12 or later version");
  for (int r = NVFATBIN_ERROR_INTERNAL; r <= NVFATBIN_ERROR_INTERNAL_PTX_OPTION; ++r)
    if (!nvFatbinGetErrorString(static_cast<nvFatbinResult>(r))) {
      check(false, "every result has a description");
      break;
    }

  // Creation.
  nvFatbinHandle h = nullptr;
  IS(nvFatbinCreate(nullptr, nullptr, 0), NVFATBIN_ERROR_NULL_POINTER);
  const char* bogus[] = {"-bogus"};
  IS(nvFatbinCreate(&h, bogus, 1), NVFATBIN_ERROR_UNRECOGNIZED_OPTION);
  if (h) nvFatbinDestroy(&h);
  const char* fine[] = {"-64", "-compress=false", "-host=linux", "-cuda", "-g"};
  IS(nvFatbinCreate(&h, fine, 5), NVFATBIN_SUCCESS);
  nvFatbinDestroy(&h);
  IS(nvFatbinCreate(&h, nullptr, 0), NVFATBIN_SUCCESS);
  size_t size = 0;
  IS(nvFatbinSize(h, &size), NVFATBIN_SUCCESS);
  check(size == 16, "an empty fatbin is its 16-byte container header");
  IS(nvFatbinSize(h, nullptr), NVFATBIN_ERROR_NULL_POINTER);

  // PTX, and what is refused on the way in. The architecture is the bare
  // number; PTX whose .target differs from it is still accepted.
  const std::string sm80 = which(80, 1), sm90 = which(90, 2);
  IS(nvFatbinAddPTX(h, sm80.c_str(), sm80.size() + 1, "sm_80", "which", nullptr), NVFATBIN_ERROR_INVALID_ARCH);
  IS(nvFatbinAddPTX(h, sm80.c_str(), sm80.size() + 1, nullptr, "which", nullptr), NVFATBIN_ERROR_NULL_POINTER);
  IS(nvFatbinAddPTX(h, nullptr, 10, "80", "which", nullptr), NVFATBIN_ERROR_NULL_POINTER);
  IS(nvFatbinAddPTX(h, sm80.c_str(), 0, "80", "which", nullptr), NVFATBIN_ERROR_EMPTY_INPUT);
  const std::string unversioned = "// no version\n.target sm_80\n.address_size 64\n";
  IS(nvFatbinAddPTX(h, unversioned.c_str(), unversioned.size() + 1, "80", "u", nullptr),
     NVFATBIN_ERROR_MISSING_PTX_VERSION);
  IS(nvFatbinAddPTX(h, sm90.c_str(), sm90.size() + 1, "90", "which", nullptr), NVFATBIN_SUCCESS);
  IS(nvFatbinAddPTX(h, sm80.c_str(), sm80.size(), "80", "which", "-O3"), NVFATBIN_SUCCESS);
  {
    // The driver JITs the newest PTX the device can run: sm_80's.
    std::vector<char> image = image_of(h);
    CUmodule mod;
    IS(cuModuleLoadData(&mod, image.data()), CUDA_SUCCESS);
    std::vector<int> out = run(mod, "which", 1);
    check(out.size() == 1 && out[0] == 1, "cuModuleLoadData picks the sm_80 PTX");
    cuModuleUnload(mod);
    IS(cuModuleLoadFatBinary(&mod, image.data()), CUDA_SUCCESS);
    out = run(mod, "which", 1);
    check(out.size() == 1 && out[0] == 1, "and so does cuModuleLoadFatBinary");
    cuModuleUnload(mod);
  }
  IS(nvFatbinDestroy(&h), NVFATBIN_SUCCESS);
  check(h == nullptr, "destroy clears the handle");
  IS(nvFatbinDestroy(nullptr), NVFATBIN_ERROR_NULL_POINTER);

  // A cubin from nvJitLink -- on VirtualGPU that is PTX, and it goes in as
  // what it is -- and the fatbin loads and runs it.
  std::vector<char> cubin = linked_cubin({{NVJITLINK_INPUT_PTX, bytes_of(kMain)},
                                          {NVJITLINK_INPUT_PTX, bytes_of(kLib)}},
                                         "-arch=sm_80");
  check(!cubin.empty(), "nvJitLink links the kernel");
  nvFatbinCreate(&h, nullptr, 0);
  IS(nvFatbinAddCubin(h, cubin.data(), cubin.size(), "80", "k"), NVFATBIN_SUCCESS);
  const char garbage[] = "this is not an ELF image";
  IS(nvFatbinAddCubin(h, garbage, sizeof garbage, "80", "g"), NVFATBIN_ERROR_ELF_SIZE_MISMATCH);
  IS(nvFatbinAddLTOIR(h, garbage, sizeof garbage, "80", "g", nullptr), NVFATBIN_ERROR_INTERNAL);
  IS(nvFatbinAddIndex(h, garbage, sizeof garbage, "g"), NVFATBIN_ERROR_INVALID_INDEX);
  {
    std::vector<char> image = image_of(h);
    CUmodule mod;
    IS(cuModuleLoadData(&mod, image.data()), CUDA_SUCCESS);
    check(computes_k(run(mod, "k", 32)), "a fatbin around nvJitLink's cubin runs it");
    cuModuleUnload(mod);
  }
  nvFatbinDestroy(&h);
  // A real cubin's architecture must be the one named. (VirtualGPU's cubin
  // is PTX, which carries no such claim.)
  if (!cubin.empty() && cubin[0] == 0x7f) {
    std::vector<char> sm86 = linked_cubin({{NVJITLINK_INPUT_PTX, bytes_of(kMain)},
                                           {NVJITLINK_INPUT_PTX, bytes_of(kLib)}},
                                          "-arch=sm_86");
    nvFatbinCreate(&h, nullptr, 0);
    IS(nvFatbinAddCubin(h, sm86.data(), sm86.size(), "80", "k"), NVFATBIN_ERROR_ELF_ARCH_MISMATCH);
    IS(nvFatbinAddCubin(h, sm86.data(), sm86.size(), "86", "k"), NVFATBIN_SUCCESS);
    nvFatbinDestroy(&h);
  }

  options_flags_comments_compression(cubin);

  // A fatbin of the library's PTX is an nvJitLink input.
  nvFatbinCreate(&h, nullptr, 0);
  IS(nvFatbinAddPTX(h, kLib, std::strlen(kLib), "80", "lib", nullptr), NVFATBIN_SUCCESS);
  {
    std::vector<char> image = image_of(h);
    std::vector<char> linked = linked_cubin({{NVJITLINK_INPUT_PTX, bytes_of(kMain)},
                                             {NVJITLINK_INPUT_FATBIN, image}},
                                            "-arch=sm_80");
    CUmodule mod;
    check(!linked.empty() && cuModuleLoadData(&mod, linked.data()) == CUDA_SUCCESS &&
              computes_k(run(mod, "k", 32)),
          "nvJitLink links the kernel with a fatbin nvFatbin wrote");
  }
  nvFatbinDestroy(&h);

  // A host object's relocatable PTX, under its module identifier, once per
  // architecture; the fatbin it makes links with the kernel too.
  if (argc >= 2) {
    std::ifstream f(argv[1], std::ios::binary);
    const std::vector<char> object((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    nvFatbinCreate(&h, nullptr, 0);
    IS(nvFatbinAddReloc(h, object.data(), object.size()), NVFATBIN_SUCCESS);
    IS(nvFatbinAddReloc(h, object.data(), object.size()), NVFATBIN_ERROR_IDENTIFIER_REUSE);
    std::vector<char> image = image_of(h);
    std::vector<char> linked = linked_cubin({{NVJITLINK_INPUT_PTX, bytes_of(kMain)},
                                             {NVJITLINK_INPUT_FATBIN, image}},
                                            "-arch=sm_80");
    CUmodule mod;
    check(!linked.empty() && cuModuleLoadData(&mod, linked.data()) == CUDA_SUCCESS &&
              computes_k(run(mod, "k", 32)),
          "a host object's relocatable PTX, packaged, links with the kernel");
    nvFatbinDestroy(&h);
  } else {
    std::printf("note: no host object given; nvFatbinAddReloc is not checked\n");
  }

  std::printf(failures ? "FAIL: %d nvFatbin checks\n" : "PASS: every nvFatbin check\n", failures);
  return failures != 0;
}
