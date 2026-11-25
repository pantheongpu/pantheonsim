// Fuzz target: the CUDA driver API the simulator presents (libcuda.so.1). Input byte 0 picks what the rest is.
//   mode 0: a module image, as cuModuleLoadData takes it (PTX text, a cubin or a fatbin), then, if it loaded, every
//           kernel it names is looked up and launched on a small grid, with the rest of the input as its arguments.
//   mode 1: a stream of memory calls (allocate, free, copy, set, launch) with sizes, offsets and device pointers
//           taken from the input, valid or not.
// Whatever the program passes, the driver answers with a CUresult: no crash, no sanitizer report, no hang, no C++
// exception out of an extern "C" function. (A host pointer that is not memory the caller owns is the caller's bug,
// as on a real driver; every host buffer here is owned and at least as long as the count passed.)
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu_cuda.h"

// Not in vgpu_cuda.h, exported by the shim all the same.
extern "C" {
CUresult cuMemsetD8(CUdeviceptr dst, unsigned char v, size_t n);
CUresult cuMemsetD32(CUdeviceptr dst, unsigned int v, size_t n);
CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
}

namespace {

struct Reader {
  const uint8_t* p;
  size_t n;
  uint8_t u8() { return n ? (--n, *p++) : 0; }
  uint32_t u32() {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t{u8()} << (8 * i);
    return v;
  }
  uint64_t u64() { return u32() | (uint64_t{u32()} << 32); }
  bool empty() const { return n == 0; }
};

// ".entry name(" in PTX text: the kernels to look up.
std::vector<std::string> entries_of(const std::string& text) {
  std::vector<std::string> out;
  for (size_t at = 0; out.size() < 4 && (at = text.find(".entry", at)) != std::string::npos; at += 6) {
    size_t b = at + 6;
    while (b < text.size() && (text[b] == ' ' || text[b] == '\n' || text[b] == '\t')) ++b;
    size_t e = b;
    while (e < text.size() && (std::isalnum(static_cast<unsigned char>(text[e])) || text[e] == '_' || text[e] == '$')) ++e;
    if (e > b) out.push_back(text.substr(b, e - b));
  }
  return out;
}

CUresult launch(CUfunction f, Reader& r, const uint8_t* args, size_t nargs) {
  const unsigned gx = 1 + r.u8() % 3, gy = 1 + r.u8() % 2, gz = 1;
  const unsigned bx = 1 + r.u8() % 96, by = 1 + r.u8() % 2, bz = 1;
  std::vector<uint8_t> buf(4096, 0);
  std::memcpy(buf.data(), args, nargs < buf.size() ? nargs : buf.size());
  size_t sz = nargs < buf.size() ? nargs : buf.size();
  void* extra[] = {reinterpret_cast<void*>(uintptr_t{1}), buf.data(), reinterpret_cast<void*>(uintptr_t{2}), &sz, nullptr};
  CUresult res = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, r.u32() % 4096, nullptr, nullptr, extra);
  if (res == CUDA_SUCCESS) res = cuCtxSynchronize();
  return res;
}

void module_mode(Reader r) {
  const bool binary = r.n >= 4 && (!std::memcmp(r.p, "\x7f""ELF", 4) || !std::memcmp(r.p, "\xb1\x43\x62\x46", 4) ||
                                   !std::memcmp(r.p, "\x50\xed\x55\xba", 4));
  // Text is passed NUL-terminated and exactly sized. A binary image carries its own size, which the driver reads
  // from the image: it sits in a zero-padded block so a header that claims more than was given is read, not guessed.
  std::vector<uint8_t> image(binary ? (1u << 20) : r.n + 1, 0);
  std::memcpy(image.data(), r.p, r.n < image.size() ? r.n : image.size() - 1);
  CUmodule mod = nullptr;
  if (cuModuleLoadData(&mod, image.data()) != CUDA_SUCCESS || !mod) return;
  std::vector<std::string> names = entries_of(std::string(reinterpret_cast<const char*>(r.p), r.n));
  names.push_back("k");
  names.push_back("_Z6kernelPf");
  for (const std::string& name : names) {
    CUfunction f = nullptr;
    if (cuModuleGetFunction(&f, mod, name.c_str()) == CUDA_SUCCESS && f) {
      Reader a{r.p, r.n};
      (void)launch(f, a, r.p, r.n < 256 ? r.n : 256);
    }
  }
  CUdeviceptr g = 0;
  size_t gb = 0;
  (void)cuModuleGetGlobal(&g, &gb, mod, "g");
  (void)cuModuleUnload(mod);
}

void memory_mode(Reader r) {
  std::vector<CUdeviceptr> live;
  std::vector<uint8_t> host(4096);
  for (auto& b : host) b = r.u8();
  CUfunction last_fn = nullptr;
  CUmodule mod = nullptr;
  const auto pick = [&]() -> CUdeviceptr {
    const uint8_t how = r.u8();
    if (!live.empty() && how < 200) return live[how % live.size()] + (how < 100 ? 0 : r.u32() % 4096);
    return how < 230 ? r.u64() : r.u32();
  };
  for (int ops = 0; ops < 200 && !r.empty(); ++ops) {
    switch (r.u8() % 11) {
      case 0: {
        CUdeviceptr p = 0;
        if (cuMemAlloc(&p, r.u32() % (1u << 20)) == CUDA_SUCCESS) live.push_back(p);
        break;
      }
      case 1: {
        CUdeviceptr p = 0;
        size_t pitch = 0;
        (void)pitch;
        if (cuMemAlloc(&p, uint64_t{r.u8()} * r.u8() * 16) == CUDA_SUCCESS) live.push_back(p);
        break;
      }
      case 2: {
        const CUdeviceptr p = pick();
        if (cuMemFree(p) == CUDA_SUCCESS) {
          for (size_t i = 0; i < live.size(); ++i)
            if (live[i] == p) { live.erase(live.begin() + i); break; }
        }
        break;
      }
      case 3: (void)cuMemcpyHtoD(pick(), host.data(), r.u32() % (host.size() + 1)); break;
      case 4: (void)cuMemcpyDtoH(host.data(), pick(), r.u32() % (host.size() + 1)); break;
      case 5: (void)cuMemcpyDtoD(pick(), pick(), r.u32() % (1u << 16)); break;
      case 6: (void)cuMemsetD8(pick(), r.u8(), r.u32() % (1u << 16)); break;
      case 7: (void)cuMemsetD32(pick(), r.u32(), r.u32() % (1u << 14)); break;
      case 8: {
        size_t fr = 0, tot = 0;
        (void)cuMemGetInfo(&fr, &tot);
        break;
      }
      case 9: {
        // A small kernel that stores to the first argument, launched with arguments from the input.
        static const char kStore[] =
            ".version 7.0\n.target sm_80\n.address_size 64\n"
            ".visible .entry k(.param .u64 p, .param .u32 n)\n{\n.reg .b64 %rd<3>;\n.reg .b32 %r<3>;\n"
            "ld.param.u64 %rd1, [p];\nld.param.u32 %r1, [n];\nmov.u32 %r2, %tid.x;\n"
            "cvta.to.global.u64 %rd2, %rd1;\nst.global.u32 [%rd2], %r2;\nret;\n}\n";
        if (!mod && cuModuleLoadData(&mod, kStore) == CUDA_SUCCESS) (void)cuModuleGetFunction(&last_fn, mod, "k");
        if (last_fn) {
          uint8_t args[16];
          const CUdeviceptr p = pick();
          std::memcpy(args, &p, 8);
          const uint32_t n = r.u32();
          std::memcpy(args + 8, &n, 4);
          (void)launch(last_fn, r, args, 12);
        }
        break;
      }
      default: (void)cuCtxSynchronize(); break;
    }
  }
  for (const CUdeviceptr p : live) (void)cuMemFree(p);
  if (mod) (void)cuModuleUnload(mod);
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  setenv("VGPU_QUIET", "1", 1);
  setenv("VGPU_GPU", "nvidia/rtx3060", 0);
  setenv("VGPU_MAX_STEPS", "200000", 0);
  if (cuInit(0) != CUDA_SUCCESS) std::abort();
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS || cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS)
    std::abort();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 1 || size > (1u << 18)) return 0;
  Reader r{data + 1, size - 1};
  if (data[0] & 1) memory_mode(r);
  else module_mode(r);
  return 0;
}
