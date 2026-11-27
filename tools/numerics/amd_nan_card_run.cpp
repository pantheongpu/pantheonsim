// Runs one kernel of amd/tests/data/asm_nan.s or asm_nan2.s (built for the card's own gfx by amd/tests/data/build.sh's recipe) on the card in this
// machine and prints its dwords in the format of amd/tests/data/asm_nan*.gfx1030.card.txt, for a diff against the RX 6800's.
//
//   hipcc -O2 amd_nan_card_run.cpp -o amd_nan_card_run
//   ROCR_VISIBLE_DEVICES=<one device> amd_nan_card_run <object.o> <kernel> <inputs.bin> <items> <ops> <dwords per result: 1 or 2> <out pitch bytes per op>
//
// inputs.bin: the work-items' dwords as the test builds them (16 dwords an item for asm_nan2.s, 8 for asm_nan.s; tools/numerics/amd_nan_inputs.py
// writes it). Work-items run in groups of 64 (the kernels are wave32). Each result is printed as 8 hex digits, or, for two dwords, as 16 where the
// high dword is not 0xa5a5a5a5 and 8 where it is (the card files' convention). Our own code.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(3); } } while (0)

int main(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "usage: amd_nan_card_run object kernel inputs.bin items ops dwords pitch\n"); return 2; }
  std::ifstream fo(argv[1], std::ios::binary), fi(argv[3], std::ios::binary);
  std::string obj((std::istreambuf_iterator<char>(fo)), {}), in((std::istreambuf_iterator<char>(fi)), {});
  const unsigned items = atoi(argv[4]), ops = atoi(argv[5]), dw = atoi(argv[6]), pitch = atoi(argv[7]);
  hipDeviceProp_t p; OK(hipGetDeviceProperties(&p, 0));
  fprintf(stderr, "# device %s %s\n", p.name, p.gcnArchName);
  hipModule_t m; OK(hipModuleLoadData(&m, obj.data()));
  hipFunction_t f; OK(hipModuleGetFunction(&f, m, argv[2]));
  void *din, *dout; OK(hipMalloc(&din, in.size())); OK(hipMalloc(&dout, (size_t)ops * pitch));
  OK(hipMemcpy(din, in.data(), in.size(), hipMemcpyHostToDevice)); OK(hipMemset(dout, 0, (size_t)ops * pitch));
  struct { void* a; void* b; } args = {din, dout}; size_t sz = sizeof args;
  void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
  OK(hipModuleLaunchKernel(f, items / 64, 1, 1, 64, 1, 1, 0, 0, nullptr, cfg));
  OK(hipDeviceSynchronize());
  std::vector<uint32_t> out((size_t)ops * pitch / 4); OK(hipMemcpy(out.data(), dout, (size_t)ops * pitch, hipMemcpyDeviceToHost));
  for (unsigned op = 0; op < ops; ++op) {
    printf("%s %u", argv[2], op);
    for (unsigned t = 0; t < items; ++t) {
      const uint32_t* w = &out[(size_t)op * pitch / 4 + (size_t)t * dw];
      if (dw == 1) printf(" %08x", w[0]);
      else if (w[1] == 0xa5a5a5a5u) printf(" %08x", w[0]);
      else printf(" %08x%08x", w[1], w[0]);
    }
    printf("\n");
  }
}
