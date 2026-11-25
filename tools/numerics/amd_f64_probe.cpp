// v_rcp_f64, v_rsq_f64 and v_sqrt_f64 of the card in this machine on operands read as hex from stdin (one 64-bit word a line): prints "operand rcp rsq sqrt" per line, the format of
// tools/numerics/amd/gfx1030.vectors64.txt. hipcc -O2 amd_f64_probe.cpp -o amd_f64_probe; ROCR_VISIBLE_DEVICES=<one device> amd_f64_probe < operands.txt. Our own code.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define OK(x) do{hipError_t e=(x); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(3);}}while(0)
__global__ void k(const uint64_t* in, uint64_t* out, uint32_t n){
  uint32_t t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return;
  double x=__longlong_as_double((long long)in[t]), a, b, c;
  asm volatile("v_rcp_f64 %0, %1":"=v"(a):"v"(x));
  asm volatile("v_rsq_f64 %0, %1":"=v"(b):"v"(x));
  asm volatile("v_sqrt_f64 %0, %1":"=v"(c):"v"(x));
  out[3*t]=(uint64_t)__double_as_longlong(a); out[3*t+1]=(uint64_t)__double_as_longlong(b); out[3*t+2]=(uint64_t)__double_as_longlong(c);
}
int main(){
  std::vector<uint64_t> in; unsigned long long v; while(scanf("%llx",&v)==1) in.push_back(v);
  uint32_t n=in.size(); std::vector<uint64_t> out(3*n);
  uint64_t *di,*dout; OK(hipMalloc(&di,n*8)); OK(hipMalloc(&dout,n*24)); OK(hipMemcpy(di,in.data(),n*8,hipMemcpyHostToDevice));
  k<<<(n+63)/64,64>>>(di,dout,n); OK(hipGetLastError()); OK(hipDeviceSynchronize()); OK(hipMemcpy(out.data(),dout,n*24,hipMemcpyDeviceToHost));
  hipDeviceProp_t p; OK(hipGetDeviceProperties(&p,0)); printf("# %s %s\n",p.name,p.gcnArchName);
  for(uint32_t i=0;i<n;i++) printf("%016llx %016llx %016llx %016llx\n",(unsigned long long)in[i],(unsigned long long)out[3*i],(unsigned long long)out[3*i+1],(unsigned long long)out[3*i+2]);
}
