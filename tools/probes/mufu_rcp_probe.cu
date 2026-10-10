// Round-5 card probe: rcp.approx.ftz.f32 (MUFU.RCP) of an array of bit patterns. usage: <in.bin of u32> <out.bin of u32>.
// The card's reciprocal differs from the correctly rounded one by one ulp on 13% of the mantissas of [1, 2).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
__global__ void k(const uint32_t* in, uint32_t* out, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float r; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(__uint_as_float(in[i]))); out[i]=__float_as_uint(r); }
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET); int n=sz/4; std::vector<uint32_t> h(n); fread(h.data(),4,n,f); fclose(f);
  uint32_t *di,*dout; cudaMalloc(&di,n*4); cudaMalloc(&dout,n*4); cudaMemcpy(di,h.data(),n*4,cudaMemcpyHostToDevice);
  k<<<(n+255)/256,256>>>(di,dout,n); cudaMemcpy(h.data(),dout,n*4,cudaMemcpyDeviceToHost);
  f=fopen(argv[2],"wb"); fwrite(h.data(),4,n,f); fclose(f);
}
