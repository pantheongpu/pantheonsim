// Round-4 measurement probe for NPP watershed labels (see nvidia/docs/libraries.md): runs nppiSegmentWatershed_8u_C1IR_Ctx on a batch of images and writes the segmented image and the marker labels.
// Batch: npp_wsb <norm 0|1> <in.bin> <out.bin>. in: u32 count, then per image u32 w,u32 h,w*h bytes. out: per image seg bytes (w*h) then labels (u32 w*h).
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
  int l1 = atoi(argv[1]);
  FILE* f = fopen(argv[2], "rb"); unsigned cnt; if (fread(&cnt, 4, 1, f) != 1) return 2;
  FILE* o = fopen(argv[3], "wb");
  NppStreamContext c{}; c.hStream = 0; cudaGetDevice(&c.nCudaDeviceId);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, c.nCudaDeviceId);
  c.nMultiProcessorCount = p.multiProcessorCount; c.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = p.maxThreadsPerBlock; c.nSharedMemPerBlock = p.sharedMemPerBlock;
  c.nCudaDevAttrComputeCapabilityMajor = p.major; c.nCudaDevAttrComputeCapabilityMinor = p.minor;
  for (unsigned k = 0; k < cnt; ++k) {
    unsigned w, h; if (fread(&w, 4, 1, f) != 1 || fread(&h, 4, 1, f) != 1) return 3;
    std::vector<unsigned char> img((size_t)w * h); if (fread(img.data(), 1, img.size(), f) != img.size()) return 3;
    int pitch = 0; Npp8u* d = nppiMalloc_8u_C1(w, h, &pitch);
    cudaMemcpy2D(d, (size_t)pitch, img.data(), w, w, h, cudaMemcpyHostToDevice);
    Npp32u* dl; cudaMalloc(&dl, (size_t)w * h * 4); cudaMemset(dl, getenv("LABFILL") ? atoi(getenv("LABFILL")) : 0xEE, (size_t)w * h * 4);
    NppiSize roi = {(int)w, (int)h};
    size_t bs = 0; nppiSegmentWatershedGetBufferSize_8u_C1R(roi, &bs);
    Npp8u* buf; cudaMalloc(&buf, bs);
    NppStatus st = nppiSegmentWatershed_8u_C1IR_Ctx(d, pitch, dl, w * 4, l1 ? nppiNormL1 : nppiNormInf, NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, roi, buf, c);
    cudaDeviceSynchronize();
    std::vector<unsigned char> out((size_t)w * h); cudaMemcpy2D(out.data(), w, d, (size_t)pitch, w, h, cudaMemcpyDeviceToHost);
    std::vector<unsigned> lab((size_t)w * h); cudaMemcpy(lab.data(), dl, lab.size() * 4, cudaMemcpyDeviceToHost);
    if (st != NPP_SUCCESS) { fprintf(stderr, "status %d\n", (int)st); }
    fwrite(out.data(), 1, out.size(), o); fwrite(lab.data(), 4, lab.size(), o);
    nppiFree(d); cudaFree(dl); cudaFree(buf);
  }
  fclose(o); return 0;
}
