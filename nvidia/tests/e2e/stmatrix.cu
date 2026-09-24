// stmatrix: the warp writes the 8x8 matrices its registers hold back to shared
// memory, in the distribution ldmatrix reads them in.
//
// This is how a kernel gets an mma result out of registers and into shared
// memory for the next stage, which is what a fused attention kernel does
// between its two multiplies. The instruction is sm_90 and up.
//
// Two things are worth pinning separately. One is the layout: every element has
// to land where the ISA says, checked here by reading the shared tile back with
// plain loads and comparing against the mapping element by element. The other is
// that stmatrix and ldmatrix are inverses -- a fragment loaded by one and stored
// by the other has to come back unchanged, which is the property a kernel
// actually leans on, and it would still hold if both had the same permutation
// wrong, so the first check is what catches that.
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

// Each lane's register holds two consecutive 16-bit elements of one row:
// lane L holds row L/4, columns 2*(L%4) and 2*(L%4)+1.
__device__ __forceinline__ unsigned fragment_of(const unsigned short* tile8x8, int lane) {
  const int row = lane / 4, col = 2 * (lane % 4);
  return (static_cast<unsigned>(tile8x8[row * 8 + col + 1]) << 16) |
         static_cast<unsigned>(tile8x8[row * 8 + col]);
}

// One matrix, written straight and transposed, plus the round trip through
// ldmatrix. `out` gets the straight tile, the transposed tile, and then 32
// words: what each lane held after loading the straight tile back.
__global__ void one(const unsigned short* in, unsigned short* straight,
                    unsigned short* transposed, unsigned* reloaded) {
  __shared__ unsigned short tile[64];
  const int lane = threadIdx.x;
  const unsigned frag = fragment_of(in, lane);

  // Every lane supplies an address; the eight that name rows are the ones used.
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&tile[(lane % 8) * 8]));

  asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};" ::"r"(addr), "r"(frag));
  __syncthreads();
  if (lane < 8)
    for (int c = 0; c < 8; ++c) straight[lane * 8 + c] = tile[lane * 8 + c];
  __syncthreads();

  // The same fragment stored transposed.
  asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};" ::"r"(addr),
               "r"(frag));
  __syncthreads();
  if (lane < 8)
    for (int c = 0; c < 8; ++c) transposed[lane * 8 + c] = tile[lane * 8 + c];
  __syncthreads();

  // Put the straight tile back and load it with ldmatrix: what a lane gets has
  // to be what it stored.
  asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};" ::"r"(addr), "r"(frag));
  __syncthreads();
  unsigned back = 0;
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(back) : "r"(addr));
  reloaded[lane] = back;
}

// Four matrices at once: lanes 0-31 each name one row of one of them, so the
// warp writes 4 x 8 x 8 elements in one instruction.
__global__ void four(const unsigned short* in, unsigned short* out, unsigned* reloaded) {
  __shared__ unsigned short tile[256];
  const int lane = threadIdx.x;
  unsigned frag[4];
  for (int mat = 0; mat < 4; ++mat) frag[mat] = fragment_of(in + mat * 64, lane);

  // Row r of matrix i comes from lane i*8+r, and the matrices sit one after the
  // other in shared memory.
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&tile[lane * 8]));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr),
               "r"(frag[0]), "r"(frag[1]), "r"(frag[2]), "r"(frag[3]));
  __syncthreads();
  for (int i = 0; i < 8; ++i) out[lane * 8 + i] = tile[lane * 8 + i];
  __syncthreads();

  unsigned back[4];
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(back[0]), "=r"(back[1]), "=r"(back[2]), "=r"(back[3])
               : "r"(addr));
  for (int mat = 0; mat < 4; ++mat) reloaded[mat * 32 + lane] = back[mat];
}

static unsigned host_fragment(const unsigned short* tile8x8, int lane) {
  const int row = lane / 4, col = 2 * (lane % 4);
  return (static_cast<unsigned>(tile8x8[row * 8 + col + 1]) << 16) |
         static_cast<unsigned>(tile8x8[row * 8 + col]);
}

int main() {
  // Distinct values, so a misplaced element is visible rather than plausible.
  unsigned short in[256];
  for (int i = 0; i < 256; ++i) in[i] = static_cast<unsigned short>(1000 + i);

  unsigned short *d_in = nullptr, *d_straight = nullptr, *d_trans = nullptr, *d_four = nullptr;
  unsigned *d_reloaded = nullptr, *d_reloaded4 = nullptr;
  CK(cudaMalloc(&d_in, sizeof in));
  CK(cudaMalloc(&d_straight, 64 * sizeof(unsigned short)));
  CK(cudaMalloc(&d_trans, 64 * sizeof(unsigned short)));
  CK(cudaMalloc(&d_four, 256 * sizeof(unsigned short)));
  CK(cudaMalloc(&d_reloaded, 32 * sizeof(unsigned)));
  CK(cudaMalloc(&d_reloaded4, 128 * sizeof(unsigned)));
  CK(cudaMemcpy(d_in, in, sizeof in, cudaMemcpyHostToDevice));

  one<<<1, 32>>>(d_in, d_straight, d_trans, d_reloaded);
  CK(cudaDeviceSynchronize());

  unsigned short straight[64], trans[64];
  unsigned reloaded[32];
  CK(cudaMemcpy(straight, d_straight, sizeof straight, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(trans, d_trans, sizeof trans, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(reloaded, d_reloaded, sizeof reloaded, cudaMemcpyDeviceToHost));

  // The straight store: the tile in shared memory is the matrix the registers held.
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 8; ++c)
      if (straight[r * 8 + c] != in[r * 8 + c]) {
        printf("FAIL straight [%d][%d] = %u, expected %u\n", r, c, straight[r * 8 + c],
               in[r * 8 + c]);
        return 1;
      }
  // The transposed store: element (r, c) of the tile is (c, r) of the matrix.
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 8; ++c)
      if (trans[r * 8 + c] != in[c * 8 + r]) {
        printf("FAIL trans [%d][%d] = %u, expected %u\n", r, c, trans[r * 8 + c], in[c * 8 + r]);
        return 1;
      }
  // The round trip: every lane got back exactly what it stored.
  for (int lane = 0; lane < 32; ++lane) {
    const unsigned want = host_fragment(in, lane);
    if (reloaded[lane] != want) {
      printf("FAIL lane %d reloaded 0x%08x, stored 0x%08x\n", lane, reloaded[lane], want);
      return 1;
    }
  }

  four<<<1, 32>>>(d_in, d_four, d_reloaded4);
  CK(cudaDeviceSynchronize());
  unsigned short got4[256];
  unsigned reloaded4[128];
  CK(cudaMemcpy(got4, d_four, sizeof got4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(reloaded4, d_reloaded4, sizeof reloaded4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 256; ++i)
    if (got4[i] != in[i]) {
      printf("FAIL x4 element %d = %u, expected %u\n", i, got4[i], in[i]);
      return 1;
    }
  for (int mat = 0; mat < 4; ++mat)
    for (int lane = 0; lane < 32; ++lane) {
      const unsigned want = host_fragment(in + mat * 64, lane);
      if (reloaded4[mat * 32 + lane] != want) {
        printf("FAIL x4 matrix %d lane %d reloaded 0x%08x, stored 0x%08x\n", mat, lane,
               reloaded4[mat * 32 + lane], want);
        return 1;
      }
    }

  CK(cudaFree(d_in));
  CK(cudaFree(d_straight));
  CK(cudaFree(d_trans));
  CK(cudaFree(d_four));
  CK(cudaFree(d_reloaded));
  CK(cudaFree(d_reloaded4));
  printf("PASS\n");
  return 0;
}
