// gfx1250 (CDNA 5, MI455X) Tensor Data Mover (tensor1250.gfx1250, wave32). The oracle is AMD's CDNA5 ISA (section 10.11): a
// descriptor ("D#") in groups of scalar registers says where a tile of a tensor of up to five dimensions is in global memory
// and in LDS, and tensor_load_to_lds and tensor_store_from_lds move it. Each case here builds the descriptor from the ISA's
// tables bit range by bit range, and works out on the host what the move leaves in LDS or in memory, by walking the tile
// the way the ISA's pseudocode does.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

constexpr int kLds = 4096;

// One wave: the move (a load into LDS, or a store from it), then the LDS copied out. Descriptor words 0-3 are group 0, 4-11
// group 1, 12-15 group 2, 16-19 group 3; `groups` says how many groups the instruction names (2 or 4).
__global__ void k_move(const uint32_t* desc, const uint8_t* lds_init, uint8_t* lds_out, int store, int groups) {
  __shared__ uint8_t lds[kLds];
  const int l = threadIdx.x;
  for (int i = l; i < kLds; i += 32) lds[i] = lds_init[i];
  __syncthreads();
  const auto word = [&](int i) { return __builtin_amdgcn_readfirstlane(static_cast<int>(desc[i])); };
  v4i g0 = {word(0), word(1), word(2), word(3)};
  v8i g1 = {word(4), word(5), word(6), word(7), word(8), word(9), word(10), word(11)};
  v4i g2 = {word(12), word(13), word(14), word(15)};
  v4i g3 = {word(16), word(17), word(18), word(19)};
  if (store) {
    if (groups == 4) __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
    else __builtin_amdgcn_tensor_store_from_lds_d2(g0, g1, 0);
  } else {
    if (groups == 4) __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
    else __builtin_amdgcn_tensor_load_to_lds_d2(g0, g1, 0);
  }
  __builtin_amdgcn_s_wait_tensorcnt(0);
  __syncthreads();
  for (int i = l; i < kLds; i += 32) lds_out[i] = lds[i];
}

// Writes `value` into bits [lo, lo + width) of the descriptor (a little-endian run of 32-bit words).
static void put(std::vector<uint32_t>& d, int lo, int width, uint64_t value) {
  for (int b = 0; b < width; ++b) {
    const int bit = lo + b;
    if ((value >> b) & 1) d[bit / 32] |= 1u << (bit % 32);
  }
}

struct Case {
  const char* name;
  int data_log2 = 2;
  uint64_t tensor_dim[5] = {8, 4, 0, 0, 0};
  uint64_t tile_dim[5] = {8, 4, 0, 0, 0};
  uint64_t stride[4] = {8, 0, 0, 0};   // in elements: the stride of dimension 1, 2, 3 and 4
  uint64_t start = 0;                  // element offset of the tile's start from the buffer's beginning
  uint32_t lds_addr = 0;
  bool pad = false;
  int pad_interval = 0, pad_amount = 0;
  bool iterate = false;
  int iterate_count = 0;
  uint64_t global_inc = 0, lds_inc = 0;
  bool gather = false, wide = false;
  std::vector<uint32_t> indices;
  bool store = false;
  int groups = 2;
};

static uint32_t seed = 777;
static uint32_t rnd() {
  seed = seed * 1664525u + 1013904223u;
  return seed >> 8;
}

int main() {
  constexpr int kBytes = 8192;
  std::vector<uint8_t> mem(kBytes), lds0(kLds);
  for (auto& b : mem) b = static_cast<uint8_t>(rnd());
  for (auto& b : lds0) b = static_cast<uint8_t>(rnd());
  uint8_t *d_mem = nullptr, *d_out = nullptr;
  CHECK(hipMalloc(&d_mem, kBytes));
  uint8_t* d_lds0 = nullptr;
  uint8_t* d_lds_out = nullptr;
  uint32_t* d_desc = nullptr;
  CHECK(hipMalloc(&d_lds0, kLds));
  CHECK(hipMalloc(&d_lds_out, kLds));
  CHECK(hipMalloc(&d_desc, 20 * 4));
  (void)d_out;
  CHECK(hipMemcpy(d_lds0, lds0.data(), kLds, hipMemcpyHostToDevice));

  std::vector<Case> cases;
  {
    Case c; c.name = "2D load, 4-byte elements, tile past the tensor's right and bottom";
    c.tensor_dim[0] = 6; c.tensor_dim[1] = 3; c.tile_dim[0] = 8; c.tile_dim[1] = 4; c.stride[0] = 10; c.start = 3; cases.push_back(c);
  }
  {
    Case c; c.name = "2D load, 1-byte elements"; c.data_log2 = 0;
    c.tensor_dim[0] = 20; c.tensor_dim[1] = 5; c.tile_dim[0] = 12; c.tile_dim[1] = 5; c.stride[0] = 20; c.start = 7; c.lds_addr = 64; cases.push_back(c);
  }
  {
    Case c; c.name = "2D load, 2-byte elements"; c.data_log2 = 1;
    c.tensor_dim[0] = 9; c.tensor_dim[1] = 6; c.tile_dim[0] = 9; c.tile_dim[1] = 6; c.stride[0] = 12; c.start = 5; c.lds_addr = 128; cases.push_back(c);
  }
  {
    Case c; c.name = "2D load, 8-byte elements"; c.data_log2 = 3;
    c.tensor_dim[0] = 5; c.tensor_dim[1] = 4; c.tile_dim[0] = 6; c.tile_dim[1] = 4; c.stride[0] = 7; c.start = 2; c.lds_addr = 256; cases.push_back(c);
  }
  {
    Case c; c.name = "2D load with padding (every 4 dwords, 2 dwords of it)";
    c.tensor_dim[0] = 8; c.tensor_dim[1] = 4; c.tile_dim[0] = 8; c.tile_dim[1] = 4; c.stride[0] = 8; c.pad = true; c.pad_interval = 1; c.pad_amount = 1; cases.push_back(c);
  }
  {
    Case c; c.name = "3D load, 4-byte elements";
    c.tensor_dim[0] = 5; c.tensor_dim[1] = 4; c.tensor_dim[2] = 3; c.tile_dim[0] = 4; c.tile_dim[1] = 3; c.tile_dim[2] = 2;
    c.stride[0] = 8; c.stride[1] = 40; c.start = 1; c.groups = 4; cases.push_back(c);
  }
  {
    Case c; c.name = "5D load, 2-byte elements"; c.data_log2 = 1;
    c.tensor_dim[0] = 3; c.tensor_dim[1] = 3; c.tensor_dim[2] = 2; c.tensor_dim[3] = 2; c.tensor_dim[4] = 2;
    c.tile_dim[0] = 3; c.tile_dim[1] = 2; c.tile_dim[2] = 2; c.tile_dim[3] = 2; c.tile_dim[4] = 2;
    c.stride[0] = 4; c.stride[1] = 12; c.stride[2] = 30; c.stride[3] = 70; c.groups = 4; cases.push_back(c);
  }
  {
    Case c; c.name = "2D load, iterated three times";
    c.tensor_dim[0] = 4; c.tensor_dim[1] = 2; c.tile_dim[0] = 4; c.tile_dim[1] = 2; c.stride[0] = 4; c.iterate = true; c.iterate_count = 2;
    c.global_inc = 16; c.lds_inc = 32; c.groups = 4; cases.push_back(c);
  }
  {
    Case c; c.name = "2D gather load, 16-bit row indices (one row past the tensor)";
    c.tensor_dim[0] = 4; c.tensor_dim[1] = 8; c.tile_dim[0] = 4; c.stride[0] = 6; c.gather = true; c.wide = false; c.indices = {5, 0, 3, 9, 7}; c.groups = 4; cases.push_back(c);
  }
  {
    Case c; c.name = "2D gather load, 32-bit row indices";
    c.tensor_dim[0] = 5; c.tensor_dim[1] = 9; c.tile_dim[0] = 5; c.stride[0] = 5; c.gather = true; c.wide = true; c.indices = {8, 2, 2, 0}; c.groups = 4; cases.push_back(c);
  }
  {
    Case c; c.name = "2D store, the part outside the tensor dropped"; c.store = true;
    c.tensor_dim[0] = 6; c.tensor_dim[1] = 3; c.tile_dim[0] = 8; c.tile_dim[1] = 4; c.stride[0] = 10; c.start = 3; c.lds_addr = 16; cases.push_back(c);
  }
  {
    Case c; c.name = "2D gather store (a scatter), 16-bit row indices"; c.store = true;
    c.tensor_dim[0] = 3; c.tensor_dim[1] = 8; c.tile_dim[0] = 3; c.stride[0] = 5; c.gather = true; c.indices = {6, 1, 4}; c.lds_addr = 8; c.groups = 4; cases.push_back(c);
  }

  int failed = 0;
  for (const Case& c : cases) {
    std::vector<uint32_t> d(20, 0);
    const uint64_t elem = 1u << c.data_log2;
    const uint64_t global_addr = reinterpret_cast<uint64_t>(d_mem) + c.start * elem;
    // Group 0.
    put(d, 0, 2, 1);                       // count: a valid tensor
    put(d, 30, 1, c.wide);                 // gather index size
    put(d, 31, 1, c.gather);               // gather mode
    put(d, 32, 32, c.lds_addr);
    put(d, 64, 57, global_addr);
    put(d, 126, 2, 2);                     // type: "image"
    // Group 1 (words 4-11).
    put(d, 128 + 16, 2, c.data_log2);
    put(d, 128 + 19, 1, c.iterate);
    put(d, 128 + 20, 1, c.pad);
    put(d, 128 + 22, 3, c.pad_interval);
    put(d, 128 + 25, 7, c.pad_amount);
    put(d, 128 + 48, 32, c.tensor_dim[0]);
    put(d, 128 + 80, 32, c.tensor_dim[1]);
    put(d, 128 + 112, 16, c.tile_dim[0]);
    put(d, 128 + 128, 16, c.gather ? c.indices.size() : c.tile_dim[1]);
    put(d, 128 + 144, 16, c.tile_dim[2]);
    put(d, 128 + 160, 48, c.stride[0]);
    put(d, 128 + 208, 48, c.stride[1]);
    // Group 2 (words 12-15), group 3 (words 16-19).
    if (c.gather) {
      for (size_t i = 0; i < c.indices.size(); ++i) {
        const int slot = static_cast<int>(i);
        if (c.wide) put(d, 12 * 32 + 32 * slot, 32, c.indices[i]);   // eight 32-bit indices over groups 2 and 3
        else put(d, 12 * 32 + 16 * slot, 16, c.indices[i]);          // sixteen 16-bit ones
      }
    } else {
      put(d, 12 * 32, 32, c.tensor_dim[2]);
      if (c.iterate) {
        put(d, 12 * 32 + 32, 32, c.lds_inc);
        put(d, 12 * 32 + 64, 48, c.global_inc);
        put(d, 12 * 32 + 112, 16, c.iterate_count);
      } else {
        put(d, 12 * 32 + 32, 32, c.tensor_dim[3]);
        put(d, 12 * 32 + 64, 48, c.stride[2]);
        put(d, 12 * 32 + 112, 16, c.tile_dim[3]);
      }
      put(d, 16 * 32, 48, c.stride[3]);
      put(d, 16 * 32 + 48, 32, c.tensor_dim[4]);
      put(d, 16 * 32 + 80, 16, c.tile_dim[4]);
    }
    CHECK(hipMemcpy(d_desc, d.data(), 20 * 4, hipMemcpyHostToDevice));

    // The walk of the tile, by the ISA's pseudocode.
    std::vector<uint8_t> want_lds = lds0, want_mem = mem;
    {
      auto dim = [&](int i) { return c.tile_dim[i] ? c.tile_dim[i] : 1; };
      const int reps = c.iterate ? c.iterate_count + 1 : 1;
      uint64_t laddr = c.lds_addr, stored = 0;
      for (int it = 0; it < reps; ++it) {
        laddr = c.lds_addr + it * c.lds_inc * elem;
        stored = 0;
        const uint64_t base = c.start * elem + it * c.global_inc * elem;
        const uint64_t rows = c.gather ? c.indices.size() : dim(1);
        for (uint64_t t4 = 0; t4 < (c.gather ? 1 : dim(4)); ++t4)
          for (uint64_t t3 = 0; t3 < (c.gather ? 1 : dim(3)); ++t3)
            for (uint64_t t2 = 0; t2 < (c.gather ? 1 : dim(2)); ++t2)
              for (uint64_t r = 0; r < rows; ++r) {
                const uint64_t y = c.gather ? c.indices[r] : r;
                for (uint64_t x = 0; x < dim(0); ++x) {
                  const bool in = x < c.tensor_dim[0] && y < c.tensor_dim[1] && (t2 == 0 || t2 < c.tensor_dim[2]) &&
                                  (t3 == 0 || t3 < c.tensor_dim[3]) && (t4 == 0 || t4 < c.tensor_dim[4]);
                  const uint64_t m = base + elem * (x + y * c.stride[0] + t2 * c.stride[1] + t3 * c.stride[2] + t4 * c.stride[3]);
                  if (c.store) {
                    if (in) std::memcpy(&want_mem[m], &lds0[laddr], elem);
                  } else if (in) {
                    std::memcpy(&want_lds[laddr], &mem[m], elem);
                  } else {
                    std::memset(&want_lds[laddr], 0, elem);
                  }
                  laddr += elem;
                  stored += elem;
                  if (c.pad && !c.store && stored >= 8u << c.pad_interval) {
                    stored = 0;
                    laddr += 4u * (c.pad_amount + 1);
                  }
                }
              }
      }
    }
    CHECK(hipMemcpy(d_mem, mem.data(), kBytes, hipMemcpyHostToDevice));
    k_move<<<1, 32>>>(d_desc, d_lds0, d_lds_out, c.store, c.groups);
    CHECK(hipDeviceSynchronize());
    std::vector<uint8_t> got_lds(kLds), got_mem(kBytes);
    CHECK(hipMemcpy(got_lds.data(), d_lds_out, kLds, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(got_mem.data(), d_mem, kBytes, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int i = 0; i < kLds; ++i) wrong += got_lds[i] != want_lds[i];
    for (int i = 0; i < kBytes; ++i) wrong += got_mem[i] != want_mem[i];
    std::printf("%s: %d of %d bytes wrong\n", c.name, wrong, kLds + kBytes);
    failed += wrong != 0;
  }
  std::printf("tensor1250: %d failed\n", failed);
  return failed != 0;
}
