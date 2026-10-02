// The library half of nvjitlink_paths: add_base(x) = twice(x) * scale + base
// with a file-scope twice (2x + 1) and scale (5), so 10x + 12. Built with
// nvcc -dc into a host object, and archived into a static library, by
// run_nvjitlink.sh.
__device__ int base = 7;
static __device__ int scale = 5;
static __device__ __noinline__ int twice(int x) { return 2 * x + 1; }
extern "C" __device__ __noinline__ int add_base(int x) { return twice(x) * scale + base; }
