// The other half of device_heap.cu: the same heap kernels in a module of their
// own, which run_device_heap.sh compiles to PTX only. The main program's
// kernels run as SASS, so these run on the other engine -- and a block one of
// them allocates is the other's to free, because a device has one heap.
#include <cstdint>

__global__ void peer_keep(void** slot, size_t bytes, unsigned char mark) {
  unsigned char* p = static_cast<unsigned char*>(malloc(bytes));
  if (p) {
    p[0] = mark;
    p[bytes - 1] = mark;
  }
  *slot = p;
}

__global__ void peer_check_free(void** slot, size_t bytes, int* seen) {
  unsigned char* p = static_cast<unsigned char*>(*slot);
  *seen = p ? p[0] | p[bytes - 1] << 8 : -1;
  free(p);
  *slot = nullptr;
}

__global__ void peer_try(size_t bytes, int* ok) {
  void* p = malloc(bytes);
  *ok = p != nullptr;
  free(p);
}

// Launched from the main program, which cannot name these kernels itself.
void launch_peer_keep(void** slot, size_t bytes, unsigned char mark) { peer_keep<<<1, 1>>>(slot, bytes, mark); }
void launch_peer_check_free(void** slot, size_t bytes, int* seen) { peer_check_free<<<1, 1>>>(slot, bytes, seen); }
void launch_peer_try(size_t bytes, int* ok) { peer_try<<<1, 1>>>(bytes, ok); }
