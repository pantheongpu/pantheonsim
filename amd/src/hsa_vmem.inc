// HSA's virtual memory (hsa_amd_vmem_*) and memory shared between processes
// (hsa_amd_ipc_memory_*). Included by hsa_api.cpp.
//
// Both are what HIP's own calls already model (hip_api.cpp): a reservation of
// addresses, physical memory made in a pool and mapped into it, its access,
// and its export as a file descriptor; and an allocation moved into a file
// another process maps. These calls are HIP's, reached through the library
// both runtimes live in, so memory made either way is one memory -- ROCm's
// HIP over this runtime makes its virtual memory and IPC handles here.

extern "C" {
// HIP's calls, by the hidden names hip_api.cpp gives them (hipError_t is an
// int's size): never the exported ones, which may be ROCm's own HIP's.
int vgpu_own_hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags);
int vgpu_own_hipMemAddressFree(void* ptr, size_t size);
int vgpu_own_hipMemCreate(void** handle, size_t size, const vgpu::amd::abi::MemAllocationProp* prop, unsigned long long flags);
int vgpu_own_hipMemRelease(void* handle);
int vgpu_own_hipMemMap(void* ptr, size_t size, size_t offset, void* handle, unsigned long long flags);
int vgpu_own_hipMemUnmap(void* ptr, size_t size);
int vgpu_own_hipMemSetAccess(void* ptr, size_t size, const vgpu::amd::abi::MemAccessDesc* desc, size_t count);
int vgpu_own_hipMemGetAccess(unsigned long long* flags, const vgpu::amd::abi::MemLocation* location, void* ptr);
int vgpu_own_hipMemExportToShareableHandle(void* out, void* handle, int type, unsigned long long flags);
int vgpu_own_hipMemImportFromShareableHandle(void** handle, void* os_handle, int type);
int vgpu_own_hipMemRetainAllocationHandle(void** handle, void* addr);
int vgpu_own_hipMemGetAllocationPropertiesFromHandle(vgpu::amd::abi::MemAllocationProp* prop, void* handle);
}

namespace {
// What HIP said, as HSA says it.
hsa_status_t from_hip(int e) {
  switch (e) {
    case 0: return HSA_STATUS_SUCCESS;
    case 1: return HSA_STATUS_ERROR_INVALID_ARGUMENT;    // hipErrorInvalidValue
    case 2: return HSA_STATUS_ERROR_OUT_OF_RESOURCES;    // hipErrorOutOfMemory
    case 101: return HSA_STATUS_ERROR_INVALID_AGENT;     // hipErrorInvalidDevice
    default: return HSA_STATUS_ERROR;
  }
}
constexpr int kFileDescriptor = 1;   // hipMemHandleTypePosixFileDescriptor
constexpr int kLocationDevice = 1, kLocationHost = 2;
}  // namespace

hsa_status_t hsa_amd_vmem_address_reserve_align(void** va, size_t size, uint64_t address, uint64_t alignment,
                                                uint64_t) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!va || !size) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return from_hip(vgpu_own_hipMemAddressReserve(va, size, alignment, reinterpret_cast<void*>(address), 0));
}
hsa_status_t hsa_amd_vmem_address_reserve(void** va, size_t size, uint64_t address, uint64_t flags) {
  return hsa_amd_vmem_address_reserve_align(va, size, address, 0, flags);
}
hsa_status_t hsa_amd_vmem_address_free(void* va, size_t size) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return from_hip(vgpu_own_hipMemAddressFree(va, size));
}

// Physical memory in a pool: a GPU's memory, or the host's (either system
// pool), made exportable as a file descriptor as ROCm's runtime makes it.
hsa_status_t hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size, hsa_amd_memory_type_t,
                                        uint64_t, hsa_amd_vmem_alloc_handle_t* handle) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  PoolId id;
  if (!pool_of(pool.handle, &id)) return HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  if (!handle || !size || id.kind == PoolKind::Group) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  vgpu::amd::abi::MemAllocationProp prop{};
  prop.type = 1;   // hipMemAllocationTypePinned
  prop.requestedHandleType = kFileDescriptor;
  prop.location.type = id.kind == PoolKind::Device ? kLocationDevice : kLocationHost;
  prop.location.id = id.kind == PoolKind::Device ? gpu_of(id.agent) : 0;
  void* h = nullptr;
  if (const hsa_status_t st = from_hip(vgpu_own_hipMemCreate(&h, size, &prop, 0)); st != HSA_STATUS_SUCCESS) return st;
  handle->handle = reinterpret_cast<uint64_t>(h);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t handle) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return from_hip(vgpu_own_hipMemRelease(reinterpret_cast<void*>(handle.handle)));
}
hsa_status_t hsa_amd_vmem_get_alloc_properties_from_handle(hsa_amd_vmem_alloc_handle_t handle,
                                                           hsa_amd_memory_pool_t* pool, hsa_amd_memory_type_t* type) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!pool || !type) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  vgpu::amd::abi::MemAllocationProp prop{};
  if (const hsa_status_t st = from_hip(vgpu_own_hipMemGetAllocationPropertiesFromHandle(&prop, reinterpret_cast<void*>(handle.handle)));
      st != HSA_STATUS_SUCCESS)
    return st;
  const hsa_agent_t agent = prop.location.type == kLocationDevice ? gpu_agent(prop.location.id) : hsa_agent_t{kCpuAgent};
  pool->handle = pools_of(agent)[0];
  *type = MEMORY_TYPE_PINNED;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_vmem_map(void* va, size_t size, size_t offset, hsa_amd_vmem_alloc_handle_t handle, uint64_t) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return from_hip(vgpu_own_hipMemMap(va, size, offset, reinterpret_cast<void*>(handle.handle), 0));
}
hsa_status_t hsa_amd_vmem_unmap(void* va, size_t size) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return from_hip(vgpu_own_hipMemUnmap(va, size));
}

// Access, agent by agent: read-only is read, write-only or read-write is
// read-write. The host's access to mapped memory is not HIP's to set; it is
// taken as given.
hsa_status_t hsa_amd_vmem_set_access(void* va, size_t size, const hsa_amd_memory_access_desc_t* desc, size_t count) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!desc && count) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::vector<vgpu::amd::abi::MemAccessDesc> hip;
  for (size_t i = 0; i < count; ++i) {
    if (!valid_agent(desc[i].agent_handle)) return HSA_STATUS_ERROR_INVALID_AGENT;
    const int gpu = gpu_of(desc[i].agent_handle);
    if (gpu < 0) continue;
    const int p = static_cast<int>(desc[i].permissions);
    hip.push_back({{kLocationDevice, gpu}, p == 0 ? 0 : p == 1 ? 1 : 3});
  }
  if (hip.empty()) return HSA_STATUS_SUCCESS;
  return from_hip(vgpu_own_hipMemSetAccess(va, size, hip.data(), hip.size()));
}
hsa_status_t hsa_amd_vmem_get_access(void* va, hsa_access_permission_t* perms, hsa_agent_t agent) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!perms) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  const int gpu = gpu_of(agent);
  if (gpu < 0) {
    *perms = HSA_ACCESS_PERMISSION_RW;
    return HSA_STATUS_SUCCESS;
  }
  unsigned long long flags = 0;
  const vgpu::amd::abi::MemLocation where{kLocationDevice, gpu};
  if (const hsa_status_t st = from_hip(vgpu_own_hipMemGetAccess(&flags, &where, va)); st != HSA_STATUS_SUCCESS) return st;
  *perms = static_cast<hsa_access_permission_t>(flags == 1 ? HSA_ACCESS_PERMISSION_RO
                                                : flags == 3 ? HSA_ACCESS_PERMISSION_RW
                                                             : 0);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_amd_vmem_export_shareable_handle(int* fd, hsa_amd_vmem_alloc_handle_t handle, uint64_t) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!fd) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return from_hip(vgpu_own_hipMemExportToShareableHandle(fd, reinterpret_cast<void*>(handle.handle), kFileDescriptor, 0));
}
hsa_status_t hsa_amd_vmem_import_shareable_handle(int fd, hsa_amd_vmem_alloc_handle_t* handle) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!handle || fd < 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  void* h = nullptr;
  if (const hsa_status_t st =
          from_hip(vgpu_own_hipMemImportFromShareableHandle(&h, reinterpret_cast<void*>(static_cast<intptr_t>(fd)), kFileDescriptor));
      st != HSA_STATUS_SUCCESS)
    return st;
  handle->handle = reinterpret_cast<uint64_t>(h);
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_vmem_retain_alloc_handle(hsa_amd_vmem_alloc_handle_t* handle, void* addr) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!handle) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  void* h = nullptr;
  if (const hsa_status_t st = from_hip(vgpu_own_hipMemRetainAllocationHandle(&h, addr)); st != HSA_STATUS_SUCCESS) return st;
  handle->handle = reinterpret_cast<uint64_t>(h);
  return HSA_STATUS_SUCCESS;
}

// ---- IPC ---------------------------------------------------------------------------
//
// A handle is 32 bytes: which process shared the memory and its number for
// the export (which name the file the bytes are in), the exporter's device,
// and the size.
namespace {
constexpr uint32_t kHsaIpcMagic = 0x48534149;   // "HSAI"
}  // namespace

hsa_status_t hsa_amd_ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || !len || !handle) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  int device = 0;
  uint64_t size = 0;
  uint32_t serial = 0;
  std::string why;
  if (!shared::ipc_share(ptr, &device, &size, &serial, &why)) return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, why);
  if (len > size) return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, "the length runs past the allocation");
  std::memset(handle, 0, sizeof *handle);
  handle->handle[0] = kHsaIpcMagic;
  handle->handle[1] = static_cast<uint32_t>(::getpid());
  handle->handle[2] = serial;
  handle->handle[3] = static_cast<uint32_t>(device);
  handle->handle[4] = static_cast<uint32_t>(size);
  handle->handle[5] = static_cast<uint32_t>(size >> 32);
  return HSA_STATUS_SUCCESS;
}
// Mapped on the first GPU named, or the exporter's device where none is.
hsa_status_t hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents,
                                       const hsa_agent_t* agents, void** mapped) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!handle || !mapped || (num_agents && !agents) || handle->handle[0] != kHsaIpcMagic)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const uint64_t size = handle->handle[4] | uint64_t{handle->handle[5]} << 32;
  if (len > size) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  int device = static_cast<int>(handle->handle[3]);
  for (uint32_t i = 0; i < num_agents; ++i) {
    if (!valid_agent(agents[i])) return HSA_STATUS_ERROR_INVALID_AGENT;
    if (gpu_of(agents[i]) >= 0) {
      device = gpu_of(agents[i]);
      break;
    }
  }
  std::string why;
  void* p = shared::ipc_attach(handle->handle[1], handle->handle[2], size, device, &why);
  if (!p) return fail(HSA_STATUS_ERROR_INVALID_ARGUMENT, why);
  *mapped = p;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_ipc_memory_detach(void* mapped) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return shared::ipc_detach(mapped) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

// ---- Device memory as a file descriptor -------------------------------------------
//
// A card's dma-buf; here the file the allocation's bytes move into, which
// another process (or this one) maps -- hipMemGetHandleForAddressRange's.
hsa_status_t hsa_amd_portable_export_dmabuf_v2(const void* ptr, size_t size, int* dmabuf, uint64_t* offset,
                                               uint64_t) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || !size || !dmabuf || !offset) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  int fd = -1;
  uint64_t at = 0;
  std::string why;
  if (!shared::export_dmabuf(ptr, size, &fd, &at, &why)) return fail(HSA_STATUS_ERROR_INVALID_ALLOCATION, why);
  *dmabuf = fd;
  *offset = at;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t hsa_amd_portable_export_dmabuf(const void* ptr, size_t size, int* dmabuf, uint64_t* offset) {
  return hsa_amd_portable_export_dmabuf_v2(ptr, size, dmabuf, offset, 0);
}
hsa_status_t hsa_amd_portable_close_dmabuf(int dmabuf) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  return ::close(dmabuf) == 0 ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_INVALID_ARGUMENT;
}
