/* HSA's virtual memory and IPC, as a program written against ROCm's runtime
 * uses them: addresses reserved, physical memory made in a GPU's pool and
 * mapped there, its access set and read back, the memory exported as a file
 * descriptor and imported again; and an allocation shared with a child
 * process through an IPC handle, which the child reads and writes. Built
 * against VirtualGPU's HSA header or ROCm's (amd/tests/e2e/run_hsa.sh). One
 * line a check, "ok <what>" or "FAIL <what>". */
#define _DEFAULT_SOURCE
#ifdef VGPU_REAL_HSA
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#else
#include "vgpu/hsa_abi.h"
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
static void check(const char* what, int ok) {
  printf("%s %s\n", ok ? "ok" : "FAIL", what);
  fflush(stdout);
  if (!ok) ++failures;
}
#define MUST(x)                                                                  \
  do {                                                                           \
    hsa_status_t s_ = (x);                                                       \
    if (s_ != HSA_STATUS_SUCCESS) {                                              \
      const char* why_ = "?";                                                    \
      hsa_status_string(s_, &why_);                                              \
      printf("FAIL %s: %s\n", #x, why_);                                         \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

static hsa_agent_t gpu, cpu;
static int found;
static hsa_status_t agents(hsa_agent_t a, void* data) {
  (void)data;
  hsa_device_type_t type;
  MUST(hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type));
  if (type == HSA_DEVICE_TYPE_GPU && !found) gpu = a, found = 1;
  if (type == HSA_DEVICE_TYPE_CPU) cpu = a;
  return HSA_STATUS_SUCCESS;
}
static hsa_amd_memory_pool_t pool;
static int have_pool;
static hsa_status_t first_global(hsa_amd_memory_pool_t p, void* data) {
  (void)data;
  hsa_amd_segment_t segment;
  MUST(hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));
  if (segment == HSA_AMD_SEGMENT_GLOBAL && !have_pool) pool = p, have_pool = 1;
  return HSA_STATUS_SUCCESS;
}

enum { kSize = 2 << 20 };

int main(void) {
  MUST(hsa_init());
  MUST(hsa_iterate_agents(agents, NULL));
  MUST(hsa_amd_agent_iterate_memory_pools(gpu, first_global, NULL));

  /* Virtual memory: reserve, make, map, give the GPU access. */
  void* va = NULL;
  MUST(hsa_amd_vmem_address_reserve_align(&va, kSize, 0, kSize, 0));
  hsa_amd_vmem_alloc_handle_t h;
  MUST(hsa_amd_vmem_handle_create(pool, kSize, MEMORY_TYPE_PINNED, 0, &h));
  MUST(hsa_amd_vmem_map(va, kSize, 0, h, 0));
  hsa_amd_memory_access_desc_t desc = {HSA_ACCESS_PERMISSION_RW, gpu};
  MUST(hsa_amd_vmem_set_access(va, kSize, &desc, 1));
  hsa_access_permission_t perms = (hsa_access_permission_t)0;
  MUST(hsa_amd_vmem_get_access(va, &perms, gpu));
  check("reserved addresses are aligned, and the GPU reads and writes the memory mapped there",
        ((uintptr_t)va % kSize) == 0 && perms == HSA_ACCESS_PERMISSION_RW);
  uint32_t in[256], out[256];
  for (int i = 0; i < 256; ++i) in[i] = 0xC0DE0000u + (uint32_t)i;
  MUST(hsa_memory_copy(va, in, sizeof in));
  memset(out, 0, sizeof out);
  MUST(hsa_memory_copy(out, va, sizeof out));
  check("bytes copied into the mapping come back out", memcmp(in, out, sizeof in) == 0);
  hsa_amd_vmem_alloc_handle_t retained;
  MUST(hsa_amd_vmem_retain_alloc_handle(&retained, va));
  hsa_amd_memory_pool_t back_pool;
  hsa_amd_memory_type_t type;
  MUST(hsa_amd_vmem_get_alloc_properties_from_handle(h, &back_pool, &type));
  check("an address gives back the memory mapped there, made in the GPU's pool",
        retained.handle == h.handle && back_pool.handle == pool.handle && type == MEMORY_TYPE_PINNED);
  MUST(hsa_amd_vmem_handle_release(retained));
  int fd = -1;
  MUST(hsa_amd_vmem_export_shareable_handle(&fd, h, 0));
  hsa_amd_vmem_alloc_handle_t imported;
  MUST(hsa_amd_vmem_import_shareable_handle(fd, &imported));
  void* va2 = NULL;
  MUST(hsa_amd_vmem_address_reserve(&va2, kSize, 0, 0));
  MUST(hsa_amd_vmem_map(va2, kSize, 0, imported, 0));
  MUST(hsa_amd_vmem_set_access(va2, kSize, &desc, 1));
  memset(out, 0, sizeof out);
  MUST(hsa_memory_copy(out, va2, sizeof out));
  check("memory exported as a file descriptor and imported is the same memory", memcmp(in, out, sizeof in) == 0);
  close(fd);
  MUST(hsa_amd_vmem_unmap(va2, kSize));
  MUST(hsa_amd_vmem_handle_release(imported));
  MUST(hsa_amd_vmem_address_free(va2, kSize));
  MUST(hsa_amd_vmem_unmap(va, kSize));
  MUST(hsa_amd_vmem_handle_release(h));
  MUST(hsa_amd_vmem_address_free(va, kSize));

  /* IPC: a child process attaches to the parent's allocation, reads what the
   * parent wrote, and writes back. */
  void* buf = NULL;
  MUST(hsa_amd_memory_pool_allocate(pool, 4096, 0, &buf));
  MUST(hsa_memory_copy(buf, in, sizeof in));
  hsa_amd_ipc_memory_t ipc;
  MUST(hsa_amd_ipc_memory_create(buf, 4096, &ipc));
  fflush(stdout);
  const pid_t child = fork();
  if (child == 0) {
    void* mapped = NULL;
    MUST(hsa_amd_ipc_memory_attach(&ipc, 4096, 1, &gpu, &mapped));
    uint32_t seen[256];
    MUST(hsa_memory_copy(seen, mapped, sizeof seen));
    int same = memcmp(seen, in, sizeof in) == 0;
    const uint32_t mark = 0x51DE51DE;
    MUST(hsa_memory_copy((char*)mapped + 2048, &mark, sizeof mark));
    MUST(hsa_amd_ipc_memory_detach(mapped));
    _exit(same ? 0 : 3);
  }
  int status = 0;
  waitpid(child, &status, 0);
  uint32_t mark = 0;
  MUST(hsa_memory_copy(&mark, (char*)buf + 2048, sizeof mark));
  check("another process attaches to an IPC handle, reads the memory and writes it",
        WIFEXITED(status) && WEXITSTATUS(status) == 0 && mark == 0x51DE51DE);
  hsa_amd_ipc_memory_t bad;
  memset(&bad, 0, sizeof bad);
  void* nothing = NULL;
  check("a handle no process made is refused",
        hsa_amd_ipc_memory_attach(&bad, 4096, 1, &gpu, &nothing) != HSA_STATUS_SUCCESS);
  MUST(hsa_amd_memory_pool_free(buf));

  /* SVM: host pages given to the GPU, advised, queried and prefetched. */
  bool svm = false;
  MUST(hsa_system_get_info((hsa_system_info_t)HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED, &svm));
  const size_t page = 4096, pages = 8;
  char* host = (char*)mmap(NULL, pages * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  hsa_amd_svm_attribute_pair_t set[] = {{HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, gpu.handle},
                                        {HSA_AMD_SVM_ATTRIB_READ_MOSTLY, 1},
                                        {HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, gpu.handle}};
  MUST(hsa_amd_svm_attributes_set(host, 4 * page, set, 3));
  hsa_amd_svm_attribute_pair_t get[] = {{HSA_AMD_SVM_ATTRIB_READ_MOSTLY, 0},
                                        {HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, 0},
                                        {HSA_AMD_SVM_ATTRIB_ACCESS_QUERY, gpu.handle},
                                        {HSA_AMD_SVM_ATTRIB_ACCESS_QUERY, cpu.handle}};
  MUST(hsa_amd_svm_attributes_get(host, 4 * page, get, 4));
  check("the runtime has SVM, and a range's advice and access read back as they were set",
        svm && get[0].value == 1 && get[1].value == gpu.handle && get[2].attribute == HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE &&
            get[2].value == gpu.handle && get[3].attribute == HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS);
  hsa_amd_svm_attribute_pair_t mixed[] = {{HSA_AMD_SVM_ATTRIB_READ_MOSTLY, 7},
                                          {HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, 7}};
  MUST(hsa_amd_svm_attributes_get(host + 2 * page, 4 * page, mixed, 2));
  check("a range advised only in part reads as not uniform", mixed[0].value == 0 && mixed[1].value == 0);
  hsa_signal_t done;
  MUST(hsa_signal_create(1, 0, NULL, &done));
  /* Not page-aligned: the driver keeps whole pages, so every page it touches goes. */
  MUST(hsa_amd_svm_prefetch_async(host + 100, pages * page - 200, gpu, 0, NULL, done));
  const hsa_signal_value_t left =
      hsa_signal_wait_scacquire(done, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  hsa_amd_svm_attribute_pair_t where = {HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION, 0};
  MUST(hsa_amd_svm_attributes_get(host, pages * page, &where, 1));
  check("a prefetch completes its signal, and each page it touched says where it went", left == 0 && where.value == gpu.handle);
  host[3] = 42;   /* host pages given to the GPU stay the host's */
  uint32_t seen = 0;
  MUST(hsa_memory_copy(&seen, host, 4));
  check("host pages given to a GPU are still the host's", (seen >> 24) == 42);
  MUST(hsa_signal_destroy(done));
  munmap(host, pages * page);
  MUST(hsa_shut_down());
  return failures ? 1 : 0;
}
