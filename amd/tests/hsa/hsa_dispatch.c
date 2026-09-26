/* An HSA program, as one is written against ROCm's runtime: it finds the CPU
 * and GPU agents and their memory pools, loads a code object into an
 * executable, and dispatches kernels by writing AQL packets into a queue and
 * ringing its doorbell. Checked on VirtualGPU's libhsa-runtime64, and built
 * against its header or, where ROCm is installed, ROCm's own
 * (amd/tests/e2e/run_hsa.sh), which is what says the two agree.
 *
 *   hsa_dispatch <vector_add.gfx942.hsaco>
 *
 * The kernels are amd/tests/data/vector_add.c's: vector_add, 256 work-items
 * a group, and reduce_sum, which reduces each group's 256 elements in LDS.
 * One line a check, "ok <what>" or "FAIL <what>". */
#ifdef VGPU_REAL_HSA
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#else
#include "vgpu/hsa_abi.h"
#endif

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
static void check(const char* what, int ok) {
  printf("%s %s\n", ok ? "ok" : "FAIL", what);
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

/* ---- Finding the agents and their pools ---------------------------------- */

typedef struct {
  hsa_agent_t cpu, gpu[8];
  int cpus, gpus;
} Agents;

static hsa_status_t each_agent(hsa_agent_t a, void* data) {
  Agents* all = (Agents*)data;
  hsa_device_type_t type;
  MUST(hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type));
  if (type == HSA_DEVICE_TYPE_CPU) all->cpu = a, all->cpus++;
  else if (type == HSA_DEVICE_TYPE_GPU && all->gpus < 8) all->gpu[all->gpus++] = a;
  return HSA_STATUS_SUCCESS;
}

typedef struct {
  hsa_amd_memory_pool_t kernarg, coarse;
  int found_kernarg, found_coarse;
} Pools;

static hsa_status_t each_pool(hsa_amd_memory_pool_t p, void* data) {
  Pools* pools = (Pools*)data;
  uint32_t flags = 0;
  bool alloc = false;
  MUST(hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags));
  MUST(hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc));
  if (!alloc) return HSA_STATUS_SUCCESS;
  if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) && !pools->found_kernarg)
    pools->kernarg = p, pools->found_kernarg = 1;
  if ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) && !pools->found_coarse)
    pools->coarse = p, pools->found_coarse = 1;
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_kernarg_region(hsa_region_t r, void* data) {
  uint32_t flags = 0;
  MUST(hsa_region_get_info(r, HSA_REGION_INFO_GLOBAL_FLAGS, &flags));
  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    *(hsa_region_t*)data = r;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t first_isa(hsa_isa_t isa, void* data) {
  *(hsa_isa_t*)data = isa;
  return HSA_STATUS_INFO_BREAK;
}

/* ---- Kernels ------------------------------------------------------------- */

typedef struct {
  uint64_t object;
  uint32_t kernarg_size, group_segment, private_segment;
} Kernel;

static Kernel kernel(hsa_executable_t exe, hsa_agent_t gpu, const char* name) {
  hsa_executable_symbol_t sym;
  MUST(hsa_executable_get_symbol_by_name(exe, name, &gpu, &sym));
  Kernel k;
  MUST(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &k.object));
  MUST(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &k.kernarg_size));
  MUST(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &k.group_segment));
  MUST(hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                      &k.private_segment));
  return k;
}

/* Writes a kernel dispatch packet into the queue's next slot and rings the
 * doorbell: the body first, then the header, atomically, which is what hands
 * the packet to the packet processor. */
static void dispatch(hsa_queue_t* q, const Kernel* k, void* kernarg, uint32_t grid, uint16_t group,
                     hsa_signal_t done) {
  const uint64_t index = hsa_queue_add_write_index_relaxed(q, 1);
  hsa_kernel_dispatch_packet_t* p =
      (hsa_kernel_dispatch_packet_t*)q->base_address + (index & (q->size - 1));
  memset((char*)p + 4, 0, sizeof *p - 4);
  p->workgroup_size_x = group;
  p->workgroup_size_y = p->workgroup_size_z = 1;
  p->grid_size_x = grid;
  p->grid_size_y = p->grid_size_z = 1;
  p->kernel_object = k->object;
  p->kernarg_address = kernarg;
  p->private_segment_size = k->private_segment;
  p->group_segment_size = k->group_segment;
  p->completion_signal = done;
  const uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                          (1 << HSA_PACKET_HEADER_BARRIER) |
                          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  const uint32_t word = header | (1u << 16); /* setup: one dimension */
  __atomic_store_n((uint32_t*)p, word, __ATOMIC_RELEASE);
  hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)index);
}

static void barrier_and(hsa_queue_t* q, hsa_signal_t dep, hsa_signal_t done) {
  const uint64_t index = hsa_queue_add_write_index_relaxed(q, 1);
  hsa_barrier_and_packet_t* p = (hsa_barrier_and_packet_t*)q->base_address + (index & (q->size - 1));
  memset((char*)p + 2, 0, sizeof *p - 2);
  p->dep_signal[0] = dep;
  p->completion_signal = done;
  __atomic_store_n(&p->header, (uint16_t)((HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
                                          (1 << HSA_PACKET_HEADER_BARRIER)),
                   __ATOMIC_RELEASE);
  hsa_signal_store_screlease(q->doorbell_signal, (hsa_signal_value_t)index);
}

static hsa_signal_value_t wait_zero(hsa_signal_t s) {
  return hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
}

static int queue_errors;
static hsa_status_t queue_error;
static void on_queue_error(hsa_status_t status, hsa_queue_t* q, void* data) {
  (void)q, (void)data;
  queue_error = status;
  ++queue_errors;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s vector_add.gfx942.hsaco\n", argv[0]);
    return 2;
  }
  MUST(hsa_init());

  /* 1. The agents: a CPU, and the GPUs, each saying what it is. */
  Agents agents = {0};
  MUST(hsa_iterate_agents(each_agent, &agents));
  hsa_agent_t gpu = agents.gpu[0];
  char name[64] = {0}, product[64] = {0}, isa_name[64] = {0};
  uint32_t wave = 0, cus = 0, queue_max = 0, feature = 0;
  MUST(hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, name));
  MUST(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, product));
  MUST(hsa_agent_get_info(gpu, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wave));
  MUST(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cus));
  MUST(hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max));
  MUST(hsa_agent_get_info(gpu, HSA_AGENT_INFO_FEATURE, &feature));
  hsa_isa_t isa;
  MUST(hsa_agent_iterate_isas(gpu, first_isa, &isa) == HSA_STATUS_INFO_BREAK ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR);
  MUST(hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name));
  check("one CPU agent and the GPU agents", agents.cpus == 1 && agents.gpus >= 1);
  check("a GPU agent names its processor, product and shape",
        strcmp(name, "gfx942") == 0 && strstr(product, "MI300X") && wave == 64 && cus == 304 &&
            queue_max >= 4096 && (feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH));
  check("its ISA is the full target name", strncmp(isa_name, "amdgcn-amd-amdhsa--gfx942", 25) == 0);
  const char* text = NULL;
  check("a status says what it means",
        hsa_status_string(HSA_STATUS_ERROR_INVALID_AGENT, &text) == HSA_STATUS_SUCCESS && strstr(text, "agent"));

  /* 2. Memory: kernel arguments from the CPU's kernarg pool, the GPU's own
   * memory from its coarse-grained pool. */
  Pools cpu_pools = {0}, gpu_pools = {0};
  MUST(hsa_amd_agent_iterate_memory_pools(agents.cpu, each_pool, &cpu_pools));
  MUST(hsa_amd_agent_iterate_memory_pools(gpu, each_pool, &gpu_pools));
  hsa_region_t kernarg_region = {0};
  hsa_agent_iterate_regions(agents.cpu, find_kernarg_region, &kernarg_region);
  check("the CPU has a kernarg pool and a region for kernel arguments",
        cpu_pools.found_kernarg && kernarg_region.handle != 0);
  check("the GPU has a pool of its own memory", gpu_pools.found_coarse);
  uint32_t access = 99;
  MUST(hsa_amd_agent_memory_pool_get_info(agents.cpu, gpu_pools.coarse, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                          &access));
  check("the CPU is not given the GPU's memory by default", access == HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT);

  /* 3. The code object, loaded from a file into an executable. */
  const int fd = open(argv[1], O_RDONLY);
  if (fd < 0) {
    printf("FAIL no code object at %s\n", argv[1]);
    return 1;
  }
  hsa_code_object_reader_t reader;
  MUST(hsa_code_object_reader_create_from_file(fd, &reader));
  hsa_executable_t exe;
  MUST(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &exe));
  MUST(hsa_executable_load_agent_code_object(exe, gpu, reader, NULL, NULL));
  MUST(hsa_executable_freeze(exe, NULL));
  const Kernel add = kernel(exe, gpu, "vector_add.kd"), sum = kernel(exe, gpu, "reduce_sum.kd");
  hsa_executable_symbol_t missing;
  check("an executable finds its kernels, and not what it lacks",
        add.object && sum.object && add.object != sum.object && add.kernarg_size >= 28 &&
            sum.group_segment >= 1024 &&
            hsa_executable_get_symbol_by_name(exe, "no_such_kernel.kd", &gpu, &missing) != HSA_STATUS_SUCCESS);

  /* 4. vector_add through a queue: inputs in system memory the GPU reads
   * where it is, the output in the GPU's memory, copied back. The kernarg
   * segment is the size the kernel declares: the runtime's hidden arguments
   * follow the program's. */
  enum { N = 4096 };
  float *a, *b, *host_out = malloc(N * sizeof(float));
  void* out;
  MUST(hsa_amd_memory_pool_allocate(cpu_pools.kernarg, N * sizeof(float), 0, (void**)&a));
  MUST(hsa_amd_memory_pool_allocate(cpu_pools.kernarg, N * sizeof(float), 0, (void**)&b));
  MUST(hsa_amd_memory_pool_allocate(gpu_pools.coarse, N * sizeof(float), 0, &out));
  for (int i = 0; i < N; ++i) a[i] = (float)i, b[i] = 0.5f * (float)i;
  hsa_agent_t both[2] = {agents.cpu, gpu};
  MUST(hsa_amd_agents_allow_access(2, both, NULL, out));
  struct {
    const float *a, *b;
    void* out;
    int n;
  } __attribute__((aligned(16))) args = {a, b, out, N};
  void* kernarg;
  MUST(hsa_amd_memory_pool_allocate(cpu_pools.kernarg, add.kernarg_size, 0, &kernarg));
  memcpy(kernarg, &args, sizeof args);
  hsa_queue_t* q;
  MUST(hsa_queue_create(gpu, 256, HSA_QUEUE_TYPE_SINGLE, on_queue_error, NULL, UINT32_MAX, UINT32_MAX, &q));
  hsa_signal_t done;
  MUST(hsa_signal_create(1, 0, NULL, &done));
  dispatch(q, &add, kernarg, N, 256, done);
  const hsa_signal_value_t left = wait_zero(done);
  MUST(hsa_memory_copy(host_out, out, N * sizeof(float)));
  int right = left == 0;
  for (int i = 0; i < N; ++i) right = right && host_out[i] == 1.5f * (float)i;
  check("a kernel dispatched through a queue adds two vectors", right);
  check("the packet processor moved past the packet",
        hsa_queue_load_read_index_scacquire(q) == 1 && hsa_queue_load_write_index_relaxed(q) == 1);

  /* 5. reduce_sum, with the LDS the kernel reserves: each group of 256 sums
   * its part. */
  float *in, *partial;
  MUST(hsa_amd_memory_pool_allocate(cpu_pools.kernarg, N * sizeof(float), 0, (void**)&in));
  MUST(hsa_amd_memory_pool_allocate(cpu_pools.kernarg, (N / 256) * sizeof(float), 0, (void**)&partial));
  for (int i = 0; i < N; ++i) in[i] = (float)(i % 7);
  struct {
    float *in, *out;
    int n;
    float scale;
  } __attribute__((aligned(16))) sargs = {in, partial, N, 2.0f};
  void* skernarg;
  MUST(hsa_amd_memory_pool_allocate(cpu_pools.kernarg, sum.kernarg_size, 0, &skernarg));
  memcpy(skernarg, &sargs, sizeof sargs);
  hsa_signal_store_relaxed(done, 1);
  dispatch(q, &sum, skernarg, N, 256, done);
  wait_zero(done);
  right = 1;
  for (int g = 0; g < N / 256; ++g) {
    float want = 0;
    for (int i = g * 256; i < (g + 1) * 256; ++i) want += 2.0f * (float)(i % 7);
    right = right && partial[g] == want;
  }
  check("a kernel reduces through the LDS it reserves", right);

  /* 6. Ordering: a barrier packet holds the queue until the host sets a
   * signal, and an asynchronous copy waits for the dispatch behind it. */
  hsa_signal_t gate, copied;
  MUST(hsa_signal_create(1, 0, NULL, &gate));
  MUST(hsa_signal_create(1, 0, NULL, &copied));
  hsa_signal_store_relaxed(done, 1);
  memset(host_out, 0, N * sizeof(float));
  for (int i = 0; i < N; ++i) a[i] = 2.0f;
  barrier_and(q, gate, (hsa_signal_t){0});
  dispatch(q, &add, kernarg, N, 256, done);
  MUST(hsa_amd_memory_async_copy(host_out, agents.cpu, out, gpu, N * sizeof(float), 1, &done, copied));
  const hsa_signal_value_t held = hsa_signal_wait_scacquire(done, HSA_SIGNAL_CONDITION_LT, 1, 50000000,
                                                            HSA_WAIT_STATE_BLOCKED);
  hsa_signal_store_screlease(gate, 0);
  wait_zero(copied);
  right = held == 1 && hsa_signal_load_relaxed(done) == 0;
  for (int i = 0; i < N; ++i) right = right && host_out[i] == 2.0f + 0.5f * (float)i;
  check("a barrier holds the queue, and a copy waits for its signal", right);

  /* 7. A grid that is not a whole number of work-groups: the last group has
   * only what is left, so exactly the first 1000 elements are written. */
  MUST(hsa_amd_memory_fill(out, 0, N));
  hsa_signal_store_relaxed(done, 1);
  dispatch(q, &add, kernarg, 1000, 256, done);
  wait_zero(done);
  MUST(hsa_memory_copy(host_out, out, N * sizeof(float)));
  right = queue_errors == 0;
  for (int i = 0; i < N; ++i) right = right && host_out[i] == (i < 1000 ? 2.0f + 0.5f * (float)i : 0.0f);
  check("a grid that is not a whole number of work-groups runs its last one short", right);

  /* And what the packet processor refuses goes to the queue's callback: a
   * kernel object no executable loaded. */
  const Kernel bogus = {add.object + 4096, add.kernarg_size, 0, 0};
  hsa_signal_store_relaxed(done, 1);
  dispatch(q, &bogus, kernarg, N, 256, done);
  wait_zero(done);
  check("a packet the processor refuses is told to the queue's callback",
        queue_errors == 1 && queue_error == HSA_STATUS_ERROR_INVALID_PACKET_FORMAT);

  /* 8. Signals: the arithmetic, and a wait that times out. */
  hsa_signal_t s;
  MUST(hsa_signal_create(10, 0, NULL, &s));
  hsa_signal_add_relaxed(s, 5);
  hsa_signal_subtract_scacq_screl(s, 3);
  const hsa_signal_value_t was = hsa_signal_cas_scacq_screl(s, 12, 40);
  const hsa_signal_value_t timed = hsa_signal_wait_relaxed(s, HSA_SIGNAL_CONDITION_EQ, 0, 1000000,
                                                           HSA_WAIT_STATE_BLOCKED);
  check("a signal adds, subtracts, compares and swaps, and a wait times out",
        was == 12 && hsa_signal_load_relaxed(s) == 40 && timed == 40);

  /* 9. Each GPU dispatches on its own queue, when there is more than one. */
  if (agents.gpus > 1) {
    hsa_agent_t gpu1 = agents.gpu[1];
    hsa_executable_t exe1;
    MUST(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL, &exe1));
    MUST(hsa_executable_load_agent_code_object(exe1, gpu1, reader, NULL, NULL));
    MUST(hsa_executable_freeze(exe1, NULL));
    const Kernel add1 = kernel(exe1, gpu1, "vector_add.kd");
    Pools gpu1_pools = {0};
    MUST(hsa_amd_agent_iterate_memory_pools(gpu1, each_pool, &gpu1_pools));
    void* out1;
    MUST(hsa_amd_memory_pool_allocate(gpu1_pools.coarse, N * sizeof(float), 0, &out1));
    void* kernarg1;
    MUST(hsa_amd_memory_pool_allocate(cpu_pools.kernarg, add1.kernarg_size, 0, &kernarg1));
    args.out = out1;
    memcpy(kernarg1, &args, sizeof args);
    hsa_queue_t* q1;
    MUST(hsa_queue_create(gpu1, 64, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &q1));
    hsa_signal_store_relaxed(done, 1);
    dispatch(q1, &add1, kernarg1, N, 256, done);
    wait_zero(done);
    MUST(hsa_memory_copy(host_out, out1, N * sizeof(float)));
    right = 1;
    for (int i = 0; i < N; ++i) right = right && host_out[i] == 2.0f + 0.5f * (float)i;
    check("a second GPU runs its own queue", right && add1.object != add.object);
    MUST(hsa_queue_destroy(q1));
    MUST(hsa_executable_destroy(exe1));
    MUST(hsa_amd_memory_pool_free(out1));
    MUST(hsa_amd_memory_pool_free(kernarg1));
  }

  MUST(hsa_queue_destroy(q));
  MUST(hsa_signal_destroy(done));
  MUST(hsa_signal_destroy(gate));
  MUST(hsa_signal_destroy(copied));
  MUST(hsa_signal_destroy(s));
  MUST(hsa_executable_destroy(exe));
  MUST(hsa_code_object_reader_destroy(reader));
  close(fd);
  MUST(hsa_amd_memory_pool_free(a));
  MUST(hsa_amd_memory_pool_free(b));
  MUST(hsa_amd_memory_pool_free(out));
  MUST(hsa_amd_memory_pool_free(kernarg));
  MUST(hsa_amd_memory_pool_free(skernarg));
  MUST(hsa_amd_memory_pool_free(in));
  MUST(hsa_amd_memory_pool_free(partial));
  free(host_out);
  MUST(hsa_shut_down());
  return failures ? 1 : 0;
}
