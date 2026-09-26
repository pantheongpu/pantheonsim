/* The HSA runtime's C interface, as far as VirtualGPU's libhsa-runtime64
 * implements it: the part of the HSA Foundation's Runtime Programmer's
 * Reference Manual (1.2) and of AMD's documented extensions a program uses to
 * find the GPUs, allocate memory, load a code object and dispatch kernels
 * through an AQL queue.
 *
 * Declared here from the published specification, with the values and
 * layouts a program built against ROCm's own headers passes: the layouts are
 * checked against ROCm's hsa.h where that is installed
 * (amd/tests/e2e/run_hsa.sh builds the same program against both). The names
 * are the specification's, so a program written for HSA builds against this
 * header unchanged. */
#ifndef VGPU_HSA_ABI_H
#define VGPU_HSA_ABI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  HSA_STATUS_SUCCESS = 0x0,
  HSA_STATUS_INFO_BREAK = 0x1,
  HSA_STATUS_ERROR = 0x1000,
  HSA_STATUS_ERROR_INVALID_ARGUMENT = 0x1001,
  HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = 0x1002,
  HSA_STATUS_ERROR_INVALID_ALLOCATION = 0x1003,
  HSA_STATUS_ERROR_INVALID_AGENT = 0x1004,
  HSA_STATUS_ERROR_INVALID_REGION = 0x1005,
  HSA_STATUS_ERROR_INVALID_SIGNAL = 0x1006,
  HSA_STATUS_ERROR_INVALID_QUEUE = 0x1007,
  HSA_STATUS_ERROR_OUT_OF_RESOURCES = 0x1008,
  HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = 0x1009,
  HSA_STATUS_ERROR_RESOURCE_FREE = 0x100A,
  HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B,
  HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = 0x100C,
  HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = 0x100D,
  HSA_STATUS_ERROR_INVALID_INDEX = 0x100E,
  HSA_STATUS_ERROR_INVALID_ISA = 0x100F,
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 0x1010,
  HSA_STATUS_ERROR_INVALID_EXECUTABLE = 0x1011,
  HSA_STATUS_ERROR_FROZEN_EXECUTABLE = 0x1012,
  HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = 0x1013,
  HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = 0x1014,
  HSA_STATUS_ERROR_VARIABLE_UNDEFINED = 0x1015,
  HSA_STATUS_ERROR_EXCEPTION = 0x1016,
  HSA_STATUS_ERROR_INVALID_ISA_NAME = 0x1017,
  HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = 0x1018,
  HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = 0x1019,
  HSA_STATUS_ERROR_INVALID_FILE = 0x1020,
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = 0x1021,
  HSA_STATUS_ERROR_INVALID_CACHE = 0x1022,
  HSA_STATUS_ERROR_INVALID_WAVEFRONT = 0x1023,
  HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = 0x1024,
  HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = 0x1025,
  HSA_STATUS_ERROR_FATAL = 0x1026,
  /* AMD's (hsa_ext_amd.h). */
  HSA_STATUS_ERROR_INVALID_MEMORY_POOL = 40,
  HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION = 41,
  HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION = 42,
  HSA_STATUS_ERROR_MEMORY_FAULT = 43,
  HSA_STATUS_ERROR_NOT_SUPPORTED = 47
} hsa_status_t;

typedef struct hsa_agent_s { uint64_t handle; } hsa_agent_t;
typedef struct hsa_signal_s { uint64_t handle; } hsa_signal_t;
typedef struct hsa_region_s { uint64_t handle; } hsa_region_t;
typedef struct hsa_isa_s { uint64_t handle; } hsa_isa_t;
typedef struct hsa_executable_s { uint64_t handle; } hsa_executable_t;
typedef struct hsa_executable_symbol_s { uint64_t handle; } hsa_executable_symbol_t;
typedef struct hsa_code_object_reader_s { uint64_t handle; } hsa_code_object_reader_t;
typedef struct hsa_loaded_code_object_s { uint64_t handle; } hsa_loaded_code_object_t;
typedef struct hsa_amd_memory_pool_s { uint64_t handle; } hsa_amd_memory_pool_t;
typedef struct hsa_dim3_s { uint32_t x, y, z; } hsa_dim3_t;
typedef int64_t hsa_signal_value_t;
typedef int hsa_file_t;

/* ---- The system -------------------------------------------------------- */

typedef enum {
  HSA_SYSTEM_INFO_VERSION_MAJOR = 0,
  HSA_SYSTEM_INFO_VERSION_MINOR = 1,
  HSA_SYSTEM_INFO_TIMESTAMP = 2,
  HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY = 3,
  HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT = 4,
  HSA_SYSTEM_INFO_ENDIANNESS = 5,
  HSA_SYSTEM_INFO_MACHINE_MODEL = 6,
  HSA_SYSTEM_INFO_EXTENSIONS = 7
} hsa_system_info_t;
typedef enum { HSA_ENDIANNESS_LITTLE = 0, HSA_ENDIANNESS_BIG = 1 } hsa_endianness_t;
typedef enum { HSA_MACHINE_MODEL_SMALL = 0, HSA_MACHINE_MODEL_LARGE = 1 } hsa_machine_model_t;
typedef enum { HSA_PROFILE_BASE = 0, HSA_PROFILE_FULL = 1 } hsa_profile_t;
typedef enum {
  HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT = 0,
  HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO = 1,
  HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR = 2
} hsa_default_float_rounding_mode_t;

hsa_status_t hsa_init(void);
hsa_status_t hsa_shut_down(void);
hsa_status_t hsa_status_string(hsa_status_t status, const char** status_string);
hsa_status_t hsa_system_get_info(hsa_system_info_t attribute, void* value);

/* ---- Agents ------------------------------------------------------------ */

typedef enum { HSA_DEVICE_TYPE_CPU = 0, HSA_DEVICE_TYPE_GPU = 1, HSA_DEVICE_TYPE_DSP = 2 } hsa_device_type_t;
typedef enum { HSA_AGENT_FEATURE_KERNEL_DISPATCH = 1, HSA_AGENT_FEATURE_AGENT_DISPATCH = 2 } hsa_agent_feature_t;
typedef enum { HSA_QUEUE_TYPE_MULTI = 0, HSA_QUEUE_TYPE_SINGLE = 1 } hsa_queue_type_t;
typedef uint32_t hsa_queue_type32_t;
typedef enum {
  HSA_AGENT_INFO_NAME = 0,
  HSA_AGENT_INFO_VENDOR_NAME = 1,
  HSA_AGENT_INFO_FEATURE = 2,
  HSA_AGENT_INFO_MACHINE_MODEL = 3,
  HSA_AGENT_INFO_PROFILE = 4,
  HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5,
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7,
  HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 8,
  HSA_AGENT_INFO_GRID_MAX_DIM = 9,
  HSA_AGENT_INFO_GRID_MAX_SIZE = 10,
  HSA_AGENT_INFO_FBARRIER_MAX_SIZE = 11,
  HSA_AGENT_INFO_QUEUES_MAX = 12,
  HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13,
  HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14,
  HSA_AGENT_INFO_QUEUE_TYPE = 15,
  HSA_AGENT_INFO_NODE = 16,
  HSA_AGENT_INFO_DEVICE = 17,
  HSA_AGENT_INFO_CACHE_SIZE = 18,
  HSA_AGENT_INFO_ISA = 19,
  HSA_AGENT_INFO_EXTENSIONS = 20,
  HSA_AGENT_INFO_VERSION_MAJOR = 21,
  HSA_AGENT_INFO_VERSION_MINOR = 22,
  HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 23,
  HSA_AGENT_INFO_FAST_F16_OPERATION = 24
} hsa_agent_info_t;
/* AMD's (hsa_ext_amd.h). */
typedef enum {
  HSA_AMD_AGENT_INFO_CHIP_ID = 0xA000,
  HSA_AMD_AGENT_INFO_CACHELINE_SIZE = 0xA001,
  HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = 0xA002,
  HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY = 0xA003,
  HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = 0xA004,
  HSA_AMD_AGENT_INFO_BDFID = 0xA006,
  HSA_AMD_AGENT_INFO_MEMORY_WIDTH = 0xA007,
  HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY = 0xA008,
  HSA_AMD_AGENT_INFO_PRODUCT_NAME = 0xA009,
  HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = 0xA00A,
  HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU = 0xA00B,
  HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES = 0xA00C,
  HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE = 0xA00D,
  HSA_AMD_AGENT_INFO_DOMAIN = 0xA00F,
  HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES = 0xA010,
  HSA_AMD_AGENT_INFO_UUID = 0xA011,
  HSA_AMD_AGENT_INFO_ASIC_REVISION = 0xA012,
  HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS = 0xA013,
  HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT = 0xA014,
  HSA_AMD_AGENT_INFO_MEMORY_AVAIL = 0xA015,
  HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY = 0xA016
} hsa_amd_agent_info_t;

hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t agent, void* data), void* data);
hsa_status_t hsa_agent_get_info(hsa_agent_t agent, hsa_agent_info_t attribute, void* value);

/* ---- Instruction sets -------------------------------------------------- */

typedef enum {
  HSA_ISA_INFO_NAME_LENGTH = 0,
  HSA_ISA_INFO_NAME = 1,
  HSA_ISA_INFO_MACHINE_MODELS = 5,
  HSA_ISA_INFO_PROFILES = 6,
  HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES = 7,
  HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 8,
  HSA_ISA_INFO_FAST_F16_OPERATION = 9,
  HSA_ISA_INFO_WORKGROUP_MAX_DIM = 12,
  HSA_ISA_INFO_WORKGROUP_MAX_SIZE = 13,
  HSA_ISA_INFO_GRID_MAX_DIM = 14,
  HSA_ISA_INFO_GRID_MAX_SIZE = 16,
  HSA_ISA_INFO_FBARRIER_MAX_SIZE = 17
} hsa_isa_info_t;

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t agent, hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                    void* data);
hsa_status_t hsa_isa_get_info_alt(hsa_isa_t isa, hsa_isa_info_t attribute, void* value);
hsa_status_t hsa_isa_from_name(const char* name, hsa_isa_t* isa);

/* ---- Signals ----------------------------------------------------------- */

typedef enum {
  HSA_SIGNAL_CONDITION_EQ = 0,
  HSA_SIGNAL_CONDITION_NE = 1,
  HSA_SIGNAL_CONDITION_LT = 2,
  HSA_SIGNAL_CONDITION_GTE = 3
} hsa_signal_condition_t;
typedef enum { HSA_WAIT_STATE_BLOCKED = 0, HSA_WAIT_STATE_ACTIVE = 1 } hsa_wait_state_t;

hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
                               const hsa_agent_t* consumers, hsa_signal_t* signal);
hsa_status_t hsa_signal_destroy(hsa_signal_t signal);
hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
                                   const hsa_agent_t* consumers, uint64_t attributes, hsa_signal_t* signal);
hsa_signal_value_t hsa_signal_load_relaxed(hsa_signal_t signal);
hsa_signal_value_t hsa_signal_load_scacquire(hsa_signal_t signal);
void hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_silent_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_silent_store_screlease(hsa_signal_t signal, hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_exchange_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_exchange_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_cas_scacq_screl(hsa_signal_t signal, hsa_signal_value_t expected,
                                              hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_cas_relaxed(hsa_signal_t signal, hsa_signal_value_t expected, hsa_signal_value_t value);
void hsa_signal_add_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_add_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_subtract_scacq_screl(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_subtract_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_and_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_or_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
void hsa_signal_xor_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_wait_scacquire(hsa_signal_t signal, hsa_signal_condition_t condition,
                                             hsa_signal_value_t compare_value, uint64_t timeout_hint,
                                             hsa_wait_state_t wait_state_hint);
hsa_signal_value_t hsa_signal_wait_relaxed(hsa_signal_t signal, hsa_signal_condition_t condition,
                                           hsa_signal_value_t compare_value, uint64_t timeout_hint,
                                           hsa_wait_state_t wait_state_hint);

/* ---- Queues and packets ------------------------------------------------ */

typedef enum { HSA_QUEUE_FEATURE_KERNEL_DISPATCH = 1, HSA_QUEUE_FEATURE_AGENT_DISPATCH = 2 } hsa_queue_feature_t;

typedef struct hsa_queue_s {
  hsa_queue_type32_t type;
  uint32_t features;
  void* base_address;   /* the ring: `size` packets of 64 bytes */
  hsa_signal_t doorbell_signal;
  uint32_t size;
  uint32_t reserved1;
  uint64_t id;
} hsa_queue_t;

typedef enum {
  HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  HSA_PACKET_TYPE_INVALID = 1,
  HSA_PACKET_TYPE_KERNEL_DISPATCH = 2,
  HSA_PACKET_TYPE_BARRIER_AND = 3,
  HSA_PACKET_TYPE_AGENT_DISPATCH = 4,
  HSA_PACKET_TYPE_BARRIER_OR = 5
} hsa_packet_type_t;
typedef enum { HSA_FENCE_SCOPE_NONE = 0, HSA_FENCE_SCOPE_AGENT = 1, HSA_FENCE_SCOPE_SYSTEM = 2 } hsa_fence_scope_t;
typedef enum {
  HSA_PACKET_HEADER_TYPE = 0,
  HSA_PACKET_HEADER_BARRIER = 8,
  HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9,
  HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11
} hsa_packet_header_t;
typedef enum { HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0 } hsa_kernel_dispatch_packet_setup_t;

typedef struct hsa_kernel_dispatch_packet_s {
  uint16_t header;
  uint16_t setup;   /* the grid's dimensions, 1 to 3 */
  uint16_t workgroup_size_x, workgroup_size_y, workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x, grid_size_y, grid_size_z;   /* in work-items */
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  void* kernarg_address;
  uint64_t reserved2;
  hsa_signal_t completion_signal;
} hsa_kernel_dispatch_packet_t;

typedef struct hsa_barrier_and_packet_s {
  uint16_t header;
  uint16_t reserved0;
  uint32_t reserved1;
  hsa_signal_t dep_signal[5];
  uint64_t reserved2;
  hsa_signal_t completion_signal;
} hsa_barrier_and_packet_t;
typedef hsa_barrier_and_packet_t hsa_barrier_or_packet_t;

hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                              void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
                              uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue);
hsa_status_t hsa_queue_destroy(hsa_queue_t* queue);
uint64_t hsa_queue_load_read_index_relaxed(const hsa_queue_t* queue);
uint64_t hsa_queue_load_read_index_scacquire(const hsa_queue_t* queue);
uint64_t hsa_queue_load_write_index_relaxed(const hsa_queue_t* queue);
uint64_t hsa_queue_load_write_index_scacquire(const hsa_queue_t* queue);
void hsa_queue_store_write_index_relaxed(const hsa_queue_t* queue, uint64_t value);
void hsa_queue_store_write_index_screlease(const hsa_queue_t* queue, uint64_t value);
uint64_t hsa_queue_add_write_index_relaxed(const hsa_queue_t* queue, uint64_t value);
uint64_t hsa_queue_add_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t value);
uint64_t hsa_queue_add_write_index_screlease(const hsa_queue_t* queue, uint64_t value);
uint64_t hsa_queue_cas_write_index_relaxed(const hsa_queue_t* queue, uint64_t expected, uint64_t value);
uint64_t hsa_queue_cas_write_index_scacq_screl(const hsa_queue_t* queue, uint64_t expected, uint64_t value);

/* ---- Memory ------------------------------------------------------------ */

typedef enum {
  HSA_REGION_SEGMENT_GLOBAL = 0,
  HSA_REGION_SEGMENT_READONLY = 1,
  HSA_REGION_SEGMENT_PRIVATE = 2,
  HSA_REGION_SEGMENT_GROUP = 3,
  HSA_REGION_SEGMENT_KERNARG = 4
} hsa_region_segment_t;
typedef enum {
  HSA_REGION_GLOBAL_FLAG_KERNARG = 1,
  HSA_REGION_GLOBAL_FLAG_FINE_GRAINED = 2,
  HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED = 4
} hsa_region_global_flag_t;
typedef enum {
  HSA_REGION_INFO_SEGMENT = 0,
  HSA_REGION_INFO_GLOBAL_FLAGS = 1,
  HSA_REGION_INFO_SIZE = 2,
  HSA_REGION_INFO_ALLOC_MAX_SIZE = 4,
  HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED = 5,
  HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE = 6,
  HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT = 7,
  HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE = 8
} hsa_region_info_t;

hsa_status_t hsa_agent_iterate_regions(hsa_agent_t agent, hsa_status_t (*callback)(hsa_region_t region, void* data),
                                       void* data);
hsa_status_t hsa_region_get_info(hsa_region_t region, hsa_region_info_t attribute, void* value);
hsa_status_t hsa_memory_allocate(hsa_region_t region, size_t size, void** ptr);
hsa_status_t hsa_memory_free(void* ptr);
hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size);
hsa_status_t hsa_memory_register(void* ptr, size_t size);
hsa_status_t hsa_memory_deregister(void* ptr, size_t size);

typedef enum {
  HSA_AMD_SEGMENT_GLOBAL = 0,
  HSA_AMD_SEGMENT_READONLY = 1,
  HSA_AMD_SEGMENT_PRIVATE = 2,
  HSA_AMD_SEGMENT_GROUP = 3
} hsa_amd_segment_t;
typedef enum {
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = 1,
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2,
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4
} hsa_amd_memory_pool_global_flag_t;
typedef enum { HSA_AMD_MEMORY_POOL_LOCATION_CPU = 0, HSA_AMD_MEMORY_POOL_LOCATION_GPU = 1 } hsa_amd_memory_pool_location_t;
typedef enum {
  HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0,
  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1,
  HSA_AMD_MEMORY_POOL_INFO_SIZE = 2,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = 7,
  HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = 15,
  HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE = 16,
  HSA_AMD_MEMORY_POOL_INFO_LOCATION = 17,
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = 18
} hsa_amd_memory_pool_info_t;
typedef enum {
  HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = 0,
  HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT = 1,
  HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT = 2
} hsa_amd_memory_pool_access_t;
typedef enum {
  HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = 0,
  HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS = 1
} hsa_amd_agent_memory_pool_info_t;

hsa_status_t hsa_amd_agent_iterate_memory_pools(hsa_agent_t agent,
                                                hsa_status_t (*callback)(hsa_amd_memory_pool_t pool, void* data),
                                                void* data);
hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t pool, hsa_amd_memory_pool_info_t attribute,
                                          void* value);
hsa_status_t hsa_amd_agent_memory_pool_get_info(hsa_agent_t agent, hsa_amd_memory_pool_t pool,
                                                hsa_amd_agent_memory_pool_info_t attribute, void* value);
hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags, void** ptr);
hsa_status_t hsa_amd_memory_pool_free(void* ptr);
hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents, const uint32_t* flags,
                                         const void* ptr);
hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
                                       size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                       hsa_signal_t completion_signal);
hsa_status_t hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count);
hsa_status_t hsa_amd_memory_lock(void* host_ptr, size_t size, hsa_agent_t* agents, int num_agent, void** agent_ptr);
hsa_status_t hsa_amd_memory_unlock(void* host_ptr);

/* ---- Code objects and executables -------------------------------------- */

typedef enum { HSA_EXECUTABLE_STATE_UNFROZEN = 0, HSA_EXECUTABLE_STATE_FROZEN = 1 } hsa_executable_state_t;
typedef enum { HSA_SYMBOL_KIND_VARIABLE = 0, HSA_SYMBOL_KIND_KERNEL = 1, HSA_SYMBOL_KIND_INDIRECT_FUNCTION = 2 } hsa_symbol_kind_t;
typedef enum { HSA_SYMBOL_LINKAGE_MODULE = 0, HSA_SYMBOL_LINKAGE_PROGRAM = 1 } hsa_symbol_linkage_t;
typedef enum {
  HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2,
  HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE = 5,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = 9,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15,
  HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION = 17,
  HSA_EXECUTABLE_SYMBOL_INFO_AGENT = 20,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = 22
} hsa_executable_symbol_info_t;

hsa_status_t hsa_code_object_reader_create_from_memory(const void* code_object, size_t size,
                                                       hsa_code_object_reader_t* reader);
hsa_status_t hsa_code_object_reader_create_from_file(hsa_file_t file, hsa_code_object_reader_t* reader);
hsa_status_t hsa_code_object_reader_destroy(hsa_code_object_reader_t reader);
hsa_status_t hsa_executable_create_alt(hsa_profile_t profile, hsa_default_float_rounding_mode_t rounding_mode,
                                       const char* options, hsa_executable_t* executable);
hsa_status_t hsa_executable_load_agent_code_object(hsa_executable_t executable, hsa_agent_t agent,
                                                   hsa_code_object_reader_t reader, const char* options,
                                                   hsa_loaded_code_object_t* loaded_code_object);
hsa_status_t hsa_executable_freeze(hsa_executable_t executable, const char* options);
hsa_status_t hsa_executable_destroy(hsa_executable_t executable);
hsa_status_t hsa_executable_get_symbol_by_name(hsa_executable_t executable, const char* symbol_name,
                                               const hsa_agent_t* agent, hsa_executable_symbol_t* symbol);
hsa_status_t hsa_executable_iterate_agent_symbols(
    hsa_executable_t executable, hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t exec, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data),
    void* data);
hsa_status_t hsa_executable_symbol_get_info(hsa_executable_symbol_t symbol, hsa_executable_symbol_info_t attribute,
                                            void* value);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_HSA_ABI_H */
