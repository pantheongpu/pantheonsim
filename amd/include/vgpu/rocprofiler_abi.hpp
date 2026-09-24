// The rocprofiler-sdk interface, as far as AMD's own profiler uses it.
//
// rocprofv3 is a front end over a tool library (librocprofiler-sdk-tool.so)
// that asks librocprofiler-sdk for the machine's agents and counters,
// subscribes to what the runtime does, and writes what it is told. VirtualGPU's
// librocprofiler-sdk answers it from the simulated devices, so the profiler
// runs unmodified. The tool reads every structure at the offsets its own
// headers give, so these declarations are those headers' -- the names are
// kept, and test_amd_rocprofiler_abi checks each field's offset against the
// headers wherever ROCm's rocprofiler-sdk is installed.
//
// Only what the tool reads and writes is here. A structure that embeds a type
// from HSA or the kernel driver's interface (an agent's cache and link types,
// an HSA agent handle) carries a stand-in of the same size and alignment; the
// argument union of a HIP API record is carried as its size alone, since
// nothing here fills it in.
//
// The declarations are carried over from AMD's rocprofiler-sdk headers, which
// are distributed under the MIT licence below; the comments are VirtualGPU's
// own.
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vgpu::amd::rocprof {

// The interface version this answers to: rocprofiler-sdk 1.0.0, ROCm 7.1's.
constexpr uint32_t kVersionMajor = 1, kVersionMinor = 0, kVersionPatch = 0;
constexpr uint32_t kVersion = kVersionMajor * 10000 + kVersionMinor * 100 + kVersionPatch;

// ---- fwd.h ----------------------------------------------------------------

typedef enum rocprofiler_status_t {
  ROCPROFILER_STATUS_SUCCESS = 0,
  ROCPROFILER_STATUS_ERROR,
  ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_THREAD_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR,
  ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID,
  ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED,
  ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT,
  ROCPROFILER_STATUS_ERROR_CONTEXT_ID_NOT_ZERO,
  ROCPROFILER_STATUS_ERROR_BUFFER_BUSY,
  ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED,
  ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED,
  ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED,
  ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI,
  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT,
  ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT,
  ROCPROFILER_STATUS_ERROR_FINALIZED,
  ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED,
  ROCPROFILER_STATUS_ERROR_DIM_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED,
  ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_AQL_NO_EVENT_COORD,
  ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL,
  ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES,
  ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND,
  ROCPROFILER_STATUS_ERROR_AGENT_DISPATCH_CONFLICT,
  ROCPROFILER_STATUS_INTERNAL_NO_AGENT_CONTEXT,
  ROCPROFILER_STATUS_ERROR_SAMPLE_RATE_EXCEEDED,
  ROCPROFILER_STATUS_ERROR_NO_PROFILE_QUEUE,
  ROCPROFILER_STATUS_ERROR_NO_HARDWARE_COUNTERS,
  ROCPROFILER_STATUS_ERROR_AGENT_MISMATCH,
  ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE,
  ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT,
  ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED,
  ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED,
  ROCPROFILER_STATUS_LAST,
} rocprofiler_status_t;

typedef enum rocprofiler_buffer_category_t {
  ROCPROFILER_BUFFER_CATEGORY_NONE = 0,
  ROCPROFILER_BUFFER_CATEGORY_TRACING,
  ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING,
  ROCPROFILER_BUFFER_CATEGORY_COUNTERS,
  ROCPROFILER_BUFFER_CATEGORY_LAST,
} rocprofiler_buffer_category_t;

typedef enum rocprofiler_agent_type_t {
  ROCPROFILER_AGENT_TYPE_NONE = 0,
  ROCPROFILER_AGENT_TYPE_CPU,
  ROCPROFILER_AGENT_TYPE_GPU,
  ROCPROFILER_AGENT_TYPE_LAST,
} rocprofiler_agent_type_t;

typedef enum rocprofiler_callback_phase_t {
  ROCPROFILER_CALLBACK_PHASE_NONE = 0,
  ROCPROFILER_CALLBACK_PHASE_ENTER,
  ROCPROFILER_CALLBACK_PHASE_LOAD = ROCPROFILER_CALLBACK_PHASE_ENTER,
  ROCPROFILER_CALLBACK_PHASE_EXIT,
  ROCPROFILER_CALLBACK_PHASE_UNLOAD = ROCPROFILER_CALLBACK_PHASE_EXIT,
  ROCPROFILER_CALLBACK_PHASE_LAST,
} rocprofiler_callback_phase_t;

typedef enum rocprofiler_callback_tracing_kind_t {
  ROCPROFILER_CALLBACK_TRACING_NONE = 0,
  ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
  ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
  ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
  ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
  ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
  ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
  ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
  ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
  ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,
  ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
  ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY,
  ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
  ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
  ROCPROFILER_CALLBACK_TRACING_RCCL_API,
  ROCPROFILER_CALLBACK_TRACING_OMPT,
  ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
  ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION,
  ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API,
  ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API,
  ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,
  ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API,
  ROCPROFILER_CALLBACK_TRACING_LAST,
} rocprofiler_callback_tracing_kind_t;

typedef enum rocprofiler_buffer_tracing_kind_t {
  ROCPROFILER_BUFFER_TRACING_NONE = 0,
  ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
  ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
  ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
  ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API,
  ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
  ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,
  ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API,
  ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API,
  ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API,
  ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
  ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
  ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,
  ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,
  ROCPROFILER_BUFFER_TRACING_RCCL_API,
  ROCPROFILER_BUFFER_TRACING_OMPT,
  ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
  ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION,
  ROCPROFILER_BUFFER_TRACING_ROCDECODE_API,
  ROCPROFILER_BUFFER_TRACING_ROCJPEG_API,
  ROCPROFILER_BUFFER_TRACING_HIP_STREAM,
  ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT,
  ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT,
  ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT,
  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE,
  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT,
  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE,
  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
  ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS,
  ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE,
  ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT,
  ROCPROFILER_BUFFER_TRACING_KFD_QUEUE,
  ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API,
  ROCPROFILER_BUFFER_TRACING_LAST,
} rocprofiler_buffer_tracing_kind_t;

typedef enum rocprofiler_code_object_operation_t {
  ROCPROFILER_CODE_OBJECT_NONE = 0,
  ROCPROFILER_CODE_OBJECT_LOAD,
  ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER,
  ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER,
  ROCPROFILER_CODE_OBJECT_LAST,
} rocprofiler_code_object_operation_t;

typedef enum rocprofiler_hip_stream_operation_t {
  ROCPROFILER_HIP_STREAM_NONE = 0,
  ROCPROFILER_HIP_STREAM_CREATE,
  ROCPROFILER_HIP_STREAM_DESTROY,
  ROCPROFILER_HIP_STREAM_SET,
  ROCPROFILER_HIP_STREAM_LAST,
} rocprofiler_hip_stream_operation_t;

typedef enum rocprofiler_memory_copy_operation_t {
  ROCPROFILER_MEMORY_COPY_NONE = 0,
  ROCPROFILER_MEMORY_COPY_HOST_TO_HOST,
  ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE,
  ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST,
  ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE,
  ROCPROFILER_MEMORY_COPY_LAST,
} rocprofiler_memory_copy_operation_t;

typedef enum rocprofiler_memory_allocation_operation_t {
  ROCPROFILER_MEMORY_ALLOCATION_NONE = 0,
  ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE,
  ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE,
  ROCPROFILER_MEMORY_ALLOCATION_FREE,
  ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE,
  ROCPROFILER_MEMORY_ALLOCATION_LAST,
} rocprofiler_memory_allocation_operation_t;

typedef enum rocprofiler_kernel_dispatch_operation_t {
  ROCPROFILER_KERNEL_DISPATCH_NONE = 0,
  ROCPROFILER_KERNEL_DISPATCH_ENQUEUE = 1,
  ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
  ROCPROFILER_KERNEL_DISPATCH_LAST,
} rocprofiler_kernel_dispatch_operation_t;

typedef enum rocprofiler_scratch_memory_operation_t {
  ROCPROFILER_SCRATCH_MEMORY_NONE = 0,
  ROCPROFILER_SCRATCH_MEMORY_ALLOC,
  ROCPROFILER_SCRATCH_MEMORY_FREE,
  ROCPROFILER_SCRATCH_MEMORY_ASYNC_RECLAIM,
  ROCPROFILER_SCRATCH_MEMORY_LAST,
} rocprofiler_scratch_memory_operation_t;

typedef enum rocprofiler_buffer_policy_t {
  ROCPROFILER_BUFFER_POLICY_NONE = 0,
  ROCPROFILER_BUFFER_POLICY_DISCARD,
  ROCPROFILER_BUFFER_POLICY_LOSSLESS,
  ROCPROFILER_BUFFER_POLICY_LAST,
} rocprofiler_buffer_policy_t;

typedef enum rocprofiler_runtime_library_t {
  ROCPROFILER_LIBRARY = (1 << 0),
  ROCPROFILER_HSA_LIBRARY = (1 << 1),
  ROCPROFILER_HIP_LIBRARY = (1 << 2),
  ROCPROFILER_MARKER_LIBRARY = (1 << 3),
  ROCPROFILER_RCCL_LIBRARY = (1 << 4),
  ROCPROFILER_ROCDECODE_LIBRARY = (1 << 5),
  ROCPROFILER_ROCJPEG_LIBRARY = (1 << 6),
  ROCPROFILER_LIBRARY_LAST = ROCPROFILER_ROCJPEG_LIBRARY,
} rocprofiler_runtime_library_t;

typedef enum rocprofiler_intercept_table_t {
  ROCPROFILER_HSA_TABLE = (1 << 0),
  ROCPROFILER_HIP_RUNTIME_TABLE = (1 << 1),
  ROCPROFILER_HIP_COMPILER_TABLE = (1 << 2),
  ROCPROFILER_MARKER_CORE_TABLE = (1 << 3),
  ROCPROFILER_MARKER_CONTROL_TABLE = (1 << 4),
  ROCPROFILER_MARKER_NAME_TABLE = (1 << 5),
  ROCPROFILER_RCCL_TABLE = (1 << 6),
  ROCPROFILER_ROCDECODE_TABLE = (1 << 7),
  ROCPROFILER_ROCJPEG_TABLE = (1 << 8),
  ROCPROFILER_TABLE_LAST = ROCPROFILER_ROCJPEG_TABLE,
} rocprofiler_intercept_table_t;

typedef enum rocprofiler_runtime_initialization_operation_t {
  ROCPROFILER_RUNTIME_INITIALIZATION_NONE = 0,
  ROCPROFILER_RUNTIME_INITIALIZATION_HSA,
  ROCPROFILER_RUNTIME_INITIALIZATION_HIP,
  ROCPROFILER_RUNTIME_INITIALIZATION_MARKER,
  ROCPROFILER_RUNTIME_INITIALIZATION_RCCL,
  ROCPROFILER_RUNTIME_INITIALIZATION_ROCDECODE,
  ROCPROFILER_RUNTIME_INITIALIZATION_ROCJPEG,
  ROCPROFILER_RUNTIME_INITIALIZATION_LAST,
} rocprofiler_runtime_initialization_operation_t;

typedef enum rocprofiler_counter_info_version_id_t {
  ROCPROFILER_COUNTER_INFO_VERSION_NONE,
  ROCPROFILER_COUNTER_INFO_VERSION_0,
  ROCPROFILER_COUNTER_INFO_VERSION_1,
  ROCPROFILER_COUNTER_INFO_VERSION_LAST,
} rocprofiler_counter_info_version_id_t;

typedef enum rocprofiler_counter_flag_t {
  ROCPROFILER_COUNTER_FLAG_NONE = 0,
  ROCPROFILER_COUNTER_FLAG_ASYNC,
  ROCPROFILER_COUNTER_FLAG_APPEND_DEFINITION,
  ROCPROFILER_COUNTER_FLAG_LAST,
} rocprofiler_counter_flag_t;

typedef uint64_t rocprofiler_timestamp_t;
typedef uint64_t rocprofiler_thread_id_t;
typedef int32_t rocprofiler_tracing_operation_t;
typedef uint64_t rocprofiler_kernel_id_t;
typedef uint64_t rocprofiler_dispatch_id_t;
typedef uint64_t rocprofiler_counter_instance_id_t;
typedef uint64_t rocprofiler_counter_dimension_id_t;

typedef union rocprofiler_user_data_t {
  uint64_t value;
  void* ptr;
} rocprofiler_user_data_t;

typedef union rocprofiler_address_t {
  uint64_t handle;
  uint64_t value;
  const void* ptr;
} rocprofiler_address_t;

typedef struct rocprofiler_uuid_t {
  uint8_t bytes[16];
} rocprofiler_uuid_t;

typedef struct rocprofiler_context_id_t {
  uint64_t handle;
} rocprofiler_context_id_t;
typedef struct rocprofiler_queue_id_t {
  uint64_t handle;
} rocprofiler_queue_id_t;
typedef struct rocprofiler_stream_id_t {
  uint64_t handle;
} rocprofiler_stream_id_t;

typedef struct rocprofiler_correlation_id_t {
  uint64_t internal;
  rocprofiler_user_data_t external;
  uint64_t ancestor;
} rocprofiler_correlation_id_t;

typedef struct rocprofiler_async_correlation_id_t {
  uint64_t internal;
  rocprofiler_user_data_t external;
} rocprofiler_async_correlation_id_t;

typedef struct rocprofiler_buffer_id_t {
  uint64_t handle;
} rocprofiler_buffer_id_t;
typedef struct rocprofiler_agent_id_t {
  uint64_t handle;
} rocprofiler_agent_id_t;
typedef struct rocprofiler_counter_id_t {
  uint64_t handle;
} rocprofiler_counter_id_t;
typedef struct rocprofiler_counter_config_id_t {
  uint64_t handle;
} rocprofiler_counter_config_id_t;

typedef struct rocprofiler_dim3_t {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} rocprofiler_dim3_t;

typedef struct rocprofiler_callback_tracing_record_t {
  rocprofiler_context_id_t context_id;
  rocprofiler_thread_id_t thread_id;
  rocprofiler_correlation_id_t correlation_id;
  rocprofiler_callback_tracing_kind_t kind;
  rocprofiler_tracing_operation_t operation;
  rocprofiler_callback_phase_t phase;
  void* payload;
} rocprofiler_callback_tracing_record_t;

typedef struct rocprofiler_record_header_t {
  union {
    struct {
      uint32_t category;
      uint32_t kind;
    };
    uint64_t hash;
  };
  void* payload;
} rocprofiler_record_header_t;

typedef struct rocprofiler_kernel_dispatch_info_t {
  uint64_t size;
  rocprofiler_agent_id_t agent_id;
  rocprofiler_queue_id_t queue_id;
  rocprofiler_kernel_id_t kernel_id;
  rocprofiler_dispatch_id_t dispatch_id;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  rocprofiler_dim3_t workgroup_size;
  rocprofiler_dim3_t grid_size;
  uint8_t reserved_padding[56];
} rocprofiler_kernel_dispatch_info_t;

typedef struct rocprofiler_counter_record_dimension_info_t {
  const char* name;
  size_t instance_size;
  rocprofiler_counter_dimension_id_t id;
} rocprofiler_counter_record_dimension_info_t;

typedef struct rocprofiler_counter_record_t {
  rocprofiler_counter_instance_id_t id;
  double counter_value;
  rocprofiler_dispatch_id_t dispatch_id;
  rocprofiler_user_data_t user_data;
  rocprofiler_agent_id_t agent_id;
} rocprofiler_counter_record_t;

// ---- agent.h --------------------------------------------------------------

typedef enum rocprofiler_agent_version_t {
  ROCPROFILER_AGENT_INFO_VERSION_NONE = 0,
  ROCPROFILER_AGENT_INFO_VERSION_0 = 1,
  ROCPROFILER_AGENT_INFO_VERSION_LAST,
} rocprofiler_agent_version_t;

// The kernel driver's words for a cache's type, a link's type and properties,
// a memory bank's heap and properties, and an engine's version, id and
// capabilities: each is 32 bits (hsakmttypes.h).
struct KfdWord {
  uint32_t Value;
};

typedef struct rocprofiler_agent_cache_t {
  uint64_t processor_id_low;
  uint64_t size;
  uint32_t level;
  uint32_t cache_line_size;
  uint32_t cache_lines_per_tag;
  uint32_t association;
  uint32_t latency;
  KfdWord type;
} rocprofiler_agent_cache_t;

typedef struct rocprofiler_agent_io_link_t {
  KfdWord type;
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t node_from;
  uint32_t node_to;
  uint32_t weight;
  uint32_t min_latency;
  uint32_t max_latency;
  uint32_t min_bandwidth;
  uint32_t max_bandwidth;
  uint32_t recommended_transfer_size;
  KfdWord flags;
} rocprofiler_agent_io_link_t;

typedef struct rocprofiler_agent_mem_bank_t {
  KfdWord heap_type;
  KfdWord flags;
  uint32_t width;
  uint32_t mem_clk_max;
  uint64_t size_in_bytes;
} rocprofiler_agent_mem_bank_t;

typedef struct rocprofiler_agent_runtime_visiblity_t {
  uint32_t hsa : 1;
  uint32_t hip : 1;
  uint32_t rccl : 1;
  uint32_t rocdecode : 1;
  uint32_t reserved : 28;
} rocprofiler_agent_runtime_visiblity_t;

typedef struct rocprofiler_agent_v0_t {
  uint64_t size;
  rocprofiler_agent_id_t id;
  rocprofiler_agent_type_t type;
  uint32_t cpu_cores_count;
  uint32_t simd_count;
  uint32_t mem_banks_count;
  uint32_t caches_count;
  uint32_t io_links_count;
  uint32_t cpu_core_id_base;
  uint32_t simd_id_base;
  uint32_t max_waves_per_simd;
  uint32_t lds_size_in_kb;
  uint32_t gds_size_in_kb;
  uint32_t num_gws;
  uint32_t wave_front_size;
  uint32_t num_xcc;
  uint32_t cu_count;
  uint32_t array_count;
  uint32_t num_shader_banks;
  uint32_t simd_arrays_per_engine;
  uint32_t cu_per_simd_array;
  uint32_t simd_per_cu;
  uint32_t max_slots_scratch_cu;
  uint32_t gfx_target_version;
  uint16_t vendor_id;
  uint16_t device_id;
  uint32_t location_id;
  uint32_t domain;
  uint32_t drm_render_minor;
  uint32_t num_sdma_engines;
  uint32_t num_sdma_xgmi_engines;
  uint32_t num_sdma_queues_per_engine;
  uint32_t num_cp_queues;
  uint32_t max_engine_clk_ccompute;
  uint32_t max_engine_clk_fcompute;
  KfdWord sdma_fw_version;
  KfdWord fw_version;
  KfdWord capability;
  uint32_t cu_per_engine;
  uint32_t max_waves_per_cu;
  uint32_t family_id;
  uint32_t workgroup_max_size;
  uint32_t grid_max_size;
  uint64_t local_mem_size;
  uint64_t hive_id;
  uint64_t gpu_id;
  rocprofiler_dim3_t workgroup_max_dim;
  rocprofiler_dim3_t grid_max_dim;
  const rocprofiler_agent_mem_bank_t* mem_banks;
  const rocprofiler_agent_cache_t* caches;
  const rocprofiler_agent_io_link_t* io_links;
  const char* name;
  const char* vendor_name;
  const char* product_name;
  const char* model_name;
  uint32_t node_id;
  int32_t logical_node_id;
  int32_t logical_node_type_id;
  rocprofiler_agent_runtime_visiblity_t runtime_visibility;
  rocprofiler_uuid_t uuid;
} rocprofiler_agent_v0_t;

typedef rocprofiler_status_t (*rocprofiler_query_available_agents_cb_t)(rocprofiler_agent_version_t version,
                                                                         const void** agents, size_t num_agents,
                                                                         void* user_data);

// ---- counters.h, counter_config.h, dispatch_counting_service.h ---------------

typedef struct rocprofiler_counter_info_v0_t {
  rocprofiler_counter_id_t id;
  const char* name;
  const char* description;
  const char* block;
  const char* expression;
  uint8_t is_constant : 1;
  uint8_t is_derived : 1;
} rocprofiler_counter_info_v0_t;

typedef struct rocprofiler_counter_dimension_info_t {
  uint64_t size;
  const char* dimension_name;
  size_t index;
} rocprofiler_counter_dimension_info_t;

typedef struct rocprofiler_counter_record_dimension_instance_info_t {
  uint64_t size;
  rocprofiler_counter_instance_id_t instance_id;
  uint64_t counter_id;
  uint64_t dimensions_count;
  const rocprofiler_counter_dimension_info_t** dimensions;
} rocprofiler_counter_record_dimension_instance_info_t;

typedef struct rocprofiler_counter_info_v1_t {
  uint64_t size;
  rocprofiler_counter_id_t id;
  const char* name;
  const char* description;
  const char* block;
  const char* expression;
  uint8_t is_constant : 1;
  uint8_t is_derived : 1;
  uint64_t dimensions_count;
  const rocprofiler_counter_record_dimension_info_t** dimensions;
  uint64_t dimensions_instances_count;
  const rocprofiler_counter_record_dimension_instance_info_t** dimensions_instances;
} rocprofiler_counter_info_v1_t;

typedef rocprofiler_status_t (*rocprofiler_available_counters_cb_t)(rocprofiler_agent_id_t agent_id,
                                                                     rocprofiler_counter_id_t* counters,
                                                                     size_t num_counters, void* user_data);

typedef struct rocprofiler_dispatch_counting_service_data_t {
  uint64_t size;
  rocprofiler_async_correlation_id_t correlation_id;
  rocprofiler_timestamp_t start_timestamp;
  rocprofiler_timestamp_t end_timestamp;
  rocprofiler_kernel_dispatch_info_t dispatch_info;
} rocprofiler_dispatch_counting_service_data_t;

typedef void (*rocprofiler_dispatch_counting_service_cb_t)(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                                           rocprofiler_counter_config_id_t* config,
                                                           rocprofiler_user_data_t* user_data,
                                                           void* callback_data_args);
typedef void (*rocprofiler_dispatch_counting_record_cb_t)(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                                          rocprofiler_counter_record_t* record_data,
                                                          size_t record_count, rocprofiler_user_data_t user_data,
                                                          void* callback_data_args);

// ---- buffer.h, buffer_tracing.h --------------------------------------------

typedef void (*rocprofiler_buffer_tracing_cb_t)(rocprofiler_context_id_t context, rocprofiler_buffer_id_t buffer_id,
                                                rocprofiler_record_header_t** headers, size_t num_headers,
                                                void* data, uint64_t drop_count);

typedef struct rocprofiler_buffer_tracing_hip_api_record_t {
  uint64_t size;
  rocprofiler_buffer_tracing_kind_t kind;
  rocprofiler_tracing_operation_t operation;
  rocprofiler_correlation_id_t correlation_id;
  rocprofiler_timestamp_t start_timestamp;
  rocprofiler_timestamp_t end_timestamp;
  rocprofiler_thread_id_t thread_id;
} rocprofiler_buffer_tracing_hip_api_record_t;

// The call's arguments and return value follow, as unions over every HIP
// function (rocprofiler_hip_api_args_t, rocprofiler_hip_api_retval_t). This
// leaves them zero: they are written out only when a tool asks for arguments.
typedef struct rocprofiler_buffer_tracing_hip_api_ext_record_t {
  uint64_t size;
  rocprofiler_buffer_tracing_kind_t kind;
  rocprofiler_tracing_operation_t operation;
  rocprofiler_correlation_id_t correlation_id;
  rocprofiler_timestamp_t start_timestamp;
  rocprofiler_timestamp_t end_timestamp;
  rocprofiler_thread_id_t thread_id;
  alignas(8) uint8_t args[88];
  alignas(8) uint8_t retval[24];
} rocprofiler_buffer_tracing_hip_api_ext_record_t;

typedef struct rocprofiler_buffer_tracing_memory_copy_record_t {
  uint64_t size;
  rocprofiler_buffer_tracing_kind_t kind;
  rocprofiler_memory_copy_operation_t operation;
  rocprofiler_async_correlation_id_t correlation_id;
  rocprofiler_thread_id_t thread_id;
  rocprofiler_timestamp_t start_timestamp;
  rocprofiler_timestamp_t end_timestamp;
  rocprofiler_agent_id_t dst_agent_id;
  rocprofiler_agent_id_t src_agent_id;
  uint64_t bytes;
  rocprofiler_address_t dst_address;
  rocprofiler_address_t src_address;
} rocprofiler_buffer_tracing_memory_copy_record_t;

typedef struct rocprofiler_buffer_tracing_memory_allocation_record_t {
  uint64_t size;
  rocprofiler_buffer_tracing_kind_t kind;
  rocprofiler_memory_allocation_operation_t operation;
  rocprofiler_correlation_id_t correlation_id;
  rocprofiler_thread_id_t thread_id;
  rocprofiler_timestamp_t start_timestamp;
  rocprofiler_timestamp_t end_timestamp;
  rocprofiler_agent_id_t agent_id;
  rocprofiler_address_t address;
  uint64_t allocation_size;
} rocprofiler_buffer_tracing_memory_allocation_record_t;

typedef struct rocprofiler_buffer_tracing_kernel_dispatch_record_t {
  uint64_t size;
  rocprofiler_buffer_tracing_kind_t kind;
  rocprofiler_kernel_dispatch_operation_t operation;
  rocprofiler_async_correlation_id_t correlation_id;
  rocprofiler_thread_id_t thread_id;
  rocprofiler_timestamp_t start_timestamp;
  rocprofiler_timestamp_t end_timestamp;
  rocprofiler_kernel_dispatch_info_t dispatch_info;
} rocprofiler_buffer_tracing_kernel_dispatch_record_t;

typedef int (*rocprofiler_buffer_tracing_kind_cb_t)(rocprofiler_buffer_tracing_kind_t kind, void* data);
typedef int (*rocprofiler_buffer_tracing_kind_operation_cb_t)(rocprofiler_buffer_tracing_kind_t kind,
                                                              rocprofiler_tracing_operation_t operation, void* data);

// ---- callback_tracing.h ----------------------------------------------------

typedef enum rocprofiler_code_object_storage_type_t {
  ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_NONE = 0,
  ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE = 1,
  ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY = 2,
  ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_LAST,
} rocprofiler_code_object_storage_type_t;

// hsa_agent_t: the HSA runtime's handle for the agent, which there is none of.
struct HsaAgent {
  uint64_t handle;
};

typedef struct rocprofiler_callback_tracing_code_object_load_data_t {
  uint64_t size;
  uint64_t code_object_id;
  union {
    rocprofiler_agent_id_t rocp_agent;
    rocprofiler_agent_id_t agent_id;
  };
  HsaAgent hsa_agent;
  const char* uri;
  uint64_t load_base;
  uint64_t load_size;
  int64_t load_delta;
  rocprofiler_code_object_storage_type_t storage_type;
  union {
    struct {
      int storage_file;
    };
    struct {
      uint64_t memory_base;
      uint64_t memory_size;
    };
  };
} rocprofiler_callback_tracing_code_object_load_data_t;

typedef struct rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t {
  uint64_t size;
  uint64_t kernel_id;
  uint64_t code_object_id;
  const char* kernel_name;
  uint64_t kernel_object;
  uint32_t kernarg_segment_size;
  uint32_t kernarg_segment_alignment;
  uint32_t group_segment_size;
  uint32_t private_segment_size;
  uint32_t sgpr_count;
  uint32_t arch_vgpr_count;
  uint32_t accum_vgpr_count;
  int64_t kernel_code_entry_byte_offset;
  rocprofiler_address_t kernel_address;
} rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

typedef struct rocprofiler_callback_tracing_runtime_initialization_data_t {
  uint64_t size;
  uint64_t version;
  uint64_t instance;
} rocprofiler_callback_tracing_runtime_initialization_data_t;

typedef struct rocprofiler_callback_tracing_hip_stream_data_t {
  uint64_t size;
  rocprofiler_stream_id_t stream_id;
  rocprofiler_address_t stream_value;
} rocprofiler_callback_tracing_hip_stream_data_t;

typedef void (*rocprofiler_callback_tracing_cb_t)(rocprofiler_callback_tracing_record_t record,
                                                  rocprofiler_user_data_t* user_data, void* callback_data);
typedef int (*rocprofiler_callback_tracing_kind_cb_t)(rocprofiler_callback_tracing_kind_t kind, void* data);
typedef int (*rocprofiler_callback_tracing_kind_operation_cb_t)(rocprofiler_callback_tracing_kind_t kind,
                                                                rocprofiler_tracing_operation_t operation,
                                                                void* data);

// ---- registration.h, internal_threading.h, external_correlation.h,
//      intercept_table.h -----------------------------------------------------

typedef struct rocprofiler_client_id_t {
  size_t size;
  const char* name;
  const uint32_t handle;
} rocprofiler_client_id_t;

typedef void (*rocprofiler_client_finalize_t)(rocprofiler_client_id_t);
typedef int (*rocprofiler_tool_initialize_t)(rocprofiler_client_finalize_t finalize_func, void* tool_data);
typedef void (*rocprofiler_tool_finalize_t)(void* tool_data);

typedef struct rocprofiler_tool_configure_result_t {
  size_t size;
  rocprofiler_tool_initialize_t initialize;
  rocprofiler_tool_finalize_t finalize;
  void* tool_data;
} rocprofiler_tool_configure_result_t;

typedef rocprofiler_tool_configure_result_t* (*rocprofiler_configure_func_t)(uint32_t version,
                                                                              const char* runtime_version,
                                                                              uint32_t priority,
                                                                              rocprofiler_client_id_t* client_id);

typedef struct rocprofiler_callback_thread_t {
  uint64_t handle;
} rocprofiler_callback_thread_t;

typedef enum rocprofiler_external_correlation_id_request_kind_t {
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_NONE = 0,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_CORE_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_AMD_EXT_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_IMAGE_EXT_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_FINALIZE_EXT_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_COMPILER_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CORE_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CONTROL_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_NAME_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_SCRATCH_MEMORY,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_RCCL_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_OMPT,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCDECODE_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCJPEG_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CORE_RANGE_API,
  ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST,
} rocprofiler_external_correlation_id_request_kind_t;

typedef int (*rocprofiler_external_correlation_id_request_cb_t)(
    rocprofiler_thread_id_t thread_id, rocprofiler_context_id_t context_id,
    rocprofiler_external_correlation_id_request_kind_t kind, rocprofiler_tracing_operation_t operation,
    uint64_t internal_corr_id_value, rocprofiler_user_data_t* external_corr_id_value, void* data);

typedef void (*rocprofiler_intercept_library_cb_t)(rocprofiler_intercept_table_t type, uint64_t lib_version,
                                                   uint64_t lib_instance, void** tables, uint64_t num_tables,
                                                   void* user_data);

}  // namespace vgpu::amd::rocprof
