/*
 * vgpu_nvcomp.hpp -- VirtualGPU's clean-room declarations of nvCOMP's C++
 * high-level interface (the "manager" API), version 5.3.
 *
 * Written from NVIDIA's public documentation and the declarations of its
 * public headers (nvcomp.hpp, nvcomp/nvcompManager.hpp, the per-format
 * managers, nvcompManagerFactory.hpp, formatSpec.hpp). It contains no NVIDIA
 * code. The classes are laid out member for member and virtual function for
 * virtual function as those headers declare them, because a program compiled
 * against NVIDIA's headers allocates the managers and calls through their
 * vtables: a program compiled with either header works with either library.
 */
#ifndef VGPU_NVCOMP_HPP_
#define VGPU_NVCOMP_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vgpu_nvcomp.h"

namespace nvcomp {

enum class nvcompFormatType_t : uint8_t {
  LZ4 = 0,
  Snappy = 1,
  ANS = 2,
  GDeflate = 3,
  Cascaded = 4,
  Bitcomp = 5,
  Zstd = 6,
  Deflate = 7,
  Gzip = 8,
  NotSupportedError = 255
};

class NVCompException : public std::runtime_error {
 public:
  NVCompException(nvcompStatus_t err, const std::string& msg)
      : std::runtime_error(msg + " : code=" + std::to_string(err) + "."), m_err(err) {}
  nvcompStatus_t get_error() const noexcept { return m_err; }

 private:
  nvcompStatus_t m_err;
};

using AllocFn_t = std::function<void*(size_t)>;
using DeAllocFn_t = std::function<void(void*, size_t)>;

/* How a compressed buffer is laid out: nvCOMP's own container (a header, the
 * chunks' offsets and sizes, the chunks), the format's bare bitstream as one
 * chunk, or that bitstream after its uncompressed size. */
enum class BitstreamKind { NVCOMP_NATIVE = 0, RAW = 1, WITH_UNCOMPRESSED_SIZE = 2 };

enum ChecksumPolicy : int {
  NoComputeNoVerify = 0,
  ComputeAndNoVerify = 1,
  NoComputeAndVerifyIfPresent = 2,
  ComputeAndVerifyIfPresent = 3,
  ComputeAndVerify = 4
};

struct CompressionConfig {
  struct CompressionConfigImpl;
  std::shared_ptr<CompressionConfigImpl> impl;
  size_t uncompressed_buffer_size;
  size_t max_compressed_buffer_size;
  size_t num_chunks;
  bool compute_checksums;

  NVCOMP_EXPORT CompressionConfig();
  NVCOMP_EXPORT CompressionConfig(size_t uncompressed_buffer_size);
  NVCOMP_EXPORT nvcompStatus_t* get_status() const;
  NVCOMP_EXPORT CompressionConfig(CompressionConfig&& other);
  NVCOMP_EXPORT CompressionConfig(const CompressionConfig& other);
  NVCOMP_EXPORT CompressionConfig& operator=(CompressionConfig&& other);
  NVCOMP_EXPORT CompressionConfig& operator=(const CompressionConfig& other);
  NVCOMP_EXPORT ~CompressionConfig() noexcept;
};

struct DecompressionConfig {
  struct DecompressionConfigImpl;
  std::shared_ptr<DecompressionConfigImpl> impl;
  size_t decomp_data_size;
  size_t num_chunks;
  bool checksums_present;

  NVCOMP_EXPORT DecompressionConfig();
  NVCOMP_EXPORT nvcompStatus_t* get_status() const;
  NVCOMP_EXPORT DecompressionConfig(DecompressionConfig&& other);
  NVCOMP_EXPORT DecompressionConfig(const DecompressionConfig& other);
  NVCOMP_EXPORT DecompressionConfig& operator=(DecompressionConfig&& other);
  NVCOMP_EXPORT DecompressionConfig& operator=(const DecompressionConfig& other);
  NVCOMP_EXPORT ~DecompressionConfig() noexcept;
};

/* The virtual functions, in the order of NVIDIA's declaration: that order is
 * the vtable's. */
struct nvcompManagerBase {
  virtual CompressionConfig configure_compression(const size_t uncomp_buffer_size) = 0;
  virtual std::vector<CompressionConfig> configure_compression(const std::vector<size_t>& uncomp_buffer_sizes) = 0;
  virtual void compress(const uint8_t* uncomp_buffer, uint8_t* comp_buffer, const CompressionConfig& comp_config,
                        size_t* comp_size = nullptr) = 0;
  virtual void compress(const uint8_t* const* uncomp_buffers, uint8_t* const* comp_buffers,
                        const std::vector<CompressionConfig>& comp_configs, size_t* comp_sizes = nullptr) = 0;
  virtual DecompressionConfig configure_decompression(const uint8_t* comp_buffer,
                                                      const size_t* comp_size = nullptr) = 0;
  virtual std::vector<DecompressionConfig> configure_decompression(const uint8_t* const* comp_buffers,
                                                                   size_t batch_size,
                                                                   const size_t* comp_sizes = nullptr) = 0;
  virtual DecompressionConfig configure_decompression(const CompressionConfig& comp_config) = 0;
  virtual std::vector<DecompressionConfig> configure_decompression(
      const std::vector<CompressionConfig>& comp_configs) = 0;
  virtual void decompress(uint8_t* decomp_buffer, const uint8_t* comp_buffer, const DecompressionConfig& decomp_config,
                          size_t* comp_size = nullptr) = 0;
  virtual void decompress(uint8_t* const* decomp_buffers, const uint8_t* const* comp_buffers,
                          const std::vector<DecompressionConfig>& decomp_configs,
                          const size_t* comp_sizes = nullptr) = 0;
  virtual void set_scratch_allocators(const AllocFn_t& alloc_fn, const DeAllocFn_t& dealloc_fn) = 0;
  virtual size_t get_compressed_output_size(const uint8_t* comp_buffer) = 0;
  virtual std::vector<size_t> get_compressed_output_size(const uint8_t* const* comp_buffers, size_t batch_size) = 0;
  virtual void deallocate_gpu_mem() = 0;
  virtual ~nvcompManagerBase() noexcept = default;
  virtual void decompress(uint8_t* const* decomp_buffers, const uint8_t* const* comp_buffers,
                          const std::vector<DecompressionConfig>& decomp_configs, const size_t* comp_sizes,
                          const size_t batch_count, const uint8_t* const* host_comp_buffers) = 0;
  virtual size_t get_decompressed_output_size(const uint8_t* comp_buffer) = 0;
  virtual std::vector<size_t> get_decompressed_output_size(const uint8_t* const* comp_buffers,
                                                           size_t batch_size) = 0;
};

namespace detail {

struct nvcompManagerInternalBase;

struct NVCOMP_EXPORT PimplManager : nvcompManagerBase {
  std::unique_ptr<nvcompManagerInternalBase> impl;

  PimplManager() noexcept = default;
  NVCOMP_EXPORT explicit PimplManager(std::unique_ptr<nvcompManagerInternalBase> p) noexcept;
  PimplManager(const PimplManager&) = delete;
  virtual ~PimplManager() noexcept = default;
  PimplManager& operator=(const PimplManager&) = delete;

  NVCOMP_EXPORT CompressionConfig configure_compression(const size_t uncomp_buffer_size) override;
  NVCOMP_EXPORT std::vector<CompressionConfig> configure_compression(
      const std::vector<size_t>& uncomp_buffer_sizes) override;
  NVCOMP_EXPORT void compress(const uint8_t* uncomp_buffer, uint8_t* comp_buffer,
                              const CompressionConfig& comp_config, size_t* comp_size = nullptr) override;
  NVCOMP_EXPORT void compress(const uint8_t* const* uncomp_buffers, uint8_t* const* comp_buffers,
                              const std::vector<CompressionConfig>& comp_configs,
                              size_t* comp_sizes = nullptr) override;
  NVCOMP_EXPORT DecompressionConfig configure_decompression(const uint8_t* comp_buffer,
                                                            const size_t* comp_size = nullptr) override;
  NVCOMP_EXPORT std::vector<DecompressionConfig> configure_decompression(const uint8_t* const* comp_buffers,
                                                                         size_t batch_size,
                                                                         const size_t* comp_sizes = nullptr) override;
  NVCOMP_EXPORT DecompressionConfig configure_decompression(const CompressionConfig& comp_config) override;
  NVCOMP_EXPORT std::vector<DecompressionConfig> configure_decompression(
      const std::vector<CompressionConfig>& comp_configs) override;
  NVCOMP_EXPORT void decompress(uint8_t* decomp_buffer, const uint8_t* comp_buffer,
                                const DecompressionConfig& decomp_config, size_t* comp_size = nullptr) override;
  NVCOMP_EXPORT void decompress(uint8_t* const* decomp_buffers, const uint8_t* const* comp_buffers,
                                const std::vector<DecompressionConfig>& decomp_configs,
                                const size_t* comp_sizes = nullptr) override;
  NVCOMP_EXPORT void set_scratch_allocators(const AllocFn_t& alloc_fn, const DeAllocFn_t& dealloc_fn) override;
  NVCOMP_EXPORT size_t get_compressed_output_size(const uint8_t* comp_buffer) override;
  NVCOMP_EXPORT std::vector<size_t> get_compressed_output_size(const uint8_t* const* comp_buffers,
                                                               size_t batch_size) override;
  NVCOMP_EXPORT void deallocate_gpu_mem() override;
  NVCOMP_EXPORT void decompress(uint8_t* const* decomp_buffers, const uint8_t* const* comp_buffers,
                                const std::vector<DecompressionConfig>& decomp_configs, const size_t* comp_sizes,
                                const size_t batch_count, const uint8_t* const* host_comp_buffers) override;
  NVCOMP_EXPORT size_t get_decompressed_output_size(const uint8_t* comp_buffer) override;
  NVCOMP_EXPORT std::vector<size_t> get_decompressed_output_size(const uint8_t* const* comp_buffers,
                                                                 size_t batch_size) override;
};

}  // namespace detail

#define VGPU_NVCOMP_MANAGER(M, F)                                                                                \
  struct M : detail::PimplManager {                                                                              \
    NVCOMP_EXPORT M(size_t uncomp_chunk_size,                                                                    \
                    const nvcompBatched##F##CompressOpts_t& compress_opts = nvcompBatched##F##CompressDefaultOpts, \
                    const nvcompBatched##F##DecompressOpts_t& decompress_opts =                                  \
                        nvcompBatched##F##DecompressDefaultOpts,                                                  \
                    cudaStream_t user_stream = 0, ChecksumPolicy checksum_policy = NoComputeNoVerify,            \
                    BitstreamKind bitstream_kind = BitstreamKind::NVCOMP_NATIVE);                                \
    NVCOMP_EXPORT ~M() noexcept;                                                                                 \
  };
VGPU_NVCOMP_MANAGER(LZ4Manager, LZ4)
VGPU_NVCOMP_MANAGER(SnappyManager, Snappy)
VGPU_NVCOMP_MANAGER(DeflateManager, Deflate)
VGPU_NVCOMP_MANAGER(GdeflateManager, Gdeflate)
VGPU_NVCOMP_MANAGER(GzipManager, Gzip)
VGPU_NVCOMP_MANAGER(ZstdManager, Zstd)
VGPU_NVCOMP_MANAGER(CascadedManager, Cascaded)
VGPU_NVCOMP_MANAGER(BitcompManager, Bitcomp)
VGPU_NVCOMP_MANAGER(ANSManager, ANS)
#undef VGPU_NVCOMP_MANAGER

NVCOMP_EXPORT std::shared_ptr<nvcompManagerBase> create_manager(
    const uint8_t* comp_buffer, cudaStream_t stream = 0, ChecksumPolicy checksum_policy = NoComputeNoVerify,
    nvcompDecompressBackend_t backend = NVCOMP_DECOMPRESS_BACKEND_DEFAULT, bool use_de_sort = false);

NVCOMP_EXPORT nvcompFormatType_t get_compression_format(const uint8_t* comp_buffer, cudaStream_t stream = 0);

}  // namespace nvcomp

#endif /* VGPU_NVCOMP_HPP_ */

// nvcomp/native/streaming_gzip.hpp: gzip from one C++ stream to another, in windows,
// with the workspace the caller allocates on the device. (C++ linkage, as NVIDIA's.)
nvcompStatus_t nvcompGzipStreamingDecompressGetTempSize(size_t* temp_bytes);
nvcompStatus_t nvcompGzipStreamingDecompress(std::istream& input_stream, std::ostream& output_stream, const size_t temp_bytes,
                                             void* const device_temp_ptr, cudaStream_t stream);
nvcompStatus_t nvcompGzipStreamingCompressGetTempSize(nvcompBatchedGzipCompressOpts_t opts, size_t* temp_bytes);
nvcompStatus_t nvcompGzipStreamingCompress(std::istream& input_stream, std::ostream& output_stream, const size_t temp_bytes,
                                           void* const device_temp_ptr, nvcompBatchedGzipCompressOpts_t opts, cudaStream_t stream);
