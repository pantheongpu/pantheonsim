// The device runtime's shared half (vgpu/exec/devrt.hpp).
#include "vgpu/exec/devrt.hpp"

#include <algorithm>
#include <cstring>

namespace vgpu::exec::devrt {

namespace {
std::atomic<uint64_t> g_streams{0}, g_events{0};

struct Entry {
  const char* base;   // the name after "cuda" / "__cudaCDP2"
  Fn fn;
};
const Entry kEntries[] = {
    {"GetParameterBufferV2", Fn::GetParameterBufferV2},
    {"GetParameterBuffer", Fn::GetParameterBuffer},
    {"LaunchDeviceV2", Fn::LaunchDeviceV2},
    {"LaunchDeviceV2_ptsz", Fn::LaunchDeviceV2},
    {"LaunchDevice", Fn::LaunchDevice},
    {"LaunchDevice_ptsz", Fn::LaunchDevice},
    {"GetLastError", Fn::GetLastError},
    {"PeekAtLastError", Fn::PeekAtLastError},
    {"GetErrorString", Fn::GetErrorString},
    {"GetErrorName", Fn::GetErrorName},
    {"GetDevice", Fn::GetDevice},
    {"GetDeviceCount", Fn::GetDeviceCount},
    {"RuntimeGetVersion", Fn::RuntimeGetVersion},
    {"StreamCreateWithFlags", Fn::StreamCreateWithFlags},
    {"StreamDestroy", Fn::StreamDestroy},
    {"StreamWaitEvent", Fn::StreamWaitEvent},
    {"StreamWaitEvent_ptsz", Fn::StreamWaitEvent},
    {"EventCreateWithFlags", Fn::EventCreateWithFlags},
    {"EventDestroy", Fn::EventDestroy},
    {"EventRecord", Fn::EventRecord},
    {"EventRecord_ptsz", Fn::EventRecord},
    {"EventRecordWithFlags", Fn::EventRecordWithFlags},
    {"EventRecordWithFlags_ptsz", Fn::EventRecordWithFlags},
    {"Malloc", Fn::Malloc},
    {"Free", Fn::Free},
    {"MemcpyAsync", Fn::MemcpyAsync},
    {"MemcpyAsync_ptsz", Fn::MemcpyAsync},
    {"Memcpy2DAsync", Fn::Memcpy2DAsync},
    {"Memcpy2DAsync_ptsz", Fn::Memcpy2DAsync},
    {"Memcpy3DAsync", Fn::Memcpy3DAsync},
    {"Memcpy3DAsync_ptsz", Fn::Memcpy3DAsync},
    {"MemsetAsync", Fn::MemsetAsync},
    {"MemsetAsync_ptsz", Fn::MemsetAsync},
    {"Memset2DAsync", Fn::Memset2DAsync},
    {"Memset2DAsync_ptsz", Fn::Memset2DAsync},
    {"Memset3DAsync", Fn::Memset3DAsync},
    {"Memset3DAsync_ptsz", Fn::Memset3DAsync},
    {"FuncGetAttributes", Fn::FuncGetAttributes},
    {"DeviceGetAttribute", Fn::DeviceGetAttribute},
    {"DeviceGetLimit", Fn::DeviceGetLimit},
    {"DeviceGetCacheConfig", Fn::DeviceGetCacheConfig},
    {"DeviceGetSharedMemConfig", Fn::DeviceGetSharedMemConfig},
    {"OccupancyMaxActiveBlocksPerMultiprocessor", Fn::OccupancyMaxActiveBlocks},
    {"OccupancyMaxActiveBlocksPerMultiprocessorWithFlags", Fn::OccupancyMaxActiveBlocksWithFlags},
    {"DeviceSynchronize", Fn::DeviceSynchronize},
};

const std::unordered_map<std::string, Fn>& table() {
  static const std::unordered_map<std::string, Fn> t = [] {
    std::unordered_map<std::string, Fn> m;
    for (const Entry& e : kEntries) {
      m.emplace(std::string("__cudaCDP2") + e.base, e.fn);
      m.emplace(std::string("cuda") + e.base, e.fn);
    }
    // CDP2 declares cudaDeviceSynchronize's replacement under this name.
    m.emplace("__cudaDeviceSynchronizeDeprecationAvoidance", Fn::DeviceSynchronize);
    return m;
  }();
  return t;
}
}  // namespace

Fn lookup(const std::string& callee) {
  const auto& t = table();
  const auto it = t.find(callee);
  return it == t.end() ? Fn::None : it->second;
}

const std::vector<std::string>& names() {
  static const std::vector<std::string> v = [] {
    std::vector<std::string> out;
    for (const Entry& e : kEntries) {
      out.push_back(std::string("__cudaCDP2") + e.base);
      out.push_back(std::string("cuda") + e.base);
    }
    out.push_back("__cudaDeviceSynchronizeDeprecationAvoidance");
    return out;
  }();
  return v;
}

StreamKind stream_kind(uint64_t h) {
  switch (h) {
    case 0: return StreamKind::Default;
    case 1: return StreamKind::Legacy;
    case 2: return StreamKind::PerThread;
    case 3: return StreamKind::Tail;
    case 4: return StreamKind::FireAndForget;
    default: break;
  }
  const uint64_t n = h - kStreamBase;
  return h >= kStreamBase && n >= 1 && n <= g_streams.load(std::memory_order_relaxed) ? StreamKind::Named
                                                                                      : StreamKind::Invalid;
}

uint64_t new_stream() { return kStreamBase + g_streams.fetch_add(1, std::memory_order_relaxed) + 1; }
uint64_t new_event() { return kEventBase + g_events.fetch_add(1, std::memory_order_relaxed) + 1; }

bool valid_event(uint64_t h) {
  const uint64_t n = h - kEventBase;
  return h >= kEventBase && n >= 1 && n <= g_events.load(std::memory_order_relaxed);
}

void MemOp::run(MemoryManager& mem) const {
  if (!width || !height || !depth) return;
  std::vector<uint8_t> stage;
  constexpr uint64_t kChunk = 1u << 20;
  for (uint64_t z = 0; z < depth; ++z)
    for (uint64_t y = 0; y < height; ++y) {
      const uint64_t d = dst + z * dslice + y * dpitch;
      if (kind == Kind::Set) {
        mem.fill(d, &value, 1, width);
        continue;
      }
      const uint64_t s = src + z * sslice + y * spitch;
      for (uint64_t off = 0; off < width; off += kChunk) {
        const uint64_t n = std::min(kChunk, width - off);
        stage.resize(n);
        mem.read(s + off, stage.data(), n);
        mem.write(d + off, stage.data(), n);
      }
    }
}

bool config_ok(const std::array<uint32_t, 3>& grid, const std::array<uint32_t, 3>& block, uint64_t shared_total,
               const DeviceProfile& p, uint64_t max_threads, const std::array<uint32_t, 3>& req_block) {
  uint64_t threads = 1;
  for (int i = 0; i < 3; ++i) {
    if (!grid[i] || !block[i]) return false;
    if (block[i] > p.limits.max_block_dim[i] || grid[i] > p.limits.max_grid_dim[i]) return false;
    threads *= block[i];
  }
  if (threads > p.limits.max_threads_per_block) return false;
  if (max_threads && threads > max_threads) return false;
  for (int i = 0; i < 3; ++i)
    if (req_block[i] && block[i] != req_block[i]) return false;
  const uint64_t limit = std::max(p.limits.shared_mem_per_block, p.limits.shared_mem_per_block_optin);
  return shared_total <= limit;
}

uint32_t kernel_thread_limit(const ptx::EntryFn& fn, const DeviceProfile& p, uint32_t regs_per_thread) {
  uint64_t limit = p.limits.max_threads_per_block;
  const uint64_t bound = uint64_t{fn.max_ntid[0]} * std::max(1u, fn.max_ntid[1]) * std::max(1u, fn.max_ntid[2]);
  if (fn.max_ntid[0] && bound) limit = std::min(limit, bound);
  if (regs_per_thread && p.limits.registers_per_block && p.warp_size) {
    const uint64_t per_warp = (uint64_t{regs_per_thread} * p.warp_size + 255) / 256 * 256;
    limit = std::min<uint64_t>(limit, p.limits.registers_per_block / per_warp * p.warp_size);
  }
  return static_cast<uint32_t>(limit);
}

FuncAttrs func_attributes(const ptx::EntryFn& fn, const DeviceProfile& p, int ptx_arch, uint64_t constant) {
  // What the host's cudaFuncGetAttributes reports for the kernel (the device
  // runtime's matched it field for field on an RTX 3060).
  const KernelResources res = kernel_resources(fn, p, p.limits.max_threads_per_block, 0);
  FuncAttrs a;
  a.regs = static_cast<int32_t>(res.usage.regs_per_thread);
  a.local = res.usage.local_bytes;
  a.shared = fn.static_shared_size;
  a.constant = constant;
  a.max_threads = static_cast<int32_t>(kernel_thread_limit(fn, p, res.usage.regs_per_thread));
  a.binary = p.cc_major * 10 + p.cc_minor;
  a.ptx = ptx_arch ? ptx_arch : a.binary;
  return a;
}

void put_func_attrs(const FuncAttrs& a, uint8_t out[40]) {
  // sharedSizeBytes, constSizeBytes, localSizeBytes (size_t), then
  // maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion (int).
  std::memcpy(out + 0, &a.shared, 8);
  std::memcpy(out + 8, &a.constant, 8);
  std::memcpy(out + 16, &a.local, 8);
  std::memcpy(out + 24, &a.max_threads, 4);
  std::memcpy(out + 28, &a.regs, 4);
  std::memcpy(out + 32, &a.ptx, 4);
  std::memcpy(out + 36, &a.binary, 4);
}

}  // namespace vgpu::exec::devrt
