#include "machine.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>

#include "vgpu/ras.hpp"
#include "vgpu/registry.hpp"

namespace vgpu::cli {

namespace {
// Injected clock-event reasons (`vgpu fault throttle`) and the readings they
// imply, applied to what every command reads so nvidia-smi, rocm-smi and
// `vgpu fault show` agree. Unreadable reliability state leaves the readings
// as they were rather than failing the report.
void apply_faults(vgpu::telemetry::Shared* snap) {
  for (uint32_t i = 0; i < snap->device_count; ++i) {
    try {
      vgpu::ras::apply_throttle(snap->devices[i]);
      vgpu::ras::apply_link(snap->devices[i]);
    } catch (const std::exception&) {
    }
  }
}
bool read_live_or_idle(vgpu::telemetry::Shared* snap);
}  // namespace

bool read_machine(vgpu::telemetry::Shared* snap) {
  if (!read_live_or_idle(snap)) return false;
  apply_faults(snap);
  return true;
}

namespace {
bool read_live_or_idle(vgpu::telemetry::Shared* snap) {
  if (vgpu::telemetry::read_snapshot(snap)) return true;
  // Nothing is publishing, which on a real machine is the ordinary case:
  // nvidia-smi answers about an idle GPU rather than failing. Monitoring
  // tools poll before and after a workload and treat a non-zero exit as "no
  // GPU", so describe the configured rack as idle instead.
  const char* gpu = std::getenv("VGPU_GPU");
  int count = 1;
  if (const char* c = std::getenv("VGPU_DEVICE_COUNT"); c && c[0]) count = std::atoi(c);
  try {
    vgpu::DeviceProfile p = vgpu::load_gpu(gpu && *gpu ? gpu : "nvidia/h100");
    vgpu::apply_vram_override(p);   // the card the session's programs see
    *snap = vgpu::telemetry::idle_snapshot(p, count);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vgpu smi: no running VirtualGPU and no usable profile (%s)\n", e.what());
    return false;
  }
  return true;
}
}  // namespace

}  // namespace vgpu::cli
