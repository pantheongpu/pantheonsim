// Reliability state: ECC counts, retired pages, remapped rows and PCIe errors,
// shared by every process that touches the machine and outliving all of them.
#include "vgpu/ras.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "vtest.hpp"

using namespace vgpu;
using ras::Location;
using ras::Retirement;
using ras::Severity;

namespace {
// Each test gets its own runtime and state directories, so none of them sees
// another's counts or the user's own.
struct TempMachine {
  std::string root;
  explicit TempMachine(const char* tag) {
    root = std::string("/tmp/vgpu-ras-test-") + tag + "-" + std::to_string(getpid());
    std::filesystem::remove_all(root);
    setenv("VGPU_TELEMETRY_PATH", (root + "/run").c_str(), 1);
    setenv("VGPU_STATE_DIR", (root + "/state").c_str(), 1);
  }
  ~TempMachine() {
    unsetenv("VGPU_TELEMETRY_PATH");
    unsetenv("VGPU_STATE_DIR");
    std::filesystem::remove_all(root);
  }
};
constexpr const char* kGpu = "GPU-00000000-0000-4000-8000-000000000000";
uint64_t ecc(const ras::Counters& c, Severity s, Location l) {
  return c.ecc[static_cast<uint32_t>(s)][static_cast<uint32_t>(l)];
}
}  // namespace

VTEST(a_device_nobody_touched_reads_zero) {
  TempMachine m("zero");
  const ras::State st = ras::read(kGpu);
  VCHECK_EQ(st.since_load.ecc_total(Severity::Corrected), 0u);
  VCHECK_EQ(st.lifetime.ecc_total(Severity::Uncorrected), 0u);
  VCHECK_EQ(st.lifetime.retired_dbe, 0u);
  VCHECK(!std::filesystem::exists(m.root));   // reading creates nothing
}

VTEST(injected_errors_count_in_both_lifetimes) {
  TempMachine m("both");
  ras::inject_ecc(kGpu, Severity::Corrected, Location::L2Cache, 3, Retirement::Pages);
  ras::inject_ecc(kGpu, Severity::Corrected, Location::DeviceMemory, 2, Retirement::Pages);
  const ras::State st = ras::read(kGpu);
  VCHECK_EQ(ecc(st.since_load, Severity::Corrected, Location::L2Cache), 3u);
  VCHECK_EQ(ecc(st.lifetime, Severity::Corrected, Location::L2Cache), 3u);
  VCHECK_EQ(st.since_load.ecc_total(Severity::Corrected), 5u);
  VCHECK_EQ(st.since_load.ecc_total(Severity::Uncorrected), 0u);
  // Another device's state is separate.
  VCHECK_EQ(ras::read("GPU-other").since_load.ecc_total(Severity::Corrected), 0u);
}

VTEST(an_uncorrected_memory_error_retires_a_page_on_gddr_and_remaps_a_row_on_hbm) {
  TempMachine m("retire");
  ras::inject_ecc("GPU-gddr", Severity::Uncorrected, Location::DeviceMemory, 2, Retirement::Pages);
  ras::inject_ecc("GPU-hbm", Severity::Uncorrected, Location::DeviceMemory, 1, Retirement::Rows);
  const ras::Counters gddr = ras::read("GPU-gddr").lifetime;
  const ras::Counters hbm = ras::read("GPU-hbm").lifetime;
  VCHECK_EQ(gddr.retired_dbe, 2u);
  VCHECK_EQ(gddr.retired_pending, 1u);
  VCHECK_EQ(gddr.rows_uncorrectable, 0u);
  VCHECK_EQ(hbm.rows_uncorrectable, 1u);
  VCHECK_EQ(hbm.rows_pending, 1u);
  VCHECK_EQ(hbm.retired_dbe, 0u);
}

VTEST(only_uncorrected_device_memory_errors_take_memory_out_of_service) {
  TempMachine m("scope");
  ras::inject_ecc(kGpu, Severity::Corrected, Location::DeviceMemory, 5, Retirement::Pages);
  ras::inject_ecc(kGpu, Severity::Uncorrected, Location::RegisterFile, 1, Retirement::Pages);
  ras::inject_ecc("GPU-noecc", Severity::Uncorrected, Location::DeviceMemory, 1, Retirement::None);
  VCHECK_EQ(ras::read(kGpu).lifetime.retired_dbe, 0u);
  VCHECK_EQ(ras::read(kGpu).lifetime.retired_pending, 0u);
  VCHECK_EQ(ras::read("GPU-noecc").lifetime.retired_dbe, 0u);
}

VTEST(a_driver_reload_zeroes_volatile_counts_and_completes_pending_retirement) {
  TempMachine m("reload");
  ras::inject_ecc(kGpu, Severity::Uncorrected, Location::DeviceMemory, 1, Retirement::Pages);
  ras::inject_pcie(kGpu, ras::Pcie::Replay, 4);
  ras::reset_volatile(kGpu);
  const ras::State st = ras::read(kGpu);
  VCHECK_EQ(st.since_load.ecc_total(Severity::Uncorrected), 0u);
  VCHECK_EQ(st.since_load.pcie[static_cast<uint32_t>(ras::Pcie::Replay)], 0u);
  VCHECK_EQ(st.lifetime.ecc_total(Severity::Uncorrected), 1u);
  VCHECK_EQ(st.lifetime.retired_dbe, 1u);
  VCHECK_EQ(st.lifetime.retired_pending, 0u);
}

VTEST(nvidia_smi_p0_zeroes_counts_but_leaves_retirement_pending) {
  TempMachine m("p0");
  ras::inject_ecc(kGpu, Severity::Uncorrected, Location::DeviceMemory, 1, Retirement::Rows);
  ras::reset_volatile(kGpu, /*driver_reload=*/false);
  const ras::State st = ras::read(kGpu);
  VCHECK_EQ(st.since_load.ecc_total(Severity::Uncorrected), 0u);
  VCHECK_EQ(st.lifetime.rows_pending, 1u);
}

VTEST(an_aggregate_reset_keeps_memory_already_taken_out_of_service) {
  TempMachine m("p1");
  ras::inject_ecc(kGpu, Severity::Uncorrected, Location::DeviceMemory, 3, Retirement::Pages);
  ras::reset_aggregate(kGpu);
  const ras::State st = ras::read(kGpu);
  VCHECK_EQ(st.lifetime.ecc_total(Severity::Uncorrected), 0u);
  VCHECK_EQ(st.lifetime.retired_dbe, 3u);
  VCHECK_EQ(st.since_load.ecc_total(Severity::Uncorrected), 3u);   // -p 1 leaves volatile alone
}

VTEST(names_parse_as_nvidia_smi_spells_them) {
  Location l;
  VCHECK(ras::parse_location("dram", &l) && l == Location::DeviceMemory);
  VCHECK(ras::parse_location("texture_memory", &l) && l == Location::TextureMemory);
  VCHECK(!ras::parse_location("total", &l));
  VCHECK(!ras::parse_location("DRAM", &l));
  ras::Pcie p;
  VCHECK(ras::parse_pcie("non_fatal", &p) && p == ras::Pcie::NonFatal);
  VCHECK(!ras::parse_pcie("nonfatal", &p));
  VCHECK_EQ(std::string(ras::location_name(Location::Cbu)), std::string("cbu"));
}

VTEST(concurrent_injections_all_count) {
  TempMachine m("threads");
  std::vector<std::thread> workers;
  for (int t = 0; t < 8; ++t)
    workers.emplace_back([] {
      for (int i = 0; i < 200; ++i)
        ras::inject_ecc(kGpu, Severity::Corrected, Location::DeviceMemory, 1, Retirement::Pages);
    });
  for (auto& w : workers) w.join();
  VCHECK_EQ(ras::read(kGpu).lifetime.ecc_total(Severity::Corrected), 1600u);
}

VTEST(a_state_file_from_another_build_is_started_again) {
  TempMachine m("stale");
  ras::inject_ecc(kGpu, Severity::Corrected, Location::DeviceMemory, 7, Retirement::Pages);
  const std::string path = ras::state_dir() + "/ras-" + kGpu + ".aggregate";
  const int fd = ::open(path.c_str(), O_WRONLY);
  VCHECK(fd >= 0);
  const uint32_t junk = 0xDEADBEEF;
  VCHECK(::pwrite(fd, &junk, sizeof junk, 0) == static_cast<ssize_t>(sizeof junk));
  ::close(fd);
  VCHECK_EQ(ras::read(kGpu).lifetime.ecc_total(Severity::Corrected), 0u);
  ras::inject_ecc(kGpu, Severity::Corrected, Location::DeviceMemory, 1, Retirement::Pages);
  VCHECK_EQ(ras::read(kGpu).lifetime.ecc_total(Severity::Corrected), 1u);
}

VTEST_MAIN
