// Reliability state: ECC counts, retired pages, remapped rows and PCIe errors,
// shared by every process that touches the machine and outliving all of them.
#include "vgpu/ras.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <stdexcept>
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

VTEST(armed_faults_are_taken_most_severe_first_and_only_once) {
  TempMachine m("armed");
  ras::ArmedFaults faults(kGpu);
  VCHECK_EQ(*faults.pending(), 0u);
  ras::arm(kGpu, ras::Armed::Corrected, 1);
  ras::arm(kGpu, ras::Armed::Bitflip, 1);
  ras::arm(kGpu, ras::Armed::Uncorrected, 1);
  VCHECK_EQ(*faults.pending(), 3u);   // the same file, seen through this process's mapping
  VCHECK(faults.take() == ras::Armed::Uncorrected);
  VCHECK(faults.take() == ras::Armed::Bitflip);
  VCHECK(faults.take() == ras::Armed::Corrected);
  VCHECK(faults.take() == ras::Armed::None);
  VCHECK_EQ(*faults.pending(), 0u);
}

VTEST(a_fault_is_taken_only_by_the_accesses_it_was_armed_on) {
  TempMachine m("targets");
  ras::ArmedFaults faults(kGpu);
  ras::arm(kGpu, ras::Armed::Bitflip, 2, ras::Target::Store);
  ras::arm(kGpu, ras::Armed::Uncorrected, 1, ras::Target::Shared);
  VCHECK_EQ(*faults.pending(ras::Target::Load), 0u);   // loads still cost one zero read
  VCHECK_EQ(*faults.pending(ras::Target::Store), 2u);
  VCHECK_EQ(*faults.pending(ras::Target::Shared), 1u);
  VCHECK(faults.take(ras::Target::Load) == ras::Armed::None);
  VCHECK(faults.take(ras::Target::Shared) == ras::Armed::Uncorrected);
  VCHECK(faults.take(ras::Target::Store) == ras::Armed::Bitflip);
  VCHECK_EQ(ras::read(kGpu).since_load.armed_total(), 1u);
}

VTEST(an_ecc_error_cannot_be_armed_on_stores) {
  TempMachine m("store-ecc");
  bool refused = false;
  try {
    ras::arm(kGpu, ras::Armed::Uncorrected, 1, ras::Target::Store);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  VCHECK(refused);
  ras::Target t{};
  VCHECK(ras::parse_target("shared", &t) && t == ras::Target::Shared);
  VCHECK(!ras::parse_target("texture", &t));
  VCHECK_EQ(std::string(ras::target_name(ras::Target::Store)), std::string("store"));
}

VTEST(concurrent_loads_take_exactly_what_was_armed) {
  TempMachine m("race");
  ras::ArmedFaults faults(kGpu);
  ras::arm(kGpu, ras::Armed::Bitflip, 100);
  std::atomic<int> taken{0};
  std::vector<std::thread> workers;
  for (int t = 0; t < 8; ++t)
    workers.emplace_back([&] {
      for (int i = 0; i < 50; ++i)
        if (faults.take() == ras::Armed::Bitflip) ++taken;
    });
  for (auto& w : workers) w.join();
  VCHECK_EQ(taken.load(), 100);
  VCHECK_EQ(*faults.pending(), 0u);
}

VTEST(a_driver_reload_disarms) {
  TempMachine m("disarm");
  ras::arm(kGpu, ras::Armed::Uncorrected, 2);
  ras::reset_volatile(kGpu);
  VCHECK_EQ(ras::read(kGpu).since_load.armed_total(), 0u);
}

VTEST(log_lines_follow_the_driver_and_kernel_forms) {
  VCHECK_EQ(ras::xid_line("00000000:01:00.0", 48, "pid=42, name=python3", "An uncorrectable error."),
            std::string("NVRM: Xid (PCI:0000:01:00): 48, pid=42, name=python3, An uncorrectable error."));
  VCHECK_EQ(ras::xid_line("00000000:3B:00.0", 63, "", "Row Remapper: New row marked."),
            std::string("NVRM: Xid (PCI:0000:3b:00): 63, pid='<unknown>', name=<unknown>, Row Remapper: New row marked."));
  VCHECK_EQ(ras::aer_line("00000000:02:00.0", ras::Pcie::Replay),
            std::string("pcieport 0000:00:01.0: AER: Corrected error received: 0000:02:00.0"));
  VCHECK_EQ(ras::aer_line("00000000:02:00.0", ras::Pcie::NonFatal),
            std::string("pcieport 0000:00:01.0: AER: Uncorrected (Non-Fatal) error received: 0000:02:00.0"));
  VCHECK_EQ(ras::aer_line("00000000:02:00.0", ras::Pcie::NaksSent), std::string(""));
}

VTEST(the_kernel_log_is_written_only_inside_a_session) {
  TempMachine m("dmesg");
  unsetenv("VGPU_SESSION");
  ras::log_kernel("nowhere to go");   // outside a session: nothing, and no error
  const std::string sess = m.root + "/session";
  std::filesystem::create_directories(sess);
  { std::ofstream(sess + "/dmesg.log") << "[    7.000000] booted\n"; }
  setenv("VGPU_SESSION", sess.c_str(), 1);
  ras::log_kernel("first");
  ras::log_kernel("second");
  unsetenv("VGPU_SESSION");
  std::ifstream in(sess + "/dmesg.log");
  std::string line;
  std::vector<double> stamps;
  std::vector<std::string> texts;
  while (std::getline(in, line)) {
    double t = 0;
    char text[64] = {0};
    if (std::sscanf(line.c_str(), "[%lf] %63[^\n]", &t, text) == 2) {
      stamps.push_back(t);
      texts.emplace_back(text);
    }
  }
  VCHECK_EQ(texts.size(), 3u);
  VCHECK_EQ(texts[2], std::string("second"));
  VCHECK(stamps[1] > stamps[0] && stamps[2] > stamps[1]);
}

VTEST(a_hang_is_armed_for_launches_not_loads) {
  TempMachine m("hang");
  ras::ArmedFaults faults(kGpu);
  ras::arm_hang(kGpu, 3);
  VCHECK_EQ(*faults.pending(), 0u);   // loads never see a hang
  uint64_t seconds = 0;
  VCHECK(faults.take_hang(&seconds));
  VCHECK_EQ(seconds, 3u);
  VCHECK(!faults.take_hang(&seconds));
}

VTEST(throttle_reasons_are_active_for_their_window_and_their_time_is_kept) {
  TempMachine m("throttle");
  VCHECK_EQ(ras::reason_bit("sw_thermal_slowdown"), ras::kSwThermalSlowdown);
  VCHECK_EQ(ras::reason_bit("gpu_idle"), 0u);   // reported, never injected
  ras::throttle(kGpu, ras::kSwThermalSlowdown | ras::kSwPowerCap, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  telemetry::DeviceSample d{};
  std::snprintf(d.uuid, sizeof d.uuid, "%s", kGpu);
  d.temperature_c = 40;
  d.temperature_max_c = 90;
  d.power_mw = 20000;
  d.power_limit_mw = 70000;
  d.sm_clock_max_mhz = 1600;
  d.sm_clock_mhz = 1600;
  VCHECK_EQ(ras::apply_throttle(d), ras::kSwThermalSlowdown | ras::kSwPowerCap);
  VCHECK_EQ(d.temperature_c, 90u);    // at the slowdown threshold
  VCHECK_EQ(d.power_mw, 70000u);      // held at the limit
  VCHECK_EQ(d.sm_clock_mhz, 1200u);   // three quarters, for software reasons
  VCHECK(ras::throttle_time_us(kGpu, ras::kSwThermalSlowdown) >= 20000);
  ras::clear_throttle(kGpu);
  const uint64_t kept = ras::throttle_time_us(kGpu, ras::kSwThermalSlowdown);
  VCHECK(kept >= 20000);
  telemetry::DeviceSample after{};
  std::snprintf(after.uuid, sizeof after.uuid, "%s", kGpu);
  VCHECK_EQ(ras::apply_throttle(after), 0u);
  VCHECK_EQ(ras::throttle_time_us(kGpu, ras::kSwThermalSlowdown), kept);   // stopped counting
}

VTEST(a_throttle_window_ends_by_itself) {
  TempMachine m("expiry");
  ras::throttle(kGpu, ras::kHwSlowdown, 1);
  telemetry::DeviceSample d{};
  std::snprintf(d.uuid, sizeof d.uuid, "%s", kGpu);
  d.sm_clock_max_mhz = 2000;
  d.sm_clock_mhz = 2000;
  VCHECK_EQ(ras::apply_throttle(d), ras::kHwSlowdown);
  VCHECK_EQ(d.sm_clock_mhz, 1000u);   // half, for a hardware reason
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  VCHECK_EQ(ras::apply_throttle(d), 0u);
  const uint64_t us = ras::throttle_time_us(kGpu, ras::kHwSlowdown);
  VCHECK(us >= 990000 && us <= 1010000);   // the window, not the time since
}

VTEST(events_are_read_in_order_from_where_the_reader_started) {
  TempMachine m("events");
  ras::record_event(kGpu, ras::kEventXid, 13);   // before the reader: not seen
  uint64_t after = ras::event_head(kGpu);
  VCHECK_EQ(after, 1u);
  ras::inject_ecc(kGpu, Severity::Corrected, Location::DeviceMemory, 4, Retirement::Pages);
  ras::record_event(kGpu, ras::kEventXid, 48);
  ras::Event e{};
  VCHECK(ras::next_event(kGpu, ~0ull, &after, &e));
  VCHECK_EQ(e.type, ras::kEventSingleBitEcc);
  VCHECK(e.time_ns > 0);
  VCHECK(ras::next_event(kGpu, ~0ull, &after, &e));
  VCHECK_EQ(e.type, ras::kEventXid);
  VCHECK_EQ(e.data, 48u);
  VCHECK(!ras::next_event(kGpu, ~0ull, &after, &e));
}

VTEST(a_reader_sees_only_the_types_it_asked_for) {
  TempMachine m("mask");
  uint64_t after = ras::event_head(kGpu);
  ras::record_event(kGpu, ras::kEventSingleBitEcc, 0);
  ras::record_event(kGpu, ras::kEventXid, 63);
  ras::Event e{};
  VCHECK(ras::next_event(kGpu, ras::kEventXid | ras::kEventDoubleBitEcc, &after, &e));
  VCHECK_EQ(e.data, 63u);
  VCHECK_EQ(after, 2u);   // the skipped event is behind it too
}

VTEST(a_reader_that_fell_behind_resumes_at_the_oldest_event_kept) {
  TempMachine m("wrap");
  uint64_t after = ras::event_head(kGpu);
  for (uint64_t i = 1; i <= ras::kEvents + 5; ++i) ras::record_event(kGpu, ras::kEventXid, i);
  ras::Event e{};
  VCHECK(ras::next_event(kGpu, ~0ull, &after, &e));
  VCHECK_EQ(e.data, 6u);
  uint64_t n = 1;
  while (ras::next_event(kGpu, ~0ull, &after, &e)) ++n;
  VCHECK_EQ(n, uint64_t{ras::kEvents});
  VCHECK_EQ(e.data, uint64_t{ras::kEvents} + 5);
}

VTEST(a_driver_reload_starts_the_events_again) {
  TempMachine m("evreset");
  ras::record_event(kGpu, ras::kEventXid, 1);
  ras::record_event(kGpu, ras::kEventXid, 2);
  uint64_t after = ras::event_head(kGpu);
  ras::reset_volatile(kGpu);
  VCHECK_EQ(ras::event_head(kGpu), 0u);
  ras::record_event(kGpu, ras::kEventXid, 79);
  ras::Event e{};
  VCHECK(ras::next_event(kGpu, ~0ull, &after, &e));
  VCHECK_EQ(e.data, 79u);
}

VTEST(an_xid_is_logged_and_raised_as_an_event) {
  TempMachine m("xid");
  const std::string sess = m.root + "/session";
  std::filesystem::create_directories(sess);
  setenv("VGPU_SESSION", sess.c_str(), 1);
  uint64_t after = ras::event_head(kGpu);
  ras::report_xid(kGpu, "00000000:01:00.0", 48, "", "An uncorrectable error.");
  unsetenv("VGPU_SESSION");
  ras::Event e{};
  VCHECK(ras::next_event(kGpu, ras::kEventXid, &after, &e));
  VCHECK_EQ(e.data, 48u);
  std::ifstream in(sess + "/dmesg.log");
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  VCHECK(text.find("NVRM: Xid (PCI:0000:01:00): 48, ") != std::string::npos);
}

VTEST_MAIN
