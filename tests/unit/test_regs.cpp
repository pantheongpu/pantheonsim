// The register database and the configuration-space access engine: the
// database is well formed, the capability chains are walkable, and every
// access follows its register's semantics.
#include "vgpu/regs.hpp"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>

#include "vgpu/ras.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {
// A machine of its own, so no test sees another's writes or the user's.
struct TempMachine {
  std::string root;
  explicit TempMachine(const char* tag) {
    root = std::string("/tmp/vgpu-regs-test-") + tag + "-" + std::to_string(getpid());
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

telemetry::DeviceSample device(const char* gpu, int ordinal = 0) {
  telemetry::DeviceSample d{};
  telemetry::describe_device(load_gpu(gpu), ordinal, &d);
  return d;
}
}  // namespace

VTEST(the_database_loads_in_offset_order_with_no_overlaps) {
  const auto& regs = regs::config_registers();
  VCHECK(regs.size() > 40);
  std::set<std::string> names;
  for (size_t i = 0; i < regs.size(); ++i) {
    VCHECK(names.insert(regs[i].name).second);
    VCHECK(regs[i].status == "done" || regs[i].status == "model");
    if (i) VCHECK(regs[i].offset >= regs[i - 1].offset + regs[i - 1].width / 8);
  }
  VCHECK(regs::find_config("link_status") && regs::find_config("link_status")->offset == 0x8a);
  VCHECK(regs::find_config("0x08a") == regs::find_config("link_status"));
  VCHECK(regs::find_config("no_such_register") == nullptr);
}

VTEST(the_capability_chains_lead_from_the_pointer_to_the_end) {
  TempMachine m("chain");
  for (const char* gpu : {"nvidia/h100", "nvidia/rtx3060", "amd/mi300x"}) {
    regs::ConfigSpace cs(device(gpu));
    const auto img = cs.image(regs::kConfigSize);
    std::set<uint8_t> ids;
    int hops = 0;
    for (uint32_t at = img[0x34]; at; at = img[at + 1]) {
      VCHECK(at >= 0x40 && at < 0x100 && ++hops < 16);
      ids.insert(img[at]);
    }
    VCHECK(ids == (std::set<uint8_t>{0x01, 0x05, 0x10}));   // power management, MSI, PCI Express
    const uint32_t ext = img[0x100] | img[0x101] << 8 | img[0x102] << 16 | static_cast<uint32_t>(img[0x103]) << 24;
    VCHECK_EQ(ext & 0xFFFF, 0x0001u);   // AER, the only extended capability
    VCHECK_EQ(ext >> 20, 0u);
  }
}

VTEST(identity_class_and_link_come_from_the_device) {
  TempMachine m("identity");
  regs::ConfigSpace h100(device("nvidia/h100"));
  VCHECK_EQ(h100.read(0x00, 2), 0x10deu);
  VCHECK_EQ(h100.read(0x08, 4) >> 8, 0x030200u);        // 3D controller
  regs::ConfigSpace geforce(device("nvidia/rtx3060"));
  VCHECK_EQ(geforce.read(0x08, 4) >> 8, 0x030000u);     // VGA compatible controller
  regs::ConfigSpace mi300x(device("amd/mi300x"));
  VCHECK_EQ(mi300x.read(0x00, 2), 0x1002u);
  VCHECK_EQ(mi300x.read(0x08, 4) >> 8, 0x120000u);      // processing accelerator
  auto d = device("nvidia/t4");
  regs::ConfigSpace t4(d);
  VCHECK_EQ(t4.read(0x84, 4) & 0x3FF, (8u << 4) | 3u);  // capable of Gen3 x8
  d.pcie_width = 2;                                      // as a degraded link reads
  regs::ConfigSpace degraded(d);
  VCHECK_EQ(degraded.read(0x8a, 2) & 0x3FF, (2u << 4) | 3u);
  VCHECK_EQ(degraded.read(0x84, 4) & 0x3FF, (8u << 4) | 3u);   // the capability does not change
}

VTEST(read_write_bits_are_kept_and_read_only_ones_are_not) {
  TempMachine m("rw");
  const auto d = device("nvidia/h100");
  {
    regs::ConfigSpace cs(d);
    VCHECK_EQ(cs.read(0x04, 2), 0x0006u);
    cs.write(0x04, 2, 0xFFFF);
    VCHECK_EQ(cs.read(0x04, 2), 0x0547u);   // only the writable bits
    cs.write(0x04, 1, 0x02);                // a byte write leaves the other byte alone
    VCHECK_EQ(cs.read(0x04, 2), 0x0502u);
    cs.write(0x00, 2, 0x1234);
    VCHECK_EQ(cs.read(0x00, 2), 0x10deu);
  }
  regs::ConfigSpace another(d);             // another process, in effect
  VCHECK_EQ(another.read(0x04, 2), 0x0502u);
}

VTEST(base_address_registers_answer_a_sizing_probe_with_their_size) {
  TempMachine m("bars");
  const auto d = device("nvidia/h100");
  regs::ConfigSpace cs(d);
  const auto bar0 = regs::bar(cs, d, 0);
  VCHECK_EQ(bar0.size, uint64_t{16} << 20);
  VCHECK(!bar0.is64 && !bar0.prefetchable);
  cs.write(0x10, 4, 0xFFFFFFFF);
  VCHECK_EQ(cs.read(0x10, 4), 0xFF000000u);   // 16 MiB, 32-bit, non-prefetchable
  cs.write(0x10, 4, 0xE1000000);
  VCHECK_EQ(regs::bar(cs, d, 0).base, uint64_t{0xE1000000});
  const auto bar1 = regs::bar(cs, d, 1);
  VCHECK(bar1.is64 && bar1.prefetchable && bar1.size >= d.vram_total_bytes);
  VCHECK_EQ(regs::bar(cs, d, 2).size, 0u);    // the upper half of BAR1
  cs.write(0x14, 4, 0xFFFFFFFF);
  cs.write(0x18, 4, 0xFFFFFFFF);
  const uint64_t probe = uint64_t{cs.read(0x18, 4)} << 32 | (cs.read(0x14, 4) & ~0xFu);
  VCHECK_EQ(~probe + 1, bar1.size);
  VCHECK_EQ(cs.read(0x14, 4) & 0xF, 0xCu);    // 64-bit, prefetchable
}

VTEST(error_status_is_set_by_errors_and_cleared_by_writing_one) {
  TempMachine m("aer");
  const auto d = device("nvidia/h100");
  regs::ConfigSpace cs(d);
  VCHECK_EQ(cs.read(0x110, 4), 0u);
  ras::inject_pcie(d.uuid, ras::Pcie::BadTlp, 1);
  ras::inject_pcie(d.uuid, ras::Pcie::Replay, 3);
  VCHECK_EQ(cs.read(0x110, 4), (1u << 6) | (1u << 12));
  VCHECK_EQ(cs.read(0x82, 2) & 0x7, 0x1u);    // device status: a correctable error
  cs.write(0x110, 4, 1u << 6);                // clear BadTLP only
  VCHECK_EQ(cs.read(0x110, 4), 1u << 12);
  ras::inject_pcie(d.uuid, ras::Pcie::Lcrc, 1);
  VCHECK_EQ(cs.read(0x110, 4), (1u << 6) | (1u << 12));   // until it happens again
  ras::inject_pcie(d.uuid, ras::Pcie::Fatal, 1);
  VCHECK_EQ(cs.read(0x104, 4), 1u << 4);      // a data link protocol error
  VCHECK_EQ(cs.read(0x82, 2) & 0x7, 0x5u);
  cs.write(0x82, 2, 0x7);
  VCHECK_EQ(cs.read(0x82, 2) & 0x7, 0u);
  ras::reset_volatile(d.uuid);                 // a driver reload: counts start again
  ras::inject_pcie(d.uuid, ras::Pcie::Fatal, 1);
  VCHECK_EQ(cs.read(0x104, 4), 1u << 4);
}

VTEST(accesses_must_be_aligned_and_inside_the_space) {
  TempMachine m("align");
  regs::ConfigSpace cs(device("nvidia/t4"));
  for (auto [off, size] : {std::pair{0x01u, 2u}, {0x02u, 4u}, {0xFFEu, 4u}, {0x00u, 3u}, {0x1000u, 1u}}) {
    bool refused = false;
    try {
      cs.read(off, size);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    VCHECK(refused);
  }
}

VTEST(every_access_is_logged_with_its_process) {
  TempMachine m("log");
  const auto d = device("nvidia/t4");
  regs::ConfigSpace cs(d);
  cs.read(0x00, 4);
  cs.write(0x04, 2, 0x0007);
  cs.image(256);
  const auto log = regs::access_log(d.uuid);
  VCHECK_EQ(log.size(), 3u);
  VCHECK(!log[0].write && log[0].offset == 0 && log[0].size == 4 && log[0].value == 0x1eb810deu);
  VCHECK(log[1].write && log[1].offset == 4 && log[1].value == 7u);
  VCHECK_EQ(log[2].size, 256u);
  VCHECK_EQ(log[2].pid, static_cast<uint32_t>(getpid()));
  VCHECK(!log[0].process.empty());
  for (uint32_t i = 0; i < regs::kLogEntries + 10; ++i) cs.read(0x00, 2);
  const auto full = regs::access_log(d.uuid);
  VCHECK_EQ(full.size(), size_t{regs::kLogEntries});   // the newest kept
  VCHECK_EQ(full.back().seq, uint64_t{regs::kLogEntries} + 13);
}

VTEST_MAIN
