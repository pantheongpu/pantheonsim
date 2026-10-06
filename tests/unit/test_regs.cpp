// The register database and the configuration-space access engine: the
// database is well formed, the capability chains are walkable, and every
// access follows its register's semantics.
#include "vgpu/regs.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>

#include "vgpu/amd_metrics.hpp"
#include "vgpu/amd_regs.hpp"
#include "vgpu/embedded_gpu_registers.hpp"
#include "vgpu/error.hpp"
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
  // The generic layout's chains; a card that replays a captured space has the
  // captured card's (a_geforce_replays_the_measured_configuration_space).
  for (const char* gpu : {"nvidia/h100", "nvidia/t4", "amd/mi300x"}) {
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
    VCHECK_EQ(cs.read(0x04, 2), 0x0406u);   // as a bound driver leaves it
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

// Both vendors have MMIO registers, each in its own space, and every offset in
// either map says which vendor header it came from.
VTEST(each_vendor_has_its_own_mmio_registers) {
  TempMachine m("mmio");
  const auto mi = device("amd/mi300x");
  const auto h100 = device("nvidia/h100");
  VCHECK(regs::has_space(mi, regs::Space::AmdMmio));
  VCHECK(regs::has_space(h100, regs::Space::AmdMmio));   // "mmio" is the GPU's own
  VCHECK(regs::resolve_space(mi, regs::Space::AmdMmio) == regs::Space::AmdMmio);
  VCHECK(regs::resolve_space(h100, regs::Space::AmdMmio) == regs::Space::NvidiaMmio);
  // Either spelling asks for the device's *own* space, which is the point of
  // resolve_space: a tool says "mmio" and gets the map that card has.
  regs::RegisterSpace amd_bar5(regs::Space::NvidiaMmio, mi);
  VCHECK(amd_bar5.space() == regs::Space::AmdMmio);
  regs::RegisterSpace nv_bar0(regs::Space::AmdMmio, h100);
  VCHECK(nv_bar0.space() == regs::Space::NvidiaMmio);
  // And the two maps are different registers: BAR0's identity register is not
  // in AMD's map, whose first register is an engine status.
  VCHECK(regs::find(regs::Space::NvidiaMmio, "pmc_boot_0") != nullptr);
  VCHECK(regs::find(regs::Space::AmdMmio, "pmc_boot_0") == nullptr);
  for (regs::Space sp : {regs::Space::AmdMmio, regs::Space::NvidiaMmio}) {
    VCHECK(!regs::registers(sp).empty());
    for (const auto& r : regs::registers(sp)) {
      VCHECK_EQ(r.width, 32u);
      VCHECK_EQ(r.offset % 4, 0u);
      // Every offset is accounted for: a vendor header names it, or a card was
      // read at it. reg_000008 is the second kind -- the published headers name
      // nothing there and a real card answered zero.
      VCHECK(!r.source.empty() || !r.measured.empty());
    }
  }
}

// UMC ECC, MCA status and thermal registers: offsets are (Aldebaran's IP base
// + the header's register offset) * 4, the counts follow injected errors, and
// every one names its header symbol and says it is not measured.
VTEST(umc_ecc_registers_follow_injected_memory_errors) {
  TempMachine m("umc");
  const auto d = device("amd/mi300x");
  const auto off = [](const char* name) {
    const auto* r = regs::find(regs::Space::AmdMmio, name);
    if (!r) throw vtest::Failure(std::string("no register ") + name);
    return r->offset;
  };
  VCHECK_EQ(off("umc0_ch0_ecc_ctrl"), (0x14000u + 0x0053) * 4);
  VCHECK_EQ(off("umc0_ch0_ecc_err_cnt_sel"), (0x14000u + 0x0328) * 4);
  VCHECK_EQ(off("umc0_ch0_ecc_err_cnt"), (0x14000u + 0x0329) * 4);
  VCHECK_EQ(off("umc0_ch3_ecc_err_cnt"), (0x14000u + 0x0f29) * 4);   // regUMCCH3_0_EccErrCnt
  VCHECK_EQ(off("umc0_mca_status_lo"), (0x14000u + 0x03c2) * 4);     // regMCA_UMC_UMC0_MCUMC_STATUST0
  VCHECK_EQ(off("thm_tcon_cur_tmp"), (0x16600u + 0x0000) * 4);       // THM_BASE segment 0
  for (const char* n : {"umc0_ch0_ecc_ctrl", "umc0_ch2_ecc_err_cnt", "umc0_mca_status_hi", "thm_tcon_cur_tmp"}) {
    const auto* r = regs::find(regs::Space::AmdMmio, n);
    VCHECK(r->source.find("_offset.h reg") != std::string::npos);
    VCHECK(r->measured.find("not measured") != std::string::npos);   // the base is assumed
    VCHECK_EQ(r->status, std::string("model"));
  }
  {
    regs::RegisterSpace clean(regs::Space::AmdMmio, d);
    for (const char* n : {"umc0_ch0_ecc_err_cnt", "umc0_ch3_ecc_err_cnt", "umc0_mca_status_lo", "umc0_mca_status_hi"})
      VCHECK_EQ(clean.read(off(n), 4), 0u);   // nothing injected, nothing counted
    VCHECK_EQ(clean.read(off("umc0_ch0_ecc_ctrl"), 4), 0x401u);   // write and read ECC on
    VCHECK_EQ(clean.read(off("thm_tcon_cur_tmp"), 4) >> 21, d.temperature_c * 8u);
    clean.write(off("umc0_ch1_ecc_err_cnt_sel"), 4, 0xffffffffu);
    VCHECK_EQ(clean.read(off("umc0_ch1_ecc_err_cnt_sel"), 4), 0xb00fu);   // only its fields
  }
  ras::inject_ecc(d.uuid, ras::Severity::Corrected, ras::Location::DeviceMemory, 6, ras::Retirement::Rows);
  {
    regs::RegisterSpace r(regs::Space::AmdMmio, d);
    VCHECK_EQ(r.read(off("umc0_ch0_ecc_err_cnt"), 4), 2u);   // 6 shared over the four mapped channels
    VCHECK_EQ(r.read(off("umc0_ch1_ecc_err_cnt"), 4), 2u);
    VCHECK_EQ(r.read(off("umc0_ch2_ecc_err_cnt"), 4), 1u);
    VCHECK_EQ(r.read(off("umc0_ch3_ecc_err_cnt"), 4), 1u);
    const uint32_t hi = r.read(off("umc0_mca_status_hi"), 4);
    VCHECK_EQ(hi >> 31, 1u);           // Val
    VCHECK_EQ((hi >> 14) & 1, 1u);     // CECC
    VCHECK_EQ((hi >> 29) & 1, 0u);     // not UC
  }
  ras::inject_ecc(d.uuid, ras::Severity::Uncorrected, ras::Location::DeviceMemory, 1, ras::Retirement::Rows);
  {
    regs::RegisterSpace r(regs::Space::AmdMmio, d);
    const uint32_t hi = r.read(off("umc0_mca_status_hi"), 4);
    VCHECK_EQ((hi >> 29) & 1, 1u);     // UC
    VCHECK_EQ((hi >> 13) & 1, 1u);     // UECC
    VCHECK_EQ(r.read(off("umc0_ch0_ecc_err_cnt"), 4), 2u);   // the correctable count is unchanged
  }
}

VTEST(engine_status_follows_whether_the_gpu_is_busy) {
  TempMachine m("grbm");
  auto d = device("amd/mi300x");
  regs::RegisterSpace idle(regs::Space::AmdMmio, d);
  const uint32_t at = regs::find(regs::Space::AmdMmio, "grbm_status")->offset;
  VCHECK_EQ(at, 0x8010u);                             // (GC base 0x2000 + 0x0004) * 4
  VCHECK_EQ(idle.read(at, 4) >> 31, 0u);              // GUI_ACTIVE clear
  VCHECK_EQ((idle.read(at, 4) >> 12) & 3, 3u);        // DB and CB clean
  d.utilization_gpu = 80;
  regs::RegisterSpace busy(regs::Space::AmdMmio, d);
  VCHECK_EQ(busy.read(at, 4) >> 31, 1u);
  VCHECK_EQ((busy.read(at, 4) >> 29) & 1, 1u);        // CP busy
}

// The link controller's speed straps follow the profile's highest generation.
VTEST(the_link_controllers_speed_straps_follow_the_profile) {
  TempMachine m("lcstrap");
  for (const auto& [gpu, gen] : {std::pair{"amd/mi300x", 5u}, std::pair{"amd/mi250x", 4u}}) {
    const telemetry::DeviceSample d = device(gpu);
    regs::RegisterSpace cs(regs::Space::AmdMmio, d);
    const uint32_t v = cs.read(regs::find(regs::Space::AmdMmio, "nbio_ep_pcie_lc_speed_cntl")->offset, 4);
    VCHECK_EQ(v, (1u << (d.pcie_gen_max - 1)) - 1);
    VCHECK_EQ(d.pcie_gen_max, gen);
  }
}

VTEST(the_smu_mailbox_answers_as_the_firmware_does) {
  TempMachine m("smu");
  regs::RegisterSpace cs(regs::Space::AmdMmio, device("amd/mi300x"));
  const auto at = [](const char* name) { return regs::find(regs::Space::AmdMmio, name)->offset; };
  VCHECK_EQ(cs.read(at("smu_response"), 4), regs::kSmuResultOk);   // ready once loaded
  auto send = [&](uint32_t msg, uint32_t arg) {
    cs.write(at("smu_response"), 4, 0);
    cs.write(at("smu_argument"), 4, arg);
    cs.write(at("smu_message"), 4, msg);
    return std::pair{cs.read(at("smu_response"), 4), cs.read(at("smu_argument"), 4)};
  };
  VCHECK(send(regs::kSmuTestMessage, 41) == (std::pair{regs::kSmuResultOk, 42u}));
  VCHECK(send(regs::kSmuGetDriverIfVersion, 0) == (std::pair{regs::kSmuResultOk, 0x08042024u}));
  VCHECK_EQ(send(regs::kSmuGetSmuVersion, 0).second >> 16, 0x55u);   // 85.x.x
  VCHECK_EQ(send(0x77, 5).first, regs::kSmuResultUnknownCmd);
  VCHECK_EQ(send(0x3c, 0).first, regs::kSmuResultUnknownCmd);   // a gap in the header's numbering
  VCHECK_EQ(send(0x5c, 0).first, regs::kSmuResultUnknownCmd);   // PPSMC_Message_Count is not a message
  // Defined by the header (GetMetricsTable, ResetVCN), not modelled: fails, not unknown.
  VCHECK_EQ(send(0x9, 0).first, regs::kSmuResultFailed);
  VCHECK_EQ(send(0x5b, 0).first, regs::kSmuResultFailed);
  VCHECK(regs::smu_message_defined(0x59) && !regs::smu_message_defined(0x41) && !regs::smu_message_defined(0));
  VCHECK_EQ(send(regs::kSmuTestMessage, 1).first, regs::kSmuResultOk);   // and it recovers
  VCHECK_EQ(send(0x77, 5).first, regs::kSmuResultUnknownCmd);
  regs::RegisterSpace again(regs::Space::AmdMmio, device("amd/mi300x"));   // another process
  VCHECK_EQ(again.read(at("smu_response"), 4), regs::kSmuResultUnknownCmd);
  VCHECK(regs::access_log(device("amd/mi300x").uuid, regs::Space::AmdMmio).size() > 10);
  VCHECK(regs::access_log(device("amd/mi300x").uuid, regs::Space::Config).empty());   // a log per space
}

VTEST(an_mmio_access_is_a_whole_aligned_dword) {
  TempMachine m("mmioalign");
  regs::RegisterSpace cs(regs::Space::AmdMmio, device("amd/mi300x"));
  VCHECK_EQ(cs.read(0x8014, 4), 0u);   // undeclared: reads as zero
  for (auto [off, size] : {std::pair{0x8010u, 2u}, {0x8012u, 4u}, {0x80000u, 4u}}) {
    bool refused = false;
    try {
      cs.read(off, size);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    VCHECK(refused);
  }
}

// The GA10x GeForce the RTX 3060 profile models replays a real RTX 3080 Ti's
// configuration space (GA102, read as root by tools/regprobe;
// nvidia/registers/measurements/rtx3080ti): all 4096 bytes must match but for
// what is this card's or this host's own -- its IDs and its BAR addresses.
VTEST(a_geforce_replays_the_measured_configuration_space) {
  TempMachine m("measured");
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/registers/measurements/rtx3080ti/config.bin",
                   std::ios::binary);
  const std::string real((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  VCHECK_EQ(real.size(), size_t{regs::kConfigSize});
  regs::ConfigSpace cs(device("nvidia/rtx3060"));
  VCHECK_EQ(std::string(cs.layout()), std::string("nvidia-rtx3080ti"));
  const auto model = cs.image(regs::kConfigSize);
  const std::set<size_t> own = {0x02, 0x03, 0x2e, 0x2f};   // device and subsystem IDs
  for (size_t i = 0; i < regs::kConfigSize; ++i) {
    // BAR addresses, not their type bits: every byte of the 64-bit BARs' upper
    // halves (BAR2 and BAR4), the address bytes of the rest.
    if (own.count(i) || (i >= 0x18 && i < 0x1c) || (i >= 0x20 && i < 0x24) ||
        (i >= 0x10 && i < 0x28 && (i & 3) != 0))
      continue;
    if (model[i] != static_cast<uint8_t>(real[i])) {
      std::fprintf(stderr, "  byte 0x%03zx: model 0x%02x, measured 0x%02x\n", i, model[i],
                   static_cast<uint8_t>(real[i]));
      VCHECK(false);
    }
  }
}

// The card itself: every byte of its configuration space is the measured
// card's, but for the BAR addresses its host's firmware chose.
VTEST(the_rtx_3080_ti_profile_is_the_measured_card_to_the_byte) {
  TempMachine m("sameCard");
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/registers/measurements/rtx3080ti/config.bin",
                   std::ios::binary);
  const std::string real((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  regs::ConfigSpace cs(device("nvidia/rtx3080ti"));
  const auto model = cs.image(regs::kConfigSize);
  for (size_t i = 0; i < regs::kConfigSize; ++i) {
    const bool bar_address = i >= 0x10 && i < 0x28 && !((i & 3) == 0 && i != 0x18 && i != 0x20);
    if (!bar_address) VCHECK_EQ(static_cast<int>(model[i]), static_cast<int>(static_cast<uint8_t>(real[i])));
  }
}

// An NVIDIA GPU's BAR0. Every NVIDIA card has the space -- the offsets and
// fields are NVIDIA's published headers', which cover every architecture here --
// and on the one card that was read it answers exactly what that card did.
VTEST(an_nvidia_bar0_answers_as_the_measured_card) {
  TempMachine m("nvmmio");
  const auto ti = device("nvidia/rtx3080ti");
  VCHECK(regs::has_space(ti, regs::Space::AmdMmio));   // "mmio" is the GPU's own
  VCHECK(regs::resolve_space(ti, regs::Space::AmdMmio) == regs::Space::NvidiaMmio);
  VCHECK(regs::has_space(device("nvidia/rtx3060"), regs::Space::AmdMmio));
  VCHECK(regs::has_space(device("nvidia/h100"), regs::Space::AmdMmio));
  regs::RegisterSpace bar0(regs::Space::AmdMmio, ti);
  VCHECK(bar0.space() == regs::Space::NvidiaMmio);
  VCHECK_EQ(bar0.read(0x0, 4), 0xb72000a1u);   // measured, to the bit
  VCHECK_EQ(bar0.read(0x4, 4), 0u);            // measured: a real card, not a virtual function
  VCHECK_EQ(bar0.read(0x8, 4), 0u);            // measured
  VCHECK_EQ(bar0.read(0xc, 4), 0xbadf5040u);   // measured: nothing there
  VCHECK_EQ(bar0.read(0x100000, 4), 0xbadf5040u);
  VCHECK_EQ(bar0.read(0x0, 4) & 0xFF, 0xa1u);   // the revision configuration space reports
}

// PMC_BOOT_0 carries the architecture and die in the fields NVIDIA's header
// defines, so software that reads a card's family off BAR0 gets the right
// answer on every profile -- and the measured card's whole value back.
VTEST(every_nvidia_card_reports_its_architecture_in_bar0) {
  TempMachine m("nvarch");
  struct Want {
    const char* gpu;
    uint32_t architecture;   // NV_PMC_BOOT_0_ARCHITECTURE_*
  };
  for (const Want& w : {Want{"nvidia/t4", 0x16},            // TU100: Turing
                        Want{"nvidia/a100", 0x17},          // GA100: Ampere
                        Want{"nvidia/rtx3080ti", 0x17},
                        Want{"nvidia/h100", 0x18},          // GH100: Hopper
                        Want{"nvidia/gh200-480gb", 0x18},
                        Want{"nvidia/l4", 0x19},            // AD100: Ada
                        Want{"nvidia/l40s", 0x19},
                        Want{"nvidia/b200", 0x1a},          // GB100: Blackwell
                        Want{"nvidia/b300", 0x1a},          // GB110 is a GB100-architecture die
                        Want{"nvidia/rtx5090", 0x1b}}) {    // GB200: the GB20x dies
    regs::RegisterSpace bar0(regs::Space::AmdMmio, device(w.gpu));
    const uint32_t v = bar0.read(0x0, 4);
    if (((v >> 24) & 0x1F) != w.architecture)
      throw vtest::Failure(std::string(w.gpu) + ": PMC_BOOT_0 architecture is " +
                           std::to_string((v >> 24) & 0x1F) + ", expected " +
                           std::to_string(w.architecture));
    VCHECK_EQ(v & 0xFFu, 0xa1u);   // the revision, as configuration space reports it
    // The die is known only where a card was read: the RTX 3080 Ti is GA102,
    // which that card reported as implementation 2.
    VCHECK_EQ((v >> 20) & 0xFu, std::string(w.gpu) == "nvidia/rtx3080ti" ? 2u : 0u);
  }
}

// The registers a driver writes keep what was written, and the counter
// software polls for time moves forward. Both are what BAR0 is read for.
VTEST(nvidia_bar0_scratch_keeps_writes_and_the_timer_advances) {
  TempMachine m("nvscratch");
  regs::RegisterSpace bar0(regs::Space::AmdMmio, device("nvidia/t4"));
  bar0.write(0x1400, 4, 0xdeadbeef);            // PBUS_SW_SCRATCH(0)
  bar0.write(0x140c, 4, 0x0000f00d);            // PBUS_SW_SCRATCH(3)
  VCHECK_EQ(bar0.read(0x1400, 4), 0xdeadbeefu);
  VCHECK_EQ(bar0.read(0x140c, 4), 0x0000f00du);
  VCHECK_EQ(bar0.read(0x1404, 4), 0u);          // its neighbours are untouched
  bar0.write(0x1704, 4, 0x80000123);            // BAR1_BLOCK: pointer, target and mode
  VCHECK_EQ(bar0.read(0x1704, 4), 0x80000123u);
  // Read-only registers ignore a write, as the database says they do.
  bar0.write(0x0, 4, 0);
  VCHECK_EQ(bar0.read(0x0, 4) >> 24, 0xb6u);    // still a Turing card
  const uint32_t t0 = bar0.read(0x9800, 4);     // PTIMER_VF_TIMER(0), nanoseconds
  uint32_t t1 = t0;
  for (int i = 0; i < 1000 && t1 == t0; ++i) t1 = bar0.read(0x9800, 4);
  VCHECK(t1 != t0);
  // Which engines a bound driver leaves running: the host and graphics engines
  // at least, and the display engine only where there are display outputs.
  const uint32_t enable = bar0.read(0x200, 4);
  VCHECK((enable & (1u << 12)) && (enable & (1u << 8)));          // pgraph, pfifo
  VCHECK(!(enable & (1u << 30)));                                 // no display on a T4
  regs::RegisterSpace geforce(regs::Space::AmdMmio, device("nvidia/rtx3080ti"));
  VCHECK(geforce.read(0x200, 4) & (1u << 30));                    // pdisp on a GeForce
}

VTEST(capability_registers_are_found_through_each_cards_chain) {
  TempMachine m("chainwalk");
  const auto* aer = regs::find_config("aer_correctable_status");
  const auto* link = regs::find_config("link_status");
  regs::ConfigSpace generic(device("nvidia/h100"));
  VCHECK_EQ(std::string(generic.layout()), std::string("generic"));
  VCHECK_EQ(generic.offset_of(*aer), 0x110u);
  const auto d = device("nvidia/rtx3060");
  regs::ConfigSpace geforce(d);
  VCHECK_EQ(geforce.offset_of(*aer), 0x430u);   // the captured card's AER is at 0x420
  VCHECK_EQ(geforce.offset_of(*link), 0x8au);
  VCHECK(geforce.at(0x430) == aer);
  ras::inject_pcie(d.uuid, ras::Pcie::BadTlp, 1);
  VCHECK_EQ(geforce.read(0x430, 4), 1u << 6);   // live, where the chain put it
  geforce.write(0x430, 4, 1u << 6);
  VCHECK_EQ(geforce.read(0x430, 4), 0u);
  VCHECK_EQ(regs::link(geforce).max_gen, 4u);    // the profile's link, through the relocated registers
}

VTEST(an_amd_gpus_metrics_table_is_the_drivers_v1_5_layout) {
  auto d = device("amd/mi300x");
  d.temperature_c = 61;
  d.power_mw = 312500;
  d.utilization_gpu = 87;
  ras::Counters c{};
  c.pcie[static_cast<uint32_t>(ras::Pcie::Replay)] = 5;
  const std::string t = amd::gpu_metrics(d, c);
  VCHECK_EQ(t.size(), amd::kGpuMetricsSize);
  const auto u8 = [&](size_t o) { return static_cast<unsigned>(static_cast<uint8_t>(t[o])); };
  const auto u16 = [&](size_t o) { return u8(o) | u8(o + 1) << 8; };
  const auto u64 = [&](size_t o) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = v << 8 | u8(o + static_cast<size_t>(i));
    return v;
  };
  VCHECK_EQ(u16(0), 360u);                 // structure size
  VCHECK_EQ(u8(2), 1u);                    // format 1
  VCHECK_EQ(u8(3), 5u);                    // content 5
  VCHECK_EQ(u16(4), 61u);                  // hotspot temperature
  VCHECK_EQ(u16(10), 312u);                // socket power, W
  VCHECK_EQ(u16(12), 87u);                 // GFX activity
  VCHECK_EQ(u16(112), d.pcie_width);       // link width
  VCHECK_EQ(u16(114), 320u);               // Gen5, in 0.1 GT/s
  VCHECK_EQ(u64(152), 5u);                 // replays
  VCHECK_EQ(u64(88), ~uint64_t{0});        // energy: not reported
}

VTEST(amdgpus_driver_files_are_in_the_hwmon_abis_units) {
  TempMachine m("hwmon");
  auto d = device("amd/mi300x");
  d.temperature_c = 47;
  d.power_mw = 412000;
  d.utilization_gpu = 33;
  const std::string dir = m.root + "/files";
  std::filesystem::create_directories(dir);
  amd::write_driver_files(d, dir);
  const auto read = [&](const std::string& f) {
    std::ifstream in(dir + "/" + f);
    std::string s;
    std::getline(in, s);
    return s;
  };
  VCHECK_EQ(read("gpu_busy_percent"), std::string("33"));
  VCHECK_EQ(read("mem_info_vram_total"), std::to_string(d.vram_total_bytes));
  VCHECK_EQ(read("hwmon/name"), std::string("amdgpu"));
  VCHECK_EQ(read("hwmon/temp2_input"), std::string("47000"));      // millidegrees
  VCHECK_EQ(read("hwmon/temp2_label"), std::string("junction"));
  VCHECK_EQ(read("hwmon/power1_average"), std::string("412000000"));   // microwatts
  VCHECK_EQ(read("hwmon/freq1_input"), std::to_string(uint64_t{d.sm_clock_mhz} * 1000000));   // hertz
  for (const char* f : amd::kHwmonFiles) VCHECK(std::filesystem::exists(dir + "/hwmon/" + f));
  VCHECK(!std::filesystem::exists(dir + "/hwmon/temp1_input"));   // an MI300 has no edge sensor
}

// ---- BAR0 registers that only some architectures' headers define --------------------
//
// A register with an `arch` list is mapped on the GPUs whose published headers
// define it and nowhere else; everywhere else its offset is as unmapped as any
// undeclared one (0xbadf5040), because no header says what is there.

namespace {
constexpr uint32_t kUnmapped = 0xbadf5040u;
uint32_t bar0_read(const char* gpu, uint32_t offset) {
  regs::RegisterSpace bar0(regs::Space::AmdMmio, device(gpu));
  return bar0.read(offset, 4);
}
}  // namespace

VTEST(an_architecture_has_only_the_registers_its_headers_define) {
  TempMachine m("nvarchmap");
  struct Probe {
    const char* gpu;
    uint32_t offset;
    bool mapped;
  };
  for (const Probe& p : {
           // Turing: the second interrupt word, SET/CLEAR, two memory ECC counters, L2 ECC.
           Probe{"nvidia/t4", 0x144, true}, Probe{"nvidia/t4", 0x160, true}, Probe{"nvidia/t4", 0x184, true},
           Probe{"nvidia/t4", 0x900488, true}, Probe{"nvidia/t4", 0x90048c, true}, Probe{"nvidia/t4", 0x900490, false},
           Probe{"nvidia/t4", 0x1404f8, true}, Probe{"nvidia/t4", 0x100e78, true}, Probe{"nvidia/t4", 0x100e44, true},
           Probe{"nvidia/t4", 0x9025a0, false},   // Hopper's counters
           Probe{"nvidia/t4", 0x600, false},      // Ampere's device enable
           Probe{"nvidia/t4", 0x1410, false},     // Turing's header gives the scratch no size past 4 words
           // Ampere: device enable, the flush address, the full scratch array; none of Turing's.
           Probe{"nvidia/a100", 0x600, true}, Probe{"nvidia/rtx3080ti", 0x600, true},
           Probe{"nvidia/a100", 0x100c10, true}, Probe{"nvidia/a100", 0x100c40, true},
           Probe{"nvidia/a100", 0x14fc, true}, Probe{"nvidia/a100", 0x1500, false},
           Probe{"nvidia/a100", 0x160, false}, Probe{"nvidia/a100", 0x900488, false},
           // Ada: only the scratch array's headers.
           Probe{"nvidia/l4", 0x1410, true}, Probe{"nvidia/l4", 0x600, false}, Probe{"nvidia/l4", 0x100c10, false},
           // Hopper: its own ECC counters (four, at another offset), flush address, thermal scratch.
           Probe{"nvidia/h100", 0x9025a0, true}, Probe{"nvidia/h100", 0x9025ac, true},
           Probe{"nvidia/h100", 0x9025b0, false}, Probe{"nvidia/h100", 0x900488, false},
           Probe{"nvidia/h100", 0x100a34, true}, Probe{"nvidia/h100", 0x100a38, true},
           Probe{"nvidia/h100", 0x200bc, true}, Probe{"nvidia/h100", 0xad00bc, false},
           Probe{"nvidia/gh200-480gb", 0x9025a0, true},
           // Blackwell, GB100: the CC scratch and the thermal scratch; the GB20x has its own.
           Probe{"nvidia/b200", 0x580, true}, Probe{"nvidia/b300", 0x5bc, true}, Probe{"nvidia/b200", 0x5c0, false},
           Probe{"nvidia/b200", 0x200bc, true}, Probe{"nvidia/b200", 0xad00bc, false},
           Probe{"nvidia/b200", 0x1410, true},
           // The topology table's version: only GB100's header gives a value.
           Probe{"nvidia/b200", 0x224fc, true}, Probe{"nvidia/b300", 0x224fc, true},
           Probe{"nvidia/rtx5090", 0x224fc, false}, Probe{"nvidia/h100", 0x224fc, false},
           Probe{"nvidia/rtx5090", 0xad00bc, true}, Probe{"nvidia/rtx5090", 0x200bc, false},
           Probe{"nvidia/rtx5090", 0x580, false}, Probe{"nvidia/rtx5090", 0x1410, false}}) {
    const uint32_t v = bar0_read(p.gpu, p.offset);
    char what[96];
    std::snprintf(what, sizeof what, "%s at 0x%x reads 0x%08x, ", p.gpu, p.offset, v);
    if (p.mapped && v == kUnmapped) throw vtest::Failure(std::string(what) + "but its headers define a register there");
    if (!p.mapped && v != kUnmapped) throw vtest::Failure(std::string(what) + "but no header it has defines one there");
  }
  // The register database says the same, by name: an absent register is
  // absent, not at a different offset.
  const auto h100 = device("nvidia/h100");
  regs::RegisterSpace bar0(regs::Space::AmdMmio, h100);
  const regs::Register* turing_ded = regs::find(regs::Space::NvidiaMmio, "pfb_fbpa_0_ecc_ded_count_0");
  const regs::Register* hopper_ded = regs::find(regs::Space::NvidiaMmio, "pfb_fbpa_0_ecc_ded_count_0_gh100");
  VCHECK(turing_ded && hopper_ded);
  VCHECK_EQ(bar0.offset_of(*turing_ded), regs::RegisterSpace::kAbsent);
  VCHECK_EQ(bar0.offset_of(*hopper_ded), 0x9025a0u);
  VCHECK(bar0.at(0x900488) == nullptr);
  VCHECK(bar0.at(0x9025a0) == hopper_ded);
  // Registers every GPU has stay where they were.
  VCHECK(bar0.offset_of(*regs::find(regs::Space::NvidiaMmio, "pmc_boot_0")) == 0u);
}

// Every register gated to an architecture says where in NVIDIA's headers it is
// defined, names only architectures GPUs here have, and keeps the repository's
// conventions (a register with device logic behind it is marked as a model).
VTEST(architecture_gated_registers_name_their_header_symbol) {
  const std::set<std::string> known = {"turing", "ampere", "ada", "hopper", "blackwell", "gb100", "gb20x"};
  size_t gated = 0;
  std::set<std::string> names;
  for (const auto& r : regs::registers(regs::Space::NvidiaMmio)) {
    VCHECK(names.insert(r.name).second);
    if (r.arch.empty()) continue;
    ++gated;
    for (const auto& a : r.arch) VCHECK(known.count(a) == 1);
    // "turing/tu102/dev_fb.h NV_PFB_..." -- the header, then the symbol.
    VCHECK(r.source.find("/dev_") != std::string::npos);
    VCHECK(r.source.find(".h NV_") != std::string::npos);
    VCHECK(r.measured.empty());   // nothing past BAR0's head was read from a card
    if (!r.backing.empty()) VCHECK_EQ(r.status, std::string("model"));
  }
  VCHECK(gated > 100);
}

// Turing's header makes the interrupt enable read-only and changes it through
// SET and CLEAR, which read as zero; the later architectures' headers define no
// SET and CLEAR, so there the enable keeps what is written.
VTEST(turing_interrupt_enables_are_set_and_cleared_not_written) {
  TempMachine m("nvintren");
  regs::RegisterSpace t4(regs::Space::AmdMmio, device("nvidia/t4"));
  t4.write(0x140, 4, 0xffffffff);           // read-only on Turing
  VCHECK_EQ(t4.read(0x140, 4), 0u);
  t4.write(0x160, 4, 0x000000f0);           // NV_PMC_INTR_EN_SET(0)
  t4.write(0x164, 4, 0x80000001);           // NV_PMC_INTR_EN_SET(1)
  VCHECK_EQ(t4.read(0x140, 4), 0x000000f0u);
  VCHECK_EQ(t4.read(0x144, 4), 0x80000001u);
  t4.write(0x180, 4, 0x00000030);           // NV_PMC_INTR_EN_CLEAR(0)
  VCHECK_EQ(t4.read(0x140, 4), 0x000000c0u);
  t4.write(0x180, 4, 0xf);                  // clearing what is clear changes nothing
  VCHECK_EQ(t4.read(0x140, 4), 0x000000c0u);
  VCHECK_EQ(t4.read(0x160, 4), 0u);         // SET and CLEAR are write-only: they read zero
  VCHECK_EQ(t4.read(0x180, 4), 0u);
  VCHECK_EQ(t4.read(0x144, 4), 0x80000001u);   // the other word is untouched
  regs::RegisterSpace a100(regs::Space::AmdMmio, device("nvidia/a100"));
  a100.write(0x140, 4, 0x12345678);
  VCHECK_EQ(a100.read(0x140, 4), 0x12345678u);
}

// The ECC counters a memory diagnostic reads: they count what `vgpu fault`
// injected as uncorrectable errors, a write sets them (so zero clears), and a
// corrected error is not counted.
VTEST(nvidia_ecc_counters_report_injected_uncorrectable_errors) {
  TempMachine m("nvecc");
  const auto t4 = device("nvidia/t4");
  const auto h100 = device("nvidia/h100");
  regs::RegisterSpace turing(regs::Space::AmdMmio, t4), hopper(regs::Space::AmdMmio, h100);
  VCHECK_EQ(turing.read(0x900488, 4), 0u);   // a machine nobody has injected into
  VCHECK_EQ(hopper.read(0x9025a0, 4), 0u);
  ras::inject_ecc(t4.uuid, ras::Severity::Uncorrected, ras::Location::DeviceMemory, 3, ras::Retirement::Pages);
  ras::inject_ecc(t4.uuid, ras::Severity::Corrected, ras::Location::DeviceMemory, 9, ras::Retirement::Pages);
  ras::inject_ecc(t4.uuid, ras::Severity::Uncorrected, ras::Location::L2Cache, 5, ras::Retirement::Pages);
  VCHECK_EQ(turing.read(0x900488, 4), 3u);   // partition 0 carries the device's count
  VCHECK_EQ(turing.read(0x90048c, 4), 0u);
  VCHECK_EQ(turing.read(0x1404f8, 4), 5u);   // L2, not memory
  VCHECK_EQ(turing.read(0x100e78, 4), 0u);   // the MMU's SRAMs have no injection location
  VCHECK_EQ(hopper.read(0x9025a0, 4), 0u);   // another GPU's counters are its own
  // Writing zero clears; the counter then counts from what was written.
  turing.write(0x900488, 4, 0);
  VCHECK_EQ(turing.read(0x900488, 4), 0u);
  ras::inject_ecc(t4.uuid, ras::Severity::Uncorrected, ras::Location::DeviceMemory, 2, ras::Retirement::Pages);
  VCHECK_EQ(turing.read(0x900488, 4), 2u);
  turing.write(0x900488, 4, 100);
  VCHECK_EQ(turing.read(0x900488, 4), 100u);
  ras::inject_ecc(t4.uuid, ras::Severity::Uncorrected, ras::Location::DeviceMemory, 1, ras::Retirement::Pages);
  VCHECK_EQ(turing.read(0x900488, 4), 101u);
  VCHECK_EQ(turing.read(0x1404f8, 4), 5u);   // clearing one counter leaves the others
  // A counter written where nothing was counted (a partition past the first) keeps its value.
  turing.write(0x90048c, 4, 7);
  VCHECK_EQ(turing.read(0x90048c, 4), 7u);
  // Hopper's, at its own offsets.
  ras::inject_ecc(h100.uuid, ras::Severity::Uncorrected, ras::Location::DeviceMemory, 4, ras::Retirement::Rows);
  VCHECK_EQ(hopper.read(0x9025a0, 4), 4u);
  VCHECK_EQ(hopper.read(0x9025a4, 4), 0u);
  hopper.write(0x9025a0, 4, 0);
  VCHECK_EQ(hopper.read(0x9025a0, 4), 0u);
  // A driver reload restarts the counts; a counter's base from before does not hide them.
  ras::reset_volatile(t4.uuid);
  VCHECK_EQ(turing.read(0x900488, 4), 0u);
  ras::inject_ecc(t4.uuid, ras::Severity::Uncorrected, ras::Location::DeviceMemory, 1, ras::Retirement::Pages);
  VCHECK_EQ(turing.read(0x900488, 4), 1u);
}

// The scratch the architectures' headers give a block keeps what is written,
// and its neighbours are untouched.
VTEST(architecture_scratch_registers_keep_writes) {
  TempMachine m("nvarchscratch");
  regs::RegisterSpace h100(regs::Space::AmdMmio, device("nvidia/h100"));
  h100.write(0x14fc, 4, 0xa5a5a5a5);        // NV_PBUS_SW_SCRATCH(63)
  h100.write(0x200bc, 4, 0x12345678);       // NV_THERM_I2CS_SCRATCH
  VCHECK_EQ(h100.read(0x14fc, 4), 0xa5a5a5a5u);
  VCHECK_EQ(h100.read(0x200bc, 4), 0x12345678u);
  VCHECK_EQ(h100.read(0x14f8, 4), 0u);
  h100.write(0x100a34, 4, 0xfeedf000);      // the FBHUB flush address, low word
  VCHECK_EQ(h100.read(0x100a34, 4), 0xfeedf000u);
  regs::RegisterSpace b200(regs::Space::AmdMmio, device("nvidia/b200"));
  b200.write(0x590, 4, 0x3);                // NV_PMC_ZB_SCRATCH_RESET_2(4): confidential-computing bits
  VCHECK_EQ(b200.read(0x590, 4), 0x3u);
  VCHECK_EQ(b200.read(0x58c, 4), 0u);       // CC off until something sets it
  regs::RegisterSpace gb20x(regs::Space::AmdMmio, device("nvidia/rtx5090"));
  gb20x.write(0xad00bc, 4, 0xcafe);
  VCHECK_EQ(gb20x.read(0xad00bc, 4), 0xcafeu);
  regs::RegisterSpace a100(regs::Space::AmdMmio, device("nvidia/a100"));
  a100.write(0x100c10, 4, 0x1234);          // NV_PFB_NISO_FLUSH_SYSMEM_ADDR
  a100.write(0x100c40, 4, 0xffffffff);      // ..._HI keeps its 24 defined bits
  VCHECK_EQ(a100.read(0x100c10, 4), 0x1234u);
  VCHECK_EQ(a100.read(0x100c40, 4), 0x00ffffffu);
  regs::RegisterSpace t4(regs::Space::AmdMmio, device("nvidia/t4"));
  t4.write(0x100e2c, 4, 0xffffffff);        // MMU_FAULT_BUFFER_GET(0): only the pointer is software's
  VCHECK_EQ(t4.read(0x100e2c, 4), 0x000fffffu);
  VCHECK_EQ(t4.read(0x100e30, 4), 0u);      // PUT is read-only: no fault was ever raised
  VCHECK_EQ(t4.read(0x100cf8, 4), 3u);      // PAGE_FAULT_CTRL starts at SEND_NONE
}

VTEST_MAIN

// ---- Each GPU model's registers file (<vendor>/registers/gpus/) ---------------------

namespace {
std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
std::string source_file(const std::string& profile) {
  return std::string(VGPU_SOURCE_DIR) + "/" + regs::gpu_registers_file(profile);
}
// A register's value as software reads it: a 24-bit one as the dword around it.
uint32_t read_register(regs::RegisterSpace& rs, const regs::GpuRegister& r) {
  if (r.width == 24) return rs.read(r.offset & ~3u, 4) >> 8;
  return rs.read(r.offset, r.width / 8);
}
// Points VGPU_REGISTERS_DIR at a directory of its own for as long as it lives.
struct RegistersDir {
  std::string dir;
  explicit RegistersDir(const std::string& root) : dir(root + "/gpus") {
    std::filesystem::create_directories(dir);
    setenv("VGPU_REGISTERS_DIR", dir.c_str(), 1);
  }
  ~RegistersDir() { unsetenv("VGPU_REGISTERS_DIR"); }
  void write(const std::string& name, const std::string& text) const { std::ofstream(dir + "/" + name) << text; }
};
std::string replaced(std::string text, const std::string& from, const std::string& to) {
  const size_t at = text.find(from);
  if (at == std::string::npos) throw std::runtime_error("no '" + from + "' in the file");
  return text.replace(at, from.size(), to);
}
}  // namespace

// Every built-in profile has its file, the build embedded the one in the
// repository, and it is what the database and the profile give today: a
// change to either that is not regenerated into the file fails here.
VTEST(every_gpu_has_its_registers_file_and_it_is_current) {
  size_t embedded_files = 0;
  for (const auto& e : embedded::kGpuRegisters) embedded_files += e.profile != nullptr;
  VCHECK_EQ(embedded_files, available_gpus().size());
  for (const std::string& gpu : available_gpus()) {
    const auto d = device(gpu.c_str());
    const regs::GpuRegisters* g = regs::gpu_registers(d);
    VCHECK(g != nullptr);
    VCHECK_EQ(g->profile, gpu);
    VCHECK_EQ(g->device_id, d.pci_device_id >> 16);
    const std::string file = read_file(source_file(gpu));
    VCHECK(!file.empty());
    if (regs::export_registers(gpu, d) != file)
      throw vtest::Failure(regs::gpu_registers_file(gpu) + " is not what the register database and the profile " +
                           "give; regenerate it at the repository root: vgpu regs export --out .");
    bool embedded_same = false;
    for (const auto& e : embedded::kGpuRegisters)
      if (e.profile && gpu == e.profile) embedded_same = file == e.yaml;
    VCHECK(embedded_same);
  }
}

// A GPU reads, register by register, what its model's file says at power-on --
// the live registers included, since the file holds their power-on values.
VTEST(every_gpu_starts_from_its_registers_file) {
  for (const std::string& gpu : available_gpus()) {
    TempMachine m(("start-" + gpu.substr(gpu.find('/') + 1)).c_str());
    const auto d = device(gpu.c_str());
    const regs::GpuRegisters* g = regs::gpu_registers(d);
    VCHECK(g != nullptr);
    regs::ConfigSpace cs(d);
    VCHECK_EQ(std::string(cs.layout()), g->layout);
    VCHECK(!g->config.empty());
    for (const auto& r : g->config) {
      VCHECK_EQ(cs.offset_of(*regs::find_config(r.name)), r.offset);
      if (read_register(cs, r) != r.value)
        throw vtest::Failure(gpu + ": " + r.name + " does not read as its file's value");
    }
    VCHECK_EQ(!g->mmio.empty(), regs::has_space(d, regs::Space::AmdMmio));
    if (g->mmio.empty()) continue;
    regs::RegisterSpace mmio(regs::Space::AmdMmio, d);
    for (const auto& r : g->mmio) {
      // A counter that follows the clock cannot read back its power-on value a
      // moment later; the file records where it starts, and that it moves is
      // what nvidia_bar0_scratch_keeps_writes_and_the_timer_advances checks.
      if (r.live == "nvidia.ptimer_nsec") continue;
      if (read_register(mmio, r) != r.value)
        throw vtest::Failure(gpu + ": " + r.name + " does not read as its file's value");
    }
  }
}

// The files are unique per model: no two share a device, and models differ in
// what they hold -- identity, class, link, BARs.
VTEST(each_gpu_models_registers_are_its_own) {
  std::set<uint32_t> ids;
  std::set<std::string> configs;
  for (const std::string& gpu : available_gpus()) {
    const regs::GpuRegisters* g = regs::gpu_registers(device(gpu.c_str()));
    // A model with no file (one added without re-running CMake, which embeds
    // them) fails here by name rather than crashing the test.
    if (!g) throw vtest::Failure(gpu + " has no register file in the build");
    VCHECK(ids.insert(g->device_id).second);
    std::string values;
    for (const auto& r : g->config) values += r.name + "=" + std::to_string(r.value) + ";";
    VCHECK(configs.insert(values).second);
  }
  const auto* t4 = regs::gpu_registers(device("nvidia/t4"));
  const auto* ti = regs::gpu_registers(device("nvidia/rtx3080ti"));
  const auto* mi = regs::gpu_registers(device("amd/mi300x"));
  VCHECK(!t4->mmio.empty());          // every NVIDIA card has BAR0 registers
  VCHECK(!ti->mmio.empty());
  VCHECK(!mi->mmio.empty());          // an AMD GPU's registers behind BAR5
  // The same registers, different identities: the two NVIDIA cards report
  // different architectures in PMC_BOOT_0, and the AMD card has another map.
  const auto boot = [](const regs::GpuRegisters* g) {
    for (const auto& r : g->mmio)
      if (r.name == "pmc_boot_0") return r.value;
    return 0u;
  };
  VCHECK_EQ(boot(ti), 0xb72000a1u);
  VCHECK_EQ(boot(t4) >> 24, 0xb6u);
  VCHECK_EQ(boot(mi), 0u);            // no such register in AMD's map
  VCHECK_EQ(ti->layout, std::string("nvidia-rtx3080ti"));
  VCHECK_EQ(t4->layout, std::string("generic"));
}

// A value changed in the file is what the GPU reads, without a rebuild: the
// simulator takes its registers from the file, not from the database.
VTEST(a_value_changed_in_the_file_is_what_the_gpu_reads) {
  TempMachine m("edited");
  const std::string root = std::string("/tmp/vgpu-regs-test-edited-") + std::to_string(getpid());
  const std::string t4 = read_file(source_file("nvidia/t4"));
  {
    RegistersDir none(root + "-none");
    VCHECK(regs::gpu_registers(device("nvidia/t4")) == nullptr);   // no file: the model derives its registers
    regs::ConfigSpace cs(device("nvidia/t4"));
    VCHECK_EQ(cs.read(0x2e, 2), 0x1eb8u);   // subsystem_id, derived
  }
  RegistersDir dir(root + "-files");
  dir.write("t4.yaml", replaced(replaced(t4, "[0x02e, 16, ro, 0x1eb8]", "[0x02e, 16, ro, 0x12a2]"),
                                "[0x00e, 8, ro, 0x00]", "[0x00e, 8, ro, 0x80]"));
  regs::ConfigSpace cs(device("nvidia/t4"));
  VCHECK_EQ(cs.read(0x2e, 2), 0x12a2u);
  VCHECK_EQ(cs.read(0x0e, 1), 0x80u);
  VCHECK_EQ(cs.read(0x00, 2), 0x10deu);   // the rest as it was
  std::filesystem::remove_all(root + "-none");
  std::filesystem::remove_all(root + "-files");
}

// A file that disagrees with the database -- another width, a register the
// database does not have, one missing, another layout -- is refused, naming
// the register and how to regenerate it, rather than half applied.
VTEST(a_registers_file_that_disagrees_with_the_database_is_refused) {
  TempMachine m("stale");
  const std::string root = std::string("/tmp/vgpu-regs-test-stale-") + std::to_string(getpid()) + "-files";
  const std::string t4 = read_file(source_file("nvidia/t4"));
  const auto refused = [&](const std::string& text, const std::string& why) {
    std::filesystem::remove_all(root + "_" + why);
    RegistersDir dir(root + "_" + why);
    dir.write("t4.yaml", text);
    auto err = VCAPTURE(Error, regs::ConfigSpace(device("nvidia/t4")));
    VCHECK_CONTAINS(err.what(), "vgpu regs export nvidia/t4");
    std::filesystem::remove_all(root + "_" + why);
    return std::string(err.what());
  };
  VCHECK_CONTAINS(refused(replaced(t4, "[0x02e, 16, ro, 0x1eb8]", "[0x02e, 32, ro, 0x1eb8]"), "width"),
                  "subsystem_id differs");
  VCHECK_CONTAINS(refused(replaced(t4, "  subsystem_id:", "  subsystem_idd:"), "unknown"),
                  "subsystem_idd is not in the register database");
  VCHECK_CONTAINS(refused(replaced(t4, "  subsystem_id:                [0x02e, 16, ro, 0x1eb8]\n", ""), "missing"),
                  "1 of the device's config registers are missing");
  VCHECK_CONTAINS(refused(replaced(t4, "layout: generic", "layout: nvidia-rtx3080ti"), "layout"),
                  "layout nvidia-rtx3080ti");
  VCHECK_CONTAINS(refused(replaced(t4, "[0x08a, 16, ro, 0x1083, link.status]", "[0x08a, 16, ro, 0x1083]"), "live"),
                  "link_status differs");
}
