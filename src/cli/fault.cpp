// `vgpu fault`: inject reliability faults into the machine, show its
// reliability state, and reset it.
//
// This is how a health tool is tested against the errors it exists to catch.
// On real hardware an uncorrectable ECC error cannot be provoked on demand;
// here it is one command, and every surface a tool reads -- nvidia-smi, NVML,
// rocm-smi -- reports it, because they all read the same state (vgpu/ras.hpp).
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.hpp"
#include "machine.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/telemetry.hpp"

namespace {

int usage(FILE* to) {
  std::fprintf(to,
               "usage: vgpu fault inject [--gpu N] --ecc corrected|uncorrected [--location LOC] [--count N]\n"
               "       vgpu fault inject [--gpu N] --pcie COUNTER [--count N]\n"
               "       vgpu fault arm [--gpu N] --ecc corrected|uncorrected|--bitflip [--count N]\n"
               "                      [--on load|store|shared|alu|copy] [--rate P [--seed S]]\n"
               "       vgpu fault arm [--gpu N] --hang [--seconds S]\n"
               "       vgpu fault stuck [--gpu N] --offset BYTES --bit B --value 0|1 | --clear\n"
               "       vgpu fault lose [--gpu N] [--clear]\n"
               "       vgpu fault link [--gpu N] [--gen G] [--width W] | --clear\n"
               "       vgpu fault throttle [--gpu N] --reason R[,R...] [--seconds S] | --clear\n"
               "       vgpu fault show [--gpu N]\n"
               "       vgpu fault reset [--gpu N] --volatile|--aggregate\n"
               "\n"
               "Injects errors into the machine nvidia-smi and rocm-smi describe, so health tools\n"
               "can be tested against them. Without --gpu, every GPU is affected.\n"
               "\n"
               "  --ecc KIND        ECC errors, corrected or uncorrected. An uncorrected error in\n"
               "                    device memory also retires a page (GDDR) or remaps a row (HBM),\n"
               "                    pending until the next driver load.\n"
               "  --location LOC    device_memory (default; also dram), register_file, l1_cache,\n"
               "                    l2_cache, texture_memory, cbu, sram\n"
               "  --pcie COUNTER    replay, replay_rollover, l0_to_recovery, correctable,\n"
               "                    naks_received, bad_tlp, naks_sent, bad_dllp, non_fatal, fatal,\n"
               "                    lcrc, lane\n"
               "  --count N         how many (default 1)\n"
               "  --volatile        reset: zero the counts since the driver loaded, as a driver\n"
               "                    reload does, and complete pending retirements\n"
               "  --aggregate       reset: zero the lifetime ECC counts (nvidia-smi -p 1)\n"
               "\n"
               "arm loads faults that a running kernel's next accesses take: a corrected error\n"
               "is counted and changes nothing, an uncorrected one is counted, logged as Xid 48\n"
               "and fails the kernel with cudaErrorECCUncorrectable, and --bitflip silently\n"
               "flips one bit, as a fault ECC does not cover would -- what memory tests exist\n"
               "to catch. Inside `vgpu shell`, errors also appear in dmesg as the driver and\n"
               "kernel log them. --on says which accesses take them:\n"
               "  load              device-memory loads (default)\n"
               "  store             device-memory stores: the flipped value is what memory\n"
               "                    holds, so every later read finds it. Bit flips only, since\n"
               "                    ECC is checked when memory is read\n"
               "  shared            shared-memory loads, whose ECC errors count as the L1\n"
               "                    cache's and retire no page or row\n"
               "  alu               arithmetic results: floating-point, math and matrix\n"
               "                    (tensor core) instructions. Bit flips only, in the upper\n"
               "                    half of the result's bits; no ECC covers an ALU, so nothing\n"
               "                    counts or reports them\n"
               "  copy              copies out of device memory (to the host, or to another\n"
               "                    buffer or GPU): an uncorrected error fails the copy with\n"
               "                    cudaErrorECCUncorrectable, a bit flip corrupts what it\n"
               "                    delivers and leaves memory as it was\n"
               "\n"
               "--rate P arms a fault at a rate instead of a count: every access at the target\n"
               "takes one with probability P (for example 1e-6), from a generator seeded with S\n"
               "(default 1), until --rate 0 or a driver reload -- so errors grow with memory\n"
               "traffic, as a failing part's do.\n"
               "\n"
               "arm --hang stalls the next kernel launch, the device shown fully busy, for S\n"
               "seconds and then fails it with cudaErrorLaunchTimeout -- or, without --seconds,\n"
               "until the process is stopped, which is what a watchdog exists to catch.\n"
               "\n"
               "stuck makes one bit of device memory read as 0 or 1 whatever is written to it,\n"
               "as a failed cell does, until --clear: for every read, a kernel's loads and\n"
               "atomics and copies back to the host alike. BYTES (decimal or 0x hex) counts\n"
               "from the start of the device's memory, where its first allocation starts;\n"
               "allocations follow in order and are never moved. B is the bit, 0-7, within\n"
               "that byte. Up to 16 cells per GPU; a driver reload does not mend them.\n"
               "\n"
               "lose makes an NVIDIA GPU fall off the bus, logged as Xid 79: NVML answers\n"
               "NVML_ERROR_GPU_IS_LOST for it, nvidia-smi reports it lost and exits 15, and a\n"
               "program's next launch on it fails with an unspecified launch failure. A driver\n"
               "reload does not bring it back; --clear does, as a reset would.\n"
               "\n"
               "link trains the PCIe link below what card and slot support -- generation G,\n"
               "W lanes, or both -- as a bad riser or a marginal slot leaves it. The current\n"
               "link reads degraded beside an unchanged maximum, until --clear.\n"
               "\n"
               "throttle makes clock-event reasons active, for S seconds or until --clear:\n"
               "sw_power_cap, hw_slowdown, sw_thermal_slowdown, hw_thermal_slowdown and\n"
               "hw_power_brake_slowdown. Readings agree with them: a thermal slowdown puts the\n"
               "temperature at the slowdown threshold, a power cap holds power at the limit,\n"
               "and a slowdown pulls the SM clock down.\n");
  return to == stdout ? 0 : 2;
}

vgpu::ras::Retirement scheme_of(const vgpu::telemetry::DeviceSample& d) {
  return static_cast<vgpu::ras::Retirement>(d.memory_retirement);
}

void show(uint32_t index, const vgpu::telemetry::DeviceSample& d) {
  using vgpu::ras::Severity;
  const vgpu::ras::State st = vgpu::ras::read(d.uuid);
  std::printf("GPU %u: %s (%s)\n", index, d.name, d.uuid);
  if (!d.ecc_enabled) {
    std::printf("  ECC                    not on this card\n");
  } else {
    std::printf("  ECC errors             %12s %12s\n", "volatile", "aggregate");
    for (Severity s : {Severity::Corrected, Severity::Uncorrected}) {
      std::printf("    %-20s %12llu %12llu\n", s == Severity::Corrected ? "corrected" : "uncorrected",
                  static_cast<unsigned long long>(st.since_load.ecc_total(s)),
                  static_cast<unsigned long long>(st.lifetime.ecc_total(s)));
      for (uint32_t l = 0; l < vgpu::ras::kLocations; ++l) {
        const auto si = static_cast<uint32_t>(s);
        if (!st.since_load.ecc[si][l] && !st.lifetime.ecc[si][l]) continue;
        std::printf("      %-18s %12llu %12llu\n",
                    vgpu::ras::location_name(static_cast<vgpu::ras::Location>(l)),
                    static_cast<unsigned long long>(st.since_load.ecc[si][l]),
                    static_cast<unsigned long long>(st.lifetime.ecc[si][l]));
      }
    }
  }
  const vgpu::ras::Counters& life = st.lifetime;
  if (scheme_of(d) == vgpu::ras::Retirement::Pages)
    std::printf("  retired pages          single-bit %llu, double-bit %llu, pending %s\n",
                static_cast<unsigned long long>(life.retired_sbe),
                static_cast<unsigned long long>(life.retired_dbe), life.retired_pending ? "yes" : "no");
  if (scheme_of(d) == vgpu::ras::Retirement::Rows)
    std::printf("  remapped rows          correctable %llu, uncorrectable %llu, pending %s\n",
                static_cast<unsigned long long>(life.rows_correctable),
                static_cast<unsigned long long>(life.rows_uncorrectable),
                life.rows_pending ? "yes" : "no");
  std::string pcie;
  for (uint32_t c = 0; c < vgpu::ras::kPcieCounters; ++c)
    if (st.since_load.pcie[c])
      pcie += std::string(pcie.empty() ? "" : ", ") +
              vgpu::ras::pcie_name(static_cast<vgpu::ras::Pcie>(c)) + " " +
              std::to_string(st.since_load.pcie[c]);
  std::printf("  PCIe errors            %s\n", pcie.empty() ? "none" : pcie.c_str());
  const vgpu::ras::Counters& now = st.since_load;
  if (d.clock_event_reasons) {
    static const std::pair<uint64_t, const char*> kNames[] = {
        {vgpu::ras::kSwPowerCap, "sw_power_cap"},
        {vgpu::ras::kHwSlowdown, "hw_slowdown"},
        {vgpu::ras::kSwThermalSlowdown, "sw_thermal_slowdown"},
        {vgpu::ras::kHwThermalSlowdown, "hw_thermal_slowdown"},
        {vgpu::ras::kHwPowerBrakeSlowdown, "hw_power_brake_slowdown"}};
    std::string names;
    for (const auto& [bit, name] : kNames)
      if (d.clock_event_reasons & bit) names += std::string(names.empty() ? "" : ", ") + name;
    std::printf("  clock-event reasons    %s%s\n", names.c_str(),
                now.throttle_until_ns ? "" : " (until cleared)");
  }
  if (now.armed_hang)
    std::printf("  armed hang             next launch, %s\n",
                now.hang_seconds ? (std::to_string(now.hang_seconds) + " s").c_str()
                                 : "until the process is stopped");
  static constexpr const char* kOn[vgpu::ras::kTargets] = {"loads", "stores", "shared", "results",
                                                           "copies"};
  for (uint32_t t = 0; t < vgpu::ras::kTargets; ++t) {
    const auto at = static_cast<vgpu::ras::Target>(t);
    std::string rates;
    for (auto [kind, name] : {std::pair{vgpu::ras::Armed::Corrected, "corrected"},
                              std::pair{vgpu::ras::Armed::Uncorrected, "uncorrected"},
                              std::pair{vgpu::ras::Armed::Bitflip, "bit flips"}})
      if (const double r = vgpu::ras::rate_of(now, at, kind); r > 0) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%s%s %g", rates.empty() ? "" : ", ", name, r);
        rates += buf;
      }
    if (!rates.empty())
      std::printf("  rate on %-14s %s per access (seed %llu)\n", kOn[t], rates.c_str(),
                  static_cast<unsigned long long>(now.rate_seed));
  }
  for (uint32_t t = 0; t < vgpu::ras::kTargets; ++t)
    if (now.armed_pending[t] % vgpu::ras::kRatePending)
      std::printf("  armed on %-13s corrected %llu, uncorrected %llu, bit flips %llu\n", kOn[t],
                  static_cast<unsigned long long>(now.armed[t][0]),
                  static_cast<unsigned long long>(now.armed[t][1]),
                  static_cast<unsigned long long>(now.armed[t][2]));
  if (now.lost) std::printf("  bus                    fallen off (Xid 79) until cleared\n");
  if (d.pcie_gen != d.pcie_gen_max || d.pcie_width != d.pcie_width_max)
    std::printf("  PCIe link              Gen%u x%u, of Gen%u x%u, until cleared\n", d.pcie_gen, d.pcie_width,
                d.pcie_gen_max, d.pcie_width_max);
  for (const auto& cell : vgpu::ras::stuck_cells(d.uuid))
    std::printf("  stuck cell             bit %u of byte 0x%llx reads as %u\n", cell.bit,
                static_cast<unsigned long long>(cell.offset), cell.value);
  if (now.bitflips_delivered)
    std::printf("  bit flips delivered    %llu\n", static_cast<unsigned long long>(now.bitflips_delivered));
}

}  // namespace

int cmd_fault(const std::vector<std::string>& args) {
  if (args.empty()) return usage(stderr);
  const std::string& verb = args[0];
  if (verb == "-h" || verb == "--help" || verb == "help") return usage(stdout);
  if (verb != "inject" && verb != "arm" && verb != "stuck" && verb != "lose" && verb != "link" &&
      verb != "throttle" && verb != "show" && verb != "reset") {
    std::fprintf(stderr,
                 "vgpu fault: unknown action '%s' (inject, arm, stuck, lose, link, throttle, show or "
                 "reset)\n",
                 verb.c_str());
    return 2;
  }

  std::string gpu, ecc, location, pcie, reasons, on, offset_text;
  double rate = -1;
  long long seed = 1;
  long long bit = -1, value = -1, gen = 0, width = 0;
  bool vol = false, agg = false, bitflip = false, hang = false, clear = false;
  long long count = 1, seconds = -1;
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    const bool takes_value = a == "--gpu" || a == "--ecc" || a == "--location" || a == "--pcie" ||
                             a == "--count" || a == "--reason" || a == "--seconds" || a == "--on" ||
                             a == "--offset" || a == "--bit" || a == "--value" ||
                             a == "--gen" || a == "--width" ||
                             a == "--rate" || a == "--seed";
    if (takes_value && i + 1 >= args.size()) {
      std::fprintf(stderr, "vgpu fault: %s needs a value\n", a.c_str());
      return 2;
    }
    if (a == "--gpu") gpu = args[++i];
    else if (a == "--ecc") ecc = args[++i];
    else if (a == "--location") location = args[++i];
    else if (a == "--on") on = args[++i];
    else if (a == "--offset") offset_text = args[++i];
    else if (a == "--rate") {
      if (!vgpu::cli::parse_double(args[++i], 0, 1, &rate)) {
        std::fprintf(stderr, "vgpu fault: --rate is a probability per access, 0 to 1, got '%s'\n",
                     args[i].c_str());
        return 2;
      }
    } else if (a == "--seed") {
      if (!vgpu::cli::parse_int(args[++i], 0, 9223372036854775807LL, &seed)) {
        std::fprintf(stderr, "vgpu fault: --seed is a whole number, got '%s'\n", args[i].c_str());
        return 2;
      }
    }
    else if (a == "--gen" || a == "--width") {
      long long v = 0;
      if (!vgpu::cli::parse_int(args[++i], 1, a == "--gen" ? 6 : 32, &v)) {
        std::fprintf(stderr, "vgpu fault: %s needs a %s, got '%s'\n", a.c_str(),
                     a == "--gen" ? "PCIe generation, 1-6" : "lane count", args[i].c_str());
        return 2;
      }
      (a == "--gen" ? gen : width) = v;
    }
    else if (a == "--bit") {
      if (!vgpu::cli::parse_int(args[++i], 0, 7, &bit)) {
        std::fprintf(stderr, "vgpu fault: --bit is 0-7, the bit within the byte, got '%s'\n",
                     args[i].c_str());
        return 2;
      }
    } else if (a == "--value") {
      if (!vgpu::cli::parse_int(args[++i], 0, 1, &value)) {
        std::fprintf(stderr, "vgpu fault: --value is 0 or 1, got '%s'\n", args[i].c_str());
        return 2;
      }
    }
    else if (a == "--pcie") pcie = args[++i];
    else if (a == "--count") {
      if (!vgpu::cli::parse_int(args[++i], 1, 1000000000000LL, &count)) {
        std::fprintf(stderr, "vgpu fault: --count needs a whole number from 1, got '%s'\n",
                     args[i].c_str());
        return 2;
      }
    } else if (a == "--seconds") {
      if (!vgpu::cli::parse_int(args[++i], 1, 86400LL * 365, &seconds)) {
        std::fprintf(stderr, "vgpu fault: --seconds needs a whole number from 1, got '%s'\n",
                     args[i].c_str());
        return 2;
      }
    } else if (a == "--reason") reasons = args[++i];
    else if (a == "--hang") hang = true;
    else if (a == "--clear") clear = true;
    else if (a == "--bitflip") bitflip = true;
    else if (a == "--volatile") vol = true;
    else if (a == "--aggregate") agg = true;
    else if (a == "-h" || a == "--help") return usage(stdout);
    else {
      std::fprintf(stderr, "vgpu fault: unknown argument '%s'\n", a.c_str());
      return 2;
    }
  }

  vgpu::telemetry::Shared snap{};
  if (!vgpu::cli::read_machine(&snap)) return 1;
  std::vector<uint32_t> sel;
  if (gpu.empty()) {
    for (uint32_t i = 0; i < snap.device_count; ++i) sel.push_back(i);
  } else {
    long long n = 0;
    if (!vgpu::cli::parse_int(gpu, 0, 1 << 20, &n) || n >= static_cast<long long>(snap.device_count)) {
      std::fprintf(stderr, "vgpu fault: there is no GPU %s; this machine has %u\n", gpu.c_str(),
                   snap.device_count);
      return 2;
    }
    sel.push_back(static_cast<uint32_t>(n));
  }

  // Options that belong to one action are refused by the others.
  if (verb != "throttle" && !reasons.empty()) {
    std::fprintf(stderr, "vgpu fault %s: --reason belongs to throttle\n", verb.c_str());
    return 2;
  }
  if (clear && verb != "throttle" && verb != "stuck" && verb != "lose" && verb != "link") {
    std::fprintf(stderr, "vgpu fault %s: --clear belongs to throttle, stuck, lose and link\n",
                 verb.c_str());
    return 2;
  }
  if (verb != "stuck" && (!offset_text.empty() || bit >= 0 || value >= 0)) {
    std::fprintf(stderr, "vgpu fault %s: --offset, --bit and --value belong to stuck\n", verb.c_str());
    return 2;
  }
  if (seconds >= 0 && verb != "throttle" && !(verb == "arm" && hang)) {
    std::fprintf(stderr, "vgpu fault %s: --seconds belongs to throttle and arm --hang\n", verb.c_str());
    return 2;
  }
  if (!on.empty() && (verb != "arm" || hang)) {
    std::fprintf(stderr, "vgpu fault %s: --on belongs to arm --ecc and arm --bitflip\n", verb.c_str());
    return 2;
  }
  if (hang && verb != "arm") {
    std::fprintf(stderr, "vgpu fault %s: --hang belongs to arm\n", verb.c_str());
    return 2;
  }

  if (verb == "throttle") {
    if (!ecc.empty() || !pcie.empty() || !location.empty() || vol || agg || bitflip) {
      std::fprintf(stderr, "vgpu fault throttle: takes --reason, --seconds, --clear and --gpu\n");
      return 2;
    }
    if (reasons.empty() == !clear) {
      std::fprintf(stderr, "vgpu fault throttle: name one of --reason or --clear\n");
      return 2;
    }
    if (clear) {
      for (uint32_t i : sel) {
        vgpu::ras::clear_throttle(snap.devices[i].uuid);
        std::printf("Cleared clock-event reasons on GPU %u (%s).\n", i, snap.devices[i].name);
      }
      return 0;
    }
    uint64_t mask = 0;
    for (size_t at = 0; at <= reasons.size();) {
      const size_t comma = reasons.find(',', at);
      const std::string r = reasons.substr(at, (comma == std::string::npos ? reasons.size() : comma) - at);
      const uint64_t bit = vgpu::ras::reason_bit(r);
      if (!bit) {
        std::fprintf(stderr, "vgpu fault throttle: unknown reason '%s' (sw_power_cap, hw_slowdown, "
                             "sw_thermal_slowdown, hw_thermal_slowdown, hw_power_brake_slowdown)\n",
                     r.c_str());
        return 2;
      }
      mask |= bit;
      at = comma == std::string::npos ? reasons.size() + 1 : comma + 1;
    }
    for (uint32_t i : sel) {
      vgpu::ras::throttle(snap.devices[i].uuid, mask, seconds > 0 ? static_cast<uint64_t>(seconds) : 0);
      if (seconds > 0)
        std::printf("Throttling GPU %u (%s): %s for %lld s.\n", i, snap.devices[i].name, reasons.c_str(),
                    seconds);
      else
        std::printf("Throttling GPU %u (%s): %s until cleared.\n", i, snap.devices[i].name,
                    reasons.c_str());
    }
    return 0;
  }

  if (verb == "stuck") {
    if (!ecc.empty() || !pcie.empty() || !location.empty() || vol || agg || bitflip || count != 1) {
      std::fprintf(stderr, "vgpu fault stuck: takes --offset, --bit, --value, --clear and --gpu\n");
      return 2;
    }
    if (clear) {
      if (!offset_text.empty() || bit >= 0 || value >= 0) {
        std::fprintf(stderr, "vgpu fault stuck: --clear takes no cell\n");
        return 2;
      }
      for (uint32_t i : sel) {
        vgpu::ras::unstick_all(snap.devices[i].uuid);
        std::printf("Cleared stuck cells on GPU %u (%s).\n", i, snap.devices[i].name);
      }
      return 0;
    }
    if (offset_text.empty() || bit < 0 || value < 0) {
      std::fprintf(stderr, "vgpu fault stuck: name the cell with --offset, --bit and --value, or --clear\n");
      return 2;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long offset = std::strtoull(offset_text.c_str(), &end, 0);
    if (offset_text[0] == '-' || errno || !end || *end) {
      std::fprintf(stderr, "vgpu fault stuck: --offset is a byte offset, decimal or 0x hex, got '%s'\n",
                   offset_text.c_str());
      return 2;
    }
    for (uint32_t i : sel) {
      const auto& d = snap.devices[i];
      if (offset >= d.vram_total_bytes) {
        std::fprintf(stderr, "vgpu fault stuck: offset 0x%llx is past GPU %u's %llu bytes of memory\n",
                     offset, i, static_cast<unsigned long long>(d.vram_total_bytes));
        return 2;
      }
    }
    for (uint32_t i : sel) {
      const auto& d = snap.devices[i];
      try {
        vgpu::ras::stick(d.uuid, offset, static_cast<uint32_t>(bit), static_cast<uint32_t>(value));
      } catch (const std::length_error& e) {
        std::fprintf(stderr, "vgpu fault stuck: %s; clear them first\n", e.what());
        return 1;
      }
      std::printf("Bit %lld of byte 0x%llx on GPU %u (%s) now reads as %lld.\n", bit, offset, i, d.name,
                  value);
    }
    return 0;
  }

  if ((rate >= 0 || seed != 1) && (verb != "arm" || hang)) {
    std::fprintf(stderr, "vgpu fault %s: --rate and --seed belong to arm --ecc and arm --bitflip\n",
                 verb.c_str());
    return 2;
  }
  if (rate >= 0 && count != 1) {
    std::fprintf(stderr, "vgpu fault arm: a fault is armed by --count or at a --rate, not both\n");
    return 2;
  }
  if (seed != 1 && rate < 0) {
    std::fprintf(stderr, "vgpu fault arm: --seed seeds a --rate\n");
    return 2;
  }
  if (verb != "link" && (gen || width)) {
    std::fprintf(stderr, "vgpu fault %s: --gen and --width belong to link\n", verb.c_str());
    return 2;
  }
  if (verb == "link") {
    if (!ecc.empty() || !pcie.empty() || !location.empty() || vol || agg || bitflip || count != 1) {
      std::fprintf(stderr, "vgpu fault link: takes --gen, --width, --clear and --gpu\n");
      return 2;
    }
    if (clear == (gen || width)) {
      std::fprintf(stderr, "vgpu fault link: name --gen, --width or both, or --clear\n");
      return 2;
    }
    if (width && (width & (width - 1))) {
      std::fprintf(stderr, "vgpu fault link: a link is 1, 2, 4, 8, 16 or 32 lanes, not %lld\n", width);
      return 2;
    }
    for (uint32_t i : sel) {
      const auto& d = snap.devices[i];
      if (!d.pcie_gen_max) {
        std::fprintf(stderr, "vgpu fault link: GPU %u (%s) has no PCIe link in its profile\n", i, d.name);
        return 2;
      }
      if ((gen && gen >= d.pcie_gen_max) || (width && width >= d.pcie_width_max)) {
        std::fprintf(stderr, "vgpu fault link: GPU %u (%s) trains at Gen%u x%u; a degraded link is "
                             "below that\n",
                     i, d.name, d.pcie_gen_max, d.pcie_width_max);
        return 2;
      }
    }
    for (uint32_t i : sel) {
      const auto& d = snap.devices[i];
      if (clear) {
        vgpu::ras::restore_link(d.uuid);
        std::printf("GPU %u (%s) is back at Gen%u x%u.\n", i, d.name, d.pcie_gen_max, d.pcie_width_max);
      } else {
        vgpu::ras::degrade_link(d.uuid, static_cast<uint32_t>(gen), static_cast<uint32_t>(width));
        std::printf("GPU %u (%s) now trains at Gen%lld x%lld, of Gen%u x%u.\n", i, d.name,
                    gen ? gen : static_cast<long long>(d.pcie_gen_max),
                    width ? width : static_cast<long long>(d.pcie_width_max), d.pcie_gen_max,
                    d.pcie_width_max);
      }
    }
    return 0;
  }

  if (verb == "lose") {
    if (!ecc.empty() || !pcie.empty() || !location.empty() || vol || agg || bitflip || count != 1) {
      std::fprintf(stderr, "vgpu fault lose: takes --clear and --gpu\n");
      return 2;
    }
    for (uint32_t i : sel)
      if (std::strcmp(snap.devices[i].vendor, "amd") == 0) {
        std::fprintf(stderr, "vgpu fault lose: GPU %u (%s) is AMD; a lost GPU is modelled for NVIDIA "
                             "only\n",
                     i, snap.devices[i].name);
        return 2;
      }
    for (uint32_t i : sel) {
      const auto& d = snap.devices[i];
      if (clear) {
        vgpu::ras::recover(d.uuid);
        std::printf("GPU %u (%s) is back on the bus.\n", i, d.name);
      } else {
        vgpu::ras::lose(d.uuid, d.bus_id);
        std::printf("GPU %u (%s) has fallen off the bus.\n", i, d.name);
      }
    }
    return 0;
  }

  if (verb == "show") {
    if (!ecc.empty() || !pcie.empty() || !location.empty() || vol || agg || bitflip) {
      std::fprintf(stderr, "vgpu fault show: takes only --gpu\n");
      return 2;
    }
    for (uint32_t i : sel) show(i, snap.devices[i]);
    return 0;
  }

  if (verb == "reset") {
    if (vol == agg) {
      std::fprintf(stderr, "vgpu fault reset: name one of --volatile or --aggregate\n");
      return 2;
    }
    for (uint32_t i : sel) {
      const auto& d = snap.devices[i];
      if (vol) vgpu::ras::reset_volatile(d.uuid);
      else vgpu::ras::reset_aggregate(d.uuid);
      std::printf("Reset %s reliability counts on GPU %u (%s).\n", vol ? "volatile" : "aggregate", i,
                  d.name);
    }
    return 0;
  }

  if (verb == "arm") {
    if (!pcie.empty() || !location.empty() || vol || agg) {
      std::fprintf(stderr, "vgpu fault arm: takes --ecc, --bitflip or --hang, with --gpu and --count\n");
      return 2;
    }
    if (hang) {
      if (!ecc.empty() || bitflip) {
        std::fprintf(stderr, "vgpu fault arm: --hang is armed on its own\n");
        return 2;
      }
      for (uint32_t i : sel) {
        for (long long k = 0; k < count; ++k)
          vgpu::ras::arm_hang(snap.devices[i].uuid, seconds > 0 ? static_cast<uint64_t>(seconds) : 0);
        if (seconds > 0)
          std::printf("Armed a %lld s hang for the next kernel launch on GPU %u (%s).\n", seconds, i,
                      snap.devices[i].name);
        else
          std::printf("Armed a hang for the next kernel launch on GPU %u (%s); it lasts until the "
                      "process is stopped.\n",
                      i, snap.devices[i].name);
      }
      return 0;
    }
    if (ecc.empty() == !bitflip) {
      std::fprintf(stderr, "vgpu fault arm: name one of --ecc or --bitflip\n");
      return 2;
    }
    vgpu::ras::Armed kind = vgpu::ras::Armed::Bitflip;
    if (!ecc.empty()) {
      if (ecc == "corrected") kind = vgpu::ras::Armed::Corrected;
      else if (ecc == "uncorrected") kind = vgpu::ras::Armed::Uncorrected;
      else {
        std::fprintf(stderr, "vgpu fault arm: --ecc is corrected or uncorrected, got '%s'\n", ecc.c_str());
        return 2;
      }
      for (uint32_t i : sel)
        if (!snap.devices[i].ecc_enabled) {
          std::fprintf(stderr, "vgpu fault arm: GPU %u (%s) has no ECC, so it cannot report an ECC "
                               "error; --bitflip corrupts data on any card\n",
                       i, snap.devices[i].name);
          return 2;
        }
    }
    vgpu::ras::Target at = vgpu::ras::Target::Load;
    if (!on.empty() && !vgpu::ras::parse_target(on, &at)) {
      std::fprintf(stderr, "vgpu fault arm: --on is load, store, shared, alu or copy, got '%s'\n",
                   on.c_str());
      return 2;
    }
    if (at == vgpu::ras::Target::Store && kind != vgpu::ras::Armed::Bitflip) {
      std::fprintf(stderr, "vgpu fault arm: an ECC error is found when memory is read, not written; "
                           "arm it on loads, or a --bitflip on stores\n");
      return 2;
    }
    if (at == vgpu::ras::Target::Alu && kind != vgpu::ras::Armed::Bitflip) {
      std::fprintf(stderr, "vgpu fault arm: no ECC covers an arithmetic result; arm a --bitflip on alu\n");
      return 2;
    }
    const char* what = kind == vgpu::ras::Armed::Bitflip ? "bit flip" : ecc == "corrected"
                                                                           ? "corrected ECC error"
                                                                           : "uncorrected ECC error";
    const char* accesses = at == vgpu::ras::Target::Store    ? "device-memory stores"
                           : at == vgpu::ras::Target::Shared ? "shared-memory loads"
                           : at == vgpu::ras::Target::Alu    ? "arithmetic results"
                           : at == vgpu::ras::Target::Copy   ? "copies out of device memory"
                                                             : "device-memory loads";
    if (rate >= 0) {
      for (uint32_t i : sel) {
        vgpu::ras::arm_rate(snap.devices[i].uuid, kind, rate, at, static_cast<uint64_t>(seed));
        if (rate > 0)
          std::printf("Armed %ss at %g per access on the %s of GPU %u (%s), seed %lld.\n", what, rate,
                      accesses, i, snap.devices[i].name, seed);
        else
          std::printf("Stopped %ss at a rate on the %s of GPU %u (%s).\n", what, accesses, i,
                      snap.devices[i].name);
      }
      return 0;
    }
    for (uint32_t i : sel) {
      vgpu::ras::arm(snap.devices[i].uuid, kind, static_cast<uint64_t>(count), at);
      std::printf("Armed %lld %s%s for the next %s on GPU %u (%s).\n", count, what,
                  count == 1 ? "" : "s", accesses, i, snap.devices[i].name);
    }
    return 0;
  }

  // inject
  if (bitflip) {
    std::fprintf(stderr, "vgpu fault inject: --bitflip belongs to arm, since a flip happens on a load\n");
    return 2;
  }
  if (vol || agg) {
    std::fprintf(stderr, "vgpu fault inject: --volatile and --aggregate belong to reset\n");
    return 2;
  }
  if (ecc.empty() == pcie.empty()) {
    std::fprintf(stderr, "vgpu fault inject: name one of --ecc or --pcie\n");
    return 2;
  }
  if (!pcie.empty()) {
    if (!location.empty()) {
      std::fprintf(stderr, "vgpu fault inject: --location applies to --ecc\n");
      return 2;
    }
    vgpu::ras::Pcie counter;
    if (!vgpu::ras::parse_pcie(pcie, &counter)) {
      std::fprintf(stderr, "vgpu fault inject: unknown PCIe counter '%s' (vgpu fault --help lists them)\n",
                   pcie.c_str());
      return 2;
    }
    for (uint32_t i : sel) {
      vgpu::ras::inject_pcie(snap.devices[i].uuid, counter, static_cast<uint64_t>(count));
      if (const std::string line = vgpu::ras::aer_line(snap.devices[i].bus_id, counter); !line.empty())
        for (long long k = 0; k < count && k < 100; ++k) vgpu::ras::log_kernel(line);
      std::printf("Injected %lld PCIe %s error%s into GPU %u (%s).\n", count, pcie.c_str(),
                  count == 1 ? "" : "s", i, snap.devices[i].name);
    }
    return 0;
  }
  vgpu::ras::Severity severity;
  if (ecc == "corrected") severity = vgpu::ras::Severity::Corrected;
  else if (ecc == "uncorrected") severity = vgpu::ras::Severity::Uncorrected;
  else {
    std::fprintf(stderr, "vgpu fault inject: --ecc is corrected or uncorrected, got '%s'\n", ecc.c_str());
    return 2;
  }
  vgpu::ras::Location loc = vgpu::ras::Location::DeviceMemory;
  if (!location.empty() && !vgpu::ras::parse_location(location, &loc)) {
    std::fprintf(stderr, "vgpu fault inject: unknown location '%s' (vgpu fault --help lists them)\n",
                 location.c_str());
    return 2;
  }
  // A card without ECC cannot report an ECC error: it has nowhere to count one.
  for (uint32_t i : sel)
    if (!snap.devices[i].ecc_enabled) {
      std::fprintf(stderr, "vgpu fault inject: GPU %u (%s) has no ECC, so it cannot report an ECC error\n",
                   i, snap.devices[i].name);
      return 2;
    }
  for (uint32_t i : sel) {
    const auto& d = snap.devices[i];
    vgpu::ras::inject_ecc(d.uuid, severity, loc, static_cast<uint64_t>(count), scheme_of(d));
    // The driver's log of it. amdgpu reports every count it collects, by
    // block; NVIDIA's driver logs Xid 48 for an uncorrectable error in device
    // memory, and Xid 63 for the page or row it takes out of service.
    if (std::strcmp(d.vendor, "amd") == 0) {
      vgpu::ras::log_kernel(vgpu::ras::amdgpu_ras_line(d.bus_id, severity, loc, static_cast<uint64_t>(count)));
    } else if (severity == vgpu::ras::Severity::Uncorrected &&
               loc == vgpu::ras::Location::DeviceMemory) {
      vgpu::ras::report_xid(
          d.uuid, d.bus_id, 48, "",
          "An uncorrectable double bit error (DBE) has been detected on GPU in the framebuffer at "
          "partition 0, subpartition 0.");
      if (scheme_of(d) == vgpu::ras::Retirement::Pages)
        vgpu::ras::report_xid(
            d.uuid, d.bus_id, 63, "",
            "ECC page retirement recording event: a page is pending retirement, reboot to activate.");
      else if (scheme_of(d) == vgpu::ras::Retirement::Rows)
        vgpu::ras::report_xid(
            d.uuid, d.bus_id, 63, "", "Row Remapper: New row marked for remapping, reset gpu to activate.");
    }
    std::printf("Injected %lld %s %s ECC error%s into GPU %u (%s).\n", count, ecc.c_str(),
                vgpu::ras::location_name(loc), count == 1 ? "" : "s", i, d.name);
  }
  return 0;
}
