// `vgpu fault`: inject reliability faults into the machine, show its
// reliability state, and reset it.
//
// This is how a health tool is tested against the errors it exists to catch.
// On real hardware an uncorrectable ECC error cannot be provoked on demand;
// here it is one command, and every surface a tool reads -- nvidia-smi, NVML,
// rocm-smi -- reports it, because they all read the same state (vgpu/ras.hpp).
#include <cstdio>
#include <cstring>
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
               "  --aggregate       reset: zero the lifetime ECC counts (nvidia-smi -p 1)\n");
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
}

}  // namespace

int cmd_fault(const std::vector<std::string>& args) {
  if (args.empty()) return usage(stderr);
  const std::string& verb = args[0];
  if (verb == "-h" || verb == "--help" || verb == "help") return usage(stdout);
  if (verb != "inject" && verb != "show" && verb != "reset") {
    std::fprintf(stderr, "vgpu fault: unknown action '%s' (inject, show or reset)\n", verb.c_str());
    return 2;
  }

  std::string gpu, ecc, location, pcie;
  bool vol = false, agg = false;
  long long count = 1;
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    const bool takes_value = a == "--gpu" || a == "--ecc" || a == "--location" || a == "--pcie" ||
                             a == "--count";
    if (takes_value && i + 1 >= args.size()) {
      std::fprintf(stderr, "vgpu fault: %s needs a value\n", a.c_str());
      return 2;
    }
    if (a == "--gpu") gpu = args[++i];
    else if (a == "--ecc") ecc = args[++i];
    else if (a == "--location") location = args[++i];
    else if (a == "--pcie") pcie = args[++i];
    else if (a == "--count") {
      if (!vgpu::cli::parse_int(args[++i], 1, 1000000000000LL, &count)) {
        std::fprintf(stderr, "vgpu fault: --count needs a whole number from 1, got '%s'\n",
                     args[i].c_str());
        return 2;
      }
    } else if (a == "--volatile") vol = true;
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

  if (verb == "show") {
    if (!ecc.empty() || !pcie.empty() || !location.empty() || vol || agg) {
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

  // inject
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
    std::printf("Injected %lld %s %s ECC error%s into GPU %u (%s).\n", count, ecc.c_str(),
                vgpu::ras::location_name(loc), count == 1 ? "" : "s", i, d.name);
  }
  return 0;
}
