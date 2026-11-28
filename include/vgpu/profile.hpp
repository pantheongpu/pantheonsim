// Device profiles: the data-driven description of one virtual GPU model.
//
// A profile holds only API-visible, functionally relevant properties. VirtualGPU
// does not model performance, so clock rates, bandwidths, and cache sizes are
// intentionally absent. Profiles ship as restricted-YAML files under profiles/
// and are embedded into the binaries at build time.
//
// Values with `verified: false` are placeholders taken from public vendor
// documentation and have NOT been confirmed against physical hardware by the
// characterization suite (which does not exist yet).
#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vgpu {

struct Limits {
  uint32_t max_threads_per_block = 0;
  std::array<uint32_t, 3> max_block_dim{};
  std::array<uint32_t, 3> max_grid_dim{};
  uint32_t shared_mem_per_block = 0;        // default per-block static limit
  uint32_t shared_mem_per_block_optin = 0;  // opt-in dynamic maximum
  uint32_t registers_per_block = 0;
  uint32_t multiprocessors = 0;
  // Residency ceilings. These are functional, not performance: they decide
  // whether a launch fits on the device at all ("too many resources requested
  // for launch") and how many warps can be resident.
  uint32_t registers_per_sm = 0;
  uint32_t max_threads_per_sm = 0;
  uint32_t max_blocks_per_sm = 0;
  uint32_t max_registers_per_thread = 255;
  uint32_t shared_mem_per_sm = 0;
  // The L2 a program is told the device has, which some size a buffer
  // against so it does not fit. Zero where the profile does not say.
  uint64_t l2_cache_bytes = 0;
};

// Presentation values for monitoring tools (nvidia-smi, rocm-smi, `vgpu smi`).
//
// These are deliberately NOT part of the functional model and are never
// consulted by the execution engine: VirtualGPU does not predict performance.
// They exist so telemetry has a plausible, per-model scale to render against,
// and every value derived from them is labelled synthetic. See
// ARCHITECTURE.md and docs/telemetry.md.
struct TelemetryClass {
  uint32_t power_limit_w = 0;
  uint32_t sm_clock_max_mhz = 0;
  uint32_t mem_clock_max_mhz = 0;
  uint32_t temperature_max_c = 0;
  uint32_t pci_vendor_id = 0;
  uint32_t pci_device_id = 0;
  // Memory the driver keeps for itself: the framebuffer nvidia-smi and NVML
  // report as "total", minus the totalGlobalMem a CUDA program sees. On a real
  // H100 those are 81559 and 81089 MiB, and every datacenter card differs by a
  // few hundred. vram_bytes stays the CUDA number -- that is what the verified
  // profiles measured and what programs size their work by -- and the monitor
  // adds this back. Kept as a difference rather than an absolute so that
  // shrinking VRAM for a laptop-scale run shrinks what nvidia-smi says too.
  uint64_t framebuffer_reserve_bytes = 0;
  // Memory reliability and the PCIe link, as monitoring tools read them.
  //   ecc                 the card has ECC and ships with it on (datasheet).
  //   hbm                 HBM memory. HBM parts remap failing rows; GDDR parts
  //                       with ECC retire pages instead.
  //   memory_temperature  the driver reports a memory sensor. Taken from real
  //                       runs, not from the memory type: most HBM cards in
  //                       the benchmark database report none.
  //   pcie_gen/width      the link real cards of the model most often run at,
  //                       which depends on how they are hosted: a T4 is an x16
  //                       card that clouds attach at x8.
  //   nvlink_version/count  the NVLink generation and the number of links the
  //                       card's data sheet gives, for the cards that have
  //                       them (SXM parts; not PCIe cards, whose optional
  //                       bridge is not what cloud machines have). 0 is none.
  bool ecc = false;
  bool hbm = false;
  bool memory_temperature = false;
  uint32_t pcie_gen = 0;
  uint32_t pcie_width = 0;
  uint32_t nvlink_version = 0;
  uint32_t nvlink_count = 0;
};

// What CUDA reports of an NVIDIA device beyond its limits (cudaDeviceProp,
// cudaDeviceGetAttribute, cuDeviceGetAttribute), where it is a fact of the
// card and not of its compute capability. Each profile's `cuda:` section says
// where its values came from: read from a card where one was at hand, else
// the public datasheet. Zero means the profile did not say, and CUDA then
// reports zero rather than a number made up for it.
struct CudaClass {
  // The memory interface's width in bits (cudaDeviceProp::memoryBusWidth).
  uint32_t memory_bus_width_bits = 0;
  // The clock CUDA calls clockRate, in kHz: the boost clock a driver reports,
  // which on GeForce cards is below the highest clock NVML lists (an RTX 3060
  // says 1837000 here and 2145 MHz there). Zero: the same as the maximum
  // SM clock in `telemetry`.
  uint32_t clock_khz = 0;
  // Copy engines (asyncEngineCount). Zero: not said.
  uint32_t async_engine_count = 0;
  // The L2 a program may set aside for persisting accesses
  // (persistingL2CacheMaxSize), in bytes. Zero: not said, or none (before
  // compute capability 8.0 there is no access policy window at all).
  uint64_t persisting_l2_bytes = 0;
  // Whether two devices of this kind can reach each other's memory: an NVLink or a PCIe peer path.
  // cudaDeviceCanAccessPeer, cudaDeviceGetP2PAttribute and the peer-access calls follow it, for a pair of
  // devices that both say so. True unless the profile says otherwise (every data-centre part has a path); a
  // GeForce pair has none: two RTX 3060s answer cudaDeviceCanAccessPeer 0 (measured), and
  // cudaDeviceEnablePeerAccess is cudaErrorPeerAccessUnsupported.
  bool peer_access = true;
  // cudaDevAttrConcurrentManagedAccess: the host and the device may touch managed memory at once and the
  // driver pages it on demand. True unless the profile says otherwise (Linux since Pascal). A device that says
  // false (an RTX 3060 under WSL, measured) refuses cudaMemPrefetchAsync with cudaErrorInvalidDevice, and
  // refuses the advice that names a device (cudaMemAdviseSetPreferredLocation, ...SetAccessedBy).
  bool concurrent_managed_access = true;
};

// How the multiprocessors of a part are grouped, which decides what thread-block
// clusters can be scheduled: a cluster lives in one GPC (CUDA programming guide).
// NVIDIA publishes the GPC and TPC counts of many parts (the architecture
// whitepapers' tables: "GPCs 8, TPCs 66, SMs 132") and not how many TPCs each
// GPC of a part with some disabled has. So a profile records the counts it has a
// source for, and the spread across GPCs is always derived, never read: the
// enabled TPCs are split as evenly as they go, the first GPCs getting the extra.
// Zero means the profile does not say, and a question that needs it is refused
// by name instead of answered from a layout made up for it.
struct GpuLayout {
  uint32_t gpcs = 0;          // GPCs with at least one TPC enabled
  uint32_t tpcs = 0;          // TPCs enabled in all
  uint32_t sms_per_tpc = 0;   // 2 on every part from Volta on
  // True where the counts above are not the part's own published figures but
  // follow from another part with the same SM count (a GH200's GPU is an
  // H100 SXM5 in everything NVIDIA says), and the profile's comment says which.
  bool counts_derived = false;

  bool known() const { return gpcs && tpcs && sms_per_tpc; }
  // SMs in each GPC, largest first: the spread described above.
  std::vector<uint32_t> gpc_sms() const {
    std::vector<uint32_t> v;
    if (!known()) return v;
    for (uint32_t g = 0; g < gpcs; ++g) v.push_back((tpcs / gpcs + (g < tpcs % gpcs ? 1 : 0)) * sms_per_tpc);
    return v;
  }
};

// What an Intel Xe part is made of, where the limits of a launch come from. The engine's "multiprocessor" is
// an Xe core (the unit that owns a work-group's shared local memory); an Xe core holds EUs, each EU runs
// hardware threads, and a hardware thread runs one sub-group out of a register file of GRFs. Everything here
// is read from public sources or named in DeviceProfile::placeholders; a profile's `placeholders:` list says
// which of its values are stand-ins the engine needs to run and not claims about the part.
struct XeClass {
  uint32_t eus_per_xe_core = 0;
  uint32_t threads_per_eu = 0;       // hardware threads per EU
  uint32_t grf_per_thread = 0;       // general register file entries a thread has (the default, not the large-GRF mode)
  uint32_t grf_bytes = 0;            // bytes in one GRF entry
  std::vector<uint32_t> sub_group_sizes;   // the sub-group widths a kernel may be compiled for; warp_size is one of them

  bool known() const { return eus_per_xe_core && threads_per_eu && grf_per_thread && grf_bytes && !sub_group_sizes.empty(); }
  // Hardware threads in one Xe core.
  uint32_t threads_per_xe_core() const { return eus_per_xe_core * threads_per_eu; }
  // 32-bit register-file words in one Xe core: the engine's registers_per_sm.
  uint64_t register_words_per_xe_core() const {
    return uint64_t{threads_per_xe_core()} * grf_per_thread * grf_bytes / 4;
  }
  bool has_sub_group_size(uint32_t w) const {
    for (uint32_t s : sub_group_sizes)
      if (s == w) return true;
    return false;
  }
};

// ---- Numerics: what a device's arithmetic units do where it is not IEEE arithmetic on the operands --------------------------------
//
// The sum an NVIDIA tensor core makes of f16, bf16 and tf32 products (wmma, mma.sync, HMMA; the rule itself is in
// vgpu/exec/tc_ampere.hpp and docs/matrix-numerics.md). A profile names its model in `numerics.tensor_sum` and the engines ask the
// loaded profile: no engine compares a compute capability for it.
//
//   Sequential  acc += a * b in f32, k in order, round to nearest even. No card is known to do this: it is what a device whose
//               tensor core has not been measured gets.
//   Ampere      chunks of 8 products (tf32: 4), a chunk's terms cut at 2^(top - 25).
//   Hopper      the same rule with other numbers: chunks of 16 (tf32: 8), cut at 2^(top - 26).
enum class TensorSum : uint8_t { Sequential, Ampere, Hopper };
struct TensorSumRule {
  const char* name;
  unsigned chunk;        // products to a chunk, f16 and bf16
  unsigned chunk_tf32;   // ... tf32
  int grid;              // a chunk's terms are cut at 2^(top - grid)
};
inline constexpr TensorSumRule kAmpereSumRule{"ampere: chunks of 8 (tf32 4), cut at 2^(top-25)", 8, 4, 25};
inline constexpr TensorSumRule kHopperSumRule{"hopper: chunks of 16 (tf32 8), cut at 2^(top-26)", 16, 8, 26};
// The rule of a model, or null for the sequential sum. These two are the rules the engines implement (scalar code for both, vector
// kernels for the first), so a profile may name no other numbers: the loader refuses them.
inline constexpr const TensorSumRule* tensor_sum_rule(TensorSum m) {
  return m == TensorSum::Ampere ? &kAmpereSumRule : m == TensorSum::Hopper ? &kHopperSumRule : nullptr;
}
// The model's name as a profile writes it.
inline constexpr const char* tensor_sum_name(TensorSum m) {
  return m == TensorSum::Ampere ? "ampere" : m == TensorSum::Hopper ? "hopper" : "sequential";
}

// The sum of the 8-bit float forms (e4m3 and e5m2 operands, mma.sync). A tensor core adds them as the f16 rule does -- chunks of products and
// the accumulator, each term cut toward zero at 2^(top - grid) -- but with other numbers, and an f32 result is cut to out_bits fraction bits
// before it is stored (an Ada core keeps 13). vgpu/exec/tc_fp8.hpp has the arithmetic, docs/matrix-numerics.md the evidence.
//
//   Sequential  products and sums in f32, k in order: what every device whose FP8 sum is not known gets.
//   Ada         chunks of 16, cut at 2^(top - 14), an f32 result cut to 13 fraction bits.
//   ViaF16      the f16 rule of numerics.tensor_sum: ptxas turns an FP8 mma.sync on these devices into a conversion to f16 and HMMA, so the
//               tensor core sums the products as it does f16 ones (chunks of 16 for Hopper). The SASS engine runs that code as it stands;
//               the PTX engine does the same sum.
enum class Fp8Sum : uint8_t { Sequential, Ada, ViaF16 };
struct Fp8SumRule {
  const char* name;
  unsigned chunk;      // products to a chunk
  int grid;            // a chunk's terms are cut at 2^(top - grid)
  int out_bits;        // fraction bits an f32 result keeps (23: all of them)
};
inline constexpr Fp8SumRule kAdaFp8Rule{"ada: chunks of 16, cut at 2^(top-14), f32 results cut to 13 fraction bits", 16, 14, 13};
inline constexpr const Fp8SumRule* fp8_sum_rule(Fp8Sum m) { return m == Fp8Sum::Ada ? &kAdaFp8Rule : nullptr; }   // ViaF16 has no numbers of its own
inline constexpr const char* fp8_sum_name(Fp8Sum m) { return m == Fp8Sum::Ada ? "ada" : m == Fp8Sum::ViaF16 ? "via_f16" : "sequential"; }

// The sum of the e4m3 and e5m2 forms of wgmma (Hopper's warpgroup MMA, HGMMA/QGMMA in SASS), which is not the sum of the same operands in
// mma.sync. Same arithmetic as Fp8Sum::Ada (vgpu/exec/tc_fp8.hpp) with other numbers; dense forms only.
//
//   Sequential  products and sums in f32, k in order.
//   Hopper      one chunk of 32 products (all of K) and the accumulator, cut at 2^(top - 14), an f32 result cut to 13 fraction bits.
enum class WgmmaFp8Sum : uint8_t { Sequential, Hopper };
inline constexpr Fp8SumRule kHopperWgmmaFp8Rule{"hopper wgmma: one chunk of 32, cut at 2^(top-14), f32 results cut to 13 fraction bits", 32, 14, 13};
inline constexpr const Fp8SumRule* wgmma_fp8_sum_rule(WgmmaFp8Sum m) { return m == WgmmaFp8Sum::Hopper ? &kHopperWgmmaFp8Rule : nullptr; }
inline constexpr const char* wgmma_fp8_sum_name(WgmmaFp8Sum m) { return m == WgmmaFp8Sum::Hopper ? "hopper" : "sequential"; }

// The sum of more matrix forms of f16, bf16 and tf32 operands, which a card does not necessarily sum as its mma.sync (issue #664). The rules are of the
// same family as TensorSum (vgpu/exec/tc_ampere.hpp, "chunks of products and the accumulator, terms cut toward zero at 2^(top - grid)") with the numbers
// of the named model; dense forms only unless the key says otherwise.
//
//   numerics.tensor_sum_wgmma    wgmma (PTX) and HGMMA (SASS), f16, bf16 and tf32 operands: Sequential or Hopper.
//   numerics.tensor_sum_tcgen05  tcgen05.mma (PTX) and UTCHMMA (SASS), .kind::f16 and .kind::tf32: Sequential or Hopper.
//   numerics.tensor_sum_m8n8k4   mma.m8n8k4 with f16 operands (Volta's form): Sequential, Volta (an HMMA.884, Turing keeps it) or ChainCLast (ptxas
//                                emulates it on sm_80 and later; an f32 accumulator only, an f16 one keeps the sequential sum).
//   numerics.tensor_sum_sparse   mma.sp (PTX) and HMMA.SP (SASS), f16, bf16 and tf32 operands: Sequential or Ampere.
//
//   Hopper      chunks of 16 products (tf32: 8), cut at 2^(top - 26): the rule of an mma.sync on a Hopper device.
//   Volta       chunks of 4 products, cut at 2^(top - 24): one HMMA.884 step, no extra alignment bit.
//   ChainCLast  D = (((p0 + p1) + p2) + p3) + C in f32, each add rounded to nearest even (the products are exact in f32).
//   Ampere      the Ampere rule over the stored products: chunks of 8 (tf32: 4), cut at 2^(top - 25). A stored product's partner in B is the one the
//               metadata selects; the places A does not store take no part (an infinity or a NaN there does not reach the sum).
enum class FormSum : uint8_t { Sequential, Hopper, Volta, ChainCLast, Ampere };
inline constexpr TensorSumRule kVoltaSumRule{"volta: chunks of 4, cut at 2^(top-24)", 4, 4, 24};
inline constexpr const TensorSumRule* form_sum_rule(FormSum m) {
  return m == FormSum::Hopper ? &kHopperSumRule : m == FormSum::Volta ? &kVoltaSumRule : m == FormSum::Ampere ? &kAmpereSumRule : nullptr;
}
inline constexpr const char* form_sum_name(FormSum m) {
  return m == FormSum::Hopper ? "hopper" : m == FormSum::Volta ? "volta" : m == FormSum::ChainCLast ? "chain_c_last" : m == FormSum::Ampere ? "ampere" : "sequential";
}

// numerics.cublas_splitk: which cuBLAS split-K table a card's tensor-core GEMMs follow (nvidia/src/cublas_splitk.hpp). cuBLAS cuts some products over K
// and the cut is its heuristic's, a function of the shape and the card's multiprocessor count (issue #734); the tables are those measured on a Tesla T4
// (40 multiprocessors) and an A10G (80). None: no table, the plain chain of chunks for every product.
enum class CublasSplitK : uint8_t { None, T4, A10g };
inline constexpr const char* cublas_splitk_name(CublasSplitK m) { return m == CublasSplitK::T4 ? "t4" : m == CublasSplitK::A10g ? "a10g" : "none"; }

// A profile's `numerics:` section. Absent (present == false): the sequential sum and no published model, which is what an AMD or
// Intel profile has -- their engines do not read this.
struct Numerics {
  bool present = false;
  // numerics.tensor_sum: the model in force by default.
  TensorSum tensor_sum = TensorSum::Sequential;
  // numerics.tensor_sum_chunk, _chunk_tf32, _grid: the named model's numbers, written out so that a profile says what its card
  // does and not only a name. Zero with the sequential sum. Always the numbers of tensor_sum_rule(tensor_sum): checked at load.
  uint32_t tensor_sum_chunk = 0;
  uint32_t tensor_sum_chunk_tf32 = 0;
  int32_t tensor_sum_grid = 0;
  // numerics.tensor_sum_published: the model the literature gives a device whose own sum is not measured (so only beside
  // tensor_sum: sequential). Never in force by default: VGPU_TC_EXACT=published turns it on. Sequential: none is published.
  TensorSum tensor_sum_published = TensorSum::Sequential;
  // numerics.tensor_sum_fp8 and its numbers (_chunk, _grid, _out_bits): the sum of the e4m3 and e5m2 forms. Optional: absent is the
  // sequential sum, with the numbers zero. Present, the numbers are always those of fp8_sum_rule(tensor_sum_fp8).
  Fp8Sum tensor_sum_fp8 = Fp8Sum::Sequential;
  uint32_t tensor_sum_fp8_chunk = 0;
  int32_t tensor_sum_fp8_grid = 0;
  int32_t tensor_sum_fp8_out_bits = 0;
  // numerics.tensor_sum_wgmma_fp8 and its numbers (_chunk, _grid, _out_bits): the same for the e4m3 and e5m2 forms of wgmma. Optional, with the
  // numbers those of wgmma_fp8_sum_rule(tensor_sum_wgmma_fp8).
  WgmmaFp8Sum tensor_sum_wgmma_fp8 = WgmmaFp8Sum::Sequential;
  uint32_t tensor_sum_wgmma_fp8_chunk = 0;
  int32_t tensor_sum_wgmma_fp8_grid = 0;
  int32_t tensor_sum_wgmma_fp8_out_bits = 0;
  // numerics.tensor_sum_wgmma, _tcgen05, _m8n8k4, _sparse (FormSum above). Optional, absent is sequential.
  FormSum tensor_sum_wgmma = FormSum::Sequential;
  FormSum tensor_sum_tcgen05 = FormSum::Sequential;
  FormSum tensor_sum_m8n8k4 = FormSum::Sequential;
  FormSum tensor_sum_sparse = FormSum::Sequential;
  // numerics.cublas_splitk (CublasSplitK above). Optional, absent is none.
  CublasSplitK cublas_splitk = CublasSplitK::None;
};

// What a value in a profile rests on. Exactly four classes; a key carries at most one.
//   Measured    read from, or reproduced bit for bit against, a physical card by this project. The evidence names the card.
//   Documented  the vendor's specification, data sheet or documentation.
//   Inferred    not known of this device itself: carried over from a measured device of the same compute capability or die, or
//               taken from a published model that no card of ours has checked.
//   Unknown     none of the above. The value is what the simulator does, not a claim about the device.
enum class Provenance : uint8_t { Measured, Documented, Inferred, Unknown };
inline constexpr const char* provenance_name(Provenance c) {
  return c == Provenance::Measured ? "measured" : c == Provenance::Documented ? "documented" : c == Provenance::Inferred ? "inferred" : "unknown";
}

struct DeviceProfile {
  std::string id;            // registry id, e.g. "nvidia/h100"
  std::string vendor;        // "nvidia" | "amd" | "intel"
  std::string model;         // marketing/device name as APIs report it
  std::string architecture;  // "ampere" | "hopper" | "blackwell" | "cdna3" | ...
  // AMD's equivalent of a compute capability: the gfx target a binary must be
  // built for. "gfx942" for CDNA3. Empty on NVIDIA.
  std::string gcn_arch;
  // The same target as HIP names the device, with the features it runs with:
  // "gfx942:sramecc+:xnack-". A library that ships code for each setting of a
  // feature picks by it. The bare target where a profile does not say.
  std::string gcn_arch_full;
  int cc_major = 0;          // compute capability (NVIDIA) / ISA generation
  int cc_minor = 0;
  uint32_t warp_size = 0;
  uint64_t vram_bytes = 0;
  bool verified = false;  // true once hardware characterization confirms values
  Limits limits;
  TelemetryClass telemetry;
  CudaClass cuda;
  GpuLayout layout;
  XeClass xe;                // Intel only; empty elsewhere
  // Keys (as "limits.max_threads_per_block" or "xe.sub_group_sizes") whose values are stand-ins the engine
  // needs in order to run, not figures from a source. Empty on a profile every value of which is sourced.
  std::vector<std::string> placeholders;
  std::map<std::string, bool> features;
  Numerics numerics;         // the `numerics:` section; see Numerics
  // The `provenance:` section: the class of each key the profile tags, by the key's name in the file ("numerics.tensor_sum",
  // "limits.multiprocessors", "vram_bytes"). Every numerics.* key a profile has is in here (the loader refuses the profile
  // otherwise); elsewhere a key need not be tagged, and provenance_of says what an untagged key is.
  std::map<std::string, Provenance> provenance;
  // The `evidence:` section: for a tagged key, or for a whole section ("numerics"), the card, driver, toolkit and date, or the
  // document, that the tag rests on. evidence_of resolves a key.
  std::map<std::string, std::string> evidence;

  bool is_placeholder(const std::string& key) const {
    for (const auto& k : placeholders)
      if (k == key) return true;
    return false;
  }

  // The class of a key. A tagged key has its tag. An untagged key that is in `placeholders` is Unknown (a stand-in is by
  // definition not a claim about the part). Any other untagged key has NO per-key class (nullopt): all that is said of it is the
  // profile-wide `verified` flag (true: the characterization was read from a card of this model; false: it was not) and the
  // comments in the file. It is deliberately not turned into "measured" or "documented": a verified profile holds data-sheet
  // values and an unverified one holds values copied from its neighbours, and the flag does not say which key is which.
  std::optional<Provenance> provenance_of(const std::string& key) const;
  // The evidence text of a key: its own entry, else its section's ("numerics" for "numerics.tensor_sum"), else empty.
  const std::string& evidence_of(const std::string& key) const;

  // Shared memory the driver keeps back in every block, on top of what the
  // kernel asks for: 1 KiB from compute capability 8.0 (an RTX 3060 reports
  // cudaDevAttrReservedSharedMemoryPerBlock = 1024), none before. It is what
  // separates a card's per-SM shared memory from its per-block opt-in, and
  // cuda_occupancy.h adds it to every block.
  uint32_t reserved_smem_per_block() const { return vendor == "nvidia" && cc_major >= 8 ? 1024 : 0; }

  // Parses a profile document. `origin` names the source in error messages.
  static DeviceProfile from_yaml(const std::string& src, const std::string& origin);
};

// VGPU_VRAM_MB resizes the card a run sees -- totalGlobalMem directly, and
// nvidia-smi's total through the reserve above. One function so its readers
// cannot disagree: it used to be copied into the runtime and the driver and
// missing from the idle nvidia-smi path, which reported the whole 80 GB card to
// a session whose programs had been given 4. A value that is not a positive
// number is ignored rather than turning the card into one with no memory.
inline void apply_vram_override(DeviceProfile& p) {
  const char* mb = std::getenv("VGPU_VRAM_MB");
  // Digits only. strtoull accepts a leading minus and wraps it -- "-5" parsed
  // as 18446744073709551611 and made a card of about 16 EiB -- and a value
  // that large overflows the multiplication below anyway.
  if (!mb || *mb < '0' || *mb > '9') return;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(mb, &end, 10);
  constexpr unsigned long long kMiB = 1024ull * 1024ull;
  if (*end != '\0' || v == 0 || v > UINT64_MAX / kMiB) return;
  p.vram_bytes = v * kMiB;
}

}  // namespace vgpu
