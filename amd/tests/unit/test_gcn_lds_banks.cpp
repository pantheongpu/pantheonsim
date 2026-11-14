// LDS bank conflicts (SQC_LDS_BANK_CONFLICT), against what real cards counted.
//
// amd/tests/data/lds_bank_conflicts.tsv holds 906 measurements from an RX 6800 and an RX 6700 XT: one wave of 1 to 64 lanes issuing
// one LDS instruction 1000 times at a stride, the counter divided by 1000 (amd/tools/lds-bank-probe says how). The model in
// vgpu/amd_memtraffic.hpp is checked against every row; and the executor, running a kernel of six LDS instructions at a stride, against
// the same rows (a wave64 is two wave32 halves). Nothing here is derived from the model.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_memtraffic.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

const std::string kData = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/";

struct Row {
  std::string card;
  uint32_t wave, stride, lanes, conflicts;
  std::string instruction;   // "ds_read2_b32 0 32": the name, then the two offsets
};

std::vector<Row> rows() {
  std::ifstream in(kData + "lds_bank_conflicts.tsv");
  if (!in) throw vtest::Failure("no measurements at " + kData + "lds_bank_conflicts.tsv");
  std::vector<Row> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> f;
    std::stringstream ss(line);
    for (std::string x; std::getline(ss, x, '\t');) f.push_back(x);
    if (f.size() != 6) throw vtest::Failure("a row that is not six columns: " + line);
    out.push_back({f[0], static_cast<uint32_t>(std::stoul(f[1])), static_cast<uint32_t>(std::stoul(f[3])),
                   static_cast<uint32_t>(std::stoul(f[4])), static_cast<uint32_t>(std::stoul(f[5])), f[2]});
  }
  return out;
}

// What the model says for a row: each access of the instruction, over the lanes the row ran.
uint32_t model(const Row& r) {
  std::stringstream ss(r.instruction);
  std::string name;
  ss >> name;
  uint32_t off[2] = {0, 0};
  ss >> off[0] >> off[1];
  const amd::gcn::LdsBankAccess a = amd::gcn::ds_bank_access(name);
  if (!a.counted) throw vtest::Failure(name + " has no bank model");
  const uint64_t exec = r.lanes >= 64 ? ~0ull : (1ull << r.lanes) - 1;
  uint32_t total = 0;
  for (uint32_t s = 0; s < (a.pair ? 2u : 1u); ++s) {
    uint32_t addr[64] = {};
    for (uint32_t l = 0; l < r.lanes; ++l) addr[l] = (l * r.stride) % 65536 + (a.pair ? off[s] * a.unit : 0);
    total += amd::gcn::lds_bank_conflicts(addr, exec, a);
  }
  return total;
}

amd::CodeObject object() {
  const std::string path = kData + "asm_lds_banks.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

amd::DispatchStats run(const amd::CodeObject& o, uint32_t stride, bool count_banks) {
  MemoryManager mem(16ull << 20);
  const amd::Kernel* k = amd::find_kernel(o, "banks");
  if (!k) throw vtest::Failure("no kernel named banks");
  std::vector<uint8_t> args(k->kernarg_size, 0);
  for (int b = 0; b < 4; ++b) args[static_cast<size_t>(b)] = static_cast<uint8_t>(stride >> (8 * b));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(args.size());
  mem.write(d.kernarg, args.data(), args.size());
  d.groups[0] = 1;
  d.group_size[0] = 64;
  d.count_lds_banks = count_banks;
  return amd::execute(d, mem);
}

}  // namespace

VTEST(the_model_reproduces_every_conflict_a_real_card_counted) {
  const std::vector<Row> all = rows();
  VCHECK(all.size() >= 900);
  size_t wrong = 0;
  std::string first;
  for (const Row& r : all) {
    const uint32_t got = model(r);
    if (got == r.conflicts) continue;
    if (!wrong++) first = r.card + " wave" + std::to_string(r.wave) + " " + r.instruction + " stride " + std::to_string(r.stride) + " lanes " +
                          std::to_string(r.lanes) + ": card " + std::to_string(r.conflicts) + ", model " + std::to_string(got);
  }
  if (wrong) throw vtest::Failure(std::to_string(wrong) + " of " + std::to_string(all.size()) + " rows differ; first: " + first);
}

VTEST(the_rows_cover_what_the_model_claims) {
  // Both cards, both wave sizes, partial waves, reads, writes, atomics and the two-address forms: a regeneration that lost one fails.
  std::map<std::string, size_t> seen;
  for (const Row& r : rows()) {
    ++seen["card " + r.card];
    ++seen["wave " + std::to_string(r.wave)];
    ++seen["lanes " + std::to_string(r.lanes)];
    ++seen[r.instruction.substr(0, r.instruction.find(' '))];
  }
  for (const char* want : {"card rx6800", "card rx6700xt", "wave 32", "wave 64", "lanes 8", "lanes 16", "lanes 20", "lanes 32", "lanes 64",
                           "ds_read_b32", "ds_read_b64", "ds_read_b96", "ds_read_b128", "ds_write_b32", "ds_write_b96", "ds_read_u8",
                           "ds_add_u32", "ds_add_rtn_u32", "ds_add_u64", "ds_read2_b32", "ds_read2_b64", "ds_read2st64_b32", "ds_write2_b32"})
    if (!seen.count(want)) throw vtest::Failure(std::string("no row for ") + want);
}

VTEST(instructions_without_a_measured_model_count_no_conflicts) {
  using amd::gcn::ds_bank_access;
  for (const char* name : {"ds_nop", "ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32", "ds_gws_barrier", "ds_append", "ds_consume",
                           "ds_read_addtid_b32", "ds_write_addtid_b32", "ds_read_b64_tr_b16", "ds_load_tr6_b96", "ds_wrxchg2_rtn_b32",
                           "ds_storexchg_2addr_rtn_b64"})
    VCHECK(!ds_bank_access(name).counted);
  for (const char* name : {"ds_read_b32", "ds_write_b128", "ds_add_u32", "ds_max_rtn_f64", "ds_read2_b64", "ds_load_2addr_b32", "ds_cmpst_b32"})
    VCHECK(ds_bank_access(name).counted);
}

VTEST(the_executor_counts_what_the_card_counted_for_six_instructions_at_a_stride) {
  const amd::CodeObject o = object();
  const std::vector<Row> all = rows();
  const auto card = [&](const std::string& instruction, uint32_t stride) {
    for (const Row& r : all)
      if (r.card == "rx6800" && r.wave == 32 && r.lanes == 32 && r.instruction == instruction && r.stride == stride) return r.conflicts;
    throw vtest::Failure("no measurement of " + instruction + " at stride " + std::to_string(stride));
  };
  // The kernel's order. A wave64 is two wave32 halves, each the card's own count.
  const char* instructions[] = {"ds_write_b32", "ds_read_b32", "ds_read_b64", "ds_read_b128", "ds_add_u32", "ds_read2_b32 0 1"};
  for (uint32_t stride : {16u, 32u, 128u, 4096u}) {
    uint64_t expect = 0;
    for (const char* i : instructions) expect += 2 * card(i, stride);
    const amd::DispatchStats s = run(o, stride, true);
    if (s.counts.lds_bank_conflicts != expect)
      throw vtest::Failure("stride " + std::to_string(stride) + ": the card counts " + std::to_string(expect) + ", the executor " +
                           std::to_string(s.counts.lds_bank_conflicts));
    VCHECK_EQ(s.counts.lds, 6u);
  }
  // Stride 128 puts every lane on one bank: the most there is. 31 conflicts a half for a dword, and so on.
  VCHECK(run(o, 128, true).counts.lds_bank_conflicts > run(o, 16, true).counts.lds_bank_conflicts);
}

VTEST(counting_is_off_unless_asked_for) {
  const amd::CodeObject o = object();
  VCHECK_EQ(run(o, 128, false).counts.lds_bank_conflicts, 0u);
  // The bytes are counted either way: the six instructions move 256 + 256 written and 256 + 512 + 1024 + 512 read.
  const amd::DispatchStats s = run(o, 128, false);
  VCHECK_EQ(s.counts.lds_bytes_wr, 512u);
  VCHECK_EQ(s.counts.lds_bytes_rd, 2304u);
}

VTEST_MAIN
