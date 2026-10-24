// A wave's vector registers start at zero, whatever the wave that had the register file before it left in them.
//
// The register files go back to a per-thread pool when a work-group ends, and the next wave to take one clears only the rows
// the code the last wave ran can have written (the highest register any decoded instruction names, and every row where the
// code indexes registers at run time). These kernels are machine code written out word by word: one writes a high register and
// ends, the next reads that register and stores it, and the two run one after the other on this thread, so the second gets the
// first's register file. It has to read zero, as a fresh file does; and a kernel that writes the register itself, which makes
// the check mean something, has to read what it wrote.
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

constexpr uint32_t kMagic = 0xDEADBEEFu;
constexpr uint32_t kHighRegister = 200;

// The instructions (gfx9 encodings, as the assembler writes them).
const std::vector<uint32_t> kEnd = {0xBF810000};   // s_endpgm
// v200 = 0xDEADBEEF, every lane.
const std::vector<uint32_t> kWriteHigh = {0x7F9002FF, kMagic};   // v_mov_b32 v200, 0xDEADBEEF
// The same through VGPR indexing: v0 is written, and the index (s4 = 200) moves it to v200. Nothing in the code names v200.
const std::vector<uint32_t> kWriteIndexed = {
    0xBE8400FF, kHighRegister,   // s_mov_b32 s4, 200
    0xBF110804,                  // s_set_gpr_idx_on s4, gpr_idx(DST)
    0x7E0002FF, kMagic,          // v_mov_b32 v0, 0xDEADBEEF
    0xBF9C0000,                  // s_set_gpr_idx_off
};
// Stores the work-group's number (s2) in out[group].
const std::vector<uint32_t> kStoreGroupId = {
    0xC0060100, 0x00000000,   // s_load_dwordx2 s[4:5], s[0:1], 0x0
    0xBF8CC07F,               // s_waitcnt lgkmcnt(0)
    0x8F068202,               // s_lshl_b32 s6, s2, 2
    0x80040604,               // s_add_u32 s4, s4, s6
    0x82058005,               // s_addc_u32 s5, s5, 0
    0x7E040204,               // v_mov_b32 v2, s4
    0x7E060205,               // v_mov_b32 v3, s5
    0x7E080202,               // v_mov_b32 v4, s2
    0xDC708000, 0x007F0402,   // global_store_dword v[2:3], v4, off
    0xBF8C0F70,               // s_waitcnt vmcnt(0)
};
// Stores v200 to the address in the kernel's first argument.
const std::vector<uint32_t> kStoreHigh = {
    0xC0060080, 0x00000000,   // s_load_dwordx2 s[2:3], s[0:1], 0x0
    0xBF8CC07F,               // s_waitcnt lgkmcnt(0)
    0x7E040202,               // v_mov_b32 v2, s2
    0x7E060203,               // v_mov_b32 v3, s3
    0xDC708000, 0x007FC802,   // global_store_dword v[2:3], v200, off
    0xBF8C0F70,               // s_waitcnt vmcnt(0)
};

std::vector<uint32_t> cat(std::initializer_list<const std::vector<uint32_t>*> parts) {
  std::vector<uint32_t> all;
  for (const auto* p : parts) all.insert(all.end(), p->begin(), p->end());
  return all;
}

amd::CodeObject object(const std::vector<uint32_t>& words) {
  amd::CodeObject o;
  o.isa = "gfx942";
  o.target = "amdgcn-amd-amdhsa--gfx942";
  o.mach = 0x4c;
  o.abi_major = 1, o.abi_minor = 2;
  o.text.resize(words.size() * 4);
  std::memcpy(o.text.data(), words.data(), o.text.size());
  amd::Kernel k;
  k.name = "k";
  k.entry = 0;
  k.size = o.text.size();
  k.kernarg_size = 8;
  k.kernarg_segment_ptr = true;
  k.user_sgpr_count = 2;
  k.max_flat_workgroup_size = 256;
  k.vgpr_count = 8;
  o.kernels.push_back(k);
  return o;
}

void run(const std::vector<uint32_t>& words, MemoryManager& mem, uint64_t out) {
  amd::CodeObject o = object(words);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = amd::find_kernel(o, "k");
  VCHECK(d.kernel != nullptr);
  d.kernarg = mem.alloc(8);
  mem.write(d.kernarg, &out, 8);
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);
}

uint32_t read_word(MemoryManager& mem, uint64_t at) {
  uint32_t v = 0;
  mem.read(at, &v, 4);
  return v;
}

// First `writer` ends (its register file goes back to the pool), then a kernel stores v200 without writing it.
uint32_t stored_after(const std::vector<uint32_t>& writer, bool writes_it_itself) {
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(4);
  const uint32_t poison = 0x5A5A5A5Au;
  mem.write(out, &poison, 4);
  run(cat({&writer, &kEnd}), mem, out);
  run(writes_it_itself ? cat({&kWriteHigh, &kStoreHigh, &kEnd}) : cat({&kStoreHigh, &kEnd}), mem, out);
  return read_word(mem, out);
}

// One kernel that writes the register and stores it.
uint32_t stored_by(const std::vector<uint32_t>& writer) {
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(4);
  run(cat({&writer, &kStoreHigh, &kEnd}), mem, out);
  return read_word(mem, out);
}

// Many work-groups on one host thread, each of two waves: every one is a recycled group but the first, and has to run.
void many_groups(uint32_t groups, uint32_t size) {
  ::setenv("VGPU_THREADS", "1", 1);
  MemoryManager mem(16ull << 20);
  const uint64_t out = mem.alloc(groups * 4);
  const std::vector<uint32_t> poison(groups, 0x5A5A5A5Au);
  mem.write(out, poison.data(), groups * 4);
  amd::CodeObject o = object(cat({&kStoreGroupId, &kEnd}));
  amd::Dispatch d;
  d.object = &o;
  d.kernel = amd::find_kernel(o, "k");
  d.kernarg = mem.alloc(8);
  mem.write(d.kernarg, &out, 8);
  d.groups[0] = groups;
  d.group_size[0] = size;
  amd::execute(d, mem);
  ::unsetenv("VGPU_THREADS");
  for (uint32_t g = 0; g < groups; ++g) VCHECK_EQ(read_word(mem, out + 4ull * g), g);
}

}  // namespace

VTEST(every_work_group_of_a_launch_runs_when_the_groups_are_set_up_again) {
  many_groups(9, 64);
  many_groups(9, 128);
  many_groups(33, 256);
}

VTEST(a_register_a_kernel_writes_reads_back_as_written) {
  VCHECK_EQ(stored_after(kWriteHigh, true), kMagic);
}

VTEST(gpr_indexing_writes_the_register_the_index_names) {
  VCHECK_EQ(stored_by(kWriteIndexed), kMagic);
}

VTEST(a_register_the_last_wave_wrote_reads_zero_in_the_next) {
  VCHECK_EQ(stored_after(kWriteHigh, false), 0u);
  VCHECK_EQ(stored_after(kWriteHigh, false), 0u);
}

VTEST(a_register_written_through_gpr_indexing_reads_zero_in_the_next_wave) {
  VCHECK_EQ(stored_after(kWriteIndexed, false), 0u);
  VCHECK_EQ(stored_after(kWriteIndexed, false), 0u);
}

VTEST_MAIN
