// VGPU_EXACT_THREADS (docs/exact-threads.md): with VGPU_THREADS=1 a launch gives the bits of its blocks run in order, and with
// VGPU_EXACT_THREADS=N beside it the SASS executor runs the blocks on N host threads on trial, under a shadow of device memory
// (MemoryManager::spec_*), and keeps the trial only if no block touched what another wrote. These tests hold it to that:
//   * kernels whose blocks are independent commit, and leave the memory and every counter of the run in order;
//   * kernels whose blocks write the same bytes, or read what an earlier block wrote, are stopped, undone and run in order, and
//     leave the memory and the counters of the run in order (which a parallel run without the check does not);
//   * a kernel that faults in block 1 leaves exactly what the run in order leaves, the later blocks' stores undone, and reports the same error;
//   * a kernel that reads its SM number on a grid larger than the device is stopped (the number is the host thread's);
//   * every method of MemoryManager that reads or writes device bytes is noted by the shadow (a method added without its hook
//     would let a trial commit what it should not), and the shadow is empty again after a trial.
// The kernels are SASS for sm_86 written out word by word; nvdisasm's text of each word is checked against the decoder's.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <unistd.h>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Insn {
  sass::Word w;
  std::string text;   // empty: not checked against the decoder (the encodings that the test only needs to behave, not to read back)
};

std::string reg(unsigned r) { return r == 255 ? "RZ" : "R" + std::to_string(r); }
std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%x", v);
  return b;
}
Insn make(uint64_t hi, uint64_t lo, std::string text) {
  Insn i;
  i.w.hi = hi, i.w.lo = lo;
  i.text = std::move(text);
  return i;
}
// lo: opcode | rd << 16 | ra << 24 | rb or immediate << 32; hi's low byte is the third register of the forms that have one.
uint64_t lo_of(uint64_t op, unsigned rd, unsigned ra, uint64_t b) { return op | uint64_t{rd} << 16 | uint64_t{ra} << 24 | b << 32; }

constexpr unsigned kRZ = 255;
// guard: 7 none (PT), 0 @P0, 8 @!P0
std::string guard_text(unsigned guard) { return guard == 7 ? "" : guard == 8 ? "@!P0 " : "@P0 "; }
Insn iadd3i(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned guard = 7) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x0810 | (guard == 7 ? 0x7000 : guard << 12), d, a, imm),
              guard_text(guard) + "IADD3 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
Insn imadi(unsigned d, unsigned a, uint32_t imm, unsigned c, unsigned guard = 7) {
  return make(0x000fe400078e0200ull | c, lo_of(0x0824 | (guard == 7 ? 0x7000 : guard << 12), d, a, imm),
              guard_text(guard) + "IMAD " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}
Insn s2r_tid(unsigned d) { return make(0x000e220000002100ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_TID.X"); }
Insn exit_() { return make(0x000fea0003800000ull, 0x000000000000794dull, "EXIT"); }
// STG.E [Ra.64 + off], Rk
Insn stg(unsigned a, unsigned k, unsigned off) {
  return make(0x000fe2000c101904ull, lo_of(0x7986, 0, a, k) | uint64_t{off} << 40,
              "STG.E [" + reg(a) + ".64" + (off ? "+" + hex(off) : "") + "], " + reg(k));
}
// IMAD.WIDE.U32 Rd, Ra, imm, Rc
Insn imad_wide_i(unsigned d, unsigned a, uint32_t imm, unsigned c) {
  return make(0x000fe200078e0000ull | c, lo_of(0x7825, d, a, imm),
              "IMAD.WIDE.U32 " + reg(d) + ", " + reg(a) + ", " + hex(imm) + ", " + reg(c));
}

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_86 with one kernel taking one 8-byte pointer.
std::string make_cubin(const std::vector<Insn>& code) {
  const std::string names("\0.shstrtab\0.text.k\0.nv.info.k\0", 31);
  std::string info;   // KPARAM_INFO: index, ordinal | offset << 16, size << 18
  put<uint8_t>(info, 4), put<uint8_t>(info, 0x17), put<uint16_t>(info, 12);
  put<uint32_t>(info, 0), put<uint32_t>(info, 0), put<uint32_t>(info, 8u << 18);
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const Insn& i : code) put(out, i.w.lo), put(out, i.w.hi);
  const uint64_t info_off = out.size();
  out += info;
  while (out.size() % 8) out.push_back('\0');
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name), put<uint32_t>(out, type);
    put<uint64_t>(out, 0), put<uint64_t>(out, 0), put<uint64_t>(out, off), put<uint64_t>(out, size);
    put<uint32_t>(out, 0), put<uint32_t>(out, 0), put<uint64_t>(out, 16), put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  shdr(11, 1, text_off, 16 * code.size());
  shdr(19, 0x70000000, info_off, info.size());
  out[0] = 0x7f, out[1] = 'E', out[2] = 'L', out[3] = 'F', out[4] = 2, out[5] = 1, out[6] = 1, out[7] = 0x41, out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 4, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (86u << 8);
  std::memcpy(&out[16], &type, 2), std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8), std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2), std::memcpy(&out[0x3c], &shnum, 2), std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

Insn s2r_ctaid(unsigned d) { return make(0x000ea20000002500ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_CTAID.X"); }
Insn s2r_smid(unsigned d) { return make(0x000e220000004300ull, lo_of(0x7919, d, 0, 0), "S2R " + reg(d) + ", SR_VIRTUALSMID"); }
Insn iadd3(unsigned d, unsigned a, unsigned b, unsigned c) {
  return make(0x000fc80007ffe000ull | c, lo_of(0x7210, d, a, b), "IADD3 " + reg(d) + ", " + reg(a) + ", " + reg(b) + ", " + reg(c));
}
// LDG.E Rd, [Ra.64 + off]
Insn ldg_off(unsigned d, unsigned a, unsigned off) {
  return make(0x000ea2000c1e1900ull, lo_of(0x7981, d, a, 4) | uint64_t{off} << 40,
              "LDG.E " + reg(d) + ", [" + reg(a) + ".64" + (off ? "+" + hex(off) : "") + "]");
}
const Insn kPtrLo = make(0x000fc60000000f00ull, 0x0000580000067a02ull, "MOV R6, c[0x0][0x160]");
const Insn kPtrHi = make(0x000fc60000000f00ull, 0x0000590000077a02ull, "MOV R7, c[0x0][0x164]");

constexpr uint32_t kThreads = 64;        // two warps a block
constexpr uint32_t kIn = 0x40000;        // the byte offset of the input half of the buffer (64 K words: more than any grid here writes)
constexpr uint32_t kWords = 2 * (kIn / 4);

// The start of every kernel: R0 = blockIdx.x, R1 = threadIdx.x, R6:R7 = the buffer, R2 = blockIdx.x * 64 + threadIdx.x, R10:R11 = &buf[R2].
std::vector<Insn> prologue() {
  return {s2r_ctaid(0), s2r_tid(1), kPtrLo, kPtrHi, imadi(2, 0, kThreads, 1), imad_wide_i(10, 2, 4, 6)};
}
// Independent blocks: buf[i] = in[i] + in[0], i = block * 64 + thread. Every block reads in[0]; each writes its own 64 words.
std::vector<Insn> independent() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {ldg_off(3, 10, kIn), ldg_off(4, 6, kIn), iadd3(5, 3, 4, kRZ), stg(10, 5, 0), exit_()});
  return c;
}
// Every block writes buf[thread] = block: in order the last block's number stays.
std::vector<Insn> same_words() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {imad_wide_i(12, 1, 4, 6), stg(12, 0, 0), exit_()});
  return c;
}
// A chain: buf[i + 64] = buf[i] + 1. Block b reads the words block b - 1 wrote: in order row b holds row 0 + b.
std::vector<Insn> chain() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {ldg_off(3, 10, 0), iadd3i(3, 3, 1, kRZ), stg(10, 3, 4 * kThreads), exit_()});
  return c;
}
// In place: buf[i] = buf[i] + 1. Every block reads and writes its own 64 words, and sixteen blocks share each 4 KiB page: a note by
// page cannot tell them apart, a note by word can.
std::vector<Insn> in_place() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {ldg_off(3, 10, 0), iadd3i(3, 3, 1, kRZ), stg(10, 3, 0), exit_()});
  return c;
}
// buf[i] = block, then a store to buf + block * 2^28: far outside every allocation for block 1 and later (block 0 writes buf[0] again).
std::vector<Insn> faults_from_block_1() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {stg(10, 0, 0), imad_wide_i(12, 0, 0x10000000, 6), stg(12, 0, 0), exit_()});
  return c;
}
// A chain on every other word: buf[2 * (i + 64)] = buf[2 * i] + 1. Block b reads the words block b - 1 wrote, on a stride of two words,
// which no run of consecutive elements is: on a host-mapped buffer the loads go through MemoryManager::host_load_batch and the stores
// through host_store_batch (lane by lane with VGPU_FASTPATH=0).
std::vector<Insn> strided_chain() {
  return {s2r_ctaid(0), s2r_tid(1), kPtrLo, kPtrHi, imadi(2, 0, kThreads, 1), imad_wide_i(10, 2, 8, 6),
          ldg_off(3, 10, 0), iadd3i(3, 3, 1, kRZ), stg(10, 3, 8 * kThreads), exit_()};
}
// buf[i] = the SM number.
std::vector<Insn> sm_number() {
  std::vector<Insn> c = prologue();
  c.insert(c.end(), {s2r_smid(3), stg(10, 3, 0), exit_()});
  return c;
}

struct Result {
  std::vector<uint32_t> words;
  exec::LaunchStats stats;
  std::string error;   // what the launch threw ("" when it returned)
  sass::ExactCounts counts;   // of this launch alone
  uint64_t buf = 0;           // where the launch's buffer was
};

struct Env {
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");   // sm_86
  uint64_t max_steps = 0;   // a warp's instruction budget for the launches (0: the default)
  bool host = false;        // the buffer is host memory mapped at its own address (pinned / registered, as the runtimes map it), not an allocation
  std::shared_ptr<sass::Module> load(const std::vector<Insn>& code) {
    for (size_t i = 0; i < code.size(); ++i) {
      const sass::Instr ins = sass::decode(code[i].w, 16 * i, 86);
      if (!code[i].text.empty() && sass::to_text(ins) != code[i].text)
        throw vtest::Failure("word " + std::to_string(i) + " is \"" + sass::to_text(ins) + "\", not \"" + code[i].text + "\"");
    }
    const std::string c = make_cubin(code);
    return sass::load(reinterpret_cast<const uint8_t*>(c.data()), c.size(), mem, prof);
  }
  // One launch of `blocks` blocks of 64 threads over a buffer whose words start as 3 * index + 7.
  Result run(sass::Module& m, uint32_t blocks) {
    std::vector<uint32_t> init(kWords);
    for (uint32_t i = 0; i < kWords; ++i) init[i] = 3 * i + 7;
    uint32_t* hp = host ? static_cast<uint32_t*>(std::aligned_alloc(4096, uint64_t{kWords} * 4)) : nullptr;
    const uint64_t buf = host ? reinterpret_cast<uint64_t>(hp) : mem.alloc(uint64_t{kWords} * 4);
    // (with a host buffer, also an allocation of the device: a device with no memory in use has nothing to shadow, and is not tried)
    const uint64_t dev = host ? mem.alloc(4096) : 0;
    if (host) {
      std::memcpy(hp, init.data(), uint64_t{kWords} * 4);
      mem.map_host(buf, hp, uint64_t{kWords} * 4);
    } else {
      mem.write(buf, init.data(), uint64_t{kWords} * 4);
    }
    exec::LaunchConfig cfg;
    cfg.grid = {blocks, 1, 1};
    cfg.block = {kThreads, 1, 1};
    if (max_steps) cfg.max_steps = max_steps;
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &buf, 8);
    Result r;
    r.buf = buf;
    const sass::ExactCounts before = sass::exact_counts();
    try {
      r.stats = sass::launch(m, "k", cfg, {arg}, mem, prof);
    } catch (const Error& e) {
      r.error = e.what();
      // (every run has a buffer of its own: the one address a message here names, buf + 2^28, is written the same for all)
      char far[32];
      std::snprintf(far, sizeof far, "0x%llx", static_cast<unsigned long long>(buf + 0x10000000ull));
      if (const size_t at = r.error.find(far); at != std::string::npos) r.error.replace(at, std::strlen(far), "buf+2^28");
    }
    const sass::ExactCounts after = sass::exact_counts();
    r.counts = {after.trials - before.trials, after.committed - before.committed, after.stopped - before.stopped, after.again - before.again};
    r.words.resize(kWords);
    if (host) {
      std::memcpy(r.words.data(), hp, uint64_t{kWords} * 4);
      mem.unmap_host(buf);
      std::free(hp);
      mem.free(dev);
    } else {
      mem.read(buf, r.words.data(), uint64_t{kWords} * 4);
      mem.free(buf);
    }
    return r;
  }
};

void in_order() {
  ::setenv("VGPU_THREADS", "1", 1);
  ::unsetenv("VGPU_EXACT_THREADS");
}
void exact(const char* n) {
  ::setenv("VGPU_THREADS", "1", 1);
  ::setenv("VGPU_EXACT_THREADS", n, 1);
  ::setenv("VGPU_POOL_MIN_WORK_US", "0", 1);   // (the width rule keeps launches as small as these on one thread: off, so that every launch is tried)
}
struct Restore {
  ~Restore() {
    ::unsetenv("VGPU_THREADS");
    ::unsetenv("VGPU_EXACT_THREADS");
    ::unsetenv("VGPU_POOL_MIN_WORK_US");
  }
};

// Every counter of the two launches, the per-opcode ones too.
void same_counters(const exec::LaunchStats& a, const exec::LaunchStats& b, const std::string& what) {
  if (std::memcmp(static_cast<const void*>(&a), static_cast<const void*>(&b), exec::kCounterWords * sizeof(uint64_t)) != 0)
    throw vtest::Failure(what + ": the counters differ (instructions " + std::to_string(a.instructions) + " and " + std::to_string(b.instructions) + ")");
  if (a.inst_by_opcode != b.inst_by_opcode) throw vtest::Failure(what + ": the per-opcode counts differ");
}

// The kernel in order, then with VGPU_EXACT_THREADS = 2, 3, 4 and 8 (three launches each, and each on a module of its own so that
// no launch is held back by an earlier stop): the same words, the same counters, the same error, and the trial ending as `commits` says.
Result same_as_in_order(const std::vector<Insn>& code, uint32_t blocks, bool commits, const std::string& name, uint64_t max_steps = 0,
                        bool host = false) {
  Restore restore;
  Env e;
  e.max_steps = max_steps;
  e.host = host;
  in_order();
  const Result ref = e.run(*e.load(code), blocks);
  VCHECK_EQ(ref.counts.trials, uint64_t{0});
  for (const char* n : {"2", "3", "4", "8"}) {
    exact(n);
    for (int rep = 0; rep < 3; ++rep) {
      const Result got = e.run(*e.load(code), blocks);
      const std::string what = name + ", VGPU_EXACT_THREADS=" + n;
      if (got.words != ref.words) {
        size_t i = 0;
        while (got.words[i] == ref.words[i]) ++i;
        throw vtest::Failure(what + ": word " + std::to_string(i) + " is " + std::to_string(got.words[i]) + ", in order " + std::to_string(ref.words[i]));
      }
      if (got.error != ref.error) throw vtest::Failure(what + ": the error is \"" + got.error + "\", in order \"" + ref.error + "\"");
      if (ref.error.empty()) same_counters(got.stats, ref.stats, what);
      if (got.counts.trials != 1 || got.counts.committed != (commits ? 1u : 0u) || got.counts.stopped != (commits ? 0u : 1u))
        throw vtest::Failure(what + ": " + std::to_string(got.counts.trials) + " trials, " + std::to_string(got.counts.committed) + " committed, " +
                             std::to_string(got.counts.stopped) + " stopped");
    }
  }
  return ref;
}

}  // namespace

VTEST(independent_blocks_commit_and_leave_the_bits_and_counters_of_the_run_in_order) {
  const Result r = same_as_in_order(independent(), 300, /*commits=*/true, "independent");
  VCHECK(r.error.empty());
  // (and the run in order is the kernel's arithmetic: buf[i] = in[i] + in[0])
  const uint32_t in0 = 3 * (kIn / 4) + 7;
  for (uint32_t i = 0; i < 300 * kThreads; ++i) VCHECK_EQ(r.words[i], 3 * (kIn / 4 + i) + 7 + in0);
  VCHECK_EQ(r.words[300 * kThreads], 3 * (300 * kThreads) + 7);
  VCHECK_EQ(r.stats.blocks, uint64_t{300});
  VCHECK_EQ(r.stats.global_stores, uint64_t{300} * kThreads);
}

VTEST(blocks_that_write_the_same_words_are_stopped_and_run_in_order) {
  const Result r = same_as_in_order(same_words(), 300, /*commits=*/false, "same words");
  for (uint32_t t = 0; t < kThreads; ++t) VCHECK_EQ(r.words[t], 299u);   // the last block's number
}

VTEST(a_block_that_reads_what_an_earlier_block_wrote_is_stopped_and_run_in_order) {
  const Result r = same_as_in_order(chain(), 200, /*commits=*/false, "chain");
  for (uint32_t b = 1; b <= 200; ++b)
    for (uint32_t t = 0; t < kThreads; ++t) VCHECK_EQ(r.words[b * kThreads + t], 3 * t + 7 + b);   // row 0 plus b: each block saw the one before it
}

VTEST(a_launch_that_faults_leaves_what_the_run_in_order_leaves) {
  const Result r = same_as_in_order(faults_from_block_1(), 300, /*commits=*/false, "fault in block 1");
  VCHECK_CONTAINS(r.error, "device memory write at buf+2^28");
  // In order: block 0 ran, block 1 stored its words and then faulted, no later block ran. (A trial runs later blocks too: their words must be put back.)
  for (uint32_t i = 2 * kThreads; i < 300 * kThreads; ++i) VCHECK_EQ(r.words[i], 3 * i + 7);
}

VTEST(a_launch_whose_blocks_fail_without_touching_memory_fails_as_the_run_in_order_does) {
  // A budget of 8 instructions a warp, three short of the kernel's end: every block fails after its loads and before its store, at no access
  // the shadow would stop. The trial must not be kept (a failed block is a stop), and the launch must report what the run in order reports.
  const Result r = same_as_in_order(independent(), 300, /*commits=*/false, "step budget", 8);
  VCHECK(!r.error.empty());
  for (uint32_t i = 0; i < 300 * kThreads; ++i) VCHECK_EQ(r.words[i], 3 * i + 7);
}

VTEST(a_kernel_that_reads_its_sm_number_on_a_grid_larger_than_the_device_is_stopped) {
  const Result r = same_as_in_order(sm_number(), 300, /*commits=*/false, "SM number");   // (the a10 has 72 SMs)
  for (uint32_t i = 0; i < 300 * kThreads; ++i) VCHECK_EQ(r.words[i], 0u);   // in order every block is on host thread 0
}

VTEST(a_block_that_reads_host_mapped_words_an_earlier_block_wrote_is_run_in_order) {
  // host_load_batch and load_host_run make no note of their reads: host memory is outside the shadow, and a trial never lets a block
  // write it (every writing method refuses host-mapped bytes before the store). So a block that reads what another block of the launch
  // wrote to host-mapped memory can only be in a trial that the write stopped. This holds the batched host path to that.
  const Result r = same_as_in_order(strided_chain(), 200, /*commits=*/false, "strided chain in host memory", 0, /*host=*/true);
  VCHECK(r.error.empty());
  for (uint32_t b = 0; b <= 200; ++b)
    for (uint32_t t = 0; t < kThreads; ++t) {
      const uint32_t i = 2 * (b * kThreads + t);
      VCHECK_EQ(r.words[i], 3 * (2 * t) + 7 + b);   // row 0 plus b: each block saw the one before it
      VCHECK_EQ(r.words[i + 1], 3 * (i + 1) + 7);   // the odd words untouched
    }
  // And it was the store to host memory that stopped it, not a word of the shadow.
  Restore restore;
  char log[] = "/tmp/vgpu_exact_host_XXXXXX";
  const int fd = ::mkstemp(log);
  VCHECK(fd >= 0);
  ::close(fd);
  ::setenv("VGPU_EXACT_LOG", log, 1);
  Env e;
  e.host = true;
  exact("4");
  const Result got = e.run(*e.load(strided_chain()), 200);
  ::unsetenv("VGPU_EXACT_LOG");
  std::string text;
  if (FILE* f = std::fopen(log, "r")) {
    char line[512];
    while (std::fgets(line, sizeof line, f)) text += line;
    std::fclose(f);
  }
  ::unlink(log);
  VCHECK_EQ(got.counts.stopped, uint64_t{1});
  VCHECK(got.words == r.words);
  VCHECK_CONTAINS(text, "\tstopped\ta block wrote outside the device's own memory\t");
}

VTEST(a_kernel_whose_trial_was_stopped_is_held_back_and_tried_again_later) {
  Restore restore;
  Env e;
  exact("4");
  const auto m = e.load(same_words());
  // Stopped, then in order for 1 launch, stopped, in order for 2, stopped, in order for 4: trials at launches 0, 2, 5, 10.
  std::vector<int> tried;
  for (int i = 0; i < 12; ++i)
    if (e.run(*m, 64).counts.trials) tried.push_back(i);
  VCHECK(tried == (std::vector<int>{0, 2, 5, 10}));
}

VTEST(nothing_is_tried_without_the_variable_or_without_blocks_in_order) {
  Restore restore;
  Env e;
  const auto m = e.load(independent());
  exact("4");
  ::unsetenv("VGPU_THREADS");   // (several threads the usual way: the variable asks for nothing)
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{0});
  ::setenv("VGPU_THREADS", "4", 1);
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{0});
  in_order();
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{0});
  exact("1");
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{0});
  exact("4");
  VCHECK_EQ(e.run(*m, 64).counts.trials, uint64_t{1});
  VCHECK_EQ(e.run(*m, 1).counts.trials, uint64_t{0});   // one block: nothing to run side by side
}

// ---- the shadow itself: every method that touches device bytes is noted ----

namespace {
// VGPU_EXACT_PAGES=0 (the fourth way this test is run): every page noted by word from its first touch.
bool pages_on() {
  const char* v = std::getenv("VGPU_EXACT_PAGES");
  return !(v && v[0] == '0' && !v[1]);
}
struct Trial {
  MemoryManager& mem;
  explicit Trial(MemoryManager& m, const MemoryManager::SpecPlan* plan = nullptr) : mem(m) {
    static const MemoryManager::SpecPlan words_only = [] {
      MemoryManager::SpecPlan p;
      p.pages = false;
      return p;
    }();
    mem.spec_begin(2, plan ? plan : pages_on() ? nullptr : &words_only);
    mem.spec_enter(0);
    mem.spec_block(0);
  }
  // This thread goes on as worker `worker` (0 or 1: each has its own memory of the pages it has seen) running block `block`.
  void as(unsigned worker, uint64_t block) {
    mem.spec_enter(worker);
    mem.spec_block(block);
  }
  ~Trial() {
    if (mem.spec_active()) end();
  }
  MemoryManager::SpecOutcome end() {
    mem.spec_leave();
    return mem.spec_end();
  }
};
bool stops(const std::function<void()>& f) {
  try {
    f();
  } catch (const MemoryManager::SpecStop&) {
    return true;
  }
  return false;
}
}  // namespace

VTEST(every_reading_method_is_stopped_at_bytes_another_block_wrote_and_every_writing_method_at_bytes_another_block_read) {
  MemoryManager mem{1 << 24};
  const uint64_t a = mem.alloc(1 << 16);
  std::vector<uint8_t> init(1 << 16);
  for (size_t i = 0; i < init.size(); ++i) init[i] = static_cast<uint8_t>(i * 7 + 1);
  mem.write(a, init.data(), init.size());
  uint64_t v[64] = {};
  uint64_t addrs[32];
  for (int i = 0; i < 32; ++i) addrs[i] = a + 0x100 + 8 * i;
  const uint8_t sizes[2] = {4, 4};
  uint8_t* ptrs[32];
  size_t done = 0;
  const auto same = +[](void*, uint64_t old) -> uint64_t { return old; };
  const auto lane = +[](void*, size_t, uint64_t old) -> uint64_t { return old; };
  // Block 0 writes the words at a + 0x100 .. a + 0x200; each reading method, called as block 1 on those bytes, must stop the trial
  // (or, for a method that declines during a trial, say no: its caller then takes a path that is noted).
  const std::pair<const char*, std::function<bool()>> readers[] = {
      {"read", [&] { return stops([&] { mem.read(a + 0x100, v, 8); }); }},
      {"load_scalar", [&] { return stops([&] { mem.load_scalar(a + 0x100, 4); }); }},
      {"load_run", [&] { return stops([&] { mem.load_run(a + 0x100, v, 128, 4); }); }},
      {"load_batch", [&] { return stops([&] { mem.load_batch(addrs, 32, 8, v); }); }},
      {"load_batch16", [&] { return stops([&] { mem.load_batch16(addrs, 4, v, v + 32); }); }},
      {"load_broadcast", [&] { return stops([&] { mem.load_broadcast(a + 0x100, 32, 4, v); }); }},
      {"load_quad", [&] { return stops([&] { mem.load_quad(a + 0x100, v); }); }},
      {"load_quad_run", [&] { return stops([&] { mem.load_quad_run(a + 0x100, 2, v); }); }},
      {"load_fields", [&] { return stops([&] { mem.load_fields(a + 0x100, sizes, 2, v); }); }},
      {"load_texels", [&] { return stops([&] { mem.load_texels(addrs, 4, 1, 4, v); }); }},
      {"batch_window", [&] { MemoryManager::BatchWindow w; return !mem.batch_window(addrs, 32, 8, &w); }},
  };
  for (const auto& [name, f] : readers) {
    Trial t(mem);
    mem.store_scalar(a + 0x100, 8, 1);
    for (int i = 1; i < 32; ++i) mem.store_scalar(addrs[i], 8, 1);
    mem.spec_block(1);
    if (!f()) throw vtest::Failure(std::string(name) + " read what another block wrote and the trial went on");
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(std::string(name) == "batch_window" ? out.committed : !out.committed);
    if (out.committed) mem.write(a, init.data(), init.size());   // (declined, not stopped: block 0's stores stayed)
  }
  // Block 0 reads them; each writing method, called as block 1, must stop the trial.
  const std::pair<const char*, std::function<bool()>> writers[] = {
      {"write", [&] { return stops([&] { mem.write(a + 0x100, v, 8); }); }},
      {"fill", [&] { const uint8_t z = 0; return stops([&] { mem.fill(a + 0x100, &z, 1, 8); }); }},
      {"store_scalar", [&] { return stops([&] { mem.store_scalar(a + 0x100, 4, 5); }); }},
      {"store_run", [&] { return stops([&] { mem.store_run(a + 0x100, v, 128, 4); }); }},
      {"store_batch", [&] { return stops([&] { mem.store_batch(addrs, 32, 8, v); }); }},
      {"store_strided", [&] { return stops([&] { mem.store_strided(a + 0x100, 8, 32, 8, v); }); }},
      {"batch_store", [&] { MemoryManager::BatchWindow w{}; return stops([&] { mem.batch_store(w, addrs, 32, 8, v); }); }},
      {"store_quad", [&] { return stops([&] { mem.store_quad(a + 0x100, v); }); }},
      {"store_quad_run", [&] { return stops([&] { mem.store_quad_run(a + 0x100, 2, v); }); }},
      {"rmw_scalar", [&] { return stops([&] { mem.rmw_scalar_fn(a + 0x100, 4, same, nullptr, v); }); }},
      {"rmw_if_changed", [&] { return stops([&] { mem.rmw_if_changed_fn(a + 0x100, 4, same, nullptr, v); }); }},
      {"atomic_rmw", [&] { return stops([&] { mem.atomic_rmw_fn(a + 0x100, 4, same, nullptr, v, false); }); }},
      {"rmw_batch", [&] { return stops([&] { mem.rmw_batch_fn(addrs, 32, 8, lane, nullptr, v, &done, false, false); }); }},
      {"atomic_word_ptrs", [&] { return !mem.atomic_word_ptrs(addrs, 32, 8, ptrs); }},
      {"copy_to", [&] { return stops([&] { mem.copy_to(mem, a + 0x100, a + 0x4000, 64); }); }},
  };
  for (const auto& [name, f] : writers) {
    Trial t(mem);
    mem.read(a + 0x100, v, 256);
    mem.spec_block(1);
    if (!f()) throw vtest::Failure(std::string(name) + " wrote what another block read and the trial went on");
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(std::string(name) == "atomic_word_ptrs" ? out.committed : !out.committed);
  }
  // What a trial may not do at all.
  const std::pair<const char*, std::function<void()>> never[] = {
      {"alloc", [&] { mem.alloc(64); }},
      {"free", [&] { mem.free(a); }},
      {"heap_alloc", [&] { mem.heap_alloc(64, 1 << 20); }},
      {"heap_free", [&] { mem.heap_free(a); }},
      {"intern_string", [&] { mem.intern_string("x"); }},
      {"a store outside the device's memory", [&] { mem.store_scalar(a + (uint64_t{1} << 40), 4, 1); }},
  };
  for (const auto& [name, f] : never) {
    Trial t(mem);
    if (!stops(f)) throw vtest::Failure(std::string(name) + " went on during a trial");
    VCHECK(!t.end().committed);
  }
  // Nothing above changed a byte: every trial that wrote was undone or wrote what was there... the stores of block 0 were undone.
  std::vector<uint8_t> now(init.size());
  mem.read(a, now.data(), now.size());
  VCHECK(now == init);
}

VTEST(host_mapped_memory_may_be_read_by_every_block_and_written_by_none) {
  // The methods that read host-mapped bytes (host_load_batch, load_host_run, and the others' host paths) make no note: host memory is
  // outside the shadow. That is sound only because no block of a trial can write it: every writing method, given a host-mapped address,
  // stops the trial before the store, whether the mapping is at the host's own address (as the runtimes map pinned and registered
  // memory) or inside the device's range.
  MemoryManager mem{1 << 24};
  const uint64_t a = mem.alloc(1 << 16);
  constexpr uint64_t kBytes = 8192;
  uint8_t* hp = static_cast<uint8_t*>(std::aligned_alloc(4096, kBytes));
  uint8_t* hq = static_cast<uint8_t*>(std::aligned_alloc(4096, kBytes));
  std::vector<uint8_t> init(kBytes);
  for (size_t i = 0; i < kBytes; ++i) init[i] = static_cast<uint8_t>(i * 11 + 3);
  std::memcpy(hp, init.data(), kBytes);
  std::memcpy(hq, init.data(), kBytes);
  const uint64_t own = reinterpret_cast<uint64_t>(hp);   // at the host's own address: outside the device's range
  const uint64_t inside = a + 0x4000;                     // over device addresses (as tests/unit/test_load_batch16.cpp maps one)
  mem.map_host(own, hp, kBytes);
  mem.map_host(inside, hq, kBytes);
  uint64_t v[64] = {};
  uint8_t bytes[256] = {};
  const uint8_t sizes[2] = {4, 4};
  size_t done = 0;
  const auto same = +[](void*, uint64_t old) -> uint64_t { return old; };
  const auto lane = +[](void*, size_t, uint64_t old) -> uint64_t { return old + 1; };
  for (const uint64_t h : {own, inside}) {
    uint64_t addrs[32];
    for (int i = 0; i < 32; ++i) addrs[i] = h + 0x100 + 8 * i;
    // Two blocks, on two workers, read the same host bytes by every method: nothing stops, the values are the host's, the trial is kept.
    if (h == own) {
      Trial t(mem);
      for (unsigned b = 0; b < 2; ++b) {
        t.as(b, b);
        uint64_t want[32];
        for (int i = 0; i < 32; ++i) std::memcpy(&want[i], hp + 0x100 + 8 * i, 8);
        VCHECK(mem.host_load_batch(addrs, 32, 8, v));
        for (int i = 0; i < 32; ++i) VCHECK_EQ(v[i], want[i]);
        VCHECK(mem.load_host_run(h + 0x100, bytes, 256, 8));
        VCHECK(std::memcmp(bytes, hp + 0x100, 256) == 0);
        VCHECK_EQ(mem.load_scalar(h + 0x100, 8), want[0]);
        mem.read(h + 0x100, bytes, 64);
        VCHECK(std::memcmp(bytes, hp + 0x100, 64) == 0);
        mem.load_run(h + 0x100, bytes, 256, 8);
        mem.load_batch(addrs, 32, 8, v);
        mem.load_batch16(addrs, 4, v, v + 32);
        mem.load_broadcast(h + 0x100, 32, 4, v);
        mem.load_quad(h + 0x100, v);
        VCHECK_EQ(v[0], want[0]);
        mem.load_quad_run(h + 0x100, 2, v);
        mem.load_fields(h + 0x100, sizes, 2, v);
        mem.load_texels(addrs, 4, 1, 4, v);
      }
      t.as(0, 0);
      VCHECK(t.end().committed);
    }
    // A single block writes host-mapped bytes by any method: the trial stops before the store.
    const std::pair<const char*, std::function<void()>> writers[] = {
        {"write", [&] { mem.write(h + 0x100, v, 8); }},
        {"fill", [&] { const uint8_t z = 0; mem.fill(h + 0x100, &z, 1, 8); }},
        {"store_scalar", [&] { mem.store_scalar(h + 0x100, 4, 5); }},
        {"store_run", [&] { mem.store_run(h + 0x100, v, 128, 4); }},
        {"store_host_run", [&] { mem.store_host_run(h + 0x100, v, 128, 4); }},
        {"store_batch", [&] { mem.store_batch(addrs, 32, 8, v); }},
        {"host_store_batch", [&] { mem.host_store_batch(addrs, 32, 8, v); }},
        {"store_strided", [&] { mem.store_strided(h + 0x100, 8, 32, 8, v); }},
        {"store_quad", [&] { mem.store_quad(h + 0x100, v); }},
        {"store_quad_run", [&] { mem.store_quad_run(h + 0x100, 2, v); }},
        {"rmw_scalar", [&] { mem.rmw_scalar_fn(h + 0x100, 4, same, nullptr, v); }},
        {"rmw_if_changed", [&] { mem.rmw_if_changed_fn(h + 0x100, 4, same, nullptr, v); }},
        {"atomic_rmw", [&] { mem.atomic_rmw_fn(h + 0x100, 4, same, nullptr, v, true); }},
        {"rmw_batch", [&] { mem.rmw_batch_fn(addrs, 32, 8, lane, nullptr, v, &done, false, true); }},
        {"host_address", [&] { mem.host_address(h + 0x100, 4); }},
    };
    for (int i = 0; i < 64; ++i) v[i] = 0x5a5a5a5a5a5a5a5aull + i;
    for (const auto& [name, f] : writers) {
      Trial t(mem);
      if (!stops(f)) throw vtest::Failure(std::string(name) + " wrote host-mapped memory and the trial went on");
      VCHECK(!t.end().committed);
    }
  }
  VCHECK(std::memcmp(hp, init.data(), kBytes) == 0);   // no store was made
  VCHECK(std::memcmp(hq, init.data(), kBytes) == 0);
  mem.unmap_host(own);
  mem.unmap_host(inside);
  std::free(hp);
  std::free(hq);
}

VTEST(blocks_may_share_what_they_only_read_and_keep_what_they_alone_write_and_a_stopped_trial_is_undone_to_the_byte) {
  MemoryManager mem{1 << 24};
  const uint64_t a = mem.alloc(3 * 65536 + 10);   // (three chunks and a tail that is no whole word)
  std::vector<uint8_t> init(3 * 65536 + 10);
  for (size_t i = 0; i < init.size(); ++i) init[i] = static_cast<uint8_t>(i * 13 + 5);
  mem.write(a + 100, init.data() + 100, init.size() - 100);   // (the first bytes stay as an allocation starts: never written)
  std::fill(init.begin(), init.begin() + 100, uint8_t{0});
  const auto all = [&] {
    std::vector<uint8_t> now(init.size());
    mem.read(a, now.data(), now.size());
    return now;
  };
  uint64_t v = 0;
  {
    // Two blocks read the same word, each writes words of its own (one of them a byte of the unwritten start, one the last bytes): committed.
    Trial t(mem);
    VCHECK_EQ(mem.load_scalar(a + 4096, 4), mem.load_scalar(a + 4096, 4));
    mem.store_scalar(a + 8, 1, 0xAB);
    mem.spec_block(1);
    mem.load_scalar(a + 4096, 4);
    mem.store_scalar(a + 3 * 65536 + 9, 1, 0xCD);
    mem.store_scalar(a + 12, 1, 0xEF);   // the next word after block 0's: its own
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(out.committed);
    VCHECK_EQ(out.kept_bytes, uint64_t{4 + 2 + 4});   // a word, the two bytes the allocation has of the last word, a word
    init[8] = 0xAB, init[3 * 65536 + 9] = 0xCD, init[12] = 0xEF;
    VCHECK(all() == init);
  }
  {
    // A block writes across two chunks and single bytes here and there, then another block touches one of its words: all of it goes back.
    Trial t(mem);
    std::vector<uint8_t> junk(70000, 0x77);
    mem.write(a + 65536 - 3000, junk.data(), junk.size());
    mem.store_scalar(a + 3, 1, 0x11);
    mem.store_scalar(a + 3 * 65536 + 8, 2, 0x2222);
    mem.spec_block(1);
    mem.store_scalar(a + 2 * 65536 + 2000, 8, 0x3333333333333333ull);   // its own: written, then undone with the rest
    VCHECK(stops([&] { mem.store_scalar(a + 65536, 1, 9); }));           // block 0's
    VCHECK(stops([&] { mem.load_scalar(a + 20000, 4); }));               // (and every access after the stop)
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(!out.committed);
    VCHECK(std::string(out.why) == "two blocks wrote the same bytes");
    VCHECK(all() == init);
  }
  {
    // The shadow is empty again: what was block 0's in the last trial is anyone's now.
    Trial t(mem);
    mem.spec_block(1);
    mem.store_scalar(a + 65536, 4, 0x01020304);
    mem.spec_block(0);
    mem.read(a + 3, &v, 1);
    VCHECK(t.end().committed);
    init[65536] = 4, init[65537] = 3, init[65538] = 2, init[65539] = 1;
    VCHECK(all() == init);
  }
  {
    // A block may read and write its own words in any order; a second reader of a word a block then writes stops that write.
    Trial t(mem);
    mem.load_scalar(a + 200, 4);
    mem.store_scalar(a + 200, 4, 1);
    mem.load_scalar(a + 200, 4);
    mem.load_scalar(a + 300, 4);
    mem.spec_block(1);
    mem.load_scalar(a + 300, 4);
    mem.spec_block(0);
    VCHECK(stops([&] { mem.store_scalar(a + 300, 4, 2); }));
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(std::string(out.why) == "a block wrote what another block read");
    VCHECK(all() == init);
  }
  {
    // ... and so does the write of the block that read it last: it is not the only reader.
    Trial t(mem);
    mem.load_scalar(a + 400, 4);
    mem.spec_block(1);
    mem.load_scalar(a + 400, 4);
    VCHECK(stops([&] { mem.store_scalar(a + 400, 4, 3); }));
    VCHECK(!t.end().committed);
    VCHECK(all() == init);
  }
}

VTEST(a_device_with_memory_the_shadow_cannot_see_is_not_tried) {
  MemoryManager plain{1 << 24};
  VCHECK(!plain.spec_possible());   // nothing allocated yet: nothing to shadow
  plain.alloc(64);
  VCHECK(plain.spec_possible());
  MemoryManager vmm{1 << 26};
  vmm.alloc(64);
  const uint64_t va = vmm.reserve(MemoryManager::kVmmGranularity, MemoryManager::kVmmGranularity);
  VCHECK(!vmm.spec_possible());   // a reservation: the same bytes may be mapped at two addresses
  vmm.address_free(va, MemoryManager::kVmmGranularity);
  VCHECK(vmm.spec_possible());
}

// ---- the note by page (round sp55): pages that are only read are noted as a whole, and nothing a note by word stops gets through ----

namespace {
constexpr uint64_t kPage = 4096;
// A device of its own with one allocation of `pages` pages and six bytes, every byte set; `p` is the first page boundary in it.
struct Paged {
  MemoryManager mem{1 << 24};
  uint64_t a = 0, p = 0, size = 0;
  std::vector<uint8_t> init;
  explicit Paged(uint64_t pages) {
    size = (pages + 1) * kPage + 6;
    a = mem.alloc(size);
    p = (a + kPage - 1) / kPage * kPage;
    init.resize(size);
    for (size_t i = 0; i < init.size(); ++i) init[i] = static_cast<uint8_t>(i * 29 + 3);
    mem.write(a, init.data(), init.size());
  }
  std::vector<uint8_t> all() {
    std::vector<uint8_t> now(size);
    mem.read(a, now.data(), size);
    return now;
  }
  void put_back() { mem.write(a, init.data(), size); }
  bool rd(uint64_t at, uint64_t len) {
    uint8_t b[64];
    return stops([&] { mem.read(at, b, len); });
  }
  bool wr(uint64_t at, uint64_t len) {
    const uint8_t b[16] = {0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0};
    return stops([&] { mem.write(at, b, len); });
  }
};
// Whether the words of [x, x + xl) and of [y, y + yl) meet (a word is four bytes of the address range).
bool words_meet(uint64_t x, uint64_t xl, uint64_t y, uint64_t yl) { return x / 4 <= (y + yl - 1) / 4 && y / 4 <= (x + xl - 1) / 4; }
// Every class of place in a page: each byte of the first, a middle and the last word, and accesses that end in the next page.
struct Place {
  uint64_t off, len;
};
std::vector<Place> places() {
  std::vector<Place> v;
  for (const uint64_t word : {uint64_t{0}, uint64_t{4}, uint64_t{2044}, uint64_t{2048}, uint64_t{4088}, uint64_t{4092}})
    for (uint64_t byte = 0; byte < 4; ++byte)
      for (const uint64_t len : {uint64_t{1}, uint64_t{2}, uint64_t{3}, uint64_t{4}, uint64_t{5}, uint64_t{8}}) v.push_back({word + byte, len});
  return v;
}
}  // namespace

VTEST(a_page_that_blocks_only_read_is_noted_as_a_whole_and_a_page_a_block_writes_in_by_word) {
  Paged d(3);
  Trial t(d.mem);
  VCHECK(!d.rd(d.p + 100, 4));
  VCHECK(!d.rd(d.p + 3000, 8));     // the same page: the worker remembers it
  t.as(1, 1);
  VCHECK(!d.rd(d.p + 100, 4));      // a second reader: the page is several blocks' now
  VCHECK(!d.rd(d.p + 7, 2));
  VCHECK(!d.wr(d.p + kPage + 8, 4));   // another page, written first: by word
  t.as(0, 0);
  VCHECK(!d.rd(d.p + kPage + 12, 4));  // ... where a word next to another block's is free
  VCHECK(!d.wr(d.p + kPage + 16, 4));
  const MemoryManager::SpecOutcome out = t.end();
  VCHECK(out.committed);
  if (pages_on()) {
    VCHECK_EQ(out.stats.pages_read, uint64_t{1});
    VCHECK_EQ(out.stats.pages_shared, uint64_t{1});
    VCHECK_EQ(out.stats.pages_by_word, uint64_t{1});
    VCHECK_EQ(out.stats.word_pages, uint64_t{1});
    VCHECK_EQ(out.stats.words_read, uint64_t{1});          // the one read in the written page
    VCHECK_EQ(out.stats.words_read_first, uint64_t{1});
  } else {
    VCHECK_EQ(out.stats.pages_read, uint64_t{0});
    VCHECK_EQ(out.stats.pages_by_word, uint64_t{2});
    VCHECK_EQ(out.stats.words_read, uint64_t{2 + 2 + 1 + 1 + 1});
  }
  VCHECK_EQ(out.stats.words_written_first, uint64_t{2});
  VCHECK_EQ(out.kept_bytes, uint64_t{8});
  VCHECK_EQ(out.shadow_pages, uint64_t{2});
  // The shadow is empty again: a block may write where two blocks read in the last trial.
  Trial again(d.mem);
  VCHECK(!d.wr(d.p + 100, 4));
  VCHECK(again.end().committed);
}

// The adversary of the note by page: block 0 claims a page whole (as its reader, or as its reader and then a writer in it), and a
// second block, on the same worker or on another, touches the page at every class of place. Nothing may get through that a note
// by word stops; whatever is stopped is undone to the byte; and what only the coarseness stopped is said to be coarse, names the
// allocation, and goes through when the trial is run again with the allocation noted by word.
VTEST(a_second_block_in_a_page_another_block_claimed_whole_is_stopped_before_its_access_at_every_place) {
  Paged d(3);
  const uint64_t x = d.p + 100, y = d.p + 200;   // what block 0 reads, and what it writes
  const std::pair<uint64_t, uint64_t> whole{d.a, d.a + d.size};
  for (const unsigned other : {0u, 1u}) {
    for (const Place& pl : places()) {
      const uint64_t at = d.p + pl.off;
      const std::string where = "offset " + std::to_string(pl.off) + ", " + std::to_string(pl.len) + " bytes, worker " + std::to_string(other);
      // (1) Block 0 read in the page; block 1 writes in it.
      {
        Trial t(d.mem);
        VCHECK(!d.rd(x, 4));
        t.as(other, 1);
        const bool stopped = d.wr(at, pl.len);
        const MemoryManager::SpecOutcome out = t.end();
        if (pages_on()) {
          if (!stopped || out.committed) throw vtest::Failure("a write in a page another block read by page went on: " + where);
          VCHECK(out.coarse);
          VCHECK(out.by_word == (std::vector<std::pair<uint64_t, uint64_t>>{whole}));
          VCHECK(d.all() == d.init);
          // Again, by word: only a write of the very word block 0 read is stopped.
          MemoryManager::SpecPlan plan;
          plan.by_word = out.by_word;
          Trial t2(d.mem, &plan);
          VCHECK(!d.rd(x, 4));
          t2.as(other, 1);
          VCHECK_EQ(d.wr(at, pl.len), words_meet(at, pl.len, x, 4));
          const MemoryManager::SpecOutcome out2 = t2.end();
          VCHECK(!out2.coarse);
          VCHECK_EQ(out2.committed, !words_meet(at, pl.len, x, 4));
        } else {
          VCHECK_EQ(stopped, words_meet(at, pl.len, x, 4));
          VCHECK(!out.coarse);
        }
        d.put_back();
      }
      // (2) Block 0 read in the page, block 1 read in it at the place; then block 0 writes at the place, and block 1 somewhere else.
      for (const bool first_writes : {true, false}) {
        Trial t(d.mem);
        VCHECK(!d.rd(x, 4));
        t.as(other, 1);
        VCHECK(!d.rd(at, pl.len));
        if (first_writes) t.as(0, 0);
        const uint64_t target = first_writes ? at : x;
        const uint64_t tlen = first_writes ? pl.len : 4;
        const bool stopped = d.wr(target, tlen);
        const MemoryManager::SpecOutcome out = t.end();
        // By word: block 0 writing where block 1 read is a conflict; block 1 writing what block 0 read is too.
        if (!stopped || out.committed) throw vtest::Failure("a write of what another block read went on: " + where);
        VCHECK(d.all() == d.init);
        if (!pages_on()) VCHECK(!out.coarse);
      }
      // (3) Block 0 read in the page and then wrote in it (the page's words all say "read by block 0"); block 1 reads at the place.
      {
        Trial t(d.mem);
        VCHECK(!d.rd(x, 4));
        VCHECK(!d.wr(y, 4));
        VCHECK(!d.rd(d.p + 300, 4));   // (its own page still)
        t.as(other, 1);
        const bool hit = words_meet(at, pl.len, y, 4);
        if (d.rd(at, pl.len) != hit) throw vtest::Failure(std::string(hit ? "a read of what another block wrote went on: " : "a read beside another block's store was stopped: ") + where);
        const MemoryManager::SpecOutcome out = t.end();
        VCHECK_EQ(out.committed, !hit);
        VCHECK(!out.coarse);   // a writer's mark is exact
        if (hit) VCHECK(d.all() == d.init);
        d.put_back();
      }
      // (4) The same, and block 1 writes at the place: stopped everywhere with pages (coarse, unless it is block 0's written word),
      // only at the words block 0 touched without.
      {
        Trial t(d.mem);
        VCHECK(!d.rd(x, 4));
        VCHECK(!d.wr(y, 4));
        t.as(other, 1);
        const bool stopped = d.wr(at, pl.len);
        const bool real = words_meet(at, pl.len, y, 4) || words_meet(at, pl.len, x, 4);
        const MemoryManager::SpecOutcome out = t.end();
        if (pages_on()) {
          if (!stopped || out.committed) throw vtest::Failure("a write in a page another block read and wrote in went on: " + where);
          if (at / 4 == y / 4) VCHECK(!out.coarse);   // the first word it meets is block 0's written one
          else if (!real) VCHECK(out.coarse);
        } else {
          VCHECK_EQ(stopped, real);
          VCHECK(!out.coarse);
        }
        if (stopped) VCHECK(d.all() == d.init);
        d.put_back();
      }
      // (5) Block 0 wrote in the untouched page first (by word from the start): block 1 is stopped at block 0's word only, never coarsely.
      for (const bool reads : {true, false}) {
        Trial t(d.mem);
        VCHECK(!d.wr(y, 4));
        t.as(other, 1);
        const bool stopped = reads ? d.rd(at, pl.len) : d.wr(at, pl.len);
        VCHECK_EQ(stopped, words_meet(at, pl.len, y, 4));
        const MemoryManager::SpecOutcome out = t.end();
        VCHECK_EQ(out.committed, !stopped);
        VCHECK(!out.coarse);
        if (stopped) VCHECK(d.all() == d.init);
        d.put_back();
      }
    }
  }
}

namespace {
// Where a worker keeps its memory of a page (src/core/memory_spec.cpp, Spec::seen_at): two pages of `d` that share a place in a
// table of `places` entries, as page numbers of `d` (`first` < `second`).
std::pair<uint64_t, uint64_t> pages_that_share_a_place(const Paged& d, uint64_t pages, uint64_t places) {
  const auto place = [&](uint64_t page) { return ((((d.p - kDeviceVaBase) / kPage + page) * 0x9E3779B97F4A7C15ull) >> 56) % places; };
  for (uint64_t j = 1; j < pages; ++j)
    for (uint64_t i = 0; i < j; ++i)
      if (place(i) == place(j)) return {i, j};
  throw vtest::Failure("no two pages share a place");
}
}  // namespace

VTEST(what_a_worker_remembers_of_a_page_is_the_block_s_and_not_the_next_block_s) {
  // One worker runs block 0 and then block 1; both read the same page. If block 1 went by what the worker remembered for block 0,
  // its read would not be noted, and block 0's later write in the page would be let through.
  Paged d(300);
  {
    Trial t(d.mem);
    VCHECK(!d.rd(d.p + 8, 4));
    t.as(0, 1);
    VCHECK(!d.rd(d.p + 8, 4));
    t.as(0, 0);
    VCHECK(d.wr(d.p + 8, 4));
    VCHECK(!t.end().committed);
  }
  {
    // Two pages that share a place in the worker's memory: the memory of one must not answer for the other.
    const auto [one, two] = pages_that_share_a_place(d, 300, 256);
    Trial t(d.mem);
    t.as(1, 1);
    VCHECK(!d.wr(d.p + two * kPage + 8, 4));
    t.as(0, 0);
    VCHECK(!d.rd(d.p + one * kPage + 8, 4));
    VCHECK(!d.rd(d.p + one * kPage + 8, 4));
    VCHECK(d.rd(d.p + two * kPage + 8, 4));   // block 1 wrote it
    VCHECK(!t.end().committed);
  }
  {
    // The same for the pages of words a worker remembers: the words of one page do not answer for another's.
    const auto [one, two] = pages_that_share_a_place(d, 300, 64);
    Trial t(d.mem);
    t.as(1, 1);
    VCHECK(!d.wr(d.p + two * kPage + 8, 4));
    t.as(0, 0);
    VCHECK(!d.wr(d.p + one * kPage + 2048, 4));   // (by word)
    VCHECK(!d.rd(d.p + one * kPage + 8, 4));
    VCHECK(!d.rd(d.p + one * kPage + 8, 4));
    VCHECK(d.rd(d.p + two * kPage + 8, 4));       // block 1 wrote it
    VCHECK(!t.end().committed);
  }
  VCHECK(d.all() == d.init);
}

VTEST(every_lane_of_a_warp_is_noted_wherever_it_lies) {
  Paged d(4);
  uint64_t addrs[32], v[64] = {};
  // Lanes one after another, one word of which is another block's: the load and the store are stopped, nothing is written.
  for (int i = 0; i < 32; ++i) addrs[i] = d.p + 64 + 4 * i;
  for (const int taken : {0, 13, 31}) {
    for (const bool store : {false, true}) {
      Trial t(d.mem);
      VCHECK(!d.wr(addrs[taken], 4));
      t.as(1, 1);
      VCHECK(stops([&] { store ? (void)d.mem.store_batch(addrs, 32, 4, v) : (void)d.mem.load_batch(addrs, 32, 4, v); }));
      const MemoryManager::SpecOutcome out = t.end();
      VCHECK(!out.committed && !out.coarse);
      VCHECK(d.all() == d.init);
    }
  }
  {
    // A warp's stores (lanes one after another, and lanes apart) keep every byte they change, and a later stop puts each back.
    Trial t(d.mem);
    for (int i = 0; i < 32; ++i) v[i] = 0xD0D0D0D0u + i;
    VCHECK(d.mem.store_batch(addrs, 32, 4, v));
    VCHECK(d.mem.store_strided(d.p + 1024, 8, 16, 8, v));
    VCHECK(d.mem.store_strided(d.p + 2048, 16, 4, 8, v));
    t.as(1, 1);
    VCHECK(d.rd(addrs[31], 1));
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(!out.committed);
    VCHECK_EQ(out.kept_bytes, uint64_t{128 + 128 + 32});
    VCHECK(d.all() == d.init);
  }
  // Lanes scattered over one page the block may read without a note, and one lane that lies in another page, or whose last byte
  // does: that lane is still noted (the page holds another block's store).
  for (const int stray : {-1, 0, 7, 31}) {
    for (const uint64_t stray_at : {d.p + 2 * kPage + 16, d.p + kPage - 4}) {
      Trial t(d.mem);
      const uint32_t size = stray_at % kPage == kPage - 4 ? 8 : 4;   // (the second: eight bytes that end in the next page)
      if (stray >= 0) VCHECK(!d.wr(stray_at + size - 4, 4));
      t.as(1, 1);
      for (int i = 0; i < 32; ++i) addrs[i] = d.p + kPage - 8 - 16 * ((i * 7) % 32);
      if (stray >= 0) addrs[stray] = stray_at;
      if (stray != 0) VCHECK(!d.rd(d.p + 8, 4));   // the block has seen the page
      const bool stopped = stops([&] { d.mem.load_batch(addrs, 32, size, v); });
      VCHECK_EQ(stopped, stray >= 0);
      const MemoryManager::SpecOutcome out = t.end();
      VCHECK_EQ(out.committed, stray < 0);
      VCHECK(d.all() == d.init);
    }
  }
  {
    // ... and a lane that starts in the page before (in another block's store) and ends in the page the block may read.
    Trial t(d.mem);
    VCHECK(!d.wr(d.p + kPage - 4, 4));
    t.as(1, 1);
    VCHECK(!d.rd(d.p + kPage + 8, 4));
    for (int i = 0; i < 32; ++i) addrs[i] = d.p + kPage + 16 * ((i * 7) % 32);
    addrs[5] = d.p + kPage - 4;
    VCHECK(stops([&] { d.mem.load_batch(addrs, 32, 8, v); }));
    VCHECK(!t.end().committed);
    VCHECK(d.all() == d.init);
  }
}

// Random interleavings of four blocks on two workers over five pages, against a model that keeps the four word states of
// docs/exact-threads.md for every word. Noted by word throughout, the shadow must stop exactly the access the model stops.
// Noted by page, it must stop every access the model stops and may stop earlier, but then says the stop was coarse, and with the
// allocation noted by word the same run is the model's again. A stopped trial leaves the bytes of before; a kept one the stores.
VTEST(random_blocks_against_a_model_of_the_word_states) {
  Paged d(4);
  const uint64_t words = (d.size + 3) / 4 + 2;
  const uint64_t w0 = d.a / 4;
  uint64_t seed = 0x9E3779B97F4A7C15ull;
  const auto rnd = [&](uint64_t n) {
    seed ^= seed << 13, seed ^= seed >> 7, seed ^= seed << 17;
    return seed % n;
  };
  struct Op {
    unsigned block, kind;   // 0 read, 1 write, 2 a warp's load, 3 a warp's store, 4 a warp's scattered load, 6 a warp's scattered store
    uint64_t at, len;
  };
  unsigned kept = 0, stopped_real = 0, stopped_coarse = 0;
  for (int round = 0; round < 600; ++round) {
    std::vector<Op> ops(8 + rnd(40));
    const bool apart = rnd(4) != 0;   // mostly blocks that write in their own quarter, so that many runs get far
    for (Op& o : ops) {
      o.block = static_cast<unsigned>(rnd(4));
      o.kind = static_cast<unsigned>(rnd(22));
      o.kind = o.kind < 8 ? 0 : o.kind < 12 ? 1 : o.kind < 20 ? 2 + (o.kind - 12) / 2 : o.kind == 20 ? 6 : 2;   // (5 is a read of a long range)
      const uint64_t quarter = d.size / 4;
      const bool writes = o.kind == 1 || o.kind == 3 || o.kind == 6;
      const uint64_t lo = writes && apart ? o.block * quarter : 0, span = writes && apart ? quarter - 200 : d.size - 200;
      o.at = d.a + lo + rnd(span);
      o.len = o.kind == 5 ? 1 + rnd(6000) : 1 + rnd(9);
      if (o.kind == 5) o.at = d.a + rnd(d.size - 6000), o.kind = 0;
      if (o.kind >= 2) o.at &= ~uint64_t{3}, o.len = 2 + rnd(o.kind == 6 ? 14 : 31);   // lanes of four bytes (kind 6: every third word)
    }
    // The run under `plan`; says how it ended and, when it stopped, at which op (-1: none).
    const auto play = [&](const MemoryManager::SpecPlan* plan, MemoryManager::SpecOutcome* out, std::vector<uint8_t>* expect) {
      std::vector<uint32_t> state(words, 0);   // the model: 0, block + 1, ~0 (several readers), 0x80000000 | block + 1
      *expect = d.init;
      Trial t(d.mem, plan);
      int at_op = -1;
      for (size_t i = 0; i < ops.size() && at_op < 0; ++i) {
        const Op& o = ops[i];
        t.as(o.block % 2, o.block);
        const uint32_t tag = o.block + 1;
        uint64_t addrs[32], v[32];
        const uint64_t lanes = o.len;
        bool model_stop = false;
        const auto touch = [&](uint64_t a, uint64_t n, bool write) {
          for (uint64_t g = a / 4; g <= (a + n - 1) / 4 && !model_stop; ++g) {
            uint32_t& s = state[g - w0];
            if (write) {
              if (s == 0 || s == tag || s == (tag | 0x80000000u)) s = tag | 0x80000000u;
              else model_stop = true;
            } else {
              if (s == 0) s = tag;
              else if (s == tag || s == ~0u || s == (tag | 0x80000000u)) {}
              else if (s & 0x80000000u) model_stop = true;
              else s = ~0u;
            }
          }
        };
        bool got = false;
        if (o.kind == 0) {
          touch(o.at, o.len, false);
          std::vector<uint8_t> b(o.len);
          got = stops([&] { d.mem.read(o.at, b.data(), o.len); });
        } else if (o.kind == 1) {
          touch(o.at, o.len, true);
          std::vector<uint8_t> b(o.len, static_cast<uint8_t>(0xC0 + i));
          got = stops([&] { d.mem.write(o.at, b.data(), o.len); });
          if (!got) std::fill(expect->begin() + (o.at - d.a), expect->begin() + (o.at - d.a) + o.len, static_cast<uint8_t>(0xC0 + i));
        } else {
          for (uint64_t l = 0; l < lanes; ++l) {
            addrs[l] = o.kind == 4 ? (o.at & ~(kPage - 1)) + 4 * ((l * 37 + i) % 1024) : o.kind == 6 ? o.at + 12 * l : o.at + 4 * l;
            if (addrs[l] + 4 > d.a + d.size || addrs[l] < d.a) addrs[l] = o.at;
            v[l] = 0x01010101u * static_cast<uint32_t>(i + 1);
          }
          for (uint64_t l = 0; l < lanes && !model_stop; ++l) touch(addrs[l], 4, o.kind == 3 || o.kind == 6);
          if (o.kind == 3 || o.kind == 6) {
            got = stops([&] {
              if (!d.mem.store_batch(addrs, lanes, 4, v))
                for (uint64_t l = 0; l < lanes; ++l) d.mem.store_scalar(addrs[l], 4, v[l]);
            });
            if (!got)
              for (uint64_t l = 0; l < lanes; ++l) std::memcpy(expect->data() + (addrs[l] - d.a), &v[l], 4);
          } else {
            got = stops([&] { d.mem.load_batch(addrs, lanes, 4, v); });
          }
        }
        if (model_stop && !got) throw vtest::Failure("round " + std::to_string(round) + ", op " + std::to_string(i) + ": the model stops this access and the shadow let it through");
        if (got) {
          at_op = static_cast<int>(i);
          if (!model_stop) at_op += 1000000;   // stopped where the model goes on
        }
      }
      *out = t.end();
      return at_op;
    };
    MemoryManager::SpecPlan words_only;
    words_only.pages = false;
    MemoryManager::SpecOutcome exact_out, page_out, again_out;
    std::vector<uint8_t> exact_expect, page_expect, again_expect;
    const int exact_at = play(&words_only, &exact_out, &exact_expect);
    if (exact_at >= 1000000) throw vtest::Failure("round " + std::to_string(round) + ": noted by word, the shadow stopped an access the model lets through");
    VCHECK_EQ(exact_out.committed, exact_at < 0);
    VCHECK(!exact_out.coarse);
    VCHECK(d.all() == (exact_at < 0 ? exact_expect : d.init));
    d.put_back();
    const int page_at = play(nullptr, &page_out, &page_expect);
    VCHECK_EQ(page_out.committed, page_at < 0);
    VCHECK(d.all() == (page_at < 0 ? page_expect : d.init));
    d.put_back();
    if (page_at >= 1000000) {
      // Earlier than the model: only ever as a coarse stop, and never without the page grain.
      VCHECK(pages_on());
      VCHECK(page_out.coarse);
      VCHECK(page_out.by_word == (std::vector<std::pair<uint64_t, uint64_t>>{{d.a, d.a + d.size}}));
      MemoryManager::SpecPlan plan;
      plan.by_word = page_out.by_word;
      const int again_at = play(&plan, &again_out, &again_expect);
      VCHECK_EQ(again_at, exact_at);
      VCHECK(!again_out.coarse);
      VCHECK(d.all() == (again_at < 0 ? again_expect : d.init));
      d.put_back();
      ++stopped_coarse;
    } else {
      VCHECK_EQ(page_at, exact_at);
      ++(page_at < 0 ? kept : stopped_real);
    }
  }
  // (every kind of ending was met: the test is not vacuous)
  VCHECK(kept >= 20);
  VCHECK(stopped_real >= 20);
  if (pages_on()) VCHECK(stopped_coarse >= 20);
  else VCHECK_EQ(stopped_coarse, 0u);
}

VTEST(a_kernel_whose_blocks_share_pages_they_read_and_write_is_tried_again_by_word_and_kept) {
  Restore restore;
  Env e;
  in_order();
  const Result ref = e.run(*e.load(in_place()), 300);
  for (uint32_t i = 0; i < 300 * kThreads; ++i) VCHECK_EQ(ref.words[i], 3 * i + 8);
  for (const char* n : {"2", "4", "8"}) {
    exact(n);
    const auto m = e.load(in_place());
    uint64_t learned = 0;
    for (int launch = 0; launch < 3; ++launch) {
      const Result got = e.run(*m, 300);
      VCHECK(got.words == ref.words);
      same_counters(got.stats, ref.stats, std::string("in place, VGPU_EXACT_THREADS=") + n);
      VCHECK_EQ(got.counts.trials, uint64_t{1});
      VCHECK_EQ(got.counts.committed, uint64_t{1});
      // The first launch meets the coarse stop and learns the buffer; a later one notes it by word from the start (when its buffer
      // is where the learned one was: every run has a buffer of its own).
      VCHECK_EQ(got.counts.again, static_cast<uint64_t>(pages_on() && got.buf != learned));
      learned = got.buf;
    }
  }
}

VTEST(a_grid_of_a_few_blocks_is_noted_by_word_from_the_start) {
  // The same kernel on 6 blocks: no trial is thrown away to find out that the buffer is updated in place (with so few blocks, and a
  // store at a block's end, that trial would be most of the launch). On 8 blocks the page grain is tried.
  Restore restore;
  Env e;
  in_order();
  const Result ref6 = e.run(*e.load(in_place()), 6), ref8 = e.run(*e.load(in_place()), 8);
  exact("4");
  const Result few = e.run(*e.load(in_place()), 6);
  VCHECK(few.words == ref6.words);
  VCHECK_EQ(few.counts.committed, uint64_t{1});
  VCHECK_EQ(few.counts.again, uint64_t{0});
  const Result more = e.run(*e.load(in_place()), 8);
  VCHECK(more.words == ref8.words);
  VCHECK_EQ(more.counts.committed, uint64_t{1});
  VCHECK_EQ(more.counts.again, static_cast<uint64_t>(pages_on()));
}

VTEST(a_chain_stopped_by_a_coarse_note_is_tried_by_word_and_then_run_in_order) {
  // chain(): block b reads the row block b - 1 wrote, in pages both touch. By page the first stop may be coarse; the second trial is
  // by word, meets the real conflict, and the launch runs in order.
  Restore restore;
  Env e;
  exact("4");
  const auto m = e.load(chain());
  const Result got = e.run(*m, 200);
  VCHECK_EQ(got.counts.trials, uint64_t{1});
  VCHECK_EQ(got.counts.stopped, uint64_t{1});
  if (!pages_on()) VCHECK_EQ(got.counts.again, uint64_t{0});
  else VCHECK(got.counts.again <= 3);
  for (uint32_t b = 1; b <= 200; ++b)
    for (uint32_t t = 0; t < kThreads; ++t) VCHECK_EQ(got.words[b * kThreads + t], 3 * t + 7 + b);
}

VTEST(a_worker_remembers_the_runs_of_words_its_block_noted_for_that_block_only) {
  // (Runs of sixteen words or more are remembered, and asked for accesses of sixteen words or more: a warp's lanes one after another.)
  Paged d(3);
  uint8_t b[256] = {};
  const auto wr = [&](uint64_t at, uint64_t len) { return stops([&] { d.mem.write(at, b, len); }); };
  {
    // A store in a loop: the second and later notes of the same words are answered by the worker's memory.
    Trial t(d.mem);
    VCHECK(!wr(d.p + 64, 64));
    VCHECK(!wr(d.p + 64, 64));
    VCHECK(!wr(d.p + 72, 8));              // (a short access does not ask the memory: its two words are looked at)
    VCHECK(!d.rd(d.p + 64, 64));           // its own stores: readable
    VCHECK(!wr(d.p + 128, 64));            // touches the run: one run of thirty-two words now
    VCHECK(!wr(d.p + 96, 64));
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(out.committed);
    VCHECK_EQ(out.stats.runs_known, uint64_t{3});
    VCHECK_EQ(out.stats.words_written, uint64_t{16 + 2 + 16});
    VCHECK_EQ(out.kept_bytes, uint64_t{128});
    d.put_back();
  }
  for (const unsigned other : {0u, 1u}) {
    // What block 0 is remembered to hold is not block 1's, on the same worker or another: its write and its read are stopped.
    for (const bool reads : {false, true}) {
      Trial t(d.mem);
      VCHECK(!wr(d.p + 64, 64));
      VCHECK(!wr(d.p + 64, 64));
      t.as(other, 1);
      VCHECK(reads ? d.rd(d.p + 64, 64) : wr(d.p + 64, 64));
      VCHECK(!t.end().committed);
      VCHECK(d.all() == d.init);
    }
    // Words block 0 noted for reading (in a page noted by word) and read again: block 1's read of them is still noted (block 0's
    // later write is stopped), and block 1's write is stopped.
    for (const bool reads : {false, true}) {
      Trial t(d.mem);
      VCHECK(!wr(d.p + 512, 4));           // the page is by word from here
      VCHECK(!d.rd(d.p + 64, 64));
      VCHECK(!d.rd(d.p + 64, 64));
      VCHECK(!d.rd(d.p + 70, 9));
      t.as(other, 1);
      if (reads) {
        VCHECK(!d.rd(d.p + 64, 64));
        VCHECK(!d.rd(d.p + 64, 64));
        t.as(0, 0);
        VCHECK(wr(d.p + 80, 4));
      } else {
        VCHECK(wr(d.p + 64, 64));
      }
      const MemoryManager::SpecOutcome out = t.end();
      VCHECK(!out.committed);
      VCHECK_EQ(out.stats.runs_known, reads ? uint64_t{2} : uint64_t{1});
      VCHECK(d.all() == d.init);
    }
  }
  for (const bool across : {true, false}) {
    // Two runs with another block's word between them are not one run: a store across the gap is stopped. And a run answers
    // only for an access that lies in it whole: a store that starts in one and ends past the other block's word is stopped.
    Trial t(d.mem);
    t.as(1, 1);
    VCHECK(!wr(d.p + 128, 4));
    t.as(0, 0);
    VCHECK(!wr(d.p + 64, 64));
    VCHECK(!wr(d.p + 132, 64));
    VCHECK(across ? wr(d.p + 64, 132) : wr(d.p + 100, 64));
    VCHECK(!t.end().committed);
    VCHECK(d.all() == d.init);
  }
}

VTEST(the_words_a_worker_finds_readable_are_looked_at_every_time_and_for_the_block_that_reads) {
  // A worker that has read in a page noted by word keeps where its words are, and a read whose words already allow it needs no
  // more. What allows block 0 does not allow block 1, and a word another block wrote since is seen.
  Paged d(3);
  for (const unsigned other : {0u, 1u}) {
    Trial t(d.mem);
    VCHECK(!d.wr(d.p + 2048, 4));          // the page is by word
    VCHECK(!d.rd(d.p + 64, 4));
    VCHECK(!d.rd(d.p + 64, 4));            // (its own reader mark: nothing to do)
    VCHECK(!d.rd(d.p + 72, 8));
    t.as(other, 1);
    VCHECK(!d.wr(d.p + 68, 4));            // a free word between block 0's
    VCHECK(!d.rd(d.p + 72, 4));            // block 1 reads what block 0 read: it must be noted (several readers) ...
    t.as(0, 0);
    VCHECK(d.wr(d.p + 72, 4));             // ... or this store would go through
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(!out.committed);
    VCHECK(std::string(out.why) == "a block wrote what another block read");
    VCHECK(d.all() == d.init);
    Trial t2(d.mem);
    VCHECK(!d.wr(d.p + 2048, 4));
    VCHECK(!d.rd(d.p + 64, 12));
    t2.as(other, 1);
    VCHECK(!d.wr(d.p + 76, 4));
    t2.as(0, 0);
    VCHECK(!d.rd(d.p + 64, 12));           // still its own
    VCHECK(d.rd(d.p + 64, 16));            // the fourth word is block 1's now
    VCHECK(!t2.end().committed);
    VCHECK(d.all() == d.init);
  }
}

VTEST(two_words_taken_at_once_are_taken_only_when_both_are_free_and_kept_like_two) {
  Paged d(3);
  uint8_t b[64];
  std::memset(b, 0x5A, sizeof b);
  // Block 1 holds one word (as its reader or its writer), at an even and at an odd place; block 0's store and load over it, starting
  // at every place around it, are stopped, and what block 0 had taken before the stop goes back.
  for (const uint64_t held : {uint64_t{128}, uint64_t{132}}) {
    for (const bool held_written : {false, true}) {
      for (uint64_t start = held - 20; start <= held; start += 4) {
        for (const int form : {0, 1, 2, 3}) {
          const bool store = form & 1, partner = form & 2;
          Trial t(d.mem);
          VCHECK(!d.wr(d.p + 2048, 4));   // (the page is by word)
          if (partner) VCHECK(!d.rd(d.p + (held ^ 4), 4));   // the other word of the held word's pair is block 0's own
          t.as(1, 1);
          VCHECK(!(held_written ? d.wr(d.p + held, 4) : d.rd(d.p + held, 4)));
          t.as(0, 0);
          const bool stopped = store ? stops([&] { d.mem.write(d.p + start, b, 40); }) : d.rd(d.p + start, 40);
          VCHECK_EQ(stopped, store || held_written);
          const MemoryManager::SpecOutcome out = t.end();
          VCHECK_EQ(out.committed, !stopped);
          if (stopped) VCHECK(d.all() == d.init);
          d.put_back();
        }
      }
    }
  }
  // A store over words the block read one by one before (its own reader marks), from an odd place: every byte kept once.
  for (const uint64_t start : {uint64_t{256}, uint64_t{260}}) {
    Trial t(d.mem);
    VCHECK(!d.wr(d.p + 2048, 4));
    VCHECK(!d.rd(d.p + start + 8, 4));
    VCHECK(!d.rd(d.p + start + 20, 8));
    VCHECK(!stops([&] { d.mem.write(d.p + start, b, 44); }));
    VCHECK(!stops([&] { d.mem.write(d.p + start + 4, b, 36); }));
    t.as(1, 1);
    VCHECK(d.rd(d.p + start + 40, 4));
    const MemoryManager::SpecOutcome out = t.end();
    VCHECK(!out.committed);
    VCHECK_EQ(out.kept_bytes, uint64_t{4 + 44});
    VCHECK_EQ(out.stats.words_written_first, uint64_t{1 + 11});
    VCHECK(d.all() == d.init);
  }
}

VTEST_MAIN
