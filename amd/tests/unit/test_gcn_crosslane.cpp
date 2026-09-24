// A lane reading another lane's register, run and checked against an answer
// worked out without reference to this model.
//
// The kernels are amd/tests/data/crosslane.c, compiled for gfx942. The first
// is the sequence a wave adds itself up with, and it is what pins the rest
// down. Shifting the row by one, two, four and eight while adding leaves each
// lane holding the sum of its row up to itself, and the two broadcasts carry
// the last lane of each row into the rows above it, so every lane ends up
// with the running total of the whole wave up to itself and the last lane
// with all of it. That answer is the inclusive prefix sum, which the test
// computes for itself.
//
// It is an answer no part of this model was used to produce, and it does not
// come out unless every part of the instruction is right: the shift has to
// go down rather than up, it has to stop at the row's edge, a lane with no
// lane to read has to contribute a zero, and the row and bank masks have to
// name the rows and banks the assembler meant. Get any one of them wrong and
// the sums are wrong. The three kernels after it pin down the rest of the
// form the same way: a broadcast within each four lanes, a row read
// backwards, and a lane with no lane to read where the instruction says to
// leave it alone.
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/crosslane.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

template <typename T>
uint64_t upload(MemoryManager& mem, const std::vector<T>& v) {
  const uint64_t p = mem.alloc(v.size() * sizeof(T));
  mem.write(p, v.data(), v.size() * sizeof(T));
  return p;
}

template <typename T>
std::vector<T> download(MemoryManager& mem, uint64_t p, size_t n) {
  std::vector<T> out(n);
  mem.read(p, out.data(), n * sizeof(T));
  return out;
}

uint64_t kernargs(MemoryManager& mem, const amd::Kernel& k, const std::vector<uint64_t>& values) {
  std::vector<uint8_t> buf(k.kernarg_size, 0);
  for (size_t i = 0; i < values.size() && i < k.args.size(); ++i) {
    const amd::KernelArg& a = k.args[i];
    for (uint32_t b = 0; b < a.size && b < 8; ++b) buf[a.offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  }
  const uint64_t p = mem.alloc(buf.empty() ? 1 : buf.size());
  if (!buf.empty()) mem.write(p, buf.data(), buf.size());
  return p;
}

void run(const amd::CodeObject& o, const char* name, MemoryManager& mem, const std::vector<uint64_t>& args) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, args);
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);
}

const int kN = 64;

}  // namespace

VTEST(the_sequence_a_wave_adds_itself_up_with_gives_a_running_total) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = i * 7 - 100;   // both signs, all different
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "wave_sum", mem, {pin, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  int32_t running = 0;
  for (int i = 0; i < kN; ++i) {
    running += in[i];
    if (out[i] != running)
      throw vtest::Failure("wave_sum[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(running) + " (the total of every lane up to this one)");
  }
  // And the last lane holds the wave's own total, which is what the sequence
  // is written for.
  int32_t total = 0;
  for (int i = 0; i < kN; ++i) total += in[i];
  VCHECK_EQ(out[kN - 1], total);
}

VTEST(four_lanes_all_taking_the_first_of_their_four) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = 1000 + i;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "quad_first", mem, {pin, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], in[i & ~3]);
}

VTEST(a_row_read_backwards) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = 1000 + i;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "mirrored", mem, {pin, pout, static_cast<uint64_t>(kN)});

  // Sixteen lanes to a row, and the row is read from its far end.
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) VCHECK_EQ(out[i], in[(i & ~15) + (15 - (i & 15))]);
}

VTEST(a_lane_with_no_lane_to_read_keeps_what_it_had) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<int32_t> in(kN);
  for (int i = 0; i < kN; ++i) in[i] = 1000 + i;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  run(o, "kept", mem, {pin, pout, static_cast<uint64_t>(kN)});

  // Shifted by one within the row, and the first lane of each row has no lane
  // below it to read, so it keeps the value the kernel gave it.
  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = (i & 15) == 0 ? -1 : in[i - 1];
    if (out[i] != want)
      throw vtest::Failure("kept[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(reading_across_the_whole_wave_is_refused_rather_than_guessed) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const std::vector<int32_t> in(kN, 1);
  const uint64_t pin = upload(mem, in), pout = mem.alloc(kN * 4);
  // The shifts that cross rows are spelled out in the listing, since the
  // decoder reads them, but which way they carry is the one thing the
  // reduction does not settle, so running one says so.
  std::string what;
  try {
    run(o, "across", mem, {pin, pout, static_cast<uint64_t>(kN)});
  } catch (const std::exception& e) {
    what = e.what();
  }
  VCHECK_CONTAINS(what, "does not model");
}

VTEST_MAIN
