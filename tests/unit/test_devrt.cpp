// The device runtime's shared half (vgpu/exec/devrt.hpp): the order a grid's
// children run in, the pending-launch count and what happens when it is full,
// the memory operations, the stream and event handles, and the entry-point
// names -- with fakes for what the engines supply, since the engines have their
// own tests (test_dynpar for the PTX interpreter, e2e_sass_archs for both).
// nvidia/docs/sass.md, "Device runtime", has the card's measurements these
// follow.
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vgpu/exec/devrt.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
namespace devrt = vgpu::exec::devrt;

namespace {

using Child = devrt::Child<int>;
using Launches = devrt::Launches<int>;

// A run function: the "kernel" is a number, and a grid's body may launch more.
struct Fake {
  std::vector<int> order;
  std::function<void(Child&, Launches&)> body;
  vgpu::exec::LaunchStats stats;
  void add_stats(const vgpu::exec::LaunchStats&) {}
  void operator()(Child& c, Launches& out) {
    order.push_back(c.kernel);
    if (body) body(c, out);
  }
};

Child grid(int kernel, bool tail = false) {
  Child c;
  c.kernel = kernel;
  c.has_kernel = true;
  c.tail = tail;
  return c;
}

// Queues `kernel` as the block's launch, as the engines do (limit as the launch saw it).
int launch(Launches& dl, MemoryManager& mem, Fake& run, Child c, int64_t limit = 2048) {
  int err = -1;
  devrt::enqueue(dl, mem, run, std::move(c), 0, limit, &err);
  return err;
}

}  // namespace

// Everything that is not a tail launch runs first, in the order issued, then the tail launches in order.
VTEST(tail_launches_run_after_the_grids_other_children) {
  MemoryManager mem{1 << 20};
  Launches dl;
  Fake run;
  VCHECK_EQ(launch(dl, mem, run, grid(1, true)), 0);
  VCHECK_EQ(launch(dl, mem, run, grid(2)), 0);
  VCHECK_EQ(launch(dl, mem, run, grid(3)), 0);
  VCHECK_EQ(launch(dl, mem, run, grid(4, true)), 0);
  devrt::complete_children(dl, mem, run);
  VCHECK((run.order == std::vector<int>{2, 3, 1, 4}));
}

// A grid is complete when what it launched is: a child's own children, and its tail launches, run before the
// grid's next sibling and before the parent's tail launches.
VTEST(a_grid_completes_after_its_children_and_their_tails) {
  MemoryManager mem{1 << 20};
  Launches dl;
  Fake run;
  run.body = [&](Child& c, Launches& out) {
    if (c.kernel == 2) {
      launch(out, mem, run, grid(20));
      launch(out, mem, run, grid(21, true));
      launch(out, mem, run, grid(22));
    }
  };
  launch(dl, mem, run, grid(5, true));
  launch(dl, mem, run, grid(2));
  launch(dl, mem, run, grid(3));
  devrt::complete_children(dl, mem, run);
  VCHECK((run.order == std::vector<int>{2, 20, 22, 21, 3, 5}));
  VCHECK_EQ(dl.tree->pending.load(), int64_t{0});   // every grid completed
}

// Launches are ordered by their block, then by their order within it, whichever order they arrived in.
VTEST(children_run_in_block_then_issue_order) {
  MemoryManager mem{1 << 20};
  Launches dl;
  Fake run;
  int err = 0;
  devrt::enqueue(dl, mem, run, grid(10), /*block=*/1, 2048, &err);
  devrt::enqueue(dl, mem, run, grid(11), /*block=*/0, 2048, &err);
  devrt::enqueue(dl, mem, run, grid(12), /*block=*/1, 2048, &err);
  devrt::enqueue(dl, mem, run, grid(13), /*block=*/0, 2048, &err);
  devrt::complete_children(dl, mem, run);
  VCHECK((run.order == std::vector<int>{11, 13, 10, 12}));
}

// The pending count: a chain of grids that each launch the next is refused at the limit (a limit under 32 is 32),
// where nothing has completed to make room.
VTEST(a_chain_stops_at_the_pending_limit_and_a_limit_under_32_is_32) {
  const struct {
    int64_t limit;
    int grids;   // grids that ran: the host's, and one below it per launch the limit allows
  } cases[] = {{64, 65}, {10, 33}, {0, 33}, {2048, 2049}};
  for (const auto& tc : cases) {
    MemoryManager mem{1 << 20};
    Launches dl;
    Fake run;
    int refused = 0, depth = 0;
    run.body = [&](Child&, Launches& out) {
      ++depth;
      const int err = launch(out, mem, run, grid(depth), tc.limit);
      if (err) refused = err;
    };
    // The host's grid is the first (not counted); its launch is the second.
    depth = 1;
    VCHECK_EQ(launch(dl, mem, run, grid(1), tc.limit), 0);
    devrt::complete_children(dl, mem, run);
    VCHECK_EQ(depth, tc.grids);
    VCHECK_EQ(refused, devrt::kLaunchPendingCountExceeded);
  }
}

// When the count is full but earlier launches are only queued, they run to completion and the launch goes
// through, as the card's would have finished while the parent went on; tail launches wait for the grid.
VTEST(a_full_pending_count_runs_the_queued_grids) {
  MemoryManager mem{1 << 20};
  Launches dl;
  Fake run;
  const int64_t limit = 32;
  VCHECK_EQ(launch(dl, mem, run, grid(1000, true), limit), 0);   // a tail launch: counts, and cannot run yet
  for (int i = 0; i < 31; ++i) VCHECK_EQ(launch(dl, mem, run, grid(i), limit), 0);
  VCHECK(run.order.empty());
  VCHECK_EQ(launch(dl, mem, run, grid(31), limit), 0);   // the 33rd: the 31 queued before it run first
  VCHECK_EQ(run.order.size(), size_t{31});
  for (int i = 0; i < 31; ++i) VCHECK_EQ(run.order[i], i);
  devrt::complete_children(dl, mem, run);
  VCHECK_EQ(run.order.size(), size_t{33});
  VCHECK_EQ(run.order[31], 31);
  VCHECK_EQ(run.order[32], 1000);   // the tail launch last
  // Only tail launches queued: nothing can run to make room.
  Launches full;
  Fake none;
  for (int i = 0; i < 32; ++i) VCHECK_EQ(launch(full, mem, none, grid(i, true), limit), 0);
  VCHECK_EQ(launch(full, mem, none, grid(99), limit), devrt::kLaunchPendingCountExceeded);
  VCHECK(none.order.empty());
}

// Copies and fills are not grids: they are not counted against the limit, and they run where they were queued.
VTEST(memory_operations_run_in_order_and_are_not_counted) {
  MemoryManager mem{1 << 20};
  const uint64_t a = mem.alloc(256), b = mem.alloc(256);
  uint8_t fill[256];
  std::memset(fill, 0xEE, sizeof fill);
  mem.write(b, fill, sizeof fill);
  for (int i = 0; i < 256; ++i) fill[i] = static_cast<uint8_t>(i);
  mem.write(a, fill, sizeof fill);
  Launches dl;
  Fake run;
  const auto op = [&](devrt::MemOp m) {
    Child c;
    c.op = std::make_shared<devrt::MemOp>(m);
    return c;
  };
  devrt::MemOp copy;   // a 2D copy: rows of 4 bytes, pitch 16 to pitch 8, three rows
  copy.dst = b;
  copy.src = a;
  copy.dpitch = 8;
  copy.spitch = 16;
  copy.width = 4;
  copy.height = 3;
  devrt::MemOp set;   // then a fill of the middle row
  set.kind = devrt::MemOp::Kind::Set;
  set.dst = b + 8;
  set.value = 0x5A;
  set.width = 4;
  VCHECK_EQ(launch(dl, mem, run, op(copy), 32), 0);
  VCHECK_EQ(launch(dl, mem, run, op(set), 32), 0);
  VCHECK_EQ(dl.tree->pending.load(), int64_t{0});
  devrt::complete_children(dl, mem, run);
  uint8_t got[32];
  mem.read(b, got, sizeof got);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c) VCHECK_EQ(got[r * 8 + c], r == 1 ? 0x5A : uint8_t(r * 16 + c));
  VCHECK_EQ(got[4], 0xEE);   // between the rows: untouched
  VCHECK_EQ(got[31], 0xEE);

  // A 3D copy: two slices of two rows, the slice stride the pitch times the rows allocated.
  devrt::MemOp cube;
  cube.dst = b + 64;
  cube.src = a;
  cube.dpitch = 16;
  cube.spitch = 16;
  cube.dslice = 16 * 4;
  cube.sslice = 16 * 4;
  cube.width = 4;
  cube.height = 2;
  cube.depth = 2;
  Launches dl2;
  launch(dl2, mem, run, op(cube));
  devrt::complete_children(dl2, mem, run);
  uint8_t out[256];
  mem.read(b, out, sizeof out);
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int c = 0; c < 4; ++c) VCHECK_EQ(out[64 + z * 64 + y * 16 + c], uint8_t(z * 64 + y * 16 + c));
}

// Streams: 0 to 4 are the default, legacy, per-thread, tail and fire-and-forget streams; created ones are
// valid for good; nothing else is. Events and flags likewise.
VTEST(stream_and_event_handles) {
  using devrt::StreamKind;
  VCHECK(devrt::stream_kind(0) == StreamKind::Default);
  VCHECK(devrt::stream_kind(1) == StreamKind::Legacy);
  VCHECK(devrt::stream_kind(2) == StreamKind::PerThread);
  VCHECK(devrt::stream_kind(3) == StreamKind::Tail);
  VCHECK(devrt::stream_kind(4) == StreamKind::FireAndForget);
  for (uint64_t bad : {uint64_t{5}, uint64_t{6}, uint64_t{8}, uint64_t{0x10}, uint64_t{0x100}, uint64_t{0x1234},
                       uint64_t{0x0100000000000000}, ~uint64_t{0}})
    VCHECK(devrt::stream_kind(bad) == StreamKind::Invalid);
  const uint64_t s1 = devrt::new_stream(), s2 = devrt::new_stream(), e1 = devrt::new_event();
  VCHECK(s1 != s2);
  VCHECK(devrt::stream_kind(s1) == StreamKind::Named);
  VCHECK(devrt::stream_kind(s2) == StreamKind::Named);
  VCHECK(devrt::stream_kind(e1) == StreamKind::Invalid);   // an event is not a stream
  VCHECK(devrt::valid_event(e1));
  VCHECK(!devrt::valid_event(s1));
  VCHECK(!devrt::valid_event(0x1234));
  VCHECK(!devrt::valid_event(0));
  for (uint32_t f = 0; f < 16; ++f) VCHECK_EQ(devrt::event_flags_ok(f), f == 2 || f == 3 || f == 10 || f == 11);
  VCHECK(devrt::stream_flags_ok(0) && devrt::stream_flags_ok(1));
  VCHECK(!devrt::stream_flags_ok(2) && !devrt::stream_flags_ok(7));
}

// The configuration a child may have: no zero dimension, within the device's, shared memory within its opt-in
// maximum, and the kernel's own launch bounds.
VTEST(child_launch_configurations) {
  const DeviceProfile p = load_gpu("nvidia/rtx3060");
  const std::array<uint32_t, 3> none{0, 0, 0}, one{1, 1, 1};
  VCHECK(devrt::config_ok(one, one, 0, p, 0, none));
  VCHECK(devrt::config_ok({1, 1, 65}, one, 0, p, 0, none));    // a grid z of 65 is fine
  VCHECK(!devrt::config_ok({0, 1, 1}, one, 0, p, 0, none));
  VCHECK(!devrt::config_ok(one, {0, 1, 1}, 0, p, 0, none));
  VCHECK(!devrt::config_ok(one, {1025, 1, 1}, 0, p, 0, none));
  VCHECK(!devrt::config_ok(one, {1, 1, 65}, 0, p, 0, none));   // block z is at most 64
  VCHECK(!devrt::config_ok({1, 65536, 1}, one, 0, p, 0, none));
  VCHECK(!devrt::config_ok(one, one, 1 << 20, p, 0, none));    // far too much shared memory
  VCHECK(devrt::config_ok(one, {64, 1, 1}, 0, p, 64, none));
  VCHECK(!devrt::config_ok(one, {65, 1, 1}, 0, p, 64, none));  // past __launch_bounds__(64)
  VCHECK(!devrt::config_ok(one, {32, 1, 1}, 0, p, 0, {64, 1, 1}));   // .reqntid
}

// The names: CDP2's and CDP1's for each call, one function.
VTEST(entry_point_names) {
  using devrt::Fn;
  VCHECK(devrt::lookup("__cudaCDP2MemcpyAsync") == Fn::MemcpyAsync);
  VCHECK(devrt::lookup("cudaMemcpyAsync") == Fn::MemcpyAsync);
  VCHECK(devrt::lookup("__cudaCDP2MemcpyAsync_ptsz") == Fn::MemcpyAsync);
  VCHECK(devrt::lookup("__cudaCDP2GetParameterBufferV2") == Fn::GetParameterBufferV2);
  VCHECK(devrt::lookup("cudaLaunchDeviceV2_ptsz") == Fn::LaunchDeviceV2);
  VCHECK(devrt::lookup("cudaDeviceSynchronize") == Fn::DeviceSynchronize);
  VCHECK(devrt::lookup("__cudaDeviceSynchronizeDeprecationAvoidance") == Fn::DeviceSynchronize);
  VCHECK(devrt::lookup("vprintf") == Fn::None);
  VCHECK(devrt::lookup("malloc") == Fn::None);   // the device heap's own, not the device runtime's
  VCHECK(devrt::lookup("__cuda_syscall_cnpv2GetLastError") == Fn::None);
  for (const std::string& n : devrt::names()) VCHECK(devrt::lookup(n) != Fn::None);
}

// What cudaFuncGetAttributes writes: seven fields where the toolkit's struct has them.
VTEST(function_attributes_layout) {
  devrt::FuncAttrs a;
  a.shared = 4096;
  a.constant = 12;
  a.local = 256;
  a.max_threads = 1024;
  a.regs = 24;
  a.ptx = 86;
  a.binary = 86;
  uint8_t out[40];
  devrt::put_func_attrs(a, out);
  uint64_t q[3];
  std::memcpy(q, out, 24);
  VCHECK_EQ(q[0], uint64_t{4096});
  VCHECK_EQ(q[1], uint64_t{12});
  VCHECK_EQ(q[2], uint64_t{256});
  int32_t i[4];
  std::memcpy(i, out + 24, 16);
  VCHECK_EQ(i[0], 1024);
  VCHECK_EQ(i[1], 24);
  VCHECK_EQ(i[2], 86);
  VCHECK_EQ(i[3], 86);
}

VTEST_MAIN
