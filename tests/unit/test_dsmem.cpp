// Distributed shared memory (sm_90): one block of a thread-block cluster
// reading, writing and signalling another's shared memory -- mapa and
// getctarank, ld/st/atom on .shared::cluster, remote mbarrier arrivals,
// st.async and red.async, and bulk copies that multicast to several blocks
// or go from one block's shared memory to another's.
//
// The PTX here is written the way CUDA and CUTLASS emit it:
// cluster.map_shared_rank() is mapa.u64 on a generic address, CUTLASS's
// ClusterBarrier::arrive(cta_id) is mapa.shared::cluster.u32 and a remote
// mbarrier.arrive, and its multicast TMA is the ctaMask form of
// cp.async.bulk. nvidia/tests/e2e/run_cutlass_hopper.sh runs CUTLASS's own
// multicast test on top of these.
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

const char* kHeader = ".version 8.3\n.target sm_90a\n.address_size 64\n";

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

void run(const std::string& body_ptx, const LaunchConfig& cfg,
         std::vector<std::vector<uint8_t>> args, MemoryManager& mem) {
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(std::string(kHeader) + body_ptx);
  exec::launch(m.entries[0], cfg, args, mem, prof);
}

LaunchConfig clusters(uint32_t grid, uint32_t cluster, uint32_t block) {
  LaunchConfig c;
  c.grid = {grid, 1, 1};
  c.block = {block, 1, 1};
  c.cluster = {cluster, 1, 1};
  return c;
}

std::string parse_error(const std::string& body) {
  try {
    ptx::parse(std::string(kHeader) + ".visible .entry k()\n{\n    .reg .b32 %r<4>;\n"
               "    .reg .b64 %rd<4>;\n    .reg .pred %p<2>;\n" + body + "\n    ret;\n}\n");
  } catch (const Error& e) {
    return e.message();
  }
  return "";
}

}  // namespace

// cluster.map_shared_rank: every block reads its neighbour's array through a
// generic pointer mapa made. The mapped pointer belongs to the neighbour
// (getctarank), is shared::cluster but not this block's shared memory
// (isspacep), and mapping to one's own rank gives the pointer back.
VTEST(mapa_generic_reads_the_neighbours_shared_memory) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(4 * 32 * 3 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    .shared .align 4 .b32 buf[32];
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %cluster_ctarank;
    mov.u32 %r2, %tid.x;
    mul.lo.u32 %r3, %r1, 100;
    add.u32 %r3, %r3, %r2;
    mov.u64 %rd2, buf;
    cvta.shared.u64 %rd2, %rd2;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd2, %rd3;
    st.u32 [%rd4], %r3;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    add.u32 %r4, %r1, 1;
    rem.u32 %r4, %r4, 4;
    mapa.u64 %rd5, %rd4, %r4;
    ld.u32 %r5, [%rd5];
    getctarank.u64 %r6, %rd5;
    isspacep.shared %p1, %rd5;
    isspacep.shared::cluster %p2, %rd5;
    mapa.u64 %rd6, %rd4, %r1;
    setp.eq.u64 %p3, %rd6, %rd4;
    selp.u32 %r7, 1, 0, %p1;
    selp.u32 %r8, 2, 0, %p2;
    selp.u32 %r9, 4, 0, %p3;
    or.b32 %r7, %r7, %r8;
    or.b32 %r7, %r7, %r9;
    mad.lo.u32 %r10, %r1, 32, %r2;
    mul.lo.u32 %r10, %r10, 12;
    cvt.u64.u32 %rd7, %r10;
    add.u64 %rd8, %rd1, %rd7;
    st.global.u32 [%rd8], %r5;
    st.global.u32 [%rd8+4], %r6;
    st.global.u32 [%rd8+8], %r7;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    ret;
}
)", clusters(4, 4, 32), {arg_u64(out)}, mem);
  for (uint32_t rank = 0; rank < 4; ++rank)
    for (uint32_t t = 0; t < 32; ++t) {
      const uint64_t at = out + (rank * 32 + t) * 12;
      const uint32_t next = (rank + 1) % 4;
      VCHECK_EQ(mem.load_scalar(at, 4), uint64_t{next * 100 + t});
      VCHECK_EQ(mem.load_scalar(at + 4, 4), uint64_t{next});
      VCHECK_EQ(mem.load_scalar(at + 8, 4), uint64_t{2 | 4});
    }
}

// A counter in rank 0 that every thread of the cluster adds to, through a
// 32-bit .shared::cluster address: atom and red reach another block, and
// each cluster of the grid has its own.
VTEST(atomics_on_shared_cluster_reach_rank_0) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(2 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 4 .b32 counter;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, counter;
    mov.u32 %r9, 0;
    mapa.shared::cluster.u32 %r2, %r1, %r9;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    atom.shared::cluster.add.u32 %r3, [%r2], 1;
    red.relaxed.cluster.shared::cluster.add.u32 [%r2], 2;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    mov.u32 %r4, %cluster_ctarank;
    mov.u32 %r5, %tid.x;
    or.b32 %r6, %r4, %r5;
    setp.ne.u32 %p1, %r6, 0;
    @%p1 bra END;
    ld.shared::cluster.u32 %r7, [%r2];
    mov.u32 %r8, %clusterid.x;
    mul.wide.u32 %rd2, %r8, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r7;
END:
    ret;
}
)", clusters(4, 2, 64), {arg_u64(out)}, mem);
  VCHECK_EQ(mem.load_scalar(out, 4), uint64_t{3 * 128});
  VCHECK_EQ(mem.load_scalar(out + 4, 4), uint64_t{3 * 128});
}

// CUTLASS's ClusterBarrier::arrive(cta_id): the other blocks publish a value
// and arrive on rank 0's barrier; rank 0 waits on it and then sees all three.
VTEST(remote_mbarrier_arrive_releases_the_owner) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(4 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<6>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, bar;
    mov.u32 %r2, %cluster_ctarank;
    mov.u32 %r3, %tid.x;
    setp.eq.u32 %p1, %r2, 0;
    setp.eq.u32 %p2, %r3, 0;
    and.pred %p3, %p1, %p2;
    @%p3 mbarrier.init.shared::cta.b64 [%r1], 3;
    fence.mbarrier_init.release.cluster;
    barrier.cluster.arrive.release.aligned;
    barrier.cluster.wait.acquire.aligned;
    @%p1 bra WAITER;
    @!%p2 bra END;
    mul.lo.u32 %r4, %r2, 11;
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r4;
    mov.u32 %r9, 0;
    mapa.shared::cluster.u32 %r5, %r1, %r9;
    mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r5];
    bra END;
WAITER:
    mbarrier.try_wait.parity.shared::cta.b64 %p4, [%r1], 0;
    @!%p4 bra WAITER;
    @!%p2 bra END;
    ld.global.u32 %r6, [%rd1+4];
    ld.global.u32 %r7, [%rd1+8];
    ld.global.u32 %r8, [%rd1+12];
    add.u32 %r6, %r6, %r7;
    add.u32 %r6, %r6, %r8;
    st.global.u32 [%rd1], %r6;
END:
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    ret;
}
)", clusters(4, 4, 32), {arg_u64(out)}, mem);
  VCHECK_EQ(mem.load_scalar(out, 4), uint64_t{11 + 22 + 33});
}

// st.async and red.async: rank 1 writes 16 bytes and adds to a word in rank
// 0's shared memory, and the bytes complete on rank 0's barrier, which
// expects all 20.
VTEST(st_async_and_red_async_complete_on_the_owners_barrier) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(5 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<6>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<6>;
    .shared .align 16 .b32 buf[4];
    .shared .align 8 .b64 bar;
    .shared .align 4 .b32 acc;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, bar;
    mov.u32 %r2, %cluster_ctarank;
    mov.u32 %r3, %tid.x;
    setp.eq.u32 %p1, %r2, 0;
    setp.eq.u32 %p2, %r3, 0;
    and.pred %p3, %p1, %p2;
    @%p3 mbarrier.init.shared::cta.b64 [%r1], 1;
    @%p3 st.shared.u32 [acc], 7;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    @%p3 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r1], 20;
    @%p1 bra WAITER;
    @!%p2 bra END;
    mov.u32 %r9, 0;
    mov.u32 %r4, buf;
    mapa.shared::cluster.u32 %r5, %r4, %r9;
    mapa.shared::cluster.u32 %r6, %r1, %r9;
    mov.u32 %r4, acc;
    mapa.shared::cluster.u32 %r7, %r4, %r9;
    mov.u32 %r10, 1;
    mov.u32 %r11, 2;
    mov.u32 %r12, 3;
    mov.u32 %r13, 4;
    st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%r5], {%r10, %r11, %r12, %r13}, [%r6];
    red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.add.u32 [%r7], 5, [%r6];
    bra END;
WAITER:
    mbarrier.try_wait.parity.shared::cta.b64 %p4, [%r1], 0;
    @!%p4 bra WAITER;
    @!%p2 bra END;
    ld.shared.v4.u32 {%r10, %r11, %r12, %r13}, [buf];
    ld.shared.u32 %r14, [acc];
    st.global.v4.u32 [%rd1], {%r10, %r11, %r12, %r13};
    st.global.u32 [%rd1+16], %r14;
END:
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    ret;
}
)", clusters(2, 2, 32), {arg_u64(out)}, mem);
  for (uint32_t i = 0; i < 4; ++i) VCHECK_EQ(mem.load_scalar(out + i * 4, 4), uint64_t{i + 1});
  VCHECK_EQ(mem.load_scalar(out + 16, 4), uint64_t{12});
}

// Multicast TMA: rank 0 issues one bulk load with ctaMask 0b11 and the same
// 256 bytes land in both blocks, each completing its own barrier.
VTEST(multicast_bulk_copy_fills_every_masked_block) {
  MemoryManager mem{1 << 20};
  const uint64_t src = mem.alloc(256), out = mem.alloc(2 * 256);
  for (uint32_t i = 0; i < 64; ++i) mem.store_scalar(src + i * 4, 4, 0xA0000000u + i);
  run(R"(
.visible .entry k(.param .u64 src, .param .u64 out)
{
    .reg .pred %p<6>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    .shared .align 128 .b8 buf[256];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [out];
    mov.u32 %r1, bar;
    mov.u32 %r2, %cluster_ctarank;
    mov.u32 %r3, %tid.x;
    setp.eq.u32 %p2, %r3, 0;
    @%p2 mbarrier.init.shared::cta.b64 [%r1], 1;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r1], 256;
    setp.eq.u32 %p1, %r2, 0;
    and.pred %p3, %p1, %p2;
    mov.u32 %r4, buf;
    mov.b16 %rs1, 3;
    @%p3 cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%r4], [%rd1], 256, [%r1], %rs1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p4, [%r1], 0;
    @!%p4 bra WAIT;
    mul.lo.u32 %r5, %r3, 8;
    add.u32 %r6, %r4, %r5;
    ld.shared.u64 %rd3, [%r6];
    mad.lo.u32 %r7, %r2, 256, %r5;
    cvt.u64.u32 %rd4, %r7;
    add.u64 %rd5, %rd2, %rd4;
    st.global.u64 [%rd5], %rd3;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    ret;
}
)", clusters(2, 2, 32), {arg_u64(src), arg_u64(out)}, mem);
  for (uint32_t r = 0; r < 2; ++r)
    for (uint32_t i = 0; i < 64; ++i)
      VCHECK_EQ(mem.load_scalar(out + r * 256 + i * 4, 4), uint64_t{0xA0000000u + i});
}

// cp.async.bulk.shared::cluster.shared::cta: rank 1 copies its tile into
// rank 0, completing on rank 0's barrier.
VTEST(bulk_copy_between_the_shared_memories_of_two_blocks) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(64);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<6>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 16 .b32 src[16];
    .shared .align 16 .b32 dst[16];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, bar;
    mov.u32 %r2, %cluster_ctarank;
    mov.u32 %r3, %tid.x;
    setp.lt.u32 %p5, %r3, 16;
    mad.lo.u32 %r4, %r2, 1000, %r3;
    mov.u32 %r5, src;
    mad.lo.u32 %r6, %r3, 4, %r5;
    @%p5 st.shared.u32 [%r6], %r4;
    setp.eq.u32 %p1, %r2, 0;
    setp.eq.u32 %p2, %r3, 0;
    and.pred %p3, %p1, %p2;
    @%p3 mbarrier.init.shared::cta.b64 [%r1], 1;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    @%p3 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r1], 64;
    @%p1 bra WAITER;
    @!%p2 bra END;
    mov.u32 %r9, 0;
    mov.u32 %r7, dst;
    mapa.shared::cluster.u32 %r7, %r7, %r9;
    mapa.shared::cluster.u32 %r8, %r1, %r9;
    cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%r7], [%r5], 64, [%r8];
    bra END;
WAITER:
    mbarrier.try_wait.parity.shared::cta.b64 %p4, [%r1], 0;
    @!%p4 bra WAITER;
    @!%p5 bra END;
    mov.u32 %r7, dst;
    mad.lo.u32 %r6, %r3, 4, %r7;
    ld.shared.u32 %r10, [%r6];
    mul.wide.u32 %rd2, %r3, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r10;
END:
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    ret;
}
)", clusters(2, 2, 32), {arg_u64(out)}, mem);
  for (uint32_t i = 0; i < 16; ++i) VCHECK_EQ(mem.load_scalar(out + i * 4, 4), uint64_t{1000 + i});
}

// cp.reduce.async.bulk into another block's shared memory: rank 1 adds its
// four words into rank 0's, and the 16 bytes complete on rank 0's barrier.
VTEST(bulk_reduce_into_another_blocks_shared_memory) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(16);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<6>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 16 .b32 buf[4];
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, bar;
    mov.u32 %r2, %cluster_ctarank;
    mov.u32 %r3, %tid.x;
    setp.lt.u32 %p5, %r3, 4;
    mov.u32 %r5, buf;
    mad.lo.u32 %r6, %r3, 4, %r5;
    mul.lo.u32 %r4, %r2, 100;
    add.u32 %r4, %r4, %r3;
    add.u32 %r4, %r4, 1;
    @%p5 st.shared.u32 [%r6], %r4;
    setp.eq.u32 %p1, %r2, 0;
    setp.eq.u32 %p2, %r3, 0;
    and.pred %p3, %p1, %p2;
    @%p3 mbarrier.init.shared::cta.b64 [%r1], 1;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    @%p3 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r1], 16;
    @%p1 bra WAITER;
    @!%p2 bra END;
    mov.u32 %r9, 0;
    mapa.shared::cluster.u32 %r7, %r5, %r9;
    mapa.shared::cluster.u32 %r8, %r1, %r9;
    fence.proxy.async.shared::cta;
    cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32 [%r7], [%r5], 16, [%r8];
    bra END;
WAITER:
    mbarrier.try_wait.parity.shared::cta.b64 %p4, [%r1], 0;
    @!%p4 bra WAITER;
    @!%p5 bra END;
    ld.shared.u32 %r10, [%r6];
    mul.wide.u32 %rd2, %r3, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r10;
END:
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    ret;
}
)", clusters(2, 2, 32), {arg_u64(out)}, mem);
  // rank 0 held t + 1, rank 1 sent 100 + t + 1.
  for (uint32_t t = 0; t < 4; ++t) VCHECK_EQ(mem.load_scalar(out + t * 4, 4), uint64_t{2 * t + 102});
}

// One instruction, lanes naming different blocks' barriers: in an 8-block
// cluster, lane i of block 0's first warp arrives on block i's barrier -- the
// shape of CUTLASS's cluster pipelines releasing a stage to every block that
// multicast into it. Each barrier expects two arrivals: block 0 sends two
// rounds, and every block must see exactly its own, no more.
VTEST(lanes_of_one_arrive_reach_different_blocks_barriers) {
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(8 * 4);
  run(R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<6>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 8 .b64 bar;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, bar;
    mov.u32 %r2, %cluster_ctarank;
    mov.u32 %r3, %tid.x;
    setp.eq.u32 %p2, %r3, 0;
    @%p2 mbarrier.init.shared::cta.b64 [%r1], 2;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    setp.ne.u32 %p1, %r2, 0;
    setp.ge.u32 %p3, %r3, 8;
    or.pred %p4, %p1, %p3;
    @%p4 bra WAIT;
    mapa.shared::cluster.u32 %r4, %r1, %r3;
    mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r4];
    mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r4];
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p5, [%r1], 0;
    @!%p5 bra WAIT;
    @!%p2 bra END;
    mbarrier.pending_count.b64 %r5, [%r1];
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    add.u32 %r6, %r5, 100;
    st.global.u32 [%rd3], %r6;
END:
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    ret;
}
)", clusters(8, 8, 32), {arg_u64(out)}, mem);
  // Every block's barrier completed its phase with nothing left over.
  for (uint32_t r = 0; r < 8; ++r) VCHECK_EQ(mem.load_scalar(out + r * 4, 4), uint64_t{100 + 2});
}

// A block that exits takes its shared memory with it. Reading it afterwards
// is the bug cluster.sync() at the end of a kernel exists to prevent, and it
// is reported rather than answered from memory that is gone.
VTEST(reading_an_exited_blocks_shared_memory_is_reported) {
  MemoryManager mem{1 << 20};
  auto err = VCAPTURE(Error, run(R"(
.visible .entry k()
{
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .shared .align 4 .b32 x;
    mov.u32 %r1, %cluster_ctarank;
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra END;
    mov.u32 %r2, 0;
SPIN:
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p2, %r2, 2000;
    @%p2 bra SPIN;
    mov.u32 %r3, x;
    mov.u32 %r4, 1;
    mapa.shared::cluster.u32 %r5, %r3, %r4;
    ld.shared::cluster.u32 %r6, [%r5];
END:
    ret;
}
)", clusters(2, 2, 32), {}, mem));
  VCHECK_CONTAINS(err.message(), "has exited");
}

// A rank the cluster does not have is an error at mapa, not an address that
// fails later; and a launch with no cluster is a cluster of one.
VTEST(mapa_checks_the_rank_against_the_cluster) {
  MemoryManager mem{1 << 20};
  const std::string ptx = R"(
.visible .entry k(.param .u32 rank)
{
    .reg .b32 %r<8>;
    .shared .align 4 .b32 x;
    ld.param.u32 %r1, [rank];
    mov.u32 %r2, x;
    mapa.shared::cluster.u32 %r3, %r2, %r1;
    st.shared::cluster.u32 [%r3], 1;
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;
    ret;
}
)";
  std::vector<uint8_t> two(4, 0), zero(4, 0);
  two[0] = 2;
  auto err = VCAPTURE(Error, run(ptx, clusters(2, 2, 32), {two}, mem));
  VCHECK_CONTAINS(err.message(), "mapa names block rank 2 of a cluster of 2 blocks");
  run(ptx, LaunchConfig{}, {zero}, mem);
}

// What the ISA does not allow on another block's barrier is refused by name.
VTEST(remote_mbarrier_forms_the_isa_does_not_define_are_refused) {
  VCHECK_CONTAINS(parse_error("mbarrier.try_wait.parity.shared::cluster.b64 %p1, [%r1], 0;"),
                  "arrive, arrive_drop, expect_tx and complete_tx only");
  VCHECK_CONTAINS(parse_error("mbarrier.arrive.shared::cluster.b64 %rd1, [%r1];"),
                  "destination must be the sink");
  VCHECK_CONTAINS(parse_error("st.async.shared::cluster.b32 [%r1], %r2, [%r3];"),
                  ".mbarrier::complete_tx::bytes");
}

VTEST_MAIN
