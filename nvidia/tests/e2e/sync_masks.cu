// Member masks of the *.sync warp instructions (vote, ballot, match, redux,
// shfl, bar.warp.sync) as an RTX 3060 (sm_86, CUDA 13.0's ptxas) runs them,
// on SASS and on its PTX alike. The expected values below are what the card
// printed for this program.
//
// ptxas compiles a mask it can see through -- a constant -- to the bare
// instruction, which ignores the mask. A mask it cannot (a lane-dependent
// value, a kernel argument) goes through code that first checks whether the
// lanes agree:
//   - one mask in every lane: vote, shfl and match ignore it too (a mask of
//     0xffff, run by all 32 lanes, is the 32-lane ballot); redux.sync's code
//     waits with WARPSYNC.EXCLUSIVE, which traps a lane the mask leaves out;
//   - different masks (shfl: masks that disagree about the lanes they share):
//     the instruction runs once for each distinct mask, over the lanes that
//     named it, after a WARPSYNC, which traps a lane its own mask leaves
//     out: "an illegal instruction was encountered" (715).
// The trap is the CUDA Samples' and HeCBench's bscan benchmark (a mask of
// (1 << lane) - 1: lane 0's is empty), which prints FAIL on a card for it.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

enum Case {
  BallotUniformRuntime, BallotOwnBit, BallotPrefixInclusive, BallotTwoGroups, AnyTwoGroups, AllTwoGroups,
  MatchTwoGroups, MatchUniformRuntime, BallotConstant, ReduxConstant, ReduxTwoGroups, ShflPrefixOwnLane,
  ShflTwoGroups, ShflTwoGroupsCross, ShflUniformRuntime, ShflFromInactive, ShflDownEdge,
  TrapBallotExclusivePrefix, TrapAnyExclusivePrefix, TrapShflExclusivePrefix, TrapSyncwarpExclusivePrefix,
  TrapReduxUniformRuntime, TrapBallotLaneOutOfOwnGroup, TrapMatchExclusivePrefix, TrapAllLaneOutOfOwnGroup,
  TrapReduxExclusivePrefix, NumCases
};

__global__ void k(unsigned* out, int which, unsigned pm) {
  const unsigned lane = threadIdx.x & 31;
  const bool p = (lane % 3) == 0 || lane > 28;
  const unsigned val = lane * 10 + 1;
  const unsigned halves = lane < 16 ? 0xffffu : 0xffff0000u;
  unsigned r = 0xdead;
  switch (which) {
    case BallotUniformRuntime: r = __ballot_sync(pm, p); break;
    case BallotOwnBit: r = __ballot_sync(1u << lane, p); break;
    case BallotPrefixInclusive: r = __ballot_sync((2u << lane) - 1, p); break;
    case BallotTwoGroups: r = __ballot_sync(halves, p); break;
    case AnyTwoGroups: r = __any_sync(halves, lane == 20); break;
    case AllTwoGroups: r = __all_sync(halves, lane != 20); break;
    case MatchTwoGroups: r = __match_any_sync(halves, lane & 3); break;
    case MatchUniformRuntime: r = __match_any_sync(pm, lane & 3); break;
    case BallotConstant: r = __ballot_sync(0xffffu, p); break;
    case ReduxConstant: r = __reduce_add_sync(0xffffu, lane); break;
    case ReduxTwoGroups: r = __reduce_add_sync(halves, lane); break;
    case ShflPrefixOwnLane: r = __shfl_sync((2u << lane) - 1, lane * 10, lane); break;
    case ShflTwoGroups: r = __shfl_sync(halves, lane * 10, lane < 16 ? 3 : 19); break;
    case ShflTwoGroupsCross: r = __shfl_sync(halves, lane * 10, 17); break;
    case ShflUniformRuntime: r = __shfl_sync(pm, lane * 10, 1); break;
    case ShflFromInactive:
      if (lane < 16) r = __shfl_sync(0xffffu, val, 17);   // lane 17 is not running this
      break;
    case ShflDownEdge:
      if (lane < 16) r = __shfl_down_sync(0xffffu, val, 1);   // lane 15 reads lane 16, which is not
      break;
    case TrapBallotExclusivePrefix: r = __ballot_sync((1u << lane) - 1, p); break;
    case TrapAnyExclusivePrefix: r = __any_sync((1u << lane) - 1, p); break;
    case TrapShflExclusivePrefix: r = __shfl_sync((1u << lane) - 1, val, 0); break;
    case TrapSyncwarpExclusivePrefix: __syncwarp((1u << lane) - 1); r = 77; break;
    case TrapReduxUniformRuntime: r = __reduce_add_sync(pm, lane); break;   // lanes 16-31 are outside 0xffff
    case TrapBallotLaneOutOfOwnGroup: r = __ballot_sync(lane < 16 ? 0xffffu : 0xfffe0000u, p); break;
    case TrapMatchExclusivePrefix: r = __match_any_sync((1u << lane) - 1, lane & 3); break;
    case TrapAllLaneOutOfOwnGroup: r = __all_sync(lane < 16 ? 0xffffu : 0xfffe0000u, p); break;
    case TrapReduxExclusivePrefix: r = __reduce_add_sync((1u << lane) - 1, lane); break;
  }
  out[threadIdx.x] = r;
}

static const char* const kNames[NumCases] = {
  "ballot, one runtime mask 0xffff", "ballot, mask 1<<lane", "ballot, mask (2<<lane)-1", "ballot, a mask per half",
  "any, a mask per half", "all, a mask per half", "match_any, a mask per half", "match_any, one runtime mask 0xffff",
  "ballot, constant 0xffff", "reduce_add, constant 0xffff", "reduce_add, a mask per half", "shfl, mask (2<<lane)-1, own lane",
  "shfl, a mask per half", "shfl, a mask per half, a lane of the other half", "shfl, one runtime mask 0xffff",
  "shfl from a lane that is not running it", "shfl_down off the end of the lanes running it",
  "TRAP ballot, mask (1<<lane)-1", "TRAP any, mask (1<<lane)-1", "TRAP shfl, mask (1<<lane)-1",
  "TRAP syncwarp, mask (1<<lane)-1", "TRAP reduce_add, one runtime mask 0xffff", "TRAP ballot, lane outside its half's mask",
  "TRAP match_any, mask (1<<lane)-1", "TRAP all, lane outside its half's mask", "TRAP reduce_add, mask (1<<lane)-1",
};

// What lane `l` must read back for the cases that do not trap.
static unsigned expected(int c, unsigned l) {
  unsigned P = 0;
  for (unsigned i = 0; i < 32; ++i)
    if ((i % 3) == 0 || i > 28) P |= 1u << i;
  const bool low = l < 16;
  const unsigned half = low ? 0xffffu : 0xffff0000u;
  const bool p = (P >> l) & 1;
  unsigned classes = 0;   // lanes of the half with this lane's l & 3
  for (unsigned i = 0; i < 32; ++i)
    if ((i & 3) == (l & 3) && ((half >> i) & 1)) classes |= 1u << i;
  switch (c) {
    case BallotUniformRuntime: case BallotConstant: return P;
    case BallotOwnBit: case BallotPrefixInclusive: return p ? 1u << l : 0;
    case BallotTwoGroups: return P & half;
    case AnyTwoGroups: return low ? 0 : 1;
    case AllTwoGroups: return low ? 1 : 0;
    case MatchTwoGroups: return classes;
    case MatchUniformRuntime: return 0x11111111u << (l & 3);
    case ReduxConstant: return 496;
    case ReduxTwoGroups: return low ? 120 : 376;
    case ShflPrefixOwnLane: return l * 10;
    case ShflTwoGroups: return low ? 30 : 190;
    case ShflTwoGroupsCross: return 170;
    case ShflUniformRuntime: return 10;
    case ShflFromInactive: return low ? 0 : 0xdead;
    case ShflDownEdge: return l < 15 ? (l + 1) * 10 + 1 : l == 15 ? 0 : 0xdead;
  }
  return 0;
}

// One case, in a process of its own: a trap ends the context, and on a card a
// new one does not always come back in the same process.
static int run_case(int c) {
  const bool traps = c >= TrapBallotExclusivePrefix;
  unsigned* d = nullptr;
  cudaError_t e = cudaMalloc(&d, 32 * sizeof(unsigned));
  if (e != cudaSuccess) { printf("FAIL %s: cudaMalloc: %s\n", kNames[c], cudaGetErrorString(e)); return 1; }
  cudaMemset(d, 0xaa, 32 * sizeof(unsigned));
  k<<<1, 32>>>(d, c, 0xffffu);
  e = cudaDeviceSynchronize();
  if (traps) {
    if (e == cudaErrorIllegalInstruction) return 0;
    printf("FAIL %s: want the illegal instruction error (715), got %d (%s)\n", kNames[c], int(e), cudaGetErrorString(e));
    return 1;
  }
  if (e != cudaSuccess) { printf("FAIL %s: %s\n", kNames[c], cudaGetErrorString(e)); return 1; }
  unsigned h[32] = {};
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  cudaFree(d);
  for (unsigned l = 0; l < 32; ++l)
    if (h[l] != expected(c, l)) {
      printf("FAIL %s: lane %u read 0x%x, want 0x%x\n", kNames[c], l, h[l], expected(c, l));
      return 1;
    }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1) return run_case(atoi(argv[1]));   // one case, as the parent below runs it
  // The parent never touches CUDA, so a fork of it is a fresh start.
  int fails = 0;
  for (int c = 0; c < NumCases; ++c) {
    fflush(stdout);
    const pid_t pid = fork();
    if (pid < 0) { printf("FAIL fork\n"); return 1; }
    if (pid == 0) _exit(run_case(c));
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      if (!WIFEXITED(status)) printf("FAIL %s: the process died (status %d)\n", kNames[c], status);
      ++fails;
    }
  }
  printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
  return fails ? 1 : 0;
}
