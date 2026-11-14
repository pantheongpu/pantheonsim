# Exact parallel launches (`VGPU_EXACT_THREADS`)

`VGPU_THREADS=1` runs the blocks of a launch one after another, in order, on one host thread. It is the setting whose results are a
function of the program alone: the reference runs of `tools/verify`, the counter checks of `tools/bench` and the result cache
(`VGPU_MEMO`, for kernels with atomics) all ask for it. Several host threads (`VGPU_THREADS=N`, the default) are faster and give the
same answer for a kernel whose blocks are independent, but nothing checks that they are: blocks that share an atomic, or race, give
one of several answers.

`VGPU_EXACT_THREADS=N`, set beside `VGPU_THREADS=1`, keeps the promise of the first and takes the speed of the second where it can
prove the two are the same. It is opt-in and off by default; without it nothing changes. It applies to the SASS engine (the default
for NVIDIA kernels). Launches of the PTX interpreter and of the AMD engine run in order as before.

## What is promised

With `VGPU_THREADS=1 VGPU_EXACT_THREADS=N`, for every launch:

* device memory after the launch,
* the launch's counters (`LaunchStats`: every counter, the per-class and per-opcode ones too; `VGPU_COUNTERS`, `VGPU_COUNTERS_FILE`,
  `vgpu ncu`),
* the error of a launch that fails, and the memory it leaves behind,

are those of `VGPU_THREADS=1` alone. The exceptions, both invisible to a kernel and to a copy back to the host, are listed under
"What is not the same".

## How

A launch that qualifies (below) is run on trial: `N` host threads take its blocks from the same hand-out the other launches use, and
while the trial lasts the memory manager keeps a shadow of the device's address range (`src/core/memory_spec.cpp`). The shadow has
two grains.

**By word.** One 32-bit word stands for four bytes of device memory and holds one of four states:

| state | meaning |
|---|---|
| 0 | no block has touched the bytes |
| `b + 1` | block `b` has read them; no other block has touched them |
| all ones | more than one block has read them; none has written them |
| top bit and `b + 1` | block `b` has written them (and may have read them); no other block has touched them |

A word only moves down that table (untouched, one reader, several readers; or untouched or its own reader mark to its writer), by
compare-and-swap, and a block makes an access only after the word allows it:

* a read of a word another block wrote, and a write of a word another block read or wrote, stop the trial before the access is
  made (`MemoryManager::SpecStop`, thrown from the access);
* the first write of a block to a word copies the bytes that are there into the block's undo log, then writes.

**By page.** Most of what a kernel touches it only reads, and a compare-and-swap per word for bytes nobody writes is
the larger part of what a trial costs. So each 4 KiB page of the range has a state of its own, and a page is noted by word only
once a block writes in it:

| page state | meaning |
|---|---|
| 0 | no block has touched the page |
| `b + 1` | block `b` has read in it; no other block has touched it; nothing is noted by word |
| all ones | more than one block has read in it; none has written in it; nothing is noted by word |
| by word | the page's words (above) say what was done to its bytes |
| by word, coarse | the same, after the page's only reader went on to write in it: its reads had not been noted by word, so every word of the page started as "read by that block" |

A page moves untouched, one reader, several readers by compare-and-swap, as a word does. It becomes "by word" in two ways only:
from untouched, by the first block that writes in it (or by the first block that touches it at all, where the trial's plan says so,
below); and from one reader, by that reader itself when it writes in the page, after it has put a page of words in place in which
every word carries its reader mark. "Several readers", "by word" and "by word, coarse" are final for the trial. A write is only
ever made in a page that is noted by word, after the word allows it; so:

* a write in a page that another block has read by page state, or that several have, stops the trial before the write is made;
* a read in a page noted by word goes by the words.

Every method of the memory manager that reads or writes device bytes makes this note first (`spec_read`, `spec_write` and their
batch forms in `src/core/memory.cpp`; with no trial on, each is one null test). The methods that hand out a pointer for the caller
to use later (`batch_window`, `atomic_word_ptrs`) decline during a trial, and the caller takes the path that is noted.

Four things make the note cheaper without changing what is noted. Each is a memory a worker keeps for itself, or a wider
compare-and-swap; none adds a state:

* **Pages a worker has seen.** Each worker remembers up to 256 pages its block may read with no further note: pages whose state
  was the block's own reader mark, valid for that block only, and pages with several readers, valid for the rest of the trial
  (that state is final). A read of such a page is one compare in the worker's own memory, with no atomic at all. The memory is
  asked the page and the block, so the next block on the same worker starts with nothing but the pages of several readers. (The
  place of a page in the memory is a hash of its number: a warp's lanes are often a fixed number of pages apart, the rows of a
  matrix, and the low bits of the page number sent them all to a few places.)
* **Runs of words a block has noted.** In a page noted by word, a worker remembers the last four runs of sixteen words or
  more (a warp's lanes one after another) that its block took as their writer (every word holds the block's writer mark, which nothing changes), and the last four it
  noted for reading (every word held the block's reader mark, its writer mark or the several-readers mark, and from those a
  word only moves to another of them). An access of the same block, of sixteen words or more, inside a remembered run has
  nothing to note: the store of `tmp[i] += ...` in a loop is one range compare instead of 32 loads and compares. Two runs of
  one block that touch are remembered as one; a run is asked the block. (One lane's few words neither enter the memory nor ask
  it: the runs of 32 lanes would push one another out, and the asking cost each of them.)
* **Pages of words a worker has read in.** The worker remembers where the words of 64 such pages are; a read whose words all
  allow it already (looked at one by one, every time) needs no call and no compare-and-swap, and a lone untouched word (a
  lane of a scattered load) is taken as its first reader where it is found.
* **Two words at once.** Two neighbouring words (the first at an even index) are read as one 64-bit value, and taken by one
  64-bit compare-and-swap when both are free for the block (both untouched for a read; each untouched or the block's own reader
  mark for a write). When either holds anything else the words are handled one at a time as before, so no state is entered
  that the single-word rules do not allow: the 64-bit compare-and-swap succeeds only if both words still hold exactly the values
  that were judged free.

The undo log of a worker keeps its memory from one trial to the next (up to 64 MiB; a launch that writes much writes as much the
next time).

When the workers are done:

* **no stop:** the trial is kept. The shadow is cleared, the undo logs are dropped, the workers' counters are added up.
* **a stop** (a conflict, or any exception in any block: a fault, a step limit, a trap): every undo log is written back, the shadow
  is cleared, and the launch runs again from block 0 in order on one thread, exactly as without the variable. What it reports is
  what the run in order reports.

### Why a kept trial is the run in order

Take two blocks and one byte. If both only read it, neither can tell. If one writes it and the other reads or writes it, the
writer wrote in a page that was noted by word at that moment (no write is made in any other), and it held the byte's word as its
writer. The other block made its note either

* by the word, in which case both went through the byte's word, and whichever came second found a state it may not enter (a
  writer finds another block's mark or the several-readers mark; a reader finds another block's writer mark) and stopped the trial
  before touching the byte; or
* by the page's state (it can only have been reading). Then the page was in the state "read by that block" or "several readers"
  when it looked. From "several readers" a page never becomes "by word", so the writer can never have written in it. From "read by
  that block" only that block itself makes the page "by word", so the writer, a different block, can only have found the page
  already noted by word, which means the reader made it so after its read; but then the reader filled every word of the page with
  its own reader mark before the page's state changed, the writer found that mark on the byte's word, and stopped. And a block
  that looks at the worker's memory instead of the page's state relies on one of these two states having been seen earlier for
  the same block (or, for "several readers", for any block): both stay true of the page for as long as the memory is used, the
  first because a page read by `b` can only be changed to "several readers" (also fine for `b`) or to "by word" by `b` itself
  with `b`'s mark on every word.

The same holds for a block that goes by a remembered run of words instead of the words: a word in a run the block wrote holds the
block's writer mark for the rest of the trial; a word in a run it noted for reading held a mark that allows the block's read, and
every state such a word can move to still stops another block's write of it (another block's reader mark on it makes it "several
readers"; the block's own write makes it the block's). Taking two words by one compare-and-swap is taking each of them by the
single-word rule at the same instant, or not at all.

So in a trial that was not stopped, no byte was written by one block and touched by another. Each block therefore read, for every
byte, either what the launch started with or what the block itself stored, and that is also what it reads when the blocks run in
order: a byte it reads was written by no other block, before it or after it. A block's execution is a function of what it reads
(its registers, shared and local memory are its own, and its warps are scheduled inside the block as always), so every block ran
as it does in order, stored the same values to bytes no other block stored to, and counted the same counts. The sums of the
counters are sums of integers, and do not depend on the order of adding.

The note by page says more was read than was (a whole page for a read of four bytes; every word of a page for a block that read
some of it and then wrote in it). More reading noted can only stop more trials, never keep one that the words would have stopped.
What it costs is dealt with under "A stop the page grain made".

The other ways a block could see the order of blocks are closed one by one:

| channel | what the trial does |
|---|---|
| atomics on device memory | noted as a write of the word: two blocks on one word stop the trial; a block's atomics on words only it touches have one answer |
| `%smid` (SR_VIRTUALSMID) on a grid larger than the device: the number is the host thread's | stops the trial (`Runner::sreg`) |
| device `printf`, `assert`, `malloc`/`free`, any device runtime call | not tried: a module with any extern runs in order |
| a kernel that launches kernels | not tried |
| a kernel that polls memory (lockstep), cooperative launches, clusters (and `clusterlaunchcontrol.try_cancel`, which takes the next block from the hand-out) | not tried: these keep their own schedules |
| textures and surfaces | not tried |
| pinned, managed or registered host memory | reads are allowed and not noted (the shadow covers the device's range only; `host_load_batch` and `load_host_run` make no note at all): no block can write it, because every writing method stops the trial before a store to a host-mapped address, wherever the mapping is; the host itself writing it during a launch is as visible in order |
| virtual-memory reservations, mappings and handles, memory shared with another process, file-backed allocations | not tried: the same bytes may be mapped at two addresses, which a shadow by address cannot see |
| an armed fault (`vgpu fault arm`), the host-GPU card gate, the result cache, a memory trace | not tried |
| `VGPU_SASS_TRACE`, `VGPU_TRACE`, `VGPU_RACE`, `VGPU_TIER_STATS` | not tried: their output is in the order things ran |
| an allocation, a free or a device-to-device copy during the launch | stops the trial |
| the warp clocks (`SR_CLOCK*`) | a count of the warp's own instructions: the same in any order |

### Why a stopped trial leaves nothing behind

A block writes a byte only after its word is the block's own writer mark, and it copies the old bytes of the word between taking
the mark and the first store. No other block writes a word that carries the mark, so what was copied is what the launch started
with, and no byte the trial wrote is outside a word some block marked and copied. Writing the copies back, in any order, gives
the memory of before the launch. Bytes of a marked word that lie outside every allocation (the tail of an allocation whose size is
no multiple of four) are not copied and cannot be written. Two words taken at once are kept like two taken one after the other
(they join the run of words taken in the call, which is copied before the call returns). When a word in the middle of a store's
range is another block's, the trial stops with the words before it marked and not yet copied, and no byte of the store has been
written (the note comes before the store), so there is nothing of theirs to put back. A store inside a remembered run of the
block's own words keeps nothing: those words were kept when the block took them. The page states hold no bytes and are cleared
with the words.

Every block of a trial sees a state of memory that could be real (the launch's start plus its own stores), so a block of a trial
that will be stopped still runs on consistent data. It may run longer than it does in order, or wait for a store that never comes;
each worker therefore looks at the stop flag after every round of a block's warps (at most 4096 instructions a warp) and at every
access. A block that waits on an earlier block's store makes the conflict itself: its read and the other's write meet at one word,
or at the page the reader took.

### A stop the page grain made

A kernel that updates an array in place (`a[i] = f(a[i])`) reads and writes words of its own, and blocks share pages: by word
nothing conflicts, by page the first block's read takes the page and the next block's write in it is stopped. Such a stop is
marked *coarse*: a write in a page other blocks read by page state, or a write that met a reader's mark (one block's, or several
readers') on a word of a page that is "by word, coarse". A stop at a writer's mark is never coarse (writers' marks are exact), nor
is any stop of a read.

A trial with a coarse stop is undone like any other. The launch is then tried again, with the allocation the stop was in noted by
word from its first touch (`MemoryManager::SpecPlan`; pages elsewhere stay by page), and what was learned is kept for the kernel's
later launches (`Module::exact_plan`, by address range). After the third coarse stop in one launch, after one with nothing new to
learn, or with more than 32 ranges learned, the kernel is noted by word everywhere, which has no coarse stops. Only a trial that
stops for another reason, or commits, ends the launch's trials; what is run in order is run in order exactly as before. The
second trial changes time only: a trial that commits is the run in order whatever its plan, by the argument above.

What a coarse stop throws away is the work up to a block's first store in a page it read. With many blocks, or a store early in
a block, that is little. With a few blocks and a store at a block's end it is most of the launch, so a launch of fewer than 8
blocks is noted by word from the start and has no coarse stop.

`VGPU_EXACT_PAGES=0` notes every page by word from its first touch (the note the mode began with), for measurements and as the
fourth form of the test.

### A kernel that keeps being stopped

A stopped trial costs the work done before the stop. A kernel whose trial was stopped is run in order for the next launch, then
tried again; after each further stop it waits twice as long (2, 4, ... 4096 launches), and a kept trial puts it back. Which
launches are tried moves time only.

The width rule of ordinary launches (`LaunchWidth`, `src/exec/worker_pool.cpp`) applies too: a launch too small to pay for a
hand-off to the pool stays on one thread and is not tried.

## What is not the same

* **How memory is held, not what it holds.** A stopped trial may leave a 64 KiB chunk materialized that the run in order would have
  left untouched or uniform (a block of the trial stored to it and the store was undone). Every byte reads the same; the host
  memory the simulator uses can be higher.
* **Time**, and what is derived from it (the pool's width rule, the telemetry's utilisation).
* **The order of the lines of `VGPU_EXACT_LOG`** relative to other output.

## Cost

While a trial runs, each access of a warp to global memory makes one call of the note on top of the access. For a page the block
may read without a note (read by it, or by several blocks, and written by none) that is a compare in the worker's memory per page
the warp touches; for a page noted by word, a range compare when the block is remembered to have noted the words, else a load
and a compare per word when the block has been there and a compare-and-swap per two words the first time, and for a store a
copy of the old bytes. The page states take four bytes of host memory per 4 KiB
of device memory the launch touches; the words 4 KiB per 4 KiB page a block writes in (or that the plan notes by word); both are
cleared after each trial. The undo log holds the bytes the launch wrote for the first time, at most 1 GiB per worker (more stops
the trial). A kernel with a coarse stop pays for the part of the trial before the stop once per learned allocation. Measurements:
Round sp47 and Round sp55 in [the performance log](perf-design/rounds/README.md).

## Seeing what happened

`VGPU_EXACT_LOG=<file>` appends a line per launch the variable applied to: kernel, blocks, threads, then `committed`, `stopped`
or `in order`, the reason, the bytes kept for undoing and the 4 KiB pages touched; a trial that was stopped coarsely and tried again
has a line of its own before it, with `again`. `VGPU_EXACT_STATS=<file>` appends a line per trial with what its notes cost
(`MemoryManager::SpecStats`): calls of the hooks, pages taken by one reader, by several, noted by word, made coarse, words looked at
and taken by compare-and-swap for reads and for writes, pieces and bytes kept, accesses answered by a remembered run.
`sass::exact_counts()` gives the process's totals to a test (`again`: the trials run a second time).

## Tests

`tests/unit/test_exact_threads.cpp` (also run with `VGPU_TIER=0` and `VGPU_FASTPATH=0`, which send the accesses through other
methods of the memory manager, and with `VGPU_EXACT_PAGES=0`): kernels with independent blocks, blocks that store to the same words,
a chain in which each block reads what the one before stored (also on every other word of a host-mapped buffer, through the batched host
path), a fault in block 1, a step budget that fails every block before its
store, and `%smid` on a grid larger than the device, each compared word for word and counter for counter with the run in order at 2,
3, 4 and 8 threads; every reading and writing method of `MemoryManager` against the shadow; every writing method stopping the trial at
a host-mapped address (at the host's own address and inside the device's range), and every reading method's host path kept; a stopped trial undone to the byte across
chunk ends and at an allocation's tail.

For the note by page: a second block, on the same worker and on another, in a page that a first block claimed whole (as its reader;
as its reader and then a writer; with a second reader), at every byte of the first, a middle and the last word of the page, with
lengths of 1 to 8 bytes and accesses that end in the next page: stopped before the access wherever a note by word stops it, undone
to the byte, said to be coarse exactly where only the grain stopped it, and kept when run again with the allocation by word; the
worker's memory of a page not answering for the next block or for another page that shares its place (nor its memory of a page
of words); a warp's lanes each noted (a word in the middle taken by another block, a lane whose first or last byte is in another
page than the rest); remembered runs of words answering for their block only, joined only when they touch; readable words
looked at every time; two words taken at once only when both are free, at even and odd places, and kept like two; 600 random
interleavings of four blocks on two workers against a model of the four word states (by word: stopped at exactly
the access the model stops; by page: never later, earlier only as a coarse stop, and the model's again with the allocation by
word); a kernel that updates its buffer in place, tried again by word and kept, with the bits and counters of the run in order.
