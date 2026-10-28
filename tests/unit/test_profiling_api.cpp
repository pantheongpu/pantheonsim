// The per-call bookkeeping of the shims (vgpu/profiling.hpp): what ApiCall reports to a recorder and to a subscriber,
// which call is the outermost, whose arguments are whose, and the two flags every call tests. These paths are taken
// by every runtime and driver call, so they are kept short; this pins what they must still do.
#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/profiling.hpp"
#include "vtest.hpp"

using namespace vgpu::profiling;

namespace {
struct Seen {
  std::string name;
  bool enter;
  int nargs;
  uint32_t correlation;
  int result;
  uint64_t first_arg;   // the value of the first argument, if there is one
  uint16_t first_size;
};
std::vector<Seen> g_seen;
void on_api(const ApiInfo& i) {
  Seen s{i.name, i.enter, i.nargs, i.correlation, i.result, 0, 0};
  if (i.nargs > 0 && i.args) {
    std::memcpy(&s.first_arg, i.args[0], i.arg_sizes ? i.arg_sizes[0] : 8);
    s.first_size = i.arg_sizes ? i.arg_sizes[0] : 0;
  }
  g_seen.push_back(s);
}

// Everything off, and nothing left from the test before.
struct Quiet {
  Quiet() { reset(); }
  ~Quiet() { reset(); }
  static void reset() {
    set_enabled(false);
    set_hooks(Hooks{});
    drain();
    g_seen.clear();
  }
};

void note(uint64_t a) {
  static thread_local uint64_t slot;
  slot = a;
  const void* argv[1] = {&slot};
  static const uint16_t sizes[2] = {8, 0};
  note_args(argv, 1, sizes);
}
}  // namespace

// The flags are read inline by every library, and mean what the functions they replaced meant.
VTEST(the_two_flags_follow_their_setters) {
  Quiet q;
  VCHECK(!enabled());
  VCHECK(!hooked());
  set_enabled(true);
  VCHECK(enabled());
  VCHECK(!hooked());
  set_enabled(false);
  Hooks h;
  h.api = on_api;
  set_hooks(h);
  VCHECK(hooked());
  VCHECK(!enabled());
  set_hooks(Hooks{});
  VCHECK(!hooked());
}

// Nobody listening: nothing is recorded, nothing reported, and the call still counts its depth, so that a call made
// through another is nested whether or not anyone listened when the outer one began.
VTEST(a_call_nobody_listens_to_leaves_no_trace) {
  Quiet q;
  {
    ApiCall outer("outer");
    ApiCall inner("inner");
    inner.set_result(3);
    outer.set_result(4);
  }
  VCHECK(drain().empty());
  VCHECK(g_seen.empty());
}

// A recorder gets one Api event for the outermost call, in the order of its end, with the result; a call made through
// it is part of it.
VTEST(a_nested_call_is_part_of_the_outermost_one_in_the_recorder) {
  Quiet q;
  set_enabled(true);
  {
    ApiCall outer("cudaMemcpy");
    {
      ApiCall inner("cudaMemcpyAsync");
      inner.set_result(9);
    }
    outer.set_result(7);
  }
  {
    ApiCall again("cudaFree");
    again.set_result(0);
  }
  const std::vector<Event> ev = drain();
  VCHECK_EQ(ev.size(), size_t{2});
  VCHECK(ev[0].kind == EventKind::Api && ev[0].name == "cudaMemcpy" && ev[0].result == 7);
  VCHECK(ev[1].kind == EventKind::Api && ev[1].name == "cudaFree" && ev[1].result == 0);
  VCHECK(ev[0].correlation != 0 && ev[1].correlation != ev[0].correlation);
  VCHECK(ev[0].end_ns >= ev[0].start_ns);
}

// The work a call issues carries the call's correlation, and a fresh one outside any call.
VTEST(work_issued_inside_a_call_carries_its_correlation) {
  Quiet q;
  set_enabled(true);
  uint32_t inside = 0, nested_inside = 0;
  {
    ApiCall outer("outer");
    inside = work_correlation();
    ApiCall inner("inner");
    nested_inside = work_correlation();
  }
  const std::vector<Event> ev = drain();
  VCHECK_EQ(ev.size(), size_t{1});
  VCHECK_EQ(inside, ev[0].correlation);
  VCHECK_EQ(nested_inside, ev[0].correlation);
  const uint32_t a = work_correlation(), b = work_correlation();
  VCHECK(a != b);   // outside a call every question is a new one
}

// A subscriber is told of the outermost call as it enters and as it returns, with its arguments (the bytes the caller
// kept, their sizes) and its result; a nested call is not told.
VTEST(a_subscriber_is_told_of_the_outermost_call_with_its_arguments) {
  Quiet q;
  Hooks h;
  h.api = on_api;
  set_hooks(h);
  note(0x1122334455667788ull);
  {
    ApiCall outer("cuMemAlloc_v2", Domain::Driver);
    {
      ApiCall inner("cuMemAlloc_inner");
    }
    outer.set_result(2);
  }
  VCHECK_EQ(g_seen.size(), size_t{2});
  VCHECK(g_seen[0].enter && g_seen[0].name == "cuMemAlloc_v2" && g_seen[0].nargs == 1);
  VCHECK_EQ(g_seen[0].first_arg, 0x1122334455667788ull);
  VCHECK_EQ(g_seen[0].first_size, uint16_t{8});
  VCHECK(!g_seen[1].enter && g_seen[1].name == "cuMemAlloc_v2" && g_seen[1].result == 2);
  VCHECK_EQ(g_seen[1].nargs, 1);   // (read again at the exit)
  VCHECK_EQ(g_seen[0].correlation, g_seen[1].correlation);
  // No recorder was on: the subscriber alone makes a call visible, and nothing is recorded.
  VCHECK(drain().empty());
}

// The arguments noted for a call are that call's, whether it reports them or not: a nested call takes the ones noted
// for it, and the next outermost call starts with none.
VTEST(arguments_noted_for_a_nested_call_do_not_leak_into_the_next_call) {
  Quiet q;
  Hooks h;
  h.api = on_api;
  set_hooks(h);
  {
    ApiCall outer("first");
    note(41);
    ApiCall inner("inner");   // takes 41, not outermost, reports nothing
  }
  {
    ApiCall second("second");   // nothing was noted for it
  }
  note(42);
  {
    ApiCall third("third");
  }
  VCHECK_EQ(g_seen.size(), size_t{6});
  VCHECK(g_seen[0].name == "first" && g_seen[0].nargs == 0);
  VCHECK(g_seen[2].name == "second" && g_seen[2].enter && g_seen[2].nargs == 0);
  VCHECK(g_seen[4].name == "third" && g_seen[4].enter && g_seen[4].nargs == 1 && g_seen[4].first_arg == 42);
}

// Arguments noted for a call that nobody listened to are gone with it: they are not handed to a later call that is
// listened to.
VTEST(arguments_noted_for_an_unlistened_call_are_gone_with_it) {
  Quiet q;
  Hooks h;
  h.api = on_api;
  set_hooks(h);
  note(5);
  set_hooks(Hooks{});   // (the listener leaves before the call it noted them for)
  {
    ApiCall nobody("nobody");   // takes the note, reports nothing
  }
  set_hooks(h);
  {
    ApiCall listened("listened");
  }
  VCHECK_EQ(g_seen.size(), size_t{2});
  VCHECK(g_seen[0].name == "listened" && g_seen[0].nargs == 0);
}

// A Silence is not the program: it counts as being inside a call (no call made under it is reported) and keeps its
// events out of the recorder.
VTEST(a_silenced_thread_reports_nothing) {
  Quiet q;
  set_enabled(true);
  VCHECK(!silenced());
  {
    Silence quiet;
    VCHECK(silenced());
    ApiCall inside("inside");
    Event e;
    e.kind = EventKind::Marker;
    record(std::move(e));
  }
  VCHECK(!silenced());
  VCHECK(drain().empty());
}

// Each thread has its own depth and noted arguments.
VTEST(threads_do_not_share_call_depth) {
  Quiet q;
  set_enabled(true);
  std::atomic<bool> inside{false}, release{false};
  std::thread t([&] {
    ApiCall held("held");   // the outermost call of this thread
    inside = true;
    while (!release.load()) std::this_thread::yield();
  });
  while (!inside.load()) std::this_thread::yield();
  {
    ApiCall mine("mine");   // the outermost call of this one, though another thread is inside one
  }
  release = true;
  t.join();
  const std::vector<Event> ev = drain();
  VCHECK_EQ(ev.size(), size_t{2});
}

// The first context made is told once, whoever makes it and however often it is asked.
VTEST(the_context_hook_runs_once) {
  static std::atomic<int> told{0};
  Quiet q;
  set_context_hook([] { ++told; });
  VCHECK(!context_made());
  std::vector<std::thread> ts;
  for (int i = 0; i < 4; ++i)
    ts.emplace_back([] {
      for (int k = 0; k < 100; ++k) note_context_made();
    });
  for (auto& t : ts) t.join();
  VCHECK(context_made());
  VCHECK_EQ(told.load(), 1);
  note_context_made();
  VCHECK_EQ(told.load(), 1);
}

VTEST_MAIN
