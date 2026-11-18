// Turbo mode's program history (include/vgpu/host_gpu_history.hpp, docs/perf-design/12-host-gpu-turbo.md section 17), with no card and no simulator: the record's life
// across runs, and the file that carries it (hit, miss, corrupt, a version that changed, writers at the same time). The rule under all of it: whatever is wrong with
// the file, the answer is "no history", which is what the mode did before the history existed.
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/host_gpu_history.hpp"
#include "vtest.hpp"

using namespace vgpu::hostgpu::history;

namespace {
std::string fresh_dir() {
  char t[] = "/tmp/vgpu-history-test.XXXXXX";
  const char* d = ::mkdtemp(t);
  return d ? d : "";
}
void remove_dir(const std::string& dir) {
  const std::string cmd = "rm -rf '" + dir + "'";
  if (std::system(cmd.c_str()) != 0) std::fprintf(stderr, "cleanup of %s failed\n", dir.c_str());
}
std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream o;
  o << f.rdbuf();
  return o.str();
}
void spit(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f << text;
}
Kernel kernel(const char* name, uint32_t launches, double card_s, double sim_s, bool ok, bool faulted, uint64_t max_threads = 0, bool loop = false) {
  return Kernel{name, launches, max_threads, card_s, sim_s, ok, faulted, loop};
}
Observed card_run(double card_s, uint32_t launches, std::vector<Kernel> ks, double loopfree_work = 1000) {
  Observed o;
  o.card_s = card_s;
  o.launches = launches;
  o.loopfree_work = loopfree_work;
  o.kernels = std::move(ks);
  return o;
}
Observed held_run(double sim_s, bool released, uint32_t launches) {
  Observed o;
  o.held = true;
  o.released = released;
  o.sim_s = sim_s;
  o.launches = launches;
  return o;
}
const Limits kLim = limits_for_start_cost(0.25);
}  // namespace

VTEST(a_missing_record_is_no_history) {
  const Decision d = decide_start(nullptr);
  VCHECK(d.start_driver);
  VCHECK(!d.hold);
  VCHECK(!known_faulted(nullptr, "k"));
  VCHECK(!kernel_outlasts(nullptr, "k", 1.0));
}

VTEST(the_first_short_card_run_makes_the_program_a_candidate_and_the_simulator_run_confirms_it) {
  Record r;
  update(&r, card_run(0.0004, 26, {kernel("bfs_k", 26, 0.0004, 0, true, false)}), kLim);
  VCHECK(r.verdict == Verdict::Candidate);
  VCHECK_EQ(r.runs, uint64_t{1});
  Decision d = decide_start(&r);
  VCHECK(!d.start_driver);
  VCHECK(d.hold);
  update(&r, held_run(0.011, false, 26), kLim);   // the trial: the simulator took 11 ms, under half the start
  VCHECK(r.verdict == Verdict::Tiny);
  VCHECK(decide_start(&r).hold);
  VCHECK_EQ(r.runs, uint64_t{2});
}

VTEST(a_trial_that_spends_its_budget_says_long_and_the_driver_comes_back) {
  Record r;
  update(&r, card_run(0.002, 10, {kernel("k", 10, 0.002, 0, true, false)}), kLim);
  VCHECK(r.verdict == Verdict::Candidate);
  update(&r, held_run(0.25, true, 10), kLim);   // released
  VCHECK(r.verdict == Verdict::Long);
  const Decision d = decide_start(&r);
  VCHECK(d.start_driver);
  VCHECK(!d.hold);
  update(&r, card_run(0.001, 10, {}), kLim);   // a later card run does not turn Long back into a candidate
  VCHECK(r.verdict == Verdict::Long);
}

VTEST(a_held_run_just_under_the_budget_but_over_half_the_start_is_not_tiny) {
  Record r;
  r.verdict = Verdict::Tiny;
  update(&r, held_run(0.2, false, 5), kLim);   // finished without release but over tiny_s
  VCHECK(r.verdict == Verdict::Long);
  Record r2;
  update(&r2, held_run(0.124, false, 5), kLim);
  VCHECK(r2.verdict == Verdict::Tiny);
  update(&r2, held_run(0.125, false, 5), kLim);   // the boundary is not tiny
  VCHECK(r2.verdict == Verdict::Long);
}

VTEST(a_long_card_run_is_not_a_candidate) {
  Record r;
  update(&r, card_run(0.5, 100, {kernel("k", 100, 0.5, 0, true, false)}), kLim);
  VCHECK(r.verdict == Verdict::Unknown);
  VCHECK(decide_start(&r).start_driver);
  Record r2;
  update(&r2, card_run(0.001, 0, {}), kLim);   // no launch at all: nothing to say
  VCHECK(r2.verdict == Verdict::Unknown);
  Record r3;
  update(&r3, card_run(0.001, 100000, {}), kLim);   // too many launches to be a short program
  VCHECK(r3.verdict == Verdict::Unknown);
  Record r4;
  update(&r4, card_run(0.001, 3, {}, 5e8), kLim);   // kernels without a loop whose threads x instructions is over what the simulator does in about a tenth of a second
  VCHECK(r4.verdict == Verdict::Unknown);
  Record r5;
  update(&r5, card_run(0.001, 3, {}, 4e8), kLim);   // the bound itself is in
  VCHECK(r5.verdict == Verdict::Candidate);
  Record r6;
  update(&r6, card_run(0.005, 1, {}, 100), kLim);   // and so is the card's time bound out
  VCHECK(r6.verdict == Verdict::Unknown);
  Record r7;   // a kernel with a loop on 65,536 threads in a launch (aobench: 1 ms on the card, 0.7 s in the simulator)
  update(&r7, card_run(0.001, 1, {kernel("ray", 1, 0.0002, 0, true, false, 65536, true)}), kLim);
  VCHECK(r7.verdict == Verdict::Unknown);
  Record r8;   // and on 4,096 (bfs)
  update(&r8, card_run(0.001, 26, {kernel("bfs", 13, 0.0003, 0, true, false, 4096, true)}), kLim);
  VCHECK(r8.verdict == Verdict::Candidate);
  VCHECK(r8.kernels[0].loop);
}

VTEST(a_kernel_that_failed_on_the_card_is_kept_out_of_it_for_good) {
  Record r;
  update(&r, card_run(0.001, 1, {kernel("srad", 1, 0, 0, false, true)}), kLim);
  VCHECK(known_faulted(&r, "srad"));
  VCHECK(!known_faulted(&r, "other"));
  VCHECK(r.verdict == Verdict::Unknown);   // a fault is no candidate: the simulator run is not a trial
  const Decision d = decide_start(&r);      // nothing known to run on the card: the driver does not start with the device
  VCHECK(!d.start_driver);
  VCHECK(!d.hold);
  update(&r, card_run(0.001, 1, {kernel("srad", 1, 0, 0, false, false)}), kLim);   // a later run that did not fault it does not clear it
  VCHECK(known_faulted(&r, "srad"));
  update(&r, card_run(0.3, 5, {kernel("good", 5, 0.3, 0, true, false)}), kLim);   // a kernel the card runs: the driver starts again
  VCHECK(decide_start(&r).start_driver);
  VCHECK(known_faulted(&r, "srad"));
}

VTEST(note_fault_adds_the_kernel_without_counting_a_run) {
  Record r;
  note_fault(&r, "k");
  VCHECK(known_faulted(&r, "k"));
  VCHECK_EQ(r.runs, uint64_t{0});
  note_fault(&r, "");
  VCHECK_EQ(r.kernels.size(), size_t{1});
}

VTEST(a_kernel_the_history_knows_to_be_long_ends_the_hold_before_its_launch) {
  Record r;
  r.kernels.push_back(kernel("big", 2, 0, 0.4, false, false));
  VCHECK(kernel_outlasts(&r, "big", 0.2));    // 0.2 s a launch against 0.2 s left
  VCHECK(!kernel_outlasts(&r, "big", 0.3));
  VCHECK(!kernel_outlasts(&r, "other", 0.001));
  // a launch much larger than the record's largest ends it too: another input
  r.kernels.push_back(kernel("small", 5, 0.0001, 0.0001, true, false, 4096));
  VCHECK(!kernel_outlasts(&r, "small", 0.2, 4096));
  VCHECK(!kernel_outlasts(&r, "small", 0.2, 8192));   // twice is allowed
  VCHECK(kernel_outlasts(&r, "small", 0.2, 8193));
  VCHECK(!kernel_outlasts(&r, "small", 0.2));          // (no threads given: only the time is asked)
}

VTEST(the_record_is_bounded) {
  Record r;
  std::vector<Kernel> many;
  for (int i = 0; i < 200; ++i) many.push_back(kernel(("k" + std::to_string(i)).c_str(), 1, 0.001, 0, true, false));
  update(&r, card_run(0.1, 200, many), kLim);
  VCHECK_EQ(r.kernels.size(), kMaxKernels);
  const std::string text = serialize("0123456789abcdef", r);
  VCHECK(text.size() < kMaxFileBytes);
}

VTEST(a_record_goes_through_the_file_and_back) {
  Record r;
  update(&r, card_run(0.001, 3, {kernel("alpha", 2, 0.0005, 0, true, false, 4096, true), kernel("beta<int>(int*, float)", 1, 0, 0.002, false, true)}), kLim);
  const std::string key = "00112233445566aa";
  Record back;
  VCHECK(parse(serialize(key, r), key, &back));
  VCHECK_EQ(back.runs, r.runs);
  VCHECK(back.verdict == r.verdict);
  VCHECK_EQ(back.kernels.size(), size_t{2});
  VCHECK(back.kernels[1].name == "beta<int>(int*, float)");   // a name with spaces survives
  VCHECK(back.kernels[1].faulted);
  VCHECK(back.kernels[0].card_ok);
  VCHECK_EQ(back.launches, uint32_t{3});
  VCHECK_EQ(back.threads_total, r.threads_total);
  VCHECK_EQ(back.kernels[0].max_threads, r.kernels[0].max_threads);
  VCHECK(back.kernels[0].loop && !back.kernels[1].loop);
}

VTEST(a_hit_reads_what_a_miss_does_not) {
  const std::string dir = fresh_dir();
  VCHECK(!dir.empty());
  const std::string key = "abcdef0123456789";
  Record r;
  VCHECK(!load(dir, key, &r));   // a miss: no file
  update(&r, card_run(0.001, 3, {kernel("k", 3, 0.001, 0, true, false)}), kLim);
  VCHECK(store(dir, key, r));
  Record back;
  VCHECK(load(dir, key, &back));
  VCHECK(back.verdict == Verdict::Candidate);
  Record other;
  VCHECK(!load(dir, "0000000000000000", &other));   // another key: a miss
  remove_dir(dir);
}

VTEST(a_missing_directory_is_made_and_an_unwritable_one_is_no_history) {
  const std::string dir = fresh_dir();
  const std::string deep = dir + "/a/b/c";
  Record r;
  r.runs = 1;
  VCHECK(store(deep, "abcdef0123456789", r));
  Record back;
  VCHECK(load(deep, "abcdef0123456789", &back));
  VCHECK(!store("", "abcdef0123456789", r));
  spit(dir + "/file", "x");
  VCHECK(!store(dir + "/file/sub", "abcdef0123456789", r));   // a path under a regular file
  remove_dir(dir);
}

VTEST(a_corrupt_file_is_a_miss_in_every_way_it_can_be_corrupt) {
  const std::string dir = fresh_dir();
  const std::string key = "abcdef0123456789";
  Record r;
  update(&r, card_run(0.001, 3, {kernel("k", 3, 0.001, 0, true, false)}), kLim);
  const std::string good = serialize(key, r);
  Record out;
  VCHECK(parse(good, key, &out));
  // truncated at every length but the whole
  for (size_t n = 0; n < good.size(); ++n) VCHECK(!parse(good.substr(0, n), key, &out));
  // every single byte flipped
  for (size_t i = 0; i < good.size(); ++i) {
    std::string bad = good;
    bad[i] = static_cast<char>(bad[i] ^ 0x01);
    VCHECK(!parse(bad, key, &out));
  }
  // garbage, empty, huge, a directory in its place, a symlink in its place
  VCHECK(!parse("", key, &out));
  VCHECK(!parse(std::string(100, 'x'), key, &out));
  VCHECK(!parse(std::string(kMaxFileBytes + 1, 'x'), key, &out));
  VCHECK(!parse(good, "0000000000000000", &out));   // the key inside is not the key asked for (a renamed file)
  spit(dir + "/" + key + ".hist", good.substr(0, good.size() / 2));
  VCHECK(!load(dir, key, &out));
  spit(dir + "/" + key + ".hist", std::string(kMaxFileBytes + 10, 'a'));
  VCHECK(!load(dir, key, &out));
  ::unlink((dir + "/" + key + ".hist").c_str());
  ::mkdir((dir + "/" + key + ".hist").c_str(), 0755);
  VCHECK(!load(dir, key, &out));
  ::rmdir((dir + "/" + key + ".hist").c_str());
  VCHECK(::symlink("/etc/passwd", (dir + "/" + key + ".hist").c_str()) == 0);
  VCHECK(!load(dir, key, &out));   // O_NOFOLLOW
  // and a good write over the bad file puts things right
  ::unlink((dir + "/" + key + ".hist").c_str());
  spit(dir + "/" + key + ".hist", "junk");
  VCHECK(store(dir, key, r));
  VCHECK(load(dir, key, &out));
  remove_dir(dir);
}

VTEST(a_checksummed_file_with_a_malformed_field_is_a_miss) {
  const std::string key = "abcdef0123456789";
  Record out;
  // A file whose checksum is right but whose content is not what a writer makes: rebuilt with the same checksum rule.
  const auto sealed = [&](const std::string& body) {
    Hasher h;
    h.text(body);
    return body + "sum " + hex16(h.h) + "\n";
  };
  const std::string head = "vgpu-host-gpu-history " + std::to_string(kVersion) + "\nkey " + key + "\nruns 1\nverdict 2\ntotals 0.01 0.02 3 99 5\n";
  VCHECK(parse(sealed(head), key, &out));
  VCHECK(!parse(sealed(head + "kernel 1 5 0 0 2 0 0 name\n"), key, &out));        // card_ok is 0 or 1
  VCHECK(!parse(sealed(head + "kernel 1 5 -1 0 1 0 0 name\n"), key, &out));       // negative seconds
  VCHECK(!parse(sealed(head + "kernel 1 5 nan 0 1 0 0 name\n"), key, &out));
  VCHECK(!parse(sealed(head + "kernel 1 5 0 0 1 0 0\n"), key, &out));             // no name
  VCHECK(!parse(sealed(head + "mystery 1\n"), key, &out));                    // a line no version of the writer makes
  VCHECK(!parse(sealed("vgpu-host-gpu-history 1\nkey " + key + "\nruns 1\nverdict 9\ntotals 0 0 0 0 0\n"), key, &out));
  VCHECK(!parse(sealed("vgpu-host-gpu-history 1\nkey " + key + "\nruns x\nverdict 1\ntotals 0 0 0 0 0\n"), key, &out));
  VCHECK(!parse(sealed("vgpu-host-gpu-history 1\nkey " + key + "\nverdict 1\ntotals 0 0 0 0 0\n"), key, &out));   // runs missing
}

VTEST(a_file_of_another_version_is_a_miss_and_is_overwritten) {
  const std::string dir = fresh_dir();
  const std::string key = "abcdef0123456789";
  Record r;
  update(&r, card_run(0.001, 3, {kernel("k", 3, 0.001, 0, true, false)}), kLim);
  // what the next version of the writer would leave: the same file with its version number and a matching checksum
  std::string text = serialize(key, r);
  const std::string from = "vgpu-host-gpu-history " + std::to_string(kVersion) + "\n";
  const std::string to = "vgpu-host-gpu-history " + std::to_string(kVersion + 1) + "\n";
  std::string body = text.substr(0, text.rfind("sum "));
  body.replace(body.find(from), from.size(), to);
  Hasher h;
  h.text(body);
  spit(dir + "/" + key + ".hist", body + "sum " + hex16(h.h) + "\n");
  Record out;
  VCHECK(!load(dir, key, &out));
  VCHECK(store(dir, key, r));
  VCHECK(load(dir, key, &out));
  VCHECK(slurp(dir + "/" + key + ".hist").compare(0, from.size(), from) == 0);
  remove_dir(dir);
}

VTEST(writers_at_the_same_time_leave_one_whole_record_and_readers_never_see_half_of_one) {
  const std::string dir = fresh_dir();
  const std::string key = "abcdef0123456789";
  std::atomic<bool> stop{false};
  std::atomic<int> bad{0}, hits{0};
  std::vector<std::thread> ts;
  for (int w = 0; w < 4; ++w)
    ts.emplace_back([&, w] {
      for (int i = 0; i < 150; ++i) {
        Record r;
        r.runs = static_cast<uint64_t>(w * 1000 + i);
        r.verdict = Verdict::Candidate;
        for (int k = 0; k < 20; ++k) r.kernels.push_back(kernel(("kernel_number_" + std::to_string(k) + "_of_writer_" + std::to_string(w)).c_str(), 1, 0.001, 0.002, true, false));
        store(dir, key, r);
      }
    });
  std::thread reader([&] {
    while (!stop.load()) {
      Record r;
      if (load(dir, key, &r)) {
        ++hits;
        // a whole record of one writer: all 20 kernels, all named for the same writer
        const std::string tag = "_of_writer_" + std::to_string(r.runs / 1000);
        bool whole = r.kernels.size() == 20;
        for (const Kernel& k : r.kernels) whole = whole && k.name.size() > tag.size() && k.name.compare(k.name.size() - tag.size(), tag.size(), tag) == 0;
        if (!whole) ++bad;
      }
    }
  });
  for (auto& t : ts) t.join();
  stop = true;
  reader.join();
  VCHECK_EQ(bad.load(), 0);
  Record r;
  VCHECK(load(dir, key, &r));
  // no temp file is left behind
  size_t tmp = 0;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (const dirent* e = ::readdir(d)) tmp += std::string(e->d_name).find(".tmp.") != std::string::npos;
    ::closedir(d);
  }
  VCHECK_EQ(tmp, size_t{0});
  remove_dir(dir);
}

VTEST(the_directory_stays_bounded) {
  const std::string dir = fresh_dir();
  Record r;
  r.runs = 1;
  for (size_t i = 0; i < kMaxFiles + 40; ++i) {
    char key[17];
    std::snprintf(key, sizeof key, "%016zx", i + 1);
    VCHECK(store(dir, key, r));
  }
  size_t n = 0;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (const dirent* e = ::readdir(d)) n += std::string(e->d_name).size() == 21;
    ::closedir(d);
  }
  VCHECK(n <= kMaxFiles);
  VCHECK(n > 0);
  remove_dir(dir);
}

VTEST(a_stale_temp_file_is_cleaned_and_a_fresh_one_is_not_touched) {
  const std::string dir = fresh_dir();
  const std::string key = "abcdef0123456789";
  Record r;
  r.runs = 1;
  const std::string old_tmp = dir + "/" + key + ".hist.tmp.1.0", new_tmp = dir + "/" + key + ".hist.tmp.2.0";
  spit(old_tmp, "x");
  spit(new_tmp, "x");
  struct timespec ts[2] = {{::time(nullptr) - 7200, 0}, {::time(nullptr) - 7200, 0}};
  VCHECK(::utimensat(AT_FDCWD, old_tmp.c_str(), ts, 0) == 0);
  VCHECK(store(dir, key, r));
  VCHECK(::access(old_tmp.c_str(), F_OK) != 0);
  VCHECK(::access(new_tmp.c_str(), F_OK) == 0);
  remove_dir(dir);
}

VTEST(the_directory_comes_from_the_variable) {
  VCHECK(directory_from(nullptr, nullptr, "/home/u").empty());
  VCHECK(directory_from("", nullptr, "/home/u").empty());
  VCHECK(directory_from("0", nullptr, "/home/u").empty());
  VCHECK(directory_from("/var/h", nullptr, nullptr) == "/var/h");
  VCHECK(directory_from("1", nullptr, "/home/u") == "/home/u/.cache/pantheonsim/host-gpu-history");
  VCHECK(directory_from("1", "/xdg", "/home/u") == "/xdg/pantheonsim/host-gpu-history");
  VCHECK(directory_from("1", nullptr, nullptr).empty());
}

VTEST(the_hash_depends_on_every_byte_and_on_the_length) {
  const std::string a(1000, 'a');
  Hasher h1, h2, h3, h4;
  h1.bytes(a.data(), a.size());
  std::string b = a;
  b[999] = 'b';
  h2.bytes(b.data(), b.size());
  std::string c = a;
  c[3] = 'b';
  h3.bytes(c.data(), c.size());
  h4.bytes(a.data(), 999);
  VCHECK(h1.h != h2.h);
  VCHECK(h1.h != h3.h);
  VCHECK(h1.h != h4.h);
  Hasher h5;
  h5.bytes(a.data(), a.size());
  VCHECK_EQ(h1.h, h5.h);
}

VTEST_MAIN
