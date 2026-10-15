// Unit tests for the settings file: its parser, the key table and what
// `vgpu config` prints (src/cli/config.cpp).
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

#include "config.hpp"
#include "vgpu/telemetry.hpp"
#include "vtest.hpp"

namespace cfg = vgpu::cli::config;
namespace fs = std::filesystem;

namespace {

const std::vector<std::string> kGpus = {"nvidia/h100", "nvidia/t4", "nvidia/a100", "amd/mi300x"};

// The message of the ConfigError that `fn` throws, or "" when it throws nothing.
template <class F>
std::string error_of(F&& fn) {
  try {
    fn();
  } catch (const cfg::ConfigError& e) {
    return e.what();
  }
  return "";
}

std::string problem(const std::string& text) {
  return error_of([&] { cfg::validate(cfg::parse(text, "f.yaml"), kGpus); });
}

cfg::GetEnv env_of(const std::map<std::string, std::string>& m) {
  return [&m](const char* name) -> const char* {
    auto it = m.find(name);
    return it == m.end() ? nullptr : it->second.c_str();
  };
}

// A fresh directory that is removed at the end of the test.
struct TempDir {
  fs::path path;
  TempDir() {
    std::string pattern = (fs::temp_directory_path() / "vgpu-config-XXXXXX").string();
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) throw std::runtime_error("mkdtemp failed");
    path = buf.data();
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  std::string file(const std::string& name, const std::string& text) const {
    const fs::path p = path / name;
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
    return p.string();
  }
};

struct Chdir {
  fs::path old = fs::current_path();
  explicit Chdir(const fs::path& to) { fs::current_path(to); }
  ~Chdir() { fs::current_path(old); }
};

}  // namespace

VTEST(a_valid_file_with_comments) {
  const auto f = cfg::parse(
      "# a header\n"
      "\n"
      "gpu: nvidia/t4        # trailing comment\n"
      "count: 4\n"
      "memory_dir: \"/var/tmp/with # hash\"\n"
      "  \n"
      "strict: true\r\n"
      "scheduler: 'random'\n",
      "f.yaml");
  VCHECK_EQ(f.entries.size(), size_t{5});
  VCHECK_EQ(f.find("gpu")->value, "nvidia/t4");
  VCHECK_EQ(f.find("gpu")->line, 3);
  VCHECK_EQ(f.find("count")->value, "4");
  VCHECK_EQ(f.find("memory_dir")->value, "/var/tmp/with # hash");
  VCHECK_EQ(f.find("strict")->value, "true");
  VCHECK_EQ(f.find("scheduler")->value, "random");
  VCHECK_EQ(f.find("scheduler")->line, 8);
  cfg::validate(f, kGpus);  // does not throw
  VCHECK(f.find("race") == nullptr);
}

VTEST(an_empty_file_and_a_comment_only_file_are_valid) {
  VCHECK_EQ(cfg::parse("", "f").entries.size(), size_t{0});
  VCHECK_EQ(cfg::parse("# nothing\n\n", "f").entries.size(), size_t{0});
  VCHECK_EQ(cfg::parse("\xEF\xBB\xBF" "gpu: nvidia/t4\n", "f").entries.size(), size_t{1});  // a BOM
}

VTEST(an_unknown_key_names_the_key_and_the_closest_valid_one) {
  std::string e = problem("gpuu: nvidia/t4\n");
  VCHECK_CONTAINS(e, "f.yaml:1:");
  VCHECK_CONTAINS(e, "unknown key 'gpuu'");
  VCHECK_CONTAINS(e, "did you mean 'gpu'?");
  VCHECK_CONTAINS(problem("count: 1\ntread: 2\n"), "did you mean 'threads'?");
  VCHECK_CONTAINS(problem("count: 1\ntread: 2\n"), "f.yaml:2:");
  VCHECK_CONTAINS(problem("vram: 100\n"), "did you mean 'vram_mb'?");
  VCHECK_CONTAINS(problem("vram-mb: 100\n"), "did you mean 'vram_mb'?");
  VCHECK_CONTAINS(problem("memory_dirr: x\n"), "did you mean 'memory_dir'?");
  VCHECK_CONTAINS(problem("scheduler_sed: 3\n"), "did you mean 'scheduler_seed'?");
  // Nothing close: no guess, but the way to the list.
  e = problem("banana: 1\n");
  VCHECK_CONTAINS(e, "unknown key 'banana'");
  VCHECK_CONTAINS(e, "vgpu config keys");
  VCHECK(e.find("did you mean") == std::string::npos);
}

VTEST(wrong_types_and_ranges_say_what_is_allowed) {
  std::string e = problem("count: abc\n");
  VCHECK_CONTAINS(e, "f.yaml:1: count:");
  VCHECK_CONTAINS(e, "a whole number from 1 to 16");
  VCHECK_CONTAINS(e, "got 'abc'");
  VCHECK_CONTAINS(problem("count: 0\n"), "a whole number from 1 to 16");
  VCHECK_CONTAINS(problem("count: 17\n"), "a whole number from 1 to 16");
  VCHECK_CONTAINS(problem("count: 1.5\n"), "a whole number");
  VCHECK_CONTAINS(problem("count: -2\n"), "a whole number from 1 to 16");
  VCHECK_CONTAINS(problem("threads: 0\n"), "a whole number from 1 to 65536");
  VCHECK_CONTAINS(problem("vram_mb: 0\n"), "vram_mb:");
  VCHECK_CONTAINS(problem("max_steps: -1\n"), "a whole number of at least 0");
  VCHECK_CONTAINS(problem("strict: maybe\n"), "true or false");
  VCHECK_CONTAINS(problem("host_guard: 2\n"), "true or false");
  VCHECK_CONTAINS(problem("scheduler: fifo\n"), "one of: deterministic, random, adversarial");
  VCHECK_CONTAINS(problem("scheduler: randon\n"), "did you mean 'random'?");
  VCHECK_CONTAINS(problem("race: sometimes\n"), "one of: off, on, all");
  VCHECK_CONTAINS(problem("sass: yes\n"), "one of: auto, off, only");
  VCHECK_CONTAINS(problem("simd: avx512\n"), "one of: auto, base, avx2, off");
  VCHECK_CONTAINS(problem("cuda_version: 13\n"), "major.minor");
  VCHECK_CONTAINS(problem("cuda_version: 13.0.1\n"), "major.minor");
  VCHECK_CONTAINS(problem("rocm_version: six\n"), "a dotted version");
  VCHECK_CONTAINS(problem("driver_version: 580\n"), "a dotted version");
  VCHECK_CONTAINS(problem("os: windows\n"), "ubuntu, rocky or debian");
  VCHECK_CONTAINS(problem("os: ubuntu:two\n"), "ubuntu, rocky or debian");
  // What is allowed passes.
  VCHECK_EQ(problem("count: 16\nthreads: 65536\nmax_steps: 0\ncuda_version: 13.0\nrocm_version: 7.2.0\n"
                    "driver_version: 580.65.06\nos: rocky\nstrict: yes\nrace: all\nsass: only\nsimd: avx2\n"),
            "");
}

VTEST(gpu_is_checked_against_the_known_profiles) {
  VCHECK_EQ(problem("gpu: nvidia/h100\n"), "");
  VCHECK_EQ(problem("gpu: NVIDIA/H100\n"), "");  // profile ids are case-insensitive
  std::string e = problem("gpu: nvidia/h10\n");
  VCHECK_CONTAINS(e, "f.yaml:1: gpu:");
  VCHECK_CONTAINS(e, "'nvidia/h10' is not a GPU profile");
  VCHECK_CONTAINS(e, "did you mean 'nvidia/h100'?");
  VCHECK_CONTAINS(e, "vgpu list-gpus");
  VCHECK_CONTAINS(problem("gpu: h100\n"), "did you mean 'nvidia/h100'?");  // the model without its vendor
  VCHECK_CONTAINS(problem("gpu: mi300x\n"), "did you mean 'amd/mi300x'?");
  e = problem("gpu: quantum/9000\n");
  VCHECK_CONTAINS(e, "not a GPU profile");
  VCHECK(e.find("did you mean") == std::string::npos);
}

VTEST(syntax_errors_name_the_line) {
  VCHECK_CONTAINS(problem("count 4\n"), "f.yaml:1: expected `key: value`");
  VCHECK_CONTAINS(problem("gpu: nvidia/t4\n  count: 4\n"), "f.yaml:2: indented line");
  VCHECK_CONTAINS(problem("count:\n"), "'count' has no value");
  VCHECK_CONTAINS(problem("count: # comment only\n"), "'count' has no value");
  VCHECK_CONTAINS(problem("memory_dir: \"\"\n"), "has no value");
  VCHECK_CONTAINS(problem("memory_dir: \"/tmp/x\n"), "unterminated quote");
  VCHECK_CONTAINS(problem("memory_dir: \"/tmp/x\" junk\n"), "unexpected text after the closing quote");
  VCHECK_CONTAINS(problem("gpu: nvidia/t4\ngpu: nvidia/h100\n"), "f.yaml:2: 'gpu' is set twice (first on line 1)");
  VCHECK_CONTAINS(problem("- gpu: x\n"), "is not a key");
  VCHECK_CONTAINS(problem("a b: 1\n"), "is not a key");
  // A colon inside the value is part of the value.
  const auto f = cfg::parse("os: ubuntu:24.04\n", "f");
  VCHECK_EQ(f.entries[0].value, "ubuntu:24.04");
}

VTEST(the_value_a_variable_gets) {
  const cfg::Key& strict = *cfg::find_key("strict");
  VCHECK_EQ(cfg::env_value(strict, "true"), "1");
  VCHECK_EQ(cfg::env_value(strict, "no"), "");  // false leaves the variable unset
  const cfg::Key& guard = *cfg::find_key("host_guard");
  VCHECK_EQ(cfg::env_value(guard, "false"), "0");
  VCHECK_EQ(cfg::env_value(guard, "true"), "");
  const cfg::Key& race = *cfg::find_key("race");
  VCHECK_EQ(cfg::env_value(race, "on"), "1");
  VCHECK_EQ(cfg::env_value(race, "all"), "2");
  VCHECK_EQ(cfg::env_value(race, "off"), "");
  const cfg::Key& sass = *cfg::find_key("sass");
  VCHECK_EQ(cfg::env_value(sass, "off"), "0");
  VCHECK_EQ(cfg::env_value(sass, "only"), "1");
  VCHECK_EQ(cfg::env_value(sass, "auto"), "");
  VCHECK_EQ(cfg::env_value(*cfg::find_key("ram_mb"), "512"), "512");
  VCHECK_EQ(cfg::find_key("ram_mb")->env, "VGPU_MEMORY_RAM_MB");
  VCHECK_EQ(cfg::find_key("host_reserve_mb")->env, "VGPU_MEMORY_HOST_RESERVE_MB");
  VCHECK_EQ(cfg::find_key("memory_dir")->env, "VGPU_MEMORY_DIR");
  VCHECK_EQ(cfg::find_key("host_guard")->env, "VGPU_MEMORY_HOST_GUARD");
  VCHECK_EQ(cfg::find_key("count")->env, "VGPU_DEVICE_COUNT");
}

VTEST(apply_sets_only_what_is_not_set) {
  const auto f = cfg::parse("gpu: nvidia/t4\ncount: 4\nstrict: true\nquiet: false\nhost_guard: false\nthreads: 2\n", "f");
  std::map<std::string, std::string> env = {{"VGPU_DEVICE_COUNT", "8"}, {"VGPU_THREADS", ""}};
  std::map<std::string, std::string> set;
  const int n = cfg::apply(f, env_of(env), [&](const std::string& k, const std::string& v) { set[k] = v; });
  VCHECK_EQ(set["VGPU_GPU"], "nvidia/t4");
  VCHECK(set.count("VGPU_DEVICE_COUNT") == 0);   // the real environment wins
  VCHECK_EQ(set["VGPU_STRICT"], "1");
  VCHECK(set.count("VGPU_QUIET") == 0);          // false is "not configured"
  VCHECK_EQ(set["VGPU_MEMORY_HOST_GUARD"], "0");
  VCHECK_EQ(set["VGPU_THREADS"], "2");           // an empty variable counts as unset, as the engine reads it
  VCHECK_EQ(n, 4);
}

VTEST(precedence_flag_then_environment_then_file_then_default) {
  const auto f = cfg::parse("gpu: nvidia/t4\ncount: 4\nthreads: 2\n", "p.yaml");
  std::map<std::string, std::string> env = {{"VGPU_DEVICE_COUNT", "8"}, {"VGPU_THREADS", "3"}};
  std::map<std::string, std::string> flags = {{"threads", "5"}};
  auto find = [](const std::vector<cfg::Setting>& s, const std::string& key) -> const cfg::Setting& {
    for (const auto& x : s)
      if (x.key->name == key) return x;
    throw std::runtime_error("no setting " + key);
  };
  const auto s = cfg::effective(&f, flags, env_of(env));
  VCHECK_EQ(s.size(), cfg::keys().size());
  // file beats default
  VCHECK_EQ(find(s, "gpu").value, "nvidia/t4");
  VCHECK(find(s, "gpu").source == cfg::Source::File);
  VCHECK_EQ(find(s, "gpu").detail, "p.yaml:1");
  // environment beats file
  VCHECK_EQ(find(s, "count").value, "8");
  VCHECK(find(s, "count").source == cfg::Source::Environment);
  VCHECK_EQ(find(s, "count").detail, "VGPU_DEVICE_COUNT");
  VCHECK_CONTAINS(find(s, "count").note, "p.yaml:2");
  // flag beats environment and file
  VCHECK_EQ(find(s, "threads").value, "5");
  VCHECK(find(s, "threads").source == cfg::Source::Flag);
  VCHECK_EQ(find(s, "threads").detail, "--threads");
  // nothing set it: the default
  VCHECK(find(s, "quiet").source == cfg::Source::Default);
  VCHECK_EQ(find(s, "quiet").value, "false");
  VCHECK_EQ(find(s, "vram_mb").value, "the profile's own");
}

VTEST(config_show_says_where_each_value_came_from) {
  const auto f = cfg::parse("gpu: nvidia/t4\ncount: 4\n", "/work/pantheonsim.yaml");
  std::map<std::string, std::string> env = {{"VGPU_DEVICE_COUNT", "8"}};
  std::map<std::string, std::string> flags = {{"quiet", "true"}};
  cfg::Located where;
  where.path = f.path;
  const std::string out = cfg::render_show(where, &f, cfg::effective(&f, flags, env_of(env)));
  VCHECK_CONTAINS(out, "settings file: /work/pantheonsim.yaml");
  VCHECK_CONTAINS(out, "file /work/pantheonsim.yaml:1");
  VCHECK_CONTAINS(out, "environment VGPU_DEVICE_COUNT");
  VCHECK_CONTAINS(out, "flag --quiet");
  VCHECK_CONTAINS(out, "default");
  // every key has a row
  for (const auto& k : cfg::keys()) VCHECK_CONTAINS(out, k.name + " ");
  // and with no file, the places that were looked at
  cfg::Located none;
  none.searched = {"pantheonsim.yaml", "/home/u/.config/pantheonsim/config.yaml"};
  const std::string bare = cfg::render_show(none, nullptr, cfg::effective(nullptr, {}, env_of({})));
  VCHECK_CONTAINS(bare, "settings file: none (looked for pantheonsim.yaml, /home/u/.config/pantheonsim/config.yaml)");
  VCHECK(bare.find("file /") == std::string::npos);
}

VTEST(no_file_is_fine_and_a_requested_missing_file_is_not) {
  TempDir dir;
  Chdir cd(dir.path);
  std::map<std::string, std::string> env = {{"HOME", (dir.path / "home").string()}};
  const auto where = cfg::locate("", env_of(env));
  VCHECK_EQ(where.path, "");
  VCHECK_EQ(where.searched.size(), size_t{2});
  VCHECK(!cfg::load(where, kGpus).has_value());
  // named by --config or VGPU_CONFIG: absent is an error that says which
  std::string e = error_of([&] { cfg::load(cfg::locate((dir.path / "nope.yaml").string(), env_of(env)), kGpus); });
  VCHECK_CONTAINS(e, "nope.yaml does not exist");
  VCHECK_CONTAINS(e, "--config");
  env["VGPU_CONFIG"] = (dir.path / "nope2.yaml").string();
  e = error_of([&] { cfg::load(cfg::locate("", env_of(env)), kGpus); });
  VCHECK_CONTAINS(e, "nope2.yaml does not exist");
  VCHECK_CONTAINS(e, "VGPU_CONFIG");
  // a directory is not a file
  e = error_of([&] { cfg::load(cfg::locate(dir.path.string(), env_of({})), kGpus); });
  VCHECK_CONTAINS(e, "is a directory");
}

VTEST(the_search_order_is_flag_then_cwd_then_xdg_then_home) {
  TempDir dir;
  const std::string xdg = dir.file("xdg/pantheonsim/config.yaml", "gpu: nvidia/a100\n");
  const std::string home = dir.file("home/.config/pantheonsim/config.yaml", "gpu: amd/mi300x\n");
  const std::string flag = dir.file("flag.yaml", "gpu: nvidia/t4\n");
  const std::string venv = dir.file("venv.yaml", "gpu: nvidia/h100\n");
  fs::create_directories(dir.path / "work");
  Chdir cd(dir.path / "work");
  std::map<std::string, std::string> env = {{"HOME", (dir.path / "home").string()}};
  auto gpu_of = [&](const std::string& flag_path, const std::map<std::string, std::string>& e) {
    const auto f = cfg::load(cfg::locate(flag_path, env_of(e)), kGpus);
    return f ? f->find("gpu")->value : std::string("none");
  };
  VCHECK_EQ(gpu_of("", env), "amd/mi300x");                  // ~/.config
  env["XDG_CONFIG_HOME"] = (dir.path / "xdg").string();
  VCHECK_EQ(gpu_of("", env), "nvidia/a100");                 // $XDG_CONFIG_HOME wins over ~/.config
  fs::create_directories(dir.path / "work");
  std::ofstream(dir.path / "work" / "pantheonsim.yaml") << "gpu: nvidia/t4\ncount: 2\n";
  VCHECK_EQ(gpu_of("", env), "nvidia/t4");                   // ./pantheonsim.yaml wins over XDG
  env["VGPU_CONFIG"] = venv;
  VCHECK_EQ(gpu_of("", env), "nvidia/h100");                 // VGPU_CONFIG wins over the search
  VCHECK_EQ(gpu_of(flag, env), "nvidia/t4");                 // --config wins over VGPU_CONFIG
  (void)xdg; (void)home;
}

VTEST(a_file_that_exists_but_is_invalid_stops_with_the_message) {
  TempDir dir;
  const std::string bad = dir.file("pantheonsim.yaml", "gpu: nvidia/t4\ncount: lots\n");
  cfg::Located where;
  where.path = bad;
  const std::string e = error_of([&] { cfg::load(where, kGpus); });
  VCHECK_CONTAINS(e, bad + ":2: count:");
  VCHECK_CONTAINS(e, "a whole number from 1 to 16");
}

VTEST(config_init_writes_a_starter_and_refuses_to_overwrite) {
  TempDir dir;
  const std::string path = (dir.path / "sub" / "pantheonsim.yaml").string();  // the parent is created
  VCHECK_EQ(cfg::init_file(path, false), "");
  auto read = [&] { std::ifstream in(path); std::ostringstream s; s << in.rdbuf(); return s.str(); };
  VCHECK_EQ(read(), cfg::starter_text());
  std::ofstream(path) << "gpu: nvidia/t4\n";  // the user edits it
  const std::string e = cfg::init_file(path, false);
  VCHECK_CONTAINS(e, "already exists");
  VCHECK_CONTAINS(e, "--force");
  VCHECK_EQ(read(), "gpu: nvidia/t4\n");       // left alone
  VCHECK_EQ(cfg::init_file(path, true), "");   // --force replaces it
  VCHECK_EQ(read(), cfg::starter_text());
  VCHECK_CONTAINS(cfg::init_file((dir.path / "sub" / "pantheonsim.yaml" / "x").string(), false), "cannot");
}

VTEST(the_starter_file_and_every_example_are_valid) {
  // The starter parses and validates as it is: gpu and count active, the rest commented out.
  const auto starter = cfg::parse(cfg::starter_text(), "starter");
  cfg::validate(starter, kGpus);
  VCHECK_EQ(starter.entries.size(), size_t{2});
  // And with every key's example switched on, still valid.
  std::string all;
  for (const auto& k : cfg::keys()) all += k.name + ": " + k.example + "\n";
  const auto f = cfg::parse(all, "all");
  VCHECK_EQ(f.entries.size(), cfg::keys().size());
  cfg::validate(f, kGpus);
  // Each of them is in the starter, commented out or not.
  for (const auto& k : cfg::keys()) {
    const std::string line = k.name + ": " + k.example + "\n";
    VCHECK_CONTAINS(cfg::starter_text(), line);
  }
}

VTEST(the_key_table_is_consistent) {
  std::set<std::string> names, envs;
  for (const auto& k : cfg::keys()) {
    VCHECK(names.insert(k.name).second);
    VCHECK(envs.insert(k.env).second);
    VCHECK(k.env.rfind("VGPU_", 0) == 0);
    VCHECK(!k.description.empty());
    VCHECK(!k.deflt.empty());
    VCHECK(!cfg::allowed(k).empty());
    VCHECK(k.flag().rfind("--", 0) == 0);
    VCHECK(k.flag().find('_') == std::string::npos);
    if (k.type == cfg::Type::Enum || k.type == cfg::Type::Bool) VCHECK(!k.choices.empty());
  }
  // The keys the brief asked for are there.
  for (const char* want : {"gpu", "count", "vram_mb", "cuda_version", "driver_version", "rocm_version", "ram_mb",
                           "memory_dir", "host_reserve_mb", "host_guard", "threads", "quiet", "strict", "race",
                           "counters", "scheduler", "scheduler_seed", "sass", "simd", "os"})
    VCHECK(cfg::find_key(want) != nullptr);
  // The rack size agrees with the simulator's.
  VCHECK_EQ(cfg::find_key("count")->hi, static_cast<long long>(vgpu::telemetry::kMaxDevices));
  VCHECK_CONTAINS(cfg::render_keys(), "VGPU_MEMORY_HOST_RESERVE_MB");
}

VTEST(closest_finds_typos_and_ignores_nonsense) {
  const std::vector<std::string> c = {"threads", "count", "scheduler", "scheduler_seed"};
  VCHECK_EQ(cfg::closest("thread", c), "threads");
  VCHECK_EQ(cfg::closest("Count", c), "count");
  VCHECK_EQ(cfg::closest("sheduler", c), "scheduler");
  VCHECK_EQ(cfg::closest("xylophone", c), "");
  VCHECK_EQ(cfg::closest("", c), "");
}

VTEST_MAIN
