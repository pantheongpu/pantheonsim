#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "args.hpp"

namespace vgpu::cli::config {
namespace {

namespace fs = std::filesystem;

// The largest rack `vgpu` presents: vgpu::telemetry::kMaxDevices. The unit test
// checks the two agree, so this file stays free of the simulator's headers.
constexpr long long kMaxCount = 16;
constexpr long long kMaxMb = 1ll << 30;

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string trim(const std::string& s) {
  const size_t a = s.find_first_not_of(" \t\r");
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(" \t\r") - a + 1);
}

std::string sq(const std::string& s) { return "'" + s + "'"; }

bool parse_bool(const std::string& v, bool* out) {
  const std::string l = lower(v);
  if (l == "true" || l == "yes" || l == "on") { *out = true; return true; }
  if (l == "false" || l == "no" || l == "off") { *out = false; return true; }
  return false;
}

Key make(const char* name, const char* env, Type type, const char* deflt, const char* example,
         const char* description) {
  Key k;
  k.name = name;
  k.env = env;
  k.type = type;
  k.deflt = deflt;
  k.example = example;
  k.description = description;
  return k;
}

Key integer(const char* name, const char* env, long long lo, long long hi, const char* deflt,
            const char* example, const char* description) {
  Key k = make(name, env, Type::Int, deflt, example, description);
  k.lo = lo;
  k.hi = hi;
  return k;
}

// `on` is what the variable holds for true, `off` for false; "" leaves it unset.
Key boolean(const char* name, const char* env, const char* on, const char* off, const char* deflt,
            const char* example, const char* description) {
  Key k = make(name, env, Type::Bool, deflt, example, description);
  k.choices = {{"true", on}, {"false", off}};
  k.switch_flag = true;
  return k;
}

Key enumerated(const char* name, const char* env, std::vector<Choice> choices, const char* deflt,
               const char* example, const char* description) {
  Key k = make(name, env, Type::Enum, deflt, example, description);
  k.choices = std::move(choices);
  return k;
}

// The one table. Order is the order of the starter file and of `config keys`.
std::vector<Key> build_keys() {
  std::vector<Key> t;
  // Device
  t.push_back(make("gpu", "VGPU_GPU", Type::Gpu, "nvidia/h100", "nvidia/h100",
                   "Virtual GPU model to present; `vgpu list-gpus` lists them."));
  t.back().shell = true;
  t.push_back(integer("count", "VGPU_DEVICE_COUNT", 1, kMaxCount, "1", "1",
                      "How many identical devices to present."));
  t.back().shell = true;
  t.push_back(integer("vram_mb", "VGPU_VRAM_MB", 1, kMaxMb, "the profile's own", "16384",
                      "Memory per device in MiB, instead of the profile's."));
  t.back().shell = true;
  t.push_back(make("cuda_version", "VGPU_CUDA_VERSION", Type::CudaVersion, "this build's", "13.0",
                   "CUDA version that nvidia-smi and NVML report (the compiler is not changed)."));
  t.back().shell = true;
  t.push_back(make("driver_version", "VGPU_DRIVER_VERSION", Type::Version, "this build's", "580.65.06",
                   "NVIDIA driver version that nvidia-smi and NVML report."));
  t.back().shell = true;
  t.push_back(make("rocm_version", "VGPU_ROCM_VERSION", Type::Version, "this build's", "6.4.0",
                   "ROCm version that rocm-smi and amd-smi report (the compiler is not changed)."));
  t.back().shell = true;
  t.push_back(make("os", "VGPU_SHELL_OS", Type::Os, "ubuntu:22.04", "ubuntu:24.04",
                   "Operating system `vgpu shell` presents: ubuntu, rocky or debian, with a version."));
  t.back().shell = true;
  // Memory
  t.push_back(integer("ram_mb", "VGPU_MEMORY_RAM_MB", 1, kMaxMb, "unlimited", "8192",
                      "Host RAM in MiB that device memory may use before the rest goes to memory_dir."));
  t.push_back(make("memory_dir", "VGPU_MEMORY_DIR", Type::Text, "never spill", "/var/tmp/pantheonsim",
                   "Directory for device memory past ram_mb, so a card larger than the machine runs, at disk speed."));
  t.push_back(integer("host_reserve_mb", "VGPU_MEMORY_HOST_RESERVE_MB", 0, kMaxMb,
                      "6% of the memory limit, 1 to 4 GiB", "2048",
                      "Host memory in MiB the guard leaves to the rest of the machine."));
  t.push_back(boolean("host_guard", "VGPU_MEMORY_HOST_GUARD", "", "0", "true", "true",
                      "false turns the host-memory guard off."));
  t.back().switch_flag = false;
  // Execution
  t.push_back(integer("threads", "VGPU_THREADS", 1, 1 << 16, "one per core", "4",
                      "Host threads that run blocks."));
  t.push_back(integer("max_steps", "VGPU_MAX_STEPS", 0, LLONG_MAX, "built in", "0",
                      "Per-thread instruction budget against a kernel that never ends; 0 removes it."));
  t.push_back(enumerated("sass", "VGPU_SASS", {{"auto", ""}, {"off", "0"}, {"only", "1"}}, "auto", "auto",
                         "Engine for NVIDIA kernels: auto uses SASS when the binary has it, off forces PTX, only insists on SASS."));
  t.push_back(enumerated("simd", "VGPU_SIMD", {{"auto", ""}, {"base", "base"}, {"avx2", "avx2"}, {"off", "off"}},
                         "auto", "auto",
                         "Caps the vector instruction set the engines use; off uses scalar code. For ruling them out."));
  t.push_back(enumerated("scheduler", "VGPU_SCHEDULER",
                         {{"deterministic", "deterministic"}, {"random", "random"}, {"adversarial", "adversarial"}},
                         "deterministic", "random",
                         "Warp scheduling order. random and adversarial find order bugs (PTX engine only: set sass: off)."));
  t.push_back(integer("scheduler_seed", "VGPU_SCHEDULER_SEED", 0, LLONG_MAX, "0", "1",
                      "Seed for random and adversarial scheduling; the same seed replays the same run."));
  // Checks and output
  t.push_back(boolean("strict", "VGPU_STRICT", "1", "", "false", "false",
                      "Refuse undefined behaviour (integer division by zero, storing an unwritten register)."));
  t.back().switch_flag = true;
  t.push_back(enumerated("race", "VGPU_RACE", {{"off", ""}, {"on", "1"}, {"all", "2"}}, "off", "off",
                         "Report unordered shared-memory access between warps; all also reports stores that change nothing."));
  t.back().switch_flag = true;
  t.push_back(boolean("counters", "VGPU_COUNTERS", "1", "", "false", "false",
                      "Print exact per-launch performance counters."));
  t.push_back(boolean("quiet", "VGPU_QUIET", "1", "", "false", "false",
                      "Silence every simulator message on stderr, errors included."));
  t.push_back(boolean("trace", "VGPU_TRACE", "1", "", "false", "false",
                      "Log every runtime and driver entry point."));
  return t;
}

size_t edit_distance(const std::string& a, const std::string& b) {
  std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
  for (size_t i = 1; i <= a.size(); ++i) {
    cur[0] = i;
    for (size_t j = 1; j <= b.size(); ++j)
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
    std::swap(prev, cur);
  }
  return prev[b.size()];
}

const Choice* find_choice(const Key& key, const std::string& value) {
  const std::string l = lower(value);
  for (const auto& c : key.choices)
    if (c.word == l) return &c;
  return nullptr;
}

std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string out;
  for (size_t i = 0; i < v.size(); ++i) out += (i ? sep : "") + v[i];
  return out;
}

std::string key_list() {
  std::vector<std::string> names;
  for (const auto& k : keys()) names.push_back(k.name);
  return join(names, ", ");
}

}  // namespace

std::string Key::flag() const {
  std::string f = "--" + name;
  std::replace(f.begin(), f.end(), '_', '-');
  return f;
}

const std::vector<Key>& keys() {
  static const std::vector<Key> table = build_keys();
  return table;
}

const Key* find_key(const std::string& name) {
  for (const auto& k : keys())
    if (k.name == name) return &k;
  return nullptr;
}

std::string closest(const std::string& word, const std::vector<std::string>& candidates) {
  std::string w = lower(word);
  std::replace(w.begin(), w.end(), '-', '_');
  std::string best;
  size_t best_d = SIZE_MAX;
  for (const auto& c : candidates) {
    const std::string lc = lower(c);
    size_t d = edit_distance(w, lc);
    // A word that is the start of a name ("vram" for vram_mb) is as good as one edit.
    if (w.size() >= 3 && lc.rfind(w, 0) == 0) d = std::min<size_t>(d, 1);
    if (d < best_d) { best_d = d; best = c; }
  }
  const size_t limit = std::max<size_t>(2, w.size() / 3);
  return best_d <= limit ? best : "";
}

std::string allowed(const Key& key) {
  switch (key.type) {
    case Type::Int:
      if (key.hi == LLONG_MAX) return "a whole number of at least " + std::to_string(key.lo);
      return "a whole number from " + std::to_string(key.lo) + " to " + std::to_string(key.hi);
    case Type::Bool: return "true or false";
    case Type::Enum: {
      std::vector<std::string> w;
      for (const auto& c : key.choices) w.push_back(c.word);
      return "one of: " + join(w, ", ");
    }
    case Type::Text: return "a path or other text";
    case Type::Gpu: return "a GPU profile id such as nvidia/h100 (vgpu list-gpus)";
    case Type::CudaVersion: return "a CUDA version as major.minor, such as 13.0";
    case Type::Version: return "a dotted version such as " + key.example;
    case Type::Os: return "ubuntu, rocky or debian, optionally with a version (ubuntu:24.04)";
  }
  return "";
}

std::string check_value(const Key& key, const std::string& value, const std::vector<std::string>& gpus) {
  const std::string want = "expected " + allowed(key) + ", got " + sq(value);
  switch (key.type) {
    case Type::Int: {
      long long n = 0;
      return vgpu::cli::parse_int(value, key.lo, key.hi, &n) ? "" : want;
    }
    case Type::Bool: {
      bool b = false;
      return parse_bool(value, &b) ? "" : want;
    }
    case Type::Enum: {
      if (find_choice(key, value)) return "";
      std::vector<std::string> w;
      for (const auto& c : key.choices) w.push_back(c.word);
      const std::string near = closest(value, w);
      return want + (near.empty() ? "" : "; did you mean " + sq(near) + "?");
    }
    case Type::Text:
      return value.empty() || value.find_first_of("\n\r") != std::string::npos ? want : "";
    case Type::Gpu: {
      if (gpus.empty()) return "";
      const std::string l = lower(value);
      for (const auto& g : gpus)
        if (lower(g) == l) return "";
      std::string hint = closest(value, gpus);
      if (hint.empty() && l.find('/') == std::string::npos) {
        // A model without its vendor: h100 for nvidia/h100.
        for (const auto& g : gpus) {
          const size_t slash = g.find('/');
          if (slash != std::string::npos && lower(g.substr(slash + 1)) == l) { hint = g; break; }
        }
      }
      return sq(value) + " is not a GPU profile" +
             (hint.empty() ? "" : "; did you mean " + sq(hint) + "?") +
             " (`vgpu list-gpus` prints them)";
    }
    case Type::CudaVersion: {
      const size_t dot = value.find('.');
      long long a = 0, b = 0;
      const bool ok = dot != std::string::npos && value.find('.', dot + 1) == std::string::npos &&
                      vgpu::cli::parse_int(value.substr(0, dot), 1, 99, &a) &&
                      vgpu::cli::parse_int(value.substr(dot + 1), 0, 99, &b) &&
                      value.find('-') == std::string::npos;
      return ok ? "" : want;
    }
    case Type::Version: return vgpu::cli::is_dotted_version(value, 2) ? "" : want;
    case Type::Os: {
      // The ids `vgpu shell --os` offers (kOsChoices in shell.cpp); the shell
      // checks the version against its list when it starts.
      const size_t colon = value.find(':');
      const std::string id = value.substr(0, colon);
      const bool id_ok = id == "ubuntu" || id == "rocky" || id == "debian";
      const bool ver_ok = colon == std::string::npos || vgpu::cli::is_dotted_version(value.substr(colon + 1), 1);
      return id_ok && ver_ok ? "" : want;
    }
  }
  return want;
}

std::string env_value(const Key& key, const std::string& value) {
  switch (key.type) {
    case Type::Bool: {
      bool b = false;
      parse_bool(value, &b);
      return key.choices[b ? 0 : 1].env;
    }
    case Type::Enum: {
      const Choice* c = find_choice(key, value);
      return c ? c->env : "";
    }
    default: return value;
  }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

const Entry* File::find(const std::string& key) const {
  for (const auto& e : entries)
    if (e.key == key) return &e;
  return nullptr;
}

File parse(const std::string& text_in, const std::string& origin) {
  File f;
  f.path = origin;
  std::string text = text_in;
  if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) text.erase(0, 3);  // a BOM from an editor
  std::vector<std::string> names;
  for (const auto& k : keys()) names.push_back(k.name);

  size_t pos = 0;
  int line = 0;
  while (pos < text.size()) {
    const size_t nl = text.find('\n', pos);
    std::string raw = text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    pos = nl == std::string::npos ? text.size() : nl + 1;
    ++line;
    auto fail = [&](const std::string& msg) {
      throw ConfigError(origin + ":" + std::to_string(line) + ": " + msg);
    };
    const std::string t = trim(raw);
    if (t.empty() || t[0] == '#') continue;
    if (raw[0] == ' ' || raw[0] == '\t')
      fail("indented line; the file is a flat list of `key: value` lines (no nesting)");
    const size_t colon = raw.find(':');
    if (colon == std::string::npos) fail("expected `key: value`, got " + sq(t));
    const std::string key = trim(raw.substr(0, colon));
    if (key.empty() || key[0] == '-' || key.find_first_of(" \t\"'#[{") != std::string::npos)
      fail(sq(key) + " is not a key; expected `key: value`");

    std::string v = trim(raw.substr(colon + 1));
    std::string value;
    if (!v.empty() && (v[0] == '"' || v[0] == '\'')) {
      const size_t end = v.find(v[0], 1);
      if (end == std::string::npos) fail("unterminated quote in the value of " + sq(key));
      const std::string rest = trim(v.substr(end + 1));
      if (!rest.empty() && rest[0] != '#') fail("unexpected text after the closing quote of " + sq(key));
      value = v.substr(1, end - 1);
    } else {
      for (size_t i = 0; i < v.size(); ++i)
        if (v[i] == '#' && (i == 0 || v[i - 1] == ' ' || v[i - 1] == '\t')) { v.erase(i); break; }
      value = trim(v);
    }
    if (value.empty())
      fail(sq(key) + " has no value (write `" + key + ": value`, or delete the line to use the default)");

    if (!find_key(key)) {
      const std::string near = closest(key, names);
      fail("unknown key " + sq(key) +
           (near.empty() ? "; `vgpu config keys` lists the valid ones (" + key_list() + ")"
                         : "; did you mean " + sq(near) + "?"));
    }
    if (const Entry* dup = f.find(key))
      fail(sq(key) + " is set twice (first on line " + std::to_string(dup->line) + ")");
    f.entries.push_back({key, value, line});
  }
  return f;
}

void validate(const File& file, const std::vector<std::string>& gpus) {
  for (const auto& e : file.entries) {
    const Key* k = find_key(e.key);
    if (!k) throw ConfigError(file.path + ":" + std::to_string(e.line) + ": unknown key " + sq(e.key));
    const std::string why = check_value(*k, e.value, gpus);
    if (!why.empty()) throw ConfigError(file.path + ":" + std::to_string(e.line) + ": " + e.key + ": " + why);
  }
}

// ---------------------------------------------------------------------------
// Finding and applying
// ---------------------------------------------------------------------------

Located locate(const std::string& flag_path, const GetEnv& env) {
  Located loc;
  std::string named = flag_path;
  loc.requested_by = flag_path.empty() ? "" : "--config";
  if (named.empty()) {
    if (const char* e = env("VGPU_CONFIG"); e && *e) {
      named = e;
      loc.requested_by = "VGPU_CONFIG";
    }
  }
  if (!named.empty()) {
    loc.path = named;
    loc.searched = {named};
    return loc;
  }
  loc.searched.push_back("pantheonsim.yaml");
  const char* xdg = env("XDG_CONFIG_HOME");
  const char* home = env("HOME");
  if (xdg && *xdg) loc.searched.push_back(std::string(xdg) + "/pantheonsim/config.yaml");
  else if (home && *home) loc.searched.push_back(std::string(home) + "/.config/pantheonsim/config.yaml");
  for (const auto& cand : loc.searched) {
    std::error_code ec;
    if (fs::exists(cand, ec)) {
      loc.path = cand;
      break;
    }
  }
  return loc;
}

std::optional<File> load(const Located& where, const std::vector<std::string>& gpus) {
  if (where.path.empty()) return std::nullopt;
  std::error_code ec;
  if (!fs::exists(where.path, ec))
    throw ConfigError("config file " + where.path + " does not exist (named by " + where.requested_by + ")");
  if (fs::is_directory(where.path, ec))
    throw ConfigError("config file " + where.path + " is a directory, not a file");
  std::ifstream in(where.path, std::ios::binary);
  if (!in) throw ConfigError("cannot read config file " + where.path + ": " + std::strerror(errno));
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) throw ConfigError("error reading config file " + where.path);
  File f = parse(ss.str(), where.path);
  validate(f, gpus);
  return f;
}

int apply(const File& file, const GetEnv& env, const SetEnv& set) {
  int n = 0;
  for (const auto& e : file.entries) {
    const Key* k = find_key(e.key);
    if (!k) continue;
    const std::string v = env_value(*k, e.value);
    if (v.empty()) continue;
    const char* cur = env(k->env.c_str());
    if (cur && *cur) continue;  // the engine reads an empty variable as unset, so does this
    set(k->env, v);
    ++n;
  }
  return n;
}

// ---------------------------------------------------------------------------
// `vgpu config`
// ---------------------------------------------------------------------------

std::vector<Setting> effective(const File* file, const std::map<std::string, std::string>& flags,
                               const GetEnv& env) {
  std::vector<Setting> out;
  for (const auto& k : keys()) {
    Setting s;
    s.key = &k;
    const Entry* e = file ? file->find(k.name) : nullptr;
    const char* ev = env(k.env.c_str());
    if (auto it = flags.find(k.name); it != flags.end()) {
      s.value = it->second;
      s.source = Source::Flag;
      s.detail = k.flag();
    } else if (ev && *ev) {
      s.value = ev;
      s.source = Source::Environment;
      s.detail = k.env;
    } else if (e) {
      s.value = e->value;
      s.source = Source::File;
      s.detail = file->path + ":" + std::to_string(e->line);
    } else {
      s.value = k.deflt;
    }
    if (e && s.source != Source::File && s.source != Source::Default)
      s.note = file->path + ":" + std::to_string(e->line) + " says " + e->value;
    out.push_back(std::move(s));
  }
  return out;
}

std::string render_show(const Located& where, const File* file, const std::vector<Setting>& settings) {
  std::ostringstream os;
  if (file) {
    os << "settings file: " << file->path << "\n";
  } else {
    os << "settings file: none (looked for " << join(where.searched, ", ") << ")\n";
  }
  os << "precedence:    flag > environment > file > default\n\n";
  size_t kw = 3, vw = 5;
  for (const auto& s : settings) {
    kw = std::max(kw, s.key->name.size());
    vw = std::max(vw, s.value.size());
  }
  char buf[512];
  std::snprintf(buf, sizeof buf, "%-*s  %-*s  %s\n", static_cast<int>(kw), "key", static_cast<int>(vw), "value", "from");
  os << buf;
  for (const auto& s : settings) {
    std::string from;
    switch (s.source) {
      case Source::Default: from = "default"; break;
      case Source::File: from = "file " + s.detail; break;
      case Source::Environment: from = "environment " + s.detail; break;
      case Source::Flag: from = "flag " + s.detail; break;
    }
    if (!s.note.empty()) from += " (" + s.note + ")";
    std::snprintf(buf, sizeof buf, "%-*s  %-*s  %s\n", static_cast<int>(kw), s.key->name.c_str(),
                  static_cast<int>(vw), s.value.c_str(), from.c_str());
    os << buf;
  }
  return os.str();
}

std::string render_keys() {
  std::ostringstream os;
  for (const auto& k : keys()) {
    os << k.name << ": " << allowed(k) << "\n";
    os << "    " << k.description << "\n";
    os << "    sets " << k.env << "; default: " << k.deflt << (k.shell ? "; `vgpu shell` default" : "") << "\n";
  }
  return os.str();
}

std::string starter_text() {
  std::ostringstream os;
  os << "# PantheonSim settings. `vgpu` reads this file from the current directory\n"
        "# (or --config PATH, or $VGPU_CONFIG, or ~/.config/pantheonsim/config.yaml).\n"
        "# A flag beats an environment variable, which beats this file.\n"
        "# `vgpu config show` prints what is in effect and where each value came from.\n"
        "# Remove the leading # from a line to use it. docs/config.md explains every key.\n\n";
  for (const auto& k : keys()) {
    const bool active = k.name == "gpu" || k.name == "count";
    os << "# " << k.description << "\n";
    os << "# " << allowed(k) << "; default: " << k.deflt << "\n";
    os << (active ? "" : "# ") << k.name << ": " << k.example << "\n\n";
  }
  return os.str();
}

std::string init_file(const std::string& path, bool force) {
  std::error_code ec;
  if (!force && fs::exists(path, ec))
    return path + " already exists; leaving it alone (use --force to overwrite it)";
  const fs::path parent = fs::path(path).parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent, ec);
    if (ec) return "cannot create " + parent.string() + ": " + ec.message();
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return "cannot write " + path + ": " + std::strerror(errno);
  out << starter_text();
  out.close();
  if (!out) return "error writing " + path;
  return "";
}

}  // namespace vgpu::cli::config
