// `vgpu config` and the start-up step that applies the settings file.
//
// config_prepare() runs in main() before a command does. It strips a leading
// `--config PATH`, finds and reads the file, and sets each VGPU_* variable the
// file names that is not set already, so the command, the simulator engine and
// the shim libraries read the file's values the way they read the environment.
// For `vgpu shell`, which does not read the environment for its defaults, it
// also puts the matching flags in front of the user's own, so a flag still wins.
// The format, the keys and their checks are in config.cpp.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "config.hpp"
#include "vgpu/registry.hpp"

namespace {

namespace cfg = vgpu::cli::config;

std::string g_flag_path;  // the global --config, for `vgpu config show`

const char* real_getenv(const char* name) { return std::getenv(name); }

// The flag `vgpu shell` uses for each key it takes as a default.
const char* shell_flag(const std::string& key) {
  static const struct { const char* key; const char* flag; } kMap[] = {
      {"gpu", "--gpu"},           {"count", "--count"},
      {"vram_mb", "--vram-mb"},   {"cuda_version", "--cuda"},
      {"rocm_version", "--rocm"}, {"driver_version", "--driver"},
      {"os", "--os"},
  };
  for (const auto& m : kMap)
    if (key == m.key) return m.flag;
  return nullptr;
}

int usage(FILE* to) {
  std::fprintf(to,
               "Usage: vgpu config show [--config PATH] [flags]   effective settings and where each came from\n"
               "       vgpu config init [PATH] [--force]          write a commented pantheonsim.yaml\n"
               "       vgpu config keys                           every key, what it accepts and sets\n"
               "\n"
               "`vgpu` reads a settings file, first found of: --config PATH (before the command)\n"
               "or $VGPU_CONFIG, ./pantheonsim.yaml, $XDG_CONFIG_HOME/pantheonsim/config.yaml\n"
               "(default ~/.config). Precedence: flag > environment > file > default.\n"
               "`config show` takes the same flags as the keys (--gpu nvidia/t4 --count 2 --strict)\n"
               "to show what a command line would resolve to. See docs/config.md.\n");
  return to == stdout ? 0 : 2;
}

int fail(const std::string& msg, int code = 1) {
  std::fprintf(stderr, "vgpu: %s\n", msg.c_str());
  return code;
}

int cmd_show(const std::vector<std::string>& args) {
  std::string path = g_flag_path;
  std::map<std::string, std::string> flags;
  const auto gpus = vgpu::available_gpus();
  std::vector<std::string> flag_names = {"--config", "--help"};
  for (const auto& k : cfg::keys()) flag_names.push_back(k.flag());
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--help" || a == "-h") return usage(stdout);
    const cfg::Key* key = nullptr;
    for (const auto& k : cfg::keys())
      if (a == k.flag()) key = &k;
    if (a == "--config" || a.rfind("--config=", 0) == 0) {
      if (a == "--config") {
        if (i + 1 >= args.size()) return fail("config show: --config needs a path", 2);
        path = args[++i];
      } else {
        path = a.substr(9);
      }
      continue;
    }
    if (!key) {
      const std::string near = cfg::closest(a, flag_names);
      return fail("config show: unknown argument '" + a + "'" +
                      (near.empty() ? "" : "; did you mean '" + near + "'?"),
                  2);
    }
    std::string value;
    if (key->switch_flag) {
      value = key->type == cfg::Type::Enum ? "on" : "true";
    } else {
      if (i + 1 >= args.size()) return fail("config show: " + a + " needs a value", 2);
      value = args[++i];
    }
    const std::string why = cfg::check_value(*key, value, gpus);
    if (!why.empty()) return fail("config show: " + a + ": " + why, 2);
    flags[key->name] = value;
  }

  try {
    const cfg::Located where = cfg::locate(path, real_getenv);
    const auto file = cfg::load(where, gpus);
    const auto settings = cfg::effective(file ? &*file : nullptr, flags, real_getenv);
    std::fputs(cfg::render_show(where, file ? &*file : nullptr, settings).c_str(), stdout);
  } catch (const cfg::ConfigError& e) {
    return fail(e.what());
  }
  return 0;
}

int cmd_init(const std::vector<std::string>& args) {
  std::string path = "pantheonsim.yaml";
  bool force = false, have_path = false;
  for (const auto& a : args) {
    if (a == "--help" || a == "-h") return usage(stdout);
    if (a == "--force") force = true;
    else if (!a.empty() && a[0] == '-') return fail("config init: unknown argument '" + a + "'", 2);
    else if (have_path) return fail("config init: more than one path given", 2);
    else { path = a; have_path = true; }
  }
  const std::string err = cfg::init_file(path, force);
  if (!err.empty()) return fail("config init: " + err);
  std::printf("wrote %s\nEdit it, then `vgpu config show` prints what is in effect.\n", path.c_str());
  return 0;
}

}  // namespace

int cmd_config(const std::vector<std::string>& args) {
  if (args.empty()) return usage(stderr);
  const std::string& sub = args[0];
  const std::vector<std::string> rest(args.begin() + 1, args.end());
  if (sub == "--help" || sub == "-h" || sub == "help") return usage(stdout);
  if (sub == "show") return cmd_show(rest);
  if (sub == "init") return cmd_init(rest);
  if (sub == "keys") {
    std::fputs(cfg::render_keys().c_str(), stdout);
    return 0;
  }
  std::fprintf(stderr, "vgpu config: unknown subcommand '%s'\n\n", sub.c_str());
  return usage(stderr);
}

// Returns -1 to carry on with the command, else the exit status.
int config_prepare(std::vector<std::string>& args) {
  g_flag_path.clear();
  if (!args.empty() && (args[0] == "--config" || args[0].rfind("--config=", 0) == 0)) {
    if (args[0] == "--config") {
      if (args.size() < 2) return fail("--config needs a path", 2);
      g_flag_path = args[1];
      args.erase(args.begin(), args.begin() + 2);
    } else {
      g_flag_path = args[0].substr(9);
      args.erase(args.begin());
    }
    if (g_flag_path.empty()) return fail("--config needs a path", 2);
  }
  if (args.empty()) return -1;
  const std::string cmd = args[0];
  // Nothing to apply for these, and a broken file must not stop `--help` or
  // `config` itself (which reports the break).
  if (cmd == "--help" || cmd == "-h" || cmd == "help" || cmd == "--version" || cmd == "config") return -1;
  bool stage2 = false;
  for (const auto& a : args) stage2 = stage2 || a == "--stage2";  // `vgpu shell` re-running itself

  try {
    const cfg::Located where = cfg::locate(g_flag_path, real_getenv);
    if (where.path.empty()) return -1;
    const auto file = cfg::load(where, vgpu::available_gpus());
    if (file)
      cfg::apply(*file, real_getenv, [](const std::string& k, const std::string& v) {
        ::setenv(k.c_str(), v.c_str(), 0);
      });
  } catch (const cfg::ConfigError& e) {
    return fail(e.what());
  }

  if (cmd == "shell" && !stage2) {
    // The shell takes no defaults from the environment, so its flags carry the
    // file's and the environment's values; the user's own come after and win.
    std::vector<std::string> pre;
    for (const auto& k : cfg::keys()) {
      const char* flag = shell_flag(k.name);
      const char* v = std::getenv(k.env.c_str());
      if (flag && v && *v) { pre.push_back(flag); pre.push_back(v); }
    }
    args.insert(args.begin() + 1, pre.begin(), pre.end());
  }
  return -1;
}
