// The settings file: `pantheonsim.yaml`, a flat list of `key: value` lines that
// stands in for the VGPU_* environment variables.
//
// The simulator engine and the shim libraries never see this file. The CLI
// reads it and, for each key, sets the matching VGPU_* variable when nothing
// has set it already, so precedence is
//
//     command-line flag  >  real environment variable  >  file  >  default
//
// (a flag already beat the environment: `vgpu run` writes its flags into the
// child's environment with overwrite). One table, keys(), says for every key
// which variable it maps to, what it may hold and what it does; the parser, the
// range checks, `vgpu config show|keys|init` and docs/config.md all come from
// it. This file and config.cpp use nothing from the simulator, so the unit test
// compiles them on their own.
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vgpu::cli::config {

// A message meant for the person who wrote the file or typed the flag. It
// already names the file and line, so the CLI prints it as it is.
struct ConfigError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class Type {
  Int,          // a whole number in [lo, hi]
  Bool,         // true or false
  Enum,         // one of the words in `choices`
  Text,         // any non-empty text (a path)
  Gpu,          // a profile id from `vgpu list-gpus`
  CudaVersion,  // exactly major.minor, as the driver accepts it
  Version,      // dotted, at least two parts: 580.65.06, 6.4.0
  Os,           // ubuntu, ubuntu:24.04, rocky, debian (what `vgpu shell --os` takes)
};

// A word the file may use and the value it puts in the variable. An empty
// `env` leaves the variable unset: that word means "as if not configured".
struct Choice {
  std::string word;
  std::string env;
};

struct Key {
  std::string name;         // as written in the file: vram_mb
  std::string env;          // the variable it sets: VGPU_VRAM_MB
  Type type = Type::Text;
  long long lo = 0, hi = 0; // Int
  std::vector<Choice> choices;  // Bool (true, false) and Enum
  std::string deflt;        // what it is when nothing sets it, for `config show`
  std::string example;      // a valid value, for the starter file and the docs
  std::string description;  // one line
  bool shell = false;       // `vgpu shell` takes it as a default
  bool switch_flag = false; // on `config show` the flag takes no value (--strict)
  std::string flag() const; // --vram-mb
};

const std::vector<Key>& keys();
const Key* find_key(const std::string& name);

// The candidate nearest to `word` when one is plausibly what was meant (a typo,
// a missing suffix, dashes for underscores), else "".
std::string closest(const std::string& word, const std::vector<std::string>& candidates);

// What the key accepts, in words: "a whole number from 1 to 16".
std::string allowed(const Key& key);

// Why `value` is not acceptable for `key`, or "" when it is. `gpus` is the list
// of known profile ids.
std::string check_value(const Key& key, const std::string& value, const std::vector<std::string>& gpus);

// The value to put in the variable for a valid `value`; "" leaves it unset.
std::string env_value(const Key& key, const std::string& value);

// ---------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------

struct Entry {
  std::string key, value;
  int line = 0;
};

struct File {
  std::string path;
  std::vector<Entry> entries;
  const Entry* find(const std::string& key) const;
};

// Syntax and key names only: `key: value` lines, comments, blank lines. Throws
// ConfigError as "<origin>:<line>: <what is wrong>".
File parse(const std::string& text, const std::string& origin);

// Every value against its key. Throws ConfigError as "<path>:<line>: <key>: ...".
void validate(const File& file, const std::vector<std::string>& gpus);

// ---------------------------------------------------------------------------
// Finding it, and applying it
// ---------------------------------------------------------------------------

using GetEnv = std::function<const char*(const char*)>;
using SetEnv = std::function<void(const std::string&, const std::string&)>;

struct Located {
  std::string path;                  // "" when none was found
  std::string requested_by;          // "--config" or "VGPU_CONFIG" when one named it, else ""
  std::vector<std::string> searched;
};

// --config <path> / VGPU_CONFIG, then ./pantheonsim.yaml, then
// $XDG_CONFIG_HOME/pantheonsim/config.yaml (default ~/.config). A file that was
// asked for by name and is not there is an error; the others are only looked at.
Located locate(const std::string& flag_path, const GetEnv& env);

// Reads, parses and validates the located file; nullopt when there is none.
// A file that exists but cannot be read or is invalid throws.
std::optional<File> load(const Located& where, const std::vector<std::string>& gpus);

// Sets, from the file, each variable that is not set (or is empty). Returns how
// many it set.
int apply(const File& file, const GetEnv& env, const SetEnv& set);

// ---------------------------------------------------------------------------
// `vgpu config`
// ---------------------------------------------------------------------------

enum class Source { Default, File, Environment, Flag };

struct Setting {
  const Key* key = nullptr;
  std::string value;   // what it is
  Source source = Source::Default;
  std::string detail;  // "pantheonsim.yaml:3", "VGPU_GPU", "--gpu"
  std::string note;    // "overrides line 3 of ..." when something lower was shadowed
};

// Every key with its effective value and where it came from. `flags` maps a key
// name to the value of the flag that was given for it (already checked).
std::vector<Setting> effective(const File* file, const std::map<std::string, std::string>& flags,
                               const GetEnv& env);

std::string render_show(const Located& where, const File* file, const std::vector<Setting>& settings);
std::string render_keys();
std::string starter_text();

// Writes the starter file. Returns "" on success, else the message. Without
// `force` an existing file is left alone.
std::string init_file(const std::string& path, bool force);

}  // namespace vgpu::cli::config
