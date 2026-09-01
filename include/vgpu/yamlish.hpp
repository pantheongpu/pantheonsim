// A deliberately restricted YAML-subset parser for device profiles.
//
// VirtualGPU profiles use a documented subset of YAML so the project stays
// dependency-free. Supported:
//   - "key: scalar" at indent 0
//   - "key:" opening a nested map whose entries are at indent 2 (one level only)
//   - scalars: quoted strings, bare strings, integers, true/false
//   - inline lists of scalars: [a, b, c]
//   - comments with '#'
// Anything else is a ProfileParse error naming the offending line.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vgpu::yamlish {

struct Value {
  enum class Kind { Str, Int, Bool, List, Map };
  Kind kind = Kind::Str;
  std::string str;
  int64_t i = 0;
  bool b = false;
  std::vector<Value> list;
  std::map<std::string, Value> map;
};

// Parses a document into a top-level map. `origin` names the source in errors.
Value parse(const std::string& src, const std::string& origin);

}  // namespace vgpu::yamlish
