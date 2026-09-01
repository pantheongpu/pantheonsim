#include "vgpu/yamlish.hpp"

#include <cctype>

#include "vgpu/error.hpp"

namespace vgpu::yamlish {
namespace {

[[noreturn]] void fail(const std::string& origin, size_t line, const std::string& msg) {
  throw Error::make(Err::ProfileParse, origin, ":", line, ": ", msg);
}

std::string strip(const std::string& s) {
  size_t a = s.find_first_not_of(" \t");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t");
  return s.substr(a, b - a + 1);
}

// Removes an inline comment. Quotes are respected.
std::string strip_comment(const std::string& s) {
  bool in_quote = false;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '"') in_quote = !in_quote;
    if (s[i] == '#' && !in_quote) return s.substr(0, i);
  }
  return s;
}

bool parse_int(const std::string& s, int64_t& out) {
  if (s.empty()) return false;
  size_t i = s[0] == '-' ? 1 : 0;
  if (i == s.size()) return false;
  for (size_t j = i; j < s.size(); ++j)
    if (!std::isdigit(static_cast<unsigned char>(s[j]))) return false;
  out = std::stoll(s);
  return true;
}

Value parse_scalar(const std::string& raw, const std::string& origin, size_t line) {
  std::string s = strip(raw);
  Value v;
  if (s.empty()) fail(origin, line, "empty value");
  if (s.front() == '"') {
    if (s.size() < 2 || s.back() != '"') fail(origin, line, "unterminated string: " + s);
    v.kind = Value::Kind::Str;
    v.str = s.substr(1, s.size() - 2);
    return v;
  }
  if (s.front() == '[') {
    if (s.back() != ']') fail(origin, line, "unterminated list: " + s);
    v.kind = Value::Kind::List;
    std::string inner = s.substr(1, s.size() - 2);
    std::string item;
    for (size_t i = 0; i <= inner.size(); ++i) {
      if (i == inner.size() || inner[i] == ',') {
        if (!strip(item).empty()) v.list.push_back(parse_scalar(item, origin, line));
        item.clear();
      } else {
        item += inner[i];
      }
    }
    return v;
  }
  if (s == "true" || s == "false") {
    v.kind = Value::Kind::Bool;
    v.b = (s == "true");
    return v;
  }
  if (int64_t i = 0; parse_int(s, i)) {
    v.kind = Value::Kind::Int;
    v.i = i;
    return v;
  }
  v.kind = Value::Kind::Str;
  v.str = s;
  return v;
}

}  // namespace

Value parse(const std::string& src, const std::string& origin) {
  Value root;
  root.kind = Value::Kind::Map;
  Value* open_map = nullptr;  // nested map currently being filled, if any

  size_t lineno = 0;
  size_t pos = 0;
  while (pos <= src.size()) {
    size_t eol = src.find('\n', pos);
    if (eol == std::string::npos) eol = src.size();
    std::string line = src.substr(pos, eol - pos);
    pos = eol + 1;
    ++lineno;
    if (!line.empty() && line.back() == '\r') line.pop_back();

    line = strip_comment(line);
    if (strip(line).empty()) continue;

    size_t indent = line.find_first_not_of(' ');
    if (line[indent] == '\t') fail(origin, lineno, "tabs are not allowed for indentation");
    if (indent != 0 && indent != 2) fail(origin, lineno, "indent must be 0 or 2 spaces (one nesting level)");

    std::string body = line.substr(indent);
    size_t colon = body.find(':');
    if (colon == std::string::npos) fail(origin, lineno, "expected 'key: value', got: " + body);
    std::string key = strip(body.substr(0, colon));
    std::string rest = body.substr(colon + 1);
    if (key.empty()) fail(origin, lineno, "empty key");

    Value* target = &root;
    if (indent == 2) {
      if (!open_map) fail(origin, lineno, "indented entry with no open map above it");
      target = open_map;
    } else if (strip(rest).empty()) {
      // "key:" opens a nested map.
      Value m;
      m.kind = Value::Kind::Map;
      auto [it, fresh] = root.map.emplace(key, std::move(m));
      if (!fresh) fail(origin, lineno, "duplicate key: " + key);
      open_map = &it->second;
      continue;
    } else {
      open_map = nullptr;
    }

    if (strip(rest).empty()) fail(origin, lineno, "nested maps are only supported one level deep");
    if (!target->map.emplace(key, parse_scalar(rest, origin, lineno)).second)
      fail(origin, lineno, "duplicate key: " + key);
  }
  return root;
}

}  // namespace vgpu::yamlish
