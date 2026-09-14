// Strict argument parsing shared by the vgpu subcommands.
//
// Every subcommand used to read numbers with atoi/atof and values with "the
// next argument, or an empty string". Both fail silently in the worst way:
// `--count abc` became 0 and was then clamped to 1, `--vram-mb 1x` became 1 MiB,
// and `vgpu shell -y -c` with the command missing took "" -- which is an
// interactive shell reading stdin, so a CI job hung instead of failing. A
// machine that quietly differs from the one asked for is worse than an error,
// so these accept the whole string or nothing, and say which flag was wrong.
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace vgpu::cli {

// An integer spelled entirely in decimal digits (with an optional leading '-'),
// inside [lo, hi]. Leading or trailing junk, overflow and an empty string all
// fail.
inline bool parse_int(const std::string& s, long long lo, long long hi, long long* out) {
  if (s.empty() || s.size() > 20) return false;
  size_t start = s[0] == '-' ? 1 : 0;
  if (start == s.size() || s.find_first_not_of("0123456789", start) != std::string::npos)
    return false;
  errno = 0;
  char* end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (errno == ERANGE || *end != '\0' || v < lo || v > hi) return false;
  *out = v;
  return true;
}

// A finite decimal number inside [lo, hi]: "0.5", "1", ".25". No hex, no
// exponent tricks beyond what strtod accepts on a whole plain string, no "nan".
inline bool parse_double(const std::string& s, double lo, double hi, double* out) {
  if (s.empty() || s.find_first_not_of("0123456789.-+eE") != std::string::npos) return false;
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (errno == ERANGE || *end != '\0' || !std::isfinite(v) || v < lo || v > hi) return false;
  *out = v;
  return true;
}

// A dotted numeric version with at least `min_parts` components: "13.0",
// "580.65.06", "6.2.0". Each component is one or more digits.
inline bool is_dotted_version(const std::string& s, int min_parts) {
  int parts = 0;
  size_t at = 0;
  while (true) {
    const size_t dot = s.find('.', at);
    const std::string part = s.substr(at, dot == std::string::npos ? std::string::npos : dot - at);
    if (part.empty() || part.size() > 9 || part.find_first_not_of("0123456789") != std::string::npos)
      return false;
    ++parts;
    if (dot == std::string::npos) break;
    at = dot + 1;
  }
  return parts >= min_parts;
}

}  // namespace vgpu::cli
