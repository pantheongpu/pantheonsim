// The comparison behind `vgpu check`: what a program printed on the real card against what it printed on the simulator.
//
// Two programs never print the same bytes twice. Timings differ, a pointer is printed, the device banner names the driver. So a
// comparison that is useful does three things before it says "different": it drops the lines that are about time or addresses, it
// sets the lines that name the machine aside (a difference there is reported, but as an identity difference), and it separates a
// last-digit difference in a number from a different answer. Everything it drops is counted and can be printed, because a filter
// that hides lines silently is a way to pass a comparison.
//
// Pure functions over strings, so the unit test (tests/unit/test_check_compare.cpp) needs neither a card nor a program.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace vgpu::check {

namespace detail {

inline std::string lower(const std::string& s) {
  std::string o = s;
  for (char& c : o) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return o;
}

inline bool word_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

// `w` (lower case) occurs in `s` (lower case) with a word boundary before it (`lead`) and/or after it (`trail`).
inline bool find_at(const std::string& s, const std::string& w, bool lead, bool trail) {
  for (size_t p = s.find(w); p != std::string::npos; p = s.find(w, p + 1)) {
    if (lead && p > 0 && word_char(s[p - 1])) continue;
    const size_t e = p + w.size();
    if (trail && e < s.size() && word_char(s[e])) continue;
    return true;
  }
  return false;
}
inline bool has_word(const std::string& s, const char* w) { return find_at(s, w, true, true); }
inline bool has_prefix(const std::string& s, const char* w) { return find_at(s, w, true, false); }
inline bool has_sub(const std::string& s, const char* w) { return find_at(s, w, false, false); }

// A digit, optional space, then a lone `s`: "3 s", "0.5s", "12 s".
inline bool number_then_s(const std::string& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) continue;
    size_t j = i + 1;
    if (j < s.size() && s[j] == ' ') ++j;
    if (j < s.size() && s[j] == 's' && (j + 1 == s.size() || !word_char(s[j + 1]))) return true;
  }
  return false;
}

// "0x" and at least six hex digits: a pointer or a context handle.
inline bool has_pointer(const std::string& s) {
  for (size_t p = s.find("0x"); p != std::string::npos; p = s.find("0x", p + 1)) {
    if (p > 0 && word_char(s[p - 1])) continue;
    size_t n = 0;
    while (p + 2 + n < s.size() && std::isxdigit(static_cast<unsigned char>(s[p + 2 + n]))) ++n;
    if (n >= 6) return true;
  }
  return false;
}

// "shm" followed by digits: a shared-memory segment name that changes every run.
inline bool has_shm_name(const std::string& s) {
  for (size_t p = s.find("shm"); p != std::string::npos; p = s.find("shm", p + 1))
    if (p + 3 < s.size() && std::isdigit(static_cast<unsigned char>(s[p + 3]))) return true;
  return false;
}

}  // namespace detail

// A line about time, speed, memory used or an address: it differs between two runs of the same program on the same machine.
inline bool is_timing_line(const std::string& line) {
  using namespace detail;
  const std::string l = lower(line);
  for (const char* w : {"time", "times", "sec", "secs", "ms", "us", "fps", "cpu", "rss", "fom", "perf", "wall", "took", "date",
                        "timestamp", "cycle", "cycles", "performance", "tops", "gops", "flop", "flops"})
    if (has_word(l, w)) return true;
  for (const char* w : {"msec", "usec", "micro", "clock"})
    if (has_prefix(l, w)) return true;
  for (const char* w : {"elapsed", "seconds", "gflop", "tflop", "gb/s", "gib/s", "mb/s", "bandwidth", "throughput", "latency", "runtime",
                        "speedup", "per second", "kernel execution", "duration", "memory used", "gpu:", "gpu mem", "time to", "free time",
                        "memcopy:", "gpu execution", "context=0x"})
    if (has_sub(l, w)) return true;
  if (find_at(l, "/s", false, true) || find_at(l, "/sec", false, true)) return true;
  if (has_prefix(l, "back:")) return true;
  return number_then_s(l) || has_pointer(l) || has_shm_name(l);
}

// A line that says which machine this is. When the simulated profile is the card these agree except for driver and runtime strings;
// a difference there is real and is reported, but it is not a difference in what the program computed. `extra` holds names that
// belong to this comparison: the card's model, the profile's model, the compute capability.
inline bool is_identity_line(const std::string& line, const std::vector<std::string>& extra) {
  using namespace detail;
  const std::string l = lower(line);
  for (const char* w : {"compute capability", "driver version", "runtime version", "multiprocessor", "multi-processor", "device name",
                        "using cuda device", "cuda device", "gpu device", "running on gpu", "ada lovelace", "geforce", "tesla"})
    if (has_sub(l, w)) return true;
  for (const char* w : {"ampere", "hopper", "blackwell", "turing", "volta", "pascal", "sms"})
    if (has_word(l, w)) return true;
  if (has_prefix(l, "sm_")) return true;
  for (const std::string& e : extra)
    if (!e.empty() && has_sub(l, lower(e).c_str())) return true;
  return false;
}

// A line that states an outcome. If such a line also mentions a time ("PASSED in 0.3 s") it is not dropped: dropping it would let a
// FAILED on one side hide behind a PASSED on the other. It is compared with only the numbers that carry a time unit masked, so a count
// of mismatches ("Non-Matching CPU-GPU Outputs ...: 0") is still compared. A verdict line whose time number has no unit differs between
// runs and is reported as a difference; --show-diff shows it.
inline bool is_verdict_line(const std::string& line) {
  const std::string l = detail::lower(line);
  for (const char* w : {"pass", "passed", "fail", "failed", "failure", "error", "errors", "mismatch", "mismatches", "valid", "invalid",
                        "correct", "incorrect", "success", "successful", "ok", "match", "matches", "wrong", "bad", "verified"})
    if (detail::has_word(l, w)) return true;
  return false;
}

struct Filtered {
  std::vector<std::string> kept;      // compared
  std::vector<std::string> identity;  // compared separately: a difference is an identity difference
  size_t dropped = 0;                 // timing lines and blank lines
  std::vector<std::string> dropped_lines;  // the timing lines, for --show-ignored
  std::vector<std::string> masked_lines;   // verdict lines that mention a time: kept, with the numbers that carry a time unit masked
};

inline std::vector<std::string> split_lines(const std::string& t) {
  std::vector<std::string> out;
  size_t s = 0;
  while (s <= t.size()) {
    const size_t e = t.find('\n', s);
    std::string l = t.substr(s, e == std::string::npos ? std::string::npos : e - s);
    while (!l.empty() && (l.back() == '\r' || l.back() == ' ' || l.back() == '\t')) l.pop_back();
    if (e != std::string::npos || !l.empty()) out.push_back(std::move(l));
    if (e == std::string::npos) break;
    s = e + 1;
  }
  return out;
}

inline bool bare_number(const std::string& l) {
  size_t i = 0;
  while (i < l.size() && std::isspace(static_cast<unsigned char>(l[i]))) ++i;
  const size_t b = i;
  while (i < l.size() && (std::isdigit(static_cast<unsigned char>(l[i])) || l[i] == '.' || l[i] == '-' || l[i] == '+' || l[i] == 'e' || l[i] == 'E')) ++i;
  const size_t e = i;
  while (i < l.size() && std::isspace(static_cast<unsigned char>(l[i]))) ++i;
  return e > b && i == l.size();
}

namespace detail {
inline std::string mask_time_numbers(const std::string& s);  // defined below
}

// A bare number on the line after a timing line belongs to it (PolyBench prints "GPU Time in seconds:" and then the value).
inline Filtered filter(const std::string& text, const std::vector<std::string>& identity_words) {
  Filtered f;
  bool prev_timing = false;
  for (const std::string& l : split_lines(text)) {
    const bool timing = is_timing_line(l);
    const bool blank = l.find_first_not_of(" \t") == std::string::npos;
    if (timing && !blank && is_verdict_line(l)) {
      f.kept.push_back(detail::mask_time_numbers(l));
      f.masked_lines.push_back(l);  // listed by --show-ignored, and what makes the pair "timing only" and not "identical"
      prev_timing = false;
      continue;
    }
    if (timing || blank || (prev_timing && bare_number(l))) {
      prev_timing = timing || (prev_timing && bare_number(l));
      if (!blank) f.dropped_lines.push_back(l);
      ++f.dropped;
      continue;
    }
    prev_timing = false;
    (is_identity_line(l, identity_words) ? f.identity : f.kept).push_back(l);
  }
  return f;
}

enum class Verdict {
  Identical,     // the same bytes
  TimingOnly,    // the same after the timing lines are dropped
  IdentityOnly,  // also the same, except lines that name the machine (driver version, device banner)
  NumericClose,  // the same text with different numbers; max_rel says how different
  Differs,
};

inline const char* verdict_name(Verdict v) {
  switch (v) {
    case Verdict::Identical: return "identical";
    case Verdict::TimingOnly: return "identical after dropping timing lines";
    case Verdict::IdentityOnly: return "identical except lines that name the machine";
    case Verdict::NumericClose: return "same text, different numbers";
    case Verdict::Differs: return "differs";
  }
  return "?";
}

struct Compared {
  Verdict verdict = Verdict::Differs;
  double max_rel = 0;           // largest relative difference over the numbers that differ (NumericClose)
  size_t numbers_differ = 0;
  std::string first_a, first_b;  // the first pair of lines that differ, for the report
};

namespace detail {

struct Num { bool ok = false; double v = 0; size_t b = 0, e = 0; };

// Scans a number starting at `i`: sign, digits, point, digits, exponent; or the words nan / inf.
inline Num scan_number(const std::string& s, size_t i) {
  Num n;
  const size_t b = i;
  auto lw = [&](const char* w) {
    const size_t k = std::char_traits<char>::length(w);
    if (s.size() - i < k) return false;
    for (size_t j = 0; j < k; ++j)
      if (std::tolower(static_cast<unsigned char>(s[i + j])) != w[j]) return false;
    return i + k == s.size() || !word_char(s[i + k]);
  };
  if (lw("nan")) { n.ok = true; n.v = std::nan(""); n.b = b; n.e = i + 3; return n; }
  if (lw("inf")) { n.ok = true; n.v = HUGE_VAL; n.b = b; n.e = i + 3; return n; }
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
  const size_t ds = i;
  while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
  size_t nd = i - ds;
  if (i < s.size() && s[i] == '.') {
    ++i;
    const size_t fs = i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
    nd += i - fs;
  }
  if (nd == 0) return n;
  if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
    size_t j = i + 1;
    if (j < s.size() && (s[j] == '-' || s[j] == '+')) ++j;
    if (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) {
      while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
      i = j;
    }
  }
  n.ok = true;
  n.b = b;
  n.e = i;
  n.v = std::strtod(s.substr(b, i - b).c_str(), nullptr);
  return n;
}

// Replaces every number by '#', collecting the values.
inline std::string skeleton(const std::string& s, std::vector<double>& nums) {
  std::string out;
  for (size_t i = 0; i < s.size();) {
    const char c = s[i];
    const bool start = std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' ||
                       ((c == 'n' || c == 'N' || c == 'i' || c == 'I') && (i == 0 || !word_char(s[i - 1])));
    if (start) {
      const Num n = scan_number(s, i);
      if (n.ok) { nums.push_back(n.v); out += '#'; i = n.e; continue; }
    }
    out += c;
    ++i;
  }
  return out;
}

// Replaces each number that is followed by a unit of time or speed ("0.3 s", "12ms", "5 GB/s") with '#'.
inline std::string mask_time_numbers(const std::string& s) {
  static const char* const kUnits[] = {"s", "ms", "us", "ns", "sec", "secs", "second", "seconds", "msec", "usec", "min", "gb/s", "gib/s",
                                       "mb/s", "gflops", "gflop/s", "tflops", "fps"};
  std::string out;
  for (size_t i = 0; i < s.size();) {
    const char c = s[i];
    if (std::isdigit(static_cast<unsigned char>(c)) || ((c == '-' || c == '+' || c == '.') && i + 1 < s.size())) {
      const Num n = scan_number(s, i);
      if (n.ok && n.e > i) {
        size_t j = n.e;
        while (j < s.size() && s[j] == ' ') ++j;
        size_t k = j;
        while (k < s.size() && (std::isalpha(static_cast<unsigned char>(s[k])) || s[k] == '/')) ++k;
        const std::string unit = lower(s.substr(j, k - j));
        bool is_unit = false;
        for (const char* u : kUnits) is_unit = is_unit || unit == u;
        if (is_unit) { out += '#'; i = n.e; continue; }
        out.append(s, i, n.e - i);
        i = n.e;
        continue;
      }
    }
    out += c;
    ++i;
  }
  return out;
}

}  // namespace detail

// Compares two filtered outputs. `mask`, if given, has one entry per kept line of `a`: true lines are skipped (lines that differ
// between two runs on the card itself, which no simulator can be expected to reproduce).
inline Compared compare(const Filtered& a, const Filtered& b, const std::vector<bool>* mask = nullptr) {
  using namespace detail;
  Compared r;
  bool same_text = a.kept.size() == b.kept.size();
  bool close = same_text;
  if (same_text) {
    for (size_t i = 0; i < a.kept.size(); ++i) {
      if (mask && i < mask->size() && (*mask)[i]) continue;
      if (a.kept[i] == b.kept[i]) continue;
      same_text = false;
      std::vector<double> na, nb;
      const std::string sa = skeleton(a.kept[i], na), sb = skeleton(b.kept[i], nb);
      if (sa != sb || na.size() != nb.size()) {
        close = false;
        if (r.first_a.empty()) { r.first_a = a.kept[i]; r.first_b = b.kept[i]; }
        continue;
      }
      for (size_t k = 0; k < na.size(); ++k) {
        const double x = na[k], y = nb[k];
        if (x == y || (std::isnan(x) && std::isnan(y))) continue;
        const double rel = std::fabs(x - y) / std::fmax(std::fabs(x), std::fabs(y));
        ++r.numbers_differ;
        if (!(rel <= r.max_rel)) r.max_rel = std::isnan(rel) ? HUGE_VAL : rel;
      }
      if (r.first_a.empty()) { r.first_a = a.kept[i]; r.first_b = b.kept[i]; }
    }
  } else {
    close = false;
    for (size_t i = 0; i < a.kept.size() && i < b.kept.size(); ++i)
      if (a.kept[i] != b.kept[i]) { r.first_a = a.kept[i]; r.first_b = b.kept[i]; break; }
    if (r.first_a.empty()) {
      r.first_a = a.kept.size() > b.kept.size() ? a.kept[b.kept.size()] : "(no more lines)";
      r.first_b = b.kept.size() > a.kept.size() ? b.kept[a.kept.size()] : "(no more lines)";
    }
  }
  if (same_text) {
    if (a.identity != b.identity) r.verdict = Verdict::IdentityOnly;
    else if (a.kept == b.kept && a.dropped_lines == b.dropped_lines && a.masked_lines == b.masked_lines) r.verdict = Verdict::Identical;
    else r.verdict = Verdict::TimingOnly;
    return r;
  }
  r.verdict = close ? Verdict::NumericClose : Verdict::Differs;
  return r;
}

// Which kept lines of `a` differ from the same line of `b`; empty if the two cannot be aligned (different number of lines).
inline std::vector<bool> noisy_lines(const Filtered& a, const Filtered& b, bool* aligned) {
  std::vector<bool> m;
  *aligned = a.kept.size() == b.kept.size();
  if (!*aligned) return m;
  m.resize(a.kept.size());
  for (size_t i = 0; i < m.size(); ++i) m[i] = a.kept[i] != b.kept[i];
  return m;
}

}  // namespace vgpu::check
