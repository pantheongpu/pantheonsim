// Device printf, formatted the same way for both engines: the PTX
// interpreter's vprintf call and the SASS executor's call to vprintf hand the
// format and the argument buffer here, so a kernel prints the same text
// whichever of its codes runs.
#pragma once

#include <bit>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

#include "vgpu/error.hpp"

namespace vgpu::exec {

// The C format, formatted the way an RTX 3060's host side formats it: glibc's
// text for every conversion (so "(nil)", "(null)", "-nan"), a `*` width taken
// from the arguments, and a conversion it does not know printed as written,
// consuming no argument. What the card does wrong is refused by name rather
// than imitated: a `*` precision, hh, %Lf and %n.
//
// fetch(size) returns the next argument of `size` bytes, aligned naturally in
// the argument buffer; cstring(address) reads a NUL-terminated string from
// the device; refuse(err, why) reports a format that is not supported and
// does not return.
template <class Fetch, class CString, class Refuse>
std::string format_device_printf(const std::string& fmt, Fetch&& fetch, CString&& cstring, Refuse&& refuse) {
  std::string out;
  auto format = [&](const std::string& spec, auto value) {
    const int n = std::snprintf(nullptr, 0, spec.c_str(), value);
    std::string text(n > 0 ? static_cast<size_t>(n) : 0, '\0');
    if (n > 0) std::snprintf(text.data(), text.size() + 1, spec.c_str(), value);
    out += text;
  };
  for (size_t i = 0; i < fmt.size(); ++i) {
    if (fmt[i] != '%') {
      out += fmt[i];
      continue;
    }
    const size_t start = i++;
    std::string spec = "%";   // flags, width and precision, lengths removed
    while (i < fmt.size() && std::string("-+ #0").find(fmt[i]) != std::string::npos) spec += fmt[i++];
    if (i < fmt.size() && fmt[i] == '*') {
      // A `*` width is an int argument before the value.
      spec += std::to_string(static_cast<int32_t>(fetch(4)));
      ++i;
    }
    while (i < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[i]))) spec += fmt[i++];
    if (i < fmt.size() && fmt[i] == '.') {
      spec += fmt[i++];
      if (i < fmt.size() && fmt[i] == '*')
        refuse(Err::UnsupportedPtx,
               "printf's `*` precision: an RTX 3060 prints the value as 0 and misreads every "
               "argument after it, so it is not imitated");
      while (i < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[i]))) spec += fmt[i++];
    }
    // Length: hh and h narrow an int, l, ll, z, j and t make it 8 bytes.
    int narrow = 0;          // how many h's
    bool wide = false, long_double = false, lflag = false;
    while (i < fmt.size() && std::string("hlzjtL").find(fmt[i]) != std::string::npos) {
      const char c = fmt[i++];
      if (c == 'h') ++narrow;
      else if (c == 'L') long_double = true;
      else {
        wide = true;
        if (c == 'l') lflag = true;
      }
    }
    if (i >= fmt.size()) refuse(Err::InvalidValue, "printf format ends inside a % specifier");
    const char conv = fmt[i];
    if (narrow >= 2)
      refuse(Err::UnsupportedPtx,
             "printf's hh length: an RTX 3060 reads the value from the wrong bytes (the format "
             "text's), so it is not imitated");
    switch (conv) {
      case '%':
        out += '%';   // "%5%" too
        break;
      case 'd': case 'i': {
        const uint64_t v = fetch(wide ? 8 : 4);
        const long long x = wide     ? static_cast<long long>(v)
                            : narrow ? static_cast<short>(v)
                                     : static_cast<int32_t>(v);
        format(spec + "lld", x);
        break;
      }
      case 'u': case 'o': case 'x': case 'X': {
        const uint64_t v = fetch(wide ? 8 : 4);
        const unsigned long long x = wide     ? v
                                     : narrow ? static_cast<unsigned short>(v)
                                              : static_cast<uint32_t>(v);
        format(spec + "ll" + conv, x);
        break;
      }
      case 'c':
        format(spec + 'c', static_cast<int>(static_cast<unsigned char>(fetch(4))));
        break;
      case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A':
        if (long_double)
          refuse(Err::UnsupportedPtx,
                 "printf's %L: long double is not a device type, and an RTX 3060 prints nan and "
                 "misreads the arguments after it");
        format(spec + conv, std::bit_cast<double>(fetch(8)));
        break;
      case 'p':
        format(spec + 'p', reinterpret_cast<void*>(static_cast<uintptr_t>(fetch(8))));
        break;
      case 's': {
        const uint64_t v = fetch(8);
        if (v == 0) {
          format(spec + 's', "(null)");
        } else {
          if (lflag) refuse(Err::UnsupportedPtx, "printf's %ls with a wide string is not implemented");
          const std::string str = cstring(v);
          format(spec + 's', str.c_str());
        }
        break;
      }
      case 'n':
        refuse(Err::UnsupportedPtx, "printf's %n is not implemented");
        break;
      default:
        // Not a conversion: the card prints it as written and moves on.
        out.append(fmt, start, i - start + 1);
    }
  }
  return out;
}

// Writes one device printf's text. One lock for the whole line, shared by
// both engines: blocks run on several threads, and interleaving two device
// printfs mid-line makes both unreadable.
inline void emit_device_printf(const std::string& text) {
  static std::mutex mu;
  std::lock_guard<std::mutex> guard(mu);
  std::fwrite(text.data(), 1, text.size(), stdout);
  std::fflush(stdout);
}

}  // namespace vgpu::exec
