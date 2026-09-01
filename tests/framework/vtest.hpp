// vtest — minimal, dependency-free test framework for VirtualGPU.
// One executable per test file: use VTEST(name) { ... } and VTEST_MAIN at the end.
#pragma once

#include <cstdio>
#include <exception>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace vtest {

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct Failure : std::exception {
  std::string msg;
  explicit Failure(std::string m) : msg(std::move(m)) {}
  const char* what() const noexcept override { return msg.c_str(); }
};

template <class T, class U>
std::string format_eq_failure(const char* file, int line, const char* ae, const char* be, const T& a, const U& b) {
  std::ostringstream os;
  os << file << ":" << line << ": VCHECK_EQ(" << ae << ", " << be << ") failed\n  left:  " << a
     << "\n  right: " << b;
  return os.str();
}

// Runs fn, expecting it to throw E. Returns the caught exception for further inspection.
template <class E, class F>
E capture(F&& fn, const char* file, int line, const char* expr) {
  try {
    fn();
  } catch (const E& e) {
    return e;
  } catch (const std::exception& e) {
    throw Failure(std::string(file) + ":" + std::to_string(line) + ": expected exception from (" + expr +
                  "), got different exception: " + e.what());
  }
  throw Failure(std::string(file) + ":" + std::to_string(line) + ": expected exception from (" + expr +
                "), but nothing was thrown");
}

inline int run_all(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int failed = 0, ran = 0;
  for (auto& c : registry()) {
    if (filter && std::string(c.name).find(filter) == std::string::npos) continue;
    ++ran;
    try {
      c.fn();
      std::printf("[ PASS ] %s\n", c.name);
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[ FAIL ] %s\n%s\n", c.name, e.what());
    }
  }
  std::printf("%d/%d tests passed\n", ran - failed, ran);
  return failed == 0 && ran > 0 ? 0 : 1;
}

}  // namespace vtest

#define VTEST(name)                                                   \
  static void vtest_fn_##name();                                      \
  static ::vtest::Registrar vtest_reg_##name(#name, vtest_fn_##name); \
  static void vtest_fn_##name()

#define VCHECK(cond)                                                                              \
  do {                                                                                            \
    if (!(cond)) throw ::vtest::Failure(std::string(__FILE__) + ":" + std::to_string(__LINE__) +  \
                                        ": VCHECK(" #cond ") failed");                            \
  } while (0)

#define VCHECK_EQ(a, b)                                                                       \
  do {                                                                                        \
    auto _va = (a);                                                                           \
    auto _vb = (b);                                                                           \
    if (!(_va == _vb))                                                                        \
      throw ::vtest::Failure(::vtest::format_eq_failure(__FILE__, __LINE__, #a, #b, _va, _vb)); \
  } while (0)

// Usage: auto err = VCAPTURE(vgpu::Error, expr);
#define VCAPTURE(EType, expr) ::vtest::capture<EType>([&] { (void)(expr); }, __FILE__, __LINE__, #expr)

#define VCHECK_CONTAINS(haystack, needle)                                                        \
  do {                                                                                           \
    std::string _h = (haystack);                                                                 \
    if (_h.find(needle) == std::string::npos)                                                    \
      throw ::vtest::Failure(std::string(__FILE__) + ":" + std::to_string(__LINE__) +            \
                             ": expected to find \"" + std::string(needle) + "\" in:\n" + _h);   \
  } while (0)

#define VTEST_MAIN                                              \
  int main(int argc, char** argv) { return ::vtest::run_all(argc, argv); }
