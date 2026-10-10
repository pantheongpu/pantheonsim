// The arguments of an API call, kept as bytes for the profiler.
//
// A profiler (CUPTI) is handed pointers to a call's arguments, and it reads them at the call's enter and again
// at its exit, so they must outlive the call: the object lives in the frame of the function that makes the call.
// The arguments come in by value and are copied as bytes into slots of their own. Nothing takes the address of
// the caller's parameters (a parameter whose address is taken is a memory object, and handing it on would load
// it as its type) and nothing loads a slot as its type: an enum argument that holds a number the enum does not
// declare -- a program's deliberate error-path check -- is undefined behaviour to load, which UBSan reports.
#pragma once

#include <cstddef>
#include <cstring>
#include <tuple>
#include <utility>

namespace vgpu_traced {

template <class T>
struct ArgSlot {
  alignas(16) unsigned char b[sizeof(T)];
};

template <std::size_t I, class Tuple>
void keep_args(Tuple&) {}

// Each argument is taken by value, so it is a parameter nobody takes the address of in this function except to
// copy its bytes.
template <std::size_t I, class Tuple, class T, class... R>
void keep_args(Tuple& slots, T v, R... rest) {
  std::memcpy(std::get<I>(slots).b, &v, sizeof v);
  keep_args<I + 1>(slots, rest...);
}

template <class... A>
class KeptArgs {
 public:
  explicit KeptArgs(A... a) {
    keep_args<0>(slots_, a...);
    fill_argv(std::index_sequence_for<A...>{});
  }
  KeptArgs(const KeptArgs&) = delete;
  KeptArgs& operator=(const KeptArgs&) = delete;
  // Pointers to the arguments in declaration order, null-terminated.
  const void* const* argv() const { return argv_; }
  static constexpr int count() { return static_cast<int>(sizeof...(A)); }

 private:
  template <std::size_t... I>
  void fill_argv(std::index_sequence<I...>) {
    ((argv_[I] = std::get<I>(slots_).b), ...);
    argv_[sizeof...(A)] = nullptr;
  }
  std::tuple<ArgSlot<A>...> slots_;
  const void* argv_[sizeof...(A) + 1];
};

}  // namespace vgpu_traced
