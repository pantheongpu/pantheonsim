// Library calls inside a captured CUDA graph, for the shims whose math runs on the host.
//
// On NVIDIA's libraries a call made while its stream is capturing is recorded into the graph and
// runs at each launch, over what the graph's own kernels have written by then, and with the
// objects it was given as they were at the call -- the caller destroys its descriptors and
// parameter objects as soon as the capture function returns. Here the math is the host's, so the
// call is handed to the graph as a closure over copies.
//
// An entry point starts with VGPU_DEFER_CALL (below). While the stream it names is capturing it
// runs once as a PROBE, with the caller's own arguments: the argument checks happen, in the entry
// point's order, and every status it would return is returned. The first thing past the checks is
// the library's first access to device memory, which calls commit_point(); during a probe that
// throws ProbeCommit, meaning "this call is good and would touch the device". The arguments are then
// copied (every argument that is a registered object, by Snapshot; the host scalars and arrays
// named by Host) and the entry point is recorded to run again at each launch of the graph on the
// copies. During that replay commit_point() does nothing.
//
// Each library keeps its own Registry (its handles and parameter objects, with how to copy each),
// and calls commit_point() from the helper it reads and writes device memory with.
#pragma once

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "vgpu/runtime/capture.hpp"

namespace vgpu_capture __attribute__((visibility("hidden"))) {

struct ProbeCommit {};

inline thread_local bool t_probe = false;
inline thread_local bool t_replay = false;
inline bool probing() { return t_probe; }
inline bool replaying() { return t_replay; }
// The library is about to read or write device memory (or wait for a stream). See the top of this file.
inline void commit_point() {
  if (t_probe) throw ProbeCommit{};
}

class Snapshot;

// The objects a library hands out (handles, descriptors, parameter sets): what each is, and how to
// copy it. An object is copied by `clone` (copy-constructing it by default, or by the function given
// when it holds pointers to other objects).
class Registry {
 public:
  using Clone = void* (*)(const void*, Snapshot&);
  using Delete = void (*)(void*);

  template <class T>
  T* track(T* p, Clone fix = nullptr) {
    Delete del = [](void* q) { delete static_cast<T*>(q); };
    Clone clone = fix;
    if (!clone && std::is_copy_constructible_v<T>) clone = [](const void* q, Snapshot&) -> void* { return copy<T>(q); };
    std::lock_guard<std::mutex> l(mu_);
    live_[p] = Entry{clone, del};
    return p;
  }
  // An object of a library that is not this one's own, copied by `clone` and deleted by `del`: registered for as
  // long as the caller keeps it so (untrack).
  void track_raw(const void* p, Clone clone, Delete del) {
    std::lock_guard<std::mutex> l(mu_);
    live_[p] = Entry{clone, del};
  }
  bool known(const void* p) const {
    std::lock_guard<std::mutex> l(mu_);
    return p && live_.count(p);
  }
  void untrack(const void* p) {
    std::lock_guard<std::mutex> l(mu_);
    live_.erase(p);
  }

 private:
  friend class Snapshot;
  struct Entry {
    Clone clone;
    Delete del;
  };
  template <class T>
  static void* copy(const void* q) {
    if constexpr (std::is_copy_constructible_v<T>) return new T(*static_cast<const T*>(q));
    return nullptr;
  }
  mutable std::mutex mu_;
  std::map<const void*, Entry> live_;
};

// A host-side input whose extent only the call knows -- a scalar, a host array -- as the entry point
// reads it: `bytes()` says how many bytes, and is asked only once the probe has validated the call.
struct Host {
  const void* p;
  std::function<size_t()> bytes;
  template <class T>
  operator T*() const {
    return const_cast<T*>(static_cast<const T*>(p));
  }
};

class Snapshot {
 public:
  explicit Snapshot(Registry& r) : reg_(r) {}
  Snapshot(const Snapshot&) = delete;
  Snapshot& operator=(const Snapshot&) = delete;
  ~Snapshot() {
    for (auto& [orig, c] : done_) {
      reg_.untrack(c.p);
      if (c.del) c.del(c.p);
    }
  }
  // The copy of a registered object (made once); anything else as it is.
  void* of(const void* p) {
    if (!p) return nullptr;
    for (auto& [orig, c] : done_)
      if (orig == p) return c.p;
    Registry::Entry e{};
    {
      std::lock_guard<std::mutex> l(reg_.mu_);
      auto it = reg_.live_.find(p);
      if (it == reg_.live_.end()) return const_cast<void*>(p);   // device memory, a function, an index
      e = it->second;
    }
    if (!e.clone || !e.del) return const_cast<void*>(p);   // not copyable: used as it is
    void* c = e.clone(p, *this);
    if (!c) return const_cast<void*>(p);
    {
      std::lock_guard<std::mutex> l(reg_.mu_);
      reg_.live_[c] = e;
    }
    done_.push_back({p, Copy{c, e.del}});
    return c;
  }
  // A host buffer kept for the life of the snapshot.
  const void* keep(const void* p, size_t bytes) {
    if (!p) return nullptr;
    const auto* b = static_cast<const uint8_t*>(p);
    bufs_.push_back(std::make_shared<std::vector<uint8_t>>(b, b + bytes));
    if (bufs_.back()->empty()) bufs_.back()->resize(1);
    return bufs_.back()->data();
  }

 private:
  struct Copy {
    void* p;
    Registry::Delete del;
  };
  Registry& reg_;
  std::vector<std::pair<const void*, Copy>> done_;
  std::vector<std::shared_ptr<std::vector<uint8_t>>> bufs_;
};

// A scalar (alpha, beta, a tolerance) of `bytes` bytes, read when the call is made.
inline Host scalar(const void* p, size_t bytes) {
  return Host{p, [bytes] { return bytes; }};
}

namespace detail {
template <class P, class A>
P snap_arg(Snapshot& s, const A& a) {
  if constexpr (std::is_same_v<A, Host>) {
    return (P)(a.p ? s.keep(a.p, a.bytes()) : nullptr);
  } else if constexpr (std::is_pointer_v<P> && std::is_convertible_v<A, const void*>) {
    return (P)s.of(static_cast<const void*>(a));
  } else {
    return static_cast<P>(a);
  }
}
}  // namespace detail

// The library's device accesses go through these (a #define of cudaMemcpy and cudaMemset in its
// translation unit), so the first one a call makes is its commit point.
inline cudaError_t memcpy_commit(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
  commit_point();
  return ::cudaMemcpy(dst, src, n, kind);
}
inline cudaError_t memset_commit(void* dst, int value, size_t n) {
  commit_point();
  return ::cudaMemset(dst, value, n);
}

inline bool stream_capturing(cudaStream_t s) {
  if (!s) return false;
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  return cudaStreamIsCapturing(s, &st) == cudaSuccess && st == cudaStreamCaptureStatusActive;
}

// The call itself, when `stream` is capturing: the status it answers (a refusal, or nothing to do on
// the device), or `Status{}` -- success -- once it is recorded. Not capturing: nullopt, and the caller
// goes on to do the work.
template <class Status, class... P, class... A>
std::optional<Status> defer_call(Registry& reg, cudaStream_t stream, Status (*fn)(P...), A... a) {
  static_assert(sizeof...(P) == sizeof...(A), "defer_call: one argument per parameter");
  if (t_probe || t_replay || !stream_capturing(stream)) return std::nullopt;
  {
    struct ProbeScope {
      ProbeScope() { t_probe = true; }
      ~ProbeScope() { t_probe = false; }
    } probe;
    try {
      return fn(a...);
    } catch (const ProbeCommit&) {
    }
  }
  auto snap = std::make_shared<Snapshot>(reg);
  auto args = std::make_shared<std::tuple<std::decay_t<P>...>>(detail::snap_arg<std::decay_t<P>>(*snap, a)...);
  if (!vgpu_record_host_op_if_capturing(stream, [fn, args, snap] {
        t_replay = true;
        std::apply(fn, *args);
        t_replay = false;
      }))
    return std::nullopt;   // the capture ended meanwhile: do it now
  return Status{};
}

// A call that NVIDIA's library cannot capture (it waits for the stream to learn something): the
// checks of the call happen, and past them it answers `refused` and invalidates the capture, as the
// library's own wait does. Not capturing: nullopt.
inline bool refuse_capture(cudaStream_t stream, const char* what) {
  using Fn = int (*)(void*, const char*);
  static const Fn fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "vgpu_capture_refuse_v1"));
  return fn && fn(stream, what) != 0;
}

// A call that allocates device memory while its stream captures: in the global capture mode (the default) that
// is refused, the capture invalidated -- as NVIDIA's library finds when it cudaMallocs in the middle of one. True
// if refused. Not capturing, or in a mode that allows it: false.
inline bool unsafe_allocation(cudaStream_t stream) {
  using Fn = int (*)(const char*);
  static const Fn fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "vgpu_capture_unsafe_gate_v1"));
  return fn && !t_replay && stream_capturing(stream) && fn("cudaMalloc") != 0;
}

template <class Status, class... P, class... A>
std::optional<Status> refuse_call(cudaStream_t stream, const char* what, Status refused, Status (*fn)(P...), A... a) {
  static_assert(sizeof...(P) == sizeof...(A), "refuse_call: one argument per parameter");
  if (t_probe || t_replay || !stream_capturing(stream)) return std::nullopt;
  struct ProbeScope {
    ProbeScope() { t_probe = true; }
    ~ProbeScope() { t_probe = false; }
  } probe;
  try {
    return fn(a...);
  } catch (const ProbeCommit&) {
  }
  t_probe = false;
  refuse_capture(stream, what);
  return refused;
}

}  // namespace vgpu_capture

#define VGPU_REFUSE_CALL(stream, status, fn, ...)                                                           \
  do {                                                                                                      \
    if (auto vgpu_refused_ = ::vgpu_capture::refuse_call((stream), #fn, (status), &fn, __VA_ARGS__))         \
      return *vgpu_refused_;                                                                                \
  } while (0)

// `stream` is the stream the call is on (null if the handle is not valid); `fn` the entry point itself.
#define VGPU_DEFER_CALL(registry, stream, fn, ...)                                                          \
  do {                                                                                                      \
    if (auto vgpu_deferred_ = ::vgpu_capture::defer_call((registry), (stream), &fn, __VA_ARGS__))           \
      return *vgpu_deferred_;                                                                               \
  } while (0)
