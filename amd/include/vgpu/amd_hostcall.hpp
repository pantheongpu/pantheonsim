// The host side of the hostcall protocol: how a kernel calls the host while it
// runs, which is what device-side printf is built on.
//
// The protocol is the one ROCm's device library speaks (ockl's
// hostcall_impl.cl and services.cl): the kernel is handed a buffer of packets
// through its hidden_hostcall_buffer argument, takes one from a free stack,
// fills in a slot per lane, pushes it onto a ready stack and raises a doorbell
// signal; with the signal's mailbox set, that ends in s_sendmsg, which is
// where this model does the host's part. The host answers each lane in its
// slot, clears the packet's ready bit, and the kernel, spinning on that bit,
// carries on.
//
// The services implemented are all the device library has: printf (and
// fprintf to stdout or stderr), device memory (what device-side malloc and
// free ask for), the address sanitizer's report, and a device function call
// (__ockl_call_host_function: the kernel gives the address of a host function
// and up to seven words, and gets two words back). A printf is a message, carried in as
// many packets as it needs: the first word says which stream, then the format
// string, eight bytes to a word, then the arguments -- a word each, or a
// string's bytes for %s. The host gives the message an id when it begins and
// formats it when it ends.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace vgpu {
class MemoryManager;
}

namespace vgpu::amd {

class Hostcall {
 public:
  // stream: 0 for stdout, 1 for stderr; text: one complete printf's output.
  using Print = std::function<void(int stream, const std::string& text)>;

  // Lays out a buffer of `packets` packets (a power of two) in `mem`.
  Hostcall(MemoryManager& mem, Print print, uint32_t packets = 64);
  ~Hostcall();
  Hostcall(const Hostcall&) = delete;
  Hostcall& operator=(const Hostcall&) = delete;

  // What a kernel's hidden_hostcall_buffer argument is given.
  uint64_t buffer() const { return buffer_; }

  // What a kernel's hidden_heap_v1 argument is given (code object version 5
  // and later; earlier ones keep the allocator's state in their own data): the
  // state block of the device library's malloc, which the runtime provides and
  // sets up before the first kernel uses it. It is 128 KiB of zeroes with each
  // of the sixteen block sizes' count of recordable slabs set to the 256 the
  // allocator's own table holds (what ockl's __ockl_dm_init_v1 does, with no
  // initial slabs: the allocator then asks for every slab through the
  // device-memory service). Made on the first call.
  uint64_t heap();

  // Does the host's part for every packet the device has made ready. Safe to
  // call from several threads at once; throws, naming it, for a service
  // the device library does not have.
  void service();

  // The protocol's service numbers, as ockl's services.cl gives them.
  static constexpr uint32_t kServiceFunctionCall = 1;   // a host function the kernel names
  static constexpr uint32_t kServicePrintf = 2;
  static constexpr uint32_t kServiceDevmem = 3;      // device malloc's slabs and large blocks
  static constexpr uint32_t kServiceSanitizer = 4;   // the address sanitizer's report

 private:
  void serve_function_call(uint64_t payload, uint64_t active);
  void serve_printf(uint64_t payload, uint64_t active);
  void serve_devmem(uint64_t payload, uint64_t active);
  void serve_sanitizer(uint64_t payload, uint64_t active);
  uint64_t alloc(uint64_t bytes);

  MemoryManager& mem_;
  Print print_;
  uint32_t packets_;
  uint64_t buffer_ = 0, headers_ = 0, payloads_ = 0, signal_ = 0, mailbox_ = 0;
  std::vector<uint64_t> allocations_;
  uint64_t heap_ = 0;
  std::mutex mu_;
  std::map<uint64_t, std::vector<uint64_t>> messages_;   // by id: the words so far
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> devmem_;   // block -> (allocation it sits in, size)
  uint64_t next_id_ = 1;
};

// The signature of a host function a kernel calls through the function call
// service: up to seven input words, two output words (ROCm's CLR documents it
// as HostcallFunctionCall in rocclr/device/devhostcall.cpp).
using HostFunction = void (*)(uint64_t* output, const uint64_t* input);

// A printf message's words -- the stream, the format, the arguments -- as the
// text the host's printf would give for them.
std::string format_printf_message(const std::vector<uint64_t>& words);

}  // namespace vgpu::amd
