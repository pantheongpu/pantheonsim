// The host side of the hostcall protocol (vgpu/amd_hostcall.hpp).
//
// The layout is what ROCm's device library reads (ockl's hostcall_impl.cl and
// services.cl, and the amd_signal_t of hsa/amd_hsa_signal.h): nothing here is
// AMD's code.
#include "vgpu/amd_hostcall.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"

namespace vgpu::amd {

namespace {

// buffer_t: where the headers and payloads are, the doorbell signal, the two
// stacks' tops and the mask that takes a packet's index out of a stack entry.
constexpr uint64_t kHeaders = 0, kPayloads = 8, kDoorbell = 16, kFreeStack = 24, kReadyStack = 32, kIndexMask = 40;
constexpr uint64_t kBufferBytes = 48;
// header_t: the next entry on whichever stack it is on, the lanes that were
// active, the service, and a control word whose low bit is "ready".
constexpr uint64_t kNext = 0, kActive = 8, kService = 16, kControl = 20, kHeaderBytes = 24;
// payload_t: eight words for each of 64 lanes.
constexpr uint64_t kSlotWords = 8, kPayloadBytes = 64 * kSlotWords * 8;
// amd_signal_t: the kind (1, a user signal), its value, the mailbox a raise
// writes its event id to -- which is what makes the device send the
// interrupt this model answers -- and that id.
constexpr uint64_t kSignalKind = 0, kSignalMailbox = 16, kSignalEventId = 24, kSignalBytes = 64;

// A message descriptor (services.cl): BEGIN, END, how many words follow, and
// the message's id.
bool begins(uint64_t d) { return d & 1; }
bool ends(uint64_t d) { return d & 2; }
uint32_t length(uint64_t d) { return static_cast<uint32_t>(d >> 5 & 7); }
uint64_t message_id(uint64_t d) { return d >> 8; }

const char* service_name(uint32_t s) {
  switch (s) {
    case 1: return "a device function call";
    case 2: return "printf";
    case 3: return "device malloc";
    case 4: return "the address sanitizer";
    default: return "an unknown service";
  }
}

}  // namespace

Hostcall::Hostcall(MemoryManager& mem, Print print, uint32_t packets)
    : mem_(mem), print_(std::move(print)), packets_(packets) {
  if (packets_ == 0 || (packets_ & (packets_ - 1)) != 0)
    throw Error::make(Err::InvalidValue, "a hostcall buffer holds a power of two packets, not ", packets_);
  buffer_ = alloc(kBufferBytes);
  headers_ = alloc(uint64_t{packets_} * kHeaderBytes);
  payloads_ = alloc(uint64_t{packets_} * kPayloadBytes);
  signal_ = alloc(kSignalBytes);
  mailbox_ = alloc(8);
  // Every packet on the free stack, in order. An entry is the packet's index
  // with a tag above it, which a kernel steps each time it returns a packet
  // so that a stale entry never matches; the tags start at one, since an
  // entry of zero is an empty stack.
  auto entry = [&](uint64_t i) { return i + packets_; };
  for (uint64_t i = 0; i < packets_; ++i)
    mem_.store_scalar(headers_ + i * kHeaderBytes + kNext, 8, i + 1 < packets_ ? entry(i + 1) : 0);
  mem_.store_scalar(buffer_ + kHeaders, 8, headers_);
  mem_.store_scalar(buffer_ + kPayloads, 8, payloads_);
  mem_.store_scalar(buffer_ + kDoorbell, 8, signal_);
  mem_.store_scalar(buffer_ + kFreeStack, 8, entry(0));
  mem_.store_scalar(buffer_ + kReadyStack, 8, 0);
  mem_.store_scalar(buffer_ + kIndexMask, 8, packets_ - 1);
  mem_.store_scalar(signal_ + kSignalKind, 8, 1);
  mem_.store_scalar(signal_ + kSignalMailbox, 8, mailbox_);
  mem_.store_scalar(signal_ + kSignalEventId, 4, 1);
}

Hostcall::~Hostcall() {
  for (uint64_t a : allocations_) {
    try {
      mem_.free(a);
    } catch (const std::exception&) {
      // The device's memory went first (a reset): nothing left to give back.
    }
  }
}

uint64_t Hostcall::alloc(uint64_t bytes) {
  const uint64_t a = mem_.alloc(bytes);
  std::vector<uint8_t> zero(bytes, 0);
  mem_.write(a, zero.data(), bytes);
  allocations_.push_back(a);
  return a;
}

void Hostcall::service() {
  std::lock_guard<std::mutex> lock(mu_);
  // The whole ready stack at once. The kernel pushes onto it with a
  // compare-and-swap, which takes the lock striped by its address when
  // work-groups run on several threads; taking the stack here under the same
  // lock is what keeps the two from losing a packet between them.
  uint64_t top = 0;
  {
    std::lock_guard<std::mutex> stripe(memory_atomic_lock(buffer_ + kReadyStack));
    top = mem_.load_scalar(buffer_ + kReadyStack, 8);
    if (top) mem_.store_scalar(buffer_ + kReadyStack, 8, 0);
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  const uint64_t mask = packets_ - 1;
  while (top) {
    const uint64_t index = top & mask;
    const uint64_t header = headers_ + index * kHeaderBytes, payload = payloads_ + index * kPayloadBytes;
    const uint64_t next = mem_.load_scalar(header + kNext, 8);
    const uint64_t active = mem_.load_scalar(header + kActive, 8);
    const uint32_t service = static_cast<uint32_t>(mem_.load_scalar(header + kService, 4));
    if (service != kServicePrintf)
      throw Error::make(Err::Unsupported, "the kernel called the host for ", service_name(service), " (hostcall service ",
                        service, "), which is not modelled; printf is");
    serve_printf(payload, active);
    // Answered: the kernel, spinning on the ready bit, reads its slots.
    std::atomic_thread_fence(std::memory_order_release);
    mem_.store_scalar(header + kControl, 4, 0);
    top = next;
  }
}

void Hostcall::serve_printf(uint64_t payload, uint64_t active) {
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!(active >> lane & 1)) continue;
    const uint64_t slot = payload + uint64_t{lane} * kSlotWords * 8;
    const uint64_t desc = mem_.load_scalar(slot, 8);
    uint64_t id = message_id(desc);
    if (begins(desc)) {
      id = next_id_++;
      messages_[id].clear();
    }
    auto it = messages_.find(id);
    if (it == messages_.end())
      throw Error::make(Err::InvalidValue, "a printf continued message ", id, ", which never began");
    for (uint32_t k = 0; k < length(desc); ++k) it->second.push_back(mem_.load_scalar(slot + 8 * (1 + k), 8));
    uint64_t answer = id << 8;   // what the lane appends to next
    if (ends(desc)) {
      const std::vector<uint64_t> words = std::move(it->second);
      messages_.erase(it);
      const std::string text = format_printf_message(words);
      print_(words.empty() ? 0 : static_cast<int>(words[0] & 1), text);
      answer = text.size();   // what printf returns
    }
    mem_.store_scalar(slot, 8, answer);
    mem_.store_scalar(slot + 8, 8, 0);
  }
}

namespace {

// The bytes of a string that starts at words[at], through its terminating
// zero; `at` moves past the words it took.
std::string take_string(const std::vector<uint64_t>& words, size_t& at) {
  std::string s;
  while (at < words.size()) {
    const uint64_t w = words[at++];
    for (int b = 0; b < 8; ++b) {
      const char c = static_cast<char>(w >> (8 * b));
      if (c == '\0') return s;
      s.push_back(c);
    }
  }
  return s;
}

// One conversion, however long it comes out.
template <typename T>
std::string one(const std::string& spec, T value) {
  const int n = std::snprintf(nullptr, 0, spec.c_str(), value);
  if (n <= 0) return "";
  std::string s(static_cast<size_t>(n) + 1, '\0');
  std::snprintf(s.data(), s.size(), spec.c_str(), value);
  s.resize(static_cast<size_t>(n));
  return s;
}

}  // namespace

// The format and its arguments as C's printf would print them. The device
// promotes arguments as C does for a variadic call -- a float to a double, a
// char or short to an int -- and sends each as a word; a %s argument comes as
// the string's own bytes. A conversion with no argument left prints as it is
// written, rather than reading past the message.
std::string format_printf_message(const std::vector<uint64_t>& words) {
  size_t at = 1;   // past the stream
  const std::string format = take_string(words, at);
  std::string out;
  auto next = [&](uint64_t* v) {
    if (at >= words.size()) return false;
    *v = words[at++];
    return true;
  };
  for (size_t i = 0; i < format.size(); ++i) {
    if (format[i] != '%') {
      out.push_back(format[i]);
      continue;
    }
    const size_t start = i++;
    if (i < format.size() && format[i] == '%') {
      out.push_back('%');
      continue;
    }
    std::string spec = "%";
    while (i < format.size() && std::strchr("-+ #0'", format[i])) spec.push_back(format[i++]);
    bool missing = false;
    auto number_or_star = [&] {
      if (i < format.size() && format[i] == '*') {
        uint64_t v = 0;
        missing = missing || !next(&v);
        spec += std::to_string(static_cast<int32_t>(v));
        ++i;
      } else {
        while (i < format.size() && format[i] >= '0' && format[i] <= '9') spec.push_back(format[i++]);
      }
    };
    number_or_star();
    if (i < format.size() && format[i] == '.') {
      spec.push_back(format[i++]);
      number_or_star();
    }
    std::string length;
    while (i < format.size() && std::strchr("hljztL", format[i])) length.push_back(format[i++]);
    if (i >= format.size()) {
      out += format.substr(start);
      break;
    }
    const char conv = format[i];
    uint64_t v = 0;
    if (conv == 's') {
      if (at >= words.size()) missing = true;
      else {
        out += one(spec + "s", take_string(words, at).c_str());
        continue;
      }
    } else if (!std::strchr("diuoxXcfFeEgGaAp", conv)) {
      out += format.substr(start, i - start + 1);   // not a conversion printf knows
      continue;
    } else {
      missing = missing || !next(&v);
    }
    if (missing) {
      out += format.substr(start, i - start + 1);
      continue;
    }
    if (conv == 'd' || conv == 'i') {
      int64_t s = length == "hh" ? static_cast<int64_t>(static_cast<int8_t>(v))
                  : length == "h" ? static_cast<int64_t>(static_cast<int16_t>(v))
                  : length.empty() ? static_cast<int64_t>(static_cast<int32_t>(v))
                                   : static_cast<int64_t>(v);
      out += one(spec + "lld", static_cast<long long>(s));
    } else if (std::strchr("uoxX", conv)) {
      uint64_t u = length == "hh" ? static_cast<uint8_t>(v)
                   : length == "h" ? static_cast<uint16_t>(v)
                   : length.empty() ? static_cast<uint32_t>(v)
                                    : v;
      out += one(spec + "ll" + conv, static_cast<unsigned long long>(u));
    } else if (conv == 'c') {
      out += one(spec + "c", static_cast<int>(static_cast<unsigned char>(v)));
    } else if (conv == 'p') {
      out += one(spec + "p", reinterpret_cast<void*>(static_cast<uintptr_t>(v)));
    } else {
      double d;
      std::memcpy(&d, &v, 8);
      out += one(spec + conv, d);
    }
  }
  return out;
}

}  // namespace vgpu::amd
