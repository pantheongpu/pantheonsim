// The host's side of a hostcall, driven the way a kernel drives it: this test
// plays the device, doing what ROCm's device library does (ockl's
// hostcall_impl.cl and services.cl) -- take a packet off the free stack, fill
// a slot per lane, push it onto the ready stack, raise the doorbell -- and
// checks what the host prints and what it answers each lane. hipcc's own
// printf runs through the same path in amd_hipcc (amd/tests/hipcc/printf.cpp).
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "vgpu/amd_hostcall.hpp"
#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

struct Printed {
  int stream;
  std::string text;
};

// The kernel's side: one hostcall, eight words per active lane.
struct Device {
  MemoryManager& mem;
  amd::Hostcall& hc;
  uint64_t buffer, headers, payloads, mask;
  Device(MemoryManager& m, amd::Hostcall& h)
      : mem(m), hc(h), buffer(h.buffer()), headers(m.load_scalar(buffer, 8)), payloads(m.load_scalar(buffer + 8, 8)),
        mask(m.load_scalar(buffer + 40, 8)) {}

  // Returns each lane's two answer words.
  std::vector<std::pair<uint64_t, uint64_t>> call(uint32_t service, uint64_t active,
                                                  const std::vector<std::vector<uint64_t>>& lanes) {
    const uint64_t top = mem.load_scalar(buffer + 24, 8);   // pop the free stack
    VCHECK(top != 0);
    const uint64_t header = headers + (top & mask) * 24, payload = payloads + (top & mask) * 4096;
    mem.store_scalar(buffer + 24, 8, mem.load_scalar(header, 8));
    mem.store_scalar(header + 8, 8, active);
    mem.store_scalar(header + 16, 4, service);
    mem.store_scalar(header + 20, 4, 1);   // ready
    for (uint32_t lane = 0; lane < 64; ++lane)
      if (active >> lane & 1)
        for (uint32_t k = 0; k < 8; ++k)
          mem.store_scalar(payload + lane * 64 + 8 * k, 8, k < lanes[lane].size() ? lanes[lane][k] : 0);
    mem.store_scalar(header, 8, mem.load_scalar(buffer + 32, 8));   // push onto the ready stack
    mem.store_scalar(buffer + 32, 8, top);
    const uint64_t signal = mem.load_scalar(buffer + 16, 8);   // the doorbell, and its mailbox
    VCHECK(mem.load_scalar(signal + 16, 8) != 0);
    hc.service();   // what s_sendmsg MSG_INTERRUPT asks for
    VCHECK_EQ(mem.load_scalar(header + 20, 4), uint64_t{0});   // answered
    std::vector<std::pair<uint64_t, uint64_t>> out(64);
    for (uint32_t lane = 0; lane < 64; ++lane)
      if (active >> lane & 1) out[lane] = {mem.load_scalar(payload + lane * 64, 8), mem.load_scalar(payload + lane * 64 + 8, 8)};
    // Back onto the free stack, its tag stepped.
    uint64_t back = top + mask + 1;
    if (back == 0) back = mask + 1;
    mem.store_scalar(header, 8, mem.load_scalar(buffer + 24, 8));
    mem.store_scalar(buffer + 24, 8, back);
    return out;
  }
};

// A string's bytes, eight to a word, through its terminating zero.
std::vector<uint64_t> words_of(const std::string& s) {
  std::vector<uint64_t> w((s.size() + 1 + 7) / 8, 0);
  for (size_t i = 0; i < s.size(); ++i) w[i / 8] |= static_cast<uint64_t>(static_cast<unsigned char>(s[i])) << (8 * (i % 8));
  return w;
}

constexpr uint64_t kBegin = 1, kEnd = 2;
uint64_t len(uint64_t n) { return n << 5; }

// printf(format, args...) from one lane, as services.cl sends it: the stream,
// then the format seven words at a time, then the arguments, the last with END.
std::pair<uint64_t, uint64_t> device_printf(Device& dev, uint32_t lane, int stream, const std::string& format,
                                            const std::vector<uint64_t>& args) {
  auto one = [&](uint64_t desc, const std::vector<uint64_t>& words) {
    std::vector<std::vector<uint64_t>> lanes(64);
    lanes[lane] = {desc};
    lanes[lane].insert(lanes[lane].end(), words.begin(), words.end());
    return dev.call(amd::Hostcall::kServicePrintf, uint64_t{1} << lane, lanes)[lane];
  };
  uint64_t desc = one(kBegin | len(1), {static_cast<uint64_t>(stream)}).first;
  const std::vector<uint64_t> f = words_of(format);
  for (size_t i = 0; i < f.size(); i += 7) {
    const size_t n = std::min<size_t>(7, f.size() - i);
    desc = one((desc & ~uint64_t{0xFF}) | len(n), std::vector<uint64_t>(f.begin() + i, f.begin() + i + n)).first;
  }
  return one((desc & ~uint64_t{0xFF}) | kEnd | len(args.size()), args);
}

uint64_t bits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, 8);
  return b;
}

}  // namespace

VTEST(a_printf_from_a_lane_is_printed_once_it_ends_and_says_how_long_it_was) {
  MemoryManager mem(64ull << 20);
  std::vector<Printed> printed;
  amd::Hostcall hc(mem, [&](int s, const std::string& t) { printed.push_back({s, t}); }, 8);
  Device dev(mem, hc);
  // A format long enough to take two packets.
  const std::string format = "lane %d found %5.2f and %c in a sentence that runs on past fifty-six bytes\n";
  const uint64_t n = device_printf(dev, 3, 0, format, {3, bits(2.5), 'i'}).first;
  VCHECK_EQ(printed.size(), size_t{1});
  VCHECK_EQ(printed[0].stream, 0);
  VCHECK_EQ(printed[0].text, std::string("lane 3 found  2.50 and i in a sentence that runs on past fifty-six bytes\n"));
  VCHECK_EQ(n, uint64_t{printed[0].text.size()});
}

VTEST(each_lane_has_a_message_of_its_own_and_stderr_is_its_own_stream) {
  MemoryManager mem(64ull << 20);
  std::vector<Printed> printed;
  amd::Hostcall hc(mem, [&](int s, const std::string& t) { printed.push_back({s, t}); }, 8);
  Device dev(mem, hc);
  // Two lanes begin in the same packet and are given different ids.
  std::vector<std::vector<uint64_t>> lanes(64);
  lanes[0] = {kBegin | len(1), 0};
  lanes[5] = {kBegin | len(1), 1};
  const auto begun = dev.call(amd::Hostcall::kServicePrintf, 1 | uint64_t{1} << 5, lanes);
  VCHECK(begun[0].first != begun[5].first);
  lanes[0] = {(begun[0].first & ~uint64_t{0xFF}) | kEnd | len(1), words_of("zero\n")[0]};
  lanes[5] = {(begun[5].first & ~uint64_t{0xFF}) | kEnd | len(1), words_of("five\n")[0]};
  dev.call(amd::Hostcall::kServicePrintf, 1 | uint64_t{1} << 5, lanes);
  VCHECK_EQ(printed.size(), size_t{2});
  VCHECK_EQ(printed[0].text, std::string("zero\n"));
  VCHECK_EQ(printed[0].stream, 0);
  VCHECK_EQ(printed[1].text, std::string("five\n"));
  VCHECK_EQ(printed[1].stream, 1);
}

VTEST(a_service_other_than_printf_is_refused_by_name) {
  MemoryManager mem(64ull << 20);
  amd::Hostcall hc(mem, [](int, const std::string&) {}, 8);
  Device dev(mem, hc);
  std::vector<std::vector<uint64_t>> lanes(64);
  lanes[0] = {1024, 0};
  bool refused = false;
  try {
    dev.call(3 /* device malloc */, 1, lanes);
  } catch (const Error& e) {
    refused = e.code() == Err::Unsupported && std::string(e.what()).find("device malloc") != std::string::npos;
  }
  VCHECK(refused);
}

VTEST(the_conversions_come_out_as_the_hosts_printf_makes_them) {
  auto message = [](const std::string& format, std::vector<uint64_t> args) {
    std::vector<uint64_t> w = {0};
    for (uint64_t x : words_of(format)) w.push_back(x);
    w.insert(w.end(), args.begin(), args.end());
    return amd::format_printf_message(w);
  };
  VCHECK_EQ(message("[%-4d|%04x|%llx|%c|%%]", {static_cast<uint64_t>(-7), 0xab, 0x123456789abcull, 'z'}),
            std::string("[-7  |00ab|123456789abc|z|%]"));
  VCHECK_EQ(message("%hhd %hu %u", {0x1ff, 0x12345, 0xffffffffull}), std::string("-1 9029 4294967295"));
  VCHECK_EQ(message("%*d|%.*f", {5, 42, 2, bits(3.14159)}), std::string("   42|3.14"));
  VCHECK_EQ(message("%e %g", {bits(2.5e-3), bits(0.0001)}), std::string("2.500000e-03 0.0001"));
  // A %s argument is the string's own bytes, however many words that takes.
  std::vector<uint64_t> s = words_of("a string of twenty-six c.");
  s.push_back(9);
  VCHECK_EQ(message("<%s> %d", s), std::string("<a string of twenty-six c.> 9"));
  // A conversion with nothing left to print is printed as it is written.
  VCHECK_EQ(message("%d and %d", {1}), std::string("1 and %d"));
}

VTEST_MAIN
