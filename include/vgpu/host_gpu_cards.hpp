// Host-GPU turbo mode on a machine with several cards (docs/perf-design/12-host-gpu-turbo.md section 17): which of the machine's cards a simulated device
// uses. No simulator state and no CUDA: unit tested on any machine (tests/unit/test_host_gpu_cards.cpp).
//
// The rule: the simulated device in device slot S (the slot its memory window is in: MemoryManager::va_base(), kDeviceVaBase + S * kDeviceVaStride; it is the
// program's ordinal unless CUDA_VISIBLE_DEVICES reorders or hides the simulated machine's devices) runs on the machine's card S, when the machine has one the
// real driver shows. A device with no card stays in the simulator.
// VGPU_HOST_GPU_DEVICES=a,b,c overrides it: slot 0 uses the real driver's device a, slot 1 its device b, and so on; a slot past the list stays in the simulator.
//
// CUDA_VISIBLE_DEVICES: the simulator reads it (runtime/shared_runtime.cpp, to choose among the SIMULATED machine's devices) and so does the real driver the
// mode loads in the same process (at cuInit, to choose among the real cards): one string, two machines. The slots the simulator shows are the machine's
// card numbers the user wrote ("1,0" shows card 1 first), so slot S is the machine's card S, and the real driver numbers that card by its place in the same
// list. measured: a real driver started with CUDA_VISIBLE_DEVICES=2,3 on four T4s reports 2 devices (tools/multicard/va_probe.cu, evidence in
// docs/perf-design/data/multicard/). inferred: the driver's list rule (decimal numbers, the list ends at the first entry it cannot use); UUID entries are
// documented by NVIDIA and not read here: the string is then taken to have chosen the cards already (slot S = the driver's device S).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu::hostgpu {

// VGPU_HOST_GPU_DEVICES: "0,1,3" -> {0, 1, 3}. Decimal, no sign, no repeat (two simulated devices on one card would each map a window on it and
// each count its whole memory free). Unset or empty: no list (`*has` false). False for anything else (the caller says so and turns the mode off).
inline bool parse_card_list(const char* text, std::vector<int>* out, bool* has) {
  out->clear();
  *has = false;
  if (!text || !*text) return true;
  const std::string t(text);
  size_t at = 0;
  while (at <= t.size()) {
    size_t end = t.find(',', at);
    if (end == std::string::npos) end = t.size();
    const std::string tok = t.substr(at, end - at);
    if (tok.empty() || tok.size() > 4) return false;
    int v = 0;
    for (char c : tok) {
      if (c < '0' || c > '9') return false;
      v = v * 10 + (c - '0');
    }
    for (int seen : *out)
      if (seen == v) return false;
    out->push_back(v);
    at = end + 1;
  }
  *has = true;
  return true;
}

// The machine's cards the real driver shows, in its numbering, from CUDA_VISIBLE_DEVICES (`text`, null when unset) and the count the driver reports.
// Entry k of the result is the machine card the driver calls device k. `decided` false: the string is not a list of decimal numbers (a UUID, say), so the
// driver has chosen by a rule this does not read; the result is then the identity over the count.
inline std::vector<int> driver_cards(const char* text, int real_count, bool* decided) {
  std::vector<int> out;
  *decided = true;
  if (real_count < 0) real_count = 0;
  if (text) {
    const std::string t(text);
    size_t at = 0;
    bool numeric = true;
    std::vector<int> listed;
    while (!t.empty() && at <= t.size()) {
      size_t end = t.find(',', at);
      if (end == std::string::npos) end = t.size();
      const std::string tok = t.substr(at, end - at);
      bool ok = !tok.empty() && tok.size() <= 4;
      int v = 0;
      for (char c : tok) {
        if (c < '0' || c > '9') { ok = false; break; }
        v = v * 10 + (c - '0');
      }
      for (int seen : listed)
        if (ok && seen == v) ok = false;   // a repeat ends the list
      if (!ok) {
        // (a token that is a UUID or other form: not read; a repeated or empty one just ends the list the way the driver ends it)
        for (char c : tok)
          if (c == '-' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) numeric = false;
        break;
      }
      listed.push_back(v);
      at = end + 1;
    }
    if (!numeric) {
      *decided = false;
    } else {
      for (int v : listed)
        if (static_cast<int>(out.size()) < real_count) out.push_back(v);
      return out;
    }
  }
  for (int i = 0; i < real_count; ++i) out.push_back(i);
  return out;
}

enum class CardWhy : uint8_t {
  Ok,          // `index` is the real driver's device
  NoCard,      // the driver shows no such card (a machine with fewer cards than simulated devices)
  Hidden,      // the machine has the card, CUDA_VISIBLE_DEVICES keeps the real driver from showing it
  ListShort,   // VGPU_HOST_GPU_DEVICES does not name this slot
};

struct CardPick {
  int index = -1;
  CardWhy why = CardWhy::NoCard;
};

// Which real-driver device the simulated device in slot `slot` runs on. `list`: VGPU_HOST_GPU_DEVICES when set (else null); `cvd`: CUDA_VISIBLE_DEVICES (null:
// unset); `real_count`: cuDeviceGetCount.
inline CardPick pick_card(int slot, const std::vector<int>* list, const char* cvd, int real_count) {
  CardPick p;
  if (slot < 0 || real_count <= 0) return p;
  if (list) {
    if (static_cast<size_t>(slot) >= list->size()) { p.why = CardWhy::ListShort; return p; }
    const int want = (*list)[static_cast<size_t>(slot)];
    if (want >= real_count) return p;
    p.index = want;
    p.why = CardWhy::Ok;
    return p;
  }
  bool decided = true;
  const std::vector<int> cards = driver_cards(cvd, real_count, &decided);
  if (!decided) {   // (the string chose the cards for the driver already: slot S is its device S)
    if (slot >= real_count) return p;
    p.index = slot;
    p.why = CardWhy::Ok;
    return p;
  }
  for (size_t k = 0; k < cards.size(); ++k)
    if (cards[k] == slot) {
      p.index = static_cast<int>(k);
      p.why = CardWhy::Ok;
      return p;
    }
  // (the machine's card `slot` is not among the ones the driver shows: past its end, or hidden by the list)
  p.why = (cvd && *cvd) ? CardWhy::Hidden : CardWhy::NoCard;
  return p;
}

inline const char* card_why_text(CardWhy w) {
  switch (w) {
    case CardWhy::Ok: return "ok";
    case CardWhy::NoCard: return "the real driver shows no card for it";
    case CardWhy::Hidden: return "CUDA_VISIBLE_DEVICES keeps its card from the real driver";
    case CardWhy::ListShort: return "VGPU_HOST_GPU_DEVICES does not name it";
  }
  return "";
}

// Whether a launch of the simulated device in `slot` may touch memory outside that device's own window: the one rule that keeps coherence with
// several cards. An address argument inside another slot's window (kDeviceVaBase + k * stride) is a peer pointer; the card holds only its own window,
// so such a launch stays in the simulator (the card would fault, and a fault kills every context of the process: measured, va_probe step 9).
inline bool addr_in_other_window(uint64_t addr, int slot, uint64_t base, uint64_t stride, uint64_t windows) {
  if (addr < base || addr >= base + windows * stride) return false;
  return static_cast<int>((addr - base) / stride) != slot;
}

}  // namespace vgpu::hostgpu
