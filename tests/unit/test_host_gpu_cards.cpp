// Turbo host-GPU mode on a machine with several cards (include/vgpu/host_gpu_cards.hpp, docs/perf-design/12-host-gpu-turbo.md section 17): which real device a
// simulated device slot runs on, how CUDA_VISIBLE_DEVICES and VGPU_HOST_GPU_DEVICES change it, and the peer-pointer rule. No card and no simulator state.
#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/host_gpu_cards.hpp"
#include "vgpu/memory.hpp"
#include "vtest.hpp"

using namespace vgpu::hostgpu;

namespace {
CardPick pick(int slot, const char* cvd, int count) { return pick_card(slot, nullptr, cvd, count); }
CardPick pick_list(int slot, const std::vector<int>& list, const char* cvd, int count) { return pick_card(slot, &list, cvd, count); }
}  // namespace

VTEST(card_list_parses_whole_lists_only) {
  std::vector<int> v;
  bool has = true;
  VCHECK(parse_card_list(nullptr, &v, &has) && !has && v.empty());
  VCHECK(parse_card_list("", &v, &has) && !has);
  VCHECK(parse_card_list("0", &v, &has) && has && v == std::vector<int>{0});
  VCHECK(parse_card_list("3,1,2", &v, &has) && has && v == (std::vector<int>{3, 1, 2}));
  for (const char* bad : {"0,", ",0", "0,,1", "a", "-1", "1,1", "0, 1", "0;1", "12345", "0x1"}) VCHECK(!parse_card_list(bad, &v, &has));
}

VTEST(each_slot_runs_on_the_card_of_its_number) {
  for (int slot = 0; slot < 4; ++slot) {
    const CardPick p = pick(slot, nullptr, 4);
    VCHECK(p.why == CardWhy::Ok);
    VCHECK_EQ(p.index, slot);
  }
}

VTEST(slots_beyond_the_machines_cards_stay_simulated) {
  VCHECK(pick(3, nullptr, 3).why == CardWhy::NoCard && pick(3, nullptr, 3).index < 0);
  VCHECK(pick(2, nullptr, 3).why == CardWhy::Ok);
  VCHECK(pick(0, nullptr, 0).index < 0);   // no card at all
  VCHECK(pick(1, nullptr, 1).index < 0);   // one card: slot 1 is simulated (today's behaviour for every slot past 0)
  VCHECK_EQ(pick(0, nullptr, 1).index, 0);   // ... and one card still serves slot 0 exactly as before
  VCHECK(pick(-1, nullptr, 4).index < 0);
}

VTEST(visible_devices_renumber_the_real_driver_not_the_slot) {
  // CUDA_VISIBLE_DEVICES=2,3: the driver shows two devices, its 0 is the machine's card 2. The simulated device in slot 2 is the machine's card 2.
  VCHECK_EQ(pick(2, "2,3", 2).index, 0);
  VCHECK_EQ(pick(3, "2,3", 2).index, 1);
  VCHECK(pick(0, "2,3", 2).why == CardWhy::Hidden);
  VCHECK(pick(1, "2,3", 2).why == CardWhy::Hidden);
  // reordered: "1,0": slot 1 (the program's device 0) is the driver's device 0
  VCHECK_EQ(pick(1, "1,0", 2).index, 0);
  VCHECK_EQ(pick(0, "1,0", 2).index, 1);
  // a single card chosen: only its slot has a card
  VCHECK_EQ(pick(1, "1", 1).index, 0);
  VCHECK(pick(0, "1", 1).why == CardWhy::Hidden);
  // the driver stopped the list early (an entry it cannot use): its count decides
  VCHECK_EQ(pick(0, "0,9,1", 1).index, 0);
  VCHECK(pick(1, "0,9,1", 1).index < 0);
  // an empty string hides everything
  VCHECK(pick(0, "", 0).index < 0);
  // a list that repeats a number ends at the repeat
  VCHECK_EQ(pick(1, "0,1,1,2", 2).index, 1);
  VCHECK(pick(2, "0,1,1,2", 2).index < 0);
}

VTEST(visible_devices_that_are_not_numbers_are_the_drivers_choice) {
  // a UUID: the driver chose by a rule this does not read; slot S is its device S
  bool decided = true;
  driver_cards("GPU-1234abcd-0000-0000-0000-000000000000", 1, &decided);
  VCHECK(!decided);
  VCHECK_EQ(pick(0, "GPU-1234abcd-0000-0000-0000-000000000000", 1).index, 0);
  VCHECK(pick(1, "GPU-1234abcd-0000-0000-0000-000000000000", 1).index < 0);
  VCHECK_EQ(pick(0, "MIG-abc", 1).index, 0);
  VCHECK_EQ(pick(0, "0,GPU-x", 2).index, 0);   // (a list with a UUID in it: not read as numbers)
}

VTEST(device_list_names_real_devices_and_overrides_the_rule) {
  VCHECK_EQ(pick_list(0, {2, 3}, nullptr, 4).index, 2);
  VCHECK_EQ(pick_list(1, {2, 3}, nullptr, 4).index, 3);
  VCHECK(pick_list(2, {2, 3}, nullptr, 4).why == CardWhy::ListShort);
  VCHECK(pick_list(1, {0, 7}, nullptr, 4).why == CardWhy::NoCard);   // names a device the driver does not show
  // the list names the driver's own numbers: CUDA_VISIBLE_DEVICES is not applied on top of it
  VCHECK_EQ(pick_list(0, {1}, "2,3", 2).index, 1);
}

VTEST(driver_cards_follow_the_drivers_count) {
  bool decided = false;
  VCHECK((driver_cards(nullptr, 3, &decided) == std::vector<int>{0, 1, 2}) && decided);
  VCHECK((driver_cards("3,1", 2, &decided) == std::vector<int>{3, 1}));
  VCHECK((driver_cards("3,1", 1, &decided) == std::vector<int>{3}));
  VCHECK(driver_cards("", 4, &decided).empty());
}

VTEST(peer_pointer_rule_uses_the_simulators_window_layout) {
  using vgpu::kDeviceVaBase;
  using vgpu::kDeviceVaStride;
  using vgpu::kDeviceVaWindows;
  const auto in_other = [&](uint64_t a, int slot) { return addr_in_other_window(a, slot, kDeviceVaBase, kDeviceVaStride, kDeviceVaWindows); };
  VCHECK(!in_other(0, 0));
  VCHECK(!in_other(0x1000, 1));                                   // a small integer
  VCHECK(!in_other(kDeviceVaBase + 0x1000, 0));                   // its own window
  VCHECK(in_other(kDeviceVaBase + 0x1000, 1));                    // slot 0's pointer in slot 1's launch
  VCHECK(in_other(kDeviceVaBase + 2 * kDeviceVaStride + 8, 0));
  VCHECK(!in_other(kDeviceVaBase + 2 * kDeviceVaStride + 8, 2));
  VCHECK(!in_other(kDeviceVaBase + kDeviceVaWindows * kDeviceVaStride, 0));   // past the last window
  VCHECK(in_other(kDeviceVaBase + kDeviceVaWindows * kDeviceVaStride - 1, 0));
  // the windows are disjoint, one per slot (the layout the card path relies on: measured on four T4s, each reserved at its own address)
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b)
      VCHECK_EQ(in_other(kDeviceVaBase + a * kDeviceVaStride + 123, b), a != b);
}

VTEST(exact_mode_serves_device_zero_only_and_turbo_every_window) {
  VCHECK_EQ(served_slot(false, 0, -1, 16), 0);
  VCHECK_EQ(served_slot(false, 0, 1, 16), 0);   // exact mode: window 0 and device 0 whatever the slot (as before the several-card work)
  VCHECK_EQ(served_slot(false, 1, 1, 16), -1);
  VCHECK_EQ(served_slot(false, 3, -1, 16), -1);
  VCHECK_EQ(served_slot(true, 0, -1, 16), 0);
  VCHECK_EQ(served_slot(true, 2, -1, 16), 2);
  VCHECK_EQ(served_slot(true, 0, 3, 16), 3);    // CUDA_VISIBLE_DEVICES=3: the program's device 0 is in window 3
  VCHECK_EQ(served_slot(true, 0, 15, 16), 15);
  VCHECK_EQ(served_slot(true, 0, 16, 16), -1);  // past the simulator's windows
  VCHECK_EQ(served_slot(true, -2, -1, 16), -1);
}

VTEST_MAIN
