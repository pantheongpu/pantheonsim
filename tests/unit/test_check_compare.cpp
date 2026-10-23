// Unit tests for the comparison behind `vgpu check` (src/cli/check_compare.hpp): which lines are dropped, which are set aside, and how a
// difference is classified. Pure string functions: no card, no program.
#include <cmath>
#include <string>
#include <vector>

#include "check_compare.hpp"
#include "vtest.hpp"

using namespace vgpu::check;

namespace {
const std::vector<std::string> kWords = {"NVIDIA A10G", "sm_86"};
Compared cmp(const std::string& a, const std::string& b, const std::vector<bool>* mask = nullptr) {
  return compare(filter(a, kWords), filter(b, kWords), mask);
}
}  // namespace

VTEST(identical_bytes) {
  VCHECK(cmp("a 1\nb 2\n", "a 1\nb 2\n").verdict == Verdict::Identical);
}

VTEST(timing_lines_are_dropped) {
  const Compared c = cmp("result 42\nelapsed time: 0.5 ms\n", "result 42\nelapsed time: 9.1 ms\n");
  VCHECK(c.verdict == Verdict::TimingOnly);
  const Filtered f = filter("x\nGPU Time in seconds:\n0.123\ny\n", kWords);
  VCHECK_EQ(f.kept.size(), 2u);  // the number after a timing line belongs to it
  VCHECK(is_timing_line("Kernel took 3 s"));
  VCHECK(is_timing_line("bandwidth 120 GB/s"));
  VCHECK(is_timing_line("ptr = 0x7f12345678"));
  VCHECK(!is_timing_line("sum = 12345"));
  VCHECK(!is_timing_line("Test passed"));
}

VTEST(a_verdict_line_with_a_time_is_not_dropped) {
  // Dropping it would let FAILED on one side hide behind PASSED on the other.
  VCHECK(cmp("PASSED in 0.3 s\n", "PASSED in 7.7 s\n").verdict == Verdict::TimingOnly);
  VCHECK(cmp("PASSED in 0.3 s\n", "FAILED in 0.3 s\n").verdict == Verdict::Differs);
  VCHECK(is_verdict_line("Result: FAILED"));
}

VTEST(identity_lines_are_set_aside) {
  const Compared c = cmp("Device 0: NVIDIA A10G\nDriver Version: 13.2\nanswer 7\n", "Device 0: NVIDIA A10G\nDriver Version: 13.0\nanswer 7\n");
  VCHECK(c.verdict == Verdict::IdentityOnly);
  VCHECK(cmp("Device 0: NVIDIA A10G\nanswer 7\n", "Device 0: NVIDIA A10G\nanswer 8\n").verdict != Verdict::IdentityOnly);
}

VTEST(numbers_that_differ_are_not_identical) {
  const Compared c = cmp("value 1.000000\nn 5\n", "value 1.000001\nn 5\n");
  VCHECK(c.verdict == Verdict::NumericClose);
  VCHECK_EQ(c.numbers_differ, 1u);
  VCHECK(c.max_rel > 0.9e-6 && c.max_rel < 1.1e-6);
  const Compared same_value = cmp("v 1e-3\n", "v 1.0e-3\n");  // different spelling of the same number
  VCHECK(same_value.verdict == Verdict::NumericClose && same_value.max_rel == 0);
}

VTEST(different_text_is_different) {
  VCHECK(cmp("alpha 1\n", "beta 1\n").verdict == Verdict::Differs);
  VCHECK(cmp("a\nb\n", "a\n").verdict == Verdict::Differs);
  const Compared c = cmp("a\nb\n", "a\nc\n");
  VCHECK_EQ(c.first_a, std::string("b"));
  VCHECK_EQ(c.first_b, std::string("c"));
}

VTEST(nan_and_inf) {
  VCHECK(cmp("x nan\n", "x nan\n").verdict == Verdict::Identical);
  const Compared c = cmp("x nan\n", "x 1.0\n");
  VCHECK(c.verdict == Verdict::NumericClose);
  VCHECK(c.max_rel == HUGE_VAL);
  VCHECK(cmp("information 3\n", "information 4\n").verdict == Verdict::NumericClose);  // "inf" inside a word is not a number
}

VTEST(noisy_lines_of_the_card_are_masked) {
  const Filtered a = filter("seed 11\nresult 5\n", kWords), b = filter("seed 99\nresult 5\n", kWords);
  bool aligned = false;
  const std::vector<bool> m = noisy_lines(a, b, &aligned);
  VCHECK(aligned);
  VCHECK(m.size() == 2 && m[0] && !m[1]);
  // The simulator's line 0 differs from the card's, but that line is masked; line 1 is not.
  VCHECK(compare(a, filter("seed 3\nresult 5\n", kWords), &m).verdict == Verdict::TimingOnly);  // masked: not "identical"
  VCHECK(compare(a, filter("seed 3\nresult 6\n", kWords), &m).verdict == Verdict::NumericClose);
  bool al2 = true;
  noisy_lines(a, filter("one line\n", kWords), &al2);
  VCHECK(!al2);
}

VTEST(crlf_and_trailing_blank_lines) {
  VCHECK(cmp("a 1\r\nb 2\r\n\n\n", "a 1\nb 2\n").verdict == Verdict::TimingOnly);  // the blank lines were dropped
}

VTEST_MAIN
