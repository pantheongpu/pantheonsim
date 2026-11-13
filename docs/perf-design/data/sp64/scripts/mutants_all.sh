#!/bin/bash
# mutants_all.sh <src> <build> <work>: round sp64's mutants, each under the machine lock (scripts/mutants.sh).
s=$1; b=$2; w=$3; m="$(dirname "$0")/mutants.sh"
L() { flock /tmp/pantheonsim-builder.lock nice -n 19 bash "$m" "$s" "$b" "$w" "$@"; }
L pristine
# A section whose compile runs is hooked again (the hook loop while Queued, as before the round).
L queued_hooked hpp 's/return s <= static_cast<int>(State::Counting) || s == static_cast<int>(State::Ready);/return s <= static_cast<int>(State::Ready);/'
# The compile log is not printed.
L no_compile_log cpp 's/for (const std::string\& l : compile_log) std::fputs(l.c_str(), out);/(void)0;/'
# The compile log is printed before the summary.
L log_before_summary cpp 's/^  if (link_requested())$/  for (const std::string\& l : compile_log) std::fputs(l.c_str(), out);\n  if (link_requested())/'
# A precompiled header that is there is never read.
L pch_never_used cpp 's/if (::stat((h + ".gch").c_str(), \&gst) == 0 \&\& read_file(h, have) \&\& have == text) {/if (false) {/'
# The precompiled header is never made.
L pch_never_built cpp 's/if (p_->pch_tried || p_->pch_h.empty()) return false;/return false;/'
# VGPU_JIT_PCH=0 is not read.
L pch_ignores_env cpp 's/if (env("VGPU_JIT_PCH") == "0") c.pch = false;//'
# The header is made before the first compile instead of after it (the first compile then reads it: counts differ).
L pch_made_first cpp 's/^  std::vector<std::string> cmd{p_->cxx_path};$/  if (pch_h.empty() \&\& build_pch()) { struct stat g2; std::string h2 = p_->pch_h; if (::stat((h2 + ".gch").c_str(), \&g2) == 0) pch_h = h2; }\n  std::vector<std::string> cmd{p_->cxx_path};/'
