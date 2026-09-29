// Every function of AMD SMI's library that amd_smi.cpp does not answer:
// AMDSMI_STATUS_NOT_SUPPORTED, as a card without the feature answers.
//
// AMD's Python package binds all of the library's functions when imported,
// so each must exist. These are weak: amd_smi.cpp's definitions replace the
// ones it has. The list is AMD's header's (amd/tools/amdsmi-symbols.py).
#define AMDSMI_SYMBOL(name) \
  extern "C" __attribute__((weak, visibility("default"))) int name(...) { return 2; }
#include "amd_smi_symbols.inc"
