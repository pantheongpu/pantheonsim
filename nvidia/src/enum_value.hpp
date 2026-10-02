// The integer an enum argument holds, read without loading it as the enum.
//
// A caller may pass a value its enum does not declare -- a bad operation
// that the library must refuse, or a type a newer header added (CUDA 12.0's
// cudaDataType has no FP4). Loading such a value as the enum type is
// undefined behaviour, which UBSan reports (and halts on) wherever the
// argument lives in memory: captured by reference, read through a pointer,
// kept in a struct. So an argument that may be out of range is read through
// here, its bytes copied into an int, and only checked as an int; it is used
// as the enum only once it is known to be one of its values.
#pragma once

#include <cstring>

template <class E>
inline int enum_value(const E& e) {
  static_assert(sizeof(E) == sizeof(int), "an enum of int's size");
  int v;
  std::memcpy(&v, &e, sizeof v);
  return v;
}
