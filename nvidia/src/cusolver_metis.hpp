// METIS's nested dissection, for cusolverSpXcsrmetisndHost. The vendored
// METIS (nvidia/third_party/metis) is built with 64-bit indices and kept
// behind this one function so that its macros and types stay out of the rest
// of the library.
#pragma once

#include <cstdint>
#include <vector>

namespace vgpu::cusolver_metis {

// METIS_NodeND on the graph (xadj, adjncy) of n vertices, default options
// when `options` is null (else METIS_NOPTIONS int64_t values, as METIS reads
// them). On success perm[i] is METIS's perm and iperm[i] its inverse.
// Returns METIS's status (1 is METIS_OK).
int node_nd(int64_t n, const std::vector<int64_t>& xadj, const std::vector<int64_t>& adjncy, const int64_t* options,
            std::vector<int64_t>* perm, std::vector<int64_t>* iperm);

// The number of entries METIS reads from an options array.
int64_t option_count();

}  // namespace vgpu::cusolver_metis
