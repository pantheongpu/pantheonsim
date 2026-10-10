// See cusolver_metis.hpp. METIS 5.1.0 (Apache-2.0, nvidia/third_party/metis/README.md).
#include "cusolver_metis.hpp"

#include <metis.h>

#include <mutex>

namespace vgpu::cusolver_metis {

int node_nd(int64_t n, const std::vector<int64_t>& xadj, const std::vector<int64_t>& adjncy, const int64_t* options,
            std::vector<int64_t>* perm, std::vector<int64_t>* iperm) {
  static_assert(sizeof(idx_t) == sizeof(int64_t), "the vendored METIS is the 64-bit build");
  perm->assign((size_t)n, 0);
  iperm->assign((size_t)n, 0);
  idx_t nvtxs = (idx_t)n;
  // METIS keeps no state between calls that this library's callers share (its
  // random generator is thread-local, see the README), but its working memory
  // is large: one call at a time keeps the peak at one graph's worth.
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  return METIS_NodeND(&nvtxs, const_cast<idx_t*>(xadj.data()), const_cast<idx_t*>(adjncy.data()), nullptr,
                      const_cast<idx_t*>(options), perm->data(), iperm->data());
}

int64_t option_count() { return METIS_NOPTIONS; }

}  // namespace vgpu::cusolver_metis
