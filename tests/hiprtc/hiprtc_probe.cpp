// hiprtc as a program calls it, against whatever libhiprtc the loader finds: amd/tests/e2e/run_hiprtc_cache.sh runs this with the
// simulator's libhiprtc in front of ROCm's and compares what comes back with the cache off, cold and warm.
//   hiprtc_probe child [--src S | --src-file F] [--name N] [--header F=T] [--opt O] [--expr E] [--how H] [--next ...] [--repeat N]
#include "session.hpp"

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "child")) {
    std::fprintf(stderr, "usage: %s child [--src S | --src-file F] [--name N] [--header F=T] [--opt O] [--expr E] [--how H] [--next ...] [--repeat N]\n", argv[0]);
    return 2;
  }
  return hiprtc_session::child(argc, argv);
}
