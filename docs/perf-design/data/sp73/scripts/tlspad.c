// LD_PRELOAD: PAD bytes of thread-local storage (compile with -DPAD=n), which moves every thread's stack top by as much
__thread char sp72_pad[PAD] __attribute__((tls_model("initial-exec"), aligned(16)));
char* sp72_touch(void) { return sp72_pad; }
