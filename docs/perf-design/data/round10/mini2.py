import time
import torch
def T(label, f):
    t=time.time(); f(); torch.cuda.synchronize(); print(f"{label:34s} {time.time()-t:.3f}s")
a=torch.randn(256,256,device="cuda"); torch.cuda.synchronize()
T("add (1st elementwise kernel)", lambda: a+1)
T("add again (same kernel)", lambda: a+1)
T("mul (new kernel, same file)", lambda: a*2)
T("relu (new kernel)", lambda: a.relu())
T("sum reduction (new kernel)", lambda: a.sum())
T("softmax (new kernel)", lambda: torch.softmax(a,-1))
T("sum again", lambda: a.sum())
T("matmul 256", lambda: a@a)
T("layer_norm", lambda: torch.nn.functional.layer_norm(a,(256,)))
T("layer_norm again", lambda: torch.nn.functional.layer_norm(a,(256,)))
