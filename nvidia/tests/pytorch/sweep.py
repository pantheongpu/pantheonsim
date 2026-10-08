"""PyTorch's CUDA build, unmodified, on a simulated NVIDIA GPU: the sweep.

Where ops.py checks one operator at a time and models.py whole models, this
widens the net over what real PyTorch software uses: nn layers forward and
backward, losses, optimizers, torch.linalg / fft / sparse corners, AMP, CUDA
graphs, torch.compile, each scaled_dot_product_attention backend forced in
turn, DataParallel on two simulated devices, torchvision / torchaudio
operators and tiny Hugging Face models (where those packages are installed).
Each check runs on the simulated GPU and on the CPU from the same inputs;
one line each, "ok <name>" or "FAIL <name>: <why>". Which groups and which
tier: sweep/harness.py. nvidia/docs/pytorch.md says how it is run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep'))
import harness  # noqa: E402

GROUPS = {
    'nn': 'g_nn', 'losses': 'g_losses', 'rnn': 'g_rnn', 'attention': 'g_attention', 'ops': 'g_ops',
    'optim': 'g_optim', 'linalg': 'g_linalg', 'fft': 'g_fft', 'sparse': 'g_sparse', 'amp': 'g_amp',
    'graphs': 'g_graphs', 'compile': 'g_compile', 'multi': 'g_multi', 'vision': 'g_vision',
    'audio': 'g_audio', 'hf': 'g_hf', 'sdpa': 'g_sdpa',
}
sys.exit(harness.main(sys.argv[1:], {g: m for g, m in GROUPS.items() if os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep', m + '.py'))}))
