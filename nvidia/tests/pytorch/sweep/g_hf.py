"""Tiny Hugging Face models built from a config (no download): BERT, GPT-2,
ViT, T5, Llama-style, each forward and backward against the CPU; and
generation with a KV cache. Needs transformers."""
import torch

from harness import check, fwd_bwd, rnd, rint, flat

L = 'hf'
T = ('transformers',)
IDS = rint(1, 90, 3, 11, seed=2)
MASK = torch.ones(3, 11, dtype=torch.long)
MASK[1, 8:] = 0


def _run(mk, inputs_fn, d, train=False, tol_params=6):
    torch.manual_seed(0)
    m = mk()
    m.train(train)
    m = m.to(d)
    kw = {k: (v.to(d) if isinstance(v, torch.Tensor) else v) for k, v in inputs_fn().items()}
    out = m(**kw)
    loss = out.loss if getattr(out, 'loss', None) is not None else out.last_hidden_state.float().square().mean() if hasattr(out, 'last_hidden_state') else out.logits.float().square().mean()
    loss.backward()
    grads = [p.grad for p in list(m.parameters())[:tol_params]]
    return [loss.detach(), out.logits if hasattr(out, 'logits') and out.logits is not None else out.last_hidden_state, grads]


def bert():
    from transformers import BertConfig, BertForSequenceClassification
    return BertForSequenceClassification(BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                                    max_position_embeddings=32, num_labels=3, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0))


def gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    return GPT2LMHeadModel(GPT2Config(vocab_size=100, n_embd=32, n_layer=2, n_head=2, n_positions=32, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))


def llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    return LlamaForCausalLM(LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, max_position_embeddings=32))


def vit():
    from transformers import ViTConfig, ViTForImageClassification
    return ViTForImageClassification(ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                               intermediate_size=64, num_labels=5, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0))


def t5():
    from transformers import T5Config, T5ForConditionalGeneration
    return T5ForConditionalGeneration(T5Config(vocab_size=100, d_model=32, d_kv=16, d_ff=64, num_layers=2, num_heads=2, dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0))


def distil():
    from transformers import DistilBertConfig, DistilBertModel
    return DistilBertModel(DistilBertConfig(vocab_size=100, dim=32, n_layers=2, n_heads=2, hidden_dim=64, max_position_embeddings=32, dropout=0.0, attention_dropout=0.0))


def _cfg(eager):
    return {'attn_implementation': eager}


check(L, 'bert-tiny: sequence classification, forward and backward, with a padding mask', 3e-3, needs=T)(
    lambda d: _run(bert, lambda: dict(input_ids=IDS, attention_mask=MASK, labels=torch.tensor([0, 2, 1])), d))
check(L, 'gpt2-tiny: language modelling loss, forward and backward', 3e-3, needs=T)(
    lambda d: _run(gpt2, lambda: dict(input_ids=IDS, labels=IDS), d))
check(L, 'llama-tiny (RMSNorm, rotary embeddings, grouped-query attention, SwiGLU): forward and backward', 3e-3, needs=T)(
    lambda d: _run(llama, lambda: dict(input_ids=IDS, labels=IDS), d))
check(L, 'vit-tiny: image classification, forward and backward', 3e-3, needs=T)(
    lambda d: _run(vit, lambda: dict(pixel_values=rnd(2, 3, 32, 32, seed=3), labels=torch.tensor([1, 4])), d))
check(L, 't5-tiny: encoder-decoder loss, forward and backward', 3e-3, needs=T, tier='full')(
    lambda d: _run(t5, lambda: dict(input_ids=IDS, attention_mask=MASK, labels=rint(1, 90, 3, 6, seed=4)), d))
check(L, 'distilbert-tiny: forward and backward', 3e-3, needs=T, tier='full')(
    lambda d: _run(distil, lambda: dict(input_ids=IDS, attention_mask=MASK), d))


def _generate(mk):
    def run(d):
        torch.manual_seed(0)
        m = mk().to(d).eval()
        ids = IDS[:2, :5].to(d)
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=0, use_cache=True)
        return out
    return run


check(L, 'gpt2-tiny: greedy generation with the KV cache (the same tokens as the CPU)', 0, needs=T)(_generate(gpt2))
check(L, 'llama-tiny: greedy generation with the KV cache', 0, needs=T, tier='full')(_generate(llama))


def _gen_beam(d):
    torch.manual_seed(0)
    m = gpt2().to(d).eval()
    with torch.no_grad():
        return m.generate(IDS[:2, :5].to(d), max_new_tokens=6, num_beams=3, do_sample=False, pad_token_id=0)


check(L, 'gpt2-tiny: beam search (3 beams)', 0, needs=T, tier='full')(_gen_beam)


def _attn_impls(d):
    # The same tiny model through each attention implementation PyTorch offers.
    from transformers import GPT2Config, GPT2LMHeadModel
    outs = []
    for impl in ('eager', 'sdpa'):
        torch.manual_seed(0)
        m = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_embd=32, n_layer=2, n_head=2, n_positions=32, attn_implementation=impl)).to(d).eval()
        with torch.no_grad():
            outs.append(m(IDS.to(d)).logits)
    return outs


check(L, 'gpt2-tiny through the eager and the sdpa attention implementations', 3e-3, needs=T, tier='full')(_attn_impls)


def _half(d):
    torch.manual_seed(0)
    m = bert()
    if d == 'cpu':
        for p in m.parameters():
            p.data = p.data.half().float()
    m = m.to(d).eval() if d == 'cpu' else m.half().to(d).eval()
    with torch.no_grad():
        return m(input_ids=IDS.to(d), attention_mask=MASK.to(d)).logits.float()


check(L, 'bert-tiny held in half', 3e-2, needs=T, tier='full')(_half)
