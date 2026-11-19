#!/usr/bin/env python3
"""Ollama's reply on a real card, as a reference for the simulated ones (run_ollama.sh).

run_ollama.sh compares each simulated GPU with Ollama's CPU backend for three tokens. The CPU is not what a
user meets on a card, and three tokens say little, so this adds the reply a real card gave for sixteen greedy
tokens of a fixed prompt, with the log-probability of every token it chose, recorded in
nvidia/tests/data/ollama_smollm2_rtx3060.card.json:

  ollama_card_ref.py check  --api URL --model M --ref F    one request to a running server; prints
                                                           "ok ..." / "FAIL ..." lines, exit 1 on a FAIL
  ollama_card_ref.py record --api URL --model M --out F --card NAME
                                                           the same request on a real card: three requests,
                                                           which must agree exactly on the text, and whose
                                                           log-probabilities set the tolerance

The text must match exactly. A log-probability may differ by LOGPROB_TOL (nats): Ollama's own CPU backend
differs from run to run by about 0.005 and a card's cold first request by 0.009, and a changed token shows as
a difference of whole nats. A model whose digest differs from the recorded one is not the model the card ran
(the registry changed it), so the check says so and does not compare.
"""
import json
import sys
import urllib.request

PROMPT = 'The capital of France is'
NUM_PREDICT = 16
LOGPROB_TOL = 0.05


def post(api, path, body=None):
    req = urllib.request.Request(api + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def generate(api, model):
    d = post(api, '/api/generate', {'model': model, 'prompt': PROMPT, 'stream': False, 'logprobs': True,
                                    'options': {'num_predict': NUM_PREDICT, 'temperature': 0, 'seed': 1}})
    if 'response' not in d:
        raise RuntimeError('no reply: ' + str(d.get('error', d))[:200])
    return d['response'], [t['logprob'] for t in d.get('logprobs', [])], [t['token'] for t in d.get('logprobs', [])]


def digest(api, model):
    for m in post(api, '/api/tags').get('models', []):
        if model in (m.get('name'), m.get('model')):
            return m.get('digest', '')
    return ''


def compare(ref, text, logprobs):
    """-> list of (ok, message) lines."""
    out = []
    out.append((text == ref['response'], 'the %d-token reply is the card\'s' % NUM_PREDICT if text == ref['response']
                else 'the reply differs: %r, the card\'s %r' % (text, ref['response'])))
    if len(logprobs) != len(ref['logprobs']):
        out.append((False, '%d log-probabilities, the card\'s %d' % (len(logprobs), len(ref['logprobs']))))
    else:
        worst = max((abs(a - b) for a, b in zip(logprobs, ref['logprobs'])), default=0.0)
        ok = worst <= LOGPROB_TOL
        out.append((ok, 'log-probabilities of the chosen tokens within %.3g of the card\'s (largest difference %.3g)' % (LOGPROB_TOL, worst)
                    if ok else 'a log-probability is %.3g from the card\'s, allowed %.3g' % (worst, LOGPROB_TOL)))
    return out


def check(api, model, ref_path):
    with open(ref_path) as f:
        ref = json.load(f)
    if model != ref['model']:
        print('SKIP card reference is for %s, not %s' % (ref['model'], model))
        return 0
    have = digest(api, model)
    if have and have != ref['digest']:
        print('SKIP card reference is for %s digest %s; this one is %s' % (model, ref['digest'][:12], have[:12]))
        return 0
    text, lps, _ = generate(api, model)
    bad = 0
    for ok, msg in compare(ref, text, lps):
        print(('ok    ' if ok else 'FAIL  ') + msg)
        bad += not ok
    return 1 if bad else 0


def record(api, model, out_path, card):
    runs = [generate(api, model) for _ in range(3)]
    if len({r[0] for r in runs}) != 1 or len({tuple(r[2]) for r in runs}) != 1:
        print('refusing to record: the card gave different tokens on three runs')
        return 1
    spread = max(max(abs(a - b) for a, b in zip(r[1], runs[1][1])) for r in runs)
    if spread * 4 > LOGPROB_TOL:
        print('refusing to record: the card\'s log-probabilities differ by %.3g between runs (tolerance %.3g)' % (spread, LOGPROB_TOL))
        return 1
    ref = {'card': card, 'ollama': post(api, '/api/version')['version'], 'model': model, 'digest': digest(api, model),
           'prompt': PROMPT, 'num_predict': NUM_PREDICT, 'options': 'temperature 0, seed 1',
           'response': runs[-1][0], 'tokens': runs[-1][2], 'logprobs': [round(x, 6) for x in runs[-1][1]],
           'note': 'Regenerate on a real GPU: tests/e2e/ollama_card_ref.py record. Tolerance: LOGPROB_TOL in that script.'}
    with open(out_path, 'w') as f:
        json.dump(ref, f, indent=1)
        f.write('\n')
    print('wrote', out_path)
    return 0


def main(argv):
    if not argv or argv[0] not in ('check', 'record'):
        print(__doc__)
        return 2
    a = dict(zip(argv[1::2], argv[2::2]))
    api, model = a.get('--api'), a.get('--model')
    if argv[0] == 'check':
        return check(api, model, a['--ref'])
    return record(api, model, a['--out'], a['--card'])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
