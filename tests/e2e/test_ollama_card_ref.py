"""ollama_card_ref.py's check against a fake Ollama server: framework-free, no Ollama, no GPU.

The committed reference (nvidia/tests/data/ollama_smollm2_rtx3060.card.json) is served back as the reply
and must pass; a changed token, a log-probability moved past the tolerance and a missing log-probability
must each fail; a model with another digest, or another name, is skipped and not compared; a server error
fails. This is what makes a drift of the simulated GPUs from the card's reply fail e2e_ollama.
"""
import copy
import http.server
import json
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '..', '..', 'nvidia', 'tests', 'data', 'ollama_smollm2_rtx3060.card.json')
TOOL = os.path.join(HERE, 'ollama_card_ref.py')
ref = json.load(open(REF))
state = {}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, {'models': [{'name': ref['model'], 'model': ref['model'], 'digest': state['digest']}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if state.get('error'):
            return self.reply(500, {'error': 'model failed to load'})
        self.reply(200, {'response': state['response'],
                         'logprobs': [{'token': t, 'logprob': lp} for t, lp in zip(ref['tokens'], state['logprobs'])]})


def run(api, model=None, **change):
    state.update({'digest': ref['digest'], 'response': ref['response'], 'logprobs': list(ref['logprobs']), 'error': False})
    state.update(change)
    p = subprocess.run([sys.executable, TOOL, 'check', '--api', api, '--model', model or ref['model'], '--ref', REF],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    srv = http.server.HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    api = 'http://127.0.0.1:%d' % srv.server_port
    bad = []
    lp_moved = list(ref['logprobs'])
    lp_moved[3] += 0.2
    lp_near = list(ref['logprobs'])
    lp_near[3] += 0.01
    cases = [
        ('the card\'s own reply passes', {}, None, 0, 'ok    the 16-token'),
        ('a small log-probability difference passes', {'logprobs': lp_near}, None, 0, 'within'),
        ('a changed text fails', {'response': ref['response'].replace('Cit', 'Cat', 1)}, None, 1, 'FAIL  the reply differs'),
        ('a log-probability 0.2 off fails', {'logprobs': lp_moved}, None, 1, 'FAIL  a log-probability'),
        ('a missing log-probability fails', {'logprobs': ref['logprobs'][:-1]}, None, 1, 'FAIL'),
        ('another digest is skipped', {'digest': 'f' * 64}, None, 0, 'SKIP'),
        ('another model is skipped', {}, 'other:1b', 0, 'SKIP'),
        ('a server error fails', {'error': True}, None, 1, ''),
    ]
    for what, change, model, rc, text in cases:
        got_rc, out = run(api, model, **change)
        if got_rc != rc or text not in out:
            bad.append('%s: exit %d, wanted %d; output %r' % (what, got_rc, rc, out[-300:]))
    # Every card's file (rtx3060, a10g, t4) is a whole record of the same request: run_ollama.sh picks one by profile.
    import glob
    # run_ollama.sh's card_check names these: a Turing profile is held to the T4's reply, an sm_86 one to the A10G's, the rest to the RTX 3060's.
    for slug in ('t4', 'a10g', 'rtx3060'):
        if not os.path.exists(os.path.join(os.path.dirname(REF), 'ollama_smollm2_%s.card.json' % slug)):
            bad.append('run_ollama.sh names ollama_smollm2_%s.card.json, which does not exist' % slug)
    for f in sorted(glob.glob(os.path.join(os.path.dirname(REF), 'ollama_smollm2_*.card.json'))):
        r = json.load(open(f))
        for k in ('card', 'ollama', 'model', 'digest', 'prompt', 'response', 'tokens', 'logprobs'):
            if k not in r:
                bad.append('%s has no %s' % (os.path.basename(f), k))
        if r.get('model') != ref['model'] or r.get('prompt') != ref['prompt'] or len(r.get('logprobs', [])) != len(ref['logprobs']):
            bad.append('%s records another request than the RTX 3060 file' % os.path.basename(f))
        if r.get('digest') != ref['digest']:
            bad.append('%s was recorded for another model digest' % os.path.basename(f))
    srv.shutdown()
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('ok   ollama card reference: %d drift cases behave as documented' % len(cases))
    return 0


if __name__ == '__main__':
    sys.exit(main())
