"""How sweep.py leaves the process: framework-free, no torch and no simulator needed.

sweep.py ends with os._exit after flushing, because PyTorch's autograd worker thread can take the GIL while
Py_FinalizeEx runs and abort the process (exit 134) after every result is printed. This test copies sweep.py
next to a fake sweep/harness.py and checks, in a child interpreter each:
  * the exit status is what sys.exit(harness.main(...)) gave: 0, the returned int, SystemExit's code, 1 for an
    uncaught exception (with its traceback on stderr) and for a non-integer exit value;
  * everything printed is delivered when stdout is a pipe (nothing is lost in a buffer);
  * an atexit handler does not run, i.e. the interpreter really was not finalised (this is what fails if the
    os._exit is turned back into sys.exit).
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = '''
import atexit, os, sys
atexit.register(lambda: print("ATEXIT-RAN"))
def main(argv, loaders):
    print("result line", flush=False)          # deliberately left in the buffer
    print("to stderr", file=sys.stderr, end="")
    mode = os.environ["FAKE_MODE"]
    if mode == "ok": return 0
    if mode == "three": return 3
    if mode == "systemexit": raise SystemExit(5)
    if mode == "systemexit_none": raise SystemExit
    if mode == "systemexit_text": raise SystemExit("bye")
    if mode == "none": return None
    raise RuntimeError("boom")
'''
CASES = [('ok', 0), ('three', 3), ('systemexit', 5), ('systemexit_none', 0), ('systemexit_text', 1), ('none', 0), ('boom', 1)]


def main():
    tmp = tempfile.mkdtemp(prefix='sweep-exit-')
    failures = []
    try:
        os.mkdir(os.path.join(tmp, 'sweep'))
        shutil.copy(os.path.join(HERE, 'sweep.py'), tmp)
        with open(os.path.join(tmp, 'sweep', 'harness.py'), 'w') as f:
            f.write(FAKE)
        for mode, want in CASES:
            env = dict(os.environ, FAKE_MODE=mode)
            p = subprocess.run([sys.executable, os.path.join(tmp, 'sweep.py')], env=env, capture_output=True, text=True)
            problems = []
            if p.returncode != want:
                problems.append(f'exit status {p.returncode}, wanted {want}')
            if 'result line' not in p.stdout:
                problems.append('stdout lost the buffered result line')
            if 'to stderr' not in p.stderr:
                problems.append('stderr lost its text')
            if 'ATEXIT-RAN' in p.stdout:
                problems.append('an atexit handler ran: the interpreter was finalised')
            if mode == 'boom' and 'RuntimeError: boom' not in p.stderr:
                problems.append('no traceback for the uncaught exception')
            if mode == 'systemexit_text' and 'bye' not in p.stderr:
                problems.append('the SystemExit text was not printed')
            print(('ok   ' if not problems else 'FAIL ') + mode + ''.join('\n     ' + x for x in problems))
            failures += problems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
