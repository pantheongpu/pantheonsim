**What this changes, and why**

**How it was checked**

- [ ] A test that CI runs covers it, and it fails without the change
- [ ] I ran the tests for what I touched (`ctest --test-dir build -R <regex>`) and say below which, and which I could not run (`nvcc`, `hipcc`, a card)
- [ ] Anything left unsupported fails with a message that names it, and is in `TODO.md`
- [ ] The documentation that describes this is updated (and `CHANGELOG.md`, for anything a user would notice)

**Where the behaviour comes from**

The document, header or card output behind a claim about what a GPU does.

**If this makes something faster**

- [ ] Results and counters are bit-identical; the old path stays behind its `VGPU_*` switch
- [ ] Numbers are medians of several runs on an idle machine (`tools/bench/bench.sh --ab`), not one run

**If this adds or changes a GPU profile**

- [ ] `verified: true` only if it was read from a card; the header says where each value came from
- [ ] `nvidia/docs/profiles.md` (or the AMD README), the counts in `docs/status.md`, and a row in `.github/workflows/workloads.yml`

CI is slow and some jobs run on the maintainer's runners; a pull request with a merge conflict is not tested, so rebase if the `merge-conflict` label appears.
