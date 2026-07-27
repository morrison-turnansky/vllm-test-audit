# vLLM Test Audit

A Claude Code plugin that finds "coincidentally correct" test assertions in the [vLLM](https://github.com/vllm-project/vllm) test suite — tests that pass today but are fragile because they rely on exact numeric or text equality without a principled contract guaranteeing that equality.

These assertions can break silently when PyTorch changes floating-point behavior.

## How it works

The plugin runs a two-phase pipeline:

1. **Phase 1 — Evidence generation** (`audit-agent`): Enumerates test functions from a PR, directory, or file. Evaluates each against three criteria (weak oracle, realistic breakage, no strong contract) and classifies it as `COINCIDENTALLY_CORRECT`, `STRONG_CONTRACT`, or `NOT_REALISTIC`.

2. **Phase 2 — Adversarial verification** (`review-agent`): Re-reads the source for each candidate flagged as coincidentally correct, challenges the Phase 1 reasoning, and either agrees or reclassifies with evidence.

A `coordinator-agent` orchestrates both phases end-to-end.


## Usage

Invoke the coordinator agent to run the full pipeline:

```
Use the vllm-test-audit coordinator to audit PR #47185
```

Or target specific scopes:

```
Use the vllm-test-audit audit-agent on tests/v1/e2e/
Use the vllm-test-audit audit-agent on tests/compile/test_full_graph.py
Use the vllm-test-audit audit-agent on tests/compile/test_full_graph.py::test_cascade_attention
```

### Output

- `audit-cc.json` — tests classified as coincidentally correct
- `audit-not-cc.json` — tests classified as strong contract or not realistic
- `review-cc.json` — adversarial verification results for CC candidates
