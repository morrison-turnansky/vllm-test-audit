---
name: auto-fix
description: >-
  Remediation playbook for tests classified COINCIDENTALLY_CORRECT by the vLLM
  numeric-stability audit. Maps each fragile-oracle finding to a concrete fix —
  tolerance oracle, batch-invariant mode, golden strings, or principled
  threshold — with recipes, alignment pitfalls, and verification steps. Use when
  fixing a brittle test, replacing exact-equality assertions, adding
  VLLM_BATCH_INVARIANT, or acting on audit-cc.json / review-cc.json output.
---

# Coincidentally-Correct Remediation

How to fix a test the audit flagged `COINCIDENTALLY_CORRECT`. Load [audit-contract](../audit-contract/SKILL.md) first — this skill reuses its criteria (c1/c2/c3), clause numbers, and classifications. A fix is complete only when it defeats **criterion 1 (weak oracle)** or **criterion 3 (no strong contract)** — i.e. the assertion either tolerates realistic drift or is backed by a contract that forbids it.

Do not "fix" a test by loosening an assertion until it passes today. Loosening without a principle just moves the coincidence. Every remedy below either installs a real contract or replaces the oracle with one whose tolerance is justified.

## Pick a remedy by what the test compares

| The two executions differ on… | Remedy | Clause it satisfies |
|---|---|---|
| Restoration path, streaming, duplicate-in-batch, spec decode, batch size (BS=1 vs BS=N), cascade vs non-cascade attention (batch-geometry only), single request vs first-of-batch (same engine, same parallelism/compile strategy — pure batch composition/transport/state) | **Batch-invariant mode** (§2) | Clauses 5/6/7/9/11/12/13 |
| Cross-runtime (vLLM vs HF), eager vs compile, TP/PP/EP, DCP degree, prompt vs prompt_embeds | **Tolerance oracle** (§1) | c1 fails — oracle now tolerates drift |
| Single deterministic engine, output you can pin | **Golden strings** (§3) | c1 fails — pinned reference, not a live second run |
| Acceptance/match ratios (spec decode, quantization) | **Principled threshold** (§4) | c1 fails — threshold is justified |

**BI is the preferred first choice whenever it applies.** It defeats criterion 3 by installing a real bitwise contract, which is a stronger fix than tolerance (which only defeats criterion 1 by accepting drift). Reach for §1 tolerance only when the differing axis is *structural* — a different runtime, compile strategy, or parallelism topology/degree — not when it's pure batch composition/geometry under an otherwise-identical engine and strategy. BI mode does **not** rescue a cross-runtime, cross-compile-strategy, or cross-parallelism-degree comparison (§2 caveat) — don't reach for it there even though it's the general first choice.

When more than one axis differs (e.g. cross-runtime **and** prompt_embeds, or TP/PP/EP **and** batch geometry), fix each axis with its own remedy — don't let a structural axis's need for tolerance talk you out of using BI for a co-occurring pure-batch-geometry axis, and vice versa.

## §1 Tolerance oracle (most common)

Replace exact text/token equality with a tolerance helper. This is the default fix for generated-output comparisons and the **only** correct fix for cross-runtime (vLLM vs HF) comparisons — nothing requires two different kernel stacks to be text-identical.

Swap the oracle:

| From (exact) | To (tolerance) |
|---|---|
| `check_outputs_equal` | `check_logprobs_close` |
| `assert out.text == ...` | `check_logprobs_close` |
| embedding exact / dot-equality | `check_embeddings_close` |
| raw tensor equality | `torch.testing.assert_close(atol=, rtol=)` |

`check_logprobs_close` allows token divergence and only checks top-k logprob-id overlap (and only at offsets where the sampled tokens differ). Generation must request logprobs:

```python
from ..models.utils import check_logprobs_close

num_logprobs = 5
hf_outputs = hf_model.generate_greedy_logprobs_limit(
    example_prompts, max_tokens, num_logprobs
)
vllm_outputs = vllm_model.generate_greedy_logprobs(
    example_prompts, max_tokens, num_logprobs
)
check_logprobs_close(
    outputs_0_lst=hf_outputs,
    outputs_1_lst=vllm_outputs,
    name_0="hf",
    name_1="vllm",
)
```

### Alignment pitfalls (read before shipping a §1 fix)

Switching helpers changes what each side returns. Getting this wrong produces a *passing-looking* fix that actually compares misaligned positions, or a spurious failure.

- **Prompt-inclusion asymmetry.** HF `generate_greedy` returns **prompt + generated** tokens; HF `generate_greedy_logprobs_limit` returns **generated-only**. vLLM completions are always **generated-only**. A helper that prepends the prompt to align the *text* oracle (e.g. a `_fix_prompt_embed_outputs`-style shim) becomes **wrong** under the logprobs oracle — it shoves the vLLM side out of alignment so position 0 compares a prompt token against a generated token ("Matched tokens: []"). When moving to logprobs, delete the prompt-prefixing shim; both sides are already generated-only.
- **Shared helpers.** If a shim is reused by another test still on the text oracle, restore it to its original signature rather than editing it in place — don't break the text-path caller.
- **Intentional offset.** When the two sides legitimately start at different offsets (e.g. one skips BOS), use `check_logprobs_close(..., num_outputs_0_skip_tokens=N)` instead of hand-slicing.
- **Don't over-tolerate.** `check_logprobs_close` with a tiny `num_logprobs` and matching tokens is still a meaningful check. Do not also pass `always_check_logprobs=False` reasoning as license to skip comparison entirely.

## §2 Batch-invariant mode

For same-engine, same-strategy paths that *should* be bitwise-identical but aren't guaranteed without it — restoration (clause 5), streaming vs non-streaming (clause 6), duplicate-in-batch (clause 7), spec decode (clause 9), batch size BS=1 vs BS=N (clause 11), cascade vs non-cascade attention when the only differing axis is batch geometry (clause 12), single request vs first-of-batch (clause 13). This is the **preferred remedy** whenever the axis is pure batch composition/geometry/transport — it installs a real contract instead of just tolerating drift. Force batch-invariant kernels on **both** compared paths, then exact equality becomes a real contract.

The code base has various ways to set VLLM_BATCH_INVARIANT. Look at the surrouinding code of the test and copy that. 

Set it so it reaches the actual generation process:
- directly in the test (or via `monkeypatch.setenv`) **before** the engine starts;
- in the **spawned** worker env for multiprocessing/distributed executors, not just the parent;
- or rely on the autouse fixture if the test lives under `tests/v1/determinism/` (clause 10) — then no manual set is needed.

**Consistency rule (from the contract):** if your fix cites clause 5/6/7/9/11/12/13, `batch_invariant_enabled` must actually be `true` end-to-end. Setting the var in the parent while generation happens in an unset child is a *non-fix* — Phase 2 will RECLASSIFY it.

**Caveat — BI does not cross runtimes, compile strategies, or parallelism degrees.** It constrains vLLM's own kernel selection and accumulation order within one fixed execution strategy. It cannot make HuggingFace match vLLM, does not license exact equality across different compile/graph-partition/fused-distributed strategies (contract "Not Strong By Default" 1–5), and does not license exact equality across different TP/PP/EP/DCP degrees (contract "Not Strong By Default" 4) — different parallelism topologies use different reduction trees and communication patterns, which BI does not unify. For those structural axes use §1.

## §3 Golden strings

When there is a single deterministic engine and no meaningful "second run" to compare against, pin the expected output as a literal instead of regenerating it live.

```python
EXPECTED = [" 1024, 1025, 1026", ...]  # captured from a known-good build
outputs = vllm_model.generate_greedy(prompts, max_tokens)
for out, expected in zip(outputs, EXPECTED):
    assert out[1] == expected
```

Use only when:
- output is short and stable on the target hardware/config, and
- you record *how* the golden was produced (model, dtype, build) in a comment.

Prefer §1 when a live reference exists — golden strings are brittle across hardware and model/version bumps, so reserve them for cases with no better oracle.

## §4 Principled threshold

For acceptance-rate / match-ratio oracles (spec decode, quantization parity), keep the ratio but make the threshold defensible rather than tuned to today's run.

- Base it on the feature's contract (e.g. spec-decode acceptance target), not the observed number minus epsilon.
- State the rationale in a comment or the assertion message.
- If no principled floor exists, the ratio oracle is the wrong tool — switch to §1.

## Verification before you submit

A remedy is not done until:

1. **Re-audit.** Re-apply the 3 criteria. The fix must make c1 **no** (oracle tolerates drift / is pinned / threshold principled) or c3 cite a now-valid clause. If all three still hold, it's not fixed.
2. **Consistency check.** If you cite clause 5/6/7/9/11/12/13, confirm `VLLM_BATCH_INVARIANT=1` reaches the generation process (§2). Otherwise the citation is invalid.
3. **Alignment check.** For §1, confirm both compared sides return the same token span (§1 pitfalls) — inspect one failing/passing pair, don't assume.
4. **Run it.** Execute the specific test on target hardware and paste the result into the PR:
   ```bash
   .venv/bin/python -m pytest path/to/test_file.py::test_name -v
   ```
   Follow `AGENTS.md`: use `uv` / `.venv/bin/python`, never system `python3`. Include model-eval results for output-affecting changes.
5. **Lint.** `pre-commit run --files path/to/test_file.py` (88-char limit).

Do not claim a fix works if you could not run it — say so and give the exact command for the human to run.

## Anti-patterns

- Loosening an assertion with no principle ("bump atol until green").
- Deleting the assertion or downgrading to a smoke check (`len(output) > 0`) — that just makes it NOT_REALISTIC by gutting coverage.
- Adding `VLLM_BATCH_INVARIANT` in the parent process for a spawned-worker generation (non-fix; RECLASSIFY).
- Using BI mode to justify cross-runtime, cross-compile-strategy, or cross-parallelism-degree (TP/PP/EP/DCP) exact equality.
- Reaching for a tolerance oracle on a pure batch-geometry axis (BS=1 vs BS=N, cascade-attention batch trigger, single-vs-batched request) when BI would install a real contract instead — that under-fixes it.
- Editing a shared alignment shim in place and breaking its other caller.
- Do not add comments, just make changes.
- Introducing a new helper function/method to carry the fix when the fix fits inline in the existing function body or into an already-existing helper — keep the diff minimal, don't add abstractions the fix doesn't need.
- Changing a test function's (or a shared helper's) signature to thread through an env-setting fixture — e.g. adding a `monkeypatch` parameter just to call `monkeypatch.setenv(...)`. Use the fixture-free `pytest.MonkeyPatch.context()` classmethod (§2) instead, so no signature changes anywhere.