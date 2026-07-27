# vLLM Test Oracle Audit — Full Results

**Date**: 2026-07-02
**Plugin version**: v2.13.0
**Method**: Two-phase pipeline — Phase 1 (audit-agent) finds and classifies, Phase 2 (review-agent) adversarially verifies

## Results

| | Phase 1 | Phase 2 | Final |
|---|---|---|---|
| Analyzed | 534 | 41 CC reviewed | — |
| COINCIDENTALLY_CORRECT | 41 | 36 confirmed + 2 skipped | **38** (36 active, 2 skipped) |
| Reclassified by Phase 2 | — | 7 removed, 3 reversed, 1 manual | **5 net removed** |

### Directories Audited

| Directory | Analyzed | Phase 1 CC | Phase 2 CC |
|-----------|----------|------------|------------|
| tests/v1/e2e/ | 56 | 14 | 12 |
| tests/compile/correctness_e2e/ | 5 | 5 | 5 |
| tests/basic_correctness/ | 13 | 2 | 2 |
| tests/lora/ | 142 | 1 | 1 |
| tests/entrypoints/ | 51 | 14 | 10 |
| tests/distributed/ | 267 | 5 | 4 |
| **Total** | **534** | **41** | **34** |

### Not Audited

- `tests/models/` — contains known CC tests (`test_load_pp_4bit_bnb_model`, `test_load_tp_4bit_bnb_model`, `test_single_and_batched_generation_match`) from init-results
- `tests/kernels/` — kernel correctness tests, generally STRONG_CONTRACT
- `tests/v1/determinism/` — has `VLLM_BATCH_INVARIANT=1` via autouse conftest

---

## Phase 2 Reclassifications (7)

| Test | Phase 1 | Phase 2 | Reason |
|------|---------|---------|--------|
| `test_pp_cudagraph` | CC | STRONG_CONTRACT | Strong Contract #4: eager vs cudagraph same execution family |
| `test_tp_language_embedding` | CC | NOT_REALISTIC | Cosine sim >= 0.999 is a tolerance-based oracle, not exact equality |
| ~~`test_batch_completions`~~ | ~~CC~~ | ~~STRONG_CONTRACT~~ | ~~Phase 2 incorrectly applied Strong Contract #7 — see correction below~~ |
| ~~`test_batch_completions[beam_search_cross_position]`~~ | ~~CC~~ | ~~STRONG_CONTRACT~~ | ~~same~~ |
| ~~`test_batch_completions[streaming_batch]`~~ | ~~CC~~ | ~~STRONG_CONTRACT~~ | ~~same~~ |
| `test_streaming_input_output_equivalence` | CC | STRONG_CONTRACT | Strong Contract #6: streaming vs non-streaming transport |
| `test_mtp_speculative_mixed_batch_short_prefill` | CC | NOT_REALISTIC | Trivial recall task, smoke test — FP drift can't change output |
| `test_single_chat_session_image_base64encoded_beamsearch` | CC | NOT_REALISTIC | Semantic term check (`check_output_matches_terms`) — tests model capability, not numeric determinism. Terms are broad ("boardwalk", "parrot"/"bird") and fundamental to image content. |

---

## 34 Confirmed COINCIDENTALLY_CORRECT

### Cross-Config Exact Parity (8)

| Test | File | Compares |
|------|------|----------|
| `test_async_tp_pass_correctness` | tests/compile/correctness_e2e/test_async_tp.py | async TP compiled vs standard TP |
| `test_async_tp_pass_nvfp4_correctness` | tests/compile/correctness_e2e/test_async_tp.py | async TP+NVFP4 vs standard TP |
| `test_tp_sp_generation` | tests/compile/correctness_e2e/test_sequence_parallel.py | compiled SP vs non-compiled TP |
| `test_tp_sp_generation_prompt_embeds` | tests/compile/correctness_e2e/test_sequence_parallel.py | compiled SP+prompt_embeds vs TP |
| `test_tp_sp_nvfp4_generation` | tests/compile/correctness_e2e/test_sequence_parallel.py | NVFP4 SP vs non-compiled TP |
| `test_tp_language_generation` | tests/distributed/test_pipeline_parallel.py | PP+TP vs TP-only |
| `test_tp_multimodal_generation` | tests/distributed/test_pipeline_parallel.py | PP+TP vs TP-only multimodal |
| `test_ep` | tests/distributed/test_expert_parallel.py | EP enabled vs disabled |

### Cross-Representation — prompt_embeds vs text (9)

| Test | File |
|------|------|
| `test_text_content_and_prompt_embeds_match` | tests/entrypoints/openai/chat_completion/test_chat_completion_with_prompt_embeds.py |
| `test_text_content_and_prompt_embeds_match_with_image_url[image_url-then-text]` | tests/entrypoints/multimodal/openai/chat_completion/test_chat_completion_with_mixed_image_embeds.py |
| `test_text_content_and_prompt_embeds_match_with_image_url[text-then-image_url]` | same |
| `test_text_content_and_prompt_embeds_match_with_image_embeds[image_embeds-then-text]` | same |
| `test_text_content_and_prompt_embeds_match_with_image_embeds[text-then-image_embeds]` | same |
| `test_text_content_and_prompt_embeds_match_with_audio_embeds[audio_embeds-then-text]` | tests/entrypoints/multimodal/openai/chat_completion/test_chat_completion_with_mixed_audio_embeds.py |
| `test_text_content_and_prompt_embeds_match_with_audio_embeds[text-then-audio_embeds]` | same |
| `test_completions_with_prompt_embeds[use-lora-facebook/opt-125m]` | tests/entrypoints/openai/completion/test_completion_with_prompt_embeds.py |
| `test_completions_with_prompt_embeds[use-lora-opt125m-lora]` | same |

### Spec Decode without BI Mode (7)

| Test | File | Threshold |
|------|------|-----------|
| `test_speculators_model_integration` | tests/v1/e2e/spec_decode/test_spec_decode.py | 66% |
| `test_eagle_correctness_light` | same | 60% |
| `test_eagle_correctness_medium` | same | 60% |
| `test_eagle_correctness_heavy` | same | 60% |
| `test_mtp_correctness` | same | 80% |
| `test_batch_inference_correctness` | tests/v1/e2e/spec_decode/test_lora_with_spec_decode.py | 90% |
| `test_with_eagle3_spec_decoding` | tests/v1/e2e/general/test_async_scheduling.py | exact |

### Cross-Scheduling Config (2)

| Test | File |
|------|------|
| `test_with_ngram_gpu_spec_decoding` | tests/v1/e2e/general/test_async_scheduling.py |
| `test_without_spec_decoding` | same |

### Cross-Runtime (2)

| Test | File | Compares |
|------|------|----------|
| `test_models` | tests/basic_correctness/test_basic_correctness.py | HF vs vLLM |
| `test_models_distributed` | same | HF vs vLLM (TP=2) |

### TP Parity (1)

| Test | File | Compares |
|------|------|----------|
| `test_quant_model_tp_equality` | tests/lora/test_quant_model.py | TP=1 vs TP=2 |

### Other Fragile Oracles (3)

| Test | File | Issue |
|------|------|-------|
| `test_sliding_window_retrieval` | tests/v1/e2e/general/test_correctness_sliding_window.py | Exact integer recall from generated text |
| `test_kv_sharing_fast_prefill` | tests/v1/e2e/general/test_kv_sharing_fast_prefill.py | Exact integer recall from generated text |
| `test_structured_output_batched...[fact_check]` | tests/entrypoints/llm/test_struct_output_generate.py | Substring "12,742" in generated text |

---

### Batch Position Invariance (6, 2 skipped)

| Test | File | Compares | Note |
|------|------|----------|------|
| `test_batch_completions` | tests/entrypoints/openai/completion/test_completion.py | same prompt at positions 0 vs 2 in batch | |
| `test_batch_completions[beam_search_cross_position]` | same | beam search cross-position | |
| `test_batch_completions[streaming_batch]` | same | streaming batch positions | |
| `test_cascade_attention` | tests/v1/e2e/general/test_cascade_attention.py | batch=1 vs batch=64 | |
| `test_qwen36_moe_mixed_2d_3d_lora_tp2` | tests/lora/test_qwen36_moe_lora.py | single-adapter-alone vs mixed-batch exact text `==` | `@pytest.mark.skip` |
| `test_qwen36_moe_mixed_2d_3d_lora_tp4` | same | same pattern, TP=4 | `@pytest.mark.skip` |

---

## Phase 2 Correction: test_batch_completions

Phase 2 incorrectly reclassified `test_batch_completions` (3 variants) to STRONG_CONTRACT citing Strong Contract #7 ("duplicate identical requests in same batch"). This was wrong — Strong Contract #7 only applies when `VLLM_BATCH_INVARIANT` is enabled. Without it, different batch positions can produce different output due to cuBLAS kernel selection and accumulation order differences. The contract clause has been updated. These 3 tests are **COINCIDENTALLY_CORRECT**.

---

## Comparison with Init-Results (29 CC)

| | Count |
|---|---|
| Init-results CC | 29 |
| Confirmed in new run | 22 |
| In init but not audited (directory not covered) | 3 |
| In init, correctly reclassified by Phase 2 | 0 |
| In init, not found by Phase 1 | 1 |
| New findings not in init | 12 |
| **Final CC after Phase 2 + corrections** | **38** (36 active, 2 skipped) |

### From init-results, not in new run (5)

| Test | Reason |
|------|--------|
| `test_batch_completions` (3 variants) | Confirmed CC — Phase 2 reclassification was reversed (Strong Contract #7 requires VLLM_BATCH_INVARIANT) |
| `test_load_pp_4bit_bnb_model` | `tests/models/quantization/` not audited |
| `test_load_tp_4bit_bnb_model` | `tests/models/quantization/` not audited |
| `test_single_and_batched_generation_match` (2 tests) | `tests/models/multimodal/` not audited |
| `test_qwen36_moe_mixed_2d_3d_lora_tp2/tp4` | CC confirmed — lora agent missed it (helper `_run_mixed_2d_3d_lora_test`). Both `@pytest.mark.skip` |

### New findings not in init-results (12)

| Test | Category |
|------|----------|
| `test_sliding_window_retrieval` | Exact integer recall |
| `test_kv_sharing_fast_prefill` | Exact integer recall |
| `test_without_spec_decoding` | Cross-scheduling config |
| `test_with_eagle3_spec_decoding` | Spec decode (also in init new findings) |
| `test_with_ngram_gpu_spec_decoding` | Spec decode (also in init new findings) |
| `test_structured_output_batched...[fact_check]` | Semantic content |
| `test_text_content_and_prompt_embeds_match_with_image_url[image_url-then-text]` | Cross-representation |
| `test_text_content_and_prompt_embeds_match_with_image_url[text-then-image_url]` | Cross-representation |
| `test_text_content_and_prompt_embeds_match_with_image_embeds[image_embeds-then-text]` | Cross-representation |
| `test_text_content_and_prompt_embeds_match_with_image_embeds[text-then-image_embeds]` | Cross-representation |
| `test_text_content_and_prompt_embeds_match_with_audio_embeds[audio_embeds-then-text]` | Cross-representation |
| `test_text_content_and_prompt_embeds_match_with_audio_embeds[text-then-audio_embeds]` | Cross-representation |
