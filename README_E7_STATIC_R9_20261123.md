# CYCLEFT — E7 Static-R9 Recovery & Tracker Reconciliation
## Comprehensive README — everything to date (2026-11-23) and the forward plan

Project deadline: **27 November 2026**.
This document is the master narrative of the E7 Static-R9 seed-42 effort:
what was executed, what defects were found and fixed, what the measured
results are, where every artifact lives, and exactly what is proposed next.

---

## 1. Project context

CYCLEFT studies whether **dynamic LoRA-rank allocation** across
continual-learning events achieves a better quality–resource trade-off than
**static-rank** adaptation, on `meta-llama/Llama-3.2-1B-Instruct` with the
`temporalwiki-drift-cl-easy` stream (3 events × 500 train / 50 test, b500
protocol, α = 2r, D100 depth, Apple MPS, greedy 24-token generation).

The authoritative plan is the frozen **E7 v5 matrix**
(`results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv`):
**22 rows = 19 NEW + 3 REUSE** (Static-R8 anchors from E2). Statistical
plan P1–P4b: Static-R9 is the near-resource-matched comparison (+3.7%) for
**P4a (Grow vs Static-R9)** and **P4b (Reduce vs Static-R9)**; single-seed
results never support significance claims.

Governance model used throughout: every execution is a separately
authorized, single-run, token-gated operation under the guarded E7 Phase-2
harness (execution locks, fresh-directory guards, fail-closed completion
semantics). Bookkeeping versions of the Master Tracker are append-only;
frozen versions are never modified.

---

## 2. Chronology of 23 November 2026 (all times local, UTC+5:30)

### 2.1 Attempt 1 — authorized run `e7-static-r9-s42` (06:11–~07:03)

- The authorized pilot (`scripts/e7_pilot_static_r9_s42.py`) was already
  mid-flight when supervision began: all **10 pre-execution gates passed**
  (run identity, config hash `406f9cc9…`, MPS, no CUDA, no CPU-fallback
  permission, local model/dataset, disk, fresh output dir).
- **Training completed successfully**: 1,500/1,500 steps across three
  events (fresh AdamW per event — proven by end-of-event optimizer step
  counters 500/500/500; a migrated optimizer would read 500/1000/1500),
  rank 9 / alpha 18 verified in every checkpoint, three E5 stage-end
  checkpoints (steps 500/1000/1500), terminal fingerprint
  `f2408eabaed9790d…` consistent across `LATEST.json`,
  `stage_end_fingerprints.json`, and `progress.json`.
- **In-workload final-only evaluation completed validly** (the frozen v5
  static schedule): EM 0.18 / 0.14 / 0.16 (aggregate 0.16), F1
  0.2127/0.206/0.206, eval loss 1.8597/1.8303/1.7040, ppl
  6.4218/6.2359/5.4957 — all finite, 50 examples per set.
- **Defect discovered during the post-training prediction sweep**: the
  sweep (`e5_adapter._default_predict_fn`) called `attach_lora` and THEN
  `load_adapter_for_training` — a **PEFT double-wrap**. The trained
  checkpoint keys (`base_model.model.model.layers…`) matched nothing in
  the doubly-nested model (`base_model.model.base_model.model…`); PEFT's
  own "Found missing adapter keys" warning (documented no-false-positive)
  appeared in the live log, and a tiny-mock reproduction confirmed **all
  `lora_B` tensors stayed at initialization** — the sweep would have
  scored the **untrained base model**. A second defect: the sweep passed
  `base.device` (CPU; `load_base_model` leaves weights on CPU) — a silent
  CPU fallback, disallowed by protocol.
- **Interruption**: per the failure-handling protocol the sweep was stopped
  BEFORE any invalid record was written (SIGINT ×2 ignored; SIGTERM used).
  Process 56039 exited; manifest honestly remains `RUNNING` (unclean
  termination; never forged, never COMPLETED); `predictions/` remained
  empty. Stale execution lock removed as a documented operator action.
  Attempt wall ≈3,050 s (ESTIMATE); failed-sweep overhead ≈275 s
  (discarded, never scientific compute).
- **Same-day fix** in `src/cycleft/e7/e5_adapter.py`: single-wrap load
  through the exact `run_workload` resume path, `resolve_device` (MPS-
  preferred, raises when unavailable), fingerprint gate against
  `bundles/LATEST.json`. Verified: mock roundtrip byte-identity; real
  terminal bundle E5-loads on MPS (128 LoRA tensors, `[9, 2048]`,
  fingerprint match); 174 E7 + 368 full tests passed (exit 0).
  *Disclosure: no tests had been run before attempt 1 started (gate miss of
  the launching session), recorded honestly.*

### 2.2 Recovery preflight (read-only; two-example MPS smoke test)

- **Part A audit**: all nine required loading properties verified.
- **Part B**: 16 checkpoint files hashed before and after — byte-identical.
- **Part C**: real MPS inference on exactly TWO deterministic examples
  (test set 1 idx 0, test set 3 idx 49): fingerprint matched on load,
  128 tensors byte-equal to checkpoint and non-zero, model on `mps:0`,
  zero PEFT warnings, generation completed. Timings 5.96 s load + 2.64 s
  inference (smoke-scale only).
- **Finding P-1 (new, genuine)**: the corrected sweep omitted the
  harness-required `test_event` field from prediction records — a full
  sweep would have run ~40 min and then been discarded at harness
  finalization (fail-closed). Encoded as a **strict xfail test** that
  forces removal of the marker together with the fix.
- **Decision**: Option A (provenance-linked prediction-only recovery)
  feasible; full rerun NOT required.
- Verdict: `PASS — PREDICTION-ONLY RECOVERY FEASIBLE`.
- 14 tests + 1 xfail added; E7 suites 188+1xfail; full suite 382+1xfail.

### 2.3 Recovery-01 — prediction-only recovery (authorized)

- **P-1 fixed** (`record["test_event"] = event.event_id` — confirmed from
  `event_orchestrator.py:1470` that `event.event_id` IS the evaluation
  test-set ID, domain {1,2,3}); strict xfail converted to a passing test
  plus 5 new P-1 regression tests.
- **Regression gate**: P-1 targeted 20 ✓; E7 adapter 13 ✓; harness 57 ✓;
  orchestrator 85 ✓; Phase 1 19 ✓; complete E7 194 ✓; **full repository
  388 passed, exit 0**.
- **Sweep**: exactly **150 predictions** (50/50/50, indices 0–49 once per
  set, full schema, finite, no duplicates) on MPS from the ORIGINAL
  terminal checkpoint — 6.31 s load/restore + 92.9 s inference; no
  training, no gradients (`torch.no_grad()`), no optimizer ever created.
- **Metric reconciliation: EXACT.** Recomputed EM/F1 from the written
  records == reported == original in-workload `final_eval.json`;
  eval_loss/perplexity deltas **0.00e+00** (bit-exact — greedy decoding on
  the identical adapter is deterministic).
- **Incident (disclosed)**: post-prediction bookkeeping raised
  `KeyError('config_hash')` (progress.json nests the hash) AFTER all
  predictions and gates completed; bookkeeping completed by a separate
  corrected script; predictions untouched; sweep script preserved as
  executed.
- Verdict: `PASS — STATIC-R9 PREDICTIONS RECOVERED AND VALIDATED`.
- Status: `PREDICTIONS_RECOVERED_AND_VALIDATED`. Attempt-1 directory
  byte-identical (32/32 files) after recovery.

### 2.4 Resealing

- Two inventory discrepancies investigated (not silently patched):
  (A) execution-log checksum staleness (4,153 → 5,327 bytes — the
  incident-disclosure append happened after the inventory snapshot);
  (B) `checkpoint_reference.json` missing from the inventory (created
  after the snapshot). Superseded copies preserved timestamped
  (`superseded_20261123T0540Z/`) with recorded reasons.
- Canonical inventory regenerated: **18 files**; canonical digest
  `a6478df6e16e705d…`; self-checksum `9bbd5289f7996c01…`; manifest digest
  `044bc5e6525db941…`; self-reference rules preserved (no circular
  checksums); atomic writes.
- Attempt-1 re-verified byte-identical (32/32); all three checkpoints
  re-verified E5-loadable.
- Verdict: `PASS — STATIC-R9 S42 RECOVERY RESEALED`.

### 2.5 Tracker versions (all append-only; frozen versions untouched)

| Version | Content |
|---|---|
| v1.1.1 (frozen) | authoritative baseline ledger (pre-execution forecasts) |
| v1.1.2 | attempt-1 recorded: AUTHORIZED → INTERRUPTED (training complete; sweep defect; measured timings) |
| v1.1.3 | recovery-01 provenance (150/150; exact reconciliation) |
| v1.1.4 | preflight/fix lineage carried; adapter fix recorded |
| v1.1.5 | run-accounting correction: s42 removed from pending forecast; remaining forecast recalculated directly from 15 unblocked rows (37,500/44,480/56,500 s); actual compute ledgered; 20/20 validation |
| **v1.1.5.1** | status-matrix consistency correction: E7 row aggregates updated (3 logically complete = 2 harness + 1 recovery-validated; 16 pending = 15 unblocked + 1 blocked); 13/13 validation |

---

## 3. Results — Static-R9 (rank 9, alpha 18, seed 42), final-only frozen v5

| Test set | EM | F1 | Eval loss | Perplexity |
|---|---:|---:|---:|---:|
| 1 | 0.18 | 0.212667 | 1.859692 | 6.421760 |
| 2 | 0.14 | 0.206000 | 1.830325 | 6.235916 |
| 3 | 0.16 | 0.206000 | 1.703968 | 5.495709 |
| **Aggregate** | **0.16** | **0.2082** | 1.7980 (mean) | 6.0511 (mean) |

Compute (measured): train 2,581.135 s (per-event walls 978.5/804.3/804.6) +
in-workload eval 203.297 s + recovery 99.21 s (6.31 + 92.9) =
**2,883.64 s ≈ 48.1 min** combined logical (TS-v1.1: model-load/checkpoint
NESTED diagnostics, never double-counted; failed attempt-1 sweep ≈275 s =
discarded overhead; attempt wall ≈3,050 s ESTIMATE).

Status: **single seed — no significance claim**; data point is
**eligible for P4a/P4b** once policy seed sets complete.

---

## 4. Defects found and fixed (with tests)

| # | Defect | Fix | CI protection |
|---|---|---|---|
| 1 | PEFT double-wrap in the prediction sweep (checkpoint keys matched nothing → untrained base model would be scored) | single-wrap load via the `run_workload` resume path; `attach_lora` removed | AST call-graph test (sweep can never call `attach_lora`/`run_workload`); loader checks in the smoke test |
| 2 | Silent CPU device fallback in the sweep | `resolve_device(None)` (MPS-preferred; raises) | `test_resolve_device_rejects_silent_cpu_fallback` |
| 3 | No fingerprint gate on the reloaded adapter | fingerprint verified vs `bundles/LATEST.json` before any prediction | `test_predict_rejects_fingerprint_mismatch`, `test_missing_adapter_weights_cause_failure` |
| 4 | **P-1**: `test_event` missing from prediction records (harness would fail-closed after a full sweep) | `record["test_event"] = event.event_id` | converted strict-xfail → passing test + 5 P-1 tests (schema, harness accept/reject, grow-precedent compatibility) |
| 5 | Bookkeeping KeyError (progress.json nesting) | corrected separate bookkeeping script; sweep preserved as executed | incident fully disclosed in logs/reports |

Final suite state: **388 passed, exit 0** (no failing/xfail tests); no
previous test weakened at any step.

---

## 5. Artifact map (everything, with digests)

### Run evidence (read-only)
- `results/e7/execution_20261123/e7-static-r9-s42/` — attempt 1 (INTERRUPTED
  evidence): manifest (RUNNING), 3 E5 checkpoints + durable copies,
  execution_summary (timing registry), final_eval.json, event/audit logs,
  stage_end_fingerprints.json. 32 files, manifest SHA-256
  `b88898470cd972d3…`. Byte-identical after recovery and reseal.
- `results/e7/recovery_static_r9_s42_20261123/` — recovery-01 (canonical):
  `predictions/after_event_{1,2,3}.jsonl` + `predictions.jsonl` (150
  records, SHA-256 `029eb527…`), `evaluation_metrics.json`,
  `evaluation_config.json`, `checkpoint_reference.json`,
  `recovery_manifest.json` (resealed; manifest digest `044bc5e6…`),
  `artifact_inventory.json` (18 files; canonical digest `a6478df6…`),
  `reconciliation_and_integrity.json`, `superseded_20261123T0540Z/`
  (pre-reseal copies), `logs/` (sweep + verification + reseal scripts and
  logs).

### Reports
- `results/e7/pilot_static_r9_s42_20261123/` — attempt-1 supervision: all 8
  required reports + `checkpoint_integrity.json` (24/24 checks).
- `results/e7/pilot_static_r9_s42_recovery_preflight_20261123/` — 8 required
  reports + smoke-test evidence/scripts + checkpoint before/after hashes.
- `results/e7/pilot_static_r9_s42_recovery_20261123/` — 12 required
  deliverables of recovery-01.
- `results/e7/pilot_static_r9_s42_recovery_reseal_20261123/` — 9 required
  reseal deliverables + `reseal_evidence.json`.

### Source changes (only file touched in production)
- `src/cycleft/e7/e5_adapter.py` — single-wrap load + `resolve_device` +
  fingerprint gate + P-1 `test_event` field (all documented in the module
  docstring).
- `tests/test_e7_e5_adapter_recovery_preflight.py` — 20 regression tests.

### Master Tracker versions
`results/project_master_tracker_20261123_v1_1_1/` (frozen) → …_v1_1_2/ →
…_v1_1_3/ → …_v1_1_4/ → **…_v1_1_5/** → **…_v1_1_5_1/** (current).

### ZIP archives (SHA-256)
| Archive | Bytes | SHA-256 |
|---|---|---|
| `results/e7/e7-static-r9-s42_attempt1_evidence_20261123.zip` | 113M | `472aacbe5f3424f83decd2e0cb749d19626fe1bda59e7aef0e329f394714e1b2` |
| `results/e7/e7-static-r9-s42_recovery_preflight_20261123.zip` | 36K | `8809c88831531d78ffa22daa60110d8b95799177357f6351fc85884c52711a49` |
| `results/e7/e7-static-r9-s42_recovery-01_20261123.zip` | 124K | `424b60994537293d2e77ba405408c2c5d440e112648b244bceaaaca00c401d98` |
| `results/e7/e7-static-r9-s42_recovery_reseal_20261123.zip` | 84K | `292b500f9343643cf4ff90e91428d486d3e79bf6672c30ffe7fa56ec571dfc1e` |
| `results/e7/project_master_tracker_20261123_v1_1_4.zip` | 32K | `d7b767e767c1b6d6ddc4ec1c103006977007db38f2557b5192e4374ef6f8f0e5` |
| `results/project_master_tracker_20261123_v1_1_5.zip` | 44K | `6e32a5c72e26ace7cd86664a9719bf1c140a27a6a0b036ac46bd84c2560f68fa` |
| `results/project_master_tracker_20261123_v1_1_5_1.zip` | 48K | `b1b42f23151f5a546280ce1fc0b19a77b5f5fcefc3d1cd5b2805b6de7513bd11` |

---

## 6. Current state (end of 23 November)

| Status | Count | Runs |
|---|---:|---|
| Harness-completed NEW | 2 | grow-s42 (EM 0.1533), reduce-s42 (EM 0.18) |
| Recovery-validated NEW | 1 | static-r9-s42 (EM 0.16, F1 0.2082) |
| Pending unblocked NEW | 15 | Grow s43/s44, Reduce s43/s44, Static-R4 ×3, Static-R9 s43/s44, Static-R16 ×3, Static-R32 ×3 |
| Pending blocked NEW | 1 | resource-aware-s42 (E5 loss-signal decision) |
| Historical REUSE | 3 | Static-R8 s42/s43/s44 (read-only) |

Remaining forecast (15 unblocked rows, direct sum): **low 37,500 s /
central 44,480 s / high 56,500 s** (10.4/12.4/15.7 h; central +10%
contingency 48,928 s ≈ 13.6 h). Resource-Aware excluded (blocked); REUSE
anchors excluded; completed runs excluded.

---

## 7. What is proposed from now on

### 7.1 Next execution (READY, awaiting explicit authorization)
**`e7-static-r9-s43`** — Static-R9, seed 43, ranks 9→9→9, alphas 18→18→18,
final-only under frozen v5. Matrix row 9, NEW, never executed. Status:
`READY FOR SEPARATE EXECUTION AUTHORIZATION`. No directory, no lock, no
authorization exists yet. On authorization it runs under the same guarded
single-run harness with the corrected adapter (which now also carries the
P-1 schema fix — attempt 1's interruption cannot recur for this cause).
Recommended: pre-flight gates identical to s42's; expect ~2,784 s active
compute (s42 measured) + ~99 s recovery-phase prediction sweep.

### 7.2 Recommended execution order (critical path to 27 Nov)
1. `e7-static-r9-s43` (2,200 s central) — completes half of the Static-R9
   seed set with s42.
2. `e7-static-r9-s44` (2,200) — completes Static-R9 → **P4a/P4b become
   computable** for the Static-R9 policy.
3. `e7-static-r4-s42/43/44` (2,100 each) — Static-R4 seed set.
4. `e7-static-r16-s42/43/44` (2,400 each) — Static-R16 seed set.
5. `e7-static-r32-s42/43/44` (2,700 each) — Static-R32 seed set.
6. `e7-grow-s43` (3,700) and `e7-grow-s44` (3,700) — Grow seed set.
7. `e7-reduce-s43` (5,540) and `e7-reduce-s44` (5,540) — Reduce seed set.
Each run: separate authorization, harness-locked, stop-on-failure, no
automatic continuation. Full plan: `NOV23_27_CRITICAL_PATH_V1_1_5.md`
inside tracker v1.1.5.

### 7.3 Decisions proposed (owner input needed)
1. **Resource-Aware unblock**: the pilot (`e7-resource-aware-s42`) stays
   blocked pending the E5 training-loss signal-persistence decision. Either
   authorize the small E5 instrumentation work it needs, or formally defer
   it out of the v5 scope (it is 1 of the 19 NEW rows; deferring keeps all
   other comparisons intact).
2. **P4a/P4b trigger point**: run the paired analyses as soon as
   Static-R9 s43+s44 AND the relevant partner policy's seed set complete,
   rather than waiting for all 15 runs (early signal, clearly labeled
   non-confirmatory at 3 seeds; Holm correction across the five primary
   comparisons per the frozen plan).
3. **Static forecast optimism**: s42's measured 2,784.4 s active compute
   exceeded the 2,200 s static central estimate by ≈27%. Keep s43/s44
   estimates unchanged for like-for-like planning (recorded observation in
   v1.1.5), or lift the static central estimates to ≈2,800 s before
   scheduling the remaining 11 static runs (total +≈6,600 s central).
4. **Optional**: one-purpose smoke gate before every future prediction
   sweep (2-example fingerprint+schema check) — cheap insurance given the
   attempt-1 incident; the corrected pathway now self-gates on fingerprint
   and schema anyway.

### 7.4 Analysis plan once seed sets complete (frozen P1–P4b)
- P4a: Grow (3 seeds) vs Static-R9 (3 seeds); P4b: Reduce vs Static-R9 —
  exact paired sign-flip at 3 seeds (minimum p = 0.25 → no confirmatory
  significance claimable); Holm correction across the five primary
  comparisons; per-seed EM reported individually.
- Static policies also feed the rank–quality curve (R4/R8[reuse]/R9/R16/R32).
- No static-vs-dynamic BWT claims (statics are final-only by frozen
  design; only Grow/Reduce have event-level evaluation).

### 7.5 Standing constraints
- Never relabel attempt 1 of static-r9-s42 as COMPLETED; recovery-01
  remains the distinct prediction-only attempt.
- Never overwrite frozen trackers, original artifacts, or the v5 design.
- One experiment at a time, each under its own explicit authorization.
- No downloads, no CUDA, no cloud; MPS only.

---

## 8. Pointers for a new session

1. Read `results/project_master_tracker_20261123_v1_1_5_1/` (current truth):
   start with `MASTER_RESEARCH_TRACKER_V1_1_5.md`,
   `V1_1_5_1_VALIDATION_REPORT.md`, `UNRESOLVED_DISCREPANCIES_V1_1_5.md`.
2. For executing s43: `scripts/e7_pilot_static_r9_s42.py` is the template —
   replicate as `e7_pilot_static_r9_s43.py` (RUN_ID, telemetry path,
   output root per the harness; the output root must not already contain
   the run directory). The harness refuses non-fresh directories and
   requires the confirmation token `i-authorize-e7-phase2-execution`.
3. The prediction sweep is safe post-P-1: single-wrap load, MPS,
   fingerprint gate, schema-complete records (20 regression tests in
   `tests/test_e7_e5_adapter_recovery_preflight.py`).
4. After any run: the harness finalizes COMPLETED only with full artifact
   evidence; then update the tracker as a new append-only version.
