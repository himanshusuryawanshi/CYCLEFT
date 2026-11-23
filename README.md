# CYCLEFT — Project README
## Continual LoRA Adaptation: Dynamic Capacity Allocation Under Real Constraints

**Version:** 2026-11-23 (post Static-R9 recovery; tracker v1.1.5.1)
**Deadline:** 27 November 2026
**Compute platform:** Apple Silicon MacBook Air (MPS) — local, no cloud spend
**Model:** `meta-llama/Llama-3.2-1B-Instruct` (fp16 base + fp32 LoRA)
**Dataset:** `saxenan3/temporalwiki-drift-cl-easy` (3-event temporal stream)

---

## 1. Research question

Does **dynamic allocation of adapter capacity and compute resources**
during continual fine-tuning achieve a better **quality–resource trade-off**
than static configurations — and can that allocation be made reliable
(resumable, checkpointed, drift-aware) on commodity hardware?

The project answers this through a sequenced experimental program
(**E1–E12**) in which each experiment builds on the verified artifacts of
the previous one, under a governance model of **separately authorized,
individually sealed runs** with append-only audit trails.

### Core hypotheses

| # | Hypothesis |
|---|---|
| H1 | Continual adaptation improves answer quality over the frozen base (E1) |
| H2 | A cost–quality Pareto frontier exists over LoRA rank and training budget (E2) |
| H3 | Drift detection avoids unnecessary adaptation compute (E3) |
| H4 | Checkpointing reduces lost work under interruptions (E5) |
| H5 | Adapter depth monotonically affects continual-learning performance (E6) |
| H6 | Dynamic LoRA-rank allocation (Grow/Reduce) matches or beats static rank at better resource cost (E7) |
| H7 | Dynamic resource allocation and PPO improve efficiency (E4/E8, partially future) |

---

## 2. Repository layout

```
CYCLEFT/
├── configs/                     # e1_stream.yaml (authoritative stream+training), base.yaml, e1.yaml
├── data/                        # raw / processed temporalwiki slices
├── src/cycleft/                 # package
│   ├── e1/                      # data stream, LM runtime (LoRA train/eval), pipeline, metrics
│   ├── e2/                      # rank×budget grid, resource sampling, Pareto analysis
│   ├── e3/                      # drift detectors, calibrated streams, runner
│   ├── e4/                      # scheduler, resource harness, inventory (dynamic allocation)
│   ├── e5/                      # ResumableTrainer, ResumeBundle, checkpoint policies P0–P5, timing
│   ├── e6/                      # adapter-depth (layer) selection
│   ├── e7/                      # guarded execution harness, E5 adapter, event orchestrator, rank transitions
│   ├── checkpoints.py, config.py, evaluation.py, experiment.py, scheduler.py, types.py
├── scripts/                     # per-experiment pilots and workers (e1_run … e7_pilot_*)
├── tests/                       # 388 tests, all passing (exit 0)
├── results/
│   ├── e1/ … e7/                # per-experiment evidence trees (raw + reports)
│   ├── project_master_tracker_* # versioned master trackers (v1 → v1.1.5.1)
│   └── research_audit_*         # independent audit reports
├── Paper/                       # reference literature
├── README_E7_STATIC_R9_20261123.md  # detailed narrative of the Static-R9 recovery effort
└── pyproject.toml, uv.lock
```

Environment: Python 3.11.16, PyTorch 2.14.1, PEFT/Transformers, macOS 26.2,
Apple MPS (CUDA deliberately absent; CPU fallback opt-in only and raised as
an error otherwise).

---

## 3. The experimental program (E1–E12)

### E1 — Real continual adaptation ✅ EXECUTION COMPLETE
Establishes the continual-learning baseline: sequential LoRA adaptation over
three temporal events (500 train / 50 test each), fresh AdamW per event,
EM/F1/perplexity evaluation. Key engineering outcomes: fp32 LoRA on fp16
base (fp16 LoRA destabilized generation — documented in the aborted
lr-1e-4 and fp16-LoRA pilots), frozen chat-template date for prompt
reproducibility, batch size 1.
**Claim: SUPPORTED** (continual adaptation improves quality over the frozen
base; single-seed caveat).
Evidence: `results/e1/full_20261113`, `results/baseline.json`.

### E2 — Cost–quality Pareto frontier ✅ EXECUTION COMPLETE
Grid over LoRA rank {4, 8, 16, 32} × budget {125, 250, 500} with resource
sampling; produces the rank/budget Pareto frontier and the **Static-R8
b500 3-seed anchors** later reused by E7 (read-only REUSE rows).
**Claim: SUPPORTED** (a Pareto frontier exists).
Evidence: `results/e2/full_20261115`, `remediation_A/B_*` (3-seed anchor
remediations).

### E3 — Drift detection ✅ EXECUTION COMPLETE
Calibrated drift detectors over the temporal stream; demonstrates detection
can avoid unnecessary adaptation compute.
**Claim: SUPPORTED.**
Evidence: `results/e3/full_20261116`, `pilot_20261116`, `remediation_E3_*`.

### E4 — Dynamic resource allocation ◐ PARTIALLY COMPLETE
Scheduler + resource harness for dynamic allocation. **Degenerate-case
result**: the tested allocation rule did not beat static configurations
(PARTIALLY SUPPORTED at best); multi-resource extension pending.
Evidence: `results/e4/local_20261117`.

### E5 — Spot-instance resilience ◐ PARTIALLY COMPLETE
The **ResumableTrainer**: faithful mid-stage resume via ResumeBundles
(adapter + optimizer + RNG + progress, fingerprint-verified), checkpoint
policies P0 (none), P1 (fixed interval + stage ends), P2 (notice checkpoint),
P3 (adaptive), P5 (oracle). Event/audit logs are append-only and
SIGKILL-survivable; TimingRegistry separates train / eval / model-load /
checkpoint categories (TS-v1.1 scope convention).
**Claim: SUPPORTED locally** (checkpointing reduces lost work);
REAL-CLOUD NOT VALIDATED.
Evidence: `results/e5/design_20261119`, `phase1_20261119`,
`phase2_20261119`, `phase2_instrumentation_20261119`.

### E6 — Adapter depth ✅ STATISTICALLY ANALYZED
Layer-depth selection (D25 vs D100): quality responds monotonically to
depth; **D100 (all 16 decoder layers, q/k/v/o) > D25**. This fixes the E7
depth convention.
Evidence: `results/e6/execution_20261119/E6_RESULTS.md`,
`e6_statistical_analysis.json`.

### E7 — Dynamic LoRA rank vs static rank ◐ PARTIALLY COMPLETE (current focus)
The centerpiece comparison, governed by the frozen **v5 matrix**
(`results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv`):
**22 rows = 19 NEW + 3 REUSE**.

| Policy | Trajectory | Seeds | Status |
|---|---|---|---|
| Static-R4 / R9 / R16 / R32 | fixed (α = 2r) | 3 each | NEW |
| Static-R8 | fixed | 3 | **REUSE** (E2 anchors, read-only) |
| Grow | 4→8→16 | 3 | NEW |
| Reduce | 16→8→4 | 3 | NEW |
| Resource-Aware | runtime-decided (convergence signal) | 1 | NEW, **BLOCKED** (E5 loss-signal decision) |

Evaluation schedules (frozen): statics = **final-only**; dynamics =
event-level (3 snapshots). Statistical plan P1–P4b with Holm correction;
exact paired sign-flip at 3 seeds (minimum p = 0.25 → descriptive, not
confirmatory). Optimizer lifecycle: **fresh AdamW per event** ( moments
never migrate across rank changes; ResumeBundle restore is for crash
recovery only).

**Current E7 state (tracker v1.1.5.1):**

| Status | Count | Runs |
|---|---:|---|
| Harness-completed NEW | 2 | grow-s42 (EM 0.1533), reduce-s42 (EM 0.18) |
| Recovery-validated NEW | 1 | static-r9-s42 (EM 0.16, F1 0.2082) |
| Pending unblocked NEW | 15 | s43/s44 of Grow+Reduce; Static-R4/R9/R16/R32 seed sets |
| Pending blocked NEW | 1 | resource-aware-s42 |
| REUSE | 3 | Static-R8 anchors |

**Dynamic pilots (measured):** grow-s42 wall 3,699.2 s (train 1,379.4);
reduce-s42 wall 5,539.1 s (train 2,333.0). Directional finding: rank
**expansion preserved quality** (0.153 mean EM after event 3).

### E8–E12 — NOT DESIGNED (future work)
Baseline competitiveness vs external CL methods (E8), generalization to
other workloads (E9), multi-seed robustness program (E10), reward-hacking
audit (E11), real-cloud validation (E12).

---

## 4. The Static-R9 seed-42 case study (23 November 2026)

The most instructive engineering narrative of the project; full narrative in
`README_E7_STATIC_R9_20261123.md`.

### What happened
1. **Attempt 1** (authorized, guarded harness): all 10 pre-execution gates
   passed; **all three training events completed** (1,500/1,500 steps,
   fresh AdamW per event proven by optimizer step counters 500/500/500,
   rank 9 / α 18 in every checkpoint); in-workload final-only evaluation
   valid (EM 0.18/0.14/0.16). The post-training prediction sweep then hit
   two implementation defects — a **PEFT double-wrap** (checkpoint keys
   matched nothing; PEFT's no-false-positive "missing adapter keys" warning
   fired; the sweep would have scored the untrained base model) and a
   **silent CPU fallback**. The sweep was interrupted before a single
   invalid record was written; the manifest honestly remains RUNNING.
2. **Same-day fixes**: single-wrap load through the E5 resume path,
   `resolve_device` (MPS-preferred, raises), fingerprint gate vs
   `LATEST.json`.
3. **Recovery preflight**: two-example MPS smoke test validated the loader
   and discovered **P-1** — the sweep omitted the harness-required
   `test_event` field (would have failed-closed after a full sweep).
4. **Recovery-01** (authorized): P-1 fixed; exactly **150 predictions**
   generated on MPS from the original terminal checkpoint (fingerprint
   verified; no training/gradients); **metric reconciliation EXACT**
   (EM/F1 match; loss/ppl deltas 0.00e+00 — greedy decoding proven
   deterministic). Status: `PREDICTIONS_RECOVERED_AND_VALIDATED`.
5. **Resealing**: inventory discrepancies investigated and explained
   (append-after-snapshot; post-snapshot file), superseded copies preserved,
   canonical inventory resealed (18 files, digests recorded).

### Measured results (final-only, frozen v5)

| Test set | EM | F1 | Eval loss | PPL |
|---|---:|---:|---:|---:|
| 1 | 0.18 | 0.212667 | 1.859692 | 6.421760 |
| 2 | 0.14 | 0.206000 | 1.830325 | 6.235916 |
| 3 | 0.16 | 0.206000 | 1.703968 | 5.495709 |
| **Aggregate** | **0.16** | **0.2082** | 1.7980 | 6.0511 |

Compute (TS-v1.1 additive): train 2,581.135 s + in-workload eval 203.297 s +
recovery 99.21 s = **2,883.64 s ≈ 48.1 min** (model-load/checkpoint NESTED;
failed attempt-1 sweep ≈275 s = discarded overhead).

### Governance guarantees demonstrated
- The original attempt remains permanently INTERRUPTED-evidence (never
  relabeled COMPLETED); recovery-01 is a distinct prediction-only attempt
  with its own manifest, lock, and provenance chain.
- Original attempt directory byte-identical (32/32 files) after recovery
  and reseal.
- Every report carries exact checksums; superseded versions preserved
  timestamped with reasons; frozen trackers never modified.

---

## 5. Engineering standards established

1. **Guarded execution**: token-gated single-run harness; fresh-directory
   guards; execution locks (stale locks removed only as documented operator
   actions); stop-on-failure with no automatic retry; fail-closed completion
   (COMPLETED requires explicit training AND evaluation evidence plus full
   recursive artifact integrity).
2. **Fail-closed integrity**: symlinks, path escapes, empty artifact dirs,
   and unexpected file types rejected; atomic JSON writes (tmp + fsync +
   os.replace); completion markers cross-checked against manifests.
3. **Timing discipline (TS-v1.1)**: train/eval/transitions additive;
   model-load and checkpoint serialization are NESTED diagnostics never
   double-counted; process_wall informational only.
4. **Provenance discipline**: evidence hierarchy (raw artifacts >
   machine-readable manifests > audited reports > summaries); append-only
   trackers; timestamped supersession; discrepancy ledgers that preserve
   incident history.
5. **Testing discipline**: 388 tests passing; fail-closed semantics pinned
   by tests; strict-xfail used to encode known defects until fixed.
6. **Device discipline**: MPS required; silent CPU fallback raises.

---

## 6. Current claims ledger (summary)

| Claim | Status |
|---|---|
| Continual adaptation beats frozen base | SUPPORTED (single-seed caveat) |
| Cost–quality Pareto frontier over rank/budget | SUPPORTED |
| Drift detection avoids unnecessary adaptation | SUPPORTED |
| Checkpointing reduces lost work | SUPPORTED (local); real-cloud NOT validated |
| Dynamic resource allocation improves efficiency | PARTIALLY SUPPORTED (degenerate case; multi-resource pending) |
| Adapter depth affects CL performance (D100 > D25) | SUPPORTED |
| Dynamic rank adapts capacity efficiently | DIRECTIONAL (expansion preserved quality) |
| PPO effective allocation / generalization / real-cloud / multi-seed / external baselines | NOT VALIDATED (future work) |
| Static-R9 s42 final-only data point | SUPPORTED — **eligible for P4a/P4b** |
| Recovered predictions identical to original generations | VERIFIED (bit-exact) |

---

## 7. What is proposed from now on

### 7.1 Next execution — `e7-static-r9-s43` (READY)
Static-R9, seed 43, ranks 9→9→9, α 18→18→18, final-only. Matrix row 9, NEW,
**READY FOR SEPARATE EXECUTION AUTHORIZATION** (no directory, no lock).
Template: `scripts/e7_pilot_static_r9_s42.py`; harness token
`i-authorize-e7-phase2-execution`; expected ≈2,784 s active compute + ≈99 s
prediction sweep (based on s42 measurements). The corrected adapter makes
the attempt-1 failure mode structurally impossible (single-wrap load +
fingerprint gate + schema-complete records).

### 7.2 Critical path to 27 November (15 pending runs, central 44,480 s ≈ 12.4 h)
1. `e7-static-r9-s43`, then `e7-static-r9-s44` — completes Static-R9;
   **P4a/P4b become computable**.
2. Static-R4 s42/43/44 → Static-R16 s42/43/44 → Static-R32 s42/43/44
   (completes the static rank–quality curve).
3. `e7-grow-s43/s44`, then `e7-reduce-s43/s44` (completes the dynamic
   seed sets).
Full plan: tracker v1.1.5 `NOV23_27_CRITICAL_PATH_V1_1_5.md`.

### 7.3 Decisions proposed for the owner
1. **Resource-Aware unblock**: authorize the small E5 loss-signal
   instrumentation, or formally defer the pilot out of v5 scope.
2. **P4a/P4b timing**: run the paired analyses as soon as the relevant
   seed sets complete (early, clearly non-confirmatory signal) vs waiting
   for all 15 runs.
3. **Static forecast optimism**: s42 measured 2,784 s vs 2,200 s central
   estimate (+27%); keep estimates frozen for like-for-like planning or lift
   static centrals to ≈2,800 s for the remaining 11 static runs.
4. **Optional pre-sweep smoke gate** (2 examples) before each future
   prediction sweep — cheap insurance; the pathway now self-gates anyway.

### 7.4 Analysis plan (frozen)
P4a (Grow vs Static-R9) and P4b (Reduce vs Static-R9) once the seed sets
complete: exact paired sign-flip at 3 seeds (min p = 0.25 → descriptive),
Holm correction across five primary comparisons, per-seed EM reported
individually. Static policies also feed the rank–quality curve
(R4/R8[reuse]/R9/R16/R32). No static-vs-dynamic BWT claims (statics are
final-only by design). E8–E12 remain future work pending design.

### 7.5 Standing constraints
One experiment at a time under explicit authorization; no downloads, no
CUDA, no cloud; frozen artifacts never modified; every execution sealed
with checksums and (when needed) its own recovery/ Reseal provenance.

---

## 8. Quick reference

- **Authoritative matrix**: `results/e7/design_20261120/v5_final_reconciliation/FINAL_RUN_MATRIX.csv`
- **Current tracker**: `results/project_master_tracker_20261123_v1_1_5_1/`
  (ZIP SHA-256 `b1b42f23151f5a546280ce1fc0b19a77b5f5fcefc3d1cd5b2805b6de7513bd11`)
- **Recovery manifest (resealed)**:
  `results/e7/recovery_static_r9_s42_20261123/recovery_manifest.json`
- **Detailed Static-R9 narrative**: `README_E7_STATIC_R9_20261123.md`
- **Test suite**: `pytest tests/ -q` → 388 passed, exit 0
- **Harness token**: `i-authorize-e7-phase2-execution`
- **Timing scope convention**: TS-v1.1 (train/eval/transitions additive;
  model-load & checkpoint NESTED diagnostics)
