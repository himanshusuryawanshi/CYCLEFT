# CHANGES.md — WikiFactDiff E1 Migration
## Branch `feature/wikifactdiff-e1` (base: main @ 6010bb2, head: a938a3f)

This file documents every change made in the WikiFactDiff E1 migration
(3 commits, 15 files). The existing TemporalWiki baseline remains
reproducible: its adapter, config and outputs are untouched, and the
default `prepare_event_datasets(...)` behavior is byte-identical.

---

## 1. Files created (10)

| File | Purpose |
|---|---|
| `src/cycleft/data/base.py` | Dataset-adapter protocol + deterministic partition helpers (`stable_hash` via sha256 — Python's salted `hash()` is never reproducible and is banned for partitioning), fail-closed adapter registry, `DATASET_TEMPORALWIKI` / `DATASET_WIKIFACTDIFF` names |
| `src/cycleft/data/wikifactdiff.py` | The WikiFactDiff adapter (the core of the migration — details in §3) |
| `src/cycleft/e1/wfd_metrics.py` | M1/M2/M3 continual-learning metrics over WikiFactDiff prediction records: NSR (new-answer success), OAR (obsolete-answer rate), IUR (invalid/unmatched — the gaming detector), UE (update efficacy = matched-new ∧ ¬matched-obsolete, scored against **fact-local** answers — never a global blacklist), M2 baseline-conditioned transfer (initially-known vs initially-unknown), M3 neighborhood retention drift + obsolete-answer persistence rate |
| `configs/e1_stream_wikifactdiff.yaml` | WikiFactDiff E1 config: pinned dataset (rev `bb17ffbff7…`), partition salt `cycleft-wfd-v1`, split fractions 0.8/0.9, `eval_population: pilot` (100/event), pilot 64 train/event, seed 42; training hyperparameters **identical** to the TemporalWiki baseline (dataset migration is not confounded with training changes) |
| `tests/test_wikifactdiff.py` | 25 tests: partition determinism/exclusivity/coverage, relation stratification reproducibility, expansion + dedup, literal normalization, prompt-mask leakage, neighborhood exclusion/dedup, multi-reference metrics, M1/M2/M3 correctness (incl. anti-gaming), full-eval safeguard, TemporalWiki compatibility, opt-in real-data integration (`CYCLEFT_WFD_INTEGRATION=1`) |
| `data/e1/wikifactdiff_manifest.json` | Verified real-data manifest (see §4) |
| `hpc/HPC_WIKIFACTDIFF.md` | IIT Delhi HPC run book: env setup (module-aware, no unverified CUDA), dataset prep, GPU job template with `<placeholders>` (no invented project IDs), checkpoint/manifest conventions, recovery instructions, M1/M3 sweep notes |
| `hpc_requirements_key_versions.txt` | Key dependency versions from the working environment (torch/transformers/peft/datasets/accelerate) for HPC env replication |
| `scripts/e7_pilot_static_r9_s43.py` | (from the earlier E7 session) Static-R9 seed-43 pilot — tracked now |
| `scripts/e7_pilot_static_r9_s44.py` | (from the earlier E7 session) Static-R9 seed-44 pilot — tracked now |

## 2. Files modified (5)

| File | Change |
|---|---|
| `src/cycleft/e1/data.py` | **Dispatch only.** `prepare_event_datasets` gains `dataset="temporalwiki"` + `dataset_config=None` kwargs; unknown names fail closed; unknown eval_limit safeguard untouched. TemporalWiki logic moved verbatim — default behavior identical, so E2–E7 callers (which pass no new kwargs) are unaffected |
| `src/cycleft/e1/pipeline.py` | `E1Run.prepare()` reads `stream.dataset.name` + `stream.dataset.adapter` from the resolved config and passes them to the dispatcher (TemporalWiki configs resolve to the same path as before) |
| `src/cycleft/e1/metrics.py` | **Added** `exact_match_multi` / `token_f1_multi` (multi-reference; empty reference list ⇒ 0.0). Existing single-reference functions untouched |
| `src/cycleft/e1/lm.py` | `evaluate_event` honors an optional `valid_answers` row key → multi-reference scoring; prediction records carry `valid_answers` when present. TemporalWiki rows (no such key) behave exactly as before |
| `scripts/e1_prepare.py` | Reads `stream.dataset.name`/`.adapter` from the config and passes them to the dispatcher; prints event counts (works for both datasets) |

## 3. The WikiFactDiff adapter (`wikifactdiff.py`) — what it implements

- **Dataset pin**: `Orange/WikiFactDiff`, config `20210104-20230227_legacy`, revision `bb17ffbff7…` (recommended config per the LREC-COLING 2024 paper).
- **Events (measured partition — mutually exclusive, 100% coverage)**:
  E1 insertion `subject_is_ph_new=true` (132,857 rows) ·
  E2 other updates `ph_new=false ∧ is_replace=false` (161,958) ·
  E3 replacement `is_replace=true` (32,873).
- **Partition `wfd-partition-v1`**: subject-disjoint 80/10/10, salt
  `cycleft-wfd-v1`, **relation-aware stratification** (subjects grouped by
  dominant relation, ordered by sha256(salt‖subject) within strata,
  cumulative-fraction assignment) — explicitly *not* a plain hash split.
- **Training construction**: one example per `new` object (deterministic
  object ordering: normalized label, literal-last, id); frozen label
  normalization (strip `+`, strip); exact duplicate (prompt, answer) pairs
  removed; `static` = eval-valid only; `obsolete` = measurement-only;
  obsolete-only rows (4,157) = persistence probes (never trained); rows
  with normalized new/obsolete overlap excluded (0 under the frozen
  normalization); `subject_popularity` never used for within-row ranking.
- **Leakage protections**: subject disjointness; prompt masking of any
  val/test prompt colliding with train-partition prompts (update or
  generalization); val/test collisions resolve in favour of test;
  neighborhood probes exclude every fact key belonging to any partition
  and dedup by fact key; all exclusions recorded with reasons.
- **Evaluation populations**: M2 held-out facts travel in `PreparedEvent.eval_rows`
  (evaluated in-pipeline after every event, existing machinery);
  M1 (trained facts' generalization prompts) and M3 (neighborhood +
  persistence) are manifest `sweep_populations` evaluated per checkpoint
  with the multi-reference evaluator.

## 4. Verified real-data counts (manifest)

327,688 rows = 327,688 unique facts; expanded 328,079 → deduplicated
327,837 (242 exact duplicates removed); decisions new 410,449 / obsolete
39,384 / static 4,532; degenerate 0 (frozen normalization); 4,157
persistence probes; partitions 112,001/13,970/13,840 subjects
(262,880/32,590/32,218 facts); 4,895 prompt-mask exclusions; 1,238,011
neighborhood probes (569,238 colliding occurrences excluded).

## 5. Test results

| Suite | Before | After |
|---|---|---|
| Baseline full suite (main) | 388 passed | — |
| New WikiFactDiff tests | — | 24 passed, 1 skipped (opt-in integration) |
| Real-data integration | — | 1 passed (full 328k rows) |
| **Full repository suite (branch head)** | 388 | **412 passed, 1 skipped, exit 0** |

## 6. Commits

1. `9ff5f7e` — feat(data): adapter + dispatch + config + manifest
2. `99a48ce` — feat(e1): multi-reference evaluation + M1/M2/M3 metrics + tests
3. `a938a3f` — docs(hpc): run book + dependency versions + E7 pilot scripts

## 7. What was NOT changed

- `src/cycleft/data/temporalwiki.py`, `configs/e1_stream.yaml`,
  TemporalWiki experiment outputs — untouched (reproducible baseline).
- E2–E7 sources (they call the dispatcher with default arguments).
- E5 ResumableTrainer, E7 harness/orchestrator, production training
  hyperparameters, the frozen E7 v5 matrix.
- No model training was run on the MacBook; the pilot/full training runs
  belong on IIT Delhi HPC (`hpc/HPC_WIKIFACTDIFF.md`).
