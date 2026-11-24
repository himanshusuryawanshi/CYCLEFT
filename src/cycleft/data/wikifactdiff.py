"""WikiFactDiff dataset adapter for E1 continual learning.

Dataset: ``Orange/WikiFactDiff`` (HF), configuration
``20210104-20230227_legacy`` — the recommended configuration whose
creation process is described in Ammar Khodja et al., *"WikiFactDiff: A
Large, Realistic, and Temporally Adaptable Dataset for Atomic Factual
Knowledge Update in Causal Language Models"*, LREC-COLING 2024.

Event definitions (Option A — update-type curriculum; measured on the
full dataset to be mutually exclusive and to cover 100% of rows):

* ``wfd_e1_insertion``  : ``subject_is_ph_new == true``            (132,857)
* ``wfd_e2_other_update``: ``subject_is_ph_new == false and
                           is_replace == false``                   (161,958)
* ``wfd_e3_replacement``: ``is_replace == true``                   (32,873)

Partitioning (``wfd-partition-v1``, subject-disjoint 80/10/10 with
relation-aware stratification):

1. Every fact belongs to exactly one subject; all facts of a subject are
   assigned to the same partition (no within-subject splits).
2. Each subject is stratified by its **dominant relation** (the relation
   with the most facts for that subject; ties broken by relation id) so
   every partition sees the same relation mix (a plain hash split does
   NOT stratify by relation).
3. Within a stratum, subjects are ordered by ``sha256(salt | subject_id)``
   (deterministic across processes — never Python's salted ``hash``) and
   assigned to train/val/test by cumulative fraction (0.8 / 0.9).

Strict train/val/test separation:

* Training, validation and final-test rows are built from their own
  partitions only; a fact's partition decides everything.
* Training prompts (update or generalization) never appear in
  validation or test (masked; exclusions recorded).
* Validation/test prompt collisions resolve in favour of **test**:
  colliding prompts are dropped from validation so validation-based
  model selection cannot expose final-test evaluation.

Answer semantics (fact-local, never a global blacklist):

* ``new_answers``      : valid at T_new — the only training targets.
* ``static_answers``   : unchanged inside a changed row — valid at
  T_new, never trained on.
* ``obsolete_answers`` : true at T_old only — measurement-only.
* ``valid_answers``    = new ∪ static (current factual correctness;
  multi-reference evaluation).
* Obsolete-only rows carry no valid current answer: they are excluded
  from ordinary QA training/evaluation and retained as **persistence
  probes** (``eval_family="persistence_probe"``).
* Perplexity is **disabled** for this dataset
  (``perplexity_disabled=True``): a defensible multi-reference
  likelihood is not implemented, and empty/obsolete completions must
  never enter the loss.  TemporalWiki perplexity is unchanged.

Sweep populations (M1/M3) are deterministically sampled to configured
sizes and evaluated at baseline and after every event by the pipeline;
sizes, method, seed and exclusions are recorded in the manifest.
"""

from __future__ import annotations

import collections
from typing import Any, Callable

import time

from cycleft.data.base import stable_hash


class _PhaseTimer:
    """Records per-phase wall time and row counts into the manifest.

    Permanent instrumentation (HPC readiness Task 2): the timing table is
    part of the dataset manifest so integration performance is auditable
    on every machine.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.phases: dict[str, dict[str, Any]] = {}

    def __init__(self, log=None) -> None:
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._log = log or (lambda m: print(m, flush=True))
        self.phases: dict[str, dict[str, Any]] = {}

    def mark(self, phase: str, records: int = 0, objects: int = 0) -> None:
        now = time.perf_counter()
        phase_s = now - self._last
        total_s = now - self._t0
        self.phases[phase] = {
            "elapsed_s": round(total_s, 3),
            "phase_s": round(phase_s, 3),
            "records": records,
            "objects_created": objects,
        }
        self._log(
            f"[wfd-timer] {phase}: phase={phase_s:.1f}s "
            f"total={total_s:.1f}s records={records} objects={objects}")
        self._last = now

DATASET_ID = "Orange/WikiFactDiff"
DATASET_CONFIG_NAME = "20210104-20230227_legacy"
DATASET_REVISION = "bb17ffbff7b2d28e4cd12e251af3db50d7fa18ea"
PARTITION_SALT = "cycleft-wfd-v1"
PARTITION_ALGORITHM = "wfd-partition-v1"
SPLIT_FRACTIONS = (0.80, 0.90)  # cumulative: train < 0.8, val < 0.9, test rest

EVENT_TAGS = ("wfd_e1_insertion", "wfd_e2_other_update", "wfd_e3_replacement")
EVENT_DEFINITIONS = {
    "wfd_e1_insertion": "subject_is_ph_new == true",
    "wfd_e2_other_update": "subject_is_ph_new == false and is_replace == false",
    "wfd_e3_replacement": "is_replace == true",
}

QUESTION_COLUMN = "question"
ANSWER_COLUMN = "object"

LABEL_NORMALIZATION_VERSION = "wfd-strip-plus-v1"
ANSWER_MATCH_NORMALIZATION_VERSION = "e1-normalize-answer-v1"

DEFAULT_SWEEP_LIMITS = {"m1": 200, "persistence": 200, "neighborhood": 2000}
FULL_SWEEP_LIMITS = {"m1": 2000, "persistence": 2000, "neighborhood": 5000}
SWEEP_FAMILIES = ("m1", "persistence", "neighborhood")


def normalize_wfd_label(label: str) -> str:
    """Frozen target normalization (v1): strip ``+`` prefix and whitespace.

    Case is preserved in training completions; evaluation matching applies
    the existing ``cycleft.e1.metrics.normalize_answer`` on both sides.
    """
    text = str(label).strip()
    return text[1:] if text.startswith("+") else text


def validate_adapter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fail clearly on invalid adapter configuration values."""
    if not isinstance(config, dict):
        raise ValueError("WikiFactDiff adapter config must be a mapping.")
    eval_population = config.get("eval_population")
    if eval_population is not None and eval_population not in ("pilot", "full"):
        raise ValueError(
            f"eval_population must be 'pilot' or 'full', got {eval_population!r}"
        )
    fractions = tuple(config.get("split_fractions", SPLIT_FRACTIONS))
    if len(fractions) != 2 or not all(0 < f < 1 for f in fractions) \
            or fractions[0] >= fractions[1]:
        raise ValueError(
            f"split_fractions must be two ascending cuts in (0,1), got {fractions!r}"
        )
    pilot_eval = int(config.get("pilot_eval_per_event", 100))
    if pilot_eval <= 0:
        raise ValueError(
            f"pilot_eval_per_event must be positive, got {pilot_eval}"
        )
    salt = str(config.get("salt", PARTITION_SALT))
    if not salt:
        raise ValueError("salt must be a non-empty string.")
    sweep_limits = config.get("sweep_limits") or {}
    if not isinstance(sweep_limits, dict):
        raise ValueError("sweep_limits must be a mapping.")
    for family, limit in sweep_limits.items():
        if family not in SWEEP_FAMILIES:
            raise ValueError(f"Unknown sweep family {family!r}")
        if int(limit) <= 0:
            raise ValueError(f"sweep limit for {family} must be positive")
    return config


def _event_tag(row: dict) -> str:
    if bool(row["subject_is_ph_new"]):
        return "wfd_e1_insertion"
    if bool(row["is_replace"]):
        return "wfd_e3_replacement"
    return "wfd_e2_other_update"


def _answer_sets(row: dict) -> tuple[set[str], set[str], set[str]]:
    """Normalized (new, static, obsolete) label sets of one row."""
    new_labels: set[str] = set()
    static_labels: set[str] = set()
    obsolete_labels: set[str] = set()
    for o in row["objects"]:
        label = normalize_wfd_label(o["label"])
        if o["decision"] == "new":
            new_labels.add(label)
        elif o["decision"] == "static":
            static_labels.add(label)
        elif o["decision"] == "obsolete":
            obsolete_labels.add(label)
    return new_labels, static_labels, obsolete_labels


def assign_partitions(
    rows: list[dict],
    salt: str = PARTITION_SALT,
    fractions: tuple[float, ...] = SPLIT_FRACTIONS,
) -> dict[str, str]:
    """Relation-aware stratified subject-disjoint assignment (v1).

    Returns ``{subject_id: partition}``.  See the module docstring for
    the algorithm; the version string is recorded in every manifest.
    """
    facts_per_subject: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.Counter()
    )
    for row in rows:
        facts_per_subject[row["subject"]["id"]][row["relation"]["id"]] += 1

    strata: dict[str, list[str]] = collections.defaultdict(list)
    for subject_id in sorted(facts_per_subject):
        counts = facts_per_subject[subject_id]
        dominant = min(counts, key=lambda rid: (-counts[rid], rid))
        strata[dominant].append(subject_id)

    assignment: dict[str, str] = {}
    for relation_id in sorted(strata):
        subjects = sorted(
            strata[relation_id],
            key=lambda sid: stable_hash(f"{salt}|{sid}"),
        )
        n = len(subjects)
        for position, subject_id in enumerate(subjects):
            fraction = position / n if n else 0.0
            if fraction < fractions[0]:
                assignment[subject_id] = "train"
            elif fraction < fractions[1]:
                assignment[subject_id] = "val"
            else:
                assignment[subject_id] = "test"
    return assignment


def _fact_key(row: dict) -> tuple[str, str]:
    return (row["subject"]["id"], row["relation"]["id"])


def _build_example(
    row: dict,
    *,
    question: str,
    primary_answer: str,
    source_row_index: int | None,
    partition: str,
    event_tag: str,
    eval_family: str | None = None,
    prompt_kind: str | None = None,
    perplexity_disabled: bool = False,
    object_index: int | None = None,
) -> dict:
    """One schema-complete example with full fact-local answer metadata."""
    new_answers = sorted(_answer_sets(row)[0])
    static_answers = sorted(_answer_sets(row)[1])
    obsolete_answers = sorted(_answer_sets(row)[2])
    valid_answers = sorted(set(new_answers) | set(static_answers))
    example: dict[str, Any] = {
        "question": question,
        "object": primary_answer,
        "new_answers": new_answers,
        "static_answers": static_answers,
        "obsolete_answers": obsolete_answers,
        "valid_answers": valid_answers,
        "decision_set": sorted({o["decision"] for o in row["objects"]}),
        "fact_key": f"{row['subject']['id']}|{row['relation']['id']}",
        "subject_id": row["subject"]["id"],
        "relation_id": row["relation"]["id"],
        "partition": partition,
        "event_tag": event_tag,
        "subject_is_ph_new": bool(row["subject_is_ph_new"]),
        "is_replace": bool(row["is_replace"]),
        "subject_popularity": float(row["subject_popularity"]),
        "gen_prompts": list(row["generalization_prompts"]),
        "perplexity_disabled": perplexity_disabled,
    }
    if eval_family is not None:
        example["eval_family"] = eval_family
    if prompt_kind is not None:
        example["prompt_kind"] = prompt_kind
    if source_row_index is not None:
        example["source_row_index"] = source_row_index
    if object_index is not None:
        example["object_index"] = object_index
    return example


def build_wikifactdiff_events(
    rows: list[dict],
    *,
    seed: int,
    train_limit: int | None = None,
    mode: str | None = None,
    adapter_config: dict[str, Any] | None = None,
    log=None,
) -> tuple[list, dict]:
    """Pure event-construction core (no HF access) — unit-testable.

    Takes materialized WikiFactDiff rows (schema of the
    ``20210104-20230227_legacy`` config) and returns
    ``(list[PreparedEvent], manifest)``.  ``mode`` ("pilot"/"full")
    determines the default evaluation population when the adapter config
    does not pin one explicitly; requesting ``pilot`` populations in
    ``full`` mode is a configuration error (never a silent downgrade).
    """
    timer = _PhaseTimer(log=log or (lambda m: print(f'[wfd] {m}', flush=True)))
    config = validate_adapter_config(dict(adapter_config or {}))
    timer.mark("config_validation")
    fractions = tuple(config.get("split_fractions", SPLIT_FRACTIONS))
    salt = str(config.get("salt", PARTITION_SALT))
    configured_population = config.get("eval_population")
    if configured_population is None:
        if mode not in ("pilot", "full"):
            raise ValueError(
                "eval_population is not pinned in the adapter config and "
                f"mode={mode!r} cannot derive it (expected 'pilot' or 'full')."
            )
        eval_population = mode
    else:
        if mode == "full" and configured_population == "pilot":
            raise ValueError(
                "Refusing to evaluate the FULL experiment with PILOT "
                "evaluation populations (would silently weaken the run). "
                "Set eval_population='full' for full mode."
            )
        eval_population = configured_population
    pilot_eval_per_event = int(config.get("pilot_eval_per_event", 100))
    sweep_limits = {
        **(FULL_SWEEP_LIMITS if eval_population == "full" else DEFAULT_SWEEP_LIMITS),
        **(config.get("sweep_limits") or {}),
    }
    if train_limit is not None and train_limit <= 0:
        raise ValueError(f"train_limit must be positive or None, got {train_limit!r}")

    timer.mark("schema_rows_materialized", records=len(rows))
    assignment = assign_partitions(rows, salt=salt, fractions=fractions)
    partition_subjects = collections.Counter(assignment.values())
    partition_facts = collections.Counter(
        assignment[row["subject"]["id"]] for row in rows
    )

    # ---- per-fact classification and exclusions ---------------------------
    degenerate: list[dict] = []
    persistence_fact_keys: set[tuple[str, str]] = set()
    decision_counts_total: collections.Counter = collections.Counter()
    for row_index, row in enumerate(rows):
        for o in row["objects"]:
            decision_counts_total[o["decision"]] += 1
        new_labels, _, obsolete_labels = _answer_sets(row)
        fk = _fact_key(row)
        if new_labels & obsolete_labels:
            degenerate.append(
                {
                    "fact_key": f"{fk[0]}|{fk[1]}",
                    "reason": "degenerate_new_obsolete_overlap",
                    "partition": assignment[row["subject"]["id"]],
                    "source_row_index": row_index,
                }
            )
        elif not new_labels and obsolete_labels:
            persistence_fact_keys.add(fk)
    timer.mark("classification_and_exclusions", records=len(rows))
    degenerate_keys = {tuple(e["fact_key"].split("|", 1)) for e in degenerate}
    partition_fact_keys = {_fact_key(r) for r in rows}
    # NOTE: obsolete-only rows are NOT excluded from the dataset — they are
    # reclassified as persistence probes (no deletion-training mechanism).

    # ---- strict train / val / test fact assignment ------------------------
    train_pool_by_event: dict[str, list[dict]] = {tag: [] for tag in EVENT_TAGS}
    raw_val_rows: dict[str, list[tuple[int, dict]]] = {
        tag: [] for tag in EVENT_TAGS
    }
    raw_test_rows: dict[str, list[tuple[int, dict]]] = {
        tag: [] for tag in EVENT_TAGS
    }
    expanded_total = 0
    for row_index, row in enumerate(rows):
        fk = _fact_key(row)
        partition = assignment[row["subject"]["id"]]
        tag = _event_tag(row)
        new_labels, _, _ = _answer_sets(row)
        if partition == "train":
            if fk in persistence_fact_keys or fk in degenerate_keys:
                continue  # persistence probes / degenerate rows never train
            ordered_new = sorted(new_labels)  # deterministic object order
            for object_index, answer in enumerate(ordered_new):
                example = _build_example(
                    row,
                    question=row["update_prompt"],
                    primary_answer=answer,
                    source_row_index=row_index,
                    partition="train",
                    event_tag=tag,
                    perplexity_disabled=True,
                    object_index=object_index,
                )
                example["expanded_index"] = expanded_total
                expanded_total += 1
                train_pool_by_event[tag].append(example)
        elif partition == "val":
            raw_val_rows[tag].append((row_index, row))
        else:
            raw_test_rows[tag].append((row_index, row))

    timer.mark("training_expansion", records=len(train_pool_by_event["wfd_e1_insertion"])
               + len(train_pool_by_event["wfd_e2_other_update"])
               + len(train_pool_by_event["wfd_e3_replacement"]),
               objects=expanded_total)
    # deduplicate exact (question, object) pairs within each event pool
    event_train_expanded: dict[str, list[dict]] = {}
    dedup_removed = 0
    for tag in EVENT_TAGS:
        seen: set[tuple[str, str]] = set()
        kept: list[dict] = []
        for example in train_pool_by_event[tag]:
            pair = (example["question"], example["object"])
            if pair in seen:
                dedup_removed += 1
                continue
            seen.add(pair)
            kept.append(example)
        event_train_expanded[tag] = kept

    # ---- prompt masks (train first; val protects final test) --------------
    train_prompts: set[str] = set()
    train_gen_prompts: set[str] = set()
    for tag in EVENT_TAGS:
        for example in event_train_expanded[tag]:
            train_prompts.add(example["question"])
        for row_index, row in raw_val_rows[tag]:
            pass
    for row_index, row in enumerate(rows):
        fk = _fact_key(row)
        if assignment[row["subject"]["id"]] != "train":
            continue
        if fk in persistence_fact_keys or fk in degenerate_keys:
            continue
        train_gen_prompts.update(row["generalization_prompts"])
    train_all_prompts = train_prompts | train_gen_prompts

    def _masked(prompts, blocked, reason, fk, partition):
        kept, dropped = [], []
        for prompt in prompts:
            if prompt in blocked:
                dropped.append(
                    {"fact_key": f"{fk[0]}|{fk[1]}", "partition": partition,
                     "reason": reason, "prompt": prompt}
                )
            else:
                kept.append(prompt)
        return kept, dropped

    # test: masked against train prompts only (final test is protected)
    final_test_rows: dict[str, list[dict]] = {tag: [] for tag in EVENT_TAGS}
    final_test_prompts: set[str] = set()
    mask_exclusions: list[dict] = []
    test_facts_dropped = 0
    for tag in EVENT_TAGS:
        for row_index, row in raw_test_rows[tag]:
            fk = _fact_key(row)
            upd_kept, upd_dropped = _masked(
                [row["update_prompt"]], train_all_prompts,
                "train_prompt_collision", fk, "test")
            gen_kept, gen_dropped = _masked(
                list(row["generalization_prompts"]), train_all_prompts,
                "train_prompt_collision", fk, "test")
            mask_exclusions.extend(upd_dropped + gen_dropped)
            if not upd_kept and not gen_kept:
                test_facts_dropped += 1
                mask_exclusions.append(
                    {"fact_key": f"{fk[0]}|{fk[1]}", "partition": "test",
                     "reason": "all_prompts_masked"})
                continue
            new_labels, static_labels, _ = _answer_sets(row)
            current = sorted(new_labels or static_labels)
            for prompt in upd_kept:
                final_test_prompts.add(prompt)
                final_test_rows[tag].append(
                    _build_example(
                        row, question=prompt,
                        primary_answer=current[0] if current else "",
                        source_row_index=row_index, partition="test",
                        event_tag=tag, eval_family="heldout_update",
                        prompt_kind="update_prompt", perplexity_disabled=True))
            for prompt in gen_kept:
                final_test_prompts.add(prompt)
                final_test_rows[tag].append(
                    _build_example(
                        row, question=prompt,
                        primary_answer=current[0] if current else "",
                        source_row_index=row_index, partition="test",
                        event_tag=tag, eval_family="heldout_paraphrase",
                        prompt_kind="generalization_prompt",
                        perplexity_disabled=True))

    # val: masked against train AND final-test prompts (test is protected).
    # The blocked set is computed ONCE — a per-fact union of ~1.5M-element
    # sets was measured at 8,550 s for ~30k facts (HPC readiness Task 2).
    final_val_rows: dict[str, list[dict]] = {tag: [] for tag in EVENT_TAGS}
    val_facts_dropped = 0
    val_blocked_prompts = train_all_prompts | final_test_prompts
    for tag in EVENT_TAGS:
        for row_index, row in raw_val_rows[tag]:
            fk = _fact_key(row)
            blocked = val_blocked_prompts
            upd_kept, upd_dropped = _masked(
                [row["update_prompt"]], blocked,
                "train_or_test_prompt_collision", fk, "val")
            gen_kept, gen_dropped = _masked(
                list(row["generalization_prompts"]), blocked,
                "train_or_test_prompt_collision", fk, "val")
            mask_exclusions.extend(upd_dropped + gen_dropped)
            if not upd_kept and not gen_kept:
                val_facts_dropped += 1
                mask_exclusions.append(
                    {"fact_key": f"{fk[0]}|{fk[1]}", "partition": "val",
                     "reason": "all_prompts_masked"})
                continue
            new_labels, static_labels, _ = _answer_sets(row)
            current = sorted(new_labels or static_labels)
            for prompt in upd_kept:
                final_val_rows[tag].append(
                    _build_example(
                        row, question=prompt,
                        primary_answer=current[0] if current else "",
                        source_row_index=row_index, partition="val",
                        event_tag=tag, eval_family="validation_update",
                        prompt_kind="update_prompt", perplexity_disabled=True))
            for prompt in gen_kept:
                final_val_rows[tag].append(
                    _build_example(
                        row, question=prompt,
                        primary_answer=current[0] if current else "",
                        source_row_index=row_index, partition="val",
                        event_tag=tag, eval_family="validation_paraphrase",
                        prompt_kind="generalization_prompt",
                        perplexity_disabled=True))

    timer.mark("eval_population_construction", records=len(final_test_rows["wfd_e1_insertion"])
               + len(final_test_rows["wfd_e2_other_update"])
               + len(final_test_rows["wfd_e3_replacement"]))
    # pilot sampling: independent deterministic samples for test and val
    if eval_population == "pilot":
        from cycleft.e1.data import deterministic_subset
        for tag in EVENT_TAGS:
            for population in (final_test_rows, final_val_rows):
                if len(population[tag]) > pilot_eval_per_event:
                    sampled, _indices = deterministic_subset(
                        population[tag], pilot_eval_per_event, seed)
                    population[tag] = sampled

    # ---- training selection ----------------------------------------------
    from cycleft.e1.data import PreparedEvent, deterministic_subset

    events = []
    manifest_events = []
    train_provenance = {}
    for event_id, tag in enumerate(EVENT_TAGS, start=1):
        pool = event_train_expanded[tag]
        train_rows, selected_indices = deterministic_subset(pool, train_limit, seed)
        eval_rows = final_test_rows[tag]
        val_rows = final_val_rows[tag]
        events.append(
            PreparedEvent(
                event_id=event_id,
                name=tag,
                slice_tag=tag,
                train_rows=train_rows,
                eval_rows=eval_rows,
                train_pool_size=len(pool),
                eval_pool_size=len(eval_rows),
                train_indices_used=selected_indices,
                validation_rows=val_rows,
                validation_pool_size=len(val_rows),
            )
        )
        train_provenance[tag] = {
            "n_expanded_pool": len(pool),
            "selected_expanded_indices": selected_indices,
            "selected_source_row_indices": sorted(
                {ex.get("source_row_index") for ex in train_rows}
            ),
            "selected_fact_keys": sorted({ex["fact_key"] for ex in train_rows}),
            "n_train_used": len(train_rows),
            "n_validation_rows": len(val_rows),
        }
        manifest_events.append(
            {
                "event_id": event_id,
                "name": tag,
                "event_definition": EVENT_DEFINITIONS[tag],
                "n_train_pool": len(pool),
                "n_train_used": len(train_rows),
                "n_eval_rows": len(eval_rows),
                "n_validation_rows": len(val_rows),
                "eval_population": eval_population,
                "question_column": QUESTION_COLUMN,
                "answer_column": ANSWER_COLUMN,
            }
        )

    # ---- sweep populations (deterministic, size-capped, two-phase) -------
    # Phase A collects LIGHTWEIGHT candidate references (no example dicts);
    # phase B deterministically samples candidate positions and
    # materializes ONLY the selected entries.  This keeps memory bounded
    # (the full pools would be ~2.2M example dicts).

    # M1 candidates: (train_example_ref, gen_index) — the example dicts
    # already exist in event_train_expanded, so candidates are references.
    m1_candidates: list[tuple[dict, int]] = []
    for tag in EVENT_TAGS:
        for example in event_train_expanded[tag]:
            for gen_index, _prompt in enumerate(example["gen_prompts"]):
                m1_candidates.append((example, gen_index))

    # persistence candidates: (row_ref, row_index)
    persistence_candidates: list[tuple[dict, int]] = []
    for row_index, row in enumerate(rows):
        if _fact_key(row) in persistence_fact_keys:
            persistence_candidates.append((row, row_index))

    partition_prompts = set(final_test_prompts) | train_all_prompts
    for tag in EVENT_TAGS:
        partition_prompts.update(ex["question"] for ex in final_val_rows[tag])
        partition_prompts.update(ex["question"] for ex in final_test_rows[tag])

    # neighborhood candidates: (row_ref, neighbor_ref, probe_key) for
    # candidates passing key-exclusion, key-dedup and prompt/label
    # viability; occurrences/unique keys/exclusions counted separately.
    neighborhood_occurrences = 0
    neighborhood_unique_keys: set[tuple[str, str]] = set()
    neighborhood_excluded_occurrences = 0
    neighborhood_candidates: list[tuple[dict, dict, tuple[str, str]]] = []
    seen_probe_keys: set[tuple[str, str]] = set()
    for row in rows:
        rid = row["relation"]["id"]
        for nb in row["neighborhood"]:
            probe_key = (nb["subject"]["id"], rid)
            neighborhood_occurrences += 1
            neighborhood_unique_keys.add(probe_key)
            if probe_key in partition_fact_keys:
                neighborhood_excluded_occurrences += 1
                continue  # belongs to a dataset partition: never a probe
            if probe_key in seen_probe_keys:
                continue  # strict dedup by fact key
            seen_probe_keys.add(probe_key)
            viable_prompts = []
            for o in nb["objects"]:
                prompt = o.get("prompt")
                entity = o.get("object") or {}
                label = entity.get("label")
                if not prompt or not label:
                    neighborhood_excluded_occurrences += 1
                    continue
                if prompt in partition_prompts:
                    neighborhood_excluded_occurrences += 1
                    continue
                viable_prompts.append((prompt, normalize_wfd_label(label)))
            if not viable_prompts:
                neighborhood_excluded_occurrences += 1
                continue
            neighborhood_candidates.append((row, nb, probe_key, viable_prompts))

    def _materialize_m1(example: dict, gen_index: int) -> dict:
        row_dict = dict(example)
        row_dict["question"] = example["gen_prompts"][gen_index]
        row_dict["prompt_kind"] = "generalization_prompt"
        row_dict["eval_family"] = "trained_fact_paraphrase"
        row_dict["gen_index"] = gen_index
        row_dict["perplexity_disabled"] = True
        return row_dict

    def _materialize_persistence(row: dict, row_index: int) -> dict:
        example = _build_example(
            row, question=row["update_prompt"], primary_answer="",
            source_row_index=row_index,
            partition=assignment[row["subject"]["id"]],
            event_tag=_event_tag(row), eval_family="persistence_probe",
            prompt_kind="update_prompt", perplexity_disabled=True)
        example["valid_answers"] = []
        example["obsolete_answers"] = sorted(_answer_sets(row)[2])
        return example

    def _materialize_neighborhood(
        row: dict, nb: dict, probe_key: tuple[str, str],
        viable_prompts: list[tuple[str, str]],
    ) -> list[dict]:
        rows_out = []
        for prompt, answer in viable_prompts:
            rows_out.append(
                {
                    "question": prompt,
                    "object": answer,
                    "valid_answers": [answer],
                    "new_answers": [],
                    "static_answers": [answer],
                    "obsolete_answers": [],
                    "decision_set": ["static"],
                    "fact_key": f"{probe_key[0]}|{probe_key[1]}",
                    "subject_id": probe_key[0],
                    "relation_id": probe_key[1],
                    "partition": "probe",
                    "event_tag": "neighborhood",
                    "eval_family": "neighborhood_probe",
                    "prompt_kind": "neighbor_prompt",
                    "perplexity_disabled": True,
                    "neighbor_of": f"{row['subject']['id']}|{row['relation']['id']}",
                    "neighbor_distance": float(nb.get("dist", 0.0)),
                }
            )
        return rows_out

    sweep_populations: dict[str, list[dict]] = {}
    sweep_selection: dict[str, dict] = {}

    def _select_and_materialize(
        family: str, pool_size: int,
        materialize: Callable[..., list[dict] | dict],
        candidate_count: int,
    ) -> list[dict]:
        limit = min(int(sweep_limits[family]), candidate_count)
        selected_indices, _ = deterministic_subset(
            list(range(candidate_count)), limit, seed)
        selected = []
        for index in selected_indices:
            item = materialize(index)
            if isinstance(item, list):
                selected.extend(item)
            else:
                selected.append(item)
        sweep_selection[family] = {
            "pool_size": candidate_count,
            "selected_size": len(selected),
            "method": "deterministic_subset(seed) over candidate positions",
            "seed": seed,
            "selected_indices": selected_indices,
        }
        return selected

    sweep_populations["m1_trained_fact_paraphrases"] = _select_and_materialize(
        "m1", len(m1_candidates),
        lambda i: _materialize_m1(*m1_candidates[i]), len(m1_candidates))
    sweep_populations["m3_persistence_probes"] = _select_and_materialize(
        "persistence", len(persistence_candidates),
        lambda i: _materialize_persistence(*persistence_candidates[i]),
        len(persistence_candidates))
    sweep_populations["m3_neighborhood_probes"] = _select_and_materialize(
        "neighborhood", len(neighborhood_candidates),
        lambda i: _materialize_neighborhood(*neighborhood_candidates[i]),
        len(neighborhood_candidates))

    # align sweep_selection keys with the full family names used by
    # sweep_populations and the pipeline
    sweep_selection = {
        ("m1_trained_fact_paraphrases" if k == "m1" else
         "m3_persistence_probes" if k == "persistence" else
         "m3_neighborhood_probes"): v
        for k, v in sweep_selection.items()
    }
    timer.mark("sweep_materialization",
               objects=sum(len(v) for v in sweep_populations.values()))

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG_NAME,
        "dataset_revision": DATASET_REVISION,
        "adapter": "wikifactdiff",
        "partition_algorithm": PARTITION_ALGORITHM,
        "partition_salt": salt,
        "split_fractions": list(fractions),
        "seed": seed,
        "train_limit": train_limit,
        "eval_limit": None,
        "eval_population": eval_population,
        "mode": mode,
        "pilot_eval_per_event": pilot_eval_per_event
        if eval_population == "pilot" else None,
        "question_column": QUESTION_COLUMN,
        "answer_column": ANSWER_COLUMN,
        "label_normalization_version": LABEL_NORMALIZATION_VERSION,
        "answer_match_normalization_version": ANSWER_MATCH_NORMALIZATION_VERSION,
        "effective_adapter_config": {
            "eval_population": eval_population,
            "pilot_eval_per_event": pilot_eval_per_event
            if eval_population == "pilot" else None,
            "salt": salt,
            "split_fractions": list(fractions),
            "sweep_limits": sweep_limits,
        },
        "events": manifest_events,
        "counts": {
            "original_rows": len(rows),
            "unique_facts": len({_fact_key(r) for r in rows}),
            "expanded_examples": expanded_total,
            "deduplicated_examples": expanded_total - dedup_removed,
            "dedup_removed": dedup_removed,
            "excluded_degenerate_rows": len(degenerate),
            "obsolete_only_persistence_probes": len(persistence_fact_keys),
            "answer_decisions_total": dict(decision_counts_total),
            "partition_subjects": dict(partition_subjects),
            "partition_facts": dict(partition_facts),
            "mask_exclusions": len(mask_exclusions),
            "val_facts_dropped_all_prompts_masked": val_facts_dropped,
            "test_facts_dropped_all_prompts_masked": test_facts_dropped,
            "neighborhood_total_occurrences": neighborhood_occurrences,
            "neighborhood_unique_keys": len(neighborhood_unique_keys),
            "neighborhood_excluded_occurrences": neighborhood_excluded_occurrences,
            "neighborhood_probes": len(sweep_populations["m3_neighborhood_probes"]),
            "neighborhood_probe_candidates": len(neighborhood_candidates),
        },
        "excluded_records": {
            "degenerate": degenerate,
            "prompt_mask": mask_exclusions,
            "note": (
                "obsolete-only rows are not excluded from the dataset; they "
                "are reclassified as persistence probes (see populations)"
            ),
        },
        "train_provenance": train_provenance,
        "performance": timer.phases,
        "sweep_populations": sweep_populations,
        "sweep_selection": sweep_selection,
        "sweep_rows_sample": {
            family: rows[:200]
            for family, rows in sweep_populations.items()
        },
        "notes": [
            "Strict train/val/test separation: one partition per subject; "
            "training prompts (update or generalization) never appear in "
            "validation or test; val/test collisions resolve in favour of "
            "test (dropped from val).",
            "Perplexity is disabled for this dataset: a defensible "
            "multi-reference likelihood is not implemented; no empty or "
            "obsolete completion enters the loss. TemporalWiki perplexity "
            "is unchanged.",
            "M1 (update efficacy) applies ONLY to facts whose update "
            "prompt was trained; M2 measures held-out transfer; M3 "
            "measures retention and locality.",
            "subject_popularity is constant per subject and is never used "
            "to rank candidate objects within a row.",
        ],
    }
    return events, manifest


def prepare_wikifactdiff_events(
    *,
    train_limit: int | None = None,
    seed: int = 42,
    mode: str | None = None,
    dataset_config: dict[str, Any] | None = None,
) -> tuple[list, dict]:
    """HF-backed entry point (dispatched from ``cycleft.e1.data``)."""
    from datasets import load_dataset

    config = dict(dataset_config or {})
    hf_revision = str(config.pop("revision", DATASET_REVISION))

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG_NAME,
        split="train",
        revision=hf_revision,
    )
    rows = [dict(r) for r in dataset]
    events, manifest = build_wikifactdiff_events(
        rows,
        seed=seed,
        train_limit=train_limit,
        mode=mode,
        adapter_config=config,
    )
    manifest["dataset_fingerprint"] = {
        "hf_fingerprint": getattr(dataset, "_fingerprint", None),
        "revision": hf_revision,
    }
    return events, manifest
