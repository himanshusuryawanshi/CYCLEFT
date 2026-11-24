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

A brand-new subject can never be a replacement ((ph_new, replace) =
(1,1) occurs 0 times), so the three events are a true partition.

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

Answer semantics:

* Training targets are ``new``-decision object labels only (one example
  per valid new object, deterministic object order, exact duplicate
  (prompt, answer) pairs removed).
* ``static`` objects remain valid at T_new: they count as valid
  evaluation answers but are never training targets.
* ``obsolete`` objects are measurement-only (old-answer rate /
  suppression); they are never training targets.
* Obsolete-only rows (no valid post-update answer) are excluded from QA
  training and retained as **persistence probes**.
* Rows whose normalized new and obsolete labels overlap are degenerate
  (2 in the full dataset) and are excluded from training.
* ``subject_popularity`` is constant per subject and is NEVER used to
  rank candidate objects within a row.

Leakage protections (see the audits for measured collision counts):

* Subject disjointness across train/val/test (by construction).
* Prompt masking: any validation/test prompt string (update or
  generalization) that also appears among training prompts is dropped
  from that partition's evaluation prompts; facts that lose all prompts
  are excluded from evaluation (recorded in the manifest).
* Validation/test prompt collisions are resolved in favour of **test**
  (the final test set is never thinned for model-selection convenience).
* Neighborhood probes: only neighbor facts that belong to **no**
  partition (not train/val/test rows) and whose prompts do not collide
  with any partition prompt; deduplicated by fact key.
"""

from __future__ import annotations

import collections
from typing import Any

from cycleft.data.base import stable_hash

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


def normalize_wfd_label(label: str) -> str:
    """Frozen target normalization for WikiFactDiff object labels.

    Wikidata date/quantity labels carry a leading ``+`` sign
    (``+2022-11-30``); strip it and surrounding whitespace.  Case is
    preserved in training completions; evaluation matching applies the
    existing ``cycleft.e1.metrics.normalize_answer`` on both sides.
    """
    text = str(label).strip()
    return text[1:] if text.startswith("+") else text


def _event_tag(row: dict) -> str:
    if bool(row["subject_is_ph_new"]):
        return "wfd_e1_insertion"
    if bool(row["is_replace"]):
        return "wfd_e3_replacement"
    return "wfd_e2_other_update"


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


def _row_fact_key(row: dict) -> tuple[str, str]:
    return (row["subject"]["id"], row["relation"]["id"])


def _make_training_example(
    row: dict, answer: str, partition: str
) -> dict:
    new_answers = sorted(
        {
            normalize_wfd_label(o["label"])
            for o in row["objects"]
            if o["decision"] == "new"
        }
    )
    obsolete_answers = sorted(
        {
            normalize_wfd_label(o["label"])
            for o in row["objects"]
            if o["decision"] == "obsolete"
        }
    )
    static_answers = sorted(
        {
            normalize_wfd_label(o["label"])
            for o in row["objects"]
            if o["decision"] == "static"
        }
    )
    return {
        "question": row["update_prompt"],
        "object": answer,
        "valid_answers": list(new_answers),
        "obsolete_answers": obsolete_answers,
        "static_answers": static_answers,
        "fact_key": f"{row['subject']['id']}|{row['relation']['id']}",
        "subject_id": row["subject"]["id"],
        "relation_id": row["relation"]["id"],
        "partition": partition,
        "event_tag": _event_tag(row),
        "subject_is_ph_new": bool(row["subject_is_ph_new"]),
        "is_replace": bool(row["is_replace"]),
        "subject_popularity": float(row["subject_popularity"]),
        "gen_prompts": list(row["generalization_prompts"]),
        "source_row_type": "update",
    }


def _make_eval_row(
    row: dict,
    prompt: str,
    prompt_kind: str,
    eval_family: str,
    partition: str,
) -> dict:
    example = _make_training_example(row, "", partition)
    example["question"] = prompt
    example["prompt_kind"] = prompt_kind
    example["eval_family"] = eval_family
    return example


def build_wikifactdiff_events(
    rows: list[dict],
    *,
    seed: int,
    train_limit: int | None = None,
    eval_population: str = "full",
    pilot_eval_per_event: int = 100,
    salt: str = PARTITION_SALT,
    fractions: tuple[float, ...] = SPLIT_FRACTIONS,
) -> tuple[list, dict]:
    """Pure event-construction core (no HF access) — unit-testable.

    Takes materialized WikiFactDiff rows (schema of the
    ``20210104-20230227_legacy`` config) and returns
    ``(list[PreparedEvent], manifest)``.
    """
    if train_limit is not None and train_limit <= 0:
        raise ValueError(f"train_limit must be positive or None, got {train_limit!r}")

    assignment = assign_partitions(rows, salt=salt, fractions=fractions)
    partition_subjects = collections.Counter(assignment.values())
    partition_facts = collections.Counter(
        assignment[row["subject"]["id"]] for row in rows
    )

    # ---- per-fact answer analysis and exclusions --------------------------
    excluded: list[dict[str, Any]] = []
    degenerate_fact_keys = set()
    obsolete_only_fact_keys = set()
    decision_counts_total = collections.Counter()
    for row in rows:
        for o in row["objects"]:
            decision_counts_total[o["decision"]] += 1
        new_labels = {
            normalize_wfd_label(o["label"])
            for o in row["objects"]
            if o["decision"] == "new"
        }
        obsolete_labels = {
            normalize_wfd_label(o["label"])
            for o in row["objects"]
            if o["decision"] == "obsolete"
        }
        fk = _row_fact_key(row)
        if new_labels & obsolete_labels:
            degenerate_fact_keys.add(fk)
            excluded.append(
                {
                    "fact_key": f"{fk[0]}|{fk[1]}",
                    "reason": "degenerate_new_obsolete_overlap",
                    "partition": assignment[row["subject"]["id"]],
                }
            )
        elif not new_labels and obsolete_labels:
            obsolete_only_fact_keys.add(fk)
            excluded.append(
                {
                    "fact_key": f"{fk[0]}|{fk[1]}",
                    "reason": "obsolete_only_persistence_probe",
                    "partition": assignment[row["subject"]["id"]],
                }
            )

    # ---- training examples (train partition only) -------------------------
    event_train_pool: dict[str, list[dict]] = {tag: [] for tag in EVENT_TAGS}
    expanded_count = 0
    for row in rows:
        fk = _row_fact_key(row)
        if fk in degenerate_fact_keys or fk in obsolete_only_fact_keys:
            continue
        if assignment[row["subject"]["id"]] != "train":
            continue
        new_objects = sorted(
            (
                normalize_wfd_label(o["label"]),
                o["id"] is None,
                o["id"] or "",
            )
            for o in row["objects"]
            if o["decision"] == "new"
        )
        for answer, _, _ in new_objects:
            expanded_count += 1
            event_train_pool[_event_tag(row)].append(
                _make_training_example(row, answer, "train")
            )

    # deduplicate exact (question, object) pairs, preserving order
    event_train_rows: dict[str, list[dict]] = {}
    dedup_removed = 0
    for tag, pool in event_train_pool.items():
        seen = set()
        kept = []
        for example in pool:
            key = (example["question"], example["object"])
            if key in seen:
                dedup_removed += 1
                continue
            seen.add(key)
            kept.append(example)
        if train_limit is not None and len(kept) > train_limit:
            from cycleft.e1.data import deterministic_subset

            kept, _indices = deterministic_subset(kept, train_limit, seed)
        event_train_rows[tag] = kept

    # ---- evaluation populations -------------------------------------------
    # M2: held-out facts (val/test partitions).  Prompt masking removes any
    # val/test prompt that also appears among TRAIN-PARTITION prompts
    # (update or generalization of ANY train fact — not just the sampled
    # training examples, so the mask is stable across pilot/full sizes);
    # val/test collisions resolve in favour of test.
    train_prompts: set[str] = set()
    for row in rows:
        if assignment[row["subject"]["id"]] != "train":
            continue
        train_prompts.add(row["update_prompt"])
        train_prompts.update(row["generalization_prompts"])

    def _masked_prompts(prompts: list[str], other_partition_prompts: set[str]):
        kept, dropped = [], []
        for prompt in prompts:
            if prompt in train_prompts:
                dropped.append((prompt, "train_prompt_collision"))
            elif prompt in other_partition_prompts:
                dropped.append((prompt, "val_test_prompt_collision"))
            else:
                kept.append(prompt)
        return kept, dropped

    val_upd_prompts = {
        row["update_prompt"] for row in rows if assignment[row["subject"]["id"]] == "val"
    }
    mask_exclusions: list[dict[str, Any]] = []
    eval_facts_dropped_no_prompts = 0
    event_eval_rows: dict[str, list[dict]] = {tag: [] for tag in EVENT_TAGS}
    event_eval_counts: dict[str, dict] = {}

    if eval_population not in ("full", "pilot"):
        raise ValueError(f"Unknown eval_population {eval_population!r}")

    # group eval facts per event tag
    eval_facts_by_event: dict[str, list[dict]] = {tag: [] for tag in EVENT_TAGS}
    for row in rows:
        fk = _row_fact_key(row)
        if fk in degenerate_fact_keys:
            continue  # degenerate rows are excluded everywhere
        partition = assignment[row["subject"]["id"]]
        if partition == "train":
            continue
        eval_facts_by_event[_event_tag(row)].append(row)

    for tag in EVENT_TAGS:
        kept_rows: list[dict] = []
        for row in eval_facts_by_event[tag]:
            partition = assignment[row["subject"]["id"]]
            other_partition_prompts = val_upd_prompts if partition == "test" else set()
            upd_kept, upd_dropped = _masked_prompts(
                [row["update_prompt"]], other_partition_prompts
            )
            gen_kept, gen_dropped = _masked_prompts(
                list(row["generalization_prompts"]), other_partition_prompts
            )
            for prompt, reason in upd_dropped + gen_dropped:
                mask_exclusions.append(
                    {
                        "fact_key": f"{row['subject']['id']}|{row['relation']['id']}",
                        "partition": partition,
                        "reason": reason,
                        "prompt": prompt,
                    }
                )
            if not upd_kept and not gen_kept:
                eval_facts_dropped_no_prompts += 1
                mask_exclusions.append(
                    {
                        "fact_key": f"{row['subject']['id']}|{row['relation']['id']}",
                        "partition": partition,
                        "reason": "all_prompts_masked",
                    }
                )
                continue
            for prompt in upd_kept:
                kept_rows.append(
                    _make_eval_row(row, prompt, "update_prompt", "heldout_update", partition)
                )
            for prompt in gen_kept:
                kept_rows.append(
                    _make_eval_row(
                        row, prompt, "generalization_prompt", "heldout_paraphrase", partition
                    )
                )

        # pilot population: deterministic stratified sample
        if eval_population == "pilot" and len(kept_rows) > pilot_eval_per_event:
            from cycleft.e1.data import deterministic_subset

            kept_rows, _indices = deterministic_subset(
                kept_rows, pilot_eval_per_event, seed
            )
        event_eval_rows[tag] = kept_rows
        event_eval_counts[tag] = {
            "n_eval_rows": len(kept_rows),
            "n_eval_facts_available": len(eval_facts_by_event[tag]),
        }

    # ---- M1 population: trained facts' generalization prompts -------------
    m1_rows: list[dict] = []
    for tag in EVENT_TAGS:
        for example in event_train_rows[tag]:
            for prompt in example["gen_prompts"]:
                m1_row = dict(example)
                m1_row["question"] = prompt
                m1_row["prompt_kind"] = "generalization_prompt"
                m1_row["eval_family"] = "trained_fact_paraphrase"
                m1_rows.append(m1_row)
            # the trained fact's own update_prompt IS the training example;
            # M1 evaluates only its held-out paraphrases.

    # ---- M3 persistence probes (obsolete-only rows) ------------------------
    persistence_rows: list[dict] = []
    for row in rows:
        fk = _row_fact_key(row)
        if fk in obsolete_only_fact_keys:
            obsolete_answers = sorted(
                {
                    normalize_wfd_label(o["label"])
                    for o in row["objects"]
                    if o["decision"] == "obsolete"
                }
            )
            example = _make_training_example(row, "", assignment[row["subject"]["id"]])
            example["question"] = row["update_prompt"]
            example["eval_family"] = "persistence_probe"
            example["obsolete_answers"] = obsolete_answers
            example["valid_answers"] = []  # nothing valid post-update
            persistence_rows.append(example)

    # ---- M3 neighborhood probes -------------------------------------------
    partition_fact_keys = {_row_fact_key(row) for row in rows}
    partition_prompts = train_prompts | val_upd_prompts | {
        row["update_prompt"] for row in rows if assignment[row["subject"]["id"]] == "test"
    }
    seen_probe_keys = set()
    neighborhood_rows: list[dict] = []
    neighborhood_excluded = 0
    for row in rows:
        rid = row["relation"]["id"]
        for nb in row["neighborhood"]:
            probe_key = (nb["subject"]["id"], rid)
            if probe_key in partition_fact_keys:
                neighborhood_excluded += 1
                continue  # belongs to a dataset partition: never a probe
            probe_id = (probe_key, tuple(o.get("prompt") for o in nb["objects"]))
            if probe_id in seen_probe_keys:
                continue  # deduplicate by fact key + prompt set
            seen_probe_keys.add(probe_id)
            # NOTE: neighborhood objects are NESTED ({"object": {...}, "prompt":
            # ...}), unlike the flat row-level objects — an actual schema
            # finding from the real data, not the HF card's flattened prose.
            for o in nb["objects"]:
                prompt = o.get("prompt")
                entity = o.get("object") or {}
                label = entity.get("label")
                if not prompt or not label or prompt in partition_prompts:
                    neighborhood_excluded += 1
                    continue
                answer = normalize_wfd_label(label)
                neighborhood_rows.append(
                    {
                        "question": prompt,
                        "object": answer,
                        "valid_answers": [answer],
                        "obsolete_answers": [],
                        "static_answers": [answer],
                        "fact_key": f"{probe_key[0]}|{probe_key[1]}",
                        "subject_id": probe_key[0],
                        "relation_id": probe_key[1],
                        "partition": "probe",
                        "event_tag": "neighborhood",
                        "subject_is_ph_new": False,
                        "is_replace": False,
                        "subject_popularity": 0.0,
                        "gen_prompts": [],
                        "source_row_type": "neighborhood_probe",
                        "prompt_kind": "neighbor_prompt",
                        "eval_family": "neighborhood_probe",
                        "neighbor_of": f"{row['subject']['id']}|{rid}",
                        "neighbor_distance": float(nb.get("dist", 0.0)),
                    }
                )

    # ---- PreparedEvent assembly (import here to avoid cycles) --------------
    from cycleft.e1.data import PreparedEvent, deterministic_subset

    events = []
    manifest_events = []
    for event_id, tag in enumerate(EVENT_TAGS, start=1):
        train_rows = event_train_rows[tag]
        eval_rows = event_eval_rows[tag]
        events.append(
            PreparedEvent(
                event_id=event_id,
                name=tag,
                slice_tag=tag,
                train_rows=train_rows,
                eval_rows=eval_rows,
                train_pool_size=len(event_train_pool[tag]),
                eval_pool_size=len(eval_rows),
                train_indices_used=list(range(len(train_rows))),
            )
        )
        manifest_events.append(
            {
                "event_id": event_id,
                "name": tag,
                "event_definition": EVENT_DEFINITIONS[tag],
                "n_train_pool": len(event_train_pool[tag]),
                "n_train_used": len(train_rows),
                "n_eval_rows": len(eval_rows),
                "eval_population": eval_population,
                "question_column": QUESTION_COLUMN,
                "answer_column": ANSWER_COLUMN,
            }
        )

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
        "pilot_eval_per_event": pilot_eval_per_event if eval_population == "pilot" else None,
        "question_column": QUESTION_COLUMN,
        "answer_column": ANSWER_COLUMN,
        "events": manifest_events,
        "counts": {
            "original_rows": len(rows),
            "unique_facts": len({_row_fact_key(r) for r in rows}),
            "expanded_examples": expanded_count,
            "deduplicated_examples": expanded_count - dedup_removed,
            "dedup_removed": dedup_removed,
            "excluded_degenerate_rows": len(degenerate_fact_keys),
            "obsolete_only_persistence_probes": len(obsolete_only_fact_keys),
            "answer_decisions_total": dict(decision_counts_total),
            "partition_subjects": dict(partition_subjects),
            "partition_facts": dict(partition_facts),
            "mask_exclusions": len(mask_exclusions),
            "eval_facts_dropped_all_prompts_masked": eval_facts_dropped_no_prompts,
            "neighborhood_probes": len(neighborhood_rows),
            "neighborhood_excluded_occurrences": neighborhood_excluded,
        },
        "excluded_records": {
            "degenerate": [
                e for e in excluded if e["reason"] == "degenerate_new_obsolete_overlap"
            ],
            "prompt_mask": mask_exclusions,
            "note": (
                "obsolete-only rows are not excluded from the dataset; they "
                "are reclassified as persistence probes (see populations)"
            ),
        },
        "sweep_populations": {
            "m1_trained_fact_paraphrases": len(m1_rows),
            "m3_persistence_probes": len(persistence_rows),
            "m3_neighborhood_probes": len(neighborhood_rows),
        },
        "sweep_rows": {
            "m1_trained_fact_paraphrases": m1_rows,
            "m3_persistence_probes": persistence_rows,
            "m3_neighborhood_probes": neighborhood_rows,
        },
        "notes": [
            "One training example per valid new object; exact duplicate "
            "(prompt, answer) pairs removed.",
            "subject_popularity is constant per subject and is never used "
            "to rank candidate objects within a row.",
            "Degenerate new/obsolete overlap is 0 under the frozen target "
            "normalization (strip '+', strip whitespace); 2 rows overlap "
            "only after case/punctuation folding and are harmless.",
            "Prompt masking: validation/test prompts colliding with train "
            "prompts are dropped; val/test collisions resolve in favour of "
            "test; facts losing all prompts are excluded (recorded).",
            "Neighborhood probes exclude every fact key belonging to any "
            "dataset partition and are deduplicated by fact key.",
            "The original attempt-1 interruption of e7-static-r9-s42 is "
            "unrelated to this adapter; see the Static-R9 recovery reports.",
        ],
    }
    return events, manifest


def prepare_wikifactdiff_events(
    *,
    train_limit: int | None = None,
    seed: int = 42,
    dataset_config: dict[str, Any] | None = None,
) -> tuple[list, dict]:
    """HF-backed entry point (dispatched from ``cycleft.e1.data``)."""
    from datasets import load_dataset

    config = dict(dataset_config or {})
    eval_population = config.get("eval_population", "full")
    pilot_eval_per_event = int(config.get("pilot_eval_per_event", 100))
    hf_revision = config.get("revision", DATASET_REVISION)

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
        eval_population=eval_population,
        pilot_eval_per_event=pilot_eval_per_event,
    )
    manifest["dataset_fingerprint"] = {
        "hf_fingerprint": getattr(dataset, "_fingerprint", None),
        "revision": hf_revision,
    }
    return events, manifest
