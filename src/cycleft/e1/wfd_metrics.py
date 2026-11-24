"""WikiFactDiff continual-learning metrics (M1 / M2 / M3).

Pure functions over prediction records produced by
``cycleft.e1.lm.evaluate_event`` when rows carry the WikiFactDiff adapter
metadata (``valid_answers`` / ``obsolete_answers`` / ``static_answers`` /
``eval_family`` / ``fact_key``).  No model or dataset access — everything
is unit-testable.

Metric naming frozen by the final research-design validation (2026-11-23):

* **M1 — update efficacy (trained facts)**: facts whose
  ``update_prompt`` was actually trained, evaluated on their held-out
  ``generalization_prompts``.  ``update_success`` is ONLY meaningful
  here — a fact excluded from training has no update to succeed.
* **M2 — held-out new-answer transfer**: completely held-out facts
  (never any prompt trained), optionally baseline-conditioned to
  separate already-known from initially-unknown facts.
* **M3 — retention & locality**: neighborhood retention on
  exclusion-filtered unaffected probes, obsolete-answer persistence on
  obsolete-only rows, plus the existing average-forgetting and
  backward-transfer calculations in :mod:`cycleft.e1.analysis`.

Anti-gaming rule: old-answer suppression is never reported alone.  Every
rate table co-reports NSR (new-answer success), OAR (obsolete-answer
rate) and IUR (invalid/unmatched rate), and the headline
**update efficacy** is the *joint* event
``prediction ∈ new answers AND prediction ∉ obsolete answers`` —
producing garbage or an empty string cannot game it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from cycleft.e1.metrics import normalize_answer


def _matches_any(prediction: str, labels: Sequence[str] | None) -> bool:
    if not labels:
        return False
    normalized = normalize_answer(prediction)
    return any(normalized == normalize_answer(label) for label in labels)


def classify_record(record: dict) -> dict[str, bool]:
    """Classify one prediction against its own fact's candidate answers.

    ``matched_new`` / ``matched_obsolete`` / ``matched_static`` use the
    record's own candidate lists (never a global answer blacklist — the
    same surface label can be ``new`` for one fact and ``obsolete`` for
    another).  ``unmatched`` means the prediction matched nothing.
    """
    prediction = str(record.get("prediction", ""))
    matched_new = _matches_any(prediction, record.get("valid_answers"))
    matched_obsolete = _matches_any(prediction, record.get("obsolete_answers"))
    matched_static = _matches_any(prediction, record.get("static_answers"))
    return {
        "matched_new": matched_new,
        "matched_obsolete": matched_obsolete,
        "matched_static": matched_static,
        "unmatched": not (matched_new or matched_obsolete or matched_static),
    }


def answer_rates(records: Iterable[dict]) -> dict[str, float]:
    """NSR / OAR / IUR / UE over a record iterable (empty -> all 0.0)."""
    records = list(records)
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "new_answer_success_rate": 0.0,
            "obsolete_answer_rate": 0.0,
            "invalid_unmatched_rate": 0.0,
            "update_efficacy": 0.0,
        }
    classifications = [classify_record(r) for r in records]
    nsr = sum(c["matched_new"] for c in classifications) / n
    oar = sum(c["matched_obsolete"] for c in classifications) / n
    iur = sum(c["unmatched"] for c in classifications) / n
    ue = sum(
        c["matched_new"] and not c["matched_obsolete"] for c in classifications
    ) / n
    return {
        "n": n,
        "new_answer_success_rate": nsr,
        "obsolete_answer_rate": oar,
        "invalid_unmatched_rate": iur,
        "update_efficacy": ue,
    }


def _by_eval_family(
    records: Iterable[dict], family: str
) -> list[dict]:
    return [r for r in records if r.get("eval_family") == family]


def m1_update_efficacy(records: Iterable[dict]) -> dict[str, float]:
    """M1: update efficacy on TRAINED facts' held-out paraphrases.

    Only records with ``eval_family == "trained_fact_paraphrase"`` count.
    These are facts whose ``update_prompt`` received a gradient update;
    their ``generalization_prompts`` were never trained.
    """
    return answer_rates(_by_eval_family(records, "trained_fact_paraphrase"))


def m2_heldout_transfer(
    records: Iterable[dict],
    baseline_records: Iterable[dict] | None = None,
) -> dict[str, Any]:
    """M2: held-out new-answer transfer (completely held-out facts).

    With ``baseline_records`` (the same held-out prompts scored on the
    base model), results are split into *initially-unknown* facts
    (baseline EM == 0 — where transfer is meaningful) and
    *already-known* facts (baseline EM == 1 — reported separately so a
    high transfer number cannot hide facts the model already knew).
    """
    records = _by_eval_family(records, "heldout_update") + _by_eval_family(
        records, "heldout_paraphrase"
    )
    result: dict[str, Any] = {
        "heldout_overall": answer_rates(records),
    }
    if baseline_records is None:
        result["baseline_conditioned"] = None
        return result
    baseline_by_key = {
        r.get("fact_key"): r for r in baseline_records
    }
    unknown, known = [], []
    for record in records:
        baseline = baseline_by_key.get(record.get("fact_key"))
        if baseline is None:
            continue
        (known if baseline.get("exact_match", 0.0) >= 1.0 else unknown).append(
            record
        )
    result["baseline_conditioned"] = {
        "initially_unknown": answer_rates(unknown),
        "initially_known": answer_rates(known),
        "n_initially_unknown": len(unknown),
        "n_initially_known": len(known),
    }
    return result


def m3_neighborhood_retention(
    baseline_records: Iterable[dict],
    *later_records: Iterable[dict],
) -> dict[str, Any]:
    """M3: neighborhood retention — EM on unaffected probes over states.

    Probes must already be exclusion-filtered by the adapter (they belong
    to no partition).  Returns per-state mean EM plus the drift from
    baseline; negative drift means collateral damage on unchanged facts.
    """
    baseline = list(baseline_records)
    baseline_by_key = {
        r.get("fact_key"): r.get("exact_match", 0.0) for r in baseline
    }
    states = [{"state": "base", "n": len(baseline),
               "mean_em": (sum(baseline_by_key.values()) / len(baseline))
               if baseline else 0.0}]
    for state_index, records in enumerate(later_records, start=1):
        records = list(records)
        by_key = {r.get("fact_key"): r.get("exact_match", 0.0) for r in records}
        paired = [
            by_key[k] for k in baseline_by_key if k in by_key
        ]
        mean_em = (sum(paired) / len(paired)) if paired else 0.0
        states.append({
            "state": f"after_event_{state_index}",
            "n": len(paired),
            "mean_em": mean_em,
        })
    drift = states[-1]["mean_em"] - states[0]["mean_em"] if len(states) > 1 else 0.0
    return {"states": states, "retention_drift": drift}


def m3_persistence_rate(records: Iterable[dict]) -> dict[str, float]:
    """M3: obsolete-answer persistence on obsolete-only rows.

    These rows have NO valid post-update answer (``valid_answers == []``);
    a matching generation means the obsolete fact **persisted** after
    training on other events.  Deletion/abstention training is NOT
    implemented, so this is a measurement-only population.
    """
    records = _by_eval_family(records, "persistence_probe")
    n = len(records)
    persisted = sum(
        1 for r in records
        if _matches_any(str(r.get("prediction", "")), r.get("obsolete_answers"))
    )
    return {
        "n": n,
        "obsolete_answer_persistence_rate": persisted / n if n else 0.0,
    }
