"""E5 position audit validator (Phase 2).

Parses step events from the (multi-process) event log(s) and produces the
corrected accounting fields — replacing the Phase 1 mislabeled fields:

- pre_kill_entries / post_resume_entries / reexecuted_entries
- logical_positions_completed (target range)
- physical_step_attempts (total step events)
- unique_logical_positions_completed
- missing_logical_positions
- replay classification: valid replay (attempts separated by a
  checkpoint_restore/interruption boundary) vs false duplicate-completion
  (two completions within the same process attempt — a progress bug)

Composite correctness assertion replaces the Phase 1 single-segment check.
"""

from __future__ import annotations

from typing import Any


def validate_positions(
    step_events: list[dict],
    target_positions: int,
    interruption_boundaries: list[int] | None = None,
) -> dict[str, Any]:
    """Validate logical-position coverage from step events.

    step_events: dicts with at least {"position", "attempt", "process_exit_boundary"?}.
                 Events are in chronological order. `attempt` is the physical
                 process attempt number (1..N). An interruption boundary is
                 the attempt number after which the process was replaced
                 (i.e. attempts >= boundary belong to the post-resume process).
    target_positions: the logical position range is 1..target_positions.
    interruption_boundaries: attempt numbers at which a restoration occurred
                 (used to classify repeated positions as valid replay).
    """
    boundaries = set(interruption_boundaries or [])
    attempts_by_position: dict[int, list[dict]] = {}
    for index, event in enumerate(step_events):
        position = int(event["position"])
        attempts_by_position.setdefault(position, []).append(
            {
                "attempt": int(event.get("attempt", 1)),
                "order": index,
                "boundary_before": event.get("boundary_before", False),
            }
        )

    physical_step_attempts = len(step_events)
    completed_positions = sorted(attempts_by_position)
    unique_completed = len(completed_positions)
    missing = [
        p for p in range(1, target_positions + 1) if p not in attempts_by_position
    ]
    multi = {
        p: entries for p, entries in attempts_by_position.items() if len(entries) > 1
    }

    # Classify repeated positions.
    valid_replays: list[dict] = []
    false_duplicates: list[dict] = []
    for position, entries in sorted(multi.items()):
        attempt_numbers = [e["attempt"] for e in entries]
        crossed_boundary = any(a in boundaries for a in attempt_numbers)
        # Valid replay: repeated attempts span a restoration boundary (the
        # physical work was repeated because it was not durably checkpointed).
        if crossed_boundary:
            valid_replays.append(
                {"position": position, "attempts": attempt_numbers}
            )
        else:
            false_duplicates.append(
                {"position": position, "attempts": attempt_numbers}
            )

    # Final-pass contiguity: the LAST attempt of each completed position must
    # form the exact sequence 1..target (positions with multiple attempts
    # count once, at their final attempt).
    last_attempt_order = {
        p: entries[-1]["order"] for p, entries in attempts_by_position.items()
    }
    final_sequence = [
        p
        for p, _ in sorted(last_attempt_order.items(), key=lambda kv: kv[1])
    ]
    final_pass_contiguous = final_sequence == list(range(1, target_positions + 1))

    # Split entries relative to the first restoration boundary (if any).
    first_boundary_attempt = min(boundaries) if boundaries else None
    pre_kill_entries = 0
    post_resume_entries = 0
    if first_boundary_attempt is not None:
        for event in step_events:
            if int(event.get("attempt", 1)) < first_boundary_attempt:
                pre_kill_entries += 1
            else:
                post_resume_entries += 1
    else:
        pre_kill_entries = physical_step_attempts
        post_resume_entries = 0

    reexecuted_entries = len(multi)
    exactly_once = (
        not missing
        and not false_duplicates
        and final_pass_contiguous
        and unique_completed == target_positions
    )

    return {
        "target_positions": target_positions,
        "physical_step_attempts": physical_step_attempts,
        "unique_logical_positions_completed": unique_completed,
        "logical_positions_completed": target_positions - len(missing),
        "missing_logical_positions": missing,
        "pre_kill_entries": pre_kill_entries,
        "post_resume_entries": post_resume_entries,
        "reexecuted_entries": reexecuted_entries,
        "reexecuted_positions": sorted(multi),
        "valid_replays": valid_replays,
        "false_duplicate_completions": false_duplicates,
        "final_pass_contiguous": final_pass_contiguous,
        "exactly_once_composite": exactly_once,
        "note": (
            "composite assertion = no missing positions AND no false duplicate "
            "completions AND final-attempt sequence contiguous 1..target; "
            "repeated positions across a restoration boundary are classified "
            "as valid replay (physical work repeated because it was not "
            "durably checkpointed)"
        ),
    }
