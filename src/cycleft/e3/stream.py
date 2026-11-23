"""E3 stream construction and temporal dataset audit.

Stream semantics (all documented in ``results/e3/<run_id>/dataset_audit.md``):

* The stream is the concatenation of the three events' 500-row training
  pools, each in the seed-42 shuffled order already used by E1/E2
  (``prepare_event_datasets``).  The dataset provides **no per-example
  timestamps** — temporal information exists only at the event level
  (``snapshot_old``/``snapshot_new``/``slice_tag``) — so the within-event
  ordering is a protocol choice, not verified chronology.  Using the E1/E2
  order keeps E3 anchored to the prior experiments' training-order
  semantics and makes the ``always_adapt`` reference policy identical to
  E2's ``rank4_b125`` protocol.
* Known ground-truth drift boundaries are therefore exactly TWO:
  stream position 500 (event 1 -> 2) and 1000 (event 2 -> 3).  No
  additional drift points are invented.
* The stream is chunked into fixed windows (default 50 examples, the size
  of one event's held-out test set) giving 30 windows; boundaries fall at
  windows 10 and 20.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from cycleft.e1.data import PreparedEvent, prepare_event_datasets

WINDOW_SIZE = 50
EXAMPLES_PER_EVENT = 500
N_EVENTS = 3


@dataclass(frozen=True)
class E3Stream:
    """The chronological monitoring stream plus ground-truth boundaries."""

    rows: list[dict] = field(repr=False)
    event_ids: list[int]
    window_size: int
    seed: int

    @property
    def n_examples(self) -> int:
        return len(self.rows)

    @property
    def n_windows(self) -> int:
        return self.n_examples // self.window_size

    @property
    def boundaries(self) -> list[int]:
        """Stream positions where a new event starts (excluding position 0)."""
        return [
            event * EXAMPLES_PER_EVENT for event in range(1, N_EVENTS)
        ]

    @property
    def boundary_windows(self) -> list[int]:
        """Window index that CONTAINS each boundary (first window of the new event)."""
        return [position // self.window_size for position in self.boundaries]

    def window(self, index: int) -> list[int]:
        """Row positions belonging to window ``index``."""
        start = index * self.window_size
        return list(range(start, min(start + self.window_size, self.n_examples)))

    def event_of_position(self, position: int) -> int:
        return self.event_ids[position]


def build_stream(
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
) -> tuple[E3Stream, list[PreparedEvent], dict[str, Any]]:
    """Build the E3 stream from the shared E1/E2 prepared events.

    The prepared events also carry the held-out evaluation sets, which the
    runner uses for boundary evaluations; they are never part of the stream.
    """
    events, manifest = prepare_event_datasets(train_limit=None, seed=seed)
    rows: list[dict] = []
    event_ids: list[int] = []
    for event in events:
        for row in event.train_rows:
            rows.append(dict(row))
            event_ids.append(event.event_id)
    stream = E3Stream(
        rows=rows, event_ids=event_ids, window_size=window_size, seed=seed
    )
    return stream, events, manifest


def dataset_audit(
    stream: E3Stream,
    events: list[PreparedEvent],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Structured facts for ``dataset_audit.md`` (measured, never assumed)."""
    per_event = []
    for event in events:
        train_questions = [str(row["question"]) for row in event.train_rows]
        eval_questions = {str(row["question"]) for row in event.eval_rows}
        per_event.append(
            {
                "event_id": event.event_id,
                "slice_tag": event.slice_tag,
                "n_train_stream": len(event.train_rows),
                "n_eval": len(event.eval_rows),
                "unique_subjects_stream": len(
                    {row["subject"] for row in event.train_rows}
                ),
                "eval_questions_in_stream": len(
                    eval_questions.intersection(train_questions)
                ),
            }
        )

    # Cross-event question/answer/fact overlap on the stream side.
    cross: list[dict] = []
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            qa = {str(r["question"]) for r in a.train_rows}
            qb = {str(r["question"]) for r in b.train_rows}
            aa = {str(r["object"]) for r in a.train_rows}
            ab = {str(r["object"]) for r in b.train_rows}
            fa = {(r["subject"], r["relation"]) for r in a.train_rows}
            fb = {(r["subject"], r["relation"]) for r in b.train_rows}
            fa_map = {(r["subject"], r["relation"]): r["object"] for r in a.train_rows}
            fb_map = {(r["subject"], r["relation"]): r["object"] for r in b.train_rows}
            shared = fa & fb
            changed = sum(1 for f in shared if fa_map[f] != fb_map[f])
            cross.append(
                {
                    "event_a": a.event_id,
                    "event_b": b.event_id,
                    "question_overlap": len(qa & qb),
                    "answer_string_overlap": len(aa & ab),
                    "shared_subject_relation_facts": len(shared),
                    "shared_facts_with_changed_answer": changed,
                }
            )

    return {
        "dataset_id": manifest["dataset_id"],
        "slices": manifest["slices"],
        "n_stream_examples": stream.n_examples,
        "n_windows": stream.n_windows,
        "window_size": stream.window_size,
        "stream_order": (
            "per-event seed-42 shuffled training order (the E1/E2 order); "
            "within-event chronology is NOT verifiable because the dataset "
            "has no per-example timestamps"
        ),
        "known_boundaries": stream.boundaries,
        "known_boundary_windows": stream.boundary_windows,
        "temporal_columns": {
            "event_level_only": ["slice_tag", "slice", "snapshot_old", "snapshot_new"],
            "per_example_timestamps": None,
        },
        "detector_usable_columns": ["question", "subject", "relation", "article"],
        "answer_column": "object",
        "per_event": per_event,
        "cross_event_overlap": cross,
        "limitations": [
            "Only TWO genuine known drift boundaries exist (stream positions "
            "500 and 1000); no additional drift points are invented.",
            "Within-event windows are used to measure false alarms under an "
            "EXPLICIT stationarity assumption; the dataset does not verify "
            "that within-event windows are stationary.",
            "Question text shifts moderately across events (333/500 shared "
            "subjects per consecutive pair, all with changed answers), so "
            "input-space drift detection is an empirical question, not a "
            "guarantee.",
        ],
    }


def load_slice_order_check(stream: E3Stream) -> dict[str, Any]:
    """Verify the loader preserves the dataset's chronological grouping.

    The raw HF rows arrive as three contiguous chronological blocks; the
    E3 stream keeps events in that order (event 1 -> 2 -> 3).
    """
    seen: list[int] = []
    for event_id in stream.event_ids:
        if not seen or seen[-1] != event_id:
            seen.append(event_id)
    return {
        "event_blocks_in_stream_order": seen,
        "contiguous": seen == sorted(set(seen)),
        "matches_chronological_order": seen == [1, 2, 3],
    }


def summarise_audit(audit: dict[str, Any], order_check: dict[str, Any]) -> str:
    """Render ``dataset_audit.md``."""
    lines = [
        "# E3 Temporal Data Audit",
        "",
        f"Dataset: `{audit['dataset_id']}` — slices `{audit['slices']}`.",
        "",
        "## Temporal boundaries",
        "",
        f"- Known boundaries (stream positions): **{audit['known_boundaries']}** — "
        f"exactly **two** genuine transitions (event 1→2 and 2→3).",
        f"- Corresponding first windows of the new event: {audit['known_boundary_windows']}.",
        f"- Stream: {audit['n_stream_examples']} examples in {audit['n_windows']} "
        f"windows of {audit['window_size']}.",
        "",
        "## Timestamps and ordering",
        "",
        "- Temporal metadata is **event-level only**: `slice_tag`, `slice`, "
        "`snapshot_old`, `snapshot_new` (one (old, new) date pair per slice).",
        "- **No per-example timestamps exist.** Sub-windows within an event "
        "cannot be verified as chronological.",
        f"- Raw dataset rows arrive as three contiguous chronological blocks "
        f"(500/500/500 train, 50/50/50 test); the loader preserves this "
        f"grouping: blocks={order_check['event_blocks_in_stream_order']}, "
        f"contiguous={order_check['contiguous']}, "
        f"chronological={order_check['matches_chronological_order']}.",
        f"- Stream within-event order: {audit['stream_order']}.",
        "",
        "## Columns available to detectors",
        "",
        f"- Genuinely observable inputs: `{', '.join(audit['detector_usable_columns'])}`.",
        f"- Target/answer column: `{audit['answer_column']}` — **never** "
        f"accessible to Regime-A (unlabeled) detectors.",
        "- `qa_style`: train rows use `direct`/`train_paraphrase`; the held-out "
        "test rows use `paraphrase` (evaluation only, never in the stream).",
        "",
        "## One transition per event boundary",
        "",
        "- Each event carries exactly ONE (snapshot_old, snapshot_new) pair, "
        "i.e. each event = one temporal transition; there are no multiple "
        "transitions inside an event.",
        "",
        "## Overlap across events (stream rows, measured)",
        "",
    ]
    for entry in audit["cross_event_overlap"]:
        lines.append(
            f"- Event {entry['event_a']} vs {entry['event_b']}: "
            f"question overlap {entry['question_overlap']}, answer-string "
            f"overlap {entry['answer_string_overlap']}, shared "
            f"(subject, relation) facts "
            f"{entry['shared_subject_relation_facts']}/500, of which "
            f"**{entry['shared_facts_with_changed_answer']} changed answer** "
            "(the drift the benchmark tests)."
        )
    lines += [
        "",
        "## Per-event stream/eval composition",
        "",
    ]
    for entry in audit["per_event"]:
        lines.append(
            f"- Event {entry['event_id']} (`{entry['slice_tag']}`): "
            f"{entry['n_train_stream']} stream rows, "
            f"{entry['unique_subjects_stream']} unique subjects, "
            f"{entry['n_eval']} held-out eval rows "
            f"({entry['eval_questions_in_stream']} duplicated questions — "
            "known from the E1 audit, EM 0/13 on them in E1)."
        )
    lines += [
        "",
        "## Limitations (prominent)",
        "",
    ]
    for limitation in audit["limitations"]:
        lines.append(f"- **{limitation}**")
    lines += [
        "",
        "Consequence for E3: detection metrics are computed against the TWO "
        "known boundaries only. False alarms inside events are reported as "
        "'false positives under the stationarity assumption', not as verified "
        "errors, and detection-recall comparisons across detectors rest on "
        "only two positive cases — no claim of statistical superiority is "
        "made.",
    ]
    return "\n".join(lines) + "\n"
