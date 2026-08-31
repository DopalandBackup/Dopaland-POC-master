"""
E_t -- derived episode features (D1 feature-block separation, CLAUDE.md).
Moved verbatim out of stage1_step4_vectors.py; no formula, threshold, or
windowing logic changed (G5). See tests/test_refactor_snapshot.py for the
byte-exact regression net and tests/test_feature_separation.py for the
machine-checked import/call-graph and runtime isolation proof that this
module never reads attention.py or audio.py.

DECISION 2 (this task, restated for anyone reading this file without the
task prompt in front of them): WindowAccumulator is E_t -- E_t = h(X_core,
dX_core, C_t), and windowed avg/peak/variance is a function of X_core over
time, which fits that definition. BUT "episode" has a specific meaning in
this study: a trial in a controlled task environment that DOES NOT YET
EXIST (blocked on the client's D2 decision -- see CLAUDE.md's BLOCKED
section). The rolling 10-second window built here is an AGGREGATION WINDOW,
not a study episode, and must never be presented as if it were. The
disclosure (episode_unit="rolling_10s_window", blocked-on-D2 note) lives in
features/manifests/episodes_v1.json and docs/D1_DEPENDENCY_MAP.md, NOT in
the runtime window_summary dict itself (adding a field there would change
the schema and break the byte-exact G5 regression net).

PERMITTED: this module may import features.x_core (X_core -> E_t) and
features.geometry (upstream, shared).
FORBIDDEN: this module must NEVER import features.attention or
features.audio, directly or transitively -- see the module-level docstring
in features/__init__.py for the full directional rule.
"""

import uuid

import numpy as np

from features.geometry import WINDOW_SECONDS, YAW_VARIANCE_CEILING_DEG2

# Session-identity globals. Historically module-level in stage1_step4_vectors.py
# (SESSION_ID generated once at import time, PERSON_LABEL set by main() after
# consent) -- WindowAccumulator.flush() reads both as bare names, which Python
# resolves against the module where flush() is DEFINED, not the caller's
# module. Moving the class here means this module needs its own copies of
# both, kept in sync by whichever orchestrator actually owns the session:
# stage1_step4_vectors.py explicitly assigns features.episodes.SESSION_ID and
# features.episodes.PERSON_LABEL after generating/receiving its own (see that
# file's module-level code and main()). Consumers that do NOT perform that
# sync (stage3_demo_ui.py, analyze_video.py) get this module's own
# independently-generated SESSION_ID / default-None PERSON_LABEL in their
# window summaries -- this is NOT new behavior: before this refactor, those
# two files already got stage1_step4_vectors.py's SESSION_ID/PERSON_LABEL
# (a different value than their own same-named globals) for the exact same
# reason, since WindowAccumulator was defined in that file. The mismatch is
# harmless in both cases because neither consumer persists window_summary's
# own session_id/person_label fields to disk (see PROJECT files) -- it is
# disclosed here rather than silently carried forward.
SESSION_ID = str(uuid.uuid4())
PERSON_LABEL = None

# SCHEMA_VERSION for E_t's own record type. Historically the SAME module-
# level SCHEMA_VERSION string as every other record type in
# stage1_step4_vectors.py (one schema version for the whole log file, not
# per-record-type) -- kept identical in value here (not independently
# versioned) so window_summary records continue to carry the same version
# tag they always have. stage1_step4_vectors.py syncs this the same way it
# syncs SESSION_ID (see that file's module-level code).
SCHEMA_VERSION = "1.6"

DETECT_RATE_FLOOR = 0.5           # <50% of the window had a detected face -> unreliable, for any face


def classify_window_confidence(detection_rate, yaw_variance_deg2):
    """Binary window-validity gate. Deliberately dumb: either condition
    alone marks the window low-confidence (OR, not a weighted score) --
    no combining, no weighting, no per-person tuning (Pitfall #5). This
    is a session-quality fact true for any face, same spirit as the
    existing face<80px/yaw>35deg per-frame gate, just measured over a
    10s window instead of one frame:

      detect_rate < DETECT_RATE_FLOOR: the face wasn't reliably tracked
      for at least half the window -- unreliable regardless of who's in
      frame.

      yaw_variance > YAW_VARIANCE_CEILING_DEG2: the head moved through
      an unstable range within this one window (std ~10deg+) -- this is
      what the step-4 finding (facial vectors are direction-reliable but
      magnitude-noisy under rotation) means downstream: don't trust the
      avg/peak from a window where yaw itself was this unsettled.

    Returns (is_low_confidence: bool, reasons: list[str]). Flagged, not
    suppressed -- step 6 should show "reading unstable" rather than
    silently going blank, per the flag-not-suppress call already made
    for this gate.
    """
    reasons = []
    if detection_rate < DETECT_RATE_FLOOR:
        reasons.append(f"detect_rate {detection_rate:.2f} < floor {DETECT_RATE_FLOOR}")
    if yaw_variance_deg2 is not None and yaw_variance_deg2 > YAW_VARIANCE_CEILING_DEG2:
        reasons.append(f"yaw_variance {yaw_variance_deg2:.1f} > ceiling {YAW_VARIANCE_CEILING_DEG2}")
    return (len(reasons) > 0), reasons


class WindowAccumulator:
    """Step 5: the 10s rolling window (avg + peak + variance).

    Two-clock cadence (CADENCE): continuous sampling (add_sample, called
    every processing cycle) feeds this buffer; a 10s tick (should_flush/
    flush, checked once per cycle against the same clock) emits one
    summary and resets — a tumbling, non-overlapping window. This is
    what makes Gap 1 hold: "a 5s spike then neutral at s10 must still
    show in peak" needs every sample in [0s,10s) considered together,
    not just whatever's live at the 10s mark.

    Composite vectors (V_bf, V_es, V_pd) and logged-only covariates
    (V_jc, V_bf's convergence_ratio, V_es's cheek_raise) are collected
    into two SEPARATE, hardcoded buckets — not one generic "window
    every key" loop — specifically so a logged-only signal can't drift
    into this summary by accident. "Composite" here means "gets its own
    deviation/z-score tracked", NOT "feeds the V/A formula": per Decision
    18 (Gate 2), only V_es and V_pd actually feed
    map_to_valence_arousal()'s Valence/Arousal -- V_bf stays in this
    bucket so its own calibrated reading keeps getting logged and shown
    (Stage 3's demo bars, Track-B), but is excluded from the formula
    itself (Decision 17: did not generalize). Add a new covariate here
    explicitly if one shows up; never widen the composite set without a
    validity check backing it (see step 4 history).

    Peak policy (explicit, per CLAUDE.md CADENCE — decide and document,
    don't leave implicit): all three composites use plain max() over
    the window, which IS "max in the expression direction" here,
    because every composite was deliberately sign-flipped during step 4
    so higher = more expression (V_bf: more furrow: V_es: more crinkle;
    V_pd: more postural volatility, already non-negative by
    construction, so max-abs and max coincide anyway). Pre-calibration,
    there's no neutral baseline to deviate below zero from, so there is
    no separate "negative-direction spike" to also catch with
    max-absolute-deviation — that distinction only becomes real once
    Stage 1.5 turns these into deviation-from-neutral values, where an
    unusually-relaxed low reading could also be informative. Revisit
    peak policy there; max() is correct for the current raw-value
    schema.

    variance is the confidence signal for that window, not just a
    logged stat: a high-variance window (mid head-turn, unstable
    tracking) is a low-trust reading, by the direction-reliable/
    magnitude-noisy-under-rotation finding from step 4 validation.
    yaw_variance_deg2 and detection_rate are logged alongside each
    window as the raw ingredients a future confidence gate/weight needs.
    They are deliberately NOT combined into one hardcoded confidence
    number here — that combination should be calibrated across the
    8-12 person Gate 2 set (Stage 1.5), not guessed from n=1 data
    (Pitfall #5).

    D1 DISCLOSURE (Decision 2): this is E_t, with episode_unit=
    "rolling_10s_window" recorded in features/manifests/episodes_v1.json
    and docs/D1_DEPENDENCY_MAP.md -- NOT in the runtime summary dict
    itself (adding a field here would change window_summary's schema and
    break the byte-exact G5 regression net in
    tests/test_refactor_snapshot.py). The study's actual episode
    definition is BLOCKED on the client's D2 decision and, when it
    lands, may differ from this window. See the module docstring above.
    """

    def __init__(self):
        self.window_start = None
        self.samples = []

    def add_sample(self, ts, detected, yaw_deg, composite, covariate):
        if self.window_start is None:
            self.window_start = ts
        self.samples.append(
            {"ts": ts, "detected": detected, "yaw_deg": yaw_deg, "composite": composite, "covariate": covariate}
        )

    def should_flush(self, now, window_seconds=WINDOW_SECONDS):
        return self.window_start is not None and (now - self.window_start) >= window_seconds

    @staticmethod
    def _stats(samples, getter):
        vals = [v for v in (getter(s) for s in samples) if v is not None]
        if not vals:
            return {"avg": None, "peak": None, "variance": None, "n": 0}
        arr = np.array(vals)
        return {"avg": float(arr.mean()), "peak": float(arr.max()), "variance": float(arr.var()), "n": len(vals)}

    def flush(self, now):
        window_start = self.window_start
        samples = self.samples
        n_samples = len(samples)
        n_detected = sum(1 for s in samples if s["detected"])

        yaw_vals = [s["yaw_deg"] for s in samples if s["yaw_deg"] is not None]
        detection_rate = (n_detected / n_samples) if n_samples else 0.0
        yaw_variance_deg2 = float(np.var(yaw_vals)) if len(yaw_vals) >= 2 else None
        low_confidence, reasons = classify_window_confidence(detection_rate, yaw_variance_deg2)

        summary = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "window_summary",
            "session_id": SESSION_ID,
            "person_label": PERSON_LABEL,
            "window_start_monotonic": window_start,
            "window_end_monotonic": now,
            "window_seconds": now - window_start,
            "n_samples": n_samples,
            "n_detected": n_detected,
            "detection_rate": detection_rate,
            "yaw_mean_deg": float(np.mean(yaw_vals)) if yaw_vals else None,
            "yaw_variance_deg2": yaw_variance_deg2,
            # window-validity gate: flagged, not suppressed -- step 6
            # should render "reading unstable", not silently drop the
            # window. See classify_window_confidence() docstring.
            "window_quality": {"low_confidence": low_confidence, "reasons": reasons},
            # composite: vectors with their own deviation/z-score tracking.
            # Only v_es and v_pd actually feed step 6's V/A mapping (Decision
            # 18); v_bf is tracked here logged-only (Decision 17) -- see class
            # docstring on why this must never be a generic loop.
            "composite": {
                "v_bf": self._stats(samples, lambda s: s["composite"].get("v_bf")),
                "v_es": self._stats(samples, lambda s: s["composite"].get("v_es")),
                "v_pd": self._stats(samples, lambda s: s["composite"].get("v_pd")),
            },
            # covariates: Track-B logging only. Never read by step 6.
            "covariates": {
                "v_jc": self._stats(samples, lambda s: s["covariate"].get("v_jc")),
                "v_bf_convergence_ratio": self._stats(samples, lambda s: s["covariate"].get("v_bf_convergence_ratio")),
                "v_es_cheek_raise": self._stats(samples, lambda s: s["covariate"].get("v_es_cheek_raise")),
            },
        }
        self.window_start = None
        self.samples = []
        return summary
