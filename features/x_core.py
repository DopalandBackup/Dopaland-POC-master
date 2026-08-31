"""
X_core -- core behavioural/affective features (D1 feature-block separation,
CLAUDE.md). Moved verbatim out of stage1_step4_vectors.py; no formula,
threshold, or calibration logic changed (G5). See
tests/test_refactor_snapshot.py for the byte-exact regression net and
tests/test_feature_separation.py for the machine-checked import/call-graph
and runtime isolation proof that this module never reads attention.py or
audio.py.

CONTENTS: the 4 geometric affect-vector formulas (compute_v_bf, compute_v_es,
compute_v_jc, compute_v_pd), per-person neutral calibration (NeutralCalibrator,
classify_calibration_quality), z-scoring (_z_score), and the V/A mapping
(map_to_valence_arousal, Decision 1: this is X_core -- a re-representation of
X_core at the same per-cycle cadence, no temporal derivation, no context
input).

PERMITTED: this module may import features.geometry (upstream, shared).
FORBIDDEN: this module must NEVER import features.attention or
features.audio, directly or transitively -- see the module-level docstring
in features/__init__.py for the full directional rule.
"""

import numpy as np

from features.geometry import _dist, YAW_VARIANCE_CEILING_DEG2

# Per-person within-session neutral calibration (Decision #5, Pitfall
# #2, MANDATORY ARCHITECTURE #6). Fixed, universal duration -- same for
# everyone, no per-person tuning of the calibration window itself
# (Pitfall #5). Within-session only: no cross-session persistence, no
# FAISS/re-identification (explicitly OUT of POC, Decision #5) -- this
# calibrator is recreated from scratch every time the process starts.
CALIBRATION_SECONDS = 25.0  # within Decision #5's ~20-30s window
CALIBRATION_DRIFT_EFFECT_SIZE = 0.8  # Cohen's "large effect" convention (Cohen 1988) -- external, not tuned to any face

PD_BUFFER_SECONDS = 1.5  # "short ~1-2s rolling buffer" (CLAUDE.md V_pd definition)


def compute_v_bf(pts, io_dist):
    """Brow furrow: drop toward the glabella (composite), inner-brow
    convergence logged as a secondary component.

    Rebuild #2, composite = drop_ratio. Rebuild #1 (ratio vs
    eye_outer_span / same-side eye width — a same-extent denominator,
    chosen for yaw-tolerance) was confirmed INVERTED by the
    max-elicitation validity check: 3/3 maximal-furrow reps moved the
    composite the WRONG way. Root cause: eye-width denominators shrink
    under real furrowing too (co-occurring squint at high effort), so
    the ratio divided out the very signal it was meant to measure —
    over-normalization, not weak elicitation.

    This version normalizes by io_dist (inter-ocular iris-center
    distance) instead — expression-independent (iris centers don't move
    with brow/eyelid action). Distance-controlled re-test (holding
    camera distance fixed — the first uncontrolled attempt showed
    escalating wrong-direction drift traced to lean-in during
    concentration, not a formula fault) confirmed BOTH candidates move
    correctly, 3/3 reps:
      drop_ratio        — inner-brow-to-glabella distance / io_dist  (chosen: 3.1-4.4 sigma, more consistent)
      convergence_ratio — inner-brow-to-inner-brow distance / io_dist (logged only: 1.9-4.3 sigma, more variable)

    KNOWN LIMITATION: unlike rebuild #1, this denominator choice
    reintroduces moderate yaw sensitivity (r~=0.25 over a 0-34deg sweep,
    both candidates similar) -- io_dist (a narrow, near-center span)
    doesn't share the same lever-arm as the wide brow-to-glabella span,
    so it doesn't fully cancel yaw-linked landmark bias the way a
    same-extent denominator would. Accepted trade-off: direction-correct
    with moderate yaw noise beats yaw-clean but always-inverted. Watch
    this at Gate 2 -- if real-person validation shows yaw confounding
    genuine furrow reads, this formula needs revisiting, not re-tuning.
    """
    from features.geometry import BROW_INNER_R, BROW_INNER_L, GLABELLA

    inner_brow_dist = _dist(pts, BROW_INNER_R, BROW_INNER_L)
    convergence_ratio = inner_brow_dist / io_dist

    drop_r = _dist(pts, BROW_INNER_R, GLABELLA) / io_dist
    drop_l = _dist(pts, BROW_INNER_L, GLABELLA) / io_dist
    drop_ratio = (drop_r + drop_l) / 2.0

    # both SHRINK when furrowing; flip sign so higher = more furrow
    composite = -drop_ratio
    return composite, {"convergence_ratio": convergence_ratio, "drop_ratio": drop_ratio}


def compute_v_es(pts, io_dist):
    """Eye crinkle (Duchenne). Composite = aperture only.

    Max-elicitation validity check (3 reps, exaggerated smiles):
    aperture moved correctly and consistently (-2.0 to -2.6 sigma vs
    neutral, 3/3 reps) -- validated as the Duchenne-specific term.
    cheek_raise (iris-to-same-side-lip-corner span, meant to proxy
    midface/cheek compression) moved in the WRONG direction on 3/3 reps
    and was diluting the composite's otherwise-clean signal, so it's
    demoted to a logged-only covariate here, not part of the composite.
    CLAUDE.md's V_es spec calls for "BOTH" aperture and cheek/lower-lid
    tightening -- aperture alone carries V_es for now because it's the
    only piece that's actually validated; a second valid Duchenne term
    can be added later if one is found (a different geometric proxy,
    not a re-tuned version of this one -- the failure was directional,
    not a scale/threshold issue).

    V_es's real validation is still a genuine-vs-posed smile test at
    Stage 1.5, not this max-elicitation check, which only confirms the
    geometry moves the right way at all -- not that it distinguishes
    real from fake smiles.
    """
    from features.geometry import (
        EYE_R_OUTER, EYE_R_UPPER1, EYE_R_UPPER2, EYE_R_INNER, EYE_R_LOWER1, EYE_R_LOWER2,
        EYE_L_OUTER, EYE_L_UPPER1, EYE_L_UPPER2, EYE_L_INNER, EYE_L_LOWER1, EYE_L_LOWER2,
        IRIS_LEFT_CENTER, IRIS_RIGHT_CENTER, LIP_CORNER_R, LIP_CORNER_L,
    )

    def ear(outer, upper1, upper2, inner, lower1, lower2):
        vert = (_dist(pts, upper1, lower1) + _dist(pts, upper2, lower2)) / 2.0
        horiz = _dist(pts, outer, inner)
        return vert / horiz if horiz > 1e-6 else 0.0

    aperture = (
        ear(EYE_R_OUTER, EYE_R_UPPER1, EYE_R_UPPER2, EYE_R_INNER, EYE_R_LOWER1, EYE_R_LOWER2)
        + ear(EYE_L_OUTER, EYE_L_UPPER1, EYE_L_UPPER2, EYE_L_INNER, EYE_L_LOWER1, EYE_L_LOWER2)
    ) / 2.0

    def cheek_raise_side(iris_idx, lip_corner_idx, eye_outer, eye_inner):
        span = _dist(pts, iris_idx, lip_corner_idx)
        local_scale = _dist(pts, eye_outer, eye_inner)
        return span / local_scale if local_scale > 1e-6 else 0.0

    # logged-only covariate, not composited -- see docstring
    cheek_raise_avg = (
        cheek_raise_side(IRIS_RIGHT_CENTER, LIP_CORNER_R, EYE_R_OUTER, EYE_R_INNER)
        + cheek_raise_side(IRIS_LEFT_CENTER, LIP_CORNER_L, EYE_L_OUTER, EYE_L_INNER)
    ) / 2.0

    # aperture shrinks when crinkling; flip sign so higher = more crinkle
    composite = -aperture
    return composite, {"aperture": aperture, "cheek_raise": cheek_raise_avg}


def compute_v_jc(pts, io_dist):
    """Jaw compression — LOGGED-ONLY, not a POC-reliable vector.

    Max-elicitation validity check (3 max-effort jaw-clench reps) found
    no repeatable directional signal: rep1 moved 8.86 sigma in the WRONG
    direction (likely the mouth opened rather than pressed shut for that
    attempt), reps 2-3 stayed within noise either direction. This
    confirms CLAUDE.md's own prediction that jaw compression is largely
    invisible to surface landmark geometry (muscle tension, beard
    occlusion) — now shown empirically even under maximal, repeated
    effort, not just casual attempts.

    Do NOT try to extract more signal here (no new landmark, no new
    ratio) — the data says the surface geometry doesn't carry it, not
    that the formula is wrong. inter_lip_dist and masseter_width keep
    being logged raw as a Track-B covariate; a trained model may
    eventually strengthen this signal from cues geometry can't isolate.
    lip_corner_dist is exploratory, also logged raw.
    """
    from features.geometry import LIP_UPPER_INNER, LIP_LOWER_INNER, LIP_CORNER_R, LIP_CORNER_L, JAW_ANGLE_R, JAW_ANGLE_L

    inter_lip_dist = _dist(pts, LIP_UPPER_INNER, LIP_LOWER_INNER) / io_dist
    lip_corner_dist = _dist(pts, LIP_CORNER_R, LIP_CORNER_L) / io_dist
    masseter_width = _dist(pts, JAW_ANGLE_R, JAW_ANGLE_L) / io_dist
    composite = -inter_lip_dist
    return composite, {
        "inter_lip_dist": inter_lip_dist,
        "lip_corner_dist": lip_corner_dist,
        "masseter_width": masseter_width,
    }


def compute_v_pd(pd_buffer, nose_world, shoulder_mid_world, now):
    """Postural volatility: variance of nose + shoulder-midpoint world
    position over its own short rolling buffer (distinct from the later
    10s summary window built in step 5)."""
    pd_buffer.append((now, nose_world, shoulder_mid_world))
    while pd_buffer and now - pd_buffer[0][0] > PD_BUFFER_SECONDS:
        pd_buffer.popleft()
    if len(pd_buffer) < 2:
        return None, {"buffer_len": len(pd_buffer)}
    noses = np.array([b[1] for b in pd_buffer])
    shoulders = np.array([b[2] for b in pd_buffer])
    nose_var = float(np.var(noses, axis=0).sum())
    shoulder_var = float(np.var(shoulders, axis=0).sum())
    return nose_var + shoulder_var, {
        "nose_pos_variance": nose_var,
        "shoulder_pos_variance": shoulder_var,
        "buffer_len": len(pd_buffer),
    }


def classify_calibration_quality(samples, composite_keys, yaw_vals):
    """In-capture contamination flag for the neutral-calibration window.
    Same pattern as classify_window_confidence: binary flags, universal
    thresholds, NOT tuned to any one person's face (Pitfall #5), and a
    FLAG not an auto-reject -- a contaminated "neutral" should be
    surfaced for the operator to consider redoing, not silently
    discarded or silently accepted.

    Two checks:
      1. Movement: reuses the SAME yaw_variance ceiling already
         established for the window-validity gate (not a new number) --
         head movement during "neutral" holding is a session-quality
         fact regardless of who's in frame.
      2. Expression drift: splits the capture in half and compares each
         composite vector's mean between halves, standardized by the
         pooled within-half std (a scale-free effect size, so it works
         identically regardless of a vector's raw magnitude). Trigger is
         Cohen's conventional "large effect" (d=0.8, Cohen 1988) -- an
         external, domain-standard convention, not reverse-engineered
         from this session's own data. Catches "started neutral,
         drifted into and held an expression" during the capture --
         exactly what a genuinely stable neutral capture should NOT
         show, and exactly what a hard variance-magnitude cutoff
         couldn't catch cleanly across vectors of very different scale.

    KNOWN BLIND SPOT (confirmed empirically, not theoretical): a
    contamination held CONSTANT from the very first sample is
    invisible to both checks. Tested directly in
    stage1_step8_calibration_repeat_test.py round 4 (a deliberate faint
    smile held for the full 25s) -- the flag returned OK. Root cause:
    both checks are WITHIN-CAPTURE statistics (movement variance,
    first-half-vs-second-half drift); a uniform bias present from t=0
    produces low variance and zero drift, identical to a genuinely
    calm, stable neutral. Detecting it would require an external
    reference -- either cross-session/population data on what "normal"
    neutral values look like (out of POC scope, Decision #5) or a
    per-face tuned reference (Pitfall #5). Both are the wrong fix right
    now. Do NOT try to close this gap in software -- the mitigation is
    procedural: the calibration UX shows live raw values on-screen
    specifically so a human running Gate 2 can eyeball a resting face
    that looks off, which this flag structurally cannot.

    PER-VECTOR REPORTING (not just a session-level yes/no): the drift
    check fired on 3 of 4 real calibration attempts in this session
    (v_es, then v_pd, then v_bf, across separate runs) -- this is NOT
    the threshold being oversensitive. For v_es specifically, it's the
    SAME neutral-drift already found independently by the repeat-
    calibration test (~0.02 spread across back-to-back captures,
    Decision 44) showing up from a second angle, inside a single
    capture instead of across several. Loosening the threshold would
    hide that real instability, not fix it. So this returns WHICH
    vector(s) drifted, tagged with their VECTOR_RELIABILITY, so a
    consumer (console output, future Gate 2 tooling) can read a v_es
    flag as "expected -- known-noisy vector, not a protocol problem"
    and a v_bf flag as "unexpected -- investigate, this vector's
    neutral is otherwise stable" (repeat-calibration range 0.0005).
    Same flag, different meaning depending on which vector tripped it.
    """
    reasons = []
    drifted_vectors = []  # structured, per-vector: [{"vector","effect_size","reliability","interpretation"}]

    yaw_variance = float(np.var(yaw_vals)) if len(yaw_vals) >= 2 else None
    if yaw_variance is not None and yaw_variance > YAW_VARIANCE_CEILING_DEG2:
        reasons.append(f"head movement during capture: yaw_variance={yaw_variance:.1f} > ceiling {YAW_VARIANCE_CEILING_DEG2}")

    n = len(samples)
    if n >= 20:
        mid = n // 2
        first_half, second_half = samples[:mid], samples[mid:]
        for key in composite_keys:
            v1 = np.array([v for v in (s["composite"].get(key) for s in first_half) if v is not None])
            v2 = np.array([v for v in (s["composite"].get(key) for s in second_half) if v is not None])
            if len(v1) < 5 or len(v2) < 5:
                continue
            pooled_std = np.sqrt((v1.var() + v2.var()) / 2.0)
            if pooled_std < 1e-9:
                continue
            effect_size = abs(v1.mean() - v2.mean()) / pooled_std
            if effect_size >= CALIBRATION_DRIFT_EFFECT_SIZE:
                reliability = VECTOR_RELIABILITY.get(key)
                interpretation = (
                    "expected -- known-noisy vector, not a protocol problem"
                    if reliability == "low"
                    else "unexpected -- investigate, this vector's neutral is otherwise stable"
                )
                drifted_vectors.append(
                    {"vector": key, "effect_size": effect_size, "reliability": reliability, "interpretation": interpretation}
                )
                reasons.append(
                    f"{key} drifted mid-capture: effect_size={effect_size:.2f} >= {CALIBRATION_DRIFT_EFFECT_SIZE} ({interpretation})"
                )

    return (len(reasons) > 0), reasons, drifted_vectors


class NeutralCalibrator:
    """Stage 1.5, step 8: per-person within-session neutral calibration.

    Runs ONCE at session start, for a fixed CALIBRATION_SECONDS -- same
    duration for everyone, no per-person tuning of the calibration
    procedure itself (Pitfall #5).

    Before completion: system is UNCALIBRATED. Per Gap 2 L1 / Pitfall
    #3, there is no confident reading to report yet during this phase
    -- callers must check is_calibrated() and surface an explicit
    "calibrating" state (cold-start), never a reading built on an
    incomplete or absent baseline.

    After completion: the neutral reference is FROZEN for the rest of
    the process. Within-session only -- no cross-session persistence,
    no FAISS/re-identification, no re-calibration trigger (Decision #5:
    explicitly OUT of POC scope). deviation() reports every subsequent
    sample relative to THIS person's own resting values, not a raw
    magnitude -- this is the fix for Pitfall #2 (one person's neutral
    brow is another's furrow): only the deviation is comparable across
    people, never the raw number alone.

    Keeps mean AND spread (std) per vector. The spread is this person's
    own noise floor -- logged so Gate 2 can later judge whether a
    deviation is a real signal or within this person's own natural
    jitter. Not used as a threshold here: that judgment needs real
    multi-person data (Pitfall #5) -- calibration only hands off the
    raw ingredient (mean, std, n), it does not decide what counts as
    "significant".

    Covariates (V_jc, V_bf's convergence_ratio, V_es's cheek_raise) get
    a neutral reference too (for Track-B completeness) but are not
    reported as deviation anywhere -- they're logged-only, same as
    their windowed raw values in step 5.
    """

    COMPOSITE_KEYS = ("v_bf", "v_es", "v_pd")
    COVARIATE_KEYS = ("v_jc", "v_bf_convergence_ratio", "v_es_cheek_raise")

    def __init__(self, calibration_seconds=CALIBRATION_SECONDS):
        self.calibration_seconds = calibration_seconds
        self.start_ts = None
        self.samples = []
        self.reference = None  # frozen once complete

    def is_calibrated(self):
        return self.reference is not None

    def seconds_remaining(self, now):
        if self.start_ts is None:
            return self.calibration_seconds
        return max(0.0, self.calibration_seconds - (now - self.start_ts))

    def add_sample(self, ts, composite, covariate, yaw_deg=None):
        if self.is_calibrated():
            return
        if self.start_ts is None:
            self.start_ts = ts
        self.samples.append({"composite": composite, "covariate": covariate, "yaw_deg": yaw_deg})

    def should_complete(self, now):
        return (
            not self.is_calibrated()
            and self.start_ts is not None
            and (now - self.start_ts) >= self.calibration_seconds
        )

    @staticmethod
    def _stats(samples, bucket, key):
        vals = [v for v in (s[bucket].get(key) for s in samples) if v is not None]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}

    def complete(self, now):
        yaw_vals = [s["yaw_deg"] for s in self.samples if s["yaw_deg"] is not None]
        possibly_not_neutral, reasons, drifted_vectors = classify_calibration_quality(
            self.samples, self.COMPOSITE_KEYS, yaw_vals
        )
        self.reference = {
            "person_label": PERSON_LABEL,
            "calibrated_at_monotonic": now,
            "calibration_seconds": now - self.start_ts,
            "composite": {k: self._stats(self.samples, "composite", k) for k in self.COMPOSITE_KEYS},
            "covariates": {k: self._stats(self.samples, "covariate", k) for k in self.COVARIATE_KEYS},
            # in-capture contamination flag -- see classify_calibration_quality.
            # Flagged, not auto-rejected: the operator decides whether to redo.
            # drifted_vectors is the structured, per-vector form (machine-
            # readable for future Gate 2 tooling); reasons is the same
            # information as human-readable strings.
            "quality": {"possibly_not_neutral": possibly_not_neutral, "reasons": reasons, "drifted_vectors": drifted_vectors},
        }
        return self.reference

    def deviation(self, key, raw_value):
        """Deviation from this person's own neutral -- what step 6 must
        consume, never the raw value (Pitfall #2)."""
        if not self.is_calibrated() or raw_value is None:
            return None
        mean = self.reference["composite"].get(key, {}).get("mean")
        if mean is None:
            return None
        return raw_value - mean


# Step 6: Valence/Arousal mapping — POST-GATE-2 (Decision 18, supersedes
# retired Decision 50). map_to_valence_arousal() below is the CANONICAL
# implementation used everywhere in this codebase (stage3_demo_ui.py and
# analyze_video.py import it from here rather than keeping their own copy)
# -- one formula, one place.
#
# Reliability tags carried forward from what Stage 1 + Stage 1.5
# characterization actually established (not vibes):
#   v_bf = high    -- repeat-calibration range 0.0005 across 3 back-to-back
#                     captures (stage1_step8_calibration_repeat_test.py)
#   v_es = low     -- neutral drift ~0.02 across the same test, comparable
#                     in size to a small genuine expression; shares the
#                     yaw-noisy io_dist denominator; only ONE validated
#                     component (aperture) after cheek_raise was demoted
#   v_pd = medium  -- world-frame (camera-distance invariant by
#                     construction) but far less repeat-tested than v_bf
#   v_jc = logged_only -- never composited, no reliable directional signal
#                         even at max effort (step 4 max-elicitation check)
#
# This is METADATA for a consumer (this file's own overlay/plot right
# now, a real demo later) to caveat with -- e.g. "V_es contribution:
# LOW CONFIDENCE". It is NOT used to reweight the mapping formula below:
# that weighting needs Gate 2's multi-person data to justify, not a
# guess baked in now (Pitfall #5).
VECTOR_RELIABILITY = {
    "v_bf": "high",
    "v_es": "low",
    "v_pd": "medium",
    "v_jc": "logged_only",
}

# NeutralCalibrator.complete() stamps this into every reference it builds.
# Historically a module-level global in stage1_step4_vectors.py, set once by
# main() after consent (step 7). Left as a module-level global here too
# (not threaded through every call) so the move is a pure relocation, not a
# behavior change (G5) -- stage1_step4_vectors.py sets
# features.x_core.PERSON_LABEL the same way it used to set its own
# module-level PERSON_LABEL.
PERSON_LABEL = None


def _z_score(deviation, std):
    """Deviation standardized by THIS PERSON's own neutral spread (std
    from calibration) -- reuses calibration's own output as the scale,
    rather than inventing a new constant (Pitfall #5)."""
    if deviation is None or std is None or std < 1e-9:
        return None
    return deviation / std


def map_to_valence_arousal(v_bf_dev, v_es_dev, v_pd_dev, neutral_ref):
    """Step 6 — rule-based V/A mapping. CANONICAL implementation of
    Decision 18, the Gate 2 result: this is the ONLY place Valence/
    Arousal get computed -- stage3_demo_ui.py and analyze_video.py import
    this function rather than keeping their own copy, so there is exactly
    one formula.

      Valence = z_es only  -- SINGLE-SOURCE, pleasure-side only. There
                               is NO validated pain axis. A near-zero or
                               negative value means "no detected
                               pleasure signal", not "detected pain" --
                               label it that way everywhere this is
                               surfaced (log, overlay, demo).
      Arousal = z_pd only  -- more postural volatility = higher arousal.
    tanh(z/2) saturates the plotted point into a bounded square; 2 std
    devs is a common, domain-general statistical convention (unchanged
    from the pre-Gate-2 version, not reverse-engineered from any
    session's data).

    RETIRED: Valence = (z_es - z_bf) / 2 (Decision 50, pre-Gate-2). Gate
    2 (B-series) found V_bf's furrow direction did NOT generalize across
    faces (0/7 correct-signed, opposite-signed vs. its own concentrate
    reading -- see Decision 17 / VECTOR_RELIABILITY history): V_bf can
    no longer stand for pain, and this formula must never be
    reintroduced. v_bf_dev is still accepted and z-scored below purely
    so callers (this file's own overlay, Track-B logs) can keep showing
    V_bf's own calibrated reading alongside the V/A point for
    transparency -- it is logged-only and plays no part in the valence/
    arousal values themselves.

    Returns None components (not zeros) wherever a vector is
    uncalibrated or undetected this cycle -- never fabricate a point
    from a missing input.
    """
    if neutral_ref is None:
        return {"valence": None, "arousal": None, "components": {}}

    z_bf = _z_score(v_bf_dev, neutral_ref["composite"]["v_bf"]["std"])
    z_es = _z_score(v_es_dev, neutral_ref["composite"]["v_es"]["std"])
    z_pd = _z_score(v_pd_dev, neutral_ref["composite"]["v_pd"]["std"])

    valence = float(np.tanh(z_es / 2.0)) if z_es is not None else None
    arousal = float(np.tanh(z_pd / 2.0)) if z_pd is not None else None

    return {
        "valence": valence,
        "arousal": arousal,
        "components": {
            # logged-only, not used in valence/arousal above -- see docstring
            "v_bf": {"z": z_bf, "reliability": VECTOR_RELIABILITY["v_bf"]},
            "v_es": {"z": z_es, "reliability": VECTOR_RELIABILITY["v_es"]},
            "v_pd": {"z": z_pd, "reliability": VECTOR_RELIABILITY["v_pd"]},
        },
    }
