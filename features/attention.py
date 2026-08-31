"""
A_t -- attention features (D1 feature-block separation, CLAUDE.md). Moved
verbatim out of stage1_step4_vectors.py; no formula, threshold, or state-
machine logic changed (G5). See tests/test_refactor_snapshot.py for the
byte-exact regression net (note: the validated path it covers is X_core/E_t/
V-A mapping only -- A_t was never part of that net, since it was never part
of Gate 2's scored, validated path) and tests/test_feature_separation.py for
the machine-checked import/call-graph and runtime isolation proof that
x_core.py and episodes.py never read this module.

CONTENTS: screen orientation (V_so, compute_v_so, its gaze-centering
secondary input _gaze_centering_score, AttentionWindowAccumulator), plus the
EXPERIMENTAL SIGNALS PART 2 additions (compute_gaze_direction, BlinkDetector)
-- ROI, orientation, dwell, persistence, switching, head-gaze coherence,
gaze direction, and blink all live here, per the D1 exclusion list in
docs/D1_DEPENDENCY_MAP.md. Everything in this module is UNVALIDATED
(module-level honest-framing rule, unchanged by this move) and PILOT, not
POC-ready, per CLAUDE.md's "Attention / screen-orientation -- PILOT, not
POC" note.

PERMITTED: this module may read core-derived values as PLAIN PARAMETERS
passed in by a caller (e.g. BlinkDetector.update(aperture, now) -- the
caller reads V_es's own aperture and passes it in; this module never imports
features.x_core to compute it itself). That is X_core -> A_t, a permitted
direction distinct from the forbidden A_t -> X_core.
FORBIDDEN: this module must NEVER import features.x_core or features.episodes,
directly or transitively -- see the module-level docstring in
features/__init__.py for the full directional rule. It also must never be
imported BY x_core.py or episodes.py (checked from the other side by
tests/test_feature_separation.py).
This module MAY import features.geometry (upstream, shared).
"""

import uuid
from collections import deque

import numpy as np

from features.geometry import (
    EYE_L_INNER, EYE_L_OUTER, EYE_R_INNER, EYE_R_OUTER,
    IRIS_LEFT_CENTER, IRIS_RIGHT_CENTER, WINDOW_SECONDS,
)

# Session-identity globals -- same pattern and same reasoning as
# features/episodes.py's own SESSION_ID/PERSON_LABEL (see that module's
# comment): AttentionWindowAccumulator.flush() reads both as bare names,
# resolved against the module where flush() is DEFINED. stage1_step4_vectors.py
# syncs these explicitly; stage3_demo_ui.py and analyze_video.py do not use
# AttentionWindowAccumulator at all (confirmed by search -- see
# docs/D1_DEPENDENCY_MAP.md's two-parallel-loops finding).
SESSION_ID = str(uuid.uuid4())
PERSON_LABEL = None

SCHEMA_VERSION = "1.6"  # same single log-schema-version string as episodes.py -- see that module's comment

# --- ATTENTION SIGNAL (V_so, screen orientation) -- Step 1, UNVALIDATED ---
# Every threshold below is a first-cut estimate, explicitly derived from
# the ALREADY-EXISTING 35deg quality-gate bound (same discipline as
# YAW_VARIANCE_CEILING_DEG2 in features/geometry.py), NOT fit to any one
# face (Pitfall #5). These are exactly the numbers a Gate-2-style directed
# test ("look at screen" vs "look away") is meant to validate or correct --
# do not tune them against the developer's own footage.
ATTENTION_YAW_THRESHOLD_DEG = 20.0    # roughly half the 35deg gate -- past this, "trackable" no longer means "facing the screen"
ATTENTION_PITCH_THRESHOLD_DEG = 20.0  # same first-cut magnitude as yaw; no existing pitch precedent to derive from yet
ATTENTION_POSE_WEIGHT = 0.7           # gaze can only ever nudge the score by <=30% -- pose stays dominant so the signal keeps working with glasses (gaze unreliable) or gaze absent entirely
ATTENTION_ORIENTED_SCORE_THRESHOLD = 0.5  # score>=this -> the boolean "oriented" call
GAZE_PLAUSIBLE_SLACK = 0.5             # how far outside the raw [0,1] eye-corner span an iris ratio may sit before it's treated as corrupted (glasses reflection etc.) rather than "looking far to the side"


def _gaze_centering_score(pts):
    """SECONDARY/bonus input to compute_v_so -- never a dependency (see
    module docstring's GLASSES-ROBUST section). Reuses the SAME iris
    and eye-corner landmark indices interocular_distance/compute_v_es
    already read (IRIS_LEFT/RIGHT_CENTER, EYE_*_OUTER/INNER) -- no new
    landmarks, no new detection.

    For each eye, computes where the iris sits between that eye's outer
    (temple-side) and inner (nose-side) corner, as a 0..1 ratio (0.5 =
    centered). Deviation-from-0.5 is what's averaged across eyes, NOT
    the raw ratio -- the two eyes' outer->inner axes point in opposite
    literal directions (confirmed by hand-tracing the geometry: gazing
    to one side moves one eye's ratio toward 0 and the other's toward 1
    for the SAME physical gaze shift), so averaging raw ratios would
    cancel out and hide lateral gaze. Averaging |deviation| instead
    sidesteps that entirely, since both eyes' deviation grows together
    regardless of which raw direction each one moved.

    Reliability is judged geometrically, not via any confidence score
    (Tasks output carries none per-landmark, Decision 14): MediaPipe
    doesn't report "the iris is disappearing behind a glasses reflection"
    -- it silently returns a landmark position, which glasses corrupt
    into being far outside the anatomically plausible eye-corner span.
    A ratio further than GAZE_PLAUSIBLE_SLACK past [0, 1] is treated as
    corrupted and excluded, not as "an extreme side-glance".

    Returns (score, reliable). score is only meaningful when
    reliable=True -- same "ignore the value when the flag says so"
    convention as every other None-on-uncomputable value in this file.
    Never raises: a degenerate (zero-width) eye span for either eye is
    just excluded, same as an implausible ratio.

    IRIS_LEFT_CENTER/IRIS_RIGHT_CENTER pair with their same-named
    EYE_L_*/EYE_R_* corners directly below -- no workaround needed here.
    (Historical note: this function originally had to invert that
    pairing locally because the two constants were defined backwards at
    the time; that has since been fixed at the constant definitions
    themselves -- see IRIS_SWAP_DIAGNOSTIC.md -- so the natural, named
    pairing is correct again.)
    """

    def eye_ratio(outer_idx, inner_idx, iris_idx):
        outer_x, inner_x = pts[outer_idx][0], pts[inner_idx][0]
        span = inner_x - outer_x
        if abs(span) < 1e-6:
            return None
        return (pts[iris_idx][0] - outer_x) / span

    r_ratio = eye_ratio(EYE_R_OUTER, EYE_R_INNER, IRIS_RIGHT_CENTER)
    l_ratio = eye_ratio(EYE_L_OUTER, EYE_L_INNER, IRIS_LEFT_CENTER)

    deviations = []
    for ratio in (r_ratio, l_ratio):
        if ratio is not None and -GAZE_PLAUSIBLE_SLACK <= ratio <= 1.0 + GAZE_PLAUSIBLE_SLACK:
            deviations.append(min(abs(ratio - 0.5) / 0.5, 1.0))  # 0 = centered, 1 = at/past either corner

    if not deviations:
        return None, False

    avg_deviation = sum(deviations) / len(deviations)
    # float(...): pts is a numpy array, so every value above is numpy-typed
    # (np.float64) -- cast to native Python here, same discipline
    # WindowAccumulator._stats already uses, so this can never trip up
    # json.dumps() when it's logged (numpy scalars are not JSON-serializable).
    return float(max(0.0, 1.0 - avg_deviation)), True


def compute_v_so(normalized_pts, yaw_deg, pitch_deg):
    """Screen orientation (V_so) -- ATTENTION SIGNAL Step 1. See the
    original module docstring's ATTENTION SIGNAL section (preserved in
    stage1_step4_vectors.py's module docstring) for the full design
    rationale (glasses-robust-by-construction, no calibration needed,
    honest-framing rule). Summary:

    PILOT-ONLY, NOT POC-ready: pitch_deg's contribution below is UNRELIABLE
    by structural limit of single-camera landmark head-pose (face
    foreshortening on look-down degrades the landmark data, not a bug in
    this formula or in yaw_pitch_roll_from_matrix) -- do NOT attempt to
    "fix" pitch. yaw_deg's contribution is reliable. Do not change the
    computation below on the strength of this comment.

    PRIMARY = head pose. yaw_deg/pitch_deg are NOT recomputed here --
    passed in straight from yaw_pitch_roll_from_matrix(matrix) (now in
    features/geometry.py), the SAME decomposition the quality gate already
    runs every cycle. Reduces linearly from 1.0 at dead-center to 0.0 at
    ATTENTION_YAW_THRESHOLD_DEG / ATTENTION_PITCH_THRESHOLD_DEG,
    whichever axis is further off-center.

    SECONDARY = gaze (_gaze_centering_score), blended in at
    ATTENTION_POSE_WEIGHT:(1-ATTENTION_POSE_WEIGHT) ONLY when
    geometrically plausible -- otherwise the score is pose alone and
    components["head_pose_only"] is stamped True so every consumer
    (overlay, log, a future validation pass) can see the fallback
    engaged, never silently.

    Returns (orientation_score, components) -- same 2-tuple shape as
    compute_v_bf/v_es/v_jc/v_pd. components["oriented"] is the
    thresholded boolean call (score >= ATTENTION_ORIENTED_SCORE_THRESHOLD),
    included for direct use in a Gate-2-style commanded-direction check
    ("look at screen" should read oriented=True; "look away"/"look
    down" should read oriented=False) -- see AttentionWindowAccumulator
    for the per-window rollup this feeds (oriented_rate) that a human
    scorer would actually compare against the commanded label. Every
    caller must treat this as UNVALIDATED (module docstring) -- it is
    not auto-scored anywhere in this codebase.
    """
    yaw_frac = min(abs(yaw_deg) / ATTENTION_YAW_THRESHOLD_DEG, 1.0) if yaw_deg is not None else 1.0
    pitch_frac = min(abs(pitch_deg) / ATTENTION_PITCH_THRESHOLD_DEG, 1.0) if pitch_deg is not None else 1.0
    pose_score = max(0.0, 1.0 - max(yaw_frac, pitch_frac))

    gaze_score, gaze_reliable = _gaze_centering_score(normalized_pts)

    if gaze_reliable:
        orientation_score = ATTENTION_POSE_WEIGHT * pose_score + (1.0 - ATTENTION_POSE_WEIGHT) * gaze_score
        head_pose_only = False
    else:
        orientation_score = pose_score
        head_pose_only = True

    oriented = orientation_score >= ATTENTION_ORIENTED_SCORE_THRESHOLD

    components = {
        "oriented": oriented,
        "pose_score": pose_score,
        "gaze_score": gaze_score if gaze_reliable else None,
        "gaze_reliable": gaze_reliable,
        "head_pose_only": head_pose_only,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
    }
    return orientation_score, components


# ============================================================
# EXPERIMENTAL SIGNALS, PART 2 -- gaze direction (L/R/C) + blink rate.
# NEW, ISOLATED, UNVALIDATED additions (distinct from V_so above, though
# compute_gaze_direction reuses the SAME landmarks/ratio approach
# _gaze_centering_score uses). Neither is composited into any affect
# vector, neither touches WindowAccumulator/NeutralCalibrator/
# AttentionWindowAccumulator, and neither has been validated across
# people -- same "keep the code, mark it clearly, don't wire it into
# anything it hasn't earned" discipline as CLAUDE.md's "Attention /
# screen-orientation -- PILOT, not POC" note. Horizontal gaze only:
# vertical/up-down gaze is NEVER computed, here or anywhere -- it would
# degrade under the exact same face-foreshortening failure mode already
# confirmed for head pitch.
# ============================================================
GAZE_DIRECTION_DEVIATION_THRESHOLD = 0.25  # first-cut estimate (0=centered..1=at eye corner, same deviation metric _gaze_centering_score uses internally) -- NOT tuned to one face, pending the same kind of cross-person directed-capture validation V_so itself is still waiting on
GAZE_LABEL_SIGN = -1  # which sign of (r_ratio - l_ratio) maps to the word "RIGHT" vs "LEFT". CONFIRMED BACKWARDS at +1 by live on-camera testing (looking to the person's own left read "RIGHT") and flipped to -1 to fix it -- the direction is now reported from the PERSON'S OWN perspective (their own left reads "LEFT"), same convention as everyday mirror-vs-camera intuition. Only relabels the word, never changes raw_shift, reliability, or the CENTER/UNKNOWN calls -- if a future camera/setup shows it backwards again, flip this single constant back rather than touching the ratio math above.


def compute_gaze_direction(pts):
    """EXPERIMENTAL / UNVALIDATED -- coarse horizontal gaze label: one
    of "LEFT" / "RIGHT" / "CENTER" / "UNKNOWN". Reuses the exact
    landmarks and per-eye ratio formula _gaze_centering_score already
    uses (IRIS_LEFT_CENTER/IRIS_RIGHT_CENTER against EYE_*_OUTER/INNER,
    same GAZE_PLAUSIBLE_SLACK corruption gate) -- this is a NEW,
    separate function rather than a change to that one because
    _gaze_centering_score deliberately discards direction (it only
    needs "how far off center" for V_so's blended score, never "which
    way"): it averages each eye's |ratio-0.5| deviation, which cancels
    sign. Its own docstring explains why that cancellation is correct
    for ITS purpose but wrong for this one: a real lateral gaze shift
    moves the two eyes' RAW ratios in OPPOSITE directions (one toward 0,
    the other toward 1) for the same physical shift, so the SIGNED
    difference (r_ratio - l_ratio) grows/shrinks consistently with real
    gaze direction, where an average-of-magnitudes would erase it.

    Reliability uses the SAME corruption gate _gaze_centering_score
    already applies (a ratio further than GAZE_PLAUSIBLE_SLACK outside
    [0,1] is glasses-reflection/occlusion corruption, not a real extreme
    glance) -- reused unchanged, not re-derived. Returns
    ("UNKNOWN", False, None) whenever either eye's ratio is missing or
    implausible: never guesses a direction from a single eye, never
    reports a stale/guessed label.

    Returns (label, reliable, raw_shift). raw_shift is the signed
    geometric value (real, kept for Track-B logging richness) -- NEVER
    surfaced in the UI as a precise angle; display only ever shows the
    discrete label, per this feature's own "coarse, not precise" rule.
    """

    def eye_ratio(outer_idx, inner_idx, iris_idx):
        outer_x, inner_x = pts[outer_idx][0], pts[inner_idx][0]
        span = inner_x - outer_x
        if abs(span) < 1e-6:
            return None
        return (pts[iris_idx][0] - outer_x) / span

    def plausible(ratio):
        return ratio is not None and -GAZE_PLAUSIBLE_SLACK <= ratio <= 1.0 + GAZE_PLAUSIBLE_SLACK

    r_ratio = eye_ratio(EYE_R_OUTER, EYE_R_INNER, IRIS_RIGHT_CENTER)
    l_ratio = eye_ratio(EYE_L_OUTER, EYE_L_INNER, IRIS_LEFT_CENTER)

    if not (plausible(r_ratio) and plausible(l_ratio)):
        return "UNKNOWN", False, None

    shift = float(r_ratio - l_ratio)
    if abs(shift) < GAZE_DIRECTION_DEVIATION_THRESHOLD:
        return "CENTER", True, shift
    return ("RIGHT" if (shift * GAZE_LABEL_SIGN) > 0 else "LEFT"), True, shift


# --- Blink rate (blinks/min) -- EXPERIMENTAL / UNVALIDATED ---
# Reuses V_es's own eye-aperture value (compute_v_es's returned
# es_components["aperture"], read-only): this section computes NO new
# landmark geometry and does not reimplement or touch ear()/aperture
# itself, per this feature's own hard constraint. The caller (this
# module's own consumer, e.g. stage1_step4_vectors.py's processing_thread
# or stage3_demo_ui.py's stage3_processing_thread) reads that value from
# X_core and passes it into BlinkDetector.update() as a plain float --
# this module never imports features.x_core to fetch it itself (X_core ->
# A_t is permitted; the reverse is not, and is never exercised here).

BLINK_ROLLING_MAX_SECONDS = 3.0      # "recent eyes-open" reference window -- adapts to lighting/face-size drift instead of needing its own calibration pass
# BLINK_CLOSE_FRACTION / BLINK_REOPEN_FRACTION -- RE-TUNED against REAL
# logged aperture data from this user (not a generic EAR-literature
# guess): open-eye baseline ~0.45-0.50; a real blink dips only to
# ~0.38-0.41, i.e. ~0.80-0.85x of baseline, before tracking typically
# LOSES the eye entirely (reads None) at full closure -- this formula/
# geometry never approaches the near-zero values classic EAR literature
# describes. The OLD close_fraction (0.6) required aperture to fall
# below 0.6x baseline (~0.28 at a 0.47 baseline) to register as
# "closing" -- a real blink here never gets remotely that low, so
# "closing" was never entered at all and the count stayed 0 no matter
# what the plausibility floor was set to (that was a red herring for
# THIS user's data, even though it was a real fix for the generic case).
# close_fraction=0.87 sits just above the shallowest observed dip ratio
# (0.85), so a 0.38-0.41 dip from a ~0.45-0.50 baseline reliably crosses
# it (0.40/0.47=0.851 < 0.87), while staying comfortably below normal
# open-eye stability (small frame-to-frame jitter, not a 13%+ dip).
# reopen_fraction=0.90 sits above close_fraction (proper hysteresis --
# harder to CONFIRM than to START, so a value that merely wobbles near
# the close boundary can't complete a false blink cycle) while still
# being a realistic recovery target given the same 0.45-0.50 open range.
BLINK_CLOSE_FRACTION = 0.87
BLINK_REOPEN_FRACTION = 0.90
BLINK_MIN_DURATION_SECONDS = 0.05    # a closure shorter than this is more likely landmark jitter than a real blink
BLINK_MAX_DURATION_SECONDS = 0.6     # a closure longer than this looks like eyes-closed/looking away, not a blink. Sits in the middle of the ~500-700ms range real-world blink-detection tolerance suggests -- generous enough for a real ~150-300ms blink plus this pipeline's ~10-15Hz sampling/tracking latency, short enough to firmly exclude a deliberate multi-second eyes-closed pause or sustained occlusion. This cap is now the PRIMARY safeguard against miscounting sustained no-face/occlusion as a blink (see update()'s None-handling below) -- not tuned to one face.
BLINK_REFRACTORY_SECONDS = 0.15      # after a confirmed blink, suppress a NEW "closing" entry for this long -- debounces one physical blink into being counted exactly once even if tracking noise wobbles right at the close/reopen boundary while settling back to steady "open". Short enough to not suppress a genuine rapid double-blink (physiologically those can be a few hundred ms apart).
# BLINK_APERTURE_PLAUSIBLE_MIN/MAX -- a sanity range on the raw aperture
# ratio, NOT the close/reopen logic (that's fraction-of-baseline, see
# above). This user's real data shows full closure typically reads as
# None (tracking lost) rather than a very small positive number, which
# is now handled explicitly in update() rather than by this floor --
# see the None-tolerance comment there. 0.01 remains a reasonable safety
# net for a genuinely degenerate near-zero reading (corrupted landmarks,
# not a closed eye) on setups where closure DOES read as a tiny number
# instead of None.
BLINK_APERTURE_PLAUSIBLE_MIN = 0.01
BLINK_APERTURE_PLAUSIBLE_MAX = 0.6
BLINK_RATE_WINDOW_SECONDS = 30.0     # rolling window the reported rate is computed over
BLINK_MIN_OBSERVATION_SECONDS = 20.0  # below this much reliable observation time, report "measuring" instead of a number -- a rate computed from a handful of seconds swings wildly per additional blink
BLINK_JUST_BLINKED_FLASH_SECONDS = 0.3  # how long snapshot()'s "just_blinked" diagnostic flag stays True after a confirmed blink -- purely a UI-flash duration, not part of detection logic


class BlinkDetector:
    """EXPERIMENTAL / UNVALIDATED. Simple relative-threshold state
    machine (open -> closing -> confirmed-or-abandoned), not a trained
    classifier -- deliberately coarse, matching this feature's own
    "simple, honest, not fabricated" brief. Isolated: never touches
    NeutralCalibrator, WindowAccumulator, or any affect vector; feeds
    nothing into Valence/Arousal.

    update(aperture, now) must be called every processing cycle,
    including with aperture=None when no face is detected.

    None-HANDLING, REVISED against real evidence: an EARLIER version
    treated any None/implausible reading as an immediate abort of an
    in-progress "closing" state. Real logged data showed the deepest
    part of a genuine blink frequently reads as None (tracking loses the
    eye at full closure) -- aborting on that discarded real blinks
    before they could ever be confirmed, which was BUG 2 behind the
    count staying 0 (alongside BUG 1, BLINK_CLOSE_FRACTION being far too
    strict -- see its own comment above).

    Now: a None/implausible reading while OPEN does nothing (there is no
    closing episode to abort or start -- it is simply "no reliable data
    this cycle"). A None/implausible reading while CLOSING is TOLERATED
    -- the state simply stays "closing" and keeps waiting for a plausible
    reopen reading, bounded ONLY by BLINK_MAX_DURATION_SECONDS (measured
    in real wall-clock time since the closing episode started, so a
    None-gap's duration counts against that cap same as a low-but-
    plausible reading would). This is what correctly tells apart a SHORT
    tracking-loss gap in the middle of a real blink (confirmed once
    tracking returns and recovers, well inside the cap) from a LONG
    None run -- occlusion, looking away, the person leaving -- which
    still gets abandoned, never confirmed, once duration exceeds the cap.
    Phantom blinks are still impossible: nothing is ever counted from a
    None reading itself, only from a real recovery-above-baseline
    reading that arrives before the cap expires.

    A BLINK_REFRACTORY_SECONDS debounce after each confirmed blink
    prevents a single physical blink's own tracking noise (settling back
    toward "open") from immediately re-triggering a second false count.

    Returns True the one time a blink is confirmed on that call, else
    False -- callers that want a per-window blink COUNT (for logging)
    should tally these return values themselves rather than reaching
    into this class's internals.

    snapshot(now) reports the rolling rate PLUS diagnostic fields
    (last_aperture, current_state, rolling_max, just_blinked) added
    specifically so a human can watch real aperture/state live and
    verify detection against real behavior, instead of trusting a
    tuned-in-the-dark threshold. Returns measuring=True (rate_per_min=
    None) until BLINK_MIN_OBSERVATION_SECONDS of reliable observation
    have accumulated -- never a number computed from too little data.
    """

    def __init__(self):
        self._recent = deque()       # (t, aperture) pairs, trimmed to BLINK_ROLLING_MAX_SECONDS -- this detector's own rolling "eyes open" reference
        self._state = "open"         # or "closing"
        self._closing_start_t = None
        self._blink_times = deque()  # confirmed-blink timestamps, trimmed to BLINK_RATE_WINDOW_SECONDS
        self._first_observation_t = None
        self._last_aperture = None          # diagnostic only -- most recent raw aperture seen, live, never carried-forward-stale (None means "no current reading", not "unknown")
        self._last_blink_confirmed_t = None  # drives BOTH snapshot()'s transient just_blinked flag AND the refractory debounce above

    def update(self, aperture, now):
        self._last_aperture = aperture  # diagnostic: reflects the RAW value every call, even None/implausible, so a human watching can see exactly what the detector saw
        plausible = aperture is not None and BLINK_APERTURE_PLAUSIBLE_MIN <= aperture <= BLINK_APERTURE_PLAUSIBLE_MAX

        if plausible:
            if self._first_observation_t is None:
                self._first_observation_t = now
            self._recent.append((now, aperture))
            while self._recent and now - self._recent[0][0] > BLINK_ROLLING_MAX_SECONDS:
                self._recent.popleft()
        rolling_max = max((a for _, a in self._recent), default=None)

        blink_confirmed = False
        if self._state == "open":
            # A None/implausible reading here is simply "no data this
            # cycle" -- nothing to start. Only a real, plausible dip
            # below the close threshold begins a closing episode.
            if plausible and rolling_max is not None and rolling_max > 1e-6 and aperture < BLINK_CLOSE_FRACTION * rolling_max:
                in_refractory = self._last_blink_confirmed_t is not None and (now - self._last_blink_confirmed_t) < BLINK_REFRACTORY_SECONDS
                if not in_refractory:
                    self._state = "closing"
                    self._closing_start_t = now
        else:  # closing
            duration = now - self._closing_start_t
            if duration > BLINK_MAX_DURATION_SECONDS:
                # Been "closing" too long -- whether that time was spent
                # at a low-but-plausible reading or a None tracking-loss
                # gap (or both), this now looks like sustained occlusion/
                # eyes-closed/looking away, not a blink. Checked BEFORE
                # the reopen check below on purpose: a late reopen after
                # already exceeding the cap should still NOT count.
                self._state = "open"
                self._closing_start_t = None
            elif plausible and rolling_max is not None and rolling_max > 1e-6 and aperture >= BLINK_REOPEN_FRACTION * rolling_max:
                if duration >= BLINK_MIN_DURATION_SECONDS:
                    self._blink_times.append(now)
                    self._last_blink_confirmed_t = now
                    blink_confirmed = True
                # else: reopened suspiciously fast to be a real blink --
                # more likely a single noisy sample; not counted
                self._state = "open"
                self._closing_start_t = None
            # else: still below the recovery bar, or this sample is
            # None/implausible (tracking still lost mid-blink) -- remain
            # "closing" and keep waiting, bounded by the max-duration
            # check above on the NEXT call.

        while self._blink_times and now - self._blink_times[0] > BLINK_RATE_WINDOW_SECONDS:
            self._blink_times.popleft()
        return blink_confirmed

    def snapshot(self, now):
        just_blinked = self._last_blink_confirmed_t is not None and (now - self._last_blink_confirmed_t) < BLINK_JUST_BLINKED_FLASH_SECONDS
        rolling_max = max((a for _, a in self._recent), default=None)
        # float(...): aperture values originate from a numpy-array-backed
        # computation (compute_v_es's ear()), so these can be numpy
        # scalars -- cast to native Python here, same discipline
        # WindowAccumulator._stats already uses, so this can never trip
        # up json.dumps() when logged.
        diagnostic = {
            "last_aperture": float(self._last_aperture) if self._last_aperture is not None else None,
            "current_state": self._state,
            "rolling_max": float(rolling_max) if rolling_max is not None else None,
            "just_blinked": just_blinked,
        }
        if self._first_observation_t is None or (now - self._first_observation_t) < BLINK_MIN_OBSERVATION_SECONDS:
            return {"rate_per_min": None, "measuring": True, "blinks_in_rate_window": len(self._blink_times), "unvalidated": True, **diagnostic}
        elapsed = min(now - self._first_observation_t, BLINK_RATE_WINDOW_SECONDS)
        rate = (len(self._blink_times) / elapsed) * 60.0 if elapsed > 0 else None
        return {"rate_per_min": rate, "measuring": False, "blinks_in_rate_window": len(self._blink_times), "unvalidated": True, **diagnostic}


class AttentionWindowAccumulator:
    """Same rolling-window PATTERN as episodes.WindowAccumulator (tumbling
    WINDOW_SECONDS window, avg+peak+variance, MANDATORY ARCHITECTURE
    #5's cadence) -- but a fully INDEPENDENT object, not a new bucket
    bolted onto WindowAccumulator itself. Two reasons:

    1. V_so needs NO per-person calibration (module docstring) -- it
       must accumulate from frame 1, regardless of calibrator.is_calibrated(),
       so it cannot share WindowAccumulator's calibration-gated lifecycle.
    2. It keeps this new, UNVALIDATED signal from ever touching the
       existing (Gate-2-tested) V_bf/V_es/V_pd windowing code at all --
       nothing here can regress it. (D1 restates this as a hard rule,
       not just a design preference: this class must never import
       features.x_core or features.episodes -- see module docstring.)

    add_sample() takes a single per-sample dict rather than the
    composite/covariate split WindowAccumulator uses, because V_so has
    no composite-vs-covariate distinction to make (Decision 17/18's
    "which vectors feed Valence/Arousal" split doesn't apply -- V_so
    feeds neither, ever)."""

    def __init__(self):
        self.window_start = None
        self.samples = []

    def add_sample(self, ts, sample):
        if self.window_start is None:
            self.window_start = ts
        self.samples.append(sample)

    def should_flush(self, now, window_seconds=WINDOW_SECONDS):
        return self.window_start is not None and (now - self.window_start) >= window_seconds

    def flush(self, now):
        """Returns the window summary as THREE separate, honestly-labeled
        signals (validation-prep requirement) -- each independently
        scoreable against a commanded direction, exactly like Gate 2
        scored V_bf/V_es/V_pd per-vector rather than as one blended
        number. No new computation: every number below is derived from
        the same per-sample orientation_score/oriented/gaze_score/
        gaze_reliable values compute_v_so already produces each cycle --
        this method only reshapes and aggregates them.

          screen_orientation -- avg/peak/variance of V_so's 0..1 score
            + oriented_rate (the stat a human scorer compares against
            "look at screen" vs "look away"/"look down").
          gaze_direction -- avg/peak/variance of the gaze score, computed
            ONLY over samples where gaze was geometrically plausible
            (never averages in a corrupted/unreliable reading), plus
            gaze_reliable_rate so a scorer can exclude low-reliability
            windows the same way Gate 2 excludes flagged baselines.
          look_away_rate -- 1 - oriented_rate, computed directly from the
            same oriented flags (not re-derived from the rounded
            oriented_rate above, so there's no double-rounding drift) --
            an honestly-labeled "distraction proxy", never "distraction"
            or "disengagement" itself.
        """
        window_start = self.window_start
        samples = self.samples
        n_samples = len(samples)
        n_detected = sum(1 for s in samples if s["detected"])
        detection_rate = (n_detected / n_samples) if n_samples else 0.0

        scores = [s["orientation_score"] for s in samples if s["orientation_score"] is not None]
        oriented_flags = [s["oriented"] for s in samples if s["oriented"] is not None]
        gaze_scores = [s["gaze_score"] for s in samples if s["gaze_score"] is not None]
        gaze_reliable_flags = [s["gaze_reliable"] for s in samples if s["gaze_reliable"] is not None]

        oriented_rate = (sum(oriented_flags) / len(oriented_flags)) if oriented_flags else None

        summary = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "attention_window_summary",
            "session_id": SESSION_ID,
            "person_label": PERSON_LABEL,
            "window_start_monotonic": window_start,
            "window_end_monotonic": now,
            "window_seconds": now - window_start,
            "n_samples": n_samples,
            "n_detected": n_detected,
            "detection_rate": detection_rate,
            # SIGNAL 1/3 -- head/eyes oriented toward screen (V_so itself).
            "screen_orientation": {
                "avg": float(np.mean(scores)) if scores else None,
                "peak": float(np.max(scores)) if scores else None,
                "variance": float(np.var(scores)) if scores else None,
                "n": len(scores),
                # The stat a Gate-2-style human scorer actually compares
                # against the commanded label ("look at screen" -> expect
                # high, "look away"/"look down" -> expect low).
                "oriented_rate": oriented_rate,
                "unvalidated": True,
                "label": "screen orientation (geometric) -- NOT attention/engagement (mental state)",
            },
            # SIGNAL 2/3 -- bonus refinement, ONLY when geometrically
            # plausible. gaze_reliable_rate ALWAYS travels with it so a
            # scorer can exclude low-reliability windows, same spirit as
            # Gate 2's low-confidence exclusion -- never trust gaze
            # silently (module docstring's GLASSES-ROBUST section).
            "gaze_direction": {
                "avg": float(np.mean(gaze_scores)) if gaze_scores else None,
                "peak": float(np.max(gaze_scores)) if gaze_scores else None,
                "variance": float(np.var(gaze_scores)) if gaze_scores else None,
                "n": len(gaze_scores),
                "gaze_reliable_rate": (sum(gaze_reliable_flags) / len(gaze_reliable_flags)) if gaze_reliable_flags else None,
                "unvalidated": True,
                "label": "gaze direction (when reliable) -- bonus signal, NOT a mental-state read; ignore avg/peak/variance when gaze_reliable_rate is low",
            },
            # SIGNAL 3/3 -- fraction of the window NOT oriented.
            "look_away_rate": {
                "value": (1.0 - oriented_rate) if oriented_rate is not None else None,
                "unvalidated": True,
                "label": "look-away rate (geometric) -- fraction of window not oriented toward screen; NOT distraction/disengagement",
            },
            "unvalidated": True,
            "label": "orientation signals (geometric) -- NOT attention/engagement/focus/distraction (mental state); UNVALIDATED pending cross-person testing",
        }
        self.window_start = None
        self.samples = []
        return summary
