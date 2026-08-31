"""
Refactor safety net (D0PA1 Batch 1, Step 1). Feeds FIXED, SEEDED SYNTHETIC
landmark inputs -- never real footage -- through every function on the
validated path (G5: ear()/the affect vector formulas, per-person
calibration, the rolling window, the V/A mapping) and records the outputs
to a committed golden file.

USAGE:
  Before the refactor (already done, golden file committed):
    python tests/test_refactor_snapshot.py --write-golden

  After the refactor (and any time this file runs in a test suite):
    python -m pytest tests/test_refactor_snapshot.py -v
    (or: python tests/test_refactor_snapshot.py)

  A mismatch means the validated path moved during the refactor. Per G5
  and this task's Step 1.3: that must be REVERTED and investigated, never
  accepted, no matter how small the difference looks.

WHAT THIS DOES NOT COVER, stated explicitly rather than skipped silently:
  - capture_thread() / processing_thread() (the two-thread loop itself,
    camera I/O, MediaPipe detection calls): these require a live camera
    or real MediaPipe Tasks objects and are integration-level, not a pure
    function of landmark inputs -- no synthetic fixture can exercise them
    without either a real camera or a real .task model file feeding real
    detection output, and this test is deliberately about the downstream
    MATH, not detection. FPS/threading behavior has its own coverage
    (logs/soak_log.jsonl).
  - draw_overlay() / draw_va_plot() / main(): rendering and orchestration,
    not feature computation -- nothing in G5's validated-path list lives
    there.
  - classify_calibration_quality() / classify_window_confidence(): these
    ARE exercised indirectly (NeutralCalibrator.complete() and
    WindowAccumulator.flush() call them internally and their output is
    captured), but not asserted on in isolation with dedicated synthetic
    edge cases -- this snapshot is a byte-for-byte regression net, not a
    correctness test suite for those two functions specifically.
"""

import copy
import hashlib
import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np

import stage1_step4_vectors as s1

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "refactor_snapshot_golden.json")


# ============================================================
# SYNTHETIC FIXTURE -- deterministic, seeded, NOT real footage.
# ============================================================

class _Landmark:
    """Minimal stand-in for mediapipe's NormalizedLandmark -- pose_normalize()
    only ever reads .x/.y/.z off each entry, so a plain object with those
    three attributes is a faithful, dependency-free substitute."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def build_synthetic_face_landmarks():
    """478 landmarks (matching FaceLandmarker's real 468 mesh + 10 iris
    shape). Every index the affect-vector formulas actually reference
    (IRIS_*, BROW_*, GLABELLA, EYE_*, LIP_*, JAW_*) is hand-placed at an
    anatomically-plausible relative position so the geometric ratios are
    non-degenerate and meaningful, not just non-crashing. Every other
    index is filled by a fixed-seed deterministic generator (small cloud
    around the face center) so pose_normalize()'s centroid computation
    (which averages ALL 478 points) is realistic without needing 478
    hand-placed points.
    """
    rng = np.random.default_rng(20260824)  # fixed seed -- reproducibility, not realism, is the point
    n = 478
    xs = 0.5 + rng.normal(0, 0.12, n)
    ys = 0.5 + rng.normal(0, 0.15, n)
    zs = rng.normal(0, 0.01, n)

    fixed = {
        s1.IRIS_RIGHT_CENTER: (0.400, 0.450, 0.000),
        s1.IRIS_LEFT_CENTER: (0.600, 0.450, 0.000),
        s1.BROW_INNER_R: (0.450, 0.400, -0.005),
        s1.BROW_INNER_L: (0.550, 0.400, -0.005),
        s1.GLABELLA: (0.500, 0.405, -0.008),
        s1.EYE_R_OUTER: (0.350, 0.450, 0.000),
        s1.EYE_R_UPPER1: (0.380, 0.442, 0.000),
        s1.EYE_R_UPPER2: (0.420, 0.442, 0.000),
        s1.EYE_R_INNER: (0.450, 0.450, 0.000),
        s1.EYE_R_LOWER1: (0.420, 0.458, 0.000),
        s1.EYE_R_LOWER2: (0.380, 0.458, 0.000),
        s1.EYE_L_OUTER: (0.650, 0.450, 0.000),
        s1.EYE_L_UPPER1: (0.620, 0.442, 0.000),
        s1.EYE_L_UPPER2: (0.580, 0.442, 0.000),
        s1.EYE_L_INNER: (0.550, 0.450, 0.000),
        s1.EYE_L_LOWER1: (0.580, 0.458, 0.000),
        s1.EYE_L_LOWER2: (0.620, 0.458, 0.000),
        s1.LIP_UPPER_INNER: (0.500, 0.600, -0.003),
        s1.LIP_LOWER_INNER: (0.500, 0.630, -0.003),
        s1.LIP_CORNER_R: (0.420, 0.615, -0.002),
        s1.LIP_CORNER_L: (0.580, 0.615, -0.002),
        s1.JAW_ANGLE_R: (0.300, 0.650, 0.010),
        s1.JAW_ANGLE_L: (0.700, 0.650, 0.010),
    }

    landmarks = []
    for i in range(n):
        if i in fixed:
            x, y, z = fixed[i]
        else:
            x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
        landmarks.append(_Landmark(x, y, z))
    return landmarks


def build_synthetic_transform_matrix():
    """A fixed, small, deterministic rotation (10deg yaw, 4deg pitch,
    2deg roll) plus a fixed translation, in the 4x4 homogeneous form
    output_facial_transformation_matrixes provides. Exercises
    yaw_pitch_roll_from_matrix and pose_normalize's R^T un-rotation with
    a genuinely non-identity, non-degenerate rotation."""
    yaw, pitch, roll = np.radians(10.0), np.radians(4.0), np.radians(2.0)

    cy, sy = np.cos(yaw), np.sin(yaw)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    cp, sp = np.cos(pitch), np.sin(pitch)
    rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    cr, sr = np.cos(roll), np.sin(roll)
    rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

    r = ry @ rp @ rr
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = [0.01, -0.02, -0.5]
    return m


def build_synthetic_pose_world_sequence():
    """A short deterministic sequence of (nose, shoulder_mid) world
    positions in meters, spanning ~1.2s -- enough for V_pd's rolling
    buffer (PD_BUFFER_SECONDS=1.5) to hold >=2 samples and produce a real
    variance-based reading, not a None (buffer_len<2) result."""
    ts = [0.0, 0.3, 0.6, 0.9, 1.2]
    samples = []
    for i, t in enumerate(ts):
        jitter = 0.002 * np.sin(i * 1.7)
        nose = np.array([0.00 + jitter, 0.10 + 0.5 * jitter, -0.30])
        shoulder_mid = np.array([0.00 - jitter, -0.15 + 0.3 * jitter, -0.35])
        samples.append((t, nose, shoulder_mid))
    return samples


def build_synthetic_calibration_samples():
    """~30 deterministic samples spanning ts=0..26s (just past
    CALIBRATION_SECONDS=25.0, reused unchanged -- not reinvented) with a
    small fixed sinusoidal wobble per vector, so NeutralCalibrator sees
    real (non-zero) mean AND std -- exercising the std branch, not just
    the degenerate all-identical-value case."""
    samples = []
    for i in range(27):
        t = float(i)
        wobble = 0.01 * np.sin(i * 0.9)
        composite = {
            "v_bf": -0.30 + wobble,
            "v_es": -0.25 + 0.6 * wobble,
            "v_pd": 0.0004 + 0.1 * wobble * 0.0004,
        }
        covariate = {
            "v_jc": 0.55 + 0.4 * wobble,
            "v_bf_convergence_ratio": 0.18 + 0.3 * wobble,
            "v_es_cheek_raise": 0.40 + 0.5 * wobble,
        }
        yaw_deg = 1.5 * np.sin(i * 0.5)
        samples.append((t, composite, covariate, yaw_deg))
    return samples


def build_synthetic_window_samples():
    """~11 deterministic samples spanning ts=0..10.5s (just past
    WINDOW_SECONDS=10.0), each with detected=True and a small
    deterministic wobble, for WindowAccumulator.flush()."""
    samples = []
    for i in range(11):
        t = float(i)
        wobble = 0.02 * np.cos(i * 1.1)
        composite = {"v_bf": -0.31 + wobble, "v_es": -0.24 + 0.5 * wobble, "v_pd": 0.00045 + 0.05 * wobble * 0.0004}
        covariate = {"v_jc": 0.56 + 0.3 * wobble, "v_bf_convergence_ratio": 0.19 + 0.2 * wobble, "v_es_cheek_raise": 0.41 + 0.4 * wobble}
        yaw_deg = 2.0 * np.cos(i * 0.4)
        samples.append((t, True, yaw_deg, composite, covariate))
    return samples


# ============================================================
# RUN EVERY FUNCTION ON THE VALIDATED PATH, CAPTURE OUTPUTS
# ============================================================

def _floatify(obj):
    """json.dumps can't handle numpy scalars/arrays -- convert
    recursively. Not a computation, purely a serialization shim."""
    if isinstance(obj, dict):
        return {k: _floatify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_floatify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _floatify(obj.tolist())
    return obj


def run_snapshot():
    out = {}

    lms = build_synthetic_face_landmarks()
    matrix = build_synthetic_transform_matrix()
    img_w, img_h = 640, 480

    normalized_pts = s1.pose_normalize(lms, matrix, img_w, img_h)
    out["pose_normalize_sample_points"] = {
        "point_0": normalized_pts[0].tolist(),
        "point_100": normalized_pts[100].tolist(),
        "point_IRIS_RIGHT_CENTER": normalized_pts[s1.IRIS_RIGHT_CENTER].tolist(),
        "point_JAW_ANGLE_L": normalized_pts[s1.JAW_ANGLE_L].tolist(),
        "centroid_check_sum": float(normalized_pts.sum()),
    }

    io_dist = s1.interocular_distance(normalized_pts)
    out["interocular_distance"] = io_dist

    yaw, pitch, roll = s1.yaw_pitch_roll_from_matrix(matrix)
    out["yaw_pitch_roll_from_matrix"] = {"yaw": yaw, "pitch": pitch, "roll": roll}

    v_bf, bf_components = s1.compute_v_bf(normalized_pts, io_dist)
    out["compute_v_bf"] = {"composite": v_bf, "components": bf_components}

    v_es, es_components = s1.compute_v_es(normalized_pts, io_dist)
    out["compute_v_es"] = {"composite": v_es, "components": es_components}

    v_jc, jc_components = s1.compute_v_jc(normalized_pts, io_dist)
    out["compute_v_jc"] = {"composite": v_jc, "components": jc_components}

    pd_buffer = deque()
    pd_results = []
    for t, nose, shoulder_mid in build_synthetic_pose_world_sequence():
        v_pd, pd_components = s1.compute_v_pd(pd_buffer, nose, shoulder_mid, t)
        pd_results.append({"t": t, "v_pd": v_pd, "components": pd_components})
    out["compute_v_pd_sequence"] = pd_results

    out["_z_score_direct"] = {
        "normal_case": s1._z_score(0.5, 0.25),
        "std_zero": s1._z_score(0.5, 0.0),
        "deviation_none": s1._z_score(None, 0.25),
    }

    calibrator = s1.NeutralCalibrator()
    cal_samples = build_synthetic_calibration_samples()
    for t, composite, covariate, yaw_deg in cal_samples:
        calibrator.add_sample(t, composite, covariate, yaw_deg)
    assert calibrator.should_complete(cal_samples[-1][0]), "fixture must span >= CALIBRATION_SECONDS"
    reference = calibrator.complete(cal_samples[-1][0])
    out["calibrator_reference"] = reference

    test_deviation_bf = calibrator.deviation("v_bf", -0.28)
    test_deviation_es = calibrator.deviation("v_es", -0.20)
    test_deviation_pd = calibrator.deviation("v_pd", 0.00046)
    out["calibrator_deviation"] = {"v_bf": test_deviation_bf, "v_es": test_deviation_es, "v_pd": test_deviation_pd}

    va_point = s1.map_to_valence_arousal(test_deviation_bf, test_deviation_es, test_deviation_pd, reference)
    out["map_to_valence_arousal"] = va_point

    window_acc = s1.WindowAccumulator()
    for t, detected, yaw_deg, composite, covariate in build_synthetic_window_samples():
        window_acc.add_sample(t, detected, yaw_deg, composite, covariate)
    win_samples = build_synthetic_window_samples()
    assert window_acc.should_flush(win_samples[-1][0]), "fixture must span >= WINDOW_SECONDS"
    window_summary = window_acc.flush(win_samples[-1][0])
    # session_id/person_label/ts are run-identity, not formula output -- excluded
    # from the golden comparison so the snapshot only asserts on the MATH.
    window_summary.pop("session_id", None)
    window_summary.pop("person_label", None)
    out["window_summary"] = window_summary

    return _floatify(out)


def _canonical_json(data):
    return json.dumps(data, indent=2, sort_keys=True)


def write_golden():
    data = run_snapshot()
    text = _canonical_json(data)
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"Wrote golden file: {GOLDEN_PATH}")
    print(f"SHA256: {digest}")
    return digest


def test_snapshot_matches_golden():
    if not os.path.exists(GOLDEN_PATH):
        raise AssertionError(
            f"No golden file at {GOLDEN_PATH} -- run `python tests/test_refactor_snapshot.py --write-golden` "
            "BEFORE any refactor, and commit the result. There is nothing to compare against yet."
        )
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_text = f.read()
    fresh_text = _canonical_json(run_snapshot())
    assert fresh_text == golden_text, (
        "SNAPSHOT MISMATCH -- the validated path (G5) produced a DIFFERENT result "
        "than the committed golden file. Per this task's Step 1.3, this must be "
        "REVERTED and investigated, not accepted, no matter how small the diff."
    )


if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        write_golden()
    else:
        test_snapshot_matches_golden()
        print("PASS: snapshot matches golden file exactly.")
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            digest = hashlib.sha256(f.read().encode("utf-8")).hexdigest()
        print(f"Golden file SHA256: {digest}")
