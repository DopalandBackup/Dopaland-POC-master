"""
Shared raw-landmark utilities (D1 feature-block separation, CLAUDE.md).

UPSTREAM of all five feature blocks (x_core, episodes, attention, audio,
context) and belongs to NONE of them -- any block may import this module.
Moved verbatim out of stage1_step4_vectors.py; no formula, threshold, or
landmark index changed (G5 -- this task must not alter the validated path,
only relocate it. See tests/test_refactor_snapshot.py for the byte-exact
regression net that proves this).

CONTENTS:
  1. Raw-landmark math primitives: apply_clahe, yaw_pitch_roll_from_matrix,
     pose_normalize, _dist, interocular_distance.
  2. Verified landmark indices (face_mesh_connections.py contour sets).
  3. TWO constants that are genuinely used by more than one feature block
     today (WINDOW_SECONDS, YAW_VARIANCE_CEILING_DEG2) -- see the honesty
     note at their definitions below. These are NOT raw-landmark utilities;
     they live here only because geometry.py is the one module every block
     may import, and duplicating a shared constant across two block modules
     would create two sources of truth for the same number.

HONESTY NOTE (mirrors the manifest): _dist and interocular_distance are
domain-free primitives that COULD be called by any block, but as of this
refactor are actually called only by X_core functions (compute_v_bf,
compute_v_es, compute_v_jc; interocular_distance is called from the two
processing-loop call sites, not from inside a feature function itself).
"Could be shared" and "is currently shared" are different claims -- see
features/manifests/x_core_v1.json.
"""

import cv2
import numpy as np

# --- Raw-landmark math primitives (verbatim from stage1_step4_vectors.py) ---


def apply_clahe(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)


def yaw_pitch_roll_from_matrix(matrix_4x4):
    r = np.asarray(matrix_4x4)[:3, :3]
    pitch = np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)))
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    roll = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    return float(yaw), float(pitch), float(roll)


def pose_normalize(face_landmarks, transform_matrix, img_w, img_h):
    """
    Undo head rotation on face landmarks, in a single self-consistent
    coordinate system (isotropic pixel-like units), NOT by mixing
    image-normalized landmarks with the transform matrix's metric
    canonical-model translation (those are different coordinate systems).

    1. Aspect-correct to isotropic units: X=x*W, Y=y*H, Z=z*W (z already
       shares x's normalization base per MediaPipe convention).
    2. Center on the landmark centroid — self-consistent, unlike the
       matrix's translation which lives in the canonical model's frame.
    3. Apply R^T to undo rotation. R is confirmed orthonormal (verified
       empirically: R @ R^T ~= I), so R^T = R^-1 exactly — a clean,
       distortion-free un-rotation.
    """
    pts = np.array([[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in face_landmarks])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    r = np.asarray(transform_matrix)[:3, :3]
    return (r.T @ centered.T).T


def _dist(pts, i, j):
    return float(np.linalg.norm(pts[i] - pts[j]))


def interocular_distance(pts):
    return _dist(pts, IRIS_LEFT_CENTER, IRIS_RIGHT_CENTER)


# --- Verified landmark indices (face_mesh_connections.py contour sets) ---
# IRIS_LEFT_CENTER/IRIS_RIGHT_CENTER: FIXED -- were swapped relative to their
# names (468/473 were assigned backwards) from Stage 1 until this fix. Found
# and root-caused during the attention-signal (V_so) build, confirmed
# empirically against real footage (pose-normalized coordinates), documented
# in IRIS_SWAP_DIAGNOSTIC.md. Verdict there was CONTAINED: only the logged-
# only cheek_raise covariate (compute_v_es) used these constants -- V_es's
# SCORED aperture/composite (via ear()) never referenced them, so the Gate-2
# 71.4% result was never affected and needs no re-scoring. This fix makes
# cheek_raise's own pairing correct going forward; it stays logged-only/
# excluded from Valence regardless (Decision 25) -- fixing the pairing does
# not change that status.
#
# Used by X_core (compute_v_es's aperture/cheek_raise) AND by A_t
# (_gaze_centering_score, compute_v_so, compute_gaze_direction) -- genuinely
# shared today, not just "could be shared". See features/manifests/*.json.
IRIS_LEFT_CENTER = 473
IRIS_RIGHT_CENTER = 468
BROW_INNER_R = 55       # member of FACEMESH_RIGHT_EYEBROW -- X_core only (compute_v_bf)
BROW_INNER_L = 285      # member of FACEMESH_LEFT_EYEBROW -- X_core only (compute_v_bf)
GLABELLA = 168          # member of FACEMESH_NOSE; between-eyebrows/nose-bridge point -- X_core only (compute_v_bf)
EYE_R_OUTER, EYE_R_UPPER1, EYE_R_UPPER2, EYE_R_INNER, EYE_R_LOWER1, EYE_R_LOWER2 = 33, 160, 158, 133, 153, 144
EYE_L_OUTER, EYE_L_UPPER1, EYE_L_UPPER2, EYE_L_INNER, EYE_L_LOWER1, EYE_L_LOWER2 = 263, 385, 387, 362, 373, 380
# EYE_*_OUTER/INNER: shared, same as IRIS_* above (X_core's compute_v_es
# aperture/cheek_raise; A_t's _gaze_centering_score/compute_v_so/
# compute_gaze_direction). EYE_*_UPPER1/UPPER2/LOWER1/LOWER2: X_core only
# (compute_v_es's ear() aperture term).
LIP_UPPER_INNER = 13    # X_core only (compute_v_jc)
LIP_LOWER_INNER = 14    # X_core only (compute_v_jc)
LIP_CORNER_R = 61       # X_core only (compute_v_jc; also compute_v_es's logged-only cheek_raise)
LIP_CORNER_L = 291      # X_core only (compute_v_jc; also compute_v_es's logged-only cheek_raise)
JAW_ANGLE_R = 172        # FACEMESH_FACE_OVAL, jaw/masseter region -- X_core only (compute_v_jc)
JAW_ANGLE_L = 397        # FACEMESH_FACE_OVAL, jaw/masseter region -- X_core only (compute_v_jc)

# BlazePose topology (stable across mediapipe versions, not re-derived here)
# -- X_core only (compute_v_pd).
POSE_NOSE = 0
POSE_SHOULDER_L = 11
POSE_SHOULDER_R = 12

# --- Constants shared by more than one feature block (see module docstring) ---

# Used by episodes.py's WindowAccumulator (E_t, avg/peak/variance cadence)
# AND by attention.py's AttentionWindowAccumulator (A_t, its own INDEPENDENT
# clock -- see that class's docstring). Same numeric value by deliberate
# design choice (both tumbling windows tick at the same cadence), not because
# one depends on the other; the two accumulators never share state or read
# each other's output. A single definition here is what keeps that "same
# cadence" fact from silently drifting into two different numbers.
WINDOW_SECONDS = 10.0  # Decision 12 / MANDATORY ARCHITECTURE #5: 10s avg+peak+variance

# Used by episodes.py's classify_window_confidence (E_t, window-validity
# gate) AND by x_core.py's classify_calibration_quality (X_core, in-capture
# contamination flag) -- both reuse the SAME per-window/per-capture head-
# movement ceiling rather than each defining their own (see
# classify_calibration_quality's own docstring: "reuses the SAME yaw_variance
# ceiling already established for the window-validity gate (not a new
# number)"). E_t importing a constant from geometry.py rather than from
# x_core.py directly is a deliberate choice: X_core -> E_t is a permitted
# direction, but this constant is not X_core CONTENT (it is not an affect
# vector or a calibration output) -- it is a session-quality threshold that
# predates and is orthogonal to both consumers, so it lives in the neutral,
# upstream module rather than being "owned" by either block.
YAW_VARIANCE_CEILING_DEG2 = 100.0  # yaw std ~10deg -- roughly a third of the existing 35deg per-frame gate
