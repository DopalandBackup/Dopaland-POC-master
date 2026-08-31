"""
Stage 1, steps 4+5, plus Stage 1.5 step 8 — 4 biometric vectors,
geometric, pose-normalized 3D (CLAUDE.md Decision 11 + MANDATORY
ARCHITECTURE #4); the 10s rolling window (avg/peak/variance, MANDATORY
ARCHITECTURE #5) with a window-validity gate; and per-person
within-session neutral calibration (Decision #5 / Pitfall #2 /
MANDATORY ARCHITECTURE #6), built deliberately BEFORE step 6's V/A
mapping so that mapping runs on calibrated (deviation-from-neutral)
vectors from the start, not raw magnitudes that would need redoing.

New file (not an edit of stage0_skeleton.py) so Stage 0's Gate-1 artifact
stays intact and reviewable on its own. Thread 1 (capture) is copied over
unchanged. Thread 2 changes in two ways from Stage 0:

1. Continuous sampling, not a 10s tick (CADENCE clarification — not new
   scope): the "every 10s" rule governs the summary/agent/DB step, which
   doesn't exist until Stage 2. Thread 2 loops at detection speed and
   logs every sample ("log per-frame", Decision 12) — required because
   V_pd is itself defined as a variance over a rolling ~1-2s buffer, and
   step 5's 10s window needs a real sample stream to summarize.
2. RunningMode.VIDEO instead of IMAGE: IMAGE mode has no inter-frame
   tracking, so min_tracking_confidence would be silently meaningless.
   VIDEO mode (detect_for_video, monotonically increasing ms timestamps)
   makes tracking — and therefore that confidence floor — real.

Step 4 validation found our facial vectors (V_bf, V_es) are direction-
reliable but magnitude-noisy under head rotation (io_dist, their shared
scale reference, carries residual yaw correlation ~0.25 at full range).
Step 5's window design leans on this finding rather than re-fighting it:
variance-per-window doubles as the confidence signal (a high-variance
window, e.g. mid head-turn, is a low-trust reading) rather than just a
logged stat — see episodes.WindowAccumulator.

Landmark indices are verified against
mediapipe/python/solutions/face_mesh_connections.py (google-ai-edge/
mediapipe, master), not assumed from memory.

=== D0PA1 BATCH 1 STEP 3 REFACTOR NOTE (feature-block separation) ===
The vector formulas, calibration, windowing, attention signals, and V/A
mapping that used to live in this file have MOVED to features/geometry.py,
features/x_core.py, features/episodes.py, and features/attention.py (D1
feature-block separation, CLAUDE.md). This file is now an ORCHESTRATOR: it
owns the two-thread loop, the webcam, MediaPipe setup, and the live
overlay/plot, and imports the actual feature computation from those
modules. No formula, threshold, or behavior changed in the move (G5) --
see tests/test_refactor_snapshot.py for the byte-exact regression net and
docs/D1_DEPENDENCY_MAP.md for the full picture. The design-rationale prose
below (ATTENTION SIGNAL / PILOT STATUS sections) is retained as accurate
HISTORY of why V_so was built the way it was; the code itself now lives in
features/attention.py.

--- ATTENTION SIGNAL, Step 1 (V_so -- screen orientation) ---
A NEW, FIFTH geometric signal, added after Gate 2, using the exact same
discipline the four affect vectors were built with: reuse the existing
head-pose decomposition (yaw_pitch_roll_from_matrix, already computed
every cycle for the quality gate -- not recomputed here), a first-cut
threshold explicitly derived from the existing 35deg per-frame gate (not
invented, not tuned to one face), and an UNVALIDATED stamp on every
logged record until it is tested across real people the same way V_bf/
V_es/V_pd were at Gate 2 (see compute_v_so's docstring for the exact
validation hook this needs).

HONEST FRAMING (as strict as the V/A rules): V_so measures a geometric
fact -- is the head/gaze pointed at the screen -- and is labeled "screen
orientation" everywhere. It NEVER claims "attention" or "engagement":
those are mental-state inferences a camera cannot measure (a person can
be squarely oriented at the screen and mentally elsewhere). Any UI/log/
report text describing V_so must say "orientation", never "attention"/
"engagement".

GLASSES-ROBUST BY CONSTRUCTION: PRIMARY signal is head pose (yaw/pitch),
which glasses do not affect at all. SECONDARY signal is gaze (iris
position within the eye, from the same iris landmarks V_es already
reuses) -- a BONUS refinement applied ONLY when the iris landmarks pass
a geometric plausibility check, NEVER a dependency, because glasses are
exactly what corrupts iris tracking (Pitfall #4) via lens reflections.
If gaze is unusable this frame, V_so falls back to head-pose-only and
says so in its own logged record -- it never fails or goes silent just
because gaze isn't available.

V_so is NOT composited into Valence or Arousal (attention is not
affect) and needs NO per-person calibration (unlike V_bf/V_es/V_pd,
"oriented toward screen" is a universal geometric threshold, same
category as the quality gate's yaw>35deg check, not a deviation-from-
neutral concept) -- so it is measured and logged from frame 1, windowed
on its own independent 10s clock (AttentionWindowAccumulator), never
touching WindowAccumulator or NeutralCalibrator.

VALIDATION PREP: the cluster is exposed as THREE separate, independently
scoreable fields (sample records and window summaries alike), same
discipline as Gate 2 scoring V_bf/V_es/V_pd per-vector rather than as
one blended number -- "screen_orientation" (head/eyes toward screen),
"gaze_direction" (bonus, ALWAYS paired with a gaze_reliable flag so a
scorer can exclude unreliable-gaze windows), and "look_away_rate" (1 -
oriented_rate, an honestly-labeled proxy -- never "distraction" or
"disengagement"). All three carry "unvalidated": True until a Gate-2-
style directed capture ("look at screen"/"look away"/"look down" on
command) scores them across real people. This is a pure reshape of
values compute_v_so already produced -- no new per-frame computation.

--- PILOT STATUS (post-validation finding, documentation only) ---
V_so is a PILOT feature, NOT POC-ready, and is NOT wired into the demo
UI (stage3_demo_ui.py never imports or reads it). Directed testing
found: YAW (left/right) is detected reliably. PITCH (looking up/down)
is UNRELIABLE -- on a maximal, sustained, verified chin-to-chest
look-down, MediaPipe-derived pitch stayed ~0.1deg (indistinguishable
from looking straight at the screen) and oriented_rate stayed 1.0.
Root cause is STRUCTURAL, not a code bug: yaw_pitch_roll_from_matrix
is provably exact on synthetic rotations, but the face foreshortens
when looking down, degrading the landmark data the pitch calculation
depends on -- fixable only by a different sensing approach (gaze
tracking and/or better hardware), not by tuning this formula. DO NOT
attempt to "fix" pitch here. Since the most common disengagement cue
is looking down, a yaw-only reliable signal cannot honestly be
presented as "attention"/"engagement"/"focus"/"distraction" -- doing
so would overclaim, the same dishonesty the pain-axis rule (V_bf)
forbids. A reliably-pitched version is pilot work (see CLAUDE.md's
"Attention / screen-orientation -- PILOT, not POC" note).
"""

import cv2
import json
import mediapipe as mp
import numpy as np
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from features import x_core, episodes, attention
from features.geometry import (
    apply_clahe,
    yaw_pitch_roll_from_matrix,
    pose_normalize,
    interocular_distance,
    POSE_NOSE,
    POSE_SHOULDER_L,
    POSE_SHOULDER_R,
    # Landmark index constants: not referenced by bare name anywhere in this
    # file's own code any more (they moved into features.x_core's function
    # bodies), but re-exported here because tests/test_refactor_snapshot.py
    # (Step 1, already committed, not to be modified by this task) builds
    # its synthetic fixture against s1.IRIS_RIGHT_CENTER etc.
    IRIS_RIGHT_CENTER,
    IRIS_LEFT_CENTER,
    BROW_INNER_R,
    BROW_INNER_L,
    GLABELLA,
    EYE_R_OUTER,
    EYE_R_UPPER1,
    EYE_R_UPPER2,
    EYE_R_INNER,
    EYE_R_LOWER1,
    EYE_R_LOWER2,
    EYE_L_OUTER,
    EYE_L_UPPER1,
    EYE_L_UPPER2,
    EYE_L_INNER,
    EYE_L_LOWER1,
    EYE_L_LOWER2,
    LIP_UPPER_INNER,
    LIP_LOWER_INNER,
    LIP_CORNER_R,
    LIP_CORNER_L,
    JAW_ANGLE_R,
    JAW_ANGLE_L,
    _dist,  # re-exported: stage1_step4_browdiag_session.py imports this by name -- see note above
)
from features.x_core import (
    compute_v_bf,
    compute_v_es,
    compute_v_jc,
    compute_v_pd,
    NeutralCalibrator,
    map_to_valence_arousal,
    CALIBRATION_SECONDS,
    _z_score,  # re-exported for tests/test_refactor_snapshot.py and stage1_step9_gate2_capture.py -- see note above
)
from features.episodes import WindowAccumulator, classify_window_confidence
from features.attention import compute_v_so, AttentionWindowAccumulator

CAMERA_INDEX = 0
FPS_REPORT_INTERVAL_SECONDS = 3.0
CONFIDENCE_THRESHOLD = 0.7  # "detected" means "cleared 0.7", per instruction

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_full.task")

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SCHEMA_VERSION = "1.6"  # 1.3 added va_point (step 6, untested hypothesis) to sample records;
                         # 1.4: va_point now uses the Decision 18 mapping (Valence=z_es only) --
                         # the retired (z_es-z_bf)/2 formula no longer appears. Historical records
                         # at 1.3 keep their old values; only new records use the new mapping.
                         # 1.5: ATTENTION SIGNAL Step 1 -- adds "screen_orientation" to sample
                         # records (new field, existing fields untouched) and a new
                         # "attention_window_summary" record type. UNVALIDATED (see module
                         # docstring) -- never composited into vectors/vectors_deviation/va_point.
                         # 1.6: orientation-cluster validation prep -- exposes the attention
                         # signal as THREE separate top-level fields (sample records AND window
                         # summaries): "screen_orientation" (unchanged in meaning, now leaner),
                         # NEW "gaze_direction" (was folded into screen_orientation's
                         # head_pose_only/gaze_reliable flags -- now its own field, always
                         # carrying gaze_reliable), NEW "look_away" per-sample / "look_away_rate"
                         # per-window (1 - oriented_rate, honestly labeled, never "distraction").
                         # No vector math changed -- pure reshape of values compute_v_so already
                         # produced. Historical 1.5 records keep their old flat shape; only new
                         # records use the 1.6 three-field shape.
SESSION_ID = str(uuid.uuid4())
# D1 feature-block separation: episodes.WindowAccumulator.flush() and
# attention.AttentionWindowAccumulator.flush() each stamp a bare SESSION_ID
# name resolved against the module where they're DEFINED, not this one --
# kept in sync here so every record this process writes (sample records
# built directly below, plus window/attention-window summaries produced by
# the imported classes) carries the SAME session_id. See features/episodes.py
# and features/attention.py's own comments on this global.
episodes.SESSION_ID = SESSION_ID
attention.SESSION_ID = SESSION_ID
# Set once by main() after consent (step 7), before any thread starts.
# One launch = one person = one session_id = one person_label -- no
# internal multi-person loop reads or writes this after that point.
PERSON_LABEL = None

# Per-person within-session neutral calibration (Decision #5, Pitfall
# #2, MANDATORY ARCHITECTURE #6). Fixed, universal duration -- same for
# everyone, no per-person tuning of the calibration window itself
# (Pitfall #5). Within-session only: no cross-session persistence, no
# FAISS/re-identification (explicitly OUT of POC, Decision #5) -- this
# calibrator is recreated from scratch every time the process starts.
# CALIBRATION_SECONDS itself now lives in features/x_core.py (imported
# above); referenced here only for the calibration-start print message.

stop_event = threading.Event()
frame_lock = threading.Lock()
latest_frame = None

readings_lock = threading.Lock()
latest_readings = {"overlay_lines": [], "face_detected": False, "calibrated": False, "va_point": None, "window_flagged": False}


def capture_thread():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[Capture] ERROR: could not open webcam.")
        stop_event.set()
        return

    frame_count = 0
    fps_window_start = time.perf_counter()
    print("[Capture] thread started.")
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            continue
        with frame_lock:
            latest_frame = frame
        frame_count += 1
        elapsed = time.perf_counter() - fps_window_start
        if elapsed >= FPS_REPORT_INTERVAL_SECONDS:
            print(f"[Capture] sustained FPS: {frame_count / elapsed:.1f}")
            frame_count = 0
            fps_window_start = time.perf_counter()

    cap.release()
    print("[Capture] thread stopped.")


def processing_thread():
    global latest_readings
    print("[Processing] thread started.")

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,  # unchanged; not the requested gate
            min_face_presence_confidence=CONFIDENCE_THRESHOLD,
            min_tracking_confidence=CONFIDENCE_THRESHOLD,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=CONFIDENCE_THRESHOLD,
            min_tracking_confidence=CONFIDENCE_THRESHOLD,
        )
    )

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"session_{SESSION_ID}.jsonl")
    print(f"[Processing] logging to {log_path}")

    pd_buffer = deque()
    window_acc = WindowAccumulator()
    attention_window_acc = AttentionWindowAccumulator()  # V_so -- independent clock, no calibration gate (see class docstring)
    calibrator = NeutralCalibrator()
    stream_start = time.perf_counter()
    cycle_count = 0
    fps_window_start = time.perf_counter()
    # step 6: confidence context comes from the LAST COMPLETED window
    # (10s cadence, per the two-clock design), not recomputed per-sample --
    # a flagged window means "don't trust readings right now", persisting
    # until the next window completes.
    last_window_low_confidence = False
    print(f"[Calibration] starting -- relax your face completely (jaw loose, as if resting alone) for {CALIBRATION_SECONDS:.0f}s...")

    with open(log_path, "a", encoding="utf-8") as log_file:
        while not stop_event.is_set():
            with frame_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.01)
                continue

            cycle_start = time.perf_counter()
            timestamp_ms = int((cycle_start - stream_start) * 1000)

            clahe_frame = apply_clahe(frame)
            rgb_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "sample",
                "session_id": SESSION_ID,
                "person_label": PERSON_LABEL,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "ts_monotonic": cycle_start,
                "vectors": {"v_bf": None, "v_es": None, "v_jc": None, "v_pd": None},
                "vectors_deviation": {"v_bf": None, "v_es": None, "v_pd": None},
                "vector_components": {},
                # ATTENTION SIGNAL -- THREE separate, own-namespace fields (never
                # touch "vectors"/"vectors_deviation" above, so they structurally
                # cannot be composited into Valence/Arousal). UNVALIDATED (module
                # docstring) until tested across real people. Kept as three
                # distinct top-level keys, not nested under one wrapper, so each
                # can be read/scored independently (validation-prep requirement).
                "screen_orientation": {
                    "score": None,
                    "oriented": None,
                    "unvalidated": True,
                    "label": "screen orientation (geometric) -- NOT attention/engagement",
                },
                "gaze_direction": {
                    "score": None,
                    "gaze_reliable": None,
                    "unvalidated": True,
                    "label": "gaze direction (when reliable) -- bonus signal, NOT a mental-state read; ignore unless gaze_reliable is true",
                },
                "look_away": {
                    "value": None,
                    "unvalidated": True,
                    "label": "look-away flag (geometric) -- per-frame input to the window's look_away_rate; NOT distraction/disengagement",
                },
                "head_pose": {"yaw_deg": None, "pitch_deg": None, "roll_deg": None},
                "quality": {
                    "face_detected": False,
                    "pose_detected": False,
                    "face_width_px": None,
                    "z_cm": None,
                    "min_face_presence_confidence": CONFIDENCE_THRESHOLD,
                    "min_tracking_confidence": CONFIDENCE_THRESHOLD,
                },
                "person_id": None,
                # cold-start (Gap 2 L1 / Pitfall #3): "calibrating" until the
                # neutral-capture phase completes -- no confident reading
                # exists yet, so vectors_deviation stays all-None until then.
                "calibration_status": "calibrated" if calibrator.is_calibrated() else "calibrating",
                "calibration_neutral_ref": calibrator.reference,
                "cycle_time_ms": None,
            }
            overlay_lines = []
            # step 5 window inputs: composite (own deviation/z-score tracked;
            # only v_es/v_pd feed V/A, Decision 18) vs covariate (Track-B
            # logging only) -- see episodes.WindowAccumulator
            window_composite = {"v_bf": None, "v_es": None, "v_pd": None}
            window_covariate = {"v_jc": None, "v_bf_convergence_ratio": None, "v_es_cheek_raise": None}
            window_yaw = None

            if face_result.face_landmarks and face_result.facial_transformation_matrixes:
                lms = face_result.face_landmarks[0]
                matrix = face_result.facial_transformation_matrixes[0]
                normalized_pts = pose_normalize(lms, matrix, w, h)
                io_dist = interocular_distance(normalized_pts)

                yaw, pitch, roll = yaw_pitch_roll_from_matrix(matrix)
                v_bf, bf_components = compute_v_bf(normalized_pts, io_dist)
                v_es, es_components = compute_v_es(normalized_pts, io_dist)
                v_jc, jc_components = compute_v_jc(normalized_pts, io_dist)
                # ATTENTION SIGNAL Step 1 -- reuses yaw/pitch already decomposed
                # above (no recomputation) and normalized_pts already built above
                # (no new landmark work); see features.attention.compute_v_so's docstring.
                v_so, so_components = compute_v_so(normalized_pts, yaw, pitch)

                record["vectors"]["v_bf"] = v_bf
                record["vectors"]["v_es"] = v_es
                record["vectors"]["v_jc"] = v_jc
                record["vector_components"]["v_bf"] = bf_components
                record["vector_components"]["v_es"] = es_components
                record["vector_components"]["v_jc"] = jc_components
                record["screen_orientation"] = {
                    "score": v_so,
                    "oriented": so_components["oriented"],
                    "unvalidated": True,
                    "label": "screen orientation (geometric) -- NOT attention/engagement",
                }
                record["gaze_direction"] = {
                    "score": so_components["gaze_score"],  # None unless gaze_reliable -- never a silent zero
                    "gaze_reliable": so_components["gaze_reliable"],
                    "unvalidated": True,
                    "label": "gaze direction (when reliable) -- bonus signal, NOT a mental-state read; ignore unless gaze_reliable is true",
                }
                record["look_away"] = {
                    "value": not so_components["oriented"],
                    "unvalidated": True,
                    "label": "look-away flag (geometric) -- per-frame input to the window's look_away_rate; NOT distraction/disengagement",
                }
                record["head_pose"] = {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}
                record["quality"]["face_detected"] = True
                record["quality"]["face_width_px"] = io_dist
                record["quality"]["z_cm"] = float(np.asarray(matrix)[2, 3])

                window_composite["v_bf"] = v_bf
                window_composite["v_es"] = v_es
                window_covariate["v_jc"] = v_jc
                window_covariate["v_bf_convergence_ratio"] = bf_components["convergence_ratio"]
                window_covariate["v_es_cheek_raise"] = es_components["cheek_raise"]
                window_yaw = yaw

                overlay_lines.append(f"V_bf={v_bf:+.3f}  (convergence_ratio={bf_components['convergence_ratio']:.3f})")
                overlay_lines.append(f"V_es={v_es:+.3f}  (aperture={es_components['aperture']:.3f})")
                overlay_lines.append(f"V_jc={v_jc:+.3f}  (inter_lip={jc_components['inter_lip_dist']:.3f})")
                overlay_lines.append(f"yaw={yaw:+.1f} pitch={pitch:+.1f} roll={roll:+.1f}")
                overlay_lines.append(
                    f"V_so={v_so:.2f} {'ORIENTED' if so_components['oriented'] else 'not-oriented'}"
                    f"{' [head-pose-only]' if so_components['head_pose_only'] else ''}  (UNVALIDATED)"
                )

            if pose_result.pose_world_landmarks:
                world = pose_result.pose_world_landmarks[0]
                nose_pos = np.array([world[POSE_NOSE].x, world[POSE_NOSE].y, world[POSE_NOSE].z])
                shoulder_mid = np.array(
                    [
                        (world[POSE_SHOULDER_L].x + world[POSE_SHOULDER_R].x) / 2.0,
                        (world[POSE_SHOULDER_L].y + world[POSE_SHOULDER_R].y) / 2.0,
                        (world[POSE_SHOULDER_L].z + world[POSE_SHOULDER_R].z) / 2.0,
                    ]
                )
                v_pd, pd_components = compute_v_pd(pd_buffer, nose_pos, shoulder_mid, cycle_start)
                record["vectors"]["v_pd"] = v_pd
                record["vector_components"]["v_pd"] = pd_components
                record["quality"]["pose_detected"] = True
                window_composite["v_pd"] = v_pd
                overlay_lines.append(f"V_pd={'n/a' if v_pd is None else f'{v_pd:.5f}'}")

            # ATTENTION SIGNAL Step 1 -- fed EVERY cycle, unconditionally
            # (unlike window_acc below, which only starts once calibrated):
            # V_so needs no per-person baseline, so its window should cover
            # the whole session from frame 1, including the calibration
            # phase -- see AttentionWindowAccumulator's docstring for why.
            attention_window_acc.add_sample(
                cycle_start,
                {
                    "detected": record["quality"]["face_detected"],
                    "orientation_score": record["screen_orientation"]["score"],
                    "oriented": record["screen_orientation"]["oriented"],
                    "gaze_score": record["gaze_direction"]["score"],
                    "gaze_reliable": record["gaze_direction"]["gaze_reliable"],
                },
            )

            cycle_ms = (time.perf_counter() - cycle_start) * 1000.0
            record["cycle_time_ms"] = cycle_ms

            # feed calibration BEFORE checking status, so this cycle's own
            # sample counts toward its own completion
            calibrator.add_sample(cycle_start, window_composite, window_covariate, window_yaw)
            if calibrator.should_complete(cycle_start):
                reference = calibrator.complete(cycle_start)
                log_file.write(json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "calibration_complete",
                    "session_id": SESSION_ID,
                    "person_label": PERSON_LABEL,
                    "reference": reference,
                }) + "\n")
                tag = "POSSIBLY NOT NEUTRAL" if reference["quality"]["possibly_not_neutral"] else "OK"
                print(f"\n=== CALIBRATION COMPLETE ({reference['calibration_seconds']:.1f}s) [{tag}] ===")
                if reference["quality"]["reasons"]:
                    for r in reference["quality"]["reasons"]:
                        print(f"  FLAG: {r}")
                for k, s in reference["composite"].items():
                    print(f"  {k:6s} neutral: mean={s['mean']:+.4f} std={s['std']:.4f} n={s['n']}")
                for k, s in reference["covariates"].items():
                    print(f"  {k:26s} neutral: mean={s['mean']}  std={s['std']}  n={s['n']}")
                print()

            record["calibration_status"] = "calibrated" if calibrator.is_calibrated() else "calibrating"
            record["calibration_neutral_ref"] = calibrator.reference

            deviation_composite = {"v_bf": None, "v_es": None, "v_pd": None}
            va_point = None
            if calibrator.is_calibrated():
                for key in ("v_bf", "v_es", "v_pd"):
                    deviation_composite[key] = calibrator.deviation(key, window_composite.get(key))
                record["vectors_deviation"] = deviation_composite
                # step 6 -- UNTESTED HYPOTHESIS (see map_to_valence_arousal
                # docstring). Computed per-sample for a live-updating point;
                # trustworthiness of that point is a separate question,
                # answered by last_window_low_confidence below, not by
                # anything in this function.
                va_point = map_to_valence_arousal(
                    deviation_composite["v_bf"], deviation_composite["v_es"], deviation_composite["v_pd"], calibrator.reference
                )
            record["va_point"] = va_point

            if not calibrator.is_calibrated():
                remaining = calibrator.seconds_remaining(cycle_start)
                # Procedural mitigation for the contamination flag's known
                # blind spot (see classify_calibration_quality docstring):
                # the flag cannot see a bias held constant from the start,
                # so (1) the instruction is explicit about full relaxation,
                # not just "neutral" (which people interpret as "posed
                # still", not "actually slack"), and (2) raw values stay
                # visible so a human running Gate 2 can eyeball a resting
                # face that looks off, which the flag structurally can't.
                overlay_lines = [
                    "CALIBRATING -- relax your face completely: jaw loose, as if resting alone",
                    f"{remaining:.0f}s remaining",
                    "(no reading yet -- Gap 2 L1: uncalibrated = no confident reading)",
                ]
                if record["quality"]["face_detected"]:
                    overlay_lines.append(f"raw: V_bf={window_composite['v_bf']:+.3f}  V_es={window_composite['v_es']:+.3f}")
                if record["quality"]["pose_detected"] and window_composite["v_pd"] is not None:
                    overlay_lines.append(f"raw: V_pd={window_composite['v_pd']:+.5f}")
                # V_so needs no calibration (module docstring) -- shown during
                # the calibration phase too, unlike the affect vectors above.
                if record["quality"]["face_detected"]:
                    so = record["screen_orientation"]
                    gz = record["gaze_direction"]
                    overlay_lines.append(
                        f"V_so={so['score']:.2f} {'ORIENTED' if so['oriented'] else 'not-oriented'}"
                        f"{'' if gz['gaze_reliable'] else ' [head-pose-only]'}  (UNVALIDATED)"
                    )
            else:
                overlay_lines = []
                if record["quality"]["face_detected"]:
                    overlay_lines.append(f"V_bf(dev)={deviation_composite['v_bf']:+.3f}  raw={window_composite['v_bf']:+.3f}")
                    overlay_lines.append(f"V_es(dev)={deviation_composite['v_es']:+.3f}  raw={window_composite['v_es']:+.3f}")
                    overlay_lines.append(f"yaw={window_yaw:+.1f}" if window_yaw is not None else "yaw=n/a")
                if record["quality"]["pose_detected"]:
                    dev_pd = deviation_composite["v_pd"]
                    overlay_lines.append(f"V_pd(dev)={'n/a' if dev_pd is None else f'{dev_pd:+.5f}'}")
                if record["quality"]["face_detected"]:
                    so = record["screen_orientation"]
                    gz = record["gaze_direction"]
                    overlay_lines.append(
                        f"V_so={so['score']:.2f} {'ORIENTED' if so['oriented'] else 'not-oriented'}"
                        f"{'' if gz['gaze_reliable'] else ' [head-pose-only]'}  (UNVALIDATED)"
                    )
                if not overlay_lines:
                    overlay_lines = ["waiting for detection..."]

            log_file.write(json.dumps(record) + "\n")

            # step 5's window operates on DEVIATION values once calibrated --
            # step 6 must consume calibrated vectors (Pitfall #2), not raw
            # magnitudes. During calibration there is no deviation yet, so
            # the window simply does not start until calibration completes.
            if calibrator.is_calibrated():
                window_acc.add_sample(cycle_start, record["quality"]["face_detected"], window_yaw, deviation_composite, window_covariate)
            if window_acc.should_flush(cycle_start):
                summary = window_acc.flush(cycle_start)
                last_window_low_confidence = summary["window_quality"]["low_confidence"]
                log_file.write(json.dumps(summary) + "\n")
                tag = "LOW-CONFIDENCE" if summary["window_quality"]["low_confidence"] else "valid"
                print(
                    f"\n[Window: {tag}] n={summary['n_samples']} detect_rate={summary['detection_rate']:.2f} "
                    f"yaw_var={summary['yaw_variance_deg2']}"
                    + (f"  reasons={summary['window_quality']['reasons']}" if summary["window_quality"]["reasons"] else "")
                )
                print(
                    f"  V_bf(avg={summary['composite']['v_bf']['avg']}, peak={summary['composite']['v_bf']['peak']}, var={summary['composite']['v_bf']['variance']}) "
                    f"V_es(avg={summary['composite']['v_es']['avg']}, peak={summary['composite']['v_es']['peak']}, var={summary['composite']['v_es']['variance']}) "
                    f"V_pd(avg={summary['composite']['v_pd']['avg']}, peak={summary['composite']['v_pd']['peak']}, var={summary['composite']['v_pd']['variance']})"
                )

            # ATTENTION SIGNAL Step 1 -- own independent flush, own clock
            # (see AttentionWindowAccumulator docstring); never gated on
            # calibrator.is_calibrated() the way window_acc's flush above is.
            if attention_window_acc.should_flush(cycle_start):
                so_summary = attention_window_acc.flush(cycle_start)
                log_file.write(json.dumps(so_summary) + "\n")
                so_win, gz_win, la_win = so_summary["screen_orientation"], so_summary["gaze_direction"], so_summary["look_away_rate"]
                print(
                    f"[Attention window -- UNVALIDATED] n={so_summary['n_samples']} "
                    f"detect_rate={so_summary['detection_rate']:.2f} "
                    f"screen_orientation(avg={so_win['avg']}, peak={so_win['peak']}, oriented_rate={so_win['oriented_rate']}) "
                    f"gaze_direction(avg={gz_win['avg']}, gaze_reliable_rate={gz_win['gaze_reliable_rate']}) "
                    f"look_away_rate={la_win['value']}"
                )

            log_file.flush()

            with readings_lock:
                latest_readings = {
                    "overlay_lines": overlay_lines,
                    "face_detected": record["quality"]["face_detected"],
                    "calibrated": calibrator.is_calibrated(),
                    "va_point": va_point,
                    "window_flagged": last_window_low_confidence,
                }

            cycle_count += 1
            elapsed = time.perf_counter() - fps_window_start
            if elapsed >= FPS_REPORT_INTERVAL_SECONDS:
                print(f"[Processing] {cycle_count / elapsed:.1f} samples/sec, last cycle {cycle_ms:.0f}ms")
                cycle_count = 0
                fps_window_start = time.perf_counter()

    face_landmarker.close()
    pose_landmarker.close()
    print("[Processing] thread stopped.")


def draw_overlay(frame):
    with readings_lock:
        lines = list(latest_readings["overlay_lines"])
        detected = latest_readings["face_detected"]

    if not lines:
        cv2.putText(frame, "waiting for detection...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    color = (0, 200, 0) if detected else (0, 0, 255)
    for i, line in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


def draw_va_plot(frame):
    """Step 6 live V/A plot. Drawn directly with OpenCV (no matplotlib)
    to avoid a second GUI event loop fighting cv2's own window/thread.

    Low-confidence windows (window-validity gate, step 5) must NOT plot
    a confident point -- per instruction, flagged rather than
    suppressed, matching the same choice already made for the window
    gate itself: render the point in red with an "UNSTABLE" label
    instead of hiding it outright, so the demo shows the failure mode
    rather than a suspicious blank.

    Honest framing (Decision 18): Valence is z_es only, pleasure-side,
    single-source -- there is NO validated pain axis. The negative-
    valence half is greyed/hatched and labeled so a negative reading
    can never be mistaken for a measured "pain" signal, matching the
    same treatment stage3_demo_ui.py's larger V/A panel uses.
    """
    with readings_lock:
        calibrated = latest_readings["calibrated"]
        va_point = latest_readings["va_point"]
        window_flagged = latest_readings["window_flagged"]

    size = 160
    margin = 10
    h, w = frame.shape[:2]
    ox, oy = w - size - margin, margin
    cx, cy = ox + size // 2, oy + size // 2

    cv2.rectangle(frame, (ox, oy), (ox + size, oy + size), (40, 40, 40), -1)

    # Negative-valence (pain) half: greyed + hatched + labeled unvalidated --
    # must never read as a measurement, same rule as stage3_demo_ui.py.
    overlay = frame.copy()
    cv2.rectangle(overlay, (ox, oy), (cx, oy + size), (12, 12, 12), -1)
    for offset in range(-size, size, 10):
        x_a, y_a = ox + offset, oy + size
        x_b, y_b = ox + offset + size, oy
        cv2.line(overlay, (x_a, y_a), (x_b, y_b), (60, 60, 60), 1, cv2.LINE_AA)
    frame[oy:oy + size, ox:cx] = cv2.addWeighted(frame[oy:oy + size, ox:cx], 0.15, overlay[oy:oy + size, ox:cx], 0.85, 0)

    cv2.rectangle(frame, (ox, oy), (ox + size, oy + size), (200, 200, 200), 1)
    cv2.line(frame, (ox, cy), (ox + size, cy), (110, 110, 110), 1)
    cv2.line(frame, (cx, oy), (cx, oy + size), (110, 110, 110), 1)
    cv2.putText(frame, "V/A (pleasure-side only)", (ox, oy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    cv2.putText(frame, "no pain axis", (ox + 4, cy - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)
    cv2.putText(frame, "not measured", (ox + 4, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)
    cv2.putText(frame, "+valence", (ox + size - 58, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
    cv2.putText(frame, "+arousal", (cx + 4, oy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

    valence = va_point.get("valence") if va_point else None
    arousal = va_point.get("arousal") if va_point else None

    if not calibrated or valence is None or arousal is None:
        cv2.putText(frame, "n/a", (cx - 14, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return frame

    px = int(cx + valence * (size // 2 - 8))
    py = int(cy - arousal * (size // 2 - 8))  # screen y is inverted vs. arousal-up

    if window_flagged:
        cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(frame, "UNSTABLE", (ox, oy + size + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    else:
        cv2.circle(frame, (px, py), 6, (0, 220, 0), -1)

    return frame


def main():
    global PERSON_LABEL

    # Stage 1.5 step 7 (Decision #6): consent gate runs BEFORE anything
    # else in this function -- no thread, no camera, no model load
    # happens above this line. Opting out returns here with nothing
    # started and nothing recorded.
    from stage1_step7_consent import run_consent_gate

    consented, person_label = run_consent_gate(SESSION_ID)
    if not consented:
        return
    PERSON_LABEL = person_label
    # D1 feature-block separation: NeutralCalibrator.complete() (x_core),
    # WindowAccumulator.flush() (episodes), and AttentionWindowAccumulator.
    # flush() (attention) each stamp a bare PERSON_LABEL name resolved
    # against the module where they're DEFINED -- synced here the same way
    # SESSION_ID is synced above, so every record carries the SAME
    # person_label regardless of which moved module produced it.
    x_core.PERSON_LABEL = person_label
    episodes.PERSON_LABEL = person_label
    attention.PERSON_LABEL = person_label

    t1 = threading.Thread(target=capture_thread, name="CaptureThread")
    t2 = threading.Thread(target=processing_thread, name="ProcessingThread")
    t1.start()
    t2.start()

    print("Press 'q' in the preview window to quit.")
    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame

        if frame is not None:
            display_frame = draw_overlay(frame.copy())
            display_frame = draw_va_plot(display_frame)
            cv2.imshow("Stage 1 step 4 - vectors (press Q to quit)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    t1.join()
    t2.join()
    print("Clean shutdown complete.")


if __name__ == "__main__":
    main()
