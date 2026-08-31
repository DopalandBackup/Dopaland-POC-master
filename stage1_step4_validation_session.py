"""
Stage 1, step 4 — characterization session. NOT part of the shipped
pipeline; throwaway diagnostic tool. Single-threaded (unlike the
production two-thread app) because this only needs to run once, briefly,
under direct supervision — architectural purity doesn't buy anything
here.

Runs one guided, phase-scripted capture in a single sitting:
  - pose-and-hold yaw extremes (closes the +/-30 deg gap every prior
    head-turn sweep fell short of — those only reached ~15 deg)
  - a directed-expression mini-session (brow furrow / smile / jaw clench)
  - a postural movement burst (needed for V_pd, which has no facial
    "expression" of its own)

Imports the vector-computation functions from stage1_step4_vectors.py
directly (not re-implemented here) so this characterizes the exact
formulas that ship, not a stale copy.

This is characterization only — it does NOT set or tune any per-person
threshold (Pitfall #5: n=1 tuning on the developer's own face is
self-deception). It only measures signal (expression response) against
noise (rest jitter + yaw-drift band) so those numbers can inform Stage
1.5's calibration design later, on 8-12 people, not here.
"""

import cv2
import json
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from stage1_step4_vectors import (
    apply_clahe,
    yaw_pitch_roll_from_matrix,
    pose_normalize,
    interocular_distance,
    compute_v_bf,
    compute_v_es,
    compute_v_jc,
    compute_v_pd,
    FACE_MODEL_PATH,
    POSE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    POSE_NOSE,
    POSE_SHOULDER_L,
    POSE_SHOULDER_R,
)

CAMERA_INDEX = 0
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SESSION_LABEL = "validation_" + time.strftime("%Y%m%d_%H%M%S")

# (phase_label, duration_seconds, on-screen/console instruction)
PHASES = [
    ("lead_in", 3, "Get ready. Sit neutral, facing the camera."),
    ("neutral_base", 6, "NEUTRAL. Relaxed face, look straight at the camera."),
    ("yaw_right_hold", 5, "TURN RIGHT ~30deg and HOLD. Keep face neutral."),
    ("yaw_center_a", 4, "Return to CENTER. Neutral face."),
    ("yaw_left_hold", 5, "TURN LEFT ~30deg and HOLD. Keep face neutral."),
    ("yaw_center_b", 4, "Return to CENTER. Neutral face."),
    ("brow_furrow", 5, "FURROW your brow (frown/concentrate) and HOLD."),
    ("neutral_c", 4, "Relax. NEUTRAL face."),
    ("smile", 5, "SMILE genuinely - think of something that makes you happy."),
    ("neutral_d", 4, "Relax. NEUTRAL face."),
    ("jaw_clench", 5, "CLENCH your jaw / press your lips and HOLD."),
    ("neutral_e", 4, "Relax. NEUTRAL face."),
    ("still_hold", 5, "Sit VERY STILL. Neutral posture."),
    ("movement_burst", 5, "SHIFT in your seat a few times (stay facing camera)."),
    ("wind_down", 3, "Done - relax."),
]


def current_phase(elapsed):
    cum = 0.0
    for i, (label, dur, instr) in enumerate(PHASES):
        cum += dur
        if elapsed < cum:
            return i, label, instr, cum - elapsed
    return None, None, None, 0.0


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
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
    log_path = os.path.join(LOG_DIR, f"{SESSION_LABEL}.jsonl")
    manifest_path = os.path.join(LOG_DIR, f"{SESSION_LABEL}_manifest.json")

    total_duration = sum(p[1] for p in PHASES)
    print(f"Validation session starting, total ~{total_duration}s. Log: {log_path}")

    pd_buffer = deque()
    stream_start = time.perf_counter()
    phase_boundaries = []
    active_idx = -1
    active_label = None
    active_start = stream_start

    with open(log_path, "w", encoding="utf-8") as log_file:
        while True:
            elapsed = time.perf_counter() - stream_start
            idx, label, instr, remaining = current_phase(elapsed)
            if idx is None:
                break
            if idx != active_idx:
                now = time.perf_counter()
                if active_label is not None:
                    phase_boundaries.append({"label": active_label, "start": active_start, "end": now})
                active_idx, active_label, active_start = idx, label, now
                print(f"\n[{label}] {instr}  ({PHASES[idx][1]}s)")

            ok, frame = cap.read()
            if not ok:
                continue

            timestamp_ms = int(elapsed * 1000)
            clahe_frame = apply_clahe(frame)
            rgb_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            record = {
                "ts_monotonic": time.perf_counter(),
                "phase": label,
                "vectors": {"v_bf": None, "v_es": None, "v_jc": None, "v_pd": None},
                "vector_components": {},
                "head_pose": {"yaw_deg": None, "pitch_deg": None, "roll_deg": None},
                "face_detected": False,
                "pose_detected": False,
            }

            if face_result.face_landmarks and face_result.facial_transformation_matrixes:
                lms = face_result.face_landmarks[0]
                matrix = face_result.facial_transformation_matrixes[0]
                normalized_pts = pose_normalize(lms, matrix, w, h)
                io_dist = interocular_distance(normalized_pts)
                yaw, pitch, roll = yaw_pitch_roll_from_matrix(matrix)

                v_bf, bf_c = compute_v_bf(normalized_pts, io_dist)
                v_es, es_c = compute_v_es(normalized_pts, io_dist)
                v_jc, jc_c = compute_v_jc(normalized_pts, io_dist)

                record["vectors"]["v_bf"] = v_bf
                record["vectors"]["v_es"] = v_es
                record["vectors"]["v_jc"] = v_jc
                record["vector_components"] = {"v_bf": bf_c, "v_es": es_c, "v_jc": jc_c}
                record["head_pose"] = {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}
                record["face_detected"] = True

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
                v_pd, pd_c = compute_v_pd(pd_buffer, nose_pos, shoulder_mid, time.perf_counter())
                record["vectors"]["v_pd"] = v_pd
                record["vector_components"]["v_pd"] = pd_c
                record["pose_detected"] = True

            log_file.write(json.dumps(record) + "\n")

            disp = frame.copy()
            cv2.putText(disp, f"[{label}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(disp, instr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(disp, f"{remaining:.1f}s left", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.imshow("Validation session (press Q to abort)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        phase_boundaries.append({"label": active_label, "start": active_start, "end": time.perf_counter()})

    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump({"session_label": SESSION_LABEL, "stream_start": stream_start, "phases": phase_boundaries}, mf, indent=2)

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()
    print(f"\nDone. Log: {log_path}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
