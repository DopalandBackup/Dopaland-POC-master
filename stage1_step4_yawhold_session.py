"""
Stage 1, step 4 — yaw-hold-only session, retry #2. NOT part of the
shipped pipeline; throwaway diagnostic.

The first attempt (stage1_step4_validation_session.py) didn't reach
+/-30deg (right_hold mean=+2.0, left_hold mean=+6.1 -- both undershoot
and same-signed) and showed elevated jitter even during "neutral" holds
(std 6-8deg vs <1.5deg frame-to-frame in a plain stillness check). Two
fixes here: (1) mirror the display -- an unmirrored feed is disorienting
for self-directed left/right turns, (2) show a live numeric yaw readout
so the target angle is hit by visual feedback instead of guessing.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from stage1_step4_vectors import (
    apply_clahe,
    yaw_pitch_roll_from_matrix,
    pose_normalize,
    interocular_distance,
    compute_v_bf,
    compute_v_es,
    FACE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SESSION_LABEL = "yawhold_" + time.strftime("%Y%m%d_%H%M%S")

PHASES = [
    ("lead_in", 3, None, "Get ready. Sit neutral, facing the camera."),
    ("neutral_base", 5, None, "NEUTRAL. Watch the yaw number settle near 0."),
    ("yaw_hold_a", 6, 30, "Turn to ONE side (your choice) until |yaw| reaches ~30, HOLD."),
    ("yaw_center_a", 4, None, "Return to CENTER (yaw near 0)."),
    ("yaw_hold_b", 6, 30, "Turn to the OPPOSITE side until |yaw| reaches ~30, HOLD."),
    ("yaw_center_b", 4, None, "Return to CENTER (yaw near 0)."),
    ("wind_down", 2, None, "Done - relax."),
]


def current_phase(elapsed):
    cum = 0.0
    for i, (label, dur, target, instr) in enumerate(PHASES):
        cum += dur
        if elapsed < cum:
            return i, label, target, instr, cum - elapsed
    return None, None, None, None, 0.0


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_presence_confidence=CONFIDENCE_THRESHOLD,
            min_tracking_confidence=CONFIDENCE_THRESHOLD,
            output_facial_transformation_matrixes=True,
        )
    )

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{SESSION_LABEL}.jsonl")
    total = sum(p[1] for p in PHASES)
    print(f"Yaw-hold session starting, ~{total}s. Log: {log_path}")

    stream_start = time.perf_counter()
    active_idx = -1

    with open(log_path, "w", encoding="utf-8") as log_file:
        while True:
            elapsed = time.perf_counter() - stream_start
            idx, label, target, instr, remaining = current_phase(elapsed)
            if idx is None:
                break
            if idx != active_idx:
                active_idx = idx
                print(f"\n[{label}] {instr}")

            ok, frame = cap.read()
            if not ok:
                continue

            ts_ms = int(elapsed * 1000)
            clahe_frame = apply_clahe(frame)
            rgb = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = face_landmarker.detect_for_video(mp_image, ts_ms)

            h, w = frame.shape[:2]
            record = {"ts_monotonic": time.perf_counter(), "phase": label,
                      "vectors": {"v_bf": None, "v_es": None}, "vector_components": {},
                      "yaw_deg": None, "face_detected": False}
            yaw = None
            if result.face_landmarks and result.facial_transformation_matrixes:
                lms = result.face_landmarks[0]
                matrix = result.facial_transformation_matrixes[0]
                normalized_pts = pose_normalize(lms, matrix, w, h)
                io_dist = interocular_distance(normalized_pts)
                yaw, pitch, roll = yaw_pitch_roll_from_matrix(matrix)
                v_bf, bf_c = compute_v_bf(normalized_pts, io_dist)
                v_es, _ = compute_v_es(normalized_pts, io_dist)
                record["vectors"]["v_bf"] = v_bf
                record["vectors"]["v_es"] = v_es
                record["vector_components"]["v_bf"] = bf_c
                record["io_dist"] = io_dist
                record["yaw_deg"] = yaw
                record["face_detected"] = True

            log_file.write(json.dumps(record) + "\n")

            disp = cv2.flip(frame, 1)  # mirror for intuitive self-directed turning
            cv2.putText(disp, f"[{label}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(disp, instr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(disp, f"{remaining:.1f}s left", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            if yaw is not None:
                # raw (unmirrored) yaw, whatever its sign convention is -- feedback
                # is "watch the number and turn until |yaw| hits target", so the
                # sign-to-physical-direction mapping never needs to be assumed
                if target is not None:
                    color = (0, 255, 0) if abs(abs(yaw) - target) < 5 else (0, 0, 255)
                    label_txt = f"yaw={yaw:+.1f}  target=|{target}|"
                else:
                    color = (0, 255, 0) if abs(yaw) < 5 else (0, 0, 255)
                    label_txt = f"yaw={yaw:+.1f}  target=0"
                cv2.putText(disp, label_txt, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(disp, "no face detected", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Yaw-hold session (press Q to abort)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
