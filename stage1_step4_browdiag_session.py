"""
Diagnostic-only: does io_dist (interocular) itself shrink during hard
brow furrowing (likely via co-occurring squint biasing the estimated
iris center), or is the numerator (inner-brow distance) the one moving
wrong? Logs both raw and normalized so they can be separated. 2 reps,
short session.
"""

import cv2
import json
import mediapipe as mp
import os
import time
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from stage1_step4_vectors import (
    apply_clahe,
    pose_normalize,
    interocular_distance,
    _dist,
    BROW_INNER_R,
    BROW_INNER_L,
    GLABELLA,
    FACE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SESSION_LABEL = "browdiag_" + time.strftime("%Y%m%d_%H%M%S")

PHASES = [
    ("lead_in", 3, "Get ready."),
    ("neutral_0", 5, "NEUTRAL baseline."),
    ("brow_furrow_1", 4, "MAX FURROW - hard."),
    ("neutral_1", 4, "Relax."),
    ("brow_furrow_2", 4, "MAX FURROW again - hard."),
    ("neutral_2", 4, "Relax."),
    ("wind_down", 2, "Done."),
]


def current_phase(elapsed):
    cum = 0.0
    for i, (label, dur, instr) in enumerate(PHASES):
        cum += dur
        if elapsed < cum:
            return i, label, instr, cum - elapsed
    return None, None, None, 0.0


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
    print(f"Brow diagnostic starting, ~{total}s. Log: {log_path}")

    stream_start = time.perf_counter()
    active_idx = -1

    with open(log_path, "w", encoding="utf-8") as log_file:
        while True:
            elapsed = time.perf_counter() - stream_start
            idx, label, instr, remaining = current_phase(elapsed)
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
            record = {"ts_monotonic": time.perf_counter(), "phase": label, "face_detected": False}

            if result.face_landmarks and result.facial_transformation_matrixes:
                lms = result.face_landmarks[0]
                matrix = result.facial_transformation_matrixes[0]
                normalized_pts = pose_normalize(lms, matrix, w, h)
                io_dist = interocular_distance(normalized_pts)
                inner_brow_dist_raw = _dist(normalized_pts, BROW_INNER_R, BROW_INNER_L)
                drop_r_raw = _dist(normalized_pts, BROW_INNER_R, GLABELLA)
                drop_l_raw = _dist(normalized_pts, BROW_INNER_L, GLABELLA)

                record["face_detected"] = True
                record["io_dist"] = io_dist
                record["inner_brow_dist_raw"] = inner_brow_dist_raw
                record["drop_raw"] = (drop_r_raw + drop_l_raw) / 2.0

            log_file.write(json.dumps(record) + "\n")

            disp = cv2.flip(frame, 1)
            cv2.putText(disp, f"[{label}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(disp, instr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(disp, f"{remaining:.1f}s left", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow("Brow diagnostic (press Q to abort)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
