"""
Stage 1, step 4 — V_bf rebuild #2 validity re-check, CONTROLLED for
camera-distance drift. NOT part of shipped pipeline; throwaway
diagnostic. Prior run (3 reps) showed 1/3 correct with escalating
wrong-direction drift; io_dist grew ~7.4% during the one furrow rep
directly measured, consistent with leaning toward the camera while
concentrating -- a natural but confounding behavior, since io_dist
and the brow distances don't scale by identical proportions under a
distance change, so the ratio doesn't fully cancel it.

This version logs io_dist alongside the ratios every frame, so if the
lean-in is actually controlled out this time, io_dist should stay flat
across reps and any remaining direction error is attributable to the
formula, not the confound.
"""

import cv2
import json
import mediapipe as mp
import os
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from stage1_step4_vectors import (
    apply_clahe,
    pose_normalize,
    interocular_distance,
    compute_v_bf,
    FACE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SESSION_LABEL = "browonly_controlled_" + time.strftime("%Y%m%d_%H%M%S")

PHASES = [
    ("lead_in", 4, "Get ready. Pick a fixed distance from screen (e.g. rest chin on hand) - do NOT lean in."),
    ("neutral_0", 5, "NEUTRAL baseline. Relax completely. Stay at this distance."),
    ("brow_furrow_1", 4, "MAX FURROW - hard - but do NOT lean toward the camera."),
    ("neutral_1", 3, "Relax. Stay at the SAME distance."),
    ("brow_furrow_2", 4, "MAX FURROW again, hard - stay put, don't lean in."),
    ("neutral_2", 3, "Relax. Stay at the SAME distance."),
    ("brow_furrow_3", 4, "MAX FURROW once more, hard - stay put."),
    ("neutral_3", 3, "Relax."),
    ("wind_down", 2, "Done - relax."),
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
    print(f"Brow-only re-test starting, ~{total}s. Log: {log_path}")

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
            record = {"ts_monotonic": time.perf_counter(), "phase": label,
                      "face_detected": False, "vectors": {}, "vector_components": {}}

            if result.face_landmarks and result.facial_transformation_matrixes:
                lms = result.face_landmarks[0]
                matrix = result.facial_transformation_matrixes[0]
                normalized_pts = pose_normalize(lms, matrix, w, h)
                io_dist = interocular_distance(normalized_pts)
                v_bf, bf_c = compute_v_bf(normalized_pts, io_dist)
                record["face_detected"] = True
                record["io_dist"] = io_dist
                record["vectors"] = {"v_bf": v_bf}
                record["vector_components"] = {"v_bf": bf_c}

            log_file.write(json.dumps(record) + "\n")

            disp = cv2.flip(frame, 1)
            cv2.putText(disp, f"[{label}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(disp, instr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(disp, f"{remaining:.1f}s left", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow("Brow-only re-test (press Q to abort)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
