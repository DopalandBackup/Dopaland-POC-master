"""
Stage 1, step 4 — max-elicitation VALIDITY check. NOT calibration, NOT
threshold-tuning (Pitfall #5) — this only asks "does the raw redesigned
component move, in the right direction, at a maximal/exaggerated
expression?" That's a yes/no validity question per vector, not a
per-person number to bake into the pipeline.

Does NOT touch V_pd (already nearest the SNR bar, not in question here)
and does NOT run the pose landmarker at all, to keep this fast and
focused on the three facial vectors under test.

3 reps each of brow_furrow / smile / jaw_clench, held 4s, maximal/
exaggerated effort, separated by neutral holds, so a weak single
attempt (as happened in the prior directed-expression session) can't
be mistaken for "the metric doesn't work."
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
    compute_v_es,
    compute_v_jc,
    FACE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SESSION_LABEL = "maxelicit_" + time.strftime("%Y%m%d_%H%M%S")

# (phase_label, duration_s, category, instruction)
PHASES = [
    ("lead_in", 3, None, "Get ready."),
    ("neutral_0", 5, "neutral", "NEUTRAL baseline. Relax completely."),
    ("brow_furrow_1", 4, "brow_furrow", "MAX FURROW - pull brows together & DOWN, hard."),
    ("neutral_b1", 3, "neutral", "Relax."),
    ("brow_furrow_2", 4, "brow_furrow", "MAX FURROW again, as hard as possible."),
    ("neutral_b2", 3, "neutral", "Relax."),
    ("brow_furrow_3", 4, "brow_furrow", "MAX FURROW once more, hard."),
    ("neutral_b3", 3, "neutral", "Relax."),
    ("smile_1", 4, "smile", "MAX SMILE - as big as possible."),
    ("neutral_s1", 3, "neutral", "Relax."),
    ("smile_2", 4, "smile", "MAX SMILE again, big."),
    ("neutral_s2", 3, "neutral", "Relax."),
    ("smile_3", 4, "smile", "MAX SMILE once more."),
    ("neutral_s3", 3, "neutral", "Relax."),
    ("jaw_clench_1", 4, "jaw_clench", "MAX JAW CLENCH / lip press, as hard as possible."),
    ("neutral_j1", 3, "neutral", "Relax."),
    ("jaw_clench_2", 4, "jaw_clench", "MAX JAW CLENCH again, hard."),
    ("neutral_j2", 3, "neutral", "Relax."),
    ("jaw_clench_3", 4, "jaw_clench", "MAX JAW CLENCH once more."),
    ("neutral_j3", 3, "neutral", "Relax."),
    ("wind_down", 2, None, "Done - relax."),
]


def current_phase(elapsed):
    cum = 0.0
    for i, (label, dur, cat, instr) in enumerate(PHASES):
        cum += dur
        if elapsed < cum:
            return i, label, cat, instr, cum - elapsed
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
    print(f"Max-elicitation session starting, ~{total}s. Log: {log_path}")

    stream_start = time.perf_counter()
    active_idx = -1

    with open(log_path, "w", encoding="utf-8") as log_file:
        while True:
            elapsed = time.perf_counter() - stream_start
            idx, label, cat, instr, remaining = current_phase(elapsed)
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
            record = {"ts_monotonic": time.perf_counter(), "phase": label, "category": cat,
                      "face_detected": False, "vectors": {}, "vector_components": {}}

            if result.face_landmarks and result.facial_transformation_matrixes:
                lms = result.face_landmarks[0]
                matrix = result.facial_transformation_matrixes[0]
                normalized_pts = pose_normalize(lms, matrix, w, h)
                io_dist = interocular_distance(normalized_pts)

                v_bf, bf_c = compute_v_bf(normalized_pts, io_dist)
                v_es, es_c = compute_v_es(normalized_pts, io_dist)
                v_jc, jc_c = compute_v_jc(normalized_pts, io_dist)

                record["face_detected"] = True
                record["vectors"] = {"v_bf": v_bf, "v_es": v_es, "v_jc": v_jc}
                record["vector_components"] = {"v_bf": bf_c, "v_es": es_c, "v_jc": jc_c}

            log_file.write(json.dumps(record) + "\n")

            disp = cv2.flip(frame, 1)
            cv2.putText(disp, f"[{label}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(disp, instr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(disp, f"{remaining:.1f}s left", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow("Max-elicitation session (press Q to abort)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    main()
