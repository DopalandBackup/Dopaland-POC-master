"""
Stage 1.5, step 8 robustness check — repeat-calibration stability +
in-capture contamination flag validation. NOT part of shipped pipeline;
throwaway diagnostic (classify_calibration_quality and the contamination
flag it exercises ARE shipped, in stage1_step4_vectors.py — this script
just characterizes the protocol, doesn't tune anything).

Runs 4 consecutive 25s neutral-calibration captures back to back, same
camera/model session (no restart overhead between rounds):
  rounds 1-3: genuinely neutral — checks whether the same person's
  neutral reference lands in the same place across repeats (protocol
  fragility check, not vector tuning — Pitfall #5).
  round 4: DELIBERATELY held with a faint smile throughout — confirms
  the in-capture contamination flag fires on a real contaminated
  capture, not just synthetic test data.
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
    NeutralCalibrator,
    FACE_MODEL_PATH,
    POSE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    POSE_NOSE,
    POSE_SHOULDER_L,
    POSE_SHOULDER_R,
    CALIBRATION_SECONDS,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SESSION_LABEL = "calib_repeat_" + time.strftime("%Y%m%d_%H%M%S")

ROUNDS = [
    ("round1_neutral", "Round 1/4: NEUTRAL. Relax completely."),
    ("round2_neutral", "Round 2/4: NEUTRAL again. Relax completely."),
    ("round3_neutral", "Round 3/4: NEUTRAL once more. Relax completely."),
    ("round4_contaminated", "Round 4/4: hold a FAINT SMILE throughout (deliberately NOT neutral)."),
]


def run_one_calibration(cap, face_landmarker, pose_landmarker, stream_start, round_label, round_instr):
    calibrator = NeutralCalibrator(calibration_seconds=CALIBRATION_SECONDS)
    pd_buffer = deque()
    print(f"\n--- {round_label} --- {round_instr}")

    while not calibrator.is_calibrated():
        ok, frame = cap.read()
        if not ok:
            continue

        cycle_start = time.perf_counter()
        elapsed_ms = int((cycle_start - stream_start) * 1000)
        clahe_frame = apply_clahe(frame)
        rgb = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = face_landmarker.detect_for_video(mp_image, elapsed_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, elapsed_ms)

        h, w = frame.shape[:2]
        composite = {"v_bf": None, "v_es": None, "v_pd": None}
        covariate = {"v_jc": None, "v_bf_convergence_ratio": None, "v_es_cheek_raise": None}
        yaw = None

        if face_result.face_landmarks and face_result.facial_transformation_matrixes:
            lms = face_result.face_landmarks[0]
            matrix = face_result.facial_transformation_matrixes[0]
            normalized_pts = pose_normalize(lms, matrix, w, h)
            io_dist = interocular_distance(normalized_pts)
            yaw, pitch, roll = yaw_pitch_roll_from_matrix(matrix)
            v_bf, bf_c = compute_v_bf(normalized_pts, io_dist)
            v_es, es_c = compute_v_es(normalized_pts, io_dist)
            v_jc, jc_c = compute_v_jc(normalized_pts, io_dist)
            composite["v_bf"] = v_bf
            composite["v_es"] = v_es
            covariate["v_jc"] = v_jc
            covariate["v_bf_convergence_ratio"] = bf_c["convergence_ratio"]
            covariate["v_es_cheek_raise"] = es_c["cheek_raise"]

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
            v_pd, pd_c = compute_v_pd(pd_buffer, nose_pos, shoulder_mid, cycle_start)
            composite["v_pd"] = v_pd

        calibrator.add_sample(cycle_start, composite, covariate, yaw)

        if calibrator.should_complete(cycle_start):
            return calibrator.complete(cycle_start)

        remaining = calibrator.seconds_remaining(cycle_start)
        disp = cv2.flip(frame, 1)
        cv2.putText(disp, round_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(disp, round_instr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(disp, f"{remaining:.0f}s remaining", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.imshow("Calibration repeat test (press Q to abort)", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    return None


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
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_presence_confidence=CONFIDENCE_THRESHOLD,
            min_tracking_confidence=CONFIDENCE_THRESHOLD,
        )
    )

    stream_start = time.perf_counter()
    results = []
    for label, instr in ROUNDS:
        reference = run_one_calibration(cap, face_landmarker, pose_landmarker, stream_start, label, instr)
        if reference is None:
            print("Aborted early.")
            break
        results.append((label, reference))

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()

    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE NEUTRAL MEANS (composite vectors)")
    print("=" * 78)
    for key in ("v_bf", "v_es", "v_pd"):
        print(f"\n{key}:")
        for label, reference in results:
            s = reference["composite"][key]
            mean_str = f"{s['mean']:+.4f}" if s["mean"] is not None else "n/a"
            std_str = f"{s['std']:.4f}" if s["std"] is not None else "n/a"
            print(f"  {label:22s} mean={mean_str:>10s}  std={std_str:>8s}  n={s['n']}")

    print("\n" + "=" * 78)
    print("CONTAMINATION FLAG")
    print("=" * 78)
    for label, reference in results:
        tag = "POSSIBLY NOT NEUTRAL" if reference["quality"]["possibly_not_neutral"] else "OK"
        print(f"  {label:22s} [{tag}]")
        for r in reference["quality"]["reasons"]:
            print(f"      - {r}")

    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, f"{SESSION_LABEL}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{"label": l, "reference": r} for l, r in results], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
