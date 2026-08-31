"""
Stage 1.5, step 9 — Gate 2 directed-expression capture tool.

DUMB CAPTURE TOOL. Records labeled trial data. Computes NO pass/fail,
NO score, NO accuracy percentage, NO "correct direction" judgement of
any kind, anywhere. Scoring happens later, separately, by a human
against a FROZEN rule this tool never sees (GATE2_SCORING_RULE.md,
Decision 14) -- keeping this tool blind to what "passing" means is
load-bearing for the >=75% number being credible. If you are reading
this file to add a checkmark or a percentage, stop -- that is the
exact failure this design forbids.

One app launch = one person: consent (step 7) -> calibration with
accept/redo (step 8) -> fixed 5-block sequence, in this order, never
randomized:
    smile -> furrow -> concentrate -> sit-still -> fidget
(neutral is the calibration reference already captured; it is not
re-scored here). sit-still is recorded before fidget, always.

Reuses, does not reimplement: consent gate (stage1_step7_consent),
vector computation + pose-normalization + z-scoring
(stage1_step4_vectors), NeutralCalibrator, classify_window_confidence,
and WindowAccumulator._stats for all avg/peak/variance math.

Mid-capture abort (Q at any time) discards every trial captured in
this run for this person -- nothing is written to disk. Trials only
reach gate2_trials.jsonl if the full 5-block sequence completes
without an abort; redone attempts are kept (not deleted), each tagged
with attempt_number and accepted.
"""

import cv2
import json
import mediapipe as mp
import numpy as np
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
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
    classify_window_confidence,
    NeutralCalibrator,
    WindowAccumulator,
    _z_score,
    FACE_MODEL_PATH,
    POSE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    POSE_NOSE,
    POSE_SHOULDER_L,
    POSE_SHOULDER_R,
)
from stage1_step7_consent import run_consent_gate

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
TRIALS_PATH = os.path.join(LOG_DIR, "gate2_trials.jsonl")
SCHEMA_VERSION = "1.0"
SESSION_ID = str(uuid.uuid4())

MAX_CALIBRATION_REDOS = 2  # rerun_count caps at 2 (0/1/2), then auto-accepts to avoid an infinite loop
HOLD_SECONDS = 10.0        # recorded -- produces ONE window per attempt

# Fixed order. Never randomized. sit-still before fidget, always.
# Instructions rewritten per VBF_DIAGNOSTIC_REPORT.md's finding: V_bf's low
# score was participants not performing a real furrow (verbal-only cue,
# easily misread as an eyebrow RAISE instead of lower-and-together), NOT a
# code/sign bug -- the fix here is concrete, unambiguous instruction text,
# not any change to compute_v_bf or its sign convention.
BLOCKS = [
    (
        "smile",
        [
            "SMILE -- a big, genuine smile",
            "- Like you just saw someone you love",
            "- Let it reach your eyes (real, not posed)",
            "Hold it.",
        ],
    ),
    (
        "furrow",
        [
            "FURROW YOUR BROW",
            "- Pull your eyebrows DOWN and TOGETHER (toward the center)",
            "- Like you're angry, or squinting into bright sun",
            "- The skin between your eyebrows should bunch up",
            "- This is NOT raising your eyebrows, and NOT just a mouth-frown",
            "Hold it until told to stop.",
        ],
    ),
    (
        "concentrate",
        [
            "CONCENTRATE HARD",
            "- Focus like you're solving a difficult math problem in your head",
            "- Let your brow tense naturally with the effort",
            "Hold it.",
        ],
    ),
    (
        "sit-still",
        [
            "SIT COMPLETELY STILL",
            "- Like posing for a photo. Don't move.",
            "Hold it.",
        ],
    ),
    (
        "fidget",
        [
            "FIDGET / SHIFT AROUND",
            "- Move in your seat, adjust yourself, like you're restless",
            "Keep moving until told to stop.",
        ],
    ),
]

WINDOW_NAME = "Gate 2 capture -- press Q at ANY time to ABORT and DISCARD"
FONT = cv2.FONT_HERSHEY_SIMPLEX
# One neutral color throughout. Never green/red keyed to any vector's
# value or direction -- that would be an implicit pass/fail signal.
NEUTRAL_COLOR = (0, 220, 220)
FLAG_COLOR = (0, 150, 255)


def compute_frame(face_landmarker, pose_landmarker, frame, elapsed_ms, pd_buffer):
    """One frame -> raw composite + raw uncomposited components + yaw +
    detection flags. Shared by calibration and block recording so the
    same detection/geometry code path runs both places."""
    clahe_frame = apply_clahe(frame)
    rgb = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    face_result = face_landmarker.detect_for_video(mp_image, elapsed_ms)
    pose_result = pose_landmarker.detect_for_video(mp_image, elapsed_ms)

    h, w = frame.shape[:2]
    out = {
        "face_detected": False,
        "pose_detected": False,
        "yaw_deg": None,
        "composite": {"v_bf": None, "v_es": None, "v_pd": None},
        "v_jc_composite": None,
        "v_bf_raw": {}, "v_es_raw": {}, "v_jc_raw": {}, "v_pd_raw": {},
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

        out["face_detected"] = True
        out["yaw_deg"] = yaw
        out["composite"]["v_bf"] = v_bf
        out["composite"]["v_es"] = v_es
        out["v_jc_composite"] = v_jc
        out["v_bf_raw"] = bf_c
        out["v_es_raw"] = es_c
        out["v_jc_raw"] = jc_c

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
        out["pose_detected"] = True
        out["composite"]["v_pd"] = v_pd
        out["v_pd_raw"] = pd_c

    return out


CALIBRATION_GET_READY_HEADER = "CALIBRATION -- hold a neutral face for 25 seconds."
CALIBRATION_GET_READY_BULLETS = [
    "Relax your face completely",
    "Do NOT smile or frown",
    "Look at the screen like you're watching TV",
    "Don't talk, don't move",
]
CALIBRATION_GET_READY_FOOTER = "Press SPACE when ready."
CALIBRATION_LIVE_REMINDER = "Keep your face relaxed -- neutral"


def show_calibration_get_ready(cap):
    """Participant-facing pre-calibration screen, shown before every
    calibration attempt (including redos, so a re-settle actually
    happens). Waits for SPACE so the participant starts settled instead
    of being rushed straight into the 25s capture. Display/instruction
    text only -- no calibration logic, no timing change. Returns False
    on abort (Q)."""
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = cv2.flip(frame, 1)
        cv2.putText(disp, CALIBRATION_GET_READY_HEADER, (10, 32), FONT, 0.65, NEUTRAL_COLOR, 2)
        y = 68
        for bullet in CALIBRATION_GET_READY_BULLETS:
            cv2.putText(disp, f"- {bullet}", (20, y), FONT, 0.52, NEUTRAL_COLOR, 1)
            y += 26
        cv2.putText(disp, CALIBRATION_GET_READY_FOOTER, (10, y + 18), FONT, 0.6, NEUTRAL_COLOR, 2)
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key == ord("q"):
            return False


def run_calibration_with_redo(cap, face_landmarker, pose_landmarker, stream_start):
    """Returns (reference, rerun_count) or (None, rerun_count) on abort."""
    rerun_count = 0
    while True:
        if not show_calibration_get_ready(cap):
            return None, rerun_count

        calibrator = NeutralCalibrator()
        pd_buffer = deque()
        print(f"\n=== CALIBRATION (attempt {rerun_count + 1}) ===")
        print("Relax your face completely: jaw loose, as if resting alone.")

        while not calibrator.is_calibrated():
            ok, frame = cap.read()
            if not ok:
                continue
            cycle_start = time.perf_counter()
            elapsed_ms = int((cycle_start - stream_start) * 1000)
            f = compute_frame(face_landmarker, pose_landmarker, frame, elapsed_ms, pd_buffer)

            calibrator.add_sample(
                cycle_start,
                f["composite"],
                {
                    "v_jc": f["v_jc_composite"],
                    "v_bf_convergence_ratio": f["v_bf_raw"].get("convergence_ratio"),
                    "v_es_cheek_raise": f["v_es_raw"].get("cheek_raise"),
                },
                f["yaw_deg"],
            )

            reference = None
            if calibrator.should_complete(cycle_start):
                reference = calibrator.complete(cycle_start)

            remaining = calibrator.seconds_remaining(cycle_start)
            disp = cv2.flip(frame, 1)
            # persistent participant-facing reminder (short, always visible)
            cv2.putText(disp, CALIBRATION_LIVE_REMINDER, (10, 25), FONT, 0.55, NEUTRAL_COLOR, 1)
            # prominent countdown, 25 -> 0 -- same seconds_remaining() the
            # calibrator already computes, just displayed larger
            cv2.putText(disp, f"{remaining:.0f}s remaining", (10, 65), FONT, 0.85, NEUTRAL_COLOR, 2)
            if f["face_detected"]:
                cv2.putText(
                    disp,
                    f"raw: V_bf={f['composite']['v_bf']:+.3f}  V_es={f['composite']['v_es']:+.3f}",
                    (10, 100), FONT, 0.5, NEUTRAL_COLOR, 1,
                )
            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None, rerun_count

            if reference is not None:
                break

        tag = "POSSIBLY NOT NEUTRAL" if reference["quality"]["possibly_not_neutral"] else "OK"
        print(f"Calibration [{tag}]")
        for r in reference["quality"]["reasons"]:
            print(f"  FLAG: {r}")

        if rerun_count >= MAX_CALIBRATION_REDOS:
            print(f"Max redos ({MAX_CALIBRATION_REDOS}) reached -- auto-accepting this calibration.")
            return reference, rerun_count

        # review screen -- accept or redo
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            disp = cv2.flip(frame, 1)
            cv2.putText(disp, f"CALIBRATION [{tag}] -- A=accept  R=redo  Q=abort", (10, 25), FONT, 0.55, NEUTRAL_COLOR, 2)
            y = 55
            for k, s in reference["composite"].items():
                cv2.putText(disp, f"{k}: mean={s['mean']:+.4f} std={s['std']:.4f}", (10, y), FONT, 0.5, NEUTRAL_COLOR, 1)
                y += 22
            for rline in reference["quality"]["reasons"]:
                cv2.putText(disp, rline[:70], (10, y), FONT, 0.42, FLAG_COLOR, 1)
                y += 18
            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("a"):
                return reference, rerun_count
            if key == ord("r"):
                rerun_count += 1
                break
            if key == ord("q"):
                return None, rerun_count


def show_block_get_ready(cap, face_landmarker, pose_landmarker, stream_start, instruction_lines, pd_buffer):
    """Participant-facing pre-block screen -- same pattern as
    show_calibration_get_ready: show the concrete instruction, wait for
    SPACE, THEN the caller begins the 10s hold. Replaces the old fixed
    2.5s auto-advance lead-in -- the keypress is the readiness signal
    instead of a timer, so the participant sets the expression before
    recording starts rather than mid-countdown. Still not recorded
    (frames are computed here only to keep pd_buffer warm for V_pd's
    short rolling window, matching the prior lead-in's behavior).
    Returns False on abort (Q)."""
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cycle_start = time.perf_counter()
        elapsed_ms = int((cycle_start - stream_start) * 1000)
        compute_frame(face_landmarker, pose_landmarker, frame, elapsed_ms, pd_buffer)  # keeps pd_buffer warm; not stored

        disp = cv2.flip(frame, 1)
        cv2.putText(disp, instruction_lines[0], (10, 30), FONT, 0.65, NEUTRAL_COLOR, 2)
        y = 62
        for line in instruction_lines[1:]:
            cv2.putText(disp, line, (20, y), FONT, 0.48, NEUTRAL_COLOR, 1)
            y += 24
        cv2.putText(disp, "(not recorded) Press SPACE when set.", (10, y + 14), FONT, 0.55, NEUTRAL_COLOR, 2)
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key == ord("q"):
            return False


def run_block_attempt(cap, face_landmarker, pose_landmarker, stream_start, instruction_lines):
    """Get-ready (not recorded, keypress-gated) then a HOLD_SECONDS
    recording. Returns a list of per-sample dicts, or None on abort."""
    pd_buffer = deque()

    if not show_block_get_ready(cap, face_landmarker, pose_landmarker, stream_start, instruction_lines, pd_buffer):
        return None

    samples = []
    rec_start = time.perf_counter()
    while time.perf_counter() - rec_start < HOLD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue
        cycle_start = time.perf_counter()
        elapsed_ms = int((cycle_start - stream_start) * 1000)
        f = compute_frame(face_landmarker, pose_landmarker, frame, elapsed_ms, pd_buffer)

        samples.append(
            {
                "detected": f["face_detected"],
                "yaw_deg": f["yaw_deg"],
                "z": {"v_bf": None, "v_es": None, "v_pd": None},  # filled in by caller (needs calibration ref)
                "raw_composite": dict(f["composite"]),
                "v_bf_raw": dict(f["v_bf_raw"]),
                "v_es_raw": dict(f["v_es_raw"]),
                "v_jc_raw": dict(f["v_jc_raw"]),
                "v_pd_raw": dict(f["v_pd_raw"]),
            }
        )

        remaining = HOLD_SECONDS - (time.perf_counter() - rec_start)
        disp = cv2.flip(frame, 1)
        cv2.putText(disp, f"HOLD: {instruction_lines[0]}", (10, 25), FONT, 0.6, NEUTRAL_COLOR, 2)
        cv2.putText(disp, f"recording -- {remaining:.1f}s left", (10, 50), FONT, 0.5, NEUTRAL_COLOR, 1)
        if f["face_detected"]:
            cv2.putText(
                disp, f"raw: V_bf={f['composite']['v_bf']:+.3f}  V_es={f['composite']['v_es']:+.3f}",
                (10, 75), FONT, 0.5, NEUTRAL_COLOR, 1,
            )
        if f["pose_detected"]:
            v_pd_val = f["composite"]["v_pd"]
            cv2.putText(
                disp, f"raw: V_pd={'n/a' if v_pd_val is None else f'{v_pd_val:+.5f}'}",
                (10, 100), FONT, 0.5, NEUTRAL_COLOR, 1,
            )
        cv2.imshow(WINDOW_NAME, disp)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return None

    return samples


def build_trial_record(person_label, block_name, attempt_number, samples, calibrator_reference, calibration_rerun_count):
    n_samples = len(samples)
    n_detected = sum(1 for s in samples if s["detected"])
    detect_rate = (n_detected / n_samples) if n_samples else 0.0
    yaw_vals = [s["yaw_deg"] for s in samples if s["yaw_deg"] is not None]
    yaw_variance = float(np.var(yaw_vals)) if len(yaw_vals) >= 2 else None
    low_conf, reasons = classify_window_confidence(detect_rate, yaw_variance)

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "gate2_trial",
        "session_id": SESSION_ID,
        "person_label": person_label,
        "commanded_label": block_name,
        "attempt_number": attempt_number,
        "accepted": False,  # set True by caller only for the accepted attempt
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "hold_seconds": HOLD_SECONDS,
        "window_z": {
            "v_bf": WindowAccumulator._stats(samples, lambda s: s["z"]["v_bf"]),
            "v_es": WindowAccumulator._stats(samples, lambda s: s["z"]["v_es"]),
            "v_pd": WindowAccumulator._stats(samples, lambda s: s["z"]["v_pd"]),
        },
        "raw_components": {
            "v_bf": {
                "convergence_ratio": WindowAccumulator._stats(samples, lambda s: s["v_bf_raw"].get("convergence_ratio")),
                "drop_ratio": WindowAccumulator._stats(samples, lambda s: s["v_bf_raw"].get("drop_ratio")),
            },
            "v_es": {
                "aperture": WindowAccumulator._stats(samples, lambda s: s["v_es_raw"].get("aperture")),
                "cheek_raise": WindowAccumulator._stats(samples, lambda s: s["v_es_raw"].get("cheek_raise")),
            },
            "v_jc": {
                "inter_lip_dist": WindowAccumulator._stats(samples, lambda s: s["v_jc_raw"].get("inter_lip_dist")),
                "lip_corner_dist": WindowAccumulator._stats(samples, lambda s: s["v_jc_raw"].get("lip_corner_dist")),
                "masseter_width": WindowAccumulator._stats(samples, lambda s: s["v_jc_raw"].get("masseter_width")),
            },
            "v_pd": {
                "nose_pos_variance": WindowAccumulator._stats(samples, lambda s: s["v_pd_raw"].get("nose_pos_variance")),
                "shoulder_pos_variance": WindowAccumulator._stats(samples, lambda s: s["v_pd_raw"].get("shoulder_pos_variance")),
            },
        },
        "calibration_baseline": calibrator_reference["composite"],
        "calibration_contamination_flags": {
            "possibly_not_neutral": calibrator_reference["quality"]["possibly_not_neutral"],
            "drifted_vectors": calibrator_reference["quality"]["drifted_vectors"],
        },
        "calibration_rerun_count": calibration_rerun_count,
        "window_confidence": {
            "low_confidence": low_conf,
            "reasons": reasons,
            "detection_rate": detect_rate,
            "yaw_variance_deg2": yaw_variance,
        },
    }


def review_block(cap, trial, block_name):
    """Shows the RAW windowed numbers only -- no interpretation, no
    pass/fail, no color keyed to value. Waits for accept/redo/abort.

    The static reminder below exists because the accept/redo decision
    is the one place self-tuning bias can re-enter an otherwise
    deliberately-dumb tool -- it pins the rule at the decision point,
    not just in a doc the operator read once."""
    REMINDER_LINES = [
        "Accept if they PERFORMED and HELD the expression.",
        "Redo ONLY for fumbles (sneeze, talk, broke the hold).",
        "Do NOT decide based on the numbers.",
    ]
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = cv2.flip(frame, 1)
        cv2.putText(
            disp, f"'{block_name}' complete (attempt {trial['attempt_number']}) -- A=accept  R=redo  Q=abort",
            (10, 25), FONT, 0.52, NEUTRAL_COLOR, 2,
        )
        y = 50
        for line in REMINDER_LINES:
            cv2.putText(disp, line, (10, y), FONT, 0.45, FLAG_COLOR, 1)
            y += 18
        y += 12
        for k in ("v_bf", "v_es", "v_pd"):
            s = trial["window_z"][k]
            avg = f"{s['avg']:+.3f}" if s["avg"] is not None else "n/a"
            peak = f"{s['peak']:+.3f}" if s["peak"] is not None else "n/a"
            cv2.putText(disp, f"{k}(z): avg={avg}  peak={peak}", (10, y), FONT, 0.5, NEUTRAL_COLOR, 1)
            y += 22
        wc = trial["window_confidence"]
        cv2.putText(
            disp, f"window: detect_rate={wc['detection_rate']:.2f}  low_confidence={wc['low_confidence']}",
            (10, y), FONT, 0.45, NEUTRAL_COLOR, 1,
        )
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("a"):
            return "accept"
        if key == ord("r"):
            return "redo"
        if key == ord("q"):
            return "abort"


def main():
    consented, person_label = run_consent_gate(SESSION_ID)
    if not consented:
        return

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

    calibrator_reference, calibration_rerun_count = run_calibration_with_redo(
        cap, face_landmarker, pose_landmarker, stream_start
    )

    trial_buffer = []  # in-memory only -- flushed to disk ONLY if the full sequence completes without abort
    aborted = calibrator_reference is None

    if not aborted:
        for block_name, instruction_lines in BLOCKS:
            attempt_number = 1
            while True:
                samples = run_block_attempt(cap, face_landmarker, pose_landmarker, stream_start, instruction_lines)
                if samples is None:
                    aborted = True
                    break

                std_bf = calibrator_reference["composite"]["v_bf"]["std"]
                std_es = calibrator_reference["composite"]["v_es"]["std"]
                std_pd = calibrator_reference["composite"]["v_pd"]["std"]
                mean_bf = calibrator_reference["composite"]["v_bf"]["mean"]
                mean_es = calibrator_reference["composite"]["v_es"]["mean"]
                mean_pd = calibrator_reference["composite"]["v_pd"]["mean"]
                for s in samples:
                    raw = s["raw_composite"]
                    s["z"]["v_bf"] = _z_score(raw["v_bf"] - mean_bf, std_bf) if raw["v_bf"] is not None and mean_bf is not None else None
                    s["z"]["v_es"] = _z_score(raw["v_es"] - mean_es, std_es) if raw["v_es"] is not None and mean_es is not None else None
                    s["z"]["v_pd"] = _z_score(raw["v_pd"] - mean_pd, std_pd) if raw["v_pd"] is not None and mean_pd is not None else None

                trial = build_trial_record(
                    person_label, block_name, attempt_number, samples, calibrator_reference, calibration_rerun_count
                )

                decision = review_block(cap, trial, block_name)
                if decision == "abort":
                    trial["accepted"] = False
                    trial_buffer.append(trial)
                    aborted = True
                    break
                if decision == "accept":
                    trial["accepted"] = True
                    trial_buffer.append(trial)
                    break
                # redo
                trial["accepted"] = False
                trial_buffer.append(trial)
                attempt_number += 1

            if aborted:
                break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    pose_landmarker.close()

    if aborted:
        print(f"\n=== SESSION ABORTED -- discarding all {len(trial_buffer)} captured trial(s) for this run. Nothing written. ===\n")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        for trial in trial_buffer:
            f.write(json.dumps(trial) + "\n")

    counts = {b: sum(1 for t in trial_buffer if t["commanded_label"] == b) for b, _ in BLOCKS}
    print(f"\n=== SESSION COMPLETE. {len(trial_buffer)} trial record(s) written to {TRIALS_PATH} ===")
    print("Attempts per block: " + ", ".join(f"{b}={n}" for b, n in counts.items()))
    print(f"Calibration rerun_count={calibration_rerun_count}, flagged_vectors={[d['vector'] for d in calibrator_reference['quality']['drifted_vectors']]}")


if __name__ == "__main__":
    main()
