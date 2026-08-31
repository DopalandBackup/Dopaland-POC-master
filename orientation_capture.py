"""
Orientation-cluster validation-prep capture tool.

PILOT STATUS (documentation only): this is a validation/study tool for
V_so, a PILOT feature, NOT POC-ready. Directed testing with this tool
found the orientation signal is PITCH-BLIND: yaw (left/right) is
detected reliably, but pitch (looking up/down) is not -- a structural
limit of single-camera landmark head-pose, not a bug (see
stage1_step4_vectors.py's module docstring and compute_v_so's
docstring for the full finding). V_so remains unvalidated overall and
is not wired into the demo UI. This tool stays dumb -- see below --
and this comment changes no logic.

DUMB CAPTURE TOOL. Records labeled orientation segments per person and
stores them. Computes NO pass/fail, NO score, NO accuracy percentage,
NO "correct direction" judgement of any kind, anywhere. Scoring happens
later, separately, by a human against a frozen rule this tool never
sees -- same discipline as stage1_step9_gate2_capture.py (Decision 14):
keeping this tool blind to what "correct" means is load-bearing for
whatever validation result comes out of it being credible. If you are
reading this file to add a checkmark, a percentage, or a threshold
comparison, stop -- that is the exact failure this design forbids.

Reuses, does not reimplement: the consent gate + anonymous participant
code (stage1_step7_consent.run_consent_gate), capture_thread (T1,
completely unmodified), and the orientation pipeline itself --
apply_clahe, pose_normalize, yaw_pitch_roll_from_matrix, compute_v_so,
AttentionWindowAccumulator, classify_window_confidence -- all imported
from stage1_step4_vectors.py, none of it re-derived here. Only
FaceLandmarker is created in this tool (no PoseLandmarker): V_so needs
head pose + iris/eye-corner landmarks only, never pose_world_landmarks,
so running a pose model here would be pure wasted work.

TWO-THREAD ARCHITECTURE (Mandatory Architecture #1, kept intact): T1 is
s1.capture_thread, reused unmodified -- capture-only, never touches
detection. T2 is orientation_processing_thread below -- all detection,
segment orchestration, display, and disk I/O; it reads frames from the
shared s1.latest_frame buffer under s1.frame_lock and never opens the
camera itself. (stage1_step9_gate2_capture.py predates this split and
reads the camera directly in one thread; this tool does not repeat
that -- the task asked for capture off the capture thread here.)

NO CALIBRATION: unlike Gate 2's affect vectors, V_so needs no per-person
neutral baseline (see stage1_step4_vectors.py's module docstring --
"oriented toward screen" is a universal geometric threshold, same
category as the quality gate's yaw>35deg check, not a deviation-from-
neutral concept). The fixed segment sequence starts immediately after
consent; there is no calibration step to add.

Fixed segment order, identical for every person, never randomized:
    look_at_screen -> look_left -> look_right -> look_down -> look_up ->
    look_away_and_back
Each segment is a ~10s HOLD_SECONDS recording, using AttentionWindowAccumulator
exactly as the live tool uses it (one fresh accumulator per segment,
fed every frame, flushed once at the end) -- not a new windowing
mechanism.

Mid-capture abort (Q at any time) discards every segment captured in
this run for this person -- nothing is written to disk, same safety
pattern as Gate 2's capture tool and the same consent principle
(a person who says stop mid-capture gets everything discarded, not
partially kept).

HONEST FRAMING: every instruction, label, and logged field describes a
geometric direction ("look left", "screen orientation", "look-away
rate") -- never "attention", "engagement", "focus", or "distraction".
Every record is stamped "unvalidated": true; this tool produces raw
material for a later, separate, human-scored validation study, not a
validated reading of anything.

RAW HEAD-POSE VISIBILITY (added to resolve PITCH_DIAGNOSTIC.md's open
question): the operator's live view now shows raw yaw/pitch/roll --
the exact yaw_pitch_roll_from_matrix values compute_v_so already
consumes, surfaced, never recomputed -- large and readable, next to the
instantaneous screen_orientation value and its oriented/not-oriented
verdict, so a human watching can see directly whether pitch moves on a
real "look down"/"look up" or stays near zero. Per-segment avg/min/max/
variance for all three angles are also logged into each trial record
(new field, existing fields untouched) so this can be checked after
the fact too, not just watched live. This is display + logging ONLY --
compute_v_so, its thresholds, and which axis it weights are unchanged.
"""

import cv2
import json
import mediapipe as mp
import numpy as np
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

import stage1_step4_vectors as s1
from stage1_step4_vectors import (
    apply_clahe,
    yaw_pitch_roll_from_matrix,
    pose_normalize,
    compute_v_so,
    AttentionWindowAccumulator,
    classify_window_confidence,
    FACE_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)
from stage1_step7_consent import run_consent_gate

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
TRIALS_PATH = os.path.join(LOG_DIR, "orientation_trials.jsonl")
SCHEMA_VERSION = "1.1"  # 1.1: adds "raw_head_pose_deg" (new field -- yaw/pitch/roll avg/min/max/variance
                         # per segment, PITCH_DIAGNOSTIC.md follow-up). Existing fields untouched;
                         # historical 1.0 records simply lack this field.
SESSION_ID = str(uuid.uuid4())

HOLD_SECONDS = 10.0  # one AttentionWindowAccumulator flush per segment -- same 10s scale the live tool windows on, not a new number

# Fixed order. Never randomized. Every line below is a plain directional
# instruction -- no mental-state word anywhere (honest-framing rule).
SEGMENTS = [
    (
        "look_at_screen",
        [
            "LOOK AT THE SCREEN",
            "- Look straight ahead at the screen in front of you",
            "- Sit normally, like you're reading something on it",
            "Hold it.",
        ],
    ),
    (
        "look_left",
        [
            "LOOK LEFT",
            "- Turn your head to look to your left, away from the screen",
            "Hold it.",
        ],
    ),
    (
        "look_right",
        [
            "LOOK RIGHT",
            "- Turn your head to look to your right, away from the screen",
            "Hold it.",
        ],
    ),
    (
        "look_down",
        [
            "LOOK DOWN",
            "- Tilt your head down, as if looking at your lap or a phone",
            "Hold it.",
        ],
    ),
    (
        "look_up",
        [
            "LOOK UP",
            "- Tilt your head up, as if looking at the ceiling",
            "Hold it.",
        ],
    ),
    (
        "look_away_and_back",
        [
            "LOOK AWAY AND BACK, REPEATEDLY",
            "- Alternate: look away (any direction), then back at the screen",
            "- Repeat this several times until told to stop",
            "Keep alternating.",
        ],
    ),
]

WINDOW_NAME = "Orientation capture -- press Q at ANY time to ABORT and DISCARD"
FONT = cv2.FONT_HERSHEY_SIMPLEX
# One neutral color throughout. Never green/red keyed to any value -- that
# would be an implicit pass/fail signal (same rule as gate2_capture.py).
NEUTRAL_COLOR = (0, 220, 220)


def compute_frame_orientation(face_landmarker, frame, elapsed_ms):
    """One frame -> compute_v_so's output PLUS the raw yaw/pitch/roll it
    was given. V_so needs FaceLandmarker only (yaw/pitch from the
    facial transformation matrix + iris/eye-corner landmarks) -- reuses
    apply_clahe/pose_normalize/yaw_pitch_roll_from_matrix/compute_v_so
    exactly as the live tool does, nothing recomputed differently here.
    pitch_deg/roll_deg are surfaced from the SAME
    yaw_pitch_roll_from_matrix call compute_v_so already consumes (that
    call is not duplicated) -- this function simply stops discarding
    two of its three return values."""
    clahe_frame = apply_clahe(frame)
    rgb = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    face_result = face_landmarker.detect_for_video(mp_image, elapsed_ms)

    h, w = frame.shape[:2]
    out = {"face_detected": False, "yaw_deg": None, "pitch_deg": None, "roll_deg": None, "v_so": None, "so_components": None}
    if face_result.face_landmarks and face_result.facial_transformation_matrixes:
        lms = face_result.face_landmarks[0]
        matrix = face_result.facial_transformation_matrixes[0]
        normalized_pts = pose_normalize(lms, matrix, w, h)
        yaw, pitch, roll = yaw_pitch_roll_from_matrix(matrix)
        v_so, so_components = compute_v_so(normalized_pts, yaw, pitch)
        out.update({
            "face_detected": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "v_so": v_so,
            "so_components": so_components,
        })
    return out


def show_segment_get_ready(instruction_lines):
    """Participant-facing pre-segment screen (like Gate 2's calibration
    get-ready screen) -- shows the instruction, waits for SPACE as the
    readiness signal, NOT recorded (mirrors gate2_capture.py's
    show_block_get_ready). Reads frames from the shared s1.latest_frame
    buffer -- T1 owns the webcam, this thread never calls
    cv2.VideoCapture itself. Returns False on abort (Q)."""
    while True:
        with s1.frame_lock:
            frame = s1.latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        disp = cv2.flip(frame, 1)
        cv2.putText(disp, instruction_lines[0], (10, 30), FONT, 0.65, NEUTRAL_COLOR, 2)
        y = 62
        for line in instruction_lines[1:]:
            cv2.putText(disp, line, (20, y), FONT, 0.48, NEUTRAL_COLOR, 1)
            y += 24
        cv2.putText(disp, "(not recorded) Press SPACE when ready.", (10, y + 14), FONT, 0.55, NEUTRAL_COLOR, 2)
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key == ord("q"):
            return False


def _live_render_and_check_abort(frame, f, remaining, instruction_lines):
    """Operator view during recording (Decision-45 pattern: the human
    watches raw values live; the tool itself makes no judgement about
    them). One neutral color, no highlighting of "good"/"bad" values.

    Shows RAW yaw/pitch/roll large and readable -- the exact
    yaw_pitch_roll_from_matrix values compute_v_so already consumes,
    not recomputed -- plus a line pairing the instantaneous pitch
    reading directly against the current screen_orientation value and
    its oriented/not-oriented verdict (PITCH_DIAGNOSTIC.md's open
    question: does pitch move to a large value on a real "look down",
    or does it stay near zero even then? nobody could see this before
    because it was never displayed anywhere)."""
    disp = cv2.flip(frame, 1)
    cv2.putText(disp, f"RECORDING: {instruction_lines[0]}", (10, 25), FONT, 0.6, NEUTRAL_COLOR, 2)
    cv2.putText(disp, f"{remaining:.1f}s left", (10, 50), FONT, 0.5, NEUTRAL_COLOR, 1)
    if f["face_detected"]:
        so = f["so_components"]
        cv2.putText(
            disp,
            f"raw: V_so={f['v_so']:.2f}  yaw={f['yaw_deg']:+.1f}  gaze_reliable={so['gaze_reliable']}",
            (10, 75), FONT, 0.48, NEUTRAL_COLOR, 1,
        )
        # Large/readable raw head-pose line -- the diagnostic surface this
        # change adds. Same values as the line above; shown bigger and
        # separately because this is the number the operator needs to
        # actually read at a glance while holding a pose.
        cv2.putText(
            disp,
            f"yaw: {f['yaw_deg']:+6.1f} deg   pitch: {f['pitch_deg']:+6.1f} deg   roll: {f['roll_deg']:+6.1f} deg",
            (10, 108), FONT, 0.62, NEUTRAL_COLOR, 2,
        )
        oriented_text = "ORIENTED" if so["oriented"] else "not-oriented"
        cv2.putText(
            disp,
            f"pitch: {f['pitch_deg']:+.1f} deg  |  orientation: {f['v_so']:.2f} {oriented_text}",
            (10, 136), FONT, 0.52, NEUTRAL_COLOR, 1,
        )
    else:
        cv2.putText(disp, "no face detected", (10, 75), FONT, 0.48, NEUTRAL_COLOR, 1)
    cv2.imshow(WINDOW_NAME, disp)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def record_segment(face_landmarker, stream_start, instruction_lines, render_fn=_live_render_and_check_abort):
    """Records ~HOLD_SECONDS of samples for ONE segment via
    AttentionWindowAccumulator (imported, not reimplemented) -- fed
    exactly the same per-sample shape the live tool feeds it, just for
    a fixed HOLD_SECONDS instead of running forever, then flushed once.
    render_fn is injected (defaults to the real interactive display +
    Q-abort check) so the core recording logic is directly testable
    against a recorded clip without a GUI event loop -- get-ready above
    is NOT recorded, so skipping it in a test changes nothing about the
    data. Also collects raw yaw/pitch/roll per sample (PITCH_DIAGNOSTIC.md
    follow-up) alongside the existing AttentionWindowAccumulator feed --
    a second, independent collection, not a change to what's fed into
    the accumulator or into compute_v_so. Returns (window_summary,
    raw_angle_samples) or (None, None) on abort; raw_angle_samples is
    {"yaw_deg": [...], "pitch_deg": [...], "roll_deg": [...]}."""
    acc = AttentionWindowAccumulator()
    yaw_vals, pitch_vals, roll_vals = [], [], []
    rec_start = time.perf_counter()

    while time.perf_counter() - rec_start < HOLD_SECONDS:
        with s1.frame_lock:
            frame = s1.latest_frame
        if frame is None:
            time.sleep(0.005)
            continue

        cycle_start = time.perf_counter()
        elapsed_ms = int((cycle_start - stream_start) * 1000)
        f = compute_frame_orientation(face_landmarker, frame, elapsed_ms)

        if f["face_detected"]:
            yaw_vals.append(f["yaw_deg"])
            pitch_vals.append(f["pitch_deg"])
            roll_vals.append(f["roll_deg"])

        so = f["so_components"]
        acc.add_sample(
            cycle_start,
            {
                "detected": f["face_detected"],
                "orientation_score": f["v_so"],
                "oriented": so["oriented"] if so else None,
                "gaze_score": so["gaze_score"] if so else None,
                "gaze_reliable": so["gaze_reliable"] if so else None,
            },
        )

        remaining = HOLD_SECONDS - (time.perf_counter() - rec_start)
        if render_fn(frame, f, remaining, instruction_lines):
            return None, None

    summary = acc.flush(time.perf_counter())
    raw_angle_samples = {"yaw_deg": yaw_vals, "pitch_deg": pitch_vals, "roll_deg": roll_vals}
    return summary, raw_angle_samples


def _raw_angle_stats(vals):
    """avg/min/max/variance for one raw signed angle's samples
    (yaw/pitch/roll, degrees). Deliberately NOT WindowAccumulator._stats
    (imported nowhere near this function) -- that helper's "peak" means
    plain max(), valid only because the affect-vector composites it was
    built for are already sign-flipped so higher=more expression. Raw
    head-pose angles are signed physical values where a large negative
    swing matters exactly as much as a large positive one, so this
    reports min AND max, not one plain "peak". A small, self-contained
    diagnostic utility -- not vector math, not reused/shared with
    anything scored."""
    if not vals:
        return {"avg": None, "min": None, "max": None, "variance": None, "n": 0}
    arr = np.array(vals)
    return {"avg": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()), "variance": float(arr.var()), "n": len(vals)}


def build_trial_record(participant_code, label, segment_index, summary, raw_angle_samples):
    """Reshapes AttentionWindowAccumulator's own output (screen_orientation/
    gaze_direction/look_away_rate, verbatim -- not recomputed) plus a
    quality flag (classify_window_confidence, imported, not reimplemented)
    into one dumb-capture trial record. Computes no score, no pass/fail,
    no threshold comparison against the commanded_label anywhere.

    raw_angle_samples (from record_segment) additionally gets aggregated
    (avg/min/max/variance, via _raw_angle_stats -- new, self-contained,
    not shared with any scored path) into the new "raw_head_pose_deg"
    field so a look_down/look_up segment can be re-examined after
    capture (PITCH_DIAGNOSTIC.md follow-up), not just watched live."""
    yaw_vals = raw_angle_samples["yaw_deg"]
    yaw_variance = float(np.var(yaw_vals)) if len(yaw_vals) >= 2 else None
    low_confidence, reasons = classify_window_confidence(summary["detection_rate"], yaw_variance)

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "orientation_trial",
        "session_id": SESSION_ID,
        "participant_code": participant_code,
        "commanded_label": label,
        "segment_index": segment_index,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "hold_seconds": HOLD_SECONDS,
        "n_samples": summary["n_samples"],
        "n_detected": summary["n_detected"],
        "detection_rate": summary["detection_rate"],
        # THREE orientation fields, reused verbatim from
        # AttentionWindowAccumulator.flush() -- same object shape the live
        # tool logs, not a parallel/rebuilt version.
        "screen_orientation": summary["screen_orientation"],
        "gaze_direction": summary["gaze_direction"],
        "look_away_rate": summary["look_away_rate"],
        # NEW field (schema 1.1, PITCH_DIAGNOSTIC.md follow-up): the raw
        # yaw/pitch/roll compute_v_so was actually given, aggregated per
        # segment -- so "did pitch ever swing large during this look_down"
        # can be checked after capture, not just watched live. Diagnostic
        # only: not itself a behavioral signal, not scored, not compared
        # against commanded_label anywhere in this file.
        "raw_head_pose_deg": {
            "yaw": _raw_angle_stats(raw_angle_samples["yaw_deg"]),
            "pitch": _raw_angle_stats(raw_angle_samples["pitch_deg"]),
            "roll": _raw_angle_stats(raw_angle_samples["roll_deg"]),
            "unvalidated": True,
            "label": (
                "raw head-pose angles (diagnostic), degrees -- the SAME yaw_pitch_roll_from_matrix "
                "values compute_v_so already consumes; not a behavioral signal itself, not scored"
            ),
        },
        # Tracking-quality flag only (was the face reliably tracked, was
        # yaw stable this window?) -- NOT a judgement of whether the
        # commanded direction was performed. Same function Gate 2's
        # capture tool reuses for the identical purpose.
        "window_quality": {
            "low_confidence": low_confidence,
            "reasons": reasons,
            "yaw_variance_deg2": yaw_variance,
        },
        "unvalidated": True,
        "label": (
            "orientation capture trial (geometric) -- NOT attention/engagement/focus/distraction "
            "(mental state); scoring is a separate, later, human-applied pass against a frozen rule "
            "this tool does not contain"
        ),
    }


def orientation_processing_thread(result_holder):
    """Thread 2: all detection, segment orchestration, display, and disk
    I/O. Never touches the webcam (Mandatory Architecture #1) -- reads
    frames from the shared s1.latest_frame buffer T1 (s1.capture_thread,
    reused unmodified) fills. Signals s1.stop_event when done so T1 exits
    too, same shutdown pattern stage3_demo_ui.py uses."""
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
    stream_start = time.perf_counter()
    trial_buffer = []
    aborted = False

    print("[Orientation capture] thread started.")
    for i, (label, instruction_lines) in enumerate(SEGMENTS, start=1):
        if not show_segment_get_ready(instruction_lines):
            aborted = True
            break

        summary, raw_angle_samples = record_segment(face_landmarker, stream_start, instruction_lines)
        if summary is None:
            aborted = True
            break

        trial = build_trial_record(result_holder["participant_code"], label, i, summary, raw_angle_samples)
        trial_buffer.append(trial)
        so, gz, la = trial["screen_orientation"], trial["gaze_direction"], trial["look_away_rate"]
        rhp = trial["raw_head_pose_deg"]
        print(
            f"[{i}/{len(SEGMENTS)}] '{label}' recorded -- "
            f"screen_orientation(avg={so['avg']}, oriented_rate={so['oriented_rate']})  "
            f"gaze_direction(avg={gz['avg']}, gaze_reliable_rate={gz['gaze_reliable_rate']})  "
            f"look_away_rate={la['value']}"
        )
        print(
            f"    raw head pose (deg) -- "
            f"yaw(avg={rhp['yaw']['avg']}, min={rhp['yaw']['min']}, max={rhp['yaw']['max']})  "
            f"pitch(avg={rhp['pitch']['avg']}, min={rhp['pitch']['min']}, max={rhp['pitch']['max']})  "
            f"roll(avg={rhp['roll']['avg']}, min={rhp['roll']['min']}, max={rhp['roll']['max']})"
        )

    face_landmarker.close()
    s1.stop_event.set()

    result_holder["trial_buffer"] = trial_buffer
    result_holder["aborted"] = aborted
    print("[Orientation capture] thread stopped.")


def main():
    consented, participant_code = run_consent_gate(SESSION_ID)
    if not consented:
        return

    result_holder = {"participant_code": participant_code, "trial_buffer": [], "aborted": True}

    t1 = threading.Thread(target=s1.capture_thread, name="CaptureThread")
    t2 = threading.Thread(target=orientation_processing_thread, args=(result_holder,), name="OrientationProcessingThread")
    t1.start()
    t2.start()

    print("Press 'q' at any time during a segment to ABORT and discard this session.")
    t2.join()
    s1.stop_event.set()  # idempotent -- harmless if T2 already set it
    t1.join()
    cv2.destroyAllWindows()

    trial_buffer = result_holder["trial_buffer"]
    aborted = result_holder["aborted"]

    if aborted:
        print(f"\n=== SESSION ABORTED -- discarding all {len(trial_buffer)} captured segment(s) for this run. Nothing written. ===\n")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(TRIALS_PATH, "a", encoding="utf-8") as f:
        for trial in trial_buffer:
            f.write(json.dumps(trial) + "\n")

    labels = [t["commanded_label"] for t in trial_buffer]
    print(f"\n=== SESSION COMPLETE. {len(trial_buffer)} segment(s) written to {TRIALS_PATH} ===")
    print("Segments recorded, in order: " + " -> ".join(labels))


if __name__ == "__main__":
    main()
