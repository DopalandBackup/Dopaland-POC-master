"""
analyze_video.py -- Steps 1+2 of the video-analysis feature: a
standalone, OFFLINE, single-video processor. Runs a recorded video
through the SAME pipeline the live tool uses -- CLAHE -> FaceLandmarker/
PoseLandmarker -> the existing vector computations -> the existing
calibration/window/z-scoring/mapping -- and produces a behavioral-
signal timeline + a templated (non-agent) report.

Every piece of math below is IMPORTED from the existing modules, never
reimplemented:
  stage1_step4_vectors.py  -- apply_clahe, pose_normalize,
                               interocular_distance, yaw_pitch_roll_from_matrix,
                               compute_v_bf/v_es/v_jc/v_pd, NeutralCalibrator,
                               WindowAccumulator, map_to_valence_arousal,
                               DETECT_RATE_FLOOR, CALIBRATION_SECONDS, ...
  stage2_personality_agent.py -- zscore_window (pure math, no agent call)

MODE A (calibrated): the first CALIBRATION_SECONDS of the video is used
as a per-person neutral baseline (NeutralCalibrator, unchanged), and the
rest of the video is analyzed as a validated deviation from it -- the
exact Step-1 pipeline.

MODE B (uncalibrated fallback, Step 2): for videos with no usable
neutral segment (movies, arbitrary clips) -- the SAME pipeline runs over
the WHOLE video with NO calibration phase, fed a fixed
POPULATION_DEFAULT_REFERENCE (mean=0, std=1) instead of a real
per-person baseline through the SAME zscore_window/map_to_valence_arousal
calls. Because there is no real per-person spread, std=1 makes the
resulting numbers equal to the raw, unscaled composite reading -- Mode B
never invents a per-person statistic it has no evidence for (the
cross-person baseline problem, Gate 2 / Pitfall #2: one person's neutral
brow is another's furrow). Every Mode B output is stamped "validated":
false plus a prominent uncalibrated/approximate label -- see MODE_B_LABEL.

Mode selection: analyze_video() tries Mode A first (unless --uncalibrated
forces Mode B directly); if the first CALIBRATION_SECONDS doesn't yield
a usable neutral (too short, or the existing contamination gate flags
it), it automatically falls back to Mode B rather than hard-failing.

STEP 3 (this step) adds two things, both built ONLY on top of Steps 1-2
-- no vector math/calibration/windowing/z-scoring/V-A-mapping/Mode-A-B
logic changed:
  (A) --folder batch processing: point at a directory of videos instead
      of one file; each is run through the exact same single-video
      pipeline, failures are caught per-video and never stop the batch.
  (B) A warm, agent-written whole-video report in the SAME pointer+
      detail format the live tool's read uses (PLEASURE/AROUSAL/
      NOT_INTERPRETED/CONFIDENCE headline+detail), built by calling the
      SAME call_agent() swap point stage2_personality_agent.py already
      defines (stub by default, real API only behind --live) and parsed
      by the SAME parse_agent_pointers(). The agent is fed ONLY this
      file's own already-computed aggregate numbers + notable-moment
      list -- never the video or any frame; it writes, it does not
      perceive. Mode B's validated=False and uncalibrated/approximate
      framing are explicitly given to the agent and it is instructed to
      state them -- see _build_video_report_prompt.

Usage:
    python analyze_video.py path/to/video.mp4
    python analyze_video.py path/to/movie_clip.mp4 --uncalibrated
    python analyze_video.py path/to/video.mp4 --live
    python analyze_video.py --folder path/to/videos [--uncalibrated] [--live]

HONEST FRAMING (unchanged from the live tool, mandatory here too):
outputs are behavioral signals / affective indicators, never "emotion",
never clinical/diagnostic language. Valence is pleasure-side only (V_es,
single-source -- there is NO validated pain/negative-valence axis).
Arousal is V_pd only. V_bf (brow furrow) and V_jc (jaw compression) are
logged-only, never interpreted, never part of Valence/Arousal (Decision
17/18 -- see stage1_step4_vectors.map_to_valence_arousal).
"""

import cv2
import json
import mediapipe as mp
import numpy as np
import os
import sys
import uuid
from collections import deque
from datetime import datetime, timezone
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

import stage1_step4_vectors as s1
from features import geometry, x_core, episodes, attention
from stage2_personality_agent import zscore_window, call_agent, parse_agent_pointers, log_agent_exchange

SCHEMA_VERSION = "2.0"  # 2.0 adds Mode B (uncalibrated) -- Mode A's own record shape is unchanged from 1.0
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Only used if the container reports a nonsensical FPS (<=0 or absurdly
# high) -- a fallback assumption, not a tuned value, so the clock driving
# calibration/windowing (see analyze_video()) stays sane rather than
# dividing by zero or producing a garbage timeline.
DEFAULT_FPS_FALLBACK = 30.0

# Threshold for flagging a window as a "notable moment" in the console
# summary -- 2 std deviations, the SAME domain-general convention already
# cited in stage1_step4_vectors.map_to_valence_arousal's own docstring
# ("2 std devs is a common, domain-general statistical convention"), not
# a new number invented for this file. In Mode B this is applied to a
# "relative" value, not a real z-score -- see MODE_B_LABEL.
NOTABLE_MOMENT_Z = 2.0

MODE_B_LABEL = (
    "UNCALIBRATED (Mode B) -- no neutral baseline. These are approximate, relative "
    "behavioral signals, NOT validated readings. Accurate readings require a "
    "calibrated video (neutral segment at the start)."
)

# Mode B's stand-in for a real per-person NeutralCalibrator.reference,
# shaped identically so it can be handed to zscore_window() and
# map_to_valence_arousal() UNCHANGED (same functions Mode A uses -- see
# module docstring). mean=0.0, std=1.0 is a domain-general "no
# information" identity reference, not fit to this project's data
# (Pitfall #5): std=1.0 makes _z_score's division a no-op, so the
# resulting number is just the raw composite reading itself, transparently
# -- Mode B does not manufacture a fake per-person spread statistic.
POPULATION_DEFAULT_REFERENCE = {
    "composite": {
        "v_bf": {"mean": 0.0, "std": 1.0, "n": None},
        "v_es": {"mean": 0.0, "std": 1.0, "n": None},
        "v_pd": {"mean": 0.0, "std": 1.0, "n": None},
    },
}
MODE_B_BASELINE_NOTE = (
    "Baseline used: population-default reference (mean=0.0, std=1.0), NOT this person's own "
    "calibrated neutral. Because std=1.0, the 'relative' values below are numerically identical "
    "to the raw, unscaled composite readings -- Mode B does not invent a per-person spread "
    "statistic it has no evidence for."
)

# Mode B only: heuristics for flagging likely scene cuts / a different
# face on screen (task requirement -- movies cut between people). These
# are NEW, self-contained report-metadata helpers, not a reimplementation
# of anything in stage1_step4_vectors.py -- they never touch V_bf/V_es/
# V_jc/V_pd math, calibration, windowing, or z-scoring, and they never
# attempt to identify or track WHICH face is which across a jump, only
# THAT the video looks unstable/multi-subject.
FACE_LANDMARKER_NUM_FACES_MODE_B = 3  # >1 so "multiple faces in one frame" is even observable; vector math still only ever reads face_landmarks[0]
POSITION_JUMP_FRACTION = 0.15  # face recentering by >15% of the frame diagonal in one frame step is not ordinary motion
SCALE_JUMP_RATIO = 1.5         # face apparent size changing by >50% in one frame step (zoom/cut, not smooth dolly)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}

# ============================================================
# LIVE AGENT MODE -- optional, OFF by default: a normal run (single file
# or batch) stays on the stub and spends zero tokens. Enabled via
# `--live` on the command line or STAGE3_LIVE=1 in the environment --
# deliberately the SAME env var stage3_demo_ui.py already defines, so
# one setting opts BOTH tools into live mode rather than inventing a
# second, parallel flag name for the identical concept ("reuse the
# existing live opt-in", per this step's own instructions).
# AGENT_LIVE is the ONLY place this flag is read; every call site below
# passes `use_stub=not AGENT_LIVE` to call_agent().
# ============================================================
AGENT_LIVE = ("--live" in sys.argv) or (os.environ.get("STAGE3_LIVE", "").strip().lower() not in ("", "0", "false", "no"))


def _print_agent_mode_banner():
    if AGENT_LIVE:
        print(">>> AGENT: LIVE (real API) -- one call per video analyzed")
    else:
        print(">>> AGENT: STUB (no API calls)")
    if AGENT_LIVE and not os.environ.get("ANTHROPIC_API_KEY"):
        print("!" * 70)
        print("!!! LIVE AGENT MODE REQUESTED BUT ANTHROPIC_API_KEY IS NOT SET")
        print("!!! Every agent report this run will fail (caught per-video); the video")
        print("!!! analysis itself is unaffected, but no agent report will be produced.")
        print("!!! Set ANTHROPIC_API_KEY before launching with --live, or drop --live for the stub.")
        print("!" * 70)


def _format_timestamp(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _open_video(path):
    if not os.path.isfile(path):
        print(f">>> ERROR: video file not found: {path}")
        return None, None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f">>> ERROR: could not open video (unsupported codec/container?): {path}")
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:  # sanity bound -- some containers/codecs report garbage
        print(f">>> WARNING: video reported an unusable FPS ({fps!r}); assuming {DEFAULT_FPS_FALLBACK:.0f} fps for timing.")
        fps = DEFAULT_FPS_FALLBACK
    return cap, fps


def _make_face_landmarker(num_faces=1):
    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=s1.FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=s1.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=s1.CONFIDENCE_THRESHOLD,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
    )


def _make_pose_landmarker():
    return mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=s1.POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=s1.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=s1.CONFIDENCE_THRESHOLD,
        )
    )


def _is_usable_calibration(reference, frames_total, frames_detected):
    """Mode A's own gate on the neutral baseline itself (distinct from
    WindowAccumulator's per-window gate below): reuses the SAME
    DETECT_RATE_FLOOR constant stage1 already defines for its window-
    validity gate (Pitfall #5 -- don't invent a new tuned number), plus
    a direct check that the two vectors Decision 18 actually needs
    (V_es, V_pd) have a real std to divide by. If this returns False,
    the caller (analyze_video) falls back to Mode B rather than
    proceeding on a baseline that can't support real z-scoring."""
    reasons = []
    if frames_total == 0:
        return False, ["no frames were read during the calibration window"]

    detect_rate = frames_detected / frames_total
    if detect_rate < episodes.DETECT_RATE_FLOOR:
        reasons.append(
            f"face detected in only {detect_rate:.0%} of calibration frames (< {episodes.DETECT_RATE_FLOOR:.0%} floor)"
        )
    for key in ("v_es", "v_pd"):
        std = reference.get("composite", {}).get(key, {}).get("std")
        if std is None:
            reasons.append(f"{key} has no usable neutral baseline (no valid samples during calibration)")
    # Deliberately NOT gating on reference["quality"]["possibly_not_neutral"]
    # here: that flag is documented in stage1_step4_vectors.py itself as
    # advisory only ("a FLAG not an auto-reject -- surfaced for the operator
    # to consider redoing, not silently discarded or silently accepted",
    # classify_calibration_quality docstring). Mode A already surfaces it
    # (console "POSSIBLY NOT NEUTRAL", result.calibration.quality) without
    # blocking on it -- same precedent the live tool follows. Only a
    # calibration that can't actually support z-scoring (too few detected
    # frames, or no std) forces a Mode B fallback.
    return (len(reasons) == 0), reasons


def _build_timeline_entry(summary, reference, mode_b=False):
    """One window -> one timeline entry. Both z-scoring (zscore_window,
    from stage2_personality_agent.py) and the canonical Decision-18
    mapping (map_to_valence_arousal, from stage1_step4_vectors.py) are
    called here, not reimplemented, for BOTH modes -- this function only
    reshapes their outputs plus the window's own metadata into one
    record. Mode B is fed POPULATION_DEFAULT_REFERENCE instead of a real
    calibration reference (see that constant's docstring) and gets
    differently-named numeric keys (*_relative instead of *_z_es/*_z_pd)
    so it can never be mistaken for a real z-score even by a consumer
    that only looks at field names -- plus "validated" and "label" are
    the FIRST keys in the dict so the caveat is unmissable."""
    window_z = zscore_window(summary, reference)
    va = x_core.map_to_valence_arousal(
        summary["composite"]["v_bf"]["avg"],
        summary["composite"]["v_es"]["avg"],
        summary["composite"]["v_pd"]["avg"],
        reference,
    )
    entry = {
        "validated": not mode_b,
    }
    if mode_b:
        entry["label"] = MODE_B_LABEL
    entry.update({
        "t_start_sec": round(summary["window_start_monotonic"], 2),
        "t_end_sec": round(summary["window_end_monotonic"], 2),
        "timestamp": _format_timestamp(summary["window_start_monotonic"]),
        "window_seconds": round(summary["window_seconds"], 2),
        "partial_window": summary["window_seconds"] < geometry.WINDOW_SECONDS - 0.5,
        # tanh-bounded point from the SAME canonical function the live UI
        # plots -- included for direct reuse by any future review UI,
        # in both modes (it's a display convenience, not a validation claim).
        "valence_plot": va["valence"],
        "arousal_plot": va["arousal"],
        "low_confidence": summary["window_quality"]["low_confidence"],
        "quality_reasons": summary["window_quality"]["reasons"],
        "detection_rate": round(summary["detection_rate"], 3),
        "n_samples": summary["n_samples"],
    })
    if mode_b:
        entry["valence_relative"] = window_z["v_es"]["avg"]
        entry["arousal_relative"] = window_z["v_pd"]["avg"]
        entry["valence_peak_relative"] = window_z["v_es"]["peak"]
        entry["arousal_peak_relative"] = window_z["v_pd"]["peak"]
        entry["v_bf_relative_logged_only"] = window_z["v_bf"]["avg"]
    else:
        # Decision 18: Valence = z_es, Arousal = z_pd -- the window-averaged
        # z-score, same quantity stage2_personality_agent's live prompt uses.
        entry["valence_z_es"] = window_z["v_es"]["avg"]
        entry["arousal_z_pd"] = window_z["v_pd"]["avg"]
        entry["valence_peak_z_es"] = window_z["v_es"]["peak"]
        entry["arousal_peak_z_pd"] = window_z["v_pd"]["peak"]
        # logged-only (Decision 17) -- never interpreted, never part of
        # Valence/Arousal above. Kept for Track-B completeness only.
        entry["v_bf_z_logged_only"] = window_z["v_bf"]["avg"]
    return entry


def _detect_face_instability(face_samples, frame_diag_px):
    """Mode B only. Heuristic-only report metadata, layered ON TOP of
    the reused pipeline -- never touches vector math, calibration,
    windowing, or z-scoring. Flags likely scene cuts / a different face
    by looking for large frame-to-frame jumps in screen position or face
    scale between temporally-adjacent detected frames (a continuous shot
    of one person can only move/zoom so much in 1/fps seconds), or any
    frame where the landmarker (num_faces=3 for Mode B) returned more
    than one face at once. Per the task's hard constraint, this NEVER
    attempts to identify or track WHICH face is which across a jump --
    it only signals THAT the video looks unstable/multi-subject, for the
    report (see MODE_B_PROCESSING's multi_face_note)."""
    if len(face_samples) < 2:
        return {
            "likely_multiple_or_changing_faces": False,
            "n_position_jumps": 0,
            "n_scale_jumps": 0,
            "multiple_faces_in_one_frame_detected": any(s["n_faces"] > 1 for s in face_samples),
            "n_face_samples": len(face_samples),
        }

    n_position_jumps = 0
    n_scale_jumps = 0
    for prev, cur in zip(face_samples, face_samples[1:]):
        if cur["t"] - prev["t"] > 1.0:
            continue  # face was lost and reacquired -- a gap, not a jump; already reflected in detect_rate
        dist = ((cur["x"] - prev["x"]) ** 2 + (cur["y"] - prev["y"]) ** 2) ** 0.5
        if frame_diag_px and dist > POSITION_JUMP_FRACTION * frame_diag_px:
            n_position_jumps += 1
        if prev["scale"] > 1e-6 and cur["scale"] > 1e-6:
            ratio = max(prev["scale"], cur["scale"]) / min(prev["scale"], cur["scale"])
            if ratio > SCALE_JUMP_RATIO:
                n_scale_jumps += 1

    multiple_faces_in_one_frame = any(s["n_faces"] > 1 for s in face_samples)
    likely = multiple_faces_in_one_frame or (n_position_jumps + n_scale_jumps) >= 2
    return {
        "likely_multiple_or_changing_faces": likely,
        "n_position_jumps": n_position_jumps,
        "n_scale_jumps": n_scale_jumps,
        "multiple_faces_in_one_frame_detected": multiple_faces_in_one_frame,
        "n_face_samples": len(face_samples),
    }


MULTI_FACE_NOTE = (
    "This video appears to contain multiple and/or rapidly changing faces (scene cuts or more "
    "than one person). No individual is tracked across the video -- each window's reading "
    "reflects whichever face was most prominent in that moment and is NOT attributable to one "
    "continuous person."
)


def analyze_video(video_path, force_uncalibrated=False):
    """Entry point for both modes. Tries Mode A first (unless
    force_uncalibrated, i.e. --uncalibrated), and falls back to Mode B
    automatically if the first CALIBRATION_SECONDS doesn't yield a
    usable neutral. Returns the result dict (also written to disk by
    main()) or None if the video couldn't be opened at all (already
    reported to console by _open_video -- nothing further to do)."""
    cap, fps = _open_video(video_path)
    if cap is None:
        return None
    cap.release()  # that was only a probe -- each mode below opens its own capture, from frame 0

    mode_a_fallback_reasons = None
    if not force_uncalibrated:
        outcome = _run_mode_a(video_path, fps)
        if outcome["outcome"] == "ok":
            return outcome["result"]
        if outcome["outcome"] == "hard_fail":
            return outcome["result"]  # e.g. zero frames at all -- Mode B can't help either
        mode_a_fallback_reasons = outcome["reasons"]
        print("[analyze_video] Mode A calibration not usable -- falling back to Mode B (uncalibrated):")
        for r in mode_a_fallback_reasons:
            print(f"    - {r}")
    else:
        print("[analyze_video] --uncalibrated forced -- running Mode B directly (no calibration attempt).")

    return _run_mode_b(video_path, fps, fallback_reasons=mode_a_fallback_reasons or [])


def _run_mode_a(video_path, fps):
    """MODE A (calibrated) -- Step 1's pipeline, UNCHANGED. Returns
    {"outcome": "ok", "result": ...} on a usable calibration,
    {"outcome": "hard_fail", "result": ...} if the video has literally
    nothing to work with (Mode B couldn't help either -- zero frames),
    or {"outcome": "fallback_to_mode_b", "reasons": [...]} if
    calibration didn't yield a usable neutral. The caller (analyze_video)
    decides whether to fall back; this function never touches Mode B."""
    cap = cv2.VideoCapture(video_path)
    face_landmarker = _make_face_landmarker(num_faces=1)
    pose_landmarker = _make_pose_landmarker()

    pd_buffer = deque()
    window_acc = episodes.WindowAccumulator()
    calibrator = x_core.NeutralCalibrator()

    timeline = []
    frame_index = 0
    n_frames_total = 0
    n_frames_face_detected = 0
    calibration_frames_total = 0
    calibration_frames_detected = 0
    last_video_time = 0.0

    print(f"[analyze_video] opening {video_path} ({fps:.1f} fps)")
    print(f"[analyze_video] Mode A: calibrating off the first {x_core.CALIBRATION_SECONDS:.0f}s of video time...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            video_time = frame_index / fps
            timestamp_ms = int(video_time * 1000)
            frame_index += 1
            n_frames_total += 1
            last_video_time = video_time

            clahe_frame = geometry.apply_clahe(frame)
            rgb_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            face_detected = False
            window_composite = {"v_bf": None, "v_es": None, "v_pd": None}
            window_covariate = {"v_jc": None, "v_bf_convergence_ratio": None, "v_es_cheek_raise": None}
            window_yaw = None

            if face_result.face_landmarks and face_result.facial_transformation_matrixes:
                lms = face_result.face_landmarks[0]
                matrix = face_result.facial_transformation_matrixes[0]
                normalized_pts = geometry.pose_normalize(lms, matrix, w, h)
                io_dist = geometry.interocular_distance(normalized_pts)
                yaw, pitch, roll = geometry.yaw_pitch_roll_from_matrix(matrix)
                v_bf, bf_components = x_core.compute_v_bf(normalized_pts, io_dist)
                v_es, es_components = x_core.compute_v_es(normalized_pts, io_dist)
                v_jc, jc_components = x_core.compute_v_jc(normalized_pts, io_dist)

                face_detected = True
                window_composite["v_bf"] = v_bf
                window_composite["v_es"] = v_es
                window_covariate["v_jc"] = v_jc
                window_covariate["v_bf_convergence_ratio"] = bf_components["convergence_ratio"]
                window_covariate["v_es_cheek_raise"] = es_components["cheek_raise"]
                window_yaw = yaw
                n_frames_face_detected += 1

            if pose_result.pose_world_landmarks:
                world = pose_result.pose_world_landmarks[0]
                nose_pos = np.array([world[geometry.POSE_NOSE].x, world[geometry.POSE_NOSE].y, world[geometry.POSE_NOSE].z])
                shoulder_mid = np.array(
                    [
                        (world[geometry.POSE_SHOULDER_L].x + world[geometry.POSE_SHOULDER_R].x) / 2.0,
                        (world[geometry.POSE_SHOULDER_L].y + world[geometry.POSE_SHOULDER_R].y) / 2.0,
                        (world[geometry.POSE_SHOULDER_L].z + world[geometry.POSE_SHOULDER_R].z) / 2.0,
                    ]
                )
                v_pd, _pd_components = x_core.compute_v_pd(pd_buffer, nose_pos, shoulder_mid, video_time)
                window_composite["v_pd"] = v_pd

            if not calibrator.is_calibrated():
                calibration_frames_total += 1
                if face_detected:
                    calibration_frames_detected += 1
                # Feed calibration BEFORE checking completion -- same
                # ordering stage3_demo_ui.py's live processing thread uses.
                calibrator.add_sample(video_time, window_composite, window_covariate, window_yaw)
                if calibrator.should_complete(video_time):
                    calibrator.complete(video_time)
                    tag = "POSSIBLY NOT NEUTRAL" if calibrator.reference["quality"]["possibly_not_neutral"] else "OK"
                    print(f"[analyze_video] calibration complete ({calibrator.reference['calibration_seconds']:.1f}s) [{tag}]")
                # No window accumulation and no deviation/z-scoring during
                # calibration -- same as the live pipeline: nothing
                # confident is reported until the neutral baseline exists.
                continue

            # Calibrated: deviation-from-neutral via the SAME call the live
            # pipeline makes (NeutralCalibrator.deviation), never
            # reimplemented here.
            deviation_composite = {
                key: calibrator.deviation(key, window_composite.get(key)) for key in ("v_bf", "v_es", "v_pd")
            }
            window_acc.add_sample(video_time, face_detected, window_yaw, deviation_composite, window_covariate)

            if window_acc.should_flush(video_time):
                summary = window_acc.flush(video_time)
                timeline.append(_build_timeline_entry(summary, calibrator.reference, mode_b=False))
    finally:
        face_landmarker.close()
        pose_landmarker.close()
        cap.release()

    video_duration_sec = n_frames_total / fps if fps else None

    if not calibrator.is_calibrated():
        if calibration_frames_total == 0:
            return {
                "outcome": "hard_fail",
                "result": _failed_result(
                    video_path, fps, video_duration_sec, "A_calibrated", "no_frames_readable",
                    ["no frames were read from the video"],
                ),
            }
        return {
            "outcome": "fallback_to_mode_b",
            "reasons": [
                f"video is only {last_video_time:.1f}s long -- shorter than the required "
                f"{x_core.CALIBRATION_SECONDS:.0f}s Mode-A calibration segment, so there is no neutral "
                f"baseline from the first part of the video."
            ],
        }

    usable, usability_reasons = _is_usable_calibration(
        calibrator.reference, calibration_frames_total, calibration_frames_detected
    )
    if not usable:
        return {"outcome": "fallback_to_mode_b", "reasons": usability_reasons}

    # Flush a trailing partial window (< WINDOW_SECONDS of leftover
    # footage) rather than silently dropping it -- marked
    # partial_window=True by _build_timeline_entry so consumers know.
    if window_acc.samples:
        summary = window_acc.flush(last_video_time)
        timeline.append(_build_timeline_entry(summary, calibrator.reference, mode_b=False))

    return {
        "outcome": "ok",
        "result": _ok_result(
            video_path, fps, video_duration_sec, calibrator.reference,
            calibration_frames_total, calibration_frames_detected, timeline,
        ),
    }


def _run_mode_b(video_path, fps, fallback_reasons):
    """MODE B (uncalibrated fallback, Step 2): the SAME pipeline (CLAHE
    -> landmarks -> existing vectors -> WindowAccumulator) run over the
    WHOLE video -- no frames reserved for calibration, since there is no
    neutral segment to calibrate against (even the first
    CALIBRATION_SECONDS, which Mode A may have just tried and failed to
    use as a baseline, is still analyzed here as ordinary content).
    Deviation/z-scoring goes through the SAME zscore_window/
    map_to_valence_arousal functions Mode A uses, fed
    POPULATION_DEFAULT_REFERENCE instead of a real per-person baseline
    -- see that constant's docstring for why. face_landmarker uses
    num_faces=FACE_LANDMARKER_NUM_FACES_MODE_B (>1) purely so the
    multi-face heuristic below can observe multiple simultaneous faces;
    vector computation still only ever reads face_landmarks[0], exactly
    as Mode A does -- no individual is tracked or identified (task hard
    constraint)."""
    cap = cv2.VideoCapture(video_path)
    face_landmarker = _make_face_landmarker(num_faces=FACE_LANDMARKER_NUM_FACES_MODE_B)
    pose_landmarker = _make_pose_landmarker()

    pd_buffer = deque()
    window_acc = episodes.WindowAccumulator()

    timeline = []
    face_samples = []  # for _detect_face_instability -- position/scale per detected frame
    frame_diag_px = None
    frame_index = 0
    n_frames_total = 0
    n_frames_face_detected = 0
    last_video_time = 0.0

    print(f"[analyze_video] opening {video_path} ({fps:.1f} fps)")
    print("[analyze_video] Mode B (uncalibrated): analyzing the whole video, no neutral baseline.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            video_time = frame_index / fps
            timestamp_ms = int(video_time * 1000)
            frame_index += 1
            n_frames_total += 1
            last_video_time = video_time

            h, w = frame.shape[:2]
            if frame_diag_px is None:
                frame_diag_px = (w ** 2 + h ** 2) ** 0.5

            clahe_frame = geometry.apply_clahe(frame)
            rgb_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            face_detected = False
            window_composite = {"v_bf": None, "v_es": None, "v_pd": None}
            window_covariate = {"v_jc": None, "v_bf_convergence_ratio": None, "v_es_cheek_raise": None}
            window_yaw = None
            n_faces_this_frame = len(face_result.face_landmarks) if face_result.face_landmarks else 0

            if face_result.face_landmarks and face_result.facial_transformation_matrixes:
                lms = face_result.face_landmarks[0]  # only ever the first face -- no individual tracking (task hard constraint)
                matrix = face_result.facial_transformation_matrixes[0]
                normalized_pts = geometry.pose_normalize(lms, matrix, w, h)
                io_dist = geometry.interocular_distance(normalized_pts)
                yaw, pitch, roll = geometry.yaw_pitch_roll_from_matrix(matrix)
                v_bf, bf_components = x_core.compute_v_bf(normalized_pts, io_dist)
                v_es, es_components = x_core.compute_v_es(normalized_pts, io_dist)
                v_jc, jc_components = x_core.compute_v_jc(normalized_pts, io_dist)

                face_detected = True
                window_composite["v_bf"] = v_bf
                window_composite["v_es"] = v_es
                window_covariate["v_jc"] = v_jc
                window_covariate["v_bf_convergence_ratio"] = bf_components["convergence_ratio"]
                window_covariate["v_es_cheek_raise"] = es_components["cheek_raise"]
                window_yaw = yaw
                n_frames_face_detected += 1

                # Raw (non-pose-normalized) iris midpoint in pixel space --
                # a cheap, stable screen-position proxy for the scene-cut
                # heuristic. pose_normalize's own output is deliberately
                # re-centered on the face itself (see its docstring), so it
                # cannot tell us WHERE on screen the face is -- only `lms`
                # (raw, normalized [0,1] image coords) can.
                cx = float((lms[geometry.IRIS_LEFT_CENTER].x + lms[geometry.IRIS_RIGHT_CENTER].x) / 2.0) * w
                cy = float((lms[geometry.IRIS_LEFT_CENTER].y + lms[geometry.IRIS_RIGHT_CENTER].y) / 2.0) * h
                face_samples.append({"t": video_time, "x": cx, "y": cy, "scale": io_dist, "n_faces": n_faces_this_frame})

            if pose_result.pose_world_landmarks:
                world = pose_result.pose_world_landmarks[0]
                nose_pos = np.array([world[geometry.POSE_NOSE].x, world[geometry.POSE_NOSE].y, world[geometry.POSE_NOSE].z])
                shoulder_mid = np.array(
                    [
                        (world[geometry.POSE_SHOULDER_L].x + world[geometry.POSE_SHOULDER_R].x) / 2.0,
                        (world[geometry.POSE_SHOULDER_L].y + world[geometry.POSE_SHOULDER_R].y) / 2.0,
                        (world[geometry.POSE_SHOULDER_L].z + world[geometry.POSE_SHOULDER_R].z) / 2.0,
                    ]
                )
                v_pd, _pd_components = x_core.compute_v_pd(pd_buffer, nose_pos, shoulder_mid, video_time)
                window_composite["v_pd"] = v_pd

            # No calibration phase at all: window_composite (raw composite
            # readings) IS what gets windowed directly -- the population-
            # default mean of 0.0 means "deviation from baseline" equals
            # the raw reading itself (see POPULATION_DEFAULT_REFERENCE).
            window_acc.add_sample(video_time, face_detected, window_yaw, window_composite, window_covariate)

            if window_acc.should_flush(video_time):
                summary = window_acc.flush(video_time)
                timeline.append(_build_timeline_entry(summary, POPULATION_DEFAULT_REFERENCE, mode_b=True))
    finally:
        face_landmarker.close()
        pose_landmarker.close()
        cap.release()

    if window_acc.samples:
        summary = window_acc.flush(last_video_time)
        timeline.append(_build_timeline_entry(summary, POPULATION_DEFAULT_REFERENCE, mode_b=True))

    video_duration_sec = n_frames_total / fps if fps else None

    if n_frames_total == 0:
        return _failed_result(
            video_path, fps, video_duration_sec, "B_uncalibrated", "no_frames_readable",
            ["no frames were read from the video"],
        )

    face_instability = _detect_face_instability(face_samples, frame_diag_px)

    return _mode_b_result(
        video_path, fps, video_duration_sec, n_frames_total, n_frames_face_detected,
        timeline, face_instability, fallback_reasons,
    )


def _honest_framing_note():
    return (
        "Behavioral signals / affective indicators only -- not emotion, not clinical, not diagnostic. "
        "Valence is pleasure-side only (from V_es, single-source) -- there is NO validated pain/"
        "negative-valence axis; a near-zero or negative value means \"no detected pleasure signal\", "
        "not \"detected pain\". Arousal is from V_pd (postural volatility) only. V_bf (brow furrow) and "
        "V_jc (jaw compression) are logged-only -- never interpreted, never part of Valence or Arousal."
    )


def _failed_result(video_path, fps, video_duration_sec, mode, failure_code, reasons,
                    calibration_frames_total=0, calibration_frames_detected=0, reference=None):
    result = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "video_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_path": os.path.abspath(video_path),
        "video_fps_used": fps,
        "video_duration_sec": round(video_duration_sec, 2) if video_duration_sec is not None else None,
        "mode": mode,
        "validated": False,
        "status": "failed",
        "failure_code": failure_code,
        "failure_reasons": reasons,
        "timeline": [],
        "aggregates": None,
        "notable_moments": [],
        "honest_framing": {"note": _honest_framing_note()},
    }
    if mode == "A_calibrated":
        result["calibration"] = {
            "calibration_seconds_target": x_core.CALIBRATION_SECONDS,
            "frames_seen": calibration_frames_total,
            "frames_face_detected": calibration_frames_detected,
            "detect_rate": (calibration_frames_detected / calibration_frames_total) if calibration_frames_total else None,
            "usable": False,
            "quality": reference["quality"] if reference else None,
        }
    return result


def _ok_result(video_path, fps, video_duration_sec, reference, calibration_frames_total, calibration_frames_detected, timeline):
    """MODE A result. Unchanged from Step 1 other than adding the
    top-level "validated": True field (Step 2 puts the same field on
    Mode B's result, set to False, as the single flag downstream code
    can rely on regardless of mode)."""
    valid_windows = [w for w in timeline if not w["low_confidence"]]
    valence_vals = [w["valence_z_es"] for w in valid_windows if w["valence_z_es"] is not None]
    arousal_vals = [w["arousal_z_pd"] for w in valid_windows if w["arousal_z_pd"] is not None]
    valence_peak_vals = [w["valence_peak_z_es"] for w in valid_windows if w["valence_peak_z_es"] is not None]
    arousal_peak_vals = [w["arousal_peak_z_pd"] for w in valid_windows if w["arousal_peak_z_pd"] is not None]
    n_low_confidence = sum(1 for w in timeline if w["low_confidence"])

    notable_moments = []
    for w in valid_windows:
        if w["arousal_peak_z_pd"] is not None and w["arousal_peak_z_pd"] >= NOTABLE_MOMENT_Z:
            notable_moments.append(
                {"timestamp": w["timestamp"], "type": "arousal_spike", "z": round(w["arousal_peak_z_pd"], 2)}
            )
        # Only POSITIVE valence spikes are reported -- pleasure-side-only,
        # per the honest-framing rule: a low/negative value means "no
        # detected pleasure signal", not a measured negative event, so it
        # is never surfaced as a "notable moment" here.
        if w["valence_peak_z_es"] is not None and w["valence_peak_z_es"] >= NOTABLE_MOMENT_Z:
            notable_moments.append(
                {"timestamp": w["timestamp"], "type": "pleasure_spike", "z": round(w["valence_peak_z_es"], 2)}
            )

    aggregates = {
        "n_windows": len(timeline),
        "n_windows_low_confidence": n_low_confidence,
        "pct_windows_low_confidence": round(n_low_confidence / len(timeline), 3) if timeline else None,
        "mean_valence_z_es": round(float(np.mean(valence_vals)), 3) if valence_vals else None,
        "peak_valence_z_es": round(float(np.max(valence_peak_vals)), 3) if valence_peak_vals else None,
        "mean_arousal_z_pd": round(float(np.mean(arousal_vals)), 3) if arousal_vals else None,
        "peak_arousal_z_pd": round(float(np.max(arousal_peak_vals)), 3) if arousal_peak_vals else None,
        "duration_analyzed_sec": round(sum(w["window_seconds"] for w in timeline), 2) if timeline else 0.0,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "video_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_path": os.path.abspath(video_path),
        "video_fps_used": fps,
        "video_duration_sec": round(video_duration_sec, 2) if video_duration_sec is not None else None,
        "mode": "A_calibrated",
        "validated": True,
        "status": "ok",
        "failure_code": None,
        "failure_reasons": [],
        "calibration": {
            "calibration_seconds_target": x_core.CALIBRATION_SECONDS,
            "calibration_seconds_actual": round(reference["calibration_seconds"], 2),
            "frames_seen": calibration_frames_total,
            "frames_face_detected": calibration_frames_detected,
            "detect_rate": round(calibration_frames_detected / calibration_frames_total, 3) if calibration_frames_total else None,
            "usable": True,
            "quality": reference["quality"],
        },
        "timeline": timeline,
        "aggregates": aggregates,
        "notable_moments": notable_moments,
        "honest_framing": {"note": _honest_framing_note()},
    }


def _mode_b_result(video_path, fps, video_duration_sec, n_frames_total, n_frames_face_detected,
                    timeline, face_instability, fallback_reasons):
    """MODE B result (Step 2). Field names mirror _ok_result's shape
    where the concept is identical (timestamps, quality flags, notable
    moments), but every VALUE field is named *_relative rather than
    *_z_es/*_z_pd so it can never be confused with a real z-score --
    plus "validated": False and "label"/"baseline" make the caveat
    unmissable at the top level too (task requirement: every output
    surface -- file, console, and each timeline entry -- carries it)."""
    valid_windows = [w for w in timeline if not w["low_confidence"]]
    valence_vals = [w["valence_relative"] for w in valid_windows if w["valence_relative"] is not None]
    arousal_vals = [w["arousal_relative"] for w in valid_windows if w["arousal_relative"] is not None]
    valence_peak_vals = [w["valence_peak_relative"] for w in valid_windows if w["valence_peak_relative"] is not None]
    arousal_peak_vals = [w["arousal_peak_relative"] for w in valid_windows if w["arousal_peak_relative"] is not None]
    n_low_confidence = sum(1 for w in timeline if w["low_confidence"])

    notable_moments = []
    for w in valid_windows:
        if w["arousal_peak_relative"] is not None and w["arousal_peak_relative"] >= NOTABLE_MOMENT_Z:
            notable_moments.append(
                {"timestamp": w["timestamp"], "type": "arousal_spike", "value": round(w["arousal_peak_relative"], 2)}
            )
        if w["valence_peak_relative"] is not None and w["valence_peak_relative"] >= NOTABLE_MOMENT_Z:
            notable_moments.append(
                {"timestamp": w["timestamp"], "type": "pleasure_spike", "value": round(w["valence_peak_relative"], 2)}
            )

    aggregates = {
        "n_windows": len(timeline),
        "n_windows_low_confidence": n_low_confidence,
        "pct_windows_low_confidence": round(n_low_confidence / len(timeline), 3) if timeline else None,
        "mean_valence_relative": round(float(np.mean(valence_vals)), 3) if valence_vals else None,
        "peak_valence_relative": round(float(np.max(valence_peak_vals)), 3) if valence_peak_vals else None,
        "mean_arousal_relative": round(float(np.mean(arousal_vals)), 3) if arousal_vals else None,
        "peak_arousal_relative": round(float(np.max(arousal_peak_vals)), 3) if arousal_peak_vals else None,
        "duration_analyzed_sec": round(sum(w["window_seconds"] for w in timeline), 2) if timeline else 0.0,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "video_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_path": os.path.abspath(video_path),
        "video_fps_used": fps,
        "video_duration_sec": round(video_duration_sec, 2) if video_duration_sec is not None else None,
        "mode": "B_uncalibrated",
        "validated": False,
        "label": MODE_B_LABEL,
        "mode_a_fallback_reasons": fallback_reasons or [],
        "status": "ok",
        "failure_code": None,
        "failure_reasons": [],
        "baseline": {"type": "population_default", "mean": 0.0, "std": 1.0, "note": MODE_B_BASELINE_NOTE},
        "frames_seen": n_frames_total,
        "frames_face_detected": n_frames_face_detected,
        "detect_rate": round(n_frames_face_detected / n_frames_total, 3) if n_frames_total else None,
        "face_stability": face_instability,
        "multi_face_note": MULTI_FACE_NOTE if face_instability["likely_multiple_or_changing_faces"] else None,
        "timeline": timeline,
        "aggregates": aggregates,
        "notable_moments": notable_moments,
        "honest_framing": {"note": _honest_framing_note()},
    }


def _build_video_report_prompt(result):
    """Builds the whole-video report prompt for the agent -- the SAME
    warm, pointer+detail contract stage2_personality_agent.build_prompt
    uses for the live 10s read (PLEASURE/AROUSAL/NOT_INTERPRETED/
    CONFIDENCE HEADLINE+DETAIL lines, parsed by the SAME
    parse_agent_pointers()), but built from this file's own whole-video
    AGGREGATE result instead of a single window. The agent is fed ONLY
    the already-computed numbers below -- it is never shown the video or
    any frame, and never told anything about vector math or thresholds,
    only the final signals (task hard constraint: it writes, it does
    not perceive).

    Mode-aware: Mode B's validated=False and the uncalibrated/
    approximate framing are explicitly given (both a natural-language
    instruction AND a dedicated CONTEXT marker line stub_agent_call()
    keys off of) and the agent is instructed to state them prominently
    -- a Mode B report must never read as confident or validated."""
    mode_b = result["mode"] == "B_uncalibrated"
    agg = result["aggregates"]

    if mode_b:
        mean_valence, peak_valence = agg["mean_valence_relative"], agg["peak_valence_relative"]
        mean_arousal, peak_arousal = agg["mean_arousal_relative"], agg["peak_arousal_relative"]
    else:
        mean_valence, peak_valence = agg["mean_valence_z_es"], agg["peak_valence_z_es"]
        mean_arousal, peak_arousal = agg["mean_arousal_z_pd"], agg["peak_arousal_z_pd"]

    def fmt(v):
        return "no reading available" if v is None else f"{v:+.2f}"

    moments_lines = []
    for m in result["notable_moments"]:
        label = "arousal spike" if m["type"] == "arousal_spike" else "pleasure spike"
        value = m.get("z", m.get("value"))
        moments_lines.append(f"- {label} around {m['timestamp']} ({value:+.2f})")
    moments_str = "\n".join(moments_lines) if moments_lines else "(none crossed the notable-moment threshold)"

    mode_marker = "MODE=B_UNCALIBRATED" if mode_b else "MODE=A_CALIBRATED"
    mode_note = (
        "MODE B, UNCALIBRATED: there is NO per-person neutral baseline for this video (validated=False). "
        "These numbers are approximate/relative, not validated readings. You MUST prominently and explicitly "
        "say this reading is approximate/uncalibrated and NOT validated -- never present it as confident or "
        "equivalent to a calibrated reading."
        if mode_b else
        "MODE A, CALIBRATED: this video had a usable per-person neutral baseline (validated=True). These are "
        "validated deviation-from-neutral readings for this person, within the limits already noted below."
    )

    multi_face_note = ""
    if result.get("multi_face_note"):
        multi_face_note = (
            "\n- MULTI-FACE WARNING: this video appears to contain multiple and/or rapidly changing faces "
            "(scene cuts or more than one person). You MUST say the reading is not attributable to one "
            "continuous person."
        )

    low_conf_note = "no window data available"
    if agg["n_windows"]:
        low_conf_note = f"{agg['n_windows_low_confidence']} of {agg['n_windows']} windows ({agg['pct_windows_low_confidence']:.0%}) were flagged low-confidence"

    prompt = f"""You are a warm, conversational voice summarizing an ENTIRE recorded video's worth of behavioral-signal readings, directly TO the person the video is about -- like a friendly, attentive observation, not a lab report. Natural and human, never dry or clinical.

CONTEXT: this is a WHOLE-VIDEO summary (not a single live window). {mode_marker}.

STRICT RULES (do not deviate -- warmth belongs in the VOICE, never in the CLAIMS):
- You are summarizing signals ALREADY COMPUTED from the whole video -- you were never shown the video or any frame; you only received the numbers below. Do not describe visual content, only the signals.
- These are BEHAVIORAL SIGNALS / AFFECTIVE INDICATORS -- never say "emotion", never use clinical or diagnostic language, never diagnose, never predict what happens next, never give advice.
- {mode_note}
- Never claim more than these two validated signals support: pleasure-side valence (from V_es only -- there is NO validated pain/negative-valence axis; a near-zero or negative value means "no detected pleasure signal", not "detected pain") and arousal (from V_pd, postural volatility, only).
- Brow furrow (V_bf) and jaw compression (V_jc) are logged-only in this system -- do not read, interpret, or describe what they show; only note that they're being logged and not interpreted.
- No invented history -- describe only this video's own signals, never compare to another session or invent continuity beyond what's given.{multi_face_note}

WHOLE-VIDEO SIGNALS ({agg['duration_analyzed_sec']}s analyzed, {agg['n_windows']} windows):
- Mean pleasure-side valence: {fmt(mean_valence)}   Peak: {fmt(peak_valence)}
- Mean arousal: {fmt(mean_arousal)}   Peak: {fmt(peak_arousal)}
- Window confidence: {low_conf_note}

NOTABLE MOMENTS:
{moments_str}

Return EXACTLY these 8 lines, one per line, in this exact order, with no extra commentary before or after:
PLEASURE_HEADLINE: <a short, warm headline, under 8 words, about the pleasure-side signal across the whole video>
PLEASURE_DETAIL: <one warm, honest sentence grounding that headline in what was actually measured, stating uncertainty if the signal is weak or missing>
AROUSAL_HEADLINE: <a short, warm headline, under 8 words, about the arousal signal across the whole video, naming a notable spike by timestamp if one is listed above>
AROUSAL_DETAIL: <one warm, honest sentence grounding that headline in what was actually measured>
NOT_INTERPRETED_HEADLINE: <a short headline noting brow/jaw are not part of this reading>
NOT_INTERPRETED_DETAIL: <one sentence -- logged only, not read or interpreted>
CONFIDENCE_HEADLINE: <a short headline naming how much to trust this reading -- MUST reflect calibrated vs approximate/uncalibrated per the MODE note above>
CONFIDENCE_DETAIL: <one sentence -- state the mode (calibrated Mode A / approximate uncalibrated Mode B), the low-confidence window count, and the multi-face caveat if one was given above>"""

    return prompt


def _generate_agent_report(result, use_live):
    """Whole-video warm agent report. Calls the SAME call_agent() swap
    point the live tool uses (stub by default, real API only when
    use_live -- exactly one call here, never a loop) and logs the
    exchange through the SAME log_agent_exchange() the live tool uses,
    so live-UI reads and video reports share one agent_log.jsonl. Never
    raises on its own; call sites still wrap this via _analyze_and_report
    so a live-API hiccup can't take down a batch run."""
    prompt = _build_video_report_prompt(result)
    response = call_agent(prompt, use_stub=not use_live)
    pointers = parse_agent_pointers(response["text"])
    log_agent_exchange(
        prompt=prompt,
        response=response,
        session_id=str(uuid.uuid4()),
        person_label=None,
        window_summary={"window_start_monotonic": 0.0, "window_end_monotonic": result.get("video_duration_sec")},
    )
    return {"response": response, "pointers": pointers}


def _print_agent_report(report):
    response = report["response"]
    mode_tag = "STUB" if response["is_stub"] else "LIVE"
    print()
    print("-" * 70)
    print(f"AGENT REPORT ({mode_tag} -- model={response['model']}):")
    for p in report["pointers"]:
        print(f"  - {p['headline']} -- {p['detail']}")
    print("-" * 70)


def _print_summary(result):
    print()
    print("=" * 70)
    print(f"VIDEO ANALYSIS SUMMARY -- {result['video_path']}")
    print("=" * 70)
    mode_display = {"A_calibrated": "CALIBRATED (Mode A)", "B_uncalibrated": "UNCALIBRATED (Mode B)"}.get(result["mode"], "n/a")
    duration_display = f"{result['video_duration_sec']}s" if result["video_duration_sec"] is not None else "n/a"
    print(f"duration: {duration_display}   mode: {mode_display}   status: {result['status']}")

    if result["mode"] == "B_uncalibrated":
        print()
        print("!" * 70)
        print(f"!!! {result['label']}")
        print("!" * 70)
        if result.get("mode_a_fallback_reasons"):
            print("Fell back from Mode A because:")
            for r in result["mode_a_fallback_reasons"]:
                print(f"  - {r}")
        if result.get("baseline"):
            print(result["baseline"]["note"])

    if result["status"] == "failed":
        print()
        print(f"COULD NOT ANALYZE ({result['failure_code']}):")
        for reason in result["failure_reasons"]:
            print(f"  - {reason}")
        print()
        print("No behavioral-signal numbers were produced -- see HONEST FRAMING below for what")
        print("this system does and does not measure even when it does run successfully.")
        print("-" * 70)
        print(_honest_framing_note())
        print("=" * 70)
        return

    if result["mode"] == "A_calibrated":
        calib = result["calibration"]
        agg = result["aggregates"]
        print(
            f"calibration: {calib['calibration_seconds_actual']}s, "
            f"detect_rate={calib['detect_rate']:.0%}"
            + (", FLAGGED: " + "; ".join(calib["quality"]["reasons"]) if calib["quality"]["possibly_not_neutral"] else ", OK")
        )
        print()
        if agg["n_windows"] == 0:
            print("No complete analysis windows -- video ended too soon after calibration to report a timeline.")
        else:
            print(f"windows analyzed: {agg['n_windows']}  ({agg['duration_analyzed_sec']}s of footage)")
            print(f"low-confidence windows: {agg['n_windows_low_confidence']} ({agg['pct_windows_low_confidence']:.0%})")
            print()
            print("VALENCE (pleasure-side only, z_es -- no validated pain axis):")
            print(f"  mean = {agg['mean_valence_z_es']}   peak = {agg['peak_valence_z_es']}")
            print("AROUSAL (postural volatility, z_pd):")
            print(f"  mean = {agg['mean_arousal_z_pd']}   peak = {agg['peak_arousal_z_pd']}")

            if result["notable_moments"]:
                print()
                print("notable moments:")
                for m in result["notable_moments"]:
                    label = "arousal spike" if m["type"] == "arousal_spike" else "pleasure spike"
                    print(f"  - {label} around {m['timestamp']}  (z = {m['z']:+.2f})")
            else:
                print()
                print(f"notable moments: none crossed the |z| >= {NOTABLE_MOMENT_Z:.1f} threshold")
    else:
        print()
        if result["detect_rate"] is not None:
            print(f"frames: {result['frames_seen']} seen, face detect_rate={result['detect_rate']:.0%}")
        if result.get("multi_face_note"):
            print()
            print("NOTE: " + result["multi_face_note"])

        agg = result["aggregates"]
        if agg["n_windows"] == 0:
            print()
            print("No complete analysis windows -- video too short to produce a timeline.")
        else:
            print()
            print(f"windows analyzed: {agg['n_windows']}  ({agg['duration_analyzed_sec']}s of footage)")
            print(f"low-confidence windows: {agg['n_windows_low_confidence']} ({agg['pct_windows_low_confidence']:.0%})")
            print()
            print("VALENCE (relative/approximate, pleasure-side only -- no validated pain axis):")
            print(f"  mean = {agg['mean_valence_relative']}   peak = {agg['peak_valence_relative']}")
            print("AROUSAL (relative/approximate, postural volatility):")
            print(f"  mean = {agg['mean_arousal_relative']}   peak = {agg['peak_arousal_relative']}")

            if result["notable_moments"]:
                print()
                print("notable moments (approximate):")
                for m in result["notable_moments"]:
                    label = "arousal spike" if m["type"] == "arousal_spike" else "pleasure spike"
                    print(f"  - {label} around {m['timestamp']}  (relative value = {m['value']:+.2f})")
            else:
                print()
                print(f"notable moments: none crossed the |value| >= {NOTABLE_MOMENT_Z:.1f} threshold")

    print("-" * 70)
    print(_honest_framing_note())
    if result["mode"] == "B_uncalibrated":
        print(result["label"])
    print("=" * 70)


def _minimal_failed_result(video_path, failure_code, reasons):
    """Same shape _failed_result() produces, for failures that happen
    OUTSIDE analyze_video() itself -- a video that couldn't be opened at
    all (video_path doesn't exist / bad codec, already reported to
    console by _open_video) or an unhandled exception during processing.
    mode=None (neither Mode A nor B was ever actually attempted/
    completed) -- _print_summary and the batch table both handle that
    as "n/a" rather than guessing a mode."""
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "video_analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_path": os.path.abspath(video_path),
        "video_fps_used": None,
        "video_duration_sec": None,
        "mode": None,
        "validated": False,
        "status": "failed",
        "failure_code": failure_code,
        "failure_reasons": reasons,
        "timeline": [],
        "aggregates": None,
        "notable_moments": [],
        "honest_framing": {"note": _honest_framing_note()},
    }


def _analyze_and_report(video_path, force_uncalibrated, use_live):
    """Runs Mode A/B selection (analyze_video, UNCHANGED from Steps 1-2)
    then, on a successful analysis, the whole-video agent report (Step
    3B). Wraps the WHOLE thing in try/except so one corrupt/unexpected-
    failure video can never crash a --folder batch run -- returns a
    result dict describing the failure instead of raising. Returns None
    only when the video couldn't be opened at all (already reported to
    console by _open_video) -- single-file mode in main() treats that
    exactly as Steps 1-2 already did; batch mode (run_batch) converts it
    to a _minimal_failed_result row instead."""
    try:
        result = analyze_video(video_path, force_uncalibrated=force_uncalibrated)
        if result is None:
            return None
        if result["status"] == "ok":
            result["agent_report"] = _generate_agent_report(result, use_live)
        return result
    except Exception as exc:  # a batch run must survive one bad video -- see docstring
        print(f">>> ERROR: unhandled exception while analyzing {video_path}: {type(exc).__name__}: {exc}")
        return _minimal_failed_result(video_path, "unhandled_exception", [f"{type(exc).__name__}: {exc}"])


def _write_result_file(video_path, result):
    os.makedirs(LOG_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(LOG_DIR, f"video_analysis_{base_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Result written to: {out_path}")
    return out_path


def _scan_video_folder(folder_path):
    if not os.path.isdir(folder_path):
        return None
    names = sorted(
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    return [os.path.join(folder_path, f) for f in names]


_MODE_DISPLAY = {"A_calibrated": "Mode A (calibrated)", "B_uncalibrated": "Mode B (uncalibrated)"}


def run_batch(folder_path, force_uncalibrated, use_live):
    """PART A (Step 3): processes every video file in folder_path through
    the exact same single-video pipeline (_analyze_and_report), one at a
    time. A failing/corrupt/no-face video is caught and recorded as a
    failed row -- it never stops the batch (task hard constraint). Every
    video still gets its own result file written, exactly as single-file
    mode does. Returns the list of batch summary rows, or None if
    folder_path itself doesn't exist."""
    video_paths = _scan_video_folder(folder_path)
    if video_paths is None:
        print(f">>> ERROR: folder not found: {folder_path}")
        return None
    if not video_paths:
        print(f">>> No video files found in {folder_path} (looked for {', '.join(sorted(VIDEO_EXTENSIONS))})")
        return []

    print(f"[analyze_video] batch mode: {len(video_paths)} video(s) found in {folder_path}")
    if use_live:
        print(f">>> This will make ONE real Anthropic API call PER VIDEO ({len(video_paths)} total) -- tokens will be spent.")

    batch_rows = []
    for i, video_path in enumerate(video_paths, start=1):
        name = os.path.basename(video_path)
        print(f"\n[{i}/{len(video_paths)}] analyzing {name} ...")

        result = _analyze_and_report(video_path, force_uncalibrated, use_live)
        if result is None:
            result = _minimal_failed_result(
                video_path, "could_not_open", ["video could not be opened -- see console output above"]
            )

        mode_display = _MODE_DISPLAY.get(result.get("mode"), "n/a")
        status_display = "OK" if result["status"] == "ok" else f"FAILED ({result.get('failure_code')})"
        print(f"[{i}/{len(video_paths)}] {name} -> {mode_display}, {status_display}")

        _print_summary(result)
        if result.get("agent_report"):
            _print_agent_report(result["agent_report"])
        _write_result_file(video_path, result)

        batch_rows.append(
            {"video": name, "mode": result.get("mode"), "status": result["status"], "failure_code": result.get("failure_code")}
        )

    _print_batch_summary(batch_rows)
    return batch_rows


def _print_batch_summary(batch_rows):
    print()
    print("=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for r in batch_rows:
        mode_display = _MODE_DISPLAY.get(r["mode"], "n/a")
        status_display = "OK" if r["status"] == "ok" else f"FAILED ({r['failure_code']})"
        print(f"  {r['video']:40s}  {mode_display:24s}  {status_display}")
    n_ok = sum(1 for r in batch_rows if r["status"] == "ok")
    print("-" * 70)
    print(f"{n_ok}/{len(batch_rows)} succeeded, {len(batch_rows) - n_ok}/{len(batch_rows)} failed")
    print("=" * 70)


def main():
    args = sys.argv[1:]
    _print_agent_mode_banner()

    folder_path = None
    if "--folder" in args:
        idx = args.index("--folder")
        if idx + 1 >= len(args):
            print("Usage: python analyze_video.py --folder path/to/videos [--uncalibrated] [--live]")
            sys.exit(1)
        folder_path = args[idx + 1]

    force_uncalibrated = "--uncalibrated" in args

    if folder_path is not None:
        batch_rows = run_batch(folder_path, force_uncalibrated, AGENT_LIVE)
        if batch_rows is None:
            sys.exit(1)
        sys.exit(0 if any(r["status"] == "ok" for r in batch_rows) or not batch_rows else 1)

    reserved = {folder_path} if folder_path else set()
    video_path = next((a for a in args if not a.startswith("--") and a not in reserved), None)
    if video_path is None:
        print("Usage: python analyze_video.py path/to/video.mp4 [--uncalibrated] [--live]")
        print("   or: python analyze_video.py --folder path/to/videos [--uncalibrated] [--live]")
        sys.exit(1)

    result = _analyze_and_report(video_path, force_uncalibrated, AGENT_LIVE)
    if result is None:
        sys.exit(1)  # already reported by _open_video

    _print_summary(result)
    if result.get("agent_report"):
        _print_agent_report(result["agent_report"])
    _write_result_file(video_path, result)

    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
