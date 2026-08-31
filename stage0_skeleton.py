"""
Stage 0, step 3 — CLAHE + FaceLandmarker + PoseLandmarker land in Thread 2
only (Gap 5).

Thread 1 is unchanged from step 2: capture-only, never blocks on anything
but the camera. This step proves detection can run inside the 10s
processing cycle without dragging Thread 1's FPS below Gate 1 (15 FPS).
Still no vectors, no quality gate — this only confirms landmarks come back
and reports how expensive one cycle of CLAHE+FaceLandmarker+PoseLandmarker
is.

Detection API note: this install (Python 3.12, mediapipe 0.10.35, Windows)
does not ship the legacy `mp.solutions.face_mesh` / `mp.solutions.pose`
API at all (verified: no `mediapipe/python/solutions` package present).
Only the newer Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`,
`PoseLandmarker`) is available, so that's what this step uses. Both
`.task` model bundles are downloaded once into `models/` (gitignored,
resolved relative to this file — no hardcoded absolute paths) rather than
committed to version control.

Design
------
Thread 1 (capture) and Thread 2 (processing) share one mutable slot,
`latest_frame`, guarded by a `threading.Lock`. Thread 1 overwrites that
slot as fast as the camera will deliver frames; Thread 2 reads whatever
is sitting there once every 10 seconds. Older un-read frames are simply
discarded — there is no queue, so the buffer never backs up.

Why not one combined loop? cap.read() runs at whatever the sensor gives
(~30 FPS budget, ~33ms/frame). Any processing sharing that same loop
(CLAHE, FaceMesh, Pose, disk I/O) adds its own latency on top, and the
loop can only run as fast as its slowest step. Once per-frame work drifts
past ~33ms, capture is throttled to processing's pace and FPS craters —
which is exactly the failure CLAUDE.md's architecture rule (2-thread split)
exists to avoid. Splitting the threads means Thread 1's cap.read() loop
never waits on anything but the camera, so capture FPS stays at the
hardware ceiling regardless of how slow processing later becomes.

The lock is held only for the few microseconds it takes to swap a
reference (assign/read `latest_frame`), never across cap.read() or any
processing — so the two threads never block on each other.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import threading
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

CAMERA_INDEX = 0
PROCESSING_CYCLE_SECONDS = 10.0
FPS_REPORT_INTERVAL_SECONDS = 3.0

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_full.task")

stop_event = threading.Event()
frame_lock = threading.Lock()
latest_frame = None


def capture_thread():
    global latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
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
            fps = frame_count / elapsed
            print(f"[Capture] sustained FPS: {fps:.1f}")
            frame_count = 0
            fps_window_start = time.perf_counter()

    cap.release()
    print("[Capture] thread stopped.")


def apply_clahe(frame_bgr):
    # CLAHE operates on a single lightness channel: convert to LAB, equalize
    # L, merge back. This boosts local contrast (helps landmark detection
    # under uneven lighting) without blowing out color the way global
    # histogram equalization would.
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_equalized = clahe.apply(l_channel)
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)


def yaw_pitch_roll_from_matrix(matrix_4x4):
    # facial_transformation_matrixes gives a 4x4 rigid transform (rotation +
    # translation) mapping the canonical 3D face model onto the detected
    # face. The upper-left 3x3 is the rotation; standard extrinsic
    # X-Y-Z Euler decomposition gives yaw/pitch/roll in degrees. This is
    # what Stage 1.5's quality gate (yaw > 35°) will read — logged here
    # only to confirm the matrix is populated and usable.
    r = np.asarray(matrix_4x4)[:3, :3]
    pitch = np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)))
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    roll = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    return yaw, pitch, roll


def processing_thread():
    print("[Processing] thread started.")

    # Detectors are created once, outside the loop — re-initializing them
    # every cycle would add setup cost to every 10s tick for no reason.
    # RunningMode.IMAGE (not VIDEO/LIVE_STREAM) because we only feed one
    # frame per 10s cycle, not a continuous timestamped stream.
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,  # not used: vectors must stay geometric, not model-derived (CLAUDE.md #4)
            output_facial_transformation_matrixes=True,  # feeds yaw/pitch/roll for the Stage 1.5 quality gate
        )
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )

    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame

        if frame is None:
            print("[Processing] no frame available yet.")
        else:
            cycle_start = time.perf_counter()

            clahe_frame = apply_clahe(frame)
            rgb_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            face_result = face_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)

            cycle_ms = (time.perf_counter() - cycle_start) * 1000.0

            face_landmark_count = (
                len(face_result.face_landmarks[0]) if face_result.face_landmarks else 0
            )
            pose_landmark_count = (
                len(pose_result.pose_landmarks[0]) if pose_result.pose_landmarks else 0
            )

            pose_line = (
                f"[Processing] face_landmarks={face_landmark_count} "
                f"pose_landmarks={pose_landmark_count} "
                f"cycle_time={cycle_ms:.1f}ms"
            )

            if face_result.facial_transformation_matrixes:
                yaw, pitch, roll = yaw_pitch_roll_from_matrix(
                    face_result.facial_transformation_matrixes[0]
                )
                pose_line += f" yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}"

            print(pose_line)

        # Sleep in short slices so shutdown (stop_event) is responsive
        # instead of blocking the full 10s on a single time.sleep().
        slept = 0.0
        while slept < PROCESSING_CYCLE_SECONDS and not stop_event.is_set():
            time.sleep(0.1)
            slept += 0.1

    face_landmarker.close()
    pose_landmarker.close()
    print("[Processing] thread stopped.")


def main():
    t1 = threading.Thread(target=capture_thread, name="CaptureThread")
    t2 = threading.Thread(target=processing_thread, name="ProcessingThread")
    t1.start()
    t2.start()

    print("Press 'q' in the preview window to quit.")
    while not stop_event.is_set():
        with frame_lock:
            frame = latest_frame

        if frame is not None:
            cv2.imshow("Stage 0 - raw capture (press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()
    t1.join()
    t2.join()
    print("Clean shutdown complete.")


if __name__ == "__main__":
    main()
