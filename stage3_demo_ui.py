"""
Stage 3 -- minimal demo UI (Track A, Phase 5).

RENDERING LAYER ONLY. Text is rendered with PIL (ImageFont/ImageDraw,
anti-aliased, real fonts) instead of cv2.putText -- shapes (rects,
lines, circles, hatching) stay on cv2, which is faster for solid fills
than PIL. All text calls are queued during a frame and composited onto
the canvas in a SINGLE BGR<->RGB round trip at the end of
draw_demo_canvas (flush_text) -- not one round trip per string, which
would be far too slow at video framerates. Font objects are loaded
ONCE at import time (FONTS dict below), never per frame.

This file is a DISPLAY LAYER over the already-built pipeline. It does
NOT reimplement vector math, pose-normalization, the window-validity
gate, calibration, or z-scoring -- all of that is imported from the D1
feature-block modules (features/geometry.py, features/x_core.py,
features/episodes.py, features/attention.py -- see
docs/D1_DEPENDENCY_MAP.md) and used as-is. `stage1_step4_vectors.py`
(aliased `s1` below) is still used directly for the two-thread runtime
state it owns (capture_thread, frame_lock/latest_frame/stop_event) and
for anything not part of the D1 feature-block split. It also does not
touch gate2_trials.jsonl, the capture tool, or any scoring script.

V/A MAPPING: this file calls x_core.map_to_valence_arousal() directly --
the CANONICAL Decision 18 implementation (Valence = z_es only,
pleasure-side/single-source; Arousal = z_pd only). There is exactly
one V/A formula in this codebase, defined once in features/x_core.py;
this file does not keep its own copy.

Threading stays exactly as CLAUDE.md mandates: Thread 1 is
s1.capture_thread, reused unmodified (it only touches the webcam and
s1's own frame_lock/latest_frame/stop_event globals). Thread 2 is
stage3_processing_thread below -- all MediaPipe + vector-math calls,
never the webcam. The render loop (main thread) only reads
already-computed state under a lock and draws; it never blocks either
worker thread.

A later pass (premium-polish + genuine real-data features: the V/A
history trail, session stats, live confidence/quality indicators, a
richer agent-read panel) touches stage3_processing_thread too, but only
ADDITIVELY -- a bounded-size history deque, a few counters, a couple of
plain values carried into latest_ui_state -- never a change to any line
that computes a vector, a deviation, a z-score, a window-validity/
calibration-quality verdict, or the V/A mapping itself; every value
those additions surface is read straight off an object stage1's
pipeline already produces (WindowAccumulator.flush(),
NeutralCalibrator.complete(), map_to_valence_arousal()), never
recomputed or approximated here.

A further pass (quit-handling fix + HORIZONTAL HEAD ORIENTATION,
experimental) does touch main(): the render loop now also detects the
window's own X-button close (cv2.getWindowProperty) and treats
Ctrl-C/KeyboardInterrupt as a clean quit path, both threads are joined
with a bounded timeout so a genuinely stuck blocking call can never
leave the process itself hanging, and "shutting down..." always prints
on the way out. Presentation- and input-handling-only -- the two-thread
split, locks, and Thread() construction themselves are unchanged. The
same pass adds one more additive read in stage3_processing_thread:
attention.compute_v_so() is called (reusing its own output, not recomputing
it) purely to surface its yaw_deg component, clearly labeled
EXPERIMENTAL/UNVALIDATED -- never composited into any vector, never
touching calibration/windowing/z-scoring/V-A mapping. Only yaw is ever
shown; pitch is confirmed unreliable (CLAUDE.md) and is never displayed
here, regardless of what compute_v_so itself computes internally.

A further pass (EXPERIMENTAL SIGNALS PART 2: gaze L/R/C + blink rate)
adds two more isolated, additive reads in stage3_processing_thread --
attention.compute_gaze_direction() (a NEW function, reusing the same
landmarks/ratio approach _gaze_centering_score already uses) and
attention.BlinkDetector (a NEW class watching V_es's own aperture value,
read-only -- ear() itself is never touched). Neither is composited into
any vector; gaze is horizontal-only by construction (no up/down, ever);
blink reports "measuring..." rather than a number until enough
observation time has passed, and never counts a blink across a no-face/
occlusion gap. The head-yaw badge from the previous pass moved out of
its video overlay into a new, shared EXPERIMENTAL SIGNALS strip
(EXPERIMENTAL_Y/EXPERIMENTAL_H, its own row) alongside the two new gaze
and blink badges -- three badges no longer fit stacked on the video
without clipping, and this also lets them join the regular cached
chrome build instead of needing the video-overlay-specific compositing
dance the single yaw badge used before. Both signals are also logged,
window by window, to a NEW, separate logs/experimental_signals_log.jsonl
(own schema, own record type, stamped unvalidated) -- see
EXPERIMENTAL_LOG_PATH below; this is the only other JSONL logging this
file does, alongside stage2's agent_log.jsonl.

A bugfix pass corrected two defects found in the above: (1) gaze L/R/C
was mirrored (confirmed by live testing) -- fixed by flipping
stage1_step4_vectors.py's GAZE_LABEL_SIGN constant, same
single-constant pattern as HEAD_YAW_LABEL_SIGN, not a change to the
ratio math. (2) blink rate always read 0 -- root-caused to
BLINK_APERTURE_PLAUSIBLE_MIN (0.05) rejecting genuinely-closed-eye
aperture values as "unreliable" and aborting detection before a blink
could ever be confirmed; fixed by lowering that floor (see its own
comment in stage1_step4_vectors.py) and widening BLINK_MAX_DURATION_
SECONDS for real sampling-rate/tracking-latency slack. A live diagnostic
readout (raw aperture + detector state) was added to the blink badge,
and raw per-sample aperture is now batched and logged per-window, so
future re-tuning has real data instead of guesswork.

No sample-level JSONL logging happens here (that is stage1's and the
Gate 2 capture tool's job, with their own frozen schemas). Disk writes
this file causes: stage2's existing agent_log.jsonl exchange log
(is_stub stamped per Mandatory Architecture #9, true by default and
false only when launched with --live/STAGE3_LIVE -- see AGENT_LIVE
below), via run_stage2_on_window -- reused unmodified; and, as of the
EXPERIMENTAL SIGNALS PART 2 pass (gaze L/R/C + blink rate), a NEW,
separate, WINDOW-level (not sample-level) logs/experimental_signals_log.jsonl,
its own record type, stamped unvalidated -- see EXPERIMENTAL_LOG_PATH.
"""

import atexit
import cv2
import json
import mediapipe as mp
import numpy as np
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

import stage1_step4_vectors as s1
from features import geometry, x_core, episodes, attention
from features.x_core import map_to_valence_arousal
from stage1_step7_consent import run_consent_gate
from stage2_personality_agent import run_stage2_on_window, parse_agent_pointers

FPS_REPORT_INTERVAL_SECONDS = 3.0
SESSION_ID = str(uuid.uuid4())
PERSON_LABEL = None

# Single definition, used by both cv2.imshow (main()) and the X-button
# close check (main()) -- a window-title mismatch between the two would
# make the close check silently never match the real window.
WINDOW_TITLE = "Stage 3 - behavioral signals demo (press Q to quit)"

# ============================================================
# EXPERIMENTAL SIGNALS LOG -- gaze L/R/C + blink rate, PART of the
# "VALIDATION READINESS" requirement for these two new, isolated,
# UNVALIDATED signals: a later human-applied cross-person check (does
# the gaze label match the commanded look direction? does the counted
# blink rate match a hand count?) needs real per-window records to check
# against. Its own file, own record type, own schema_version -- NEVER
# written into gate2_trials.jsonl, agent_log.jsonl, soak_log.jsonl, or
# consent_log.jsonl, so this can't disturb any existing, already-scored
# log. Always ON (not gated behind a flag like --soak/--live): unlike
# the agent call, writing a small JSONL line is not a network call and
# costs nothing worth gating, and CLAUDE.md's own TRAINING-GRADE DATA
# section treats logging as default-on, opt-out-never for exactly this
# reason -- every logged reading is a future validation/training row.
# ============================================================
EXPERIMENTAL_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "experimental_signals_log.jsonl")


def _log_experimental_window(record):
    """Simple append -- same lightweight style stage1_step7_consent.py's
    own _log_consent uses (makedirs + one write), not SoakTracker's
    fsync-level rigor: this is instrumentation/validation-prep data, not
    the frozen Gate-2-grade dataset. Never raises into the caller --
    logging failure must not take down the demo it's watching."""
    try:
        os.makedirs(os.path.dirname(EXPERIMENTAL_LOG_PATH), exist_ok=True)
        with open(EXPERIMENTAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # instrumentation must never crash the demo it's watching
        print(f"[Stage3] experimental signals log write failed: {exc}")


# ============================================================
# SOAK LOGGING -- optional, OFF by default (a normal run is byte-
# identical to before: SOAK_MODE gates every soak-related call site
# below to a cheap `if` on a None object). Enabled via `--soak` on the
# command line or STAGE3_SOAK=1 in the environment.
#
# CLAUDE.md's Stage 3 plan calls for a 30-60 min soak against the stub
# (no FPS decay / memory growth / thread deadlock) -- this makes that
# soak hands-off: every SOAK_INTERVAL_SECONDS, one JSONL line is
# appended to logs/soak_log.jsonl with capture FPS, processing
# throughput, process memory, and thread health, plus a final summary
# line at clean exit. Read the log back afterward instead of watching
# the window live for an hour.
#
# Every value logged is either read from state the app ALREADY
# computes (samples_per_sec, Thread.is_alive(), the hardcoded stub
# flag) or from a dedicated, separate polling thread that watches
# s1.latest_frame's identity (see SoakTracker._watch_loop) -- nothing
# here adds per-frame image processing, and the only file I/O happens
# on the render/main thread at the SOAK_INTERVAL_SECONDS cadence, never
# on s1.capture_thread and never on stage3_processing_thread.
# ============================================================
SOAK_MODE = ("--soak" in sys.argv) or (os.environ.get("STAGE3_SOAK", "").strip().lower() not in ("", "0", "false", "no"))
SOAK_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "soak_log.jsonl")
SOAK_INTERVAL_SECONDS = float(os.environ.get("STAGE3_SOAK_INTERVAL_SECONDS", "30"))  # override for a short smoke test


def _print_soak_banner():
    """Loud, unmistakable ON/OFF confirmation, printed on every actual
    launch (see the `if __name__ == "__main__"` call site -- not on a
    plain `import stage3_demo_ui`). This exists because a real soak run
    silently produced zero records when the --soak flag didn't take --
    that must be visible in the first line of output, not discoverable
    only by later going looking for a file that isn't there. SOAK_LOG_PATH
    is already an absolute path (built from __file__'s abspath), so
    there is never ambiguity about which folder it went to. No ANSI
    color codes -- the rest of this file's console output is plain text
    and not every terminal renders escape codes cleanly; the ">>>"
    prefix and ALL-CAPS ON/OFF are the "obvious" part instead."""
    if SOAK_MODE:
        print(f">>> SOAK LOGGING: ON  -- writing to {SOAK_LOG_PATH} every {SOAK_INTERVAL_SECONDS:.0f}s")
    else:
        print(">>> SOAK LOGGING: OFF -- run with --soak (or STAGE3_SOAK=1) to record")


def _loud_error(lines):
    """A failure banner that cannot be mistaken for routine log output --
    used for soak-logging setup/write failures and live-agent
    misconfiguration, neither of which may fail silently."""
    print("!" * 70)
    for line in lines:
        print(f"!!! {line}")
    print("!" * 70)


# ============================================================
# LIVE AGENT MODE -- optional, OFF by default: a normal run and the
# soak stay on the stub and spend zero tokens (Mandatory Architecture
# #8, plus this task's own hard constraint). Enabled via `--live` on
# the command line or STAGE3_LIVE=1 in the environment, mirroring the
# --soak / STAGE3_SOAK pattern above.
#
# AGENT_LIVE is the ONLY place this flag is read; every call site below
# passes `use_stub=not AGENT_LIVE` -- one flag, one meaning, everywhere
# (same "one formula, one place" discipline the V/A mapping already
# follows). The key itself is never touched here: call_real_agent in
# stage2_personality_agent.py reads ANTHROPIC_API_KEY from the
# environment via the SDK's own default resolution, and this file never
# hardcodes, prints, or logs it.
# ============================================================
AGENT_LIVE = ("--live" in sys.argv) or (os.environ.get("STAGE3_LIVE", "").strip().lower() not in ("", "0", "false", "no"))


def _print_agent_mode_banner():
    if AGENT_LIVE:
        print(">>> AGENT: LIVE (real API)")
    else:
        print(">>> AGENT: STUB (no API calls)")
    if AGENT_LIVE and not os.environ.get("ANTHROPIC_API_KEY"):
        _loud_error(
            [
                "LIVE AGENT MODE REQUESTED BUT ANTHROPIC_API_KEY IS NOT SET",
                "Every agent call this run will fail -- each failure is caught per-window so the",
                "demo keeps running, but the read panel will just keep showing its last successful",
                "text (or the startup placeholder) instead of updating.",
                "Set ANTHROPIC_API_KEY before launching with --live, or drop --live to use the stub.",
            ]
        )


def _memory_rss_mb():
    """Process RSS in MB, plus which method produced it.

    Tries psutil first (cross-platform, used automatically if the user
    ever installs it) then falls back to calling the Windows API
    directly via ctypes -- ctypes is stdlib, so this adds NO new
    dependency. psutil is deliberately NOT in requirements.txt: it
    isn't installed in this project's environment today, and the
    ctypes path below already covers CLAUDE.md's documented target
    ("Windows laptop, native Python"), so there's nothing this
    optional feature actually needs psutil for here. If neither path
    works (e.g. non-Windows without psutil), returns (None,
    "unavailable") rather than guessing.
    """
    try:
        import psutil  # optional -- see docstring; not required for this feature to work
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), "psutil"
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Explicit restype/argtypes are required here: ctypes' default
        # return type (c_int, 32-bit signed) truncates/misreads the
        # 64-bit HANDLE pseudo-value GetCurrentProcess() returns,
        # which then makes GetProcessMemoryInfo fail with an invalid-
        # handle error on 64-bit Python -- confirmed by testing without
        # this fix first.
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024), "windows_ctypes"
    except Exception:
        pass
    return None, "unavailable"


class SoakTracker:
    """Owns all soak-run state. Only instantiated in main() when
    SOAK_MODE is on -- when off, nothing in this class ever runs, and
    no extra thread is ever started.

    CAPTURE FPS MEASUREMENT: s1.capture_thread has no accessible FPS
    value to read -- it only prints its own local calculation, and
    s1.py is not touched by this instrumentation. The only option left
    is to observe s1.latest_frame from outside and count distinct
    frame objects (a fresh ndarray every capture read()). The first
    version of this did that check once per render-loop iteration --
    WRONG, confirmed by testing: the render loop paces at ~30ms
    (matching capture) but its own draw cost can push a cycle well past
    that, so on a loaded system it silently under-samples and reports
    a "capture FPS" that is actually the render loop's own rate (e.g.
    logged ~10fps while s1's own console print showed a steady ~30fps
    the whole time). Fixed by watching on a SEPARATE, dedicated,
    lightweight thread that polls s1.latest_frame's identity every 5ms
    (up to 200 checks/sec -- comfortably faster than any real capture
    rate, so every distinct frame gets caught) and sleeps the rest of
    the time, releasing the GIL -- decoupled from render/draw speed
    entirely, and NOT the capture thread itself. This thread only
    exists in soak mode and does nothing but this.

    maybe_report()/write_summary() (file I/O) still run on the
    render/main thread, gated to SOAK_INTERVAL_SECONDS -- cheap,
    infrequent, and separate from the frame-watching thread.
    """

    def __init__(self):
        self.start_time = time.perf_counter()
        self.last_report_time = self.start_time
        self._last_frame_id = None
        self._frames_since_report = 0
        self._count_lock = threading.Lock()
        self.fps_samples = []
        self.mem_samples = []
        self.any_thread_death = False
        self.any_exception = False
        self.n_samples = 0
        self._summary_written = False  # guards against writing soak_summary twice (clean exit + atexit fallback)
        self.write_ok = False  # set True only once a real write to disk is confirmed

        # Directory creation and the first write are the ONLY way to know
        # up front whether logging can actually work (bad path, no
        # permission, read-only mount, ...). Both are wrapped so a
        # failure here degrades to "no soak logging" with a LOUD, visible
        # reason -- not a crashed demo and not a silent no-op, which is
        # exactly the failure mode this hardening pass exists to close.
        try:
            os.makedirs(os.path.dirname(SOAK_LOG_PATH), exist_ok=True)
        except OSError as exc:
            _loud_error(
                [
                    "SOAK LOGGING COULD NOT START",
                    f"Failed to create log directory: {os.path.dirname(SOAK_LOG_PATH)}",
                    f"Reason: {exc}",
                    "Check folder permissions and that the path is valid (not on a read-only or missing drive).",
                    "The demo will continue running WITHOUT soak logging.",
                ]
            )
            self.any_exception = True
        else:
            self.write_ok = self._write(
                {
                    "record_type": "soak_start",
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "session_id": SESSION_ID,
                    "person_label": PERSON_LABEL,
                    "soak_interval_seconds": SOAK_INTERVAL_SECONDS,
                    "log_path": SOAK_LOG_PATH,
                }
            )
            if not self.write_ok:
                _loud_error(
                    [
                        "SOAK LOGGING COULD NOT START",
                        f"Failed to write the first record to: {SOAK_LOG_PATH}",
                        "The directory exists but the write itself failed -- check that the file",
                        "isn't open/locked elsewhere, and that this process has write permission.",
                        "The demo will continue running WITHOUT soak logging.",
                    ]
                )

        self._watch_stop = threading.Event()
        self._watch_thread = threading.Thread(target=self._watch_loop, name="SoakFPSWatcher", daemon=True)
        self._watch_thread.start()

        # Best-effort backstop for Ctrl-C / an unexpected exception: the
        # normal path is main()'s try/finally (deterministic, runs
        # first); this is a second, redundant net in case even that
        # somehow doesn't execute (e.g. an exception before the try
        # block is entered). write_summary() is idempotent, so if the
        # normal path already ran this is a silent no-op.
        atexit.register(self._atexit_fallback)

        if self.write_ok:
            print(f"[Soak] logging enabled -- writing to {SOAK_LOG_PATH} every {SOAK_INTERVAL_SECONDS:.0f}s")

    def _watch_loop(self):
        """Runs on its own thread only -- never the capture thread,
        never the render loop. Pure polling + an id() comparison; sleeps
        between checks so it spends almost all its time off the GIL.

        Uses time.sleep(), NOT Event.wait(), and a tight 2ms interval --
        confirmed by direct A/B testing against the live capture thread's
        own printed FPS (both across dozens of runs): Event.wait(0.005)
        measured a "capture FPS" of ~14-20 while the real rate was a
        steady ~30 the whole time (verified via s1.capture_thread's own
        console output) -- Event.wait's internal condition-variable
        locking wakes up measurably later under this app's GIL
        contention (render thread + processing thread both doing
        image/text work) than a plain sleep does. Swapping to
        time.sleep(0.002) alone closed the gap to within ~3% of the
        true rate in the same A/B setup. self._watch_stop is still
        checked every iteration for prompt (~2ms) shutdown."""
        while not self._watch_stop.is_set():
            with s1.frame_lock:
                frame = s1.latest_frame
            if frame is not None and id(frame) != self._last_frame_id:
                self._last_frame_id = id(frame)
                with self._count_lock:
                    self._frames_since_report += 1
            time.sleep(0.002)

    def stop(self):
        self._watch_stop.set()
        self._watch_thread.join(timeout=2)

    def _write(self, record):
        """Appends one JSONL line and forces it to disk before
        returning: f.flush() pushes Python's buffer to the OS, then
        os.fsync() forces the OS to commit it to physical storage
        rather than leaving it in the page cache. Needed so that a
        hard kill (not a clean 'q') still leaves every soak_sample
        written up to that moment recoverable -- only the final
        soak_summary (written after the loop exits) can be missing.
        Returns True/False so callers (the soak_start write especially)
        can tell whether logging is actually working, not just assume
        it silently is."""
        try:
            with open(SOAK_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as exc:  # instrumentation must never crash the demo it's watching
            self.any_exception = True
            print(f"[Soak] log write failed: {exc}")
            return False

    def maybe_report(self, t1, t2):
        now = time.perf_counter()
        elapsed_since_report = now - self.last_report_time
        if elapsed_since_report < SOAK_INTERVAL_SECONDS:
            return
        try:
            with self._count_lock:
                frames_seen = self._frames_since_report
                self._frames_since_report = 0
            capture_fps = frames_seen / elapsed_since_report
            self.last_report_time = now

            with ui_lock:  # existing value the processing thread already computes -- not a new measurement
                processing_rate = latest_ui_state.get("samples_per_sec")

            mem_mb, mem_source = _memory_rss_mb()

            capture_alive, processing_alive = t1.is_alive(), t2.is_alive()
            both_alive = capture_alive and processing_alive
            if not both_alive:
                self.any_thread_death = True

            self.fps_samples.append(capture_fps)
            if mem_mb is not None:
                self.mem_samples.append(mem_mb)
            self.n_samples += 1

            self._write(
                {
                    "record_type": "soak_sample",
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": round(now - self.start_time, 1),
                    "capture_fps": round(capture_fps, 2),
                    "processing_samples_per_sec": processing_rate,
                    "memory_rss_mb": round(mem_mb, 1) if mem_mb is not None else None,
                    "memory_source": mem_source,
                    "capture_thread_alive": capture_alive,
                    "processing_thread_alive": processing_alive,
                    "both_threads_alive": both_alive,
                    # Reflects the actual AGENT_LIVE flag this run was launched
                    # with -- not a live per-call measurement, just an honest
                    # record of what was wired for the soak (a soak should
                    # normally run on the stub, but if --live was also passed,
                    # this must say so truthfully rather than always claiming stub).
                    "agent_is_stub": not AGENT_LIVE,
                }
            )
            print(
                f"[Soak] t={now - self.start_time:.0f}s capture_fps={capture_fps:.1f} "
                f"processing={processing_rate} mem_mb={mem_mb} threads_alive={both_alive}"
            )
        except Exception as exc:  # instrumentation must never crash the demo it's watching
            self.any_exception = True
            print(f"[Soak] sample failed: {exc}")

    def write_summary(self, clean_exit=True):
        """Idempotent -- safe to call from both main()'s normal
        post-loop path AND the atexit fallback without producing two
        summary records. clean_exit distinguishes a real 'q' shutdown
        from a best-effort write triggered by the atexit safety net
        (Ctrl-C, an unexpected exception, ...) so a reader of the log
        knows which kind they're looking at."""
        if self._summary_written:
            return
        self._summary_written = True
        try:
            total_runtime = time.perf_counter() - self.start_time
            self._write(
                {
                    "record_type": "soak_summary",
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "clean_exit": clean_exit,
                    "total_runtime_seconds": round(total_runtime, 1),
                    "n_samples": self.n_samples,
                    "start_fps": round(self.fps_samples[0], 2) if self.fps_samples else None,
                    "end_fps": round(self.fps_samples[-1], 2) if self.fps_samples else None,
                    "min_fps": round(min(self.fps_samples), 2) if self.fps_samples else None,
                    "mean_fps": round(sum(self.fps_samples) / len(self.fps_samples), 2) if self.fps_samples else None,
                    "start_mem_mb": round(self.mem_samples[0], 1) if self.mem_samples else None,
                    "end_mem_mb": round(self.mem_samples[-1], 1) if self.mem_samples else None,
                    "peak_mem_mb": round(max(self.mem_samples), 1) if self.mem_samples else None,
                    "any_thread_death": self.any_thread_death,
                    "any_exception": self.any_exception,
                }
            )
            print(f"[Soak] summary written to {SOAK_LOG_PATH} (clean_exit={clean_exit})")
        except Exception as exc:
            print(f"[Soak] summary write failed: {exc}")

    def _atexit_fallback(self):
        if not self._summary_written:
            print("[Soak] atexit: process is exiting without a clean 'q' -- writing best-effort summary")
            self.write_summary(clean_exit=False)


def print_soak_report(log_path=None):
    """Reads back a soak_log.jsonl and prints a human-readable summary --
    `python stage3_demo_ui.py --soak-report [path]`. No camera, no
    consent, no threads: pure offline log parsing, so the human doesn't
    have to hand-parse JSONL after a long run.

    Computes duration/FPS/memory stats directly from the soak_sample
    records rather than only trusting a soak_summary record, because
    samples flush+fsync to disk immediately (see SoakTracker._write)
    while the summary is only written at the end -- this report is
    exactly as usable after a hard kill as after a clean 'q' exit,
    which is the whole point of flushing every sample immediately.
    """
    path = log_path or SOAK_LOG_PATH
    print(f">>> SOAK REPORT: {path}")

    if not os.path.isfile(path):
        print("    no log file found -- either soak logging was never enabled for a run, or this path is wrong.")
        print("    enable it with: python stage3_demo_ui.py --soak   (or STAGE3_SOAK=1)")
        return

    starts, samples, summaries = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(f"    [line {line_no}] skipping malformed JSON: {exc}")
                continue
            rtype = record.get("record_type")
            if rtype == "soak_sample":
                samples.append(record)
            elif rtype == "soak_start":
                starts.append(record)
            elif rtype == "soak_summary":
                summaries.append(record)

    if starts:
        s = starts[0]
        print(f"    run started: {s.get('ts_utc')}  session={s.get('session_id')}  interval={s.get('soak_interval_seconds')}s")

    if not samples:
        print("    0 soak_sample records -- either the run ended before the first interval elapsed, or logging failed at startup.")
        if summaries:
            print(f"    a soak_summary record exists anyway: {summaries[-1]}")
        return

    first_elapsed = samples[0].get("elapsed_seconds")
    last_elapsed = samples[-1].get("elapsed_seconds")
    span = (last_elapsed - first_elapsed) if (first_elapsed is not None and last_elapsed is not None) else None
    print(f"    {len(samples)} samples, covering elapsed {first_elapsed}s -> {last_elapsed}s" + (f"  (span {span:.0f}s)" if span is not None else ""))

    fps_vals = [s["capture_fps"] for s in samples if s.get("capture_fps") is not None]
    if fps_vals:
        print(
            f"    capture FPS:  start={fps_vals[0]:.2f}  end={fps_vals[-1]:.2f}  "
            f"min={min(fps_vals):.2f}  mean={sum(fps_vals) / len(fps_vals):.2f}"
        )
    else:
        print("    capture FPS: no data in any sample")

    mem_vals = [s["memory_rss_mb"] for s in samples if s.get("memory_rss_mb") is not None]
    if mem_vals:
        start_mem, end_mem, peak_mem = mem_vals[0], mem_vals[-1], max(mem_vals)
        print(f"    memory RSS:   start={start_mem:.1f}MB  end={end_mem:.1f}MB  peak={peak_mem:.1f}MB")
        if start_mem > 0:
            pct = (end_mem - start_mem) / start_mem * 100.0
            if pct > 10:
                trend = f"RISING (+{pct:.1f}% from start) -- worth a longer soak to confirm a real leak"
            elif pct < -10:
                trend = f"FALLING ({pct:.1f}% from start)"
            else:
                trend = f"stable ({pct:+.1f}% from start)"
            print(f"    memory trend: {trend}")
    else:
        print("    memory RSS: no data in any sample (memory_source may be 'unavailable' -- see samples)")

    dead = [s for s in samples if s.get("both_threads_alive") is False]
    if dead:
        first_dead = dead[0]
        print(f"    !!! {len(dead)} sample(s) show both_threads_alive=False -- first at elapsed={first_dead.get('elapsed_seconds')}s")
    else:
        print("    thread health: both threads alive on every sample")

    if summaries:
        s = summaries[-1]
        print(
            f"    soak_summary: present, clean_exit={s.get('clean_exit')}  "
            f"total_runtime={s.get('total_runtime_seconds')}s  "
            f"any_thread_death={s.get('any_thread_death')}  any_exception={s.get('any_exception')}"
        )
    else:
        print("    soak_summary: MISSING -- run never reached a clean 'q' exit or atexit fallback (process likely killed outright).")
        print("    (the stats above still stand -- they're computed from soak_sample records, which are flushed to disk immediately)")


# ============================================================
# PALETTE -- one definition, used everywhere. RGB (PIL-native);
# bgr() converts for cv2 shape calls. Saturated color is reserved for
# meaning: ACCENT = interpreted signal / stable live point, WARN =
# unstable/flagged, HONEST = the honest-framing accent (header/footer/
# calibrating notices). Everything else is low-chroma grey.
# ============================================================
BG = (16, 17, 20)
PANEL_BG = (25, 26, 31)
PANEL_BG_RECESSED = (20, 21, 25)     # logged-only sub-block: visually recedes
TRACK_BG = (36, 38, 45)
BORDER = (44, 46, 54)
BORDER_STRONG = (64, 67, 78)
TEXT_PRIMARY = (230, 232, 237)
TEXT_SECONDARY = (158, 163, 175)
TEXT_MUTED = (98, 102, 113)
ACCENT = (78, 216, 158)              # interpreted / live stable V/A point
ACCENT_MUTED = (92, 95, 105)         # logged-only bar fill -- desaturated, not colorful
WARN = (232, 96, 96)                 # unstable / low-confidence
HONEST = (223, 178, 92)              # honest-framing accent (amber)
HATCH = (40, 42, 49)


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def lerp_color(c1, c2, t):
    """Linear interpolation between two RGB tuples, t clamped to [0,1].
    Used for the V/A trail's age-fade (old points near PANEL_BG_RECESSED,
    the newest segment near the live point's own color) -- a rendering
    detail, not a new data value."""
    t = max(0.0, min(1.0, t))
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(3))


# Trail endpoints, derived from existing palette colors (not new hues) --
# faint/near-background at the oldest end, full ACCENT/WARN at the newest,
# same "saturated color reserved for meaning" rule as the rest of the palette.
TRAIL_FAINT = lerp_color(PANEL_BG_RECESSED, ACCENT, 0.18)
TRAIL_FAINT_WARN = lerp_color(PANEL_BG_RECESSED, WARN, 0.18)


# ============================================================
# TYPOGRAPHY -- 3-tier hierarchy (header / label / caption), plus a
# monospace tier for numeric readouts (instrument-panel feel, aligned
# digits). Loaded ONCE here, never per frame -- this runs at ~30 FPS.
# ============================================================
_FONT_DIR = r"C:\Windows\Fonts"


def _load_font(name, size):
    try:
        return ImageFont.truetype(os.path.join(_FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


FONTS = {
    "header": _load_font("segoeuib.ttf", 15),      # panel titles
    "label": _load_font("segoeui.ttf", 14),         # bar labels
    "label_bold": _load_font("segoeuib.ttf", 14),   # group headers, emphasis
    "caption": _load_font("segoeui.ttf", 12),       # secondary / notes / axis ticks
    "mono": _load_font("consola.ttf", 13),          # numeric z-value readouts
    "body": _load_font("segoeui.ttf", 18),          # agent-read paragraph -- the panel's main content, sized to fill it
}
LINE_H = {"header": 19, "label": 18, "label_bold": 18, "caption": 15, "mono": 16, "body": 26}

# --- Batched text: cv2 draws shapes immediately; every text string is
# queued here and composited onto the canvas in ONE PIL round trip per
# frame (flush_text), not one round trip per string.
_text_queue = []


def qtext(x, y, text, font_key="label", color=TEXT_PRIMARY, anchor="lt"):
    _text_queue.append((int(round(x)), int(round(y)), text, font_key, color, anchor))


def wrap_text(text, font_key, max_width):
    font = FONTS[font_key]
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if font.getlength(trial) > max_width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _format_mmss(total_seconds):
    total_seconds = max(0, int(total_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_ago(seconds):
    """None-safe 'Xs ago' / 'Xm ago' formatter for a real elapsed duration
    (never a fabricated freshness value -- callers pass None when there is
    genuinely no successful update yet)."""
    if seconds is None:
        return None
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s ago"
    return f"{int(seconds // 60)}m ago"


def flush_text(canvas):
    global _text_queue
    if not _text_queue:
        return canvas
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    for x, y, text, font_key, color, anchor in _text_queue:
        draw.text((x, y), text, font=FONTS[font_key], fill=color, anchor=anchor)
    _text_queue = []
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================
# LAYOUT -- FIXED CANVAS, everything must fit with NO overflow and NO
# scrolling: this is a live dashboard, not a document. 1280x720 is a
# safe, universal target (fits any modern display, including a 1366x768
# laptop screen with window chrome to spare).
#
# Every panel rect below is COMPUTED from CANVAS_W/H + the spacing
# constants (OUTER/GUTTER/PAD) -- nothing is a separately hand-tuned
# pixel that can silently grow past the canvas edge, which is exactly
# how the previous pass regressed (BARS_H was bumped to fit its own
# content with no check against the actual window size). If content
# ever doesn't fit a row's budget, shrink FONTS/LINE_H/the BAR_* metrics
# below -- never grow CANVAS_H/W to chase it.
#
# Grid: header band / [video | V/A plane] row / [vector bars | agent
# read] row / footer band. The AGENT READ row (ROW2) is sized FIRST and
# is the priority -- it must never be the panel that's clipped -- video
# gets whatever vertical budget is left over, then is scaled DOWN for
# DISPLAY to fit that budget (cv2.resize at render time). The actual
# camera CAPTURE stays at 640x480, untouched, in s1.capture_thread --
# this is a display-only scale, computed from CAMERA_W/H below.
# ============================================================
CANVAS_W = 1280
CANVAS_H = 720

OUTER = 10    # canvas edge margin
GUTTER = 8    # space between sibling panels
PAD = 12      # space between a panel's border and its content

HEADER_H = 24
STATUS_H = 22   # session-stats / quality-indicator strip, directly beneath the header
FOOTER_H = 22

CAMERA_W, CAMERA_H = 640, 480  # native capture resolution (s1.capture_thread) -- for display aspect-ratio math only

# V/A history trail: real per-cycle va_point values, downsampled at a fixed
# interval (not every processing cycle) purely to bound render cost -- the
# values themselves are never fabricated or interpolated, only sampled less
# often. 0.5s * 60s => at most ~120 points, cheap to redraw as a polyline.
VA_HISTORY_SECONDS = 60.0
VA_HISTORY_SAMPLE_INTERVAL_SECONDS = 0.5

STATUS_Y = OUTER + HEADER_H  # no gutter -- reads as one instrument cluster with the header above it
ROW1_Y = STATUS_Y + STATUS_H + GUTTER
FOOTER_Y = CANVAS_H - OUTER - FOOTER_H
_content_h = (FOOTER_Y - GUTTER) - ROW1_Y  # total budget for both content rows + the gutter between them

# ROW2_H (vector bars + agent read) is the priority row -- measured
# content need for the bars panel (the tighter of the two) is 354px at
# the current FONTS/BAR_* metrics (see tools/measure in dev notes,
# instrumented against the real draw_bar() calls, not hand math); 362
# keeps a real margin without starving ROW1 (video + V/A plane).
ROW2_H = 362

# EXPERIMENTAL_H: a THIRD row, between ROW1 and ROW2, holding all
# EXPERIMENTAL/UNVALIDATED overlays (horizontal head orientation, gaze
# L/R/C, blink rate) side by side -- deliberately its OWN row, not
# layered onto the video like an earlier version of the head-orientation
# badge alone was, because three such badges no longer fit inside the
# small video panel without overlap/clipping once gaze and blink joined
# it. Height was measured against the real wrap_text() output for the
# longest mandated disclaimer text at EXPERIMENTAL_BADGE_W (worst case:
# a 1-line title + 1 value line + a 2-line wrapped body = 4 text lines,
# ~87px including padding at the current FONTS/LINE_H) -- 96 keeps a
# small margin, same "measured, not guessed" discipline as ROW2_H above.
# Taken entirely out of ROW1 (video + V/A plane) rather than ROW2 (bars +
# agent), since ROW2 has near-zero slack already (see its own comment).
EXPERIMENTAL_H = 96

ROW1_H = _content_h - GUTTER - EXPERIMENTAL_H - GUTTER - ROW2_H  # video + V/A plane -- gets what's left after the priority row and the experimental strip
EXPERIMENTAL_Y = ROW1_Y + ROW1_H + GUTTER
ROW2_Y = EXPERIMENTAL_Y + EXPERIMENTAL_H + GUTTER

# Row 1: video, scaled DOWN for display to fit ROW1_H (native 4:3 aspect
# preserved) + V/A plane. The plane panel fills the FULL remaining row
# width (no cap) -- it is rendered as a filled rectangle (not forced
# into a small centered square), so the panel's whole footprint is
# actually used instead of leaving a large dead margin around a small
# plot. See draw_va_plane.
VIDEO_H = ROW1_H
VIDEO_W = int(VIDEO_H * CAMERA_W / CAMERA_H)
VIDEO_X, VIDEO_Y = OUTER, ROW1_Y

PLANE_PANEL_W = CANVAS_W - 2 * OUTER - GUTTER - VIDEO_W
PLANE_PANEL_H = ROW1_H

# EXPERIMENTAL SIGNALS strip: 3 equal-width badges (head yaw, gaze,
# blink) spanning the row, canvas-relative (not video-relative, unlike
# the old single video-overlay badge) since they now live in their own
# full-width row. 2 gutters between 3 badges.
EXPERIMENTAL_BADGE_W = (CANVAS_W - 2 * OUTER - 2 * GUTTER) // 3
EXPERIMENTAL_BADGE_X0 = OUTER
PLANE_X, PLANE_Y = VIDEO_X + VIDEO_W + GUTTER, ROW1_Y

# Row 2: vector bars (wide -- needs room for label + track + value) +
# agent read (remainder -- the priority panel gets whatever's left, and
# more width is strictly better for it, so it is NOT capped). BARS_W is
# trimmed from its earlier 700 to 660 -- still comfortable margin for the
# longest bar label at the current font (measured well under 660-2*PAD) --
# so the priority agent-read panel gets the freed 40px.
BARS_W = 660
BARS_X, BARS_Y = OUTER, ROW2_Y
BARS_H = ROW2_H

AGENT_W = CANVAS_W - 2 * OUTER - GUTTER - BARS_W
AGENT_X, AGENT_Y = BARS_X + BARS_W + GUTTER, ROW2_Y
AGENT_H = ROW2_H

# Vector-bar row metrics -- named constants so every row is computed,
# never hand-placed (instruction: "define as constants, don't
# hand-place pixels"). A "noted" row is one with a LOGGED-ONLY caption
# line beneath it (V_bf / V_jc); a "plain" row has none (V_es / V_pd).
BAR_TRACK_GAP = 4
BAR_TRACK_H = 12
BAR_NOTE_GAP = 4
BAR_ROW_GAP = 12
BAR_AFTER_DIVIDER_GAP = 14
BAR_VALUE_COL_W = 80  # reserved width for the right-aligned mono z-value
ROW_H_PLAIN = LINE_H["label"] + BAR_TRACK_GAP + BAR_TRACK_H
ROW_H_NOTED = ROW_H_PLAIN + BAR_NOTE_GAP + LINE_H["caption"]

ui_lock = threading.Lock()

# Agent-call state -- guarded by its OWN small lock (_agent_call_lock),
# separate from ui_lock, because it protects a plain module-level flag
# ("is a call in flight"), not the shared render state. See
# _launch_agent_call.
_agent_call_lock = threading.Lock()
_agent_call_in_flight = False

_WAITING_POINTERS = [
    {
        "headline": "Waiting for the first window…",
        "detail": "The behavioral read appears once calibration finishes and the first 10-second window completes.",
    }
]

latest_ui_state = {
    "face_detected": False,
    "pose_detected": False,
    "calibrated": False,
    "calibration_seconds_remaining": x_core.CALIBRATION_SECONDS,
    "z_scores": {"v_bf": None, "v_es": None, "v_pd": None, "v_jc": None},
    "va_point": {"valence": None, "arousal": None},
    "window_flagged": False,
    "agent_text": "Waiting for calibration and the first 10s window...",
    "agent_pointers": _WAITING_POINTERS,
    "agent_pending": False,
    "agent_last_updated_monotonic": None,
    "yaw_deg": None,
    "samples_per_sec": 0.0,
    # --- genuine, real-data-only additions (presentation layer; every
    # value below is read from state stage1's own pipeline already
    # computes -- see stage3_processing_thread) ---
    "va_history": [],           # last VA_HISTORY_SECONDS of real va_point samples, for the trail
    "windows_processed": 0,     # real count of WindowAccumulator.flush() calls this session
    "calibration_quality": None,  # NeutralCalibrator.complete()'s own "quality" dict, once calibrated
    "last_window": {"low_confidence": False, "reasons": [], "detection_rate": None, "n_samples": None},
    "session_elapsed_seconds": 0.0,
    "head_yaw_deg": None,  # HORIZONTAL HEAD ORIENTATION (experimental) -- see stage3_processing_thread
    # EXPERIMENTAL SIGNALS, PART 2 -- gaze L/R/C + blink rate. Both
    # UNVALIDATED, both isolated from the affect vectors -- see
    # stage3_processing_thread and stage1_step4_vectors.py's
    # compute_gaze_direction / BlinkDetector.
    "gaze_label": None,             # "LEFT" / "RIGHT" / "CENTER" / "UNKNOWN" / None (no face this cycle)
    "gaze_reliable": None,
    "blink_rate_per_min": None,
    "blink_measuring": True,
    "blink_count_in_rate_window": 0,
    # Blink DIAGNOSTIC fields (BlinkDetector.snapshot()'s own live
    # readout) -- added to debug/verify the blink-count-always-0 bug
    # against real behavior instead of re-tuning blind. See
    # BLINK_APERTURE_PLAUSIBLE_MIN's comment in stage1_step4_vectors.py.
    "blink_last_aperture": None,
    "blink_current_state": None,   # "open" / "closing"
    "blink_just_blinked": False,
}


def draw_panel_frame(canvas, x, y, w, h, title, subtitle=None):
    """Shared panel chrome: background, border, uppercase header, an
    optional caption subtitle, and a divider. Returns content_top -- the
    y-coordinate callers should start drawing body content at, so every
    panel's content begins from a consistently-derived offset instead of
    a separately hand-tuned one."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr(PANEL_BG), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr(BORDER), 1)

    ty = y + PAD
    qtext(x + PAD, ty, title.upper(), "header", TEXT_SECONDARY)
    ty += LINE_H["header"]
    if subtitle:
        qtext(x + PAD, ty, subtitle, "caption", TEXT_MUTED)
        ty += LINE_H["caption"]

    divider_y = ty + 2
    cv2.line(canvas, (x + PAD, divider_y), (x + w - PAD, divider_y), bgr(BORDER), 1)
    return divider_y + PAD


def draw_bar(canvas, x, y, w, label, z_value, interpreted, note=None):
    """One bar row: label, a centered-at-zero horizontal track (z
    clipped to [-3, 3]), and a right-hand mono z-value readout.
    interpreted=False desaturates the fill and appends a LOGGED-ONLY
    caption line -- the visual cue that this vector feeds no reading,
    not just a text note. Returns the y just below this row's content,
    so callers can stack rows without hand-placed offsets."""
    label_color = TEXT_PRIMARY if interpreted else TEXT_SECONDARY
    fill_color = ACCENT if interpreted else ACCENT_MUTED
    value_color = ACCENT if interpreted else TEXT_SECONDARY

    qtext(x, y, label, "label", label_color)

    track_y = y + LINE_H["label"] + BAR_TRACK_GAP
    track_x0, track_x1 = x, x + w
    cx = (track_x0 + track_x1) // 2
    cv2.rectangle(canvas, (track_x0, track_y), (track_x1, track_y + BAR_TRACK_H), bgr(TRACK_BG), -1)
    cv2.line(canvas, (cx, track_y - 2), (cx, track_y + BAR_TRACK_H + 2), bgr(BORDER_STRONG), 1)

    value_mid_y = track_y + BAR_TRACK_H / 2
    if z_value is None:
        qtext(track_x1 + 14, value_mid_y, "n/a", "mono", TEXT_MUTED, anchor="lm")
    else:
        clipped = max(-3.0, min(3.0, z_value))
        half = (track_x1 - track_x0) // 2
        offset = int(clipped / 3.0 * half)
        if offset >= 0:
            cv2.rectangle(canvas, (cx, track_y), (cx + offset, track_y + BAR_TRACK_H), bgr(fill_color), -1)
        else:
            cv2.rectangle(canvas, (cx + offset, track_y), (cx, track_y + BAR_TRACK_H), bgr(fill_color), -1)
        qtext(track_x1 + 14, value_mid_y, f"{z_value:+.2f}", "mono", value_color, anchor="lm")

    bottom = track_y + BAR_TRACK_H
    if not interpreted:
        note_y = bottom + BAR_NOTE_GAP
        qtext(x, note_y, "LOGGED-ONLY", "caption", ACCENT_MUTED)
        if note:
            note_x = x + FONTS["caption"].getlength("LOGGED-ONLY") + 10
            qtext(note_x, note_y, f"· {note}", "caption", TEXT_MUTED)
        bottom = note_y + LINE_H["caption"]
    return bottom


def draw_vector_bars(canvas, state):
    content_top = draw_panel_frame(canvas, BARS_X, BARS_Y, BARS_W, BARS_H, "Behavioral Signals", subtitle="live · this window")
    x = BARS_X + PAD
    w = BARS_W - 2 * PAD - BAR_VALUE_COL_W

    if not state["calibrated"]:
        y = content_top + 6
        qtext(x, y, "Calibrating…", "label_bold", HONEST)
        y += LINE_H["label_bold"] + 6
        qtext(x, y, f"{state['calibration_seconds_remaining']:.0f}s remaining — relax your face", "caption", TEXT_MUTED)
        y += LINE_H["caption"] + 10
        qtext(x, y, "No interpreted reading until calibration completes.", "caption", TEXT_MUTED)
        return

    z = state["z_scores"]
    y = content_top

    group1_top = y
    qtext(x, y, "INTERPRETED — drives Valence / Arousal", "label_bold", ACCENT)
    y += LINE_H["label_bold"] + 10
    y = draw_bar(canvas, x, y, w, "V_es    eye-crinkle / pleasure  →  Valence", z["v_es"], True) + BAR_ROW_GAP
    y = draw_bar(canvas, x, y, w, "V_pd    postural volatility  →  Arousal", z["v_pd"], True)
    group1_bottom = y
    cv2.rectangle(canvas, (BARS_X + 8, group1_top - 2), (BARS_X + 11, group1_bottom + 2), bgr(ACCENT), -1)

    y += BAR_ROW_GAP
    cv2.line(canvas, (x, y), (BARS_X + BARS_W - PAD, y), bgr(BORDER), 1)
    y += BAR_AFTER_DIVIDER_GAP

    group2_top = y
    qtext(x, y, "LOGGED-ONLY — not interpreted, feeds no reading", "label_bold", TEXT_MUTED)
    y += LINE_H["label_bold"] + 10

    # Recessed tint behind the logged-only pair, drawn before the bars
    # so the bars render on top of it -- the instant, at-a-glance visual
    # demotion the instruction asks for, beyond the text note alone.
    recess_top = y - 10
    recess_bottom = y + ROW_H_NOTED + BAR_ROW_GAP + ROW_H_NOTED + 8
    cv2.rectangle(canvas, (BARS_X + PAD - 8, recess_top), (BARS_X + BARS_W - PAD + 8, recess_bottom), bgr(PANEL_BG_RECESSED), -1)

    y = draw_bar(canvas, x, y, w, "V_bf    brow furrow", z["v_bf"], False, note="did not generalize at Gate 2") + BAR_ROW_GAP
    y = draw_bar(canvas, x, y, w, "V_jc    jaw compression", z["v_jc"], False, note="no reliable signal found")
    group2_bottom = y
    cv2.rectangle(canvas, (BARS_X + 8, group2_top - 2), (BARS_X + 11, group2_bottom + 2), bgr(ACCENT_MUTED), -1)


def draw_va_plane(canvas, state):
    """Honest-framing rule (non-negotiable, Decision 18): Valence is
    z_es only, pleasure-side, single-source -- there is NO validated
    pain axis. The negative-valence half is hatched, tinted, and
    labeled so a negative reading can never be mistaken for a measured
    "pain" signal. A flagged (low-confidence) window renders red /
    UNSTABLE and must never look like a confident point.

    The plot fills the panel's full content rectangle (width AND
    height, minus padding) rather than being forced into a square --
    this panel is much wider than it is tall (video's 4:3 aspect at a
    reasonable row height leaves a lot of row width for it), and a
    square sized to the tighter (height) dimension left most of that
    width as dead space. Filling the rectangle uses the space that's
    already allocated to this panel instead of leaving it empty; valence
    and arousal simply get independent horizontal/vertical scales (half_w
    / half_h below) rather than one shared square scale -- sign and
    relative magnitude within each axis are unaffected, nothing about
    what's measured or how changes."""
    # No subtitle here (unlike the other panels) -- the footer already
    # carries "not emotion, not clinical, not diagnostic", and this
    # panel is the tightest on vertical room; the freed line goes to
    # the plot itself instead.
    content_top = draw_panel_frame(canvas, PLANE_X, PLANE_Y, PLANE_PANEL_W, PLANE_PANEL_H, "Valence / Arousal")

    caption_band_h = 20
    plane_x0, plane_y0 = PLANE_X + PAD, content_top
    plane_x1 = PLANE_X + PLANE_PANEL_W - PAD
    plane_y1 = (PLANE_Y + PLANE_PANEL_H) - PAD - caption_band_h
    cx, cy = (plane_x0 + plane_x1) // 2, (plane_y0 + plane_y1) // 2

    cv2.rectangle(canvas, (plane_x0, plane_y0), (plane_x1, plane_y1), bgr(PANEL_BG_RECESSED), -1)

    # Negative-valence (pain) half: subtle hatch, low-contrast tint --
    # tasteful, not loud, but always present. Must never read as a
    # measurement. Diagonal reach is plane_h (not the width) so the
    # 45-degree hatch stays correctly angled regardless of how wide the
    # half is; the offset range just tiles it across the full width.
    plane_h = plane_y1 - plane_y0
    half_span = cx - plane_x0
    overlay = canvas.copy()
    for offset in range(-plane_h, half_span, 16):
        x_a, y_a = plane_x0 + offset, plane_y1
        x_b, y_b = plane_x0 + offset + plane_h, plane_y0
        cv2.line(overlay, (x_a, y_a), (x_b, y_b), bgr(HATCH), 1, cv2.LINE_AA)
    canvas[plane_y0:plane_y1, plane_x0:cx] = cv2.addWeighted(
        canvas[plane_y0:plane_y1, plane_x0:cx], 0.6, overlay[plane_y0:plane_y1, plane_x0:cx], 0.4, 0
    )
    cv2.rectangle(canvas, (plane_x0, plane_y0), (plane_x1, plane_y1), bgr(BORDER), 1)
    cv2.line(canvas, (plane_x0, cy), (plane_x1, cy), bgr(BORDER), 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, plane_y0), (cx, plane_y1), bgr(BORDER), 1, cv2.LINE_AA)

    # Plenty of width now (the rectangle fills the panel), so the full
    # sentence fits on one line -- no more forced short fragments.
    qtext((plane_x0 + cx) / 2, plane_y0 + 12, "No validated pain axis — not measured", "caption", TEXT_MUTED, anchor="mm")
    qtext((cx + plane_x1) / 2, plane_y0 + 12, "pleasure-side signal", "caption", TEXT_MUTED, anchor="mm")

    qtext(cx, plane_y0 + 8, "arousal", "caption", TEXT_SECONDARY, anchor="mt")
    qtext(plane_x1 - 10, cy, "valence", "caption", TEXT_SECONDARY, anchor="rm")

    if not state["calibrated"]:
        qtext(cx, cy, "CALIBRATING…", "header", HONEST, anchor="mm")
        return

    va = state["va_point"]
    valence, arousal = va.get("valence"), va.get("arousal")
    if valence is None or arousal is None:
        qtext(cx, cy, "n/a", "label_bold", TEXT_MUTED, anchor="mm")
        return

    half_w = (plane_x1 - plane_x0) // 2 - 24
    half_h = (plane_y1 - plane_y0) // 2 - 20

    # V/A HISTORY TRAIL (headline feature) -- real per-cycle va_point
    # samples from the last VA_HISTORY_SECONDS (state["va_history"],
    # built in stage3_processing_thread; see its own comment there for
    # how it's sampled). Drawn as a fading polyline so movement over
    # time is visible, not just the instantaneous point -- every vertex
    # is a value this pipeline actually computed, nothing interpolated
    # or fabricated between real windows. Older segments fade toward the
    # panel background; the newest approaches the live point's own
    # color -- age and the segment's own window_flagged state (WARN
    # instead of ACCENT) are the only two things that change a segment's
    # color, both real, same rule the live point already follows.
    history = state["va_history"]
    if len(history) >= 2:
        n = len(history)
        for i in range(1, n):
            p0, p1 = history[i - 1], history[i]
            x0 = int(cx + p0["valence"] * half_w)
            y0 = int(cy - p0["arousal"] * half_h)
            x1 = int(cx + p1["valence"] * half_w)
            y1 = int(cy - p1["arousal"] * half_h)
            age_t = i / (n - 1)  # 0 = oldest segment, 1 = newest
            seg_color = lerp_color(TRAIL_FAINT_WARN, WARN, age_t) if p1["flagged"] else lerp_color(TRAIL_FAINT, ACCENT, age_t)
            cv2.line(canvas, (x0, y0), (x1, y1), bgr(seg_color), 2 if age_t > 0.75 else 1, cv2.LINE_AA)

    px = int(cx + valence * half_w)
    py = int(cy - arousal * half_h)  # screen y is inverted vs. arousal-up

    point_color = WARN if state["window_flagged"] else ACCENT
    cv2.circle(canvas, (px, py), 13, bgr(point_color), 2, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), 6, bgr(point_color), -1, cv2.LINE_AA)

    if state["window_flagged"]:
        qtext(cx, plane_y1 + 10, "UNSTABLE — low-confidence window", "caption", WARN, anchor="mt")


POINTER_GROUP_GAP = 10   # vertical gap between one pointer's block and the next
POINTER_DETAIL_INDENT = 16
POINTER_DETAIL_MAX_LINES = 2


def draw_agent_panel(canvas, state):
    """The priority panel -- its reading is the main point of the demo.
    Restructured as POINTERS + DETAIL (Decision: short, warm headlines
    the eye catches first, each grounded by one honest detail line
    underneath) instead of one dense paragraph -- state["agent_pointers"]
    is an ordered list of {"headline", "detail"} dicts, identical in
    shape whether the agent call was stubbed or live (see
    parse_agent_pointers in stage2_personality_agent.py)."""
    subtitle = "LIVE · real API · refreshes each 10s window" if AGENT_LIVE else "auto-refreshes every 10s · stub agent, no live calls"
    if state.get("agent_pending"):
        subtitle += "  ·  updating…"
    else:
        # Real freshness readout -- time since the last successful agent
        # response actually landed (agent_last_updated_monotonic, stamped
        # in _launch_agent_call's worker thread), not a fabricated status.
        # None before the first successful call this session -- omitted
        # rather than shown as "0s ago" or similar.
        updated = state.get("agent_last_updated_monotonic")
        if updated is not None:
            ago = _format_ago(time.perf_counter() - updated)
            subtitle += f"  ·  updated {ago}"
    content_top = draw_panel_frame(canvas, AGENT_X, AGENT_Y, AGENT_W, AGENT_H, "Behavioral Read", subtitle=subtitle)

    x = AGENT_X + PAD
    max_w = AGENT_W - 2 * PAD
    bottom_limit = (AGENT_Y + AGENT_H) - PAD
    y = content_top

    pointers = state["agent_pointers"]
    for idx, pointer in enumerate(pointers):
        if y + LINE_H["label_bold"] > bottom_limit:
            break  # defensive only -- panel is sized to fit 4 pointers at these metrics
        qtext(x, y, f"●  {pointer['headline']}", "label_bold", ACCENT)
        y += LINE_H["label_bold"] + 4

        if pointer["detail"]:
            for line in wrap_text(pointer["detail"], "label", max_w - POINTER_DETAIL_INDENT)[:POINTER_DETAIL_MAX_LINES]:
                if y + LINE_H["label"] > bottom_limit:
                    break
                qtext(x + POINTER_DETAIL_INDENT, y, line, "label", TEXT_SECONDARY)
                y += LINE_H["label"]

        # Subtle divider between pointer blocks -- purely visual grouping,
        # drawn within the existing POINTER_GROUP_GAP so it costs no extra
        # vertical room (nothing shrinks to make space for it).
        if idx < len(pointers) - 1:
            cv2.line(canvas, (x, y + POINTER_GROUP_GAP // 2), (x + max_w, y + POINTER_GROUP_GAP // 2), bgr(BORDER), 1)
        y += POINTER_GROUP_GAP


def draw_header(canvas, state):
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, OUTER + HEADER_H), bgr(PANEL_BG), -1)
    cv2.line(canvas, (0, OUTER + HEADER_H), (CANVAS_W, OUTER + HEADER_H), bgr(BORDER), 1)

    mid_y = (OUTER + HEADER_H) / 2
    qtext(OUTER, mid_y, "Behavioral Signals Demo", "header", TEXT_PRIMARY, anchor="lm")

    status_color = ACCENT if state["calibrated"] else HONEST
    status_text = f"● {'calibrated' if state['calibrated'] else 'calibrating'}"
    agent_mode_text = "live agent · real API calls" if AGENT_LIVE else "stub agent · no live API calls"
    stats_text = f"{agent_mode_text}    person={PERSON_LABEL}    {state['samples_per_sec']:.1f}/s"

    status_w = FONTS["caption"].getlength(status_text)
    qtext(CANVAS_W - OUTER, mid_y, status_text, "caption", status_color, anchor="rm")
    qtext(CANVAS_W - OUTER - status_w - 18, mid_y, stats_text, "caption", TEXT_MUTED, anchor="rm")


def draw_status_strip(canvas, state):
    """Session-stats + live quality-indicator strip, directly beneath the
    header (STATUS_Y/STATUS_H). Every value drawn here is read straight
    off state -- real counts/timers/flags stage3_processing_thread already
    computes (elapsed session time, WindowAccumulator flush count, the
    last flushed window's classify_window_confidence result, the neutral
    calibration's own classify_calibration_quality flag). Nothing here is
    a new measurement, only a new, more visible place to show existing
    ones -- same discipline as the vector bars' LOGGED-ONLY demotion:
    surface the real signal, don't invent a friendlier one.

    A low-confidence window or a flagged calibration must still render
    clearly as such (WARN / HONEST), never softened into the same color
    as a clean reading -- this is the same "flag, don't suppress" rule
    the V/A plane's UNSTABLE marker already follows."""
    cv2.rectangle(canvas, (0, STATUS_Y), (CANVAS_W, STATUS_Y + STATUS_H), bgr(PANEL_BG), -1)
    cv2.line(canvas, (0, STATUS_Y + STATUS_H), (CANVAS_W, STATUS_Y + STATUS_H), bgr(BORDER), 1)
    mid_y = STATUS_Y + STATUS_H / 2

    left_text = f"session {_format_mmss(state['session_elapsed_seconds'])}   ·   windows processed {state['windows_processed']}"
    qtext(OUTER, mid_y, left_text, "mono", TEXT_SECONDARY, anchor="lm")

    lw = state["last_window"]
    if state["windows_processed"] == 0:
        window_text, window_color = "window: pending first 10s window", TEXT_MUTED
    elif lw["low_confidence"]:
        reason = lw["reasons"][0] if lw["reasons"] else "flagged"
        window_text, window_color = f"window: LOW-CONFIDENCE ({reason})", WARN
    else:
        rate_text = f"{lw['detection_rate'] * 100:.0f}% detected" if lw["detection_rate"] is not None else ""
        window_text = f"window: stable ({rate_text})" if rate_text else "window: stable"
        window_color = ACCENT

    cq = state["calibration_quality"]
    if cq is None:
        calib_text, calib_color = "calibration: in progress", TEXT_MUTED
    elif cq["possibly_not_neutral"]:
        calib_text, calib_color = "calibration: check neutral", HONEST
    else:
        calib_text, calib_color = "calibration: clean", ACCENT

    # Right-aligned pair, positioned from the edge inward -- same
    # measure-then-place idiom draw_header already uses for its own
    # two-segment right-hand readout.
    window_w = FONTS["mono"].getlength(window_text)
    qtext(CANVAS_W - OUTER, mid_y, window_text, "mono", window_color, anchor="rm")
    qtext(CANVAS_W - OUTER - window_w - 18, mid_y, calib_text, "mono", calib_color, anchor="rm")


def draw_footer(canvas):
    cv2.rectangle(canvas, (0, FOOTER_Y), (CANVAS_W, CANVAS_H), bgr((13, 14, 16)), -1)
    cv2.line(canvas, (0, FOOTER_Y), (CANVAS_W, FOOTER_Y), bgr(BORDER), 1)
    text = "Behavioral signals / affective indicators only — not emotion, not clinical, not diagnostic. No score. No pass/fail."
    qtext(OUTER, (FOOTER_Y + CANVAS_H) / 2, text, "caption", HONEST, anchor="lm")


# --- Chrome cache -------------------------------------------------
# PERFORMANCE NOTE (measured, not guessed): a full PIL text composite
# (~33 strings + BGR<->RGB round trip on the ~1168x1100 canvas) costs
# ~50ms. Python's GIL means that cost is NOT free just because it runs
# in the main/render thread -- it competes for interpreter time with
# stage3_processing_thread's own numpy/MediaPipe work. Redoing the full
# composite on every uncapped render-loop iteration measurably dropped
# processing throughput (~14.5 samples/sec -> ~9-11/sec) during testing.
#
# Fix: everything except the live video pixels ("chrome" -- header,
# both panels row, footer) only actually changes at the PROCESSING
# thread's own cadence (~10-15Hz, one update per detection cycle), not
# at render-loop speed. So the expensive PIL composite is rebuilt only
# when latest_ui_state's content actually changes (_chrome_state_key),
# cached, and reused for every render-loop pass in between -- those
# passes just numpy-copy the cached chrome and blit the fresh frame in,
# no PIL/color-conversion involved. This is the "cache, don't revert to
# putText" instruction: an explicit, measured optimization, not a
# logic change -- what gets drawn and when it must reflect fresh state
# (face_detected, window_flagged, every z-score) is unchanged; only how
# often the expensive draw path re-runs is different.
_chrome_cache = {"key": None, "canvas": None}
_badge_cache = {"canvas": None}


def _chrome_state_key(state):
    z, va = state["z_scores"], state["va_point"]
    pointers_key = tuple((p["headline"], p["detail"]) for p in state["agent_pointers"])
    history = state["va_history"]
    # Fingerprint, not the full history: (count, newest timestamp) already
    # changes on every append AND on every time-based trim at the tail, so
    # it correctly detects both cases without hashing up to ~120 points'
    # worth of floats on every rebuild-check.
    history_key = (len(history), round(history[-1]["t"], 1) if history else None)
    lw = state["last_window"]
    cq = state["calibration_quality"]
    agent_updated = state.get("agent_last_updated_monotonic")
    # Bucketed to 5s: the freshness caption ("updated Xs ago") doesn't need
    # a full chrome rebuild every single second to stay honest, just often
    # enough that it doesn't visibly go stale.
    agent_updated_bucket = round(agent_updated / 5.0) if agent_updated is not None else None
    return (
        state["calibrated"],
        None if state["calibrated"] else round(state["calibration_seconds_remaining"]),
        z["v_bf"], z["v_es"], z["v_pd"], z["v_jc"],
        va.get("valence"), va.get("arousal"),
        state["window_flagged"],
        pointers_key,
        state["agent_pending"],
        round(state["samples_per_sec"], 1),
        history_key,
        state["windows_processed"],
        lw["low_confidence"], lw["detection_rate"], lw["n_samples"],
        cq["possibly_not_neutral"] if cq else None,
        round(state["session_elapsed_seconds"]),  # ticks the status strip's mm:ss at least once/sec
        agent_updated_bucket,
        # EXPERIMENTAL SIGNALS, PART 2 -- yaw already covered by nothing
        # else in this key (it's not one of the affect vectors above), so
        # it's added explicitly; gaze/blink likewise.
        round(state.get("head_yaw_deg"), 1) if state.get("head_yaw_deg") is not None else None,
        state.get("gaze_label"), state.get("gaze_reliable"),
        round(state.get("blink_rate_per_min"), 1) if state.get("blink_rate_per_min") is not None else None,
        state.get("blink_measuring"),
        # Blink DIAGNOSTIC fields -- these change essentially every cycle
        # a face is detected (same as the affect z-scores above already
        # do), so this doesn't introduce a new class of rebuild-frequency
        # cost, just correctness: without these, the live diagnostic
        # readout in the blink badge would silently lag behind the
        # cached chrome.
        round(state.get("blink_last_aperture"), 3) if state.get("blink_last_aperture") is not None else None,
        state.get("blink_current_state"), state.get("blink_just_blinked"),
    )


def _build_chrome(state):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = bgr(BG)
    draw_header(canvas, state)
    draw_status_strip(canvas, state)
    draw_vector_bars(canvas, state)
    draw_va_plane(canvas, state)
    draw_experimental_strip(canvas, state)
    draw_agent_panel(canvas, state)
    draw_footer(canvas)
    return flush_text(canvas)


def _get_no_face_badge():
    """Pre-rendered once (static content, same every time it's shown)
    so the badge never needs a per-frame PIL round trip either -- it's
    pasted as a plain numpy slice, same as the video frame itself."""
    if _badge_cache["canvas"] is None:
        bw, bh = 176, 30
        badge = np.zeros((bh, bw, 3), dtype=np.uint8)
        badge[:] = bgr((42, 22, 22))
        cv2.rectangle(badge, (0, 0), (bw - 1, bh - 1), bgr(WARN), 1)
        global _text_queue
        saved_queue, _text_queue = _text_queue, []
        qtext(12, bh / 2, "NO FACE DETECTED", "caption", WARN, anchor="lm")
        badge = flush_text(badge)
        _text_queue = saved_queue
        _badge_cache["canvas"] = badge
    return _badge_cache["canvas"]


# ============================================================
# EXPERIMENTAL SIGNALS STRIP -- horizontal head orientation, gaze L/R/C,
# blink rate. ALL THREE are EXPERIMENTAL / UNVALIDATED and share ONE
# visual language (amber-bordered badge, same shared renderer below) so
# "this isn't validated" reads as one consistent vocabulary rather than
# three ad hoc treatments. Deliberately their OWN full-width row
# (EXPERIMENTAL_Y/EXPERIMENTAL_H), NOT overlaid on the video and NOT a
# peer panel next to V_es/V_pd -- three badges no longer fit inside the
# small video panel without overlap once gaze and blink joined the
# original head-orientation badge, and placing them among the validated
# cards would blur the "must not look like a validated reading" line.
# Pitch/vertical gaze are NEVER computed for display here, or anywhere
# in this file -- CLAUDE.md's own finding is that pitch is structurally
# unreliable (a maximal look-down reads as ~0.1deg, flat), and vertical
# gaze would degrade the same way.
# ============================================================
HEAD_ORIENTATION_LABEL_TITLE = "Horizontal head orientation — EXPERIMENTAL / UNVALIDATED"
HEAD_ORIENTATION_LABEL_BODY = (
    "Detects left/right head turn only. Does NOT detect looking up/down. "
    "This is NOT an attention, engagement, or focus measure."
)
GAZE_LABEL_TITLE = "Approximate gaze (L/R/C) — EXPERIMENTAL / UNVALIDATED"
GAZE_LABEL_BODY = (
    "Horizontal only, NOT up/down. Degrades with glasses. "
    "Not a validated attention measure."
)
BLINK_LABEL_TITLE = "Blink rate (blinks/min) — EXPERIMENTAL / UNVALIDATED"
BLINK_LABEL_BODY = "Approximate. Not counted when eyes aren't reliably visible."

# Display-only: which sign of yaw_deg maps to the word "RIGHT" vs "LEFT".
# yaw_deg itself comes straight from compute_v_so's own output, unchanged
# -- this constant only relabels TEXT and was not independently
# re-verified against a live camera in this pass (CLAUDE.md: "state what
# was confirmed -- do not assume"). If a live on-camera check shows the
# word is backwards for your setup, flip this to -1 -- it can only ever
# change which word is printed, never yaw_deg itself, never compute_v_so,
# and never anything the affect signals or V/A mapping read. (Gaze's own
# equivalent, GAZE_LABEL_SIGN, lives in stage1_step4_vectors.py --
# baked into compute_gaze_direction's OWN return value there rather than
# split display-side like this one, because gaze's LABEL, not just its
# raw number, is what gets logged for later cross-person validation, so
# the sign convention must be identical between logging and display.)
HEAD_YAW_LABEL_SIGN = 1

_UNSET = object()  # sentinel distinct from any real cache key, including None (a legitimate "no reading" value for several of these badges)
_yaw_badge_cache = {"key": _UNSET, "canvas": None}
_gaze_badge_cache = {"key": _UNSET, "canvas": None}
_blink_badge_cache = {"key": _UNSET, "canvas": None}


def _head_yaw_direction_label(yaw_deg):
    """CENTERED vs TURNED uses only magnitude, against the SAME threshold
    compute_v_so itself already uses for its own yaw contribution
    (attention.ATTENTION_YAW_THRESHOLD_DEG) -- reused, not a new number. The
    LEFT/RIGHT word additionally reads yaw_deg's sign; see
    HEAD_YAW_LABEL_SIGN above for that mapping's own caveat."""
    if yaw_deg is None:
        return "n/a"
    if abs(yaw_deg) < attention.ATTENTION_YAW_THRESHOLD_DEG:
        return "CENTERED"
    return "TURNED RIGHT" if (yaw_deg * HEAD_YAW_LABEL_SIGN) > 0 else "TURNED LEFT"


def _build_experimental_badge(width, title, value_text, body_text, diagnostic_text=None):
    """Shared renderer for EVERY badge in the experimental strip (head
    yaw, gaze, blink). Both title AND body are wrapped -- an earlier
    version of the (then video-overlaid, yaw-only) badge only wrapped
    the body, and the title silently lost its trailing "UNVALIDATED"
    past the canvas edge (PIL draws onto a fixed-size buffer; text past
    its bounds is just never rendered, no error). That word is part of
    the MANDATORY label text, so it must always be visible -- confirmed
    by rendering and inspecting before that fix, and title-wrapping
    stays mandatory here for exactly that reason. Height is computed
    from the actual wrapped line count, never hardcoded, so mandatory
    text can never silently clip if it re-wraps differently.

    diagnostic_text (optional): one extra muted caption line between the
    value and the body, for a live debugging readout (currently only the
    blink badge uses this, for raw aperture + detector state -- see
    _get_blink_badge). Not wrapped: kept intentionally short/compact by
    its caller so it always fits on one line."""
    pad = 8
    inner_w = width - 2 * pad
    title_lines = wrap_text(title, "caption", inner_w)
    body_lines = wrap_text(body_text, "caption", inner_w)

    lines = [(line, "caption", HONEST, LINE_H["caption"]) for line in title_lines]
    lines[-1] = (lines[-1][0], lines[-1][1], lines[-1][2], LINE_H["caption"] + 3)  # extra gap after the title block
    lines.append((value_text, "label", TEXT_PRIMARY, LINE_H["label"] + 5))
    if diagnostic_text:
        lines.append((diagnostic_text, "caption", TEXT_SECONDARY, LINE_H["caption"] + 3))
    lines.extend((line, "caption", TEXT_MUTED, LINE_H["caption"]) for line in body_lines)

    badge_h = pad + sum(advance for _, _, _, advance in lines) + pad
    badge = np.zeros((badge_h, width, 3), dtype=np.uint8)
    badge[:] = bgr((30, 25, 16))  # dark amber-tinted panel -- distinct from both the red no-face badge and the neutral-grey chrome panels
    cv2.rectangle(badge, (0, 0), (width - 1, badge_h - 1), bgr(HONEST), 1)

    global _text_queue
    saved_queue, _text_queue = _text_queue, []
    y = pad
    for text, font_key, color, advance in lines:
        qtext(pad, y, text, font_key, color)
        y += advance
    badge = flush_text(badge)
    _text_queue = saved_queue
    return badge


def _get_yaw_badge(yaw_deg):
    key = round(yaw_deg, 1) if yaw_deg is not None else None
    if _yaw_badge_cache["key"] != key:
        direction = _head_yaw_direction_label(yaw_deg)
        value_text = f"yaw {yaw_deg:+.1f}°   {direction}" if yaw_deg is not None else "yaw n/a"
        _yaw_badge_cache["canvas"] = _build_experimental_badge(EXPERIMENTAL_BADGE_W, HEAD_ORIENTATION_LABEL_TITLE, value_text, HEAD_ORIENTATION_LABEL_BODY)
        _yaw_badge_cache["key"] = key
    return _yaw_badge_cache["canvas"]


def _get_gaze_badge(gaze_label, gaze_reliable):
    key = (gaze_label, gaze_reliable)
    if _gaze_badge_cache["key"] != key:
        if gaze_label is None:
            value_text = "gaze: n/a (no face)"
        elif gaze_label == "UNKNOWN":
            value_text = "gaze: UNKNOWN (unreliable -- glasses/occlusion?)"
        else:
            value_text = f"gaze: {gaze_label}"
        _gaze_badge_cache["canvas"] = _build_experimental_badge(EXPERIMENTAL_BADGE_W, GAZE_LABEL_TITLE, value_text, GAZE_LABEL_BODY)
        _gaze_badge_cache["key"] = key
    return _gaze_badge_cache["canvas"]


def _get_blink_badge(rate_per_min, measuring, last_aperture, current_state, just_blinked):
    """last_aperture/current_state/just_blinked are BlinkDetector's own
    live diagnostic fields (snapshot()), surfaced here so a human can
    watch real aperture values and detector state while blinking --
    added specifically to debug/verify the blink-count-always-0 bug
    (see BLINK_APERTURE_PLAUSIBLE_MIN's comment in stage1_step4_vectors.py
    for the root cause and fix) instead of re-tuning blind."""
    key = (round(rate_per_min, 1) if rate_per_min is not None else None, measuring, round(last_aperture, 3) if last_aperture is not None else None, current_state, just_blinked)
    if _blink_badge_cache["key"] != key:
        value_text = "blink rate: measuring…" if measuring or rate_per_min is None else f"blink rate: {rate_per_min:.1f}/min"
        state_label = "BLINK!" if just_blinked else (current_state or "n/a")
        aperture_text = f"{last_aperture:.3f}" if last_aperture is not None else "n/a"
        diagnostic_text = f"diag: aperture={aperture_text}  state={state_label}"
        _blink_badge_cache["canvas"] = _build_experimental_badge(EXPERIMENTAL_BADGE_W, BLINK_LABEL_TITLE, value_text, BLINK_LABEL_BODY, diagnostic_text=diagnostic_text)
        _blink_badge_cache["key"] = key
    return _blink_badge_cache["canvas"]


def draw_experimental_strip(canvas, state):
    """Draws all three EXPERIMENTAL badges left-to-right, spanning
    EXPERIMENTAL_Y/EXPERIMENTAL_H. Runs as part of the regular cached
    chrome build (_build_chrome), NOT the special post-video-paste
    compositing path the old video-overlaid yaw badge needed -- these
    badges no longer sit on top of the live video pixels, so there's no
    ordering hazard to work around; the shared chrome cache (see
    _chrome_state_key) already covers them."""
    x = EXPERIMENTAL_BADGE_X0
    for badge in (
        _get_yaw_badge(state.get("head_yaw_deg")),
        _get_gaze_badge(state.get("gaze_label"), state.get("gaze_reliable")),
        _get_blink_badge(
            state.get("blink_rate_per_min"),
            state.get("blink_measuring"),
            state.get("blink_last_aperture"),
            state.get("blink_current_state"),
            state.get("blink_just_blinked"),
        ),
    ):
        bh, bw = badge.shape[:2]
        canvas[EXPERIMENTAL_Y:EXPERIMENTAL_Y + bh, x:x + bw] = badge
        x += EXPERIMENTAL_BADGE_W + GUTTER


def draw_demo_canvas(frame):
    with ui_lock:
        state = dict(latest_ui_state)
        state["z_scores"] = dict(state["z_scores"])
        state["va_point"] = dict(state["va_point"])

    key = _chrome_state_key(state)
    if _chrome_cache["key"] != key:
        _chrome_cache["canvas"] = _build_chrome(state)
        _chrome_cache["key"] = key

    canvas = _chrome_cache["canvas"].copy()
    # Display-only downscale to fit VIDEO_W/H (the ROW1_H budget) -- the
    # camera itself still captures at CAMERA_W/H (640x480) in
    # s1.capture_thread, untouched. Cheap (~1ms) and must run every
    # frame since it's tied to the live pixels, not cacheable.
    if frame.shape[1] != VIDEO_W or frame.shape[0] != VIDEO_H:
        frame = cv2.resize(frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
    canvas[VIDEO_Y:VIDEO_Y + VIDEO_H, VIDEO_X:VIDEO_X + VIDEO_W] = frame
    cv2.rectangle(canvas, (VIDEO_X, VIDEO_Y), (VIDEO_X + VIDEO_W, VIDEO_Y + VIDEO_H), bgr(BORDER_STRONG), 2)
    if not state["face_detected"]:  # read fresh every frame -- never lags behind the cache
        badge = _get_no_face_badge()
        bh, bw = badge.shape[:2]
        by, bx = VIDEO_Y + 12, VIDEO_X + 12
        canvas[by:by + bh, bx:bx + bw] = badge

    return canvas


def _launch_agent_call(summary, calibration_reference, session_id, person_label, use_stub):
    """Fires one Stage 2 call on its OWN short-lived daemon thread --
    never on s1.capture_thread (Thread 1) and never on
    stage3_processing_thread (Thread 2, the mandated two-thread
    architecture's processing side). This is a third, non-vision
    utility thread for a network call only -- the same class of
    exception SoakTracker's own watcher thread above already
    establishes; it does not touch capture or vision processing and
    does not change the mandated two-thread model.

    A live network call can take seconds; running it inline on T2 would
    freeze detection/z-scoring for that long even though capture (T1)
    stays unaffected. Offloading it here means the previous read simply
    stays on screen (latest_ui_state is untouched until this thread's
    result lands) -- never a blocking wait anywhere in the render or
    processing loop.

    Guards against overlapping calls: if a call is still in flight when
    the next 10s window flushes (only realistic if a live call runs
    long), that window's call is skipped rather than queued -- the next
    window tries again in ~10s regardless."""
    global _agent_call_in_flight

    with _agent_call_lock:
        if _agent_call_in_flight:
            print("[Stage3] previous agent call still in flight -- skipping this window's call")
            return
        _agent_call_in_flight = True

    with ui_lock:
        latest_ui_state["agent_pending"] = True

    def worker():
        global _agent_call_in_flight
        try:
            result = run_stage2_on_window(
                window_summary=summary,
                calibration_reference=calibration_reference,
                session_id=session_id,
                person_label=person_label,
                use_stub=use_stub,
            )
            text = result["response"]["text"]
            pointers = parse_agent_pointers(text)
            with ui_lock:
                latest_ui_state["agent_text"] = text
                latest_ui_state["agent_pointers"] = pointers
                latest_ui_state["agent_pending"] = False
                latest_ui_state["agent_last_updated_monotonic"] = time.perf_counter()
        except Exception as exc:  # keep the demo alive even if the agent step errors
            print(f"[Stage3] agent step failed: {exc}")
            with ui_lock:
                latest_ui_state["agent_pending"] = False
        finally:
            with _agent_call_lock:
                _agent_call_in_flight = False

    threading.Thread(target=worker, name="AgentCallThread", daemon=True).start()


def stage3_processing_thread():
    """Thread 2: all MediaPipe + vector-math calls live here, never on
    the capture thread. Landmarker setup mirrors s1.processing_thread's
    (same confidence thresholds, same VIDEO running mode, same
    output_face_blendshapes=False -- geometric-only, per Decision 14 /
    Mandatory Architecture #4) since that configuration is not vector
    math to reuse a function for, just options -- and s1 doesn't expose
    it as a callable, only inlined in its own processing_thread."""
    global latest_ui_state
    print("[Stage3 Processing] thread started.")

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=s1.FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=s1.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=s1.CONFIDENCE_THRESHOLD,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=s1.POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=s1.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=s1.CONFIDENCE_THRESHOLD,
        )
    )

    pd_buffer = deque()
    window_acc = episodes.WindowAccumulator()
    calibrator = x_core.NeutralCalibrator()
    stream_start = time.perf_counter()
    cycle_count = 0
    fps_window_start = time.perf_counter()
    samples_per_sec = 0.0
    last_window_low_confidence = False

    # --- real-data-only additions for the demo UI's new panels (see
    # latest_ui_state's docstring-comment) -- all state below is derived
    # from values this thread already computes; nothing here changes
    # vector math, calibration, windowing, or z-scoring. ---
    va_history = deque()
    last_va_history_append_t = None
    windows_processed = 0
    calibration_quality_ref = None
    last_window_detection_rate = None
    last_window_n_samples = None
    last_window_reasons = []

    # EXPERIMENTAL SIGNALS, PART 2 -- gaze L/R/C + blink rate. blink_detector
    # is the only piece of new PER-CYCLE state (a small isolated class from
    # stage1_step4_vectors.py); the rest are plain per-window tally
    # counters, reset every time the experimental log record below is
    # written (same 10s cadence as window_acc's own flush, reused as a
    # shared clock tick -- WindowAccumulator itself is never touched).
    blink_detector = attention.BlinkDetector()
    window_gaze_counts = {"LEFT": 0, "RIGHT": 0, "CENTER": 0, "UNKNOWN": 0}
    window_gaze_reliable_count = 0
    window_gaze_n_samples = 0
    window_blinks_confirmed = 0
    # BLINK DIAGNOSTIC: raw aperture per sample, batched in memory and
    # flushed once per 10s window (not written to disk every cycle -- see
    # EXPERIMENTAL_LOG_PATH's own file-cost reasoning) so the threshold
    # fix above can be checked against real recorded data, per this bug's
    # own "diagnose, don't blind-guess" instruction.
    window_aperture_samples = []

    print(f"[Stage3] calibration starting -- relax your face for {x_core.CALIBRATION_SECONDS:.0f}s...")

    while not s1.stop_event.is_set():
        with s1.frame_lock:
            frame = s1.latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        cycle_start = time.perf_counter()
        timestamp_ms = int((cycle_start - stream_start) * 1000)

        clahe_frame = geometry.apply_clahe(frame)
        rgb_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame.shape[:2]
        face_detected = False
        pose_detected = False
        window_composite = {"v_bf": None, "v_es": None, "v_pd": None}
        window_covariate = {"v_jc": None, "v_bf_convergence_ratio": None, "v_es_cheek_raise": None}
        window_yaw = None
        head_yaw_deg = None  # HORIZONTAL HEAD ORIENTATION (experimental) -- see below
        gaze_label = None    # EXPERIMENTAL gaze L/R/C -- None this cycle means "no face", distinct from "UNKNOWN" (face present, iris unreliable)
        gaze_reliable = None
        aperture_this_cycle = None  # EXPERIMENTAL blink detector input -- V_es's own aperture, read-only; None when no face this cycle

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

            # HORIZONTAL HEAD ORIENTATION (experimental, pilot, UNVALIDATED --
            # CLAUDE.md's "Attention / screen-orientation -- PILOT, not POC"
            # note). Calls attention.compute_v_so() unchanged -- reusing its own
            # output, not recomputing or re-deriving yaw a second way -- but
            # only the "yaw_deg" field of its returned components is ever
            # kept. compute_v_so's own return value (_v_so, the blended
            # score) and so_components["pitch_deg"]/["oriented"]/
            # ["pose_score"] are deliberately discarded here: those mix in
            # pitch (confirmed UNRELIABLE -- a maximal look-down reads as
            # ~0.1deg pitch, flat) and/or gaze, so displaying them would
            # imply a validated orientation reading this pipeline cannot
            # honestly make. yaw_deg alone is the one component CLAUDE.md's
            # own finding says is reliable.
            _v_so, so_components = attention.compute_v_so(normalized_pts, yaw, pitch)
            head_yaw_deg = so_components["yaw_deg"]

            # EXPERIMENTAL SIGNALS, PART 2 -- gaze L/R/C (isolated new
            # function, reuses the same landmarks/ratio approach
            # _gaze_centering_score already uses -- see its docstring)
            # and blink-detector input (V_es's OWN aperture value, read
            # straight off es_components -- no new landmark geometry).
            # Neither is composited into any affect vector.
            gaze_label, gaze_reliable, _gaze_raw_shift = attention.compute_gaze_direction(normalized_pts)
            aperture_this_cycle = es_components["aperture"]

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
            v_pd, _pd_components = x_core.compute_v_pd(pd_buffer, nose_pos, shoulder_mid, cycle_start)
            pose_detected = True
            window_composite["v_pd"] = v_pd

        # EXPERIMENTAL SIGNALS, PART 2 -- runs EVERY cycle, including with
        # aperture_this_cycle=None when no face was detected: BlinkDetector
        # needs that None to correctly ABORT an in-progress blink rather
        # than let an occlusion/no-face gap resolve into a phantom blink
        # (see its own docstring). Per-window tallies only count cycles
        # where a face WAS present (gaze_label is None otherwise, distinct
        # from "UNKNOWN" -- a real attempted-but-unreliable reading).
        if blink_detector.update(aperture_this_cycle, cycle_start):
            window_blinks_confirmed += 1
        live_blink_snapshot = blink_detector.snapshot(cycle_start)  # cheap (O(1)-ish); recomputed every cycle so the live badge is never stale
        # BLINK DIAGNOSTIC: batch this cycle's raw aperture (in-memory
        # list.append only -- no disk I/O here; the batch is written once
        # per window below). None is recorded too (not skipped), so a
        # no-face gap is visible in the data rather than silently absent.
        window_aperture_samples.append(
            {"t": round(cycle_start, 3), "aperture": round(aperture_this_cycle, 4) if aperture_this_cycle is not None else None}
        )
        if gaze_label is not None:
            window_gaze_counts[gaze_label] += 1
            window_gaze_n_samples += 1
            if gaze_reliable:
                window_gaze_reliable_count += 1

        # feed calibration BEFORE checking status, same ordering as s1
        calibrator.add_sample(cycle_start, window_composite, window_covariate, window_yaw)
        if calibrator.should_complete(cycle_start):
            reference = calibrator.complete(cycle_start)
            calibration_quality_ref = reference["quality"]  # real, frozen at calibration -- see classify_calibration_quality
            tag = "POSSIBLY NOT NEUTRAL" if reference["quality"]["possibly_not_neutral"] else "OK"
            print(f"[Stage3] calibration complete ({reference['calibration_seconds']:.1f}s) [{tag}]")

        deviation_composite = {"v_bf": None, "v_es": None, "v_pd": None}
        z_scores = {"v_bf": None, "v_es": None, "v_pd": None, "v_jc": None}
        va_point = {"valence": None, "arousal": None}

        if calibrator.is_calibrated():
            ref = calibrator.reference
            for key in ("v_bf", "v_es", "v_pd"):
                deviation_composite[key] = calibrator.deviation(key, window_composite.get(key))
            z_scores["v_bf"] = x_core._z_score(deviation_composite["v_bf"], ref["composite"]["v_bf"]["std"])
            z_scores["v_es"] = x_core._z_score(deviation_composite["v_es"], ref["composite"]["v_es"]["std"])
            z_scores["v_pd"] = x_core._z_score(deviation_composite["v_pd"], ref["composite"]["v_pd"]["std"])

            jc_mean = ref["covariates"]["v_jc"]["mean"]
            jc_std = ref["covariates"]["v_jc"]["std"]
            jc_dev = (window_covariate["v_jc"] - jc_mean) if (window_covariate["v_jc"] is not None and jc_mean is not None) else None
            z_scores["v_jc"] = x_core._z_score(jc_dev, jc_std)

            # Decision 18, live per-frame point -- canonical formula, see
            # stage1_step4_vectors.map_to_valence_arousal().
            va_point = map_to_valence_arousal(
                deviation_composite["v_bf"], deviation_composite["v_es"], deviation_composite["v_pd"], ref
            )

            window_acc.add_sample(cycle_start, face_detected, window_yaw, deviation_composite, window_covariate)

            # V/A HISTORY TRAIL -- real per-cycle va_point values, the SAME
            # value drawn as the live point on the plane, just also kept
            # for VA_HISTORY_SECONDS. Downsampled to one sample every
            # VA_HISTORY_SAMPLE_INTERVAL_SECONDS purely to bound how many
            # points the render layer has to draw -- not a recomputation,
            # not smoothing, not fabricated: every kept point is a real
            # va_point this thread actually produced this session.
            if va_point.get("valence") is not None and va_point.get("arousal") is not None:
                if last_va_history_append_t is None or (cycle_start - last_va_history_append_t) >= VA_HISTORY_SAMPLE_INTERVAL_SECONDS:
                    va_history.append(
                        {
                            "t": cycle_start,
                            "valence": va_point["valence"],
                            "arousal": va_point["arousal"],
                            "flagged": last_window_low_confidence,
                        }
                    )
                    last_va_history_append_t = cycle_start
            while va_history and (cycle_start - va_history[0]["t"]) > VA_HISTORY_SECONDS:
                va_history.popleft()

        if window_acc.should_flush(cycle_start):
            summary = window_acc.flush(cycle_start)
            last_window_low_confidence = summary["window_quality"]["low_confidence"]
            windows_processed += 1
            last_window_detection_rate = summary["detection_rate"]
            last_window_n_samples = summary["n_samples"]
            last_window_reasons = summary["window_quality"]["reasons"]
            tag = "LOW-CONFIDENCE" if last_window_low_confidence else "valid"
            print(f"[Stage3] window flushed [{tag}] n={summary['n_samples']} detect_rate={summary['detection_rate']:.2f}")

            # EXPERIMENTAL SIGNALS, PART 2 -- own window record, own
            # cadence tick (reusing window_acc's should_flush() boolean
            # as a shared clock, WindowAccumulator itself untouched), own
            # file. Reset the tallies right after writing so the next
            # window starts clean.
            blink_snap = live_blink_snapshot  # same `now` (cycle_start) as this cycle's own snapshot -- no need to recompute
            dominant_gaze = max(window_gaze_counts, key=window_gaze_counts.get) if window_gaze_n_samples > 0 else None
            _log_experimental_window(
                {
                    "schema_version": "1.0",
                    "record_type": "experimental_signals_window",
                    "unvalidated": True,
                    "session_id": SESSION_ID,
                    "person_label": PERSON_LABEL,
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "window_end_monotonic": cycle_start,
                    "gaze": {
                        "label_counts": dict(window_gaze_counts),
                        "n_samples": window_gaze_n_samples,
                        "reliable_rate": (window_gaze_reliable_count / window_gaze_n_samples) if window_gaze_n_samples else None,
                        "dominant_label": dominant_gaze,
                        "unvalidated": True,
                        "label": "approximate horizontal gaze (L/R/C) -- EXPERIMENTAL, NOT a validated attention/engagement measure, horizontal only",
                    },
                    "blink": {
                        "rate_per_min": blink_snap["rate_per_min"],
                        "measuring": blink_snap["measuring"],
                        "blinks_in_window": window_blinks_confirmed,
                        "blinks_in_rolling_rate_window": blink_snap["blinks_in_rate_window"],
                        "current_state": blink_snap["current_state"],
                        # DIAGNOSTIC: every raw aperture sample this window
                        # saw (own field, per this bug's own instruction --
                        # "log raw aperture per sample... so the threshold
                        # can be checked against real data"), plus the
                        # tuned threshold constants active when this record
                        # was written, so a later look at the log can
                        # directly see what open-vs-blink values looked
                        # like and what boundary they were judged against.
                        "raw_aperture_samples": window_aperture_samples,
                        "aperture_thresholds": {
                            "plausible_min": attention.BLINK_APERTURE_PLAUSIBLE_MIN,
                            "plausible_max": attention.BLINK_APERTURE_PLAUSIBLE_MAX,
                            "close_fraction": attention.BLINK_CLOSE_FRACTION,
                            "reopen_fraction": attention.BLINK_REOPEN_FRACTION,
                        },
                        "unvalidated": True,
                        "label": "approximate blink rate from eye-aperture threshold crossings -- EXPERIMENTAL, coarse heuristic, not a validated physiological measure",
                    },
                }
            )
            window_gaze_counts = {"LEFT": 0, "RIGHT": 0, "CENTER": 0, "UNKNOWN": 0}
            window_gaze_reliable_count = 0
            window_gaze_n_samples = 0
            window_blinks_confirmed = 0
            window_aperture_samples = []
            # use_stub is gated on the single AGENT_LIVE flag (default stub;
            # --live / STAGE3_LIVE opts in -- see the module banner). The
            # call itself runs off-thread (_launch_agent_call) so a slow
            # live response never stalls this detection loop; the panel
            # keeps showing the previous read until the new one lands.
            _launch_agent_call(
                summary=summary,
                calibration_reference=calibrator.reference,
                session_id=SESSION_ID,
                person_label=PERSON_LABEL,
                use_stub=not AGENT_LIVE,
            )

        with ui_lock:
            # agent_text/agent_pointers/agent_pending are carried forward from
            # whatever is already in latest_ui_state rather than tracked in a
            # local var: they're written asynchronously by _launch_agent_call's
            # worker thread (see its docstring), and reading them here inside
            # the same ui_lock the worker writes under is what keeps this
            # rebuild from ever clobbering a just-landed update with stale data.
            latest_ui_state = {
                "face_detected": face_detected,
                "pose_detected": pose_detected,
                "calibrated": calibrator.is_calibrated(),
                "calibration_seconds_remaining": calibrator.seconds_remaining(cycle_start),
                "z_scores": z_scores,
                "va_point": va_point,
                "window_flagged": last_window_low_confidence,
                "agent_text": latest_ui_state["agent_text"],
                "agent_pointers": latest_ui_state["agent_pointers"],
                "agent_pending": latest_ui_state["agent_pending"],
                "agent_last_updated_monotonic": latest_ui_state["agent_last_updated_monotonic"],
                "yaw_deg": window_yaw,
                "samples_per_sec": samples_per_sec,
                # snapshot, not a live reference -- va_history is never mutated
                # again after this point, so downstream readers (draw_va_plane)
                # don't need their own defensive copy.
                "va_history": list(va_history),
                "windows_processed": windows_processed,
                "calibration_quality": calibration_quality_ref,
                "last_window": {
                    "low_confidence": last_window_low_confidence,
                    "reasons": last_window_reasons,
                    "detection_rate": last_window_detection_rate,
                    "n_samples": last_window_n_samples,
                },
                "session_elapsed_seconds": cycle_start - stream_start,
                # HORIZONTAL HEAD ORIENTATION (experimental) -- raw yaw
                # degrees only, straight from compute_v_so's own output
                # (see above). None whenever no face is detected this
                # cycle, same convention as every other per-cycle value
                # here -- never carried forward/stale.
                "head_yaw_deg": head_yaw_deg,
                # EXPERIMENTAL SIGNALS, PART 2 -- live values for the demo's
                # experimental strip. gaze_label/reliable follow the same
                # None-this-cycle convention as head_yaw_deg above (never
                # carried forward/stale); blink rate/measuring come from
                # BlinkDetector.snapshot(), which is itself never stale
                # (recomputed every cycle) even though the underlying blink
                # COUNT only changes when a blink is actually confirmed.
                "gaze_label": gaze_label,
                "gaze_reliable": gaze_reliable,
                "blink_rate_per_min": live_blink_snapshot["rate_per_min"],
                "blink_measuring": live_blink_snapshot["measuring"],
                "blink_count_in_rate_window": live_blink_snapshot["blinks_in_rate_window"],
                "blink_last_aperture": live_blink_snapshot["last_aperture"],
                "blink_current_state": live_blink_snapshot["current_state"],
                "blink_just_blinked": live_blink_snapshot["just_blinked"],
            }

        cycle_count += 1
        elapsed = time.perf_counter() - fps_window_start
        if elapsed >= FPS_REPORT_INTERVAL_SECONDS:
            samples_per_sec = cycle_count / elapsed
            print(f"[Stage3 Processing] {samples_per_sec:.1f} samples/sec")
            cycle_count = 0
            fps_window_start = time.perf_counter()

    face_landmarker.close()
    pose_landmarker.close()
    print("[Stage3 Processing] thread stopped.")


def main():
    global PERSON_LABEL

    # Consent gate runs before any thread, camera, or model load -- same
    # ordering s1.main() uses, reusing the same camera-free consent module.
    consented, person_label = run_consent_gate(SESSION_ID)
    if not consented:
        return
    PERSON_LABEL = person_label

    t1 = threading.Thread(target=s1.capture_thread, name="CaptureThread")
    t2 = threading.Thread(target=stage3_processing_thread, name="Stage3ProcessingThread")
    t1.start()
    t2.start()
    window_shown = False  # gates the X-button check below -- see its comment

    # None (and every `if soak:` below a no-op) unless --soak / STAGE3_SOAK=1 --
    # a normal run never instantiates SoakTracker, never touches soak_log.jsonl.
    # Construction itself is guarded too: SoakTracker's own internals already
    # degrade to "no logging" with a loud banner on a bad path/permission
    # (see its __init__), but if something still goes wrong in a way that
    # isn't caught there, the demo must still run rather than crash outright
    # just because instrumentation failed.
    soak = None
    if SOAK_MODE:
        try:
            soak = SoakTracker()
        except Exception as exc:
            _loud_error(["SOAK LOGGING COULD NOT START", f"Unexpected error constructing SoakTracker: {exc}", "The demo will continue running WITHOUT soak logging."])

    print("Press 'q' (or Esc, or close the window) to quit.")
    try:
        while not s1.stop_event.is_set():
            with s1.frame_lock:
                frame = s1.latest_frame

            if frame is not None:
                canvas = draw_demo_canvas(frame.copy())
                cv2.imshow(WINDOW_TITLE, canvas)
                window_shown = True

            if soak:
                soak.maybe_report(t1, t2)  # no-op until SOAK_INTERVAL_SECONDS has elapsed; frame-watching is its own thread

            # ~30ms pace (matches the 30fps capture rate -- no benefit
            # rendering faster than the video source itself updates). An
            # uncapped waitKey(1) spins this loop hundreds of times/sec; on
            # the GIL, every one of those extra iterations (ui_lock
            # acquire + state dict copy) steals interpreter time from
            # stage3_processing_thread even when draw_demo_canvas's cached
            # path is cheap. Measured: waitKey(1) here measurably dropped
            # processing throughput versus this pacing. See _chrome_cache
            # docstring for the matching cache-side fix. waitKey is still
            # checked every single loop iteration (every ~30ms), so 'q' is
            # never missed -- the pacing controls how OFTEN we check, not
            # whether a keypress in that window gets caught.
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:  # 27 = Esc, a second quit key for the same clean path
                s1.stop_event.set()
                break

            # WINDOW X-BUTTON: waitKey only ever reports KEYBOARD input --
            # clicking the window's own close control delivers no key
            # event at all, so without this check the loop would keep
            # running forever with the window gone (exactly the "doesn't
            # close" symptom, just triggered by the X button instead of a
            # focus issue with 'q'). WND_PROP_VISIBLE drops below 1 the
            # frame the user clicks close. Gated on window_shown because
            # calling this before any window has ever been created (e.g.
            # the camera hasn't produced a first frame yet) raises/returns
            # garbage, not "closed".
            if window_shown:
                try:
                    closed = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
                except cv2.error:
                    closed = True  # the window handle itself is gone -- same as closed
                if closed:
                    s1.stop_event.set()
                    break
    except KeyboardInterrupt:
        # Ctrl-C: without this, the traceback still triggers `finally`
        # below (cleanup always runs), but re-raises afterward and prints
        # a scary stack trace for what is, here, a completely normal way
        # to quit. Catching it here gives the same clean path as 'q' /
        # the X button, just with its own message.
        print("\n[Stage3] Ctrl-C received.")
        s1.stop_event.set()
    finally:
        # Runs on EVERY exit path above -- the normal 'q'/Esc break, the
        # X-button break, KeyboardInterrupt, or any unexpected exception
        # -- not just the clean-quit path. stop_event.set() is idempotent
        # (harmless if already set) and is what lets the worker threads'
        # own loops end so the joins below don't hang forever waiting for
        # a thread that was never told to stop. This is the primary
        # mechanism for "best-effort summary even on Ctrl-C";
        # SoakTracker's atexit registration is a second, redundant
        # backstop behind it (see its docstring).
        print("[Stage3] shutting down...")
        s1.stop_event.set()
        cv2.destroyAllWindows()

        # Bounded joins, not indefinite ones: both worker loops check
        # s1.stop_event every cycle (capture_thread's own per-frame check,
        # stage3_processing_thread's per-cycle check), so under normal
        # operation both return within about one frame/detection cycle --
        # these timeouts are a safety net, not the expected path. Without
        # a bound, a genuinely stuck blocking call (e.g. a camera driver's
        # read() that never returns) would hang this join forever, which
        # from the user's side looks exactly like "the app doesn't
        # close": the window is already gone (destroyAllWindows just ran)
        # but the process itself never exits.
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        if soak:
            soak.stop()
            soak.write_summary(clean_exit=True)

        if t1.is_alive() or t2.is_alive():
            # Cleanup above has already run (windows closed, soak summary
            # flushed) -- this is a last-resort hard stop so a stuck
            # worker thread can never leave the process itself hanging.
            # os._exit skips further Python-level cleanup (there isn't
            # any left to skip at this point) and terminates immediately,
            # unlike sys.exit() which only raises SystemExit and would
            # itself block on the same stuck non-daemon thread at
            # interpreter shutdown.
            stuck = [name for name, alive in (("CaptureThread", t1.is_alive()), ("Stage3ProcessingThread", t2.is_alive())) if alive]
            print(f"[Stage3] warning: {', '.join(stuck)} did not stop within 3s (likely blocked in a camera/model call) -- forcing exit.")
            os._exit(1)

    print("Clean shutdown complete.")


def _cli_soak_report_path():
    """Returns the explicit path argument after --soak-report if one was
    given (`--soak-report <path>`), else None (use the default
    SOAK_LOG_PATH)."""
    if "--soak-report" not in sys.argv:
        return None
    idx = sys.argv.index("--soak-report")
    if len(sys.argv) > idx + 1 and not sys.argv[idx + 1].startswith("--"):
        return sys.argv[idx + 1]
    return None


if __name__ == "__main__":
    _print_soak_banner()
    _print_agent_mode_banner()
    if "--soak-report" in sys.argv:
        # Report-only mode: no camera, no consent, no threads -- just read
        # back whatever soak_log.jsonl already has and summarize it.
        print_soak_report(_cli_soak_report_path())
    else:
        main()
