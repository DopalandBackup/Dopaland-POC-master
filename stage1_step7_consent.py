"""
Stage 1.5, step 7 — consent + opt-out gate (Decision #6), extended for
per-person Gate 2 validation runs with an anonymous participant code.

Runs BEFORE any capture or calibration: this module never imports or
touches cv2.VideoCapture, mediapipe, or any camera/model resource. The
only side effect of importing this file is nothing happening until
run_consent_gate() is explicitly called, and even then, opting out
still opens no camera and starts no thread, and never prompts for a
participant code.

Honest framing (CLAUDE.md #10): this is an opt-in wellness/validation
tool. Never "monitoring", never clinical, never a claim of emotion
detection -- "behavioral signals / affective indicators" only.

No name or identity is collected. person_label is an anonymous,
operator-entered participant code (e.g. "P01") -- a join key for
associating this session's records with a validation-study
participant, not an identity field. Along with session_id (already
generated at import time by stage1_step4_vectors, not tied to any
person), the consent log is an audit trail that consent was obtained
for a given session and code, not a record of who gave it.

Operating model: one app launch = one person = one consent = one
calibration = one session_id = one person_label. No internal
multi-person loop lives here or anywhere else in this file.
"""

import json
import os
import time
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
CONSENT_LOG_PATH = os.path.join(LOG_DIR, "consent_log.jsonl")

MAX_PERSON_LABEL_LENGTH = 16
INVALID_PERSON_LABEL_CHARS = set("/\\'\"")

CONSENT_TEXT = """
============================================================
 Behavioral Signals Session -- Consent
============================================================

This is an opt-in wellness/validation tool. It is NOT a
monitoring or clinical system. It does not diagnose, predict,
or identify you.

What happens if you continue:
  - Your webcam turns on and tracks facial and postural
    GEOMETRY (positions of points on your face and body) --
    it does not save video or audio.
  - That geometry is used to compute a few behavioral signals
    (e.g. brow tension, eye expression, posture) for this
    session only.
  - Everything is stored LOCALLY on this machine, tagged with
    a random session ID -- no name, no identity, nothing sent
    anywhere.
  - Nothing persists across sessions. Each run starts fresh.

You can stop at any time by closing the window or pressing Q.

Do you consent to continue?
Type "yes" to proceed. Press Enter or type anything else to
exit now without recording anything.
"""


def _sanitize_person_label(raw):
    """Minimal sanity check only -- strip whitespace, reject empty, cap
    length, reject path separators/quotes. This is a free-form
    anonymous participant code (e.g. "P01"), NOT a validated identifier
    format -- do not over-validate beyond making it safe to log."""
    label = raw.strip()
    if not label:
        return None, "Participant code cannot be empty."
    if len(label) > MAX_PERSON_LABEL_LENGTH:
        return None, f"Participant code must be {MAX_PERSON_LABEL_LENGTH} characters or fewer."
    if any(ch in INVALID_PERSON_LABEL_CHARS for ch in label):
        return None, "Participant code cannot contain / \\ ' or \"."
    return label, None


def _prompt_person_label():
    """Only reached after consent is already confirmed -- re-prompting
    here never touches consent state or opt-out behavior."""
    while True:
        raw = input("Participant code (e.g. P01 -- anonymous, not a name): ")
        label, error = _sanitize_person_label(raw)
        if label is not None:
            return label
        print(f"  {error} Try again.")


def _log_consent(session_id, person_label):
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "event": "consent_given",
        "session_id": session_id,
        "person_label": person_label,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "ts_monotonic": time.perf_counter(),
    }
    with open(CONSENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_consent_gate(session_id):
    """Blocking, console-based.

    Returns (consented: bool, person_label: str | None).
      - Opt-out (blank / "no" / anything but "yes"): (False, None).
        No participant-code prompt, nothing logged.
      - Opt-in ("yes"): prompts for an anonymous participant code,
        logs {event, session_id, person_label, ts_utc, ts_monotonic},
        returns (True, person_label).
    """
    print(CONSENT_TEXT)
    response = input("> ").strip().lower()

    if response != "yes":
        print("\nNo consent given -- exiting. Nothing was recorded.\n")
        return False, None

    person_label = _prompt_person_label()
    _log_consent(session_id, person_label)
    print(f"\nConsent recorded for participant '{person_label}'. Starting session...\n")
    return True, person_label
