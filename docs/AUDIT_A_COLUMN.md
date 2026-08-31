# Audit — "Already Implemented" Column

**Purpose:** substantiate or reclassify each of 11 previously-claimed capabilities, per the client's stated rule: *any item that cannot produce both a commit ID and an inspectable artefact is reclassified "not yet done" until verified.*

**Method:** read-only inspection of the repository as it exists on disk today. No source files were changed to produce this report. Every claim below is backed by a file:line citation and, where one exists, a real logged artefact — file path, line count, and a representative excerpt.

**Audit date:** 2026-08-22 (original write-up; see Finding 0 for why no commit-based date existed at that time). **Updated 2026-08-24** with real commit hashes once version control was established — see the "UPDATE — 2026-08-24" section below Finding 0, and the "SUMMARY UPDATE" at the end. Original findings are preserved unedited; updates are added, not substituted, so the document's own history stays honest about what changed and when.

---

## FINDING 0 — THE REPOSITORY IS NOT UNDER VERSION CONTROL

This has to be stated before anything else, because it invalidates the "commit hash" column for every single item below, regardless of code quality.

```
$ git status
fatal: not a git repository (or any of the parent directories): .git
```

There is no `.git` directory anywhere in this tree. **No commit has ever been made in this repository.** Consequently:

- **Zero of the 11 items below can produce a commit hash today.** Not "an old one" — none at all.
- By the client's own stated rule ("any item which cannot produce both a commit ID and an inspectable artefact is reclassified as not yet done until verified"), **all 11 items currently fail that bar**, independent of whatever the code actually does.
- I have **not** initialized git myself, and I'm not going to without you telling me to. Two reasons:
  1. It's your call how a paid, client-facing deliverable's provenance story begins, not mine to decide silently.
  2. G4 requires a media-blocking pre-commit hook and hardened `.gitignore` to exist **before** the first commit, specifically so raw capture data can never enter history (irreversible once it does). Neither exists yet. I checked — there is no raw video/audio/image data anywhere in the current tree (confirmed by an explicit search: zero `.mp4/.avi/.mov/.wav/.mp3/.jpg/.jpeg/.png/.bmp` files found), so a naive `git init && git add -A && git commit` would not currently commit raw media — but the safeguard that makes that true *going forward* isn't built yet, and building it is a decision, not a formality.
  3. A backdated-looking single "baseline" commit for work actually done across many past sessions would give every item the *same* commit hash and *today's* timestamp — which would look like fabricated provenance to a client doing exactly the kind of verification you're describing. That is arguably worse than no git history at all.

**What every "commit hash" cell below actually contains:** `N/A — no git history exists`. I am not going to write anything else there.

Where useful I've cited file modification timestamps (`mtime`) as a *weak, non-evidentiary* substitute — these are trivially alterable by any copy/checkout operation and prove nothing about authorship or history. I've marked them as such every time.

---

## UPDATE — 2026-08-24: version control now exists

Finding 0 above is preserved unedited as the historical record of what was
true when this audit was first written — that is the honest record, and
rewriting it to pretend the gap was never there would be exactly the kind of
laundering this document exists to prevent.

As of this update, git has been initialized in this repository, with a
hardened `.gitignore` and a media/secret-blocking pre-commit hook in place
**before** the first file was ever staged (see `PROVENANCE.md` and
`docs/PRIVACY_EVIDENCE.md` for the full disclosure and the verification that
the hook actually fires). Every item below now carries a real commit hash.

**What a commit hash below does and does not mean, stated plainly:**
- It means: this exact code, byte for byte, is inspectable by a third party
  at that commit, in this repository, today.
- It does **not** mean: the code was written on the date of that commit, or
  that the commit history reflects when the underlying development work
  happened. Every item's code predates this repository's git history — see
  `PROVENANCE.md`. The commit hash evidences **current code state**, not
  **authorship date**.
- It does **not**, by itself, upgrade a PARTIAL finding to VERIFIED. Items 7
  and 8 (blink, gaze) still lack a live-camera artefact generated *after*
  their respective fixes — a commit hash proves the fixed code exists and is
  pinned, not that it has been demonstrated working on a real camera. They
  remain PARTIAL below, unchanged.

All 11 items' current code lives in a single commit, `1854609` ("chore:
baseline import of existing pipeline source") — every source file in this
repository was added to version control simultaneously, in one baseline
import, so "the most recent commit touching" any of these files is the same
commit for all of them. That is itself worth stating plainly rather than
letting eleven identical-looking hashes imply eleven independent
verification events.

---

## THE 11 ITEMS

### 1. Webcam capture pipeline, real-time, two-thread (T1 capture-only, T2 all processing)

**VERIFIED**

- **Files/functions:** `stage1_step4_vectors.py:1373` `capture_thread()` (T1); `stage1_step4_vectors.py:1403` `processing_thread()` (T2). `stage3_demo_ui.py` reuses `s1.capture_thread` **unmodified** as its own T1, and defines `stage3_demo_ui.py:1684` `stage3_processing_thread()` as its T2.
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Artefact:** `logs/soak_log.jsonl` (84 lines) — every sample carries `capture_thread_alive` and `processing_thread_alive` as independent booleans, both `true` across all 82 samples of a 41-minute run (see Item 11).
- **Separation verified by direct grep**, not just by reading the docstring: every call to `cv2.VideoCapture`, `cap.read()`, `cap.set()`, `cap.release()` in `stage1_step4_vectors.py` occurs **only** inside `capture_thread()` (lines 1375–1399). Zero occurrences anywhere else in the file. T2 never touches the camera.

### 2. Face / head / visible-torso tracking (FaceLandmarker + PoseLandmarker)

**VERIFIED**

- **Files/functions:** `stage1_step4_vectors.py:1403–1436` (both landmarkers instantiated together in `processing_thread`); `stage3_demo_ui.py:1684+` (same pair, `stage3_processing_thread`).
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Artefact:** `logs/session_*.jsonl` (19 files, re-counted precisely on 2026-08-24 — an earlier pass of this audit said 17, which was imprecise) and `logs/gate2_trials.jsonl` — sample records carry `head_pose.yaw_deg/pitch_deg/roll_deg` (from FaceLandmarker's transformation matrix) and V_pd is computed from `pose_world_landmarks` shoulder + nose positions (`POSE_SHOULDER_L/R`, `POSE_NOSE` — `stage1_step4_vectors.py:211-213`), which is genuine torso tracking, not face-only.
- `output_face_blendshapes=False` confirmed at `stage1_step4_vectors.py:1415` — geometric-only, no trained-model shortcut, matching the stated thesis.

### 3. Sustained ≥15 FPS with full processing running

**VERIFIED**

- **Artefact:** `logs/soak_log.jsonl`, `soak_summary` record: `min_fps: 27.72`, `mean_fps: 29.35`, over 2486.9s. Every one of the 82 samples is therefore ≥27.7 FPS — comfortably and consistently above the 15 FPS bar, never dipping near it.
- This run is confirmed to be **full processing**, not bare capture: the `agent_is_stub` field only appears in `stage3_demo_ui.py`'s `SoakTracker` records, meaning this soak includes vector computation, calibration, windowing, V/A mapping, and stub-agent calls — not just the camera loop.
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.

### 4. Candidate facial signals computed geometrically from landmarks

**VERIFIED**

- **Files/functions:** `stage1_step4_vectors.py` — `compute_v_bf` (line 272), `compute_v_es` (317), `compute_v_jc` (366), `compute_v_pd` (396), plus the two experimental additions `compute_v_so` (483) and `compute_gaze_direction` (567) and `BlinkDetector` (668).
- All operate on `pose_normalize()`-corrected landmark coordinates (line 242) — 3D geometry, not blendshapes, not a trained classifier.
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Related finding, not one of the 11 but directly relevant:** CLAUDE.md and in-code comments (e.g. `stage1_step4_vectors.py:808`) repeatedly refer to "the existing face<80px/yaw>35deg per-frame gate" as already built. It is **not**. `record["quality"]["face_width_px"]` is computed and logged (line 1565) but is never compared against 80 anywhere in the file. No yaw>35° check exists in the per-frame path either — `pose_normalize()` and `interocular_distance()` are unconditional, they never reject a frame. The only per-frame gate actually enforced is MediaPipe's own confidence floor (`CONFIDENCE_THRESHOLD=0.7`, passed as `min_face_presence_confidence`/`min_tracking_confidence`). The face-size/yaw half of the documented per-frame gate is aspirational text, not code. This matters directly for Q5 below.

### 5. Per-person WITHIN-SESSION neutral calibration

**VERIFIED**

- **File/function:** `stage1_step4_vectors.py:929` `class NeutralCalibrator`. Fixed `CALIBRATION_SECONDS=25.0` (line 161), frozen `reference` after completion (line 1004 `complete()`), `deviation()` method (line 1024) reports every subsequent sample relative to that person's own baseline.
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Artefact:** sample records in `logs/session_*.jsonl` carry `calibration_status` and the frozen `calibration_neutral_ref` once complete.

### 6. Rolling-window statistics: avg, peak AND variance per vector

**VERIFIED**

- **File/function:** `stage1_step4_vectors.py:1035` `class WindowAccumulator`, `_stats()` at line 1101–1107:
  ```python
  return {"avg": float(arr.mean()), "peak": float(arr.max()), "variance": float(arr.var()), "n": len(vals)}
  ```
  All three (not two of three) are present for every vector, every 10s window (`WINDOW_SECONDS=10.0`, line 153).
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Artefact:** `window_summary` records inside `logs/session_*.jsonl`.

### 7. Blink detection (flagged experimental / unvalidated)

**PARTIAL**

- **File/function:** `stage1_step4_vectors.py:668` `class BlinkDetector`. Code exists, is labeled `EXPERIMENTAL / UNVALIDATED` throughout, and — as of this session — was substantially reworked against real logged aperture evidence (relative close/reopen thresholds re-tuned from 0.6/0.85 to 0.87/0.90, tolerance for tracking-loss gaps mid-blink added, a refractory debounce added).
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Artefact:** `logs/experimental_signals_log.jsonl` (128 records, spanning 2026-08-09 to 2026-08-15). This is a **real artefact from real camera sessions** — but it entirely **predates** this session's fix. Checked programmatically: **all 128 records report `blink.rate_per_min` as either `0.0` or `null`. Zero records show a nonzero blink rate.**
- **What's missing:** a persisted log entry, generated on a real camera after the fix, showing a nonzero blink count. I do not have camera access in this environment to produce one myself. The fix is reasoned and simulation-tested (synthetic aperture sequences matching the evidence you supplied reproduce exactly 1 confirmed blink per real dip, and 0 false positives sitting still) — but "I ran synthetic Python sequences" is not the artefact standard this audit is holding everything else to, and I'm not going to pretend it is. **Reclassify as not-yet-demonstrated until a live run produces a post-fix log entry.**
- **Now having a commit hash (`1854609`) does not change this.** The hash proves the fixed code is real and pinned; it does not supply the missing artefact. **Status remains PARTIAL.**

### 8. Coarse left/right/centre gaze orientation (flagged experimental / unvalidated)

**PARTIAL**

- **File/function:** `stage1_step4_vectors.py:567` `compute_gaze_direction()`. `GAZE_LABEL_SIGN` (line 564) was flipped from `+1` to `-1` this session after you reported the mirroring bug.
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Artefact:** the same `logs/experimental_signals_log.jsonl`. One representative window record shows `label_counts: {LEFT: 2, RIGHT: 5, CENTER: 334, UNKNOWN: 12}` — real variety across all four labels, including `UNKNOWN` firing (the glasses/occlusion path is demonstrably reachable on real data). This is genuine evidence the **mechanism** works on a real camera.
- **What's missing:** every one of those 128 records predates the sign fix, so they demonstrate the label mechanism working, but under the **mirrored** mapping. No log entry exists yet confirming the corrected (person's-own-left-reads-"LEFT") mapping on real camera data. Same caveat as Item 7 — verified in isolated synthetic tests only.
- **Now having a commit hash (`1854609`) does not change this.** Same reasoning as Item 7. **Status remains PARTIAL.**

### 9. Consent + opt-out step

**VERIFIED**

- **File/function:** `stage1_step7_consent.py:108` `run_consent_gate()`.
- Camera-free claim independently confirmed, not just taken from the docstring: grepped the file for `cv2`/`mediapipe` — the only match is the docstring's own sentence describing the guarantee; there is no actual `import cv2` or `import mediapipe` anywhere in the file.
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Artefact:** `logs/consent_log.jsonl`, 41 real records, e.g. `{"event": "consent_given", "session_id": "990d58ac-...", "ts_utc": "2026-07-04T14:52:40..."}`.

### 10. Personality/agent read WITH full request/response logging to disk

**VERIFIED**

- **Files/functions:** `stage2_personality_agent.py:299` `call_agent()`, `:310` `log_agent_exchange()`, `:335` `run_stage2_on_window()`.
- `log_agent_exchange` writes `prompt`, `response_text`, `model`, and `is_stub` for every single exchange (line 316–331).
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.
- **Artefact:** `logs/agent_log.jsonl`, **622 real records**. Checked the `is_stub` distribution directly: **593 stub, 29 live** (`is_stub: False`, real `claude-sonnet-5` calls). Both code paths have real logged evidence, not just the stub.

### 11. Stability soak with no memory growth, FPS decay or thread deadlock

**VERIFIED**

- **Artefact:** `logs/soak_log.jsonl`, `soak_summary` record (read directly, not from any doc):
  ```json
  {"total_runtime_seconds": 2486.9, "n_samples": 82, "start_fps": 27.72, "end_fps": 29.81,
   "min_fps": 27.72, "mean_fps": 29.35, "start_mem_mb": 295.5, "end_mem_mb": 313.4,
   "peak_mem_mb": 335.9, "any_thread_death": false, "any_exception": false, "clean_exit": true}
  ```
- 2486.9s = **41 minutes 27 seconds**. Memory rose from 295.5→313.4MB (peak 335.9MB) then plateaued — bounded, not a runaway leak, though 41 minutes is a modest window for a *definitive* no-leak claim on its own. FPS never dipped below 27.72. No thread death, no exception, clean exit.
- **This is one single soak run.** No second, longer, or repeated soak log exists anywhere in the repo.
- **Commit hash:** `1854609` ("chore: baseline import of existing pipeline source") — see the 2026-08-24 update above for what this hash does and does not evidence.

---

## THE FIVE QUESTIONS

### Q1. Is FPS logged continuously as a metric to a file, or only displayed/measured ad hoc?

**These are different claims, and the honest answer is the second one, by default.**

- During a **normal run** (no flags), FPS is: (a) computed in-memory and shown on the live UI overlay, and (b) printed to the console via `print(f"[Capture] sustained FPS: ...")` (`stage1_step4_vectors.py:1395`) and `print(f"[Stage3 Processing] {samples_per_sec:.1f} samples/sec")`. **Neither of these writes to a file.** Console output is not a persisted artefact.
- Continuous FPS-to-file logging **only** happens when the process is launched with the opt-in `--soak` flag (`SoakTracker`, 30s interval). This is not the default, and it has been exercised exactly once (Item 11).
- **Verdict: "FPS is logged continuously as a metric to a file" is currently false as a general/default claim.** It is true only for soak-mode runs. If this was claimed unconditionally, it needs to be narrowed to "logged continuously to a file, opt-in soak mode only."

### Q2. Does EVERY emitted signal carry a missingness flag AND a confidence value, or only some?

**No. None of the seven signals has both, as two genuinely separate fields.** (Full per-signal breakdown was pulled from the code directly, not summarized from memory.)

| Signal | Explicit missingness field | Separate confidence/reliability field |
|---|---|---|
| V_bf | No — only a shared `quality.face_detected` flag (not V_bf-specific) + implicit `None` | No per-sample field. `VECTOR_RELIABILITY["v_bf"]="high"` is a **static, hardcoded, module-level label**, not attached per-sample. |
| V_es | Same as V_bf | Same pattern — static `"low"` label only. |
| V_jc | Same as V_bf | Same pattern — static `"logged_only"` label only. |
| V_pd | Only a shared `quality.pose_detected` flag + implicit `None`/`buffer_len` | Static `"medium"` label only. |
| V_so (screen orientation) | No dedicated field | `gaze_reliable`/`head_pose_only` cover only the gaze *sub-component*, not V_so itself |
| gaze (`compute_gaze_direction`) | `"UNKNOWN"` label doubles as the missingness sentinel — not a distinct field | `reliable` boolean is the *same* underlying check as the `"UNKNOWN"` label, not independent of it |
| blink (`BlinkDetector`) | None — a raw `None` aperture is the only sentinel, no boolean/reason | None at all — `"measuring"` means "not enough time has elapsed," not "this reading is untrustworthy" |

**Signals with neither a real missingness field nor a real per-sample confidence field: V_bf, V_es, V_jc, V_pd, blink.** V_so and gaze have a partial, conflated version where one flag does double duty. This is a genuine gap against D0PA1's "missingness + confidence on every signal" requirement (WHAT D0PA1 ADDS) — none of that work has been built yet.

### Q3. Is there any MAD-based (median absolute deviation) robust baseline anywhere?

**No.** Confirmed two ways:
1. Exhaustive text search for `MAD`, `median`, `median_abs`, `1.4826` across every `.py` file in the repository — zero matches, anywhere, not just in the calibration file.
2. Direct read of the calibration class: `NeutralCalibrator._stats()` (`stage1_step4_vectors.py:996-1002`) —
   ```python
   arr = np.array(vals)
   return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}
   ```
   Population mean and std, exclusively. The class's own docstring says so too ("Keeps mean AND spread (std) per vector," line 951).

**Calibration uses mean/std. The MAD-based robust baseline (`1.4826 × MAD`) that D0PA1's "signal completions" work item calls for does not exist in this codebase in any form.**

### Q4. What is the longest stability soak for which a log actually exists?

**2486.9 seconds — 41 minutes, 27 seconds.** Read directly from `logs/soak_log.jsonl`'s `soak_summary` record (`total_runtime_seconds: 2486.9`), cross-checked against the file's own first (`soak_start`, `ts_utc: 2026-07-20T17:14:48`) and last (`soak_summary`, `ts_utc: 2026-07-20T17:56:15`) timestamps, which are 2487 seconds apart — consistent. There is exactly **one** soak log in the repository; no longer or repeated run exists. This matches what CLAUDE.md itself claims ("~41 min") — this particular number was not overstated.

### Q5. Is a coverage metric computed and stored anywhere?

**No field literally named "coverage" exists anywhere in the repository** (confirmed by exhaustive search). The closest analog is `detection_rate` (`n_detected / n_samples`), computed identically in two places — `WindowAccumulator.flush()` (`stage1_step4_vectors.py:1116`) and `AttentionWindowAccumulator.flush()` (line 1217) — and stored in `window_summary` and `attention_window_summary` records respectively.

**But it measures a narrower thing than "usable signal after the quality gate" as specified.** `detected` here means only "MediaPipe returned landmarks" — i.e., cleared MediaPipe's *own* internal 0.7 confidence floor. It does **not** reflect the face-size/yaw half of the quality gate CLAUDE.md describes, because — per the Item 4 finding above — that half was never actually implemented as a per-frame rejection. So `detection_rate` is coverage-*adjacent*, not coverage against the full documented gate, because half of that gate is aspirational text rather than code.

### Q6. Every hardcoded threshold, cutoff, or magic constant, with file and line

Exhaustive for `stage1_step4_vectors.py` (where essentially all measurement/detection logic lives). **Scope note, stated honestly:** this is not an exhaustive repo-wide sweep of every `.py` file — `stage3_demo_ui.py` (UI layout pixel constants, not measurement-relevant), `gate2_score.py` (frozen historical scoring constants, already under its own change-control regime per CLAUDE.md), and `stage2_personality_agent.py` (prompt-construction strings) were not itemized line-by-line. If you need those too, say so and I'll do a second pass.

**28 threshold/config constants:**

| Constant | Line | Value | Purpose |
|---|---|---|---|
| `CAMERA_INDEX` | 120 | 0 | camera device index |
| `FPS_REPORT_INTERVAL_SECONDS` | 121 | 3.0 | console FPS print interval |
| `CONFIDENCE_THRESHOLD` | 122 | 0.7 | "detected" = cleared 0.7 |
| `WINDOW_SECONDS` | 153 | 10.0 | rolling-window length |
| `CALIBRATION_SECONDS` | 161 | 25.0 | neutral-calibration duration |
| `CALIBRATION_DRIFT_EFFECT_SIZE` | 162 | 0.8 | Cohen's-d contamination flag threshold |
| `DETECT_RATE_FLOOR` | 168 | 0.5 | window-validity gate |
| `YAW_VARIANCE_CEILING_DEG2` | 169 | 100.0 | window-validity gate |
| `ATTENTION_YAW_THRESHOLD_DEG` | 178 | 20.0 | V_so pose-orientation cutoff |
| `ATTENTION_PITCH_THRESHOLD_DEG` | 179 | 20.0 | V_so pose-orientation cutoff |
| `ATTENTION_POSE_WEIGHT` | 180 | 0.7 | V_so pose/gaze blend weight |
| `ATTENTION_ORIENTED_SCORE_THRESHOLD` | 181 | 0.5 | V_so oriented/not boolean cut |
| `GAZE_PLAUSIBLE_SLACK` | 182 | 0.5 | iris-ratio corruption gate |
| `PD_BUFFER_SECONDS` | 215 | 1.5 | V_pd rolling buffer length |
| `GAZE_DIRECTION_DEVIATION_THRESHOLD` | 563 | 0.25 | CENTER vs L/R cut |
| `GAZE_LABEL_SIGN` | 564 | -1 | L/R word mapping (fixed this session, was +1) |
| `BLINK_ROLLING_MAX_SECONDS` | 626 | 3.0 | blink open-baseline window |
| `BLINK_CLOSE_FRACTION` | 647 | 0.87 | re-tuned this session (was 0.6) |
| `BLINK_REOPEN_FRACTION` | 648 | 0.90 | re-tuned this session (was 0.85) |
| `BLINK_MIN_DURATION_SECONDS` | 649 | 0.05 | jitter-vs-blink floor |
| `BLINK_MAX_DURATION_SECONDS` | 650 | 0.6 | sustained-closure/occlusion ceiling |
| `BLINK_REFRACTORY_SECONDS` | 651 | 0.15 | debounce, added this session |
| `BLINK_APERTURE_PLAUSIBLE_MIN` | 661 | 0.01 | degenerate-reading floor |
| `BLINK_APERTURE_PLAUSIBLE_MAX` | 662 | 0.6 | degenerate-reading ceiling |
| `BLINK_RATE_WINDOW_SECONDS` | 663 | 30.0 | rate rolling window |
| `BLINK_MIN_OBSERVATION_SECONDS` | 664 | 20.0 | "measuring..." floor |
| `BLINK_JUST_BLINKED_FLASH_SECONDS` | 665 | 0.3 | UI flash duration only |

**14 landmark-index constants** (MediaPipe topology IDs, not tunable thresholds, but hardcoded numeric literals nonetheless): `IRIS_LEFT_CENTER=473` (196), `IRIS_RIGHT_CENTER=468` (197), `BROW_INNER_R=55` (198), `BROW_INNER_L=285` (199), `GLABELLA=168` (200), `LIP_UPPER_INNER=13` (203), `LIP_LOWER_INNER=14` (204), `LIP_CORNER_R=61` (205), `LIP_CORNER_L=291` (206), `JAW_ANGLE_R=172` (207), `JAW_ANGLE_L=397` (208), `POSE_NOSE=0` (211), `POSE_SHOULDER_L=11` (212), `POSE_SHOULDER_R=12` (213).

**Total: 42 hardcoded numeric constants in this one file alone**, none of them in a hashed/versioned config today.

### Q7. Are capture and analysis separable — could analysis re-run from stored files with no camera attached?

**Partially yes, via two separate, purpose-built tools — not via one general mechanism.**

1. **`analyze_video.py`** re-runs the entire pipeline (landmark detection → vectors → optional calibration → V/A → agent) against a **stored video file**. Confirmed by grep: every `cv2.VideoCapture` call in this file (lines 183, 405, 570) takes a file path argument; the file never imports or calls `s1.capture_thread` or references `CAMERA_INDEX`. This is real, working separation for the "stored video" half of D0PA1 HARD CONSTRAINT #2.
2. **`gate2_score.py`** re-runs scoring purely from an **already-logged JSONL feature stream** (`gate2_trials.jsonl`) with zero camera/video/MediaPipe dependency at all — its only imports are `json`, `os`, `collections.defaultdict`.

**What doesn't exist:** a single, general "one command, reproduce this specific prior session's output from its archived inputs" tool, as D4 (Reproducibility) specifies. What exists today is two separately-built, task-specific scripts, not a unified reproduction path. The client's clean-machine reproduction test is achievable **for the two specific cases these tools cover**, not yet as a general capability across arbitrary logged sessions.

---

## SUMMARY (as originally written, 2026-08-22)

**By code-and-artefact standard alone** (ignoring the commit-hash requirement): **9 VERIFIED, 2 PARTIAL, 0 NOT FOUND.**

**By the client's own stated rule** (commit ID + artefact, or reclassify as not-yet-done): **0 of 11 items can currently produce a commit ID.** All 11 would have to be reclassified "not yet done" today, purely on that technicality, regardless of what the code does.

**The finding you'll least want to hear:** it isn't Items 7 or 8 (blink/gaze) — those are honest, bounded, fixable gaps with a clear next step (run it on a real camera once). It's Finding 0. Nine of these eleven claims are backed by real code and real logged data that I'd stand behind — but right now, none of them can be handed to a client as "verified" under the rule the client themselves set, because there is no commit history to point to. That's a one-time, fixable problem (git init, `.gitignore` hardening, a media-blocking pre-commit hook, then real commits going forward) — but it has to be a decision you make, not one I make for you mid-audit.

## SUMMARY UPDATE — 2026-08-24

**By code-and-artefact standard alone:** unchanged — **9 VERIFIED, 2 PARTIAL, 0 NOT FOUND.** Nothing about the underlying code or artefacts changed today; only version control was established.

**By the client's own stated rule** (commit ID + artefact, or reclassify as not-yet-done), re-evaluated now that a commit ID exists:

- **9 of 11 items now satisfy the client's rule in full** — a real commit hash (`1854609`) *and* a real inspectable artefact both exist for Items 1, 2, 3, 4, 5, 6, 9, 10, 11.
- **2 of 11 items (7, 8 — blink, gaze) still do not**, for a narrower, different reason than before: they now have a commit hash, but the only artefact that exists (`logs/experimental_signals_log.jsonl`) predates the fix and does not demonstrate the current code's behavior. A commit hash pointing at fixed code, next to an artefact showing the old broken code, does not satisfy "commit ID and inspectable artefact" for the claim being made — it satisfies it for a different, weaker claim ("this code exists"), not "this code works." **These two remain not-yet-verified under the client's rule**, exactly as before, just no longer for the git-history reason.

The finding that mattered most on 2026-08-22 (no version control at all) is resolved. The finding that will matter most going forward is unchanged: Items 7 and 8 need one live-camera session run after the fix, logged, before they can honestly move to VERIFIED.
