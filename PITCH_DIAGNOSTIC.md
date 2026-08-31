# Pitch-Blindness Diagnostic — compute_v_so and "look down"/"look up"

**Type:** READ-ONLY diagnostic. No code, thresholds, or data were changed.
**Symptom:** across 3 real capture sessions (`logs/orientation_trials.jsonl`), `look_down` reads `oriented_rate = 1.0` every time; `look_up` reads 0.86–1.0. Both should read LOW.

## Bottom line

**VERDICT: PITCH ANGLE UNRELIABLE.** The orientation formula treats pitch and yaw identically (not ignored, not unweighted), and the yaw↔pitch decomposition formula is mathematically exact (verified against synthetic ground-truth rotations). But the actual pitch VALUE reported by `yaw_pitch_roll_from_matrix` during real, instructed "look down" attempts stays small (mostly 0–5°) even though the same pitch channel demonstrably CAN register large values (15–18°) elsewhere in the same footage. This is a tracking-reliability problem specific to sustained downward head nods, not a formula bug and not a weighting/threshold oversight — there is no cheap one-line fix here.

---

## 1. How orientation is computed

`compute_v_so`, `stage1_step4_vectors.py:464-508`:

```python
yaw_frac = min(abs(yaw_deg) / ATTENTION_YAW_THRESHOLD_DEG, 1.0) if yaw_deg is not None else 1.0
pitch_frac = min(abs(pitch_deg) / ATTENTION_PITCH_THRESHOLD_DEG, 1.0) if pitch_deg is not None else 1.0
pose_score = max(0.0, 1.0 - max(yaw_frac, pitch_frac))
```

**Pitch IS used, and it is not ignored or down-weighted.** `pitch_frac` is computed by the exact same formula as `yaw_frac` (`|angle| / threshold`, capped at 1.0), against the exact same threshold magnitude:

```python
ATTENTION_YAW_THRESHOLD_DEG = 20.0
ATTENTION_PITCH_THRESHOLD_DEG = 20.0  # same first-cut magnitude as yaw
```

`pose_score` takes `max(yaw_frac, pitch_frac)` — **whichever axis is further off-center dominates**, so a large pitch alone is fully sufficient to drop the score to 0, exactly like a large yaw alone. There is no code path where pitch is dropped, capped below yaw's influence, or excluded from the boolean `oriented` call (`orientation_score >= ATTENTION_ORIENTED_SCORE_THRESHOLD`, line 508, which reads the same `orientation_score` yaw and pitch both feed into). Gaze (`_gaze_centering_score`) only ever nudges the score by ≤30% (`ATTENTION_POSE_WEIGHT=0.7`) and is unrelated to this symptom.

**Conclusion for part 1: pitch is used, symmetrically with yaw, at the same threshold. The orientation-logic formula itself is not the problem.**

---

## 2. Is pitch data available upstream?

`yaw_pitch_roll_from_matrix`, `stage1_step4_vectors.py:215-220`:

```python
def yaw_pitch_roll_from_matrix(matrix_4x4):
    r = np.asarray(matrix_4x4)[:3, :3]
    pitch = np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)))
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    roll = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    return float(yaw), float(pitch), float(roll)
```

`compute_v_so`'s own docstring confirms `yaw_deg`/`pitch_deg` are "NOT recomputed here -- passed in straight from yaw_pitch_roll_from_matrix(matrix)" — the SAME decomposition, computed once per cycle, that already fed the quality gate. So pitch is genuinely computed and passed through, not silently dropped anywhere in the call chain.

### 2a. Is the decomposition FORMULA correct?

Tested against synthetic rotation matrices with **known, exact** angles (ZYX = Rz(yaw)·Ry(pitch)·Rx(roll), the convention this formula's structure implies):

| Input (yaw, pitch, roll) | Extracted (yaw, pitch, roll) |
|---|---|
| (0, +40, 0) — pure 40° pitch-down | (+0.0, **+40.0**, +0.0) |
| (0, -40, 0) | (+0.0, **-40.0**, +0.0) |
| (+40, 0, 0) — pure yaw | (+40.0, -0.0, +0.0) |
| (+30, +40, 0) — combined | (+30.0, **+40.0**, +0.0) |
| (0, +40, +15) — pitch + roll | (+0.0, **+40.0**, +15.0) |

**The formula recovers pitch exactly, including when combined with yaw and roll.** This is not a coincidence — I hand-derived the same result algebraically before testing (the code's three lines are the textbook ZYX Tait-Bryan extraction: `pitch = asin(-R[2,0])` via the atan2/sqrt form, `yaw = atan2(R[1,0], R[0,0])`, `roll = atan2(R[2,1], R[2,2])`, each valid whenever `cos(pitch) > 0`, i.e., outside gimbal-lock at ±90°, which "look down" is nowhere near). **The arithmetic itself is not the bug**, assuming MediaPipe's canonical face-model matrix follows this same rotation-order convention — which is a reasonable assumption given yaw extraction from this identical matrix was already independently validated (CLAUDE.md Decision 20: "Yaw decomposition validated sound... 6 Euler conventions agree") and this task's own footage confirms left/right yaw tracking works correctly.

**CLAUDE.md never makes the equivalent claim for pitch.** Decision 20's validation is stated for yaw only; no analogous pitch validation exists anywhere in the project's decision history. Pitch was computed and logged from Stage 0 onward but, before V_so, was **never consumed by any calculation** (grep confirms: before this feature, `pitch` only ever went into `record["head_pose"]["pitch_deg"]` for storage and an overlay text string — never into `compute_v_bf`/`v_es`/`v_jc`/`v_pd`, and `classify_window_confidence` only ever takes `yaw_variance_deg2`). **V_so is the first thing in this codebase that ever acted on the pitch value — so its accuracy was never exercised or checked before now.**

### 2b. What does pitch actually do on real footage?

Scanned two available real clips frame-by-frame with the actual, unmodified `yaw_pitch_roll_from_matrix`:

**`test_clip.mp4`** (816 detected frames, contains large real yaw excursions): `|pitch|` ranged 0.1°–31.2° (mean 8.8°), `|yaw|` ranged 16.1°–78.4° (mean 36.3° — every frame has yaw>10°, so this clip has no "clean" low-yaw pitch read). At the most extreme yaw (-78.4°), pitch stayed small (-1.0 to -1.5°) — pitch does **not** blow up at extreme yaw, which argues against a gimbal-lock/axis-coupling artifact as the explanation.

**`directed_clip.mp4`** (803 detected frames, mostly low yaw): 802/803 frames have `|yaw|<10°` (a "clean" pitch read, uncontaminated by any yaw crosstalk). The 5 largest `|pitch|` values in this entire clip are **-17.6°, -17.3°, -15.9°, -14.6°, -14.0°**, all around t=4.0–4.2s — **with yaw only 7–9° at the same moments.** This proves the pitch channel is capable of registering a real, substantial, yaw-independent value when the head is actually tilted.

However, t≈4s in this clip falls in its opening "look at screen" phase (during the settle-in moment right after the get-ready countdown, not a deliberate "look down" instruction) — i.e., the one clear large-pitch reading in this footage is incidental, not from the labeled look-down segment. During that clip's own actual "look down" phase (recorded for an earlier, differently-structured test — see caveat below), pitch stayed consistently small (mostly 0–5° across the whole ~10s window, never exceeding ~4.4°).

**This directly answers the key question: pitch is COMPUTED, and the channel demonstrably CAN produce large (15–18°) values — but during the specific windows labeled as a deliberate, sustained "look down," the value it actually reports stays small.** That is a tracking-reliability gap for sustained downward nods specifically, not a "never computed" or "computed but ignored" situation.

*Caveat on this specific clip:* `directed_clip.mp4` was recorded for an earlier task with a 4-phase structure (look_at_screen / look_away / look_down / glasses+look_at_screen), not the current 6-segment `orientation_capture.py` sequence, so its single "look down" phase is the only same-clip ground truth available here, not a fresh confirmation from the actual capture tool's own footage (no video is saved by the real tool — see §3).

---

## 3. Cross-check against the real captured data

`logs/orientation_trials.jsonl` (18 records = 3 real sessions × 6 segments) — the `look_down`/`look_up` rows:

| session | label | screen_orientation.avg | peak | oriented_rate | n_detected/n_samples |
|---|---|---|---|---|---|
| ddbc2c0f | look_down | 0.775 | 0.878 | **1.00** | 311/311 |
| 1fd37ee4 | look_down | 0.813 | 0.953 | **1.00** | 299/300 |
| ad539d4a | look_down | 0.888 | 0.981 | **1.00** | 107/316 |
| ddbc2c0f | look_up | 0.767 | 0.949 | 0.93 | 118/314 |
| 1fd37ee4 | look_up | 0.741 | 0.948 | 0.86 | 123/397 |
| ad539d4a | look_up | 0.893 | 0.990 | 1.00 | 42/347 |

**The record schema does NOT store raw pitch anywhere** — confirmed by listing every field on a record: `schema_version, record_type, session_id, participant_code, commanded_label, segment_index, ts_utc, hold_seconds, n_samples, n_detected, detection_rate, screen_orientation, gaze_direction, look_away_rate, window_quality, unvalidated, label`. `window_quality` carries `yaw_variance_deg2` only (mirroring `classify_window_confidence`'s own yaw-only signature) — no pitch field of any kind, raw or aggregated. **So there is no way to confirm from the log itself whether the head was genuinely pitched down during these three sessions.**

Also notable: `screen_orientation.peak` is ≥0.878 in all three `look_down` sessions — meaning not even a single sample within any of the three ~10s windows dipped low. If a real, brief, strong downward nod had occurred at any point, `peak` (a plain max, per `AttentionWindowAccumulator`) would have caught it even if brief. It never does, across three independent sessions.

Separately, confirmed the capture tool's own live operator overlay (`orientation_capture.py:218`) shows `V_so`, `yaw`, and `gaze_reliable` to the person running the session — **it does not display pitch at all.** So even a careful, attentive operator watching the live values (Decision-45 pattern) had no way to notice pitch wasn't moving during a "look down" segment, in real time or afterward.

---

## 4. Verdict

**PITCH ANGLE UNRELIABLE (harder).**

Not "PITCH IGNORED BY ORIENTATION LOGIC" — §1 shows pitch is weighted identically to yaw in `compute_v_so`; there is no threshold or weight to loosen.
Not "PITCH NOT COMPUTED AT ALL" — §2 shows pitch is computed every cycle and the channel can and does produce large (15–18°) values in real footage.

The evidence instead points to the pitch VALUE itself being unreliable specifically for a sustained, deliberate downward head nod: the decomposition formula is mathematically exact (§2a, synthetic ground truth), yaw extraction from the identical matrix is independently validated and confirmed working in this same investigation, yet the "look down" segments — one from an older test clip and three from real capture-tool sessions — all show pitch failing to register a large, sustained value, while the SAME pitch channel clearly can register 15–18° elsewhere in the same footage.

**Residual uncertainty, stated plainly:** I cannot fully rule out that the specific humans/attempts behind these four "look down" windows simply didn't perform a large enough sustained tilt — CLAUDE.md already documents an almost identical prior finding from the V_so build itself ("a headless directed test recording... did NOT reliably elicit large yaw/pitch during 'look away'/'look down' phases... this is exactly why the task demands real cross-person Gate-2-style validation with an operator watching live, not a solo/scripted directed test"). No video is retained by the real capture sessions, pitch isn't logged, and it isn't shown to the operator — so there is no way to check, after the fact, whether the three real sessions' participants held a genuine large downward tilt for the full window or not. What tips this toward "unreliable" rather than "just execution" is that the *one* clean, real, substantial pitch reading found anywhere in the available footage (-14° to -18°) was **incidental** (settling into position), not produced during any of the four deliberate, sustained "look down" attempts examined — suggesting sustained deliberate nods specifically may be harder for this pipeline to register than brief natural head-bob motions, though this remains a hypothesis, not a proven mechanism.

## 5. What would need to change (not done — report only)

N/A in the "cheap fix" sense — the verdict is not "pitch ignored," so there is no single weight or threshold line to point at. If a human wants to pursue this further, the next diagnostic (not a fix) that would close the residual uncertainty in §4 is: a monitored, in-person directed capture (operator watching live, exactly as Decision-45 already prescribes for calibration) with an explicit, verified maximal chin-to-chest hold, cross-checked frame-by-frame against the printed pitch value in real time — plus, since neither the log nor the operator overlay currently exposes raw pitch at all, that observability gap would need addressing before such a test could even be run.
