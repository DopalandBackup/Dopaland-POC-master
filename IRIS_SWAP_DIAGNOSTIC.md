# IRIS_LEFT_CENTER / IRIS_RIGHT_CENTER Mislabeling — Diagnostic Report

**Type:** READ-ONLY diagnostic. No code, data, or logs were changed. Nothing was re-scored.
**Source file investigated:** `stage1_step4_vectors.py`
**Trigger:** flagged during the attention-signal (V_so) build as a real, pre-existing constant mislabeling, not a new bug.

## Bottom line

**VERDICT: CONTAINED** — the swap only affects `cheek_raise`, which was already demoted to logged-only and excluded from Valence before this diagnostic. **V_es's scored formula (the eye-aperture/Duchenne term that passed Gate 2 at 71.4%) does not use these constants at all.** The V/A mapping and the Gate-2 V_es result stand. Only the *explanation* for `cheek_raise`'s "wrong-signed" finding changes — from an assumed biological/directional issue to a landmark-pairing bug — and that explanation should be corrected in the record even though `cheek_raise` itself needed no rescoring (it was never scored to begin with — Gate 2 excludes it by design, same as V_jc).

---

## 1. Confirm the swap

Definitions, `stage1_step4_vectors.py`:

```
144: IRIS_LEFT_CENTER = 468
145: IRIS_RIGHT_CENTER = 473
146: BROW_INNER_R = 55       # member of FACEMESH_RIGHT_EYEBROW
147: BROW_INNER_L = 285      # member of FACEMESH_LEFT_EYEBROW
149: EYE_R_OUTER, EYE_R_UPPER1, EYE_R_UPPER2, EYE_R_INNER, EYE_R_LOWER1, EYE_R_LOWER2 = 33, 160, 158, 133, 153, 144
150: EYE_L_OUTER, EYE_L_UPPER1, EYE_L_UPPER2, EYE_L_INNER, EYE_L_LOWER1, EYE_L_LOWER2 = 263, 385, 387, 362, 373, 380
153: LIP_CORNER_R = 61
154: LIP_CORNER_L = 291
```

**Direct empirical confirmation** (not assumed): ran the actual, unmodified `pose_normalize()` against real recorded footage and printed the pose-normalized (x, y, z) coordinate for every constant above, at 3 independent, widely-spaced frames. Excerpt (frame at t=0.00s; the other two frames show the identical pattern):

| Constant | idx | x |
|---|---|---|
| `EYE_R_OUTER`/`UPPER1`/`UPPER2`/`INNER`/`LOWER1`/`LOWER2` | 33,160,158,133,153,144 | **−66.3 to −28.4** (all cluster together) |
| `BROW_INNER_R` | 55 | **−22.6** |
| `LIP_CORNER_R` | 61 | **−27.9** |
| `IRIS_LEFT_CENTER` | 468 | **−49.7** ← lands in the R-group's range |
| `EYE_L_OUTER`/`UPPER1`/`UPPER2`/`INNER`/`LOWER1`/`LOWER2` | 263,385,387,362,373,380 | **+20.1 to +55.1** (all cluster together) |
| `BROW_INNER_L` | 285 | **+14.5** |
| `LIP_CORNER_L` | 291 | **+37.5** |
| `IRIS_RIGHT_CENTER` | 473 | **+39.6** ← lands in the L-group's range |

**Plainly stated: `IRIS_LEFT_CENTER` (468) is on the same physical side as every other `*_R` landmark (`EYE_R_*`, `BROW_INNER_R`, `LIP_CORNER_R`), and `IRIS_RIGHT_CENTER` (473) is on the same physical side as every other `*_L` landmark. The two iris constants are crossed relative to every other left/right pair in this file, which are all mutually self-consistent with each other.** This is not a fluke of one frame — the pattern holds identically across all 3 independently-sampled frames from real footage.

Secondary, non-verified-in-this-session corroboration: MediaPipe's own public FaceLandmarker documentation (recalled, not re-fetched in this session — flagged as such rather than presented as confirmed) associates iris index 468 with the **right** eye and 473 with the **left** eye, which — if accurate — means this file's naming has been backwards from the start, independent of and consistent with the empirical finding above. The empirical, in-session, real-coordinate evidence is the load-bearing proof; treat the MediaPipe-recollection point as corroborating color only.

---

## 2. Blast radius — every function that touches these constants

Exhaustive repo-wide search (`grep -rn "IRIS_LEFT_CENTER\|IRIS_RIGHT_CENTER"` across all `.py` files) found exactly **two files** that reference these constants: `stage1_step4_vectors.py` and `analyze_video.py`. Nowhere else in the codebase (not `gate2_score.py`, not `stage2_personality_agent.py`, not `stage3_demo_ui.py`) touches them.

| Function | Uses iris constants? | Affected by the swap? |
|---|---|---|
| `interocular_distance()` (`stage1_step4_vectors.py:217`) | Yes — `_dist(pts, IRIS_LEFT_CENTER, IRIS_RIGHT_CENTER)` | **NOT AFFECTED.** Euclidean distance between two points is symmetric — distance(A,B) == distance(B,A) regardless of which one is *called* "left" and which "right". This is a mathematical certainty, not just an empirical observation — confirms the report that flagged this bug was correct that it's "invisible" here. This value feeds face-width normalization for every vector (`io_dist` denominator) — unaffected. |
| `compute_v_bf()` — brow furrow (`stage1_step4_vectors.py:246`) | **No.** Uses `BROW_INNER_R`, `BROW_INNER_L`, `GLABELLA` only. | **NOT AFFECTED.** Confirmed by direct code read — no `IRIS_*` reference anywhere in the function body. Both the scored composite and the `convergence_ratio` covariate are clean. |
| `compute_v_es()` — eye crinkle (`stage1_step4_vectors.py:265`) | **Split — see below, this is the key question.** | **Composite/scored value: NOT AFFECTED. `cheek_raise` covariate: AFFECTED.** |
| — `aperture` (the SCORED, composited term, Decision-25-validated at Gate 2, 71.4%) | **No.** Computed entirely by `ear()` using only `EYE_R_OUTER/UPPER1/UPPER2/INNER/LOWER1/LOWER2` and `EYE_L_*` equivalents (lines 293-296). No `IRIS_*` symbol appears in `ear()` or in the `aperture` expression. | **NOT AFFECTED.** `composite = -aperture` (line 310) — the value actually returned, composited, and fed to Gate 2 scoring — is built exclusively from this iris-free calculation. |
| — `cheek_raise_avg` (logged-only covariate, already excluded from composite before this diagnostic) | **Yes.** `cheek_raise_side(IRIS_RIGHT_CENTER, LIP_CORNER_R, EYE_R_OUTER, EYE_R_INNER)` + `cheek_raise_side(IRIS_LEFT_CENTER, LIP_CORNER_L, EYE_L_OUTER, EYE_L_INNER)` (lines 304-307) | **DIRECTLY AFFECTED.** Given the confirmed swap, the first call is actually pairing the mislabeled-as-"right" iris (which is really the LEFT iris, per §1) with the RIGHT lip corner and RIGHT eye corners — a genuine cross-side, anatomically incoherent span. This is a real bug in `cheek_raise`'s geometry, not a biological finding. |
| `compute_v_jc()` — jaw compression, already logged-only (`stage1_step4_vectors.py:314`) | **No.** Uses `LIP_UPPER_INNER`, `LIP_LOWER_INNER`, `LIP_CORNER_R`, `LIP_CORNER_L`, `JAW_ANGLE_R`, `JAW_ANGLE_L`. | **NOT AFFECTED.** No `IRIS_*` reference. |
| `compute_v_pd()` — postural volatility | **No.** Uses `pose_world_landmarks` (BlazePose topology: `POSE_NOSE`, `POSE_SHOULDER_L/R`) — an entirely separate landmark model from FaceLandmarker's face mesh. No overlap is possible. | **NOT AFFECTED.** |
| `NeutralCalibrator`, `WindowAccumulator`, `classify_window_confidence`, `classify_calibration_quality`, `_z_score`, `map_to_valence_arousal` (all calibration/windowing/z-scoring/mapping) | **No.** None reference `IRIS_*`/`EYE_*`/`BROW_*`/`LIP_*` anywhere (confirmed via the same exhaustive grep — zero matches inside any of these). They only ever consume the already-computed scalar vector outputs (floats) that `compute_v_*()` functions return. | **NOT AFFECTED**, structurally — these operate one layer downstream of any raw landmark, so there is no path for the swap to reach them except by first passing through a compute_v_* function that itself uses the swapped constants (only `cheek_raise`, as established above). |
| New attention/gaze code — `_gaze_centering_score()`, `compute_v_so()` (`stage1_step4_vectors.py:364-467`) | **Yes, deliberately, with the swap already accounted for.** `r_ratio = eye_ratio(EYE_R_OUTER, EYE_R_INNER, IRIS_LEFT_CENTER)` and `l_ratio = eye_ratio(EYE_L_OUTER, EYE_L_INNER, IRIS_RIGHT_CENTER)` (lines 423-424) — the constant NAMES are used in the inverted pairing on purpose, with an explicit inline comment ("see KNOWN PRE-EXISTING CONSTANT MISLABEL above") and a docstring section documenting exactly this finding. | **NOT AFFECTED — confirmed isolated.** This is new, UNVALIDATED code (V_so, Step 1) built after this bug was found; it was written to already compensate for it. It does not "fix" the underlying constants (per its own docstring, deliberately out of scope), it just pairs by verified real geometry rather than by the misleading names, locally, only within this one function. |
| `analyze_video.py:641-642` (Mode B multi-face/scene-cut heuristic) | Yes — `(lms[IRIS_LEFT_CENTER].x + lms[IRIS_RIGHT_CENTER].x) / 2.0` (an average, for a rough on-screen face-position proxy, not a scored vector) | **NOT AFFECTED.** Same reasoning as `interocular_distance()`: the average of A and B equals the average of B and A regardless of which is labeled which — order/labeling-independent. |
| `stage1_step4_browdiag_session.py` | Uses `BROW_INNER_R`/`BROW_INNER_L` only, no iris. Also an explicitly-labeled throwaway diagnostic tool, not part of the scored pipeline. | **NOT AFFECTED, and not relevant to Gate 2 either way.** |

---

## 3. Gate-2 impact assessment

- `cheek_raise` was **already excluded from Valence and never scored at Gate 2** before this diagnostic (Decision 25: "cheek-raise DEMOTED to logged-only covariate — it was wrong-signed"). The swap found here gives a concrete, mechanical explanation for *why* it read wrong-signed — a genuine cross-side landmark pairing bug, not necessarily (or not only) a biological/directional failure of the geometric idea itself. That explanation is worth correcting in the record (CLAUDE.md's Decision 25 note currently attributes the wrong sign to something else), but it changes **no score**, because `cheek_raise` was never composited or scored to begin with.
- **V_es's scored formula — the piece that actually passed Gate 2 at 71.4% and feeds Valence — traces cleanly through `ear()`, which never references `IRIS_LEFT_CENTER` or `IRIS_RIGHT_CENTER` at all.** This was verified by direct, literal code inspection (§2 table), not inference.
- No other scored or composited vector (V_bf, V_pd) or the V/A mapping itself touches these constants anywhere in the codebase.

**Which of the task's two scenarios applies: the swap affects ONLY `cheek_raise` (already logged-only, excluded from Valence). It does not reach V_es's scored formula.**

---

## 4. Verdict

**CONTAINED — swap only affects already-demoted `cheek_raise`; validated results (V_es scored, V/A mapping) are NOT affected.**

Supporting facts, each independently verifiable by re-reading the cited line numbers or re-running the same read-only coordinate dump against any face:
1. The swap is empirically confirmed via direct, real-footage landmark coordinates (§1), not assumed.
2. `interocular_distance()` and the one usage in `analyze_video.py` are mathematically immune to the swap (symmetric combination of the two swapped values).
3. `compute_v_bf`, `compute_v_jc`, `compute_v_pd` never reference these constants at all.
4. `compute_v_es`'s scored `aperture`/`composite` path never references these constants — only the already-excluded `cheek_raise` covariate does.
5. Calibration, windowing, z-scoring, and the V/A mapping never touch raw landmark constants at all — only downstream scalar vector outputs.
6. The new V_so gaze code already accounts for the swap locally and does not depend on it being fixed.

No code, data, or score was changed to produce this report.
