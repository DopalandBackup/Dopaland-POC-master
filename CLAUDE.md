# CLAUDE.md — Affective AI (POC CLOSED · D0PA1 VENDOR PROJECT ACTIVE)

You are the coding agent for a solo-built real-time Behavioral Intelligence system.
This file is the source of truth for what to build **now**. Read it every session.
If a request conflicts with this file, STOP and say so before writing code.

**Quality bar: this is meant to be world-class, not throwaway.** "World-class" here means
*measured right and instrumented well* — NOT "build more" and NOT "skip the sequencing."
The staged, one-step-per-commit, no-trained-model-in-POC discipline is HOW world-class ships:
it is the only way the Gate 2 accuracy claim is credible. Rigor goes into the geometry,
the pose-normalization, and the data schema — never into prematurely enlarging scope.

---

# ⚠️ READ FIRST — ACTIVE PHASE HAS CHANGED

**The webcam POC is COMPLETE and CLOSED. It is history. Do not build more POC.**

**The active work is now the D0PA1 VENDOR PROJECT** — a paid, client-governed,
pre-registered study built on this same repo. Its rules are in the section
**"D0PA1 VENDOR PROJECT — ACTIVE PHASE"** below. **Read that section before writing
any code.**

Everything between here and that section is the POC record: what was built, what was
measured, and the guardrails that produced a defensible result. It is still binding as
history and as engineering discipline — but "the POC is complete, do not build" refers
to POC *feature* scope. **D0PA1 infrastructure work is authorised and expected.**

If a task looks like it conflicts with "POC COMPLETE — do not build", check whether it
is D0PA1 work. If it is, it is in scope. If you cannot tell, STOP and ask.

**Naming collision — this WILL confuse you if you skip it:**

| Term | POC meaning (history) | D0PA1 meaning (active) |
|---|---|---|
| **Gate 2** | the ≥75% vector-direction test, n≈5, already run and scored | the client's **attention-validity** test (D8), not yet run |
| **Gate 3** | does not exist | `M_core` vs `M0b` decision rule |
| **calibration** | 25s within-session neutral | same, plus a cross-session persistent baseline (D7) |
| **pre-registration** | `GATE2_SCORING_RULE.md` (outside repo) | the client's v0.7 addendum sign-off matrix |

When either term appears in a task, establish which one is meant before acting.

---

## STATUS (update as stages complete)

- **Stage 0 — DONE.** Environment; two-thread skeleton; CLAHE + FaceLandmarker + PoseLandmarker in Thread 2. **Gate 1 PASSED:** ~30 FPS with processing (threads fully decoupled). Steady-state detection ~120ms/pass.
- **Stage 1 — DONE (n=1, developer's own face; generalization UNPROVEN until Gate 2).** Steps 4–6 shipped in `stage1_step4_vectors.py`: 4 vectors (pose-normalized geometry) → rolling 10s window (avg/peak/variance) → window-validity gate → per-person neutral calibration → V/A mapping + live plot. The V/A plot is deliberately labeled **"untested hypothesis"** (the vector→V/A mapping is the shakiest link — it is what Gate 2 tests). Final per-vector status is under ARCHITECTURE #4 below — that IS the core Stage-1 result, not a placeholder.
- **Stage 1.5 — DONE.** Consent + opt-out (architecturally camera-free: the consent module imports no `cv2`/`mediapipe`, so declining cannot touch a camera). Anonymous `person_label` stamped on all records. `stage1_step9_gate2_capture.py` = DUMB capture tool (no pass/fail anywhere), fixed block order smile→furrow→concentrate→sit-still→fidget, accept-criterion reminder on the review screen, clear participant instructions + get-ready beats.
- **⛔ GATE 2 — RUN AND SCORED. RESULT IS IN → see `GATE2_FINDINGS.md` (authoritative).** Pilot-scale directional result, **n≈5** (labels P01B–P06B, but P01B spans 2 session_ids ⇒ one person double-weighted; n<8 so per Decision 15 this is reported as pilot-scale with n stated openly, **NOT** a passed/failed Gate 2). Scored by `gate2_score.py` against the frozen rule, **B-series only** (originals retained but not scored — unclear furrow instruction).
  - **V_pd (arousal): 100% (5/5).** Best performer. **CAVEAT — state it honestly:** the neutral std is near-zero (~0.0001), so any movement yields enormous z. This proves *fidget vs sit-still is trivially separable*, NOT graded arousal sensing.
  - **V_es (pleasure): 71.4% (5/7); 60% excluding-flagged.** Predicted weak → is the best *facial* vector. Both failures were correct-signed but below |z|≥1.0. Baseline-sensitive (Decision 44).
  - **V_bf (pain): FAILED to generalize → DEMOTED to logged-only (Decision 17).** furrow 0/7, concentrate 71%. See ARCHITECTURE #4.
  - **The vector predicted strongest failed; the two predicted weak worked.** Gate 2 doing its job — n=1 could never have revealed this.
- **Stage 2 — DONE (stub built, then wired LIVE).** `stage2_personality_agent.py`: window summary → prompt → agent → parse → request/response pairs logged to `agent_log.jsonl` (stamped session_id + person_label + **`is_stub` flag**). Decision 18 mapping. STUB is the default (zero tokens); LIVE is opt-in via `use_stub=False`, key from `ANTHROPIC_API_KEY` env (never hardcoded). **Live path verified working on `claude-sonnet-5`** — one real call returns a warm, honest, pointer+detail read that correctly interprets only V_es/V_pd and states brow/jaw are logged-only.
- **Stage 3 — DONE.** `stage3_demo_ui.py`: two-column 1280×720 dashboard (live video + interpreted/logged-only vector bars + V/A plane with pain-side hatched "not measured" + agent read panel). Warm pointer+detail read, refreshes each 10s window in live mode. **Stability soak PASSED** (~41 min, `soak_log.jsonl`): FPS flat 27.7–29.9 (min 27.7, well above the ≥15 bar), memory bounded (295→313 MB, peak 336 — warm-up then plateau, no leak), no thread death, no exceptions. Live mode is opt-in (`--live` / `STAGE3_LIVE`), STUB default so normal/soak runs spend zero tokens.
- **VIDEO ANALYSIS FEATURE — DONE (`analyze_video.py`, Decision 19).** Offline video → same pipeline (imported, not reimplemented) → V/A timeline + warm agent report. **Mode A (calibrated):** video with ~25s neutral start → per-person calibration → validated timeline, labeled "CALIBRATED / validated". **Mode B (uncalibrated):** any video/movie with no neutral → population-default baseline, auto-selected, every output stamped `mode: B_uncalibrated`, `validated: false`, fields renamed (`valence_relative` not `valence_z_es`) so approximate numbers can't be mistaken for real z-scores. `--folder` batch (one bad file can't kill the run). Warm agent report reuses the live pointer+detail format; STUB default, `--live` opt-in (one call per video). Honest framing enforced in every path including Mode B and failure reports. Multi-face/scene-cut flagged, never tracked. (Known: MediaPipe can fire a false multi-face flag on ordinary solo footage if the background has face-like objects — handle multi-face safeguard separately, not yet done.)

**★ POC STATUS: COMPLETE.** All four pass criteria met — Gate 1 (FPS ~29), Gate 2 (per-vector result, honest, pilot-scale), stability (41-min soak clean), live demo on an unseen face (working, live agent). Plus the video feature extends it to recorded/uploaded video. **The build phase is done. Next work is pilot/pitch, NOT more POC building** (see OUT OF POC and the pilot roadmap in the master instruction).

- **ATTENTION SIGNAL, Step 1 — DONE (post-POC-complete addition, `stage1_step4_vectors.py`).** A NEW, FIFTH geometric signal, V_so (screen orientation): "is the head/gaze pointed at the screen" — a measurable geometric fact, explicitly NEVER labeled "attention"/"engagement" (mental-state inferences this system cannot measure). PRIMARY = head pose (yaw/pitch, reused unchanged from `yaw_pitch_roll_from_matrix`, glasses do not affect it). SECONDARY = gaze (iris-in-eye-socket position) — a BONUS refinement only (`ATTENTION_POSE_WEIGHT=0.7` keeps pose dominant), used only when geometrically plausible; falls back to head-pose-only (stamped as such) when not, so it degrades gracefully with glasses rather than depending on gaze. Needs NO per-person calibration (universal geometric threshold, like the yaw>35° quality gate) — logged from frame 1, on its own independent `AttentionWindowAccumulator` (own 10s clock, avg/peak/variance + `oriented_rate`), never touching `WindowAccumulator`/`NeutralCalibrator`. Own field (`record["screen_orientation"]`), own record type (`attention_window_summary`) — structurally cannot be composited into Valence/Arousal. Every threshold (`ATTENTION_YAW_THRESHOLD_DEG=20`, `ATTENTION_PITCH_THRESHOLD_DEG=20`, `ATTENTION_ORIENTED_SCORE_THRESHOLD=0.5`) is a first-cut estimate derived from the existing 35° gate, NOT tuned to one face — **UNVALIDATED, stamped as such on every logged record**, pending a Gate-2-style directed capture ("look at screen" vs "look away"/"look down" on command) across real people, scored by a human against `oriented_rate`, same discipline that caught V_bf's failure to generalize. Tested this session: unit tests (synthetic yaw/pitch/iris inputs) confirm correct oriented/not-oriented calls at threshold, correct gaze fallback, and JSON-serializability; a real end-to-end run of the actual `processing_thread()` (fed a recorded clip instead of a live webcam) confirmed no crash, FPS unaffected (27.8–29.2 samples/sec), and correct independent windowing.
  - **✅ FOUND AND FIXED: `IRIS_LEFT_CENTER`/`IRIS_RIGHT_CENTER` were mislabeled** (468/473 assigned backwards) — verified directly against real footage, confirmed CONTAINED (`IRIS_SWAP_DIAGNOSTIC.md`), then fixed: the constants now correctly refer to their named side, verified again against the same real footage post-fix (exact same coordinates now sit under the correct names). V_es's SCORED aperture/composite never used these constants (confirmed byte-identical before/after on 77 sampled real frames across two clips) — the 71.4% Gate-2 result was never affected and was not re-scored. Only `cheek_raise_side()` (already logged-only) used them; it now pairs correctly going forward, but **stays logged-only/excluded from Valence regardless** (Decision 25 — see its corrected note below) — fixing the pairing does not change that status, and no historical logs were rewritten. V_so's local gaze-pairing workaround was removed (now double-correcting would have re-broken it) — verified as a behavioral no-op (0.000000 max delta in V_so's score across both re-tested clips, identical `oriented_rate` per phase including the glasses phase).
  - **Also found (real-footage limitation, not a code bug):** a headless directed test recording (printed phase instructions, no live monitoring) reliably elicited a centered/oriented head position but did NOT reliably elicit large yaw/pitch during "look away"/"look down" phases (movement showed up mostly as roll, which V_so's design correctly ignores). The "not-oriented" case was still confirmed from real footage via a different, less-controlled clip's genuine large yaw excursions (-16° to -45°, correctly scored ~0). This is exactly why the task demands real cross-person Gate-2-style validation with an operator watching live, not a solo/scripted directed test — same lesson as V_bf.

---

## POC PHASE: COMPLETE AND CLOSED. Do NOT start new POC building.

**The POC is finished — all four pass criteria met (see STATUS).** The build discipline
below is retained as the record of HOW it was built and the guardrails that still apply to
any touch-up. **Do not scaffold pilot/deferred architecture.** If asked to build something
new, first check it against OUT OF POC and the pilot roadmap — most "next" ideas
(trained model, more emotions, pain axis, more data) are PILOT work, not POC.

**→ The active phase is D0PA1. See "D0PA1 VENDOR PROJECT — ACTIVE PHASE" below.**
D0PA1 adds provenance, controls, reliability and reproducibility infrastructure AROUND
the existing pipeline. It does not add new affect signals. Infrastructure work under
D0PA1 is authorised; new POC-style signal building is not.

**Forced order (was non-negotiable, and held): de-risk the pipeline FIRST, then paint the
demo layer.** That is done. The system has no name — do not invent or reference one.

---

## ★ D0PA1 VENDOR PROJECT — ACTIVE PHASE ★

A paid engagement with an external client (DOPALAND / Gargi). Governing documents,
in precedence order:

1. `D0PA1 POC Scope & Acceptance v0.5.1` — **FROZEN**. Scope, architecture, exclusions.
2. `D0PA1 Pre-Registration Clarifications v0.7` — the operational addendum (D1–D8,
   controls, decision rules, §19 sign-off matrix).
3. The completed, signed pre-registration — becomes the execution record.
4. Later substantive changes go through documented change control.

**What this project actually is:** a single-subject study asking which candidate
representations carry predictive information about that person's subsequent on-screen
action. It is NOT an emotion-detection demo. The client explicitly wants a *clean
answer*, positive or negative. **A DROP or INCONCLUSIVE result is a legitimate,
valuable outcome. An uninterpretable result is the only failure.**

### THE FIVE D0PA1 GUARDRAILS — apply to every task, without being asked

**G1. NEVER implement a pass/fail, score, threshold or verdict.**
This project is pre-registered. Thresholds (`δ_Gate3`, `δ_attention`, `δ_audio`,
`δ_latent`), the oriented utility `U`, and the RETAIN/DROP/INCONCLUSIVE rule are set by
a human in a document you will never see, and applied AFTER collection. Your code
computes and stores numbers. It never decides what they mean.
If you find yourself writing `if metric > X: return PASS` — stop, that is out of scope.
*(This is the same rule that made the POC's Gate 2 defensible. It now has a client
contract behind it.)*

**G2. NEVER tune anything against existing data.**
There is no held-out data. Adjusting a formula, constant or threshold so that logged
results look better is forbidden — this is Pitfall #5, and V_bf is the proof of what it
costs. Any change that would alter a previously reported number must be flagged loudly,
not absorbed.

**G3. Report gaps honestly; never upgrade partial to complete.**
The client will independently run the reproduction command and inspect commits. A claim
that overstates the code is worse than a missing feature. "PARTIAL — X exists, Y does
not" is always the correct answer when true.

**G4. Never commit raw or identifying data.** Git history is permanent. A repo that has
ever contained face or audio recordings is compromised for its whole life. `.gitignore`
first, media-blocking pre-commit hook, raw data outside the repo, manifest committed
instead.

**G5. Do not modify the validated path** — `ear()`, the affect vector formulas,
per-person calibration, the rolling window, the V/A mapping, the two-thread
architecture — unless a task explicitly says to. D0PA1 builds *around* the pipeline.

### D1 — FEATURE-BLOCK SEPARATION (the load-bearing requirement)

Five blocks. The constraint is **DIRECTIONAL**:

```
PERMITTED    X_core -> E_t          C_t -> E_t
FORBIDDEN    A_t -> X_core          A_t -> E_t
             U_t -> X_core          U_t -> E_t
```

- `X_core` core behavioural/affective features · `E_t` derived episode features ·
  `A_t` attention (ROI, orientation, dwell, persistence, switching, head-gaze
  coherence, and ANY downstream attention derivative) · `U_t` audio · `C_t` context.
- `E_t = h(X_core, ΔX_core, C_t)` by definition — E_t and X_core sharing information is
  the architecture, not a defect. Only attention- and audio-derived information flowing
  *into* the core is forbidden.
- **BOTH exclusions apply.** The audio one is easy to lose because attention dominates
  the discussion. Handle both explicitly, every time.
- **Why:** if attention reaches the core, `Δ_attention` compares a model against itself.
  If audio reaches the core, `Δ_audio` measures nothing. The comparison becomes invalid
  and the study is void.
- **WATCH `C_t`.** `C_t -> E_t` is permitted, so anything ROI-derived or gaze-derived
  hiding in context is a legal-looking back door for attention into the core. Flag it
  loudly if found.
- Separation must be **machine-checkable** (import-graph test + monkeypatch smoke test),
  not just documented.
- Existing V_so / gaze / blink code is **attention-class** — it belongs in `A_t` and must
  never reach `X_core` or `E_t`.

### WHAT D0PA1 ADDS (authorised work)

- **Gate 0 provenance:** experiment ID, config hash, pinned deps, variant log,
  canonical versioned log schema, data manifest.
- **Signal completions:** MAD-based robust baseline (`1.4826 × MAD`), missingness +
  confidence on *every* signal, continuous FPS logging as a metric, coverage metric.
- **Controls:** null-input, negative control (carried through every test), time-shuffle
  (diagnostic only, never a p-value), leakage harness (post-action / pre-action /
  timestamp-shift), positive blink control.
- **Baselines (D7):** three representations — raw, session-z, persistent-z.
- **Reliability (D3):** SEM, repeatability coefficient, Bland-Altman, within-unit CV.
- **Precision simulation (D6)** and synthetic latent recovery.
- **Reproducibility (D4):** one command, from archived inputs.
- **Privacy and retention**, implemented and verifiable.

### D0PA1 HARD CONSTRAINTS — do not attempt to engineer around these

1. **No ICC.** One subject = one unit = no between-unit variance = an uninterpretable
   ICC. Use absolute reliability measures (SEM / RC / Bland-Altman / CV). An ICC
   function may exist but must REFUSE to run on an invalid unit structure rather than
   silently returning a number.
2. **Reproduction runs from ARCHIVED INPUTS** — stored video and logged feature streams.
   Never from a live camera. Live capture is inherently non-reproducible (frame timing,
   exposure, auto-gain, dropped frames) and no seed touches that. Capture and analysis
   must be cleanly separated.
3. **Temporal rule, baseline AND scale.** `B_person,t = f(x_{1:t-1})` AND
   `S_person,t = g(x_{1:t-1})`. A prospective numerator over a retrospective denominator
   is not prospective. The scale carries identical leakage risk to the baseline and is
   the easier one to get wrong. Session 1 has no prior history ⇒ no persistent baseline
   ⇒ emit missing with reason, never fall back.
4. **Resample at episode / session / day level.** Never bootstrap frames or individual
   trials as if independent — these data are autocorrelated and frame-level resampling
   badly understates variance. Permutation must preserve temporal dependence and has its
   own exchangeability unit, which need not match the bootstrap unit.
5. **MAD == 0 must be handled explicitly** — emit missing with reason `zero_dispersion`.
   Do not divide by zero, do not add a silent epsilon. V_pd's neutral std is ~0.0001;
   pretending otherwise manufactures enormous z-scores out of nothing.
6. **Sensor swap must be two cameras recording simultaneously**, during collection.
   Sequential re-recording is not a substitute and the opportunity is permanently lost
   afterwards. If omitted, the omission is documented explicitly.
7. **`subject_id` on every record from the first observation**, even with one subject.
   `context_id` and `device_id` are required and must never be dropped.
8. **Missingness is logged, never dropped.** A missing value is a row with a reason, not
   an absent row.

### BLOCKED — do not build until the client answers

- **D2 prediction target**: action classes, horizon, tie and rapid-succession handling.
  Blocked on: *who builds the controlled software environment that produces the ROIs and
  the logged on-screen actions?* It does not exist in this repo.
- **A_t attention features / ROI features** — blocked on the same question.
- **Leakage controls on real action data** — harness only, synthetic data, pluggable
  source.
- **`U_t` audio module** — stub only. No microphone, no capture, no clock sync exists.
  Pending a keep-or-formally-remove decision.
- **Second-camera capture** — pending hardware and an FPS feasibility test.

If a task requires any of the above, STOP and say so. Do not invent a placeholder
definition to keep moving.

### HONEST-FRAMING WORDING (contractual, not stylistic)

- V_es / V_pd: **"implemented; pilot-scale candidate signals; not validated as
  psychological constructs."**
- V_bf / V_jc: logged-only. **There is NO validated pain axis.** Valence is
  single-source, pleasure-side only.
- Blink and gaze: experimental / unvalidated. The positive control validates *blink-count
  detection* only — detector validation is not construct validation.
- **LLM read is TERMINAL OUTPUT.** `Features → model → LLM interpretation → TERMINAL`.
  It may be logged, displayed and inspected. It must NEVER feed back into feature
  extraction, calibration, state estimation, prediction, target construction or model
  updating. It is a demonstration and interface, not a measurement and not a POC outcome.

---

## ENVIRONMENT (decided against reality — never re-suggest)

- Windows laptop, **native Python** — NO WSL2, NO Docker, NO local GPU.
- **Python 3.12** in a local `venv`. (3.11 unavailable; 3.13 has NO mediapipe wheels; 3.12 fully supported by mediapipe 0.10.35. Supersedes any earlier "3.11" note.)
- **OpenCV = `opencv-contrib-python` ONLY** (MediaPipe's own dep; superset, includes CLAHE). Never install `opencv-python` alongside — two `cv2` providers is non-deterministic.
- Installed: `mediapipe==0.10.35 opencv-contrib-python numpy`. Add packages only when a stage needs them; `pip freeze > requirements.txt` after each add.
- After any recommendation with a real tradeoff, give exactly 3 options: A (simpler/faster), B (more robust), My recommendation (tied to: solo dev, Windows laptop, flexible timeline, Indian market). Explain WHY, not just HOW. Flag pitfalls before they happen.

---

## VISION API (decided against reality — never re-suggest legacy)

- Use the **MediaPipe Tasks API**: `FaceLandmarker` + `PoseLandmarker`. The legacy `mp.solutions` API is deprecated and ABSENT in 0.10.35.
- Both landmarkers run in **VIDEO mode** (`detect_for_video`) — makes `min_tracking_confidence` meaningful and is cheaper. VIDEO-mode tracking proved highly robust to head motion (Decision 36) — good for demo stability.
- **FaceLandmarker** returns **478 landmarks** (468 mesh + 10 iris; iris bundled by default). **PoseLandmarker** returns **33 landmarks**.
- Model bundles (`.task`) live in `models/`, resolved via `os.path.dirname(__file__)`, out of version control (in `.gitignore`).
- **Head pose (yaw/pitch/roll):** from `output_facial_transformation_matrixes=True` (decompose rotation). Yaw decomposition validated sound (<1.5° at stillness; 6 Euler conventions agree — Decision 20). Feeds the quality gate's yaw > 35° check AND the pose-normalization below.
- **No per-frame confidence float exists in Tasks output** — encode the 0.7 threshold into `min_face_presence`/`min_tracking_confidence` inputs; "detected" = "passed 0.7" (Decision 14).
- **`output_face_blendshapes` STAYS FALSE.** Blendshapes come from an internal *trained* model; POC vectors must be geometric-only. Enabling them outsources V_bf/V_es to someone else's classifier and defeats the thesis.

---

## MEASUREMENT APPROACH (the world-class core of Stage 1)

- **Measure all facial vectors in POSE-NORMALIZED space, not raw 2D pixels.** Use the 3D landmarks + the facial transformation matrix to factor out head rotation and scale BEFORE computing any vector.
- **Why:** 2D distances (even after inter-ocular normalization) shift when the head rotates — a change from turning the head is then indistinguishable from a real expression change. That corruption is exactly what breaks Gate 2 across real people who won't hold their heads still.
- **Correct normalization order (Decision 13):** aspect-correct → centroid-center → Rᵀ → empirical head-turn validation. The `Rᵀ @ (point − translation)` form was **REJECTED** (mixed coordinate systems). Do not reintroduce it.
- **Face-width normalization still applies** (Pitfall #1) for scale, on top of pose-normalization for rotation. Reference: inter-ocular iris centers (most stable, barely move under expression).
- **V_pd uses `pose_world_landmarks`** (metric meters, hip-origin) — camera-distance-invariant by construction.
- **Stage-1 finding (keep):** facial vectors are **DIRECTION-reliable but MAGNITUDE-noisy under yaw** (io_dist denominator r≈−0.24 with yaw at full range; NOT rebuilt — direction survives, Decision 28). **Acceptance = reliable SIGN in the windowed + calibrated regime (SNR ~1.5–2), NOT yaw-correlation, NOT magnitude accuracy** (Decision 17/21). Per-frame understates SNR — measure windowed + deviation-from-neutral.
- Before implementing anything new: VERIFY against the live API what the landmark scale is and what the matrix encodes. State what was confirmed — do not assume.

---

## CADENCE (clarification — not new scope)

The "every 10s" rule governs the **summary → agent → DB** step ONLY. It is NOT the sampling rate.
- Thread 2 runs a **continuous sampling loop** computing the 4 vectors at detection speed (~20–30/s), buffering each reading.
- Every 10s it does the heavy summary: avg + peak + variance over the window → (later) agent → disk.
- Two clocks in one thread: fast sampling, slow summarizing. You cannot compute variance-over-10s from one sample at second 10, and live overlays require continuous sampling.

---

## TRAINING-GRADE DATA (highest-leverage, in-scope)

The POC IS the seed of the Track-B dataset. Every logged reading is a future training row, so the schema must be rich and clean. Get it wrong and you re-collect everything.
- Raw sample stream = **JSONL** (versioned, session-grouped, component-level, null-on-no-detect). SQLite retained for the later session/summary layer (Decision 15).
- Log per-sample: timestamp, all 4 raw vectors + uncomposited components, head-pose (yaw/pitch/roll), quality metadata (face size px, detect state), `session_id`, `person_label`, and (post-calibration) the frozen `calibration_neutral_ref`.
- Windowed summaries are a **separate versioned JSONL record type** (Decision 30/32).
- Propose any schema change before implementing.

---

## MANDATORY ARCHITECTURE — applies IN the POC

1. **Two-thread architecture (built, keep intact):** T1 capture-only ~30 FPS, single-slot buffer + lock (held only for the reference swap), stale frames discarded, no queue. T2 all processing, never touches the webcam. (See CADENCE for T2's two clocks.)
2. **CLAHE on every frame** before any detection.
3. **Quality gate:** skip frames where face < 80px, yaw > 35° (from transformation matrix), or confidence < 0.7. Plus a **window-validity gate** (Decision 33/35): binary flag if `detect_rate < 0.5` OR `yaw_var > 100 deg²` — thresholds first-principles-derived from the 35° gate, NOT fit to data. Low-confidence windows are **flagged, not suppressed silently**; the V/A point must render red/"UNSTABLE" for them (Decision 42/47).
4. **4 biometric vectors — GEOMETRIC, no trained model in the POC. FINAL Stage-1 status (reliability tags shipped as `VECTOR_RELIABILITY`):**
   - **V_bf — brow furrow (PAIN). ⛔ DEMOTED TO LOGGED-ONLY at Gate 2 (Decision 17). DID NOT GENERALIZE.** Formula `-drop_ratio` (inner-brow-to-glabella ÷ inter-ocular iris) is unchanged and NOT to be "fixed" — the issue is not a code bug. **Gate 2 (B-series, corrected instruction): furrow 0/7 pass — 6/7 wrong-signed at large magnitude (−1.3 to −6.3); concentrate 5/7 (71%) correct-signed.** Same formula, same expected sign, opposite directions by gesture ⇒ V_bf does not measure one consistent quantity across elicitations. Flipping the sign would fix furrow and break concentrate. Looked HIGH on n=1 (dev's own face, 3/3) — that did NOT hold on other faces (Pitfall #2 realized). **NOT composited, NOT used for Valence, NOT scored.** Do NOT tune it against the existing 6 faces (no held-out data — Pitfall #5); any future fix requires NEW faces.
   - **V_es — eye crinkle / Duchenne (PLEASURE). Reliability LOW — known Gate-2 risk.** Final formula = **eye-aperture only** (the Duchenne-specific part). **cheek-raise DEMOTED to logged-only covariate** — originally attributed to a directional/wrong-signed finding (Decision 25); root cause found post-POC and fixed: `IRIS_LEFT_CENTER`/`IRIS_RIGHT_CENTER` were swapped, so `cheek_raise_side()` paired each iris with the wrong-side eye/lip corners — not a biological failure of the idea itself (see `IRIS_SWAP_DIAGNOSTIC.md`). **No score changes**: cheek-raise was never scored at Gate 2 (only V_es's aperture term was) and stays logged-only/excluded from Valence regardless. Neutral drifts ~0.02 between captures (≈ a small expression); carried as a known risk, NOT fixed further on n=1 — that residual is what the trained model + pilot data are for (Decision 44/49). **May not pass Gate 2, and that is planned-for, not feared.**
   - **V_jc — jaw compression (STRESS). LOGGED-ONLY, NOT composited.** No reliable signal even at max effort (surface landmarks can't see muscle tension; beards occlude). Kept as a raw Track-B covariate. Confirms Pitfall #4 and is an ARGUMENT FOR THE PILOT (Decision 26).
   - **V_pd — postural volatility (AROUSAL). Reliability MEDIUM.** Variance of shoulder + nose over a ~1–2s buffer, using `pose_world_landmarks`. Temporal by nature — needs its own small buffer.
   - **V/A MAPPING — POST-GATE-2 (Decision 18, supersedes Decision 50):**
     `Valence = z_es` (**SINGLE-SOURCE — pleasure-side only. There is NO validated pain axis.** Must be labeled as such in UI and pitch.)
     `Arousal = z_pd`
     **The old `Valence = (z_es − z_bf)/2` is RETIRED — do not reintroduce it.** V_bf can no longer stand for pain (see #4). Composite vs covariate by construction (hardcoded key lists): **only V_es feeds Valence; only V_pd feeds Arousal. V_bf, V_jc and cheek-raise are logged-only**, excluded from composites (Decision 30/32 + 17).
5. **Rolling window (mandatory):** agent receives avg + peak + variance per vector over the full 10s window. NEVER a single snapshot. A 5s spike then neutral at s10 must still show in `peak`. Peak = plain max (valid because composites are sign-flipped so higher = more expression, Decision 32).
6. **Per-person within-session neutral calibration (BUILT — Decision 39):** 25s cold-start capture, **nothing reported during it** (Gap 2 L1 framing / Pitfall #3). Neutral mean + std per vector; report all vectors as deviation-from-neutral, z-scored against the person's own std. std kept as an unthresholded noise floor. Within-session only, no persistence, no FAISS. `calibration_complete` record + frozen `calibration_neutral_ref` on every post-calibration sample. Load-bearing for multi-person — Pitfall #2.
   - **Contamination flag is PER-VECTOR** (names the triggering vector, Decision 48): a V_bf flag = investigate; a V_es flag = expected. Structural blind spot: an expression held constant from frame 1 is invisible to any within-capture reference — closed PROCEDURALLY, not in software (Decision 45): operator instruction "relax completely, as if resting alone" + human watches raw values live.
7. **Consent + opt-out (BUILT — step 7).** One app launch = one person = one consent = one calibration = one `session_id` (see GATE 2 section). Camera-free opt-out. Anonymous `person_label` (P01…), never PII.
8. **Single personality agent — Claude API only.** Speak ONLY about the live reading, "just getting started" framing, NO invented history.
9. **Log every Claude request/response pair to disk.**
10. **Honest framing in UI:** "behavioral signals / affective indicators" — NEVER "emotion," NEVER clinical.

Capture at 640×480 (do not raise — Gate 1 headroom depends on it).

---

## POC GATE 2 — OPERATING MODEL & FROZEN SCORING (HISTORICAL — already run and scored)

> **NOTE — this is the POC's Gate 2 (vector-direction test, n≈5, DONE), NOT D0PA1's
> Gate 2 (attention validity / D8, not yet run).** See the naming-collision table at the
> top. The section is retained because its "dumb tool, frozen rule, human-applied
> scoring" discipline is exactly what D0PA1 guardrail **G1** now requires by contract.

Gate 2 = the **≥75%-correct-direction-across-≥8-people** test. It is the single result the Rs 45k customer and the investors decide on. The capture tool must be built to NOT fool itself.

1. **One app launch = one person** = one consent = one calibration = one `session_id`. **NO internal multi-person loop.** Each person is a clean process run — no cross-person state leakage. (Consistent with within-session-only, no persistence, no FAISS.)
2. **Anonymous participant code (P01…)** stamped on every record is the cross-record join key. It is NOT PII and NOT Face-ID (Face-ID stays OUT). Never persist as identity, never a name. (Shipped in step 7.)
3. **Fixed expression order** for every person: `neutral → smile → furrow → concentrate`. Do NOT randomize. (Position is confounded with fatigue/learning — concentrate always last; documented and accepted at n≈10.)
4. **Capture tool is DUMB; scoring is SEPARATE, FROZEN, HUMAN-APPLIED.** The tool stores, per trial: the **commanded** expression label (ground truth, recorded at command time), full windowed vector values (calibrated, deviation, z-scored), the calibration baseline used, per-vector contamination flags, window-confidence state, and calibration re-run count. It computes **NO pass/fail, NO auto-score, NO green checkmarks, NO dashboard.** Scoring runs AFTER all collection, against the frozen `GATE2_SCORING_RULE.md`, by a human (or a script written after collection that reads the frozen rule). **Never let the pipeline score itself** — it would inherit V_es's baseline drift and launder it into a "result."
   - Gate 2 bar is **PER-VECTOR** (V_bf %, V_es % separately; V_pd per the rule's §7; V_jc excluded). **Never one blended number** — a blend lets strong V_bf mask weak V_es, destroying the exact signal the pitch depends on.
   - **Flagged baselines: KEEP the person, MARK that vector low-confidence, report the score BOTH ways** (all-people AND excluding-flagged). The gap between the two is itself a finding. Re-run a person's calibration **max 2×** for obvious disturbances — never "re-run until green" (that selects for lucky-clean baselines).
   - **Mid-capture opt-out/delete belongs in THIS tool:** a person who consents then says stop → abort and discard that person's windows. (Not in step 7; build it here.)

**Do NOT build the capture tool until `GATE2_SCORING_RULE.md` is frozen.** The frozen rule defines the capture schema; building the tool first lets the schema silently become the rule (the contamination the whole design removes).

---

## OUT OF POC — do NOT build

- Trained vision model (EfficientNet-B2, AffectNet/RAF-DB). POC V/A mapping is rule-based — geometry only, no training in POC. **Prediction/forecasting is OUT** (trajectory, attention-drop, accuracy tracking). Reading the *current* state = in; guessing the *next* = out.
- Multi-user face recognition (FaceNet/DeepFace/FAISS) → single active person.
- Cross-session baseline / Gap-2 L2/L3 → POC is L1-style, within-session only.
- Prediction accuracy tracking, accurate/inaccurate feedback capture.
- Weekly summary reports. 4 separate agents (→1). Ollama/local Llama.
- LSTM, self-retraining, action prediction, intervention, voice/audio, diagnosis.

If a task seems to need any of the above, STOP and confirm — the scope line was likely crossed.

**⚠️ D0PA1 EXCEPTION — read this before applying the list above.**
The OUT OF POC list governs the *POC*. Three of its items are now IN SCOPE for D0PA1 and
must not be refused on the strength of this list:

- **Prediction / forecasting.** "Reading the current state = in, guessing the next = out"
  was the POC rule. D0PA1's entire confirmatory target IS a prediction (`Y_{t+1}`, the
  next on-screen action). Prediction is in scope — but only via the simple model families
  the client's scope permits, and only once D2 is answered.
- **Cross-session baseline.** POC was within-session only. D0PA1's D7 explicitly requires
  a persistent cross-session baseline representation, computed under the strict temporal
  rule. In scope.
- **Voice / audio.** Excluded from the POC, and currently BLOCKED in D0PA1 pending a
  keep-or-remove decision — but as a `U_t` *stub and manifest entry* only. Do not build
  audio capture or analysis.

Still OUT under D0PA1: trained vision model, face recognition/FAISS, LSTM,
self-retraining, intervention, clinical diagnosis, weekly reports, multiple agents,
local Llama. And **no new affect signals** — D0PA1 measures what exists, it does not
add to it.

---

## BUILD ORDER (gates are hard stops)

**Stage 0 — Foundation ✅ DONE (Gate 1 passed, ~30 FPS with processing).**

**Stage 1 — Signal extraction ✅ DONE (n=1; steps 4–6 shipped, generalization unproven until Gate 2)**
- ✅ step 4: 4 vectors geometrically (pose-normalized), live overlay, training-grade logging.
- ✅ step 5: rolling window — avg + peak + variance over 10s.
- ✅ step 6: vectors → Valence/Arousal, live plot. **Resolved: vectors DRIVE the V/A point** (`Valence=(z_es−z_bf)/2`, `Arousal=z_pd`) — the thesis-testing choice — plot labeled "untested hypothesis."
- (Note: per-person calibration, originally step 8, was built here — resequenced so V/A mapping consumes calibrated deviation and is built once, Decision 38/40.)

**Stage 1.5 — Multi-person enablement ✅ DONE**
- ✅ Consent + opt-out + anonymous participant code. ✅ Per-person 25s calibration → deviation (Decision 39).
- ✅ `GATE2_SCORING_RULE.md` frozen (lives OUTSIDE this repo — the coding agent must never see it). ✅ Step 9 dumb capture tool + accept-criterion reminder + clear instructions.
- ✅ **GATE 2 RUN AND SCORED** → `GATE2_FINDINGS.md`. Result: **V_pd 100% (with near-zero-std caveat), V_es 71.4%, V_bf failed→logged-only.** n≈5, pilot-scale, honestly reported (Decision 15). **De-risking phase COMPLETE.**

**Stage 2 — Personality ✅ DONE (stub built → wired live)**
- ✅ `stage2_personality_agent.py`: rolling-window summary → prompt → agent → parse → log every request/response pair to `agent_log.jsonl` (with `is_stub` flag). Live-only L1 framing, no invented history. Decision 18 mapping (Valence = z_es, Arousal = z_pd).
- ✅ Real Anthropic API wired and verified (`claude-sonnet-5`), key from env, STUB default / LIVE opt-in.

**Stage 3 — Demo-ready ✅ DONE**
- ✅ Two-column dashboard: live video + interpreted/logged-only vector bars + V/A plane (pain-side hatched "not measured") + warm pointer+detail agent read that refreshes each 10s window.
- ✅ **Stability soak PASSED** (~41 min): FPS flat ≥27.7, memory bounded (no leak), no deadlock/exceptions. STUB default, `--live` opt-in.

**Video analysis ✅ DONE (`analyze_video.py`, Decision 19)**
- ✅ Mode A (calibrated, neutral-start video → validated timeline) + Mode B (any video/movie, uncalibrated, stamped `validated:false`, auto-selected).
- ✅ `--folder` batch (one bad file can't kill the run) + warm agent report (same pointer+detail format, STUB default / `--live` opt-in).

**★ POC COMPLETE — all four pass criteria met. Build phase done; next is pilot/pitch.**
- ⚠️ Open (not blocking): multi-face "use primary/largest face" safeguard (MediaPipe can false-flag multi-face on solo footage) — handle separately if pursued.

---

## Attention / screen-orientation — PILOT, not POC

V_so (screen orientation, `stage1_step4_vectors.py`) and its study tool
(`orientation_capture.py`) are a **PILOT feature, NOT POC-ready.** Keep the
code — do not delete it — but it stays marked as such.

- **Finding:** directed testing confirmed YAW (left/right) is detected
  reliably. PITCH (looking up/down) is **UNRELIABLE** — on a maximal,
  sustained, verified chin-to-chest look-down, pitch stayed ~0.1° and
  `oriented_rate` stayed 1.0, indistinguishable from looking straight at
  the screen.
- **Root cause is STRUCTURAL, not a bug:** `yaw_pitch_roll_from_matrix` is
  provably exact on synthetic rotations; the face foreshortens when
  looking down, degrading the landmark data pitch depends on. **Do NOT
  attempt to "fix" pitch** in this approach — it is not fixable here.
- **Why this matters:** the most common disengagement cue is looking
  down, which this signal can't see. Shipping a yaw-only signal as
  "attention"/"engagement"/"focus"/"distraction" would **overclaim** —
  the same dishonesty the pain-axis (V_bf) rule forbids.
- **Reliable version = pilot work:** needs gaze tracking and/or better
  sensing hardware, plus real cross-person validation data — out of POC
  scope, same as everything else in OUT OF POC.
- **Update:** `stage3_demo_ui.py` now DOES read `compute_v_so()` — but
  ONLY its `yaw_deg` field, surfaced as a small, clearly-labeled
  "Horizontal head orientation — EXPERIMENTAL / UNVALIDATED" badge in the
  demo's own EXPERIMENTAL SIGNALS strip. Pitch, the blended
  `orientation_score`, and `oriented` are still never shown. Not
  composited into V/A; not called "attention" anywhere in the UI.

### Gaze direction (L/R/C) + blink rate — NEW, EXPERIMENTAL, not POC

Two more signals added alongside V_so's yaw badge, same discipline:
isolated new code (`compute_gaze_direction` + `BlinkDetector` in
`stage1_step4_vectors.py`), never composited into any affect vector,
shown only in the demo's amber "EXPERIMENTAL SIGNALS" strip (never
among the validated V_es/V_pd cards), and **UNVALIDATED across people**.

- **Gaze:** coarse **LEFT / RIGHT / CENTER / UNKNOWN** only — reuses
  `_gaze_centering_score`'s own landmarks/ratio approach and corruption
  gate, never a precise angle, **never up/down** (same pitch-unreliable
  reasoning as V_so). UNKNOWN (not a guessed direction) whenever iris
  tracking is implausible (glasses, occlusion).
- **Blink rate:** blinks/minute from a simple relative-threshold state
  machine over V_es's own `aperture` value (read-only — `ear()` itself is
  untouched). Reports "measuring…" instead of a number until enough
  observation time has passed; a no-face/occluded reading aborts an
  in-progress blink rather than counting a phantom one.
- **Logged** per-window (10s cadence, same tick as the affect windows)
  to `logs/experimental_signals_log.jsonl` — its own file, own record
  type, `"unvalidated": true` on every record — so a later human-applied
  cross-person check (commanded look direction vs. logged label; hand-
  counted blinks vs. logged rate) has real data to score against. Do NOT
  auto-score this file. Do NOT tune either signal's thresholds against
  the developer's own face.

---

## PITFALLS — apply without being asked

1. **Camera-distance drift:** normalize by face-width (inter-ocular) every frame. (Pose-normalization handles rotation; this handles scale.)
2. **Cross-person threshold failure (the real trap):** different resting geometry — one person's neutral brow = another's furrow. Fix: per-person neutral calibration, report as deviation. Without it, "works on everyone" is untestable. **Scoring corollary:** the pass/fail rule must be FROZEN before capture and applied by a human afterward — never auto-scored — or the same contamination re-enters through the scorer.
3. **LLM inventing history:** always L1 → agent speaks only about the live reading.
4. **Known failure cases:** beards occlude jaw (V_jc); glasses reflect into iris; atypical resting faces. Honest claim: "works across a range of faces with quick per-person calibration," never literally everyone.
5. **Self-tuning bias:** do NOT hand-tune thresholds to the developer's own face; validate against the 8–12 people. n=1 tuning feels like accuracy and is self-deception. **Applies to scoring too:** pre-register the Gate 2 rule before seeing the 8 people's data.

---

## CODING CONVENTIONS

- Small, reviewable commits per Build-Order step. Never build multiple stages in one shot.
- Threads genuinely decoupled (shared latest-frame buffer + lock; no cross-thread blocking).
- Capture 640×480 through the POC.
- API keys via environment / `.env` — never hardcoded, never committed.
- Clarity over cleverness; this runs unattended for a multi-hour soak.
