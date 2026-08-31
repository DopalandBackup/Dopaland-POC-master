# PROJECT STATUS REPORT — Affective AI POC

**Generated:** 2026-07-27 (read-only verification pass — no code, logs, or data were modified)
**Scope:** verify CLAUDE.md's "POC COMPLETE" claim against the actual code and evidence, not just repeat it.

## Bottom line

**CLAUDE.md's "POC STATUS: COMPLETE" claim is SUBSTANTIALLY VERIFIED.** All seven core files exist, import cleanly, and are wired consistently with Decision 18. All three re-runnable pass-criteria (Gate 1 FPS, Gate 2 scoring, stability soak) reproduce the exact numbers CLAUDE.md quotes, from raw data, in this session. The fourth (a live agent call on an unseen face) could not be re-verified in this session — no `ANTHROPIC_API_KEY` is present — so it rests on CLAUDE.md's own prior claim, not fresh evidence gathered here. One genuine, unresolved documentation inconsistency was found (`GATE2_FINDINGS.md` is called "authoritative" but does not exist — see Part F). No code was changed to investigate any of this.

---

## A) Component Inventory

All seven core files exist and import cleanly (`python -c "import <module>"`, exit code 0, no output):

| File | Exists | Imports cleanly | Role |
|---|---|---|---|
| `stage1_step4_vectors.py` (1186 lines) | ✅ | ✅ | Vectors (V_bf/V_es/V_jc/V_pd), pose-normalization, `NeutralCalibrator`, `WindowAccumulator`, `map_to_valence_arousal` (the one canonical V/A mapping) |
| `stage1_step9_gate2_capture.py` (614 lines) | ✅ | ✅ | Gate 2 dumb capture tool (no pass/fail, fixed block order) |
| `gate2_score.py` (307 lines) | ✅ | ✅ | Frozen-rule Gate 2 scorer, read-only over `gate2_trials.jsonl` |
| `stage2_personality_agent.py` (387 lines) | ✅ | ✅ | Personality agent: prompt build, `call_agent` (stub/live swap point), `parse_agent_pointers`, `log_agent_exchange` |
| `stage3_demo_ui.py` (1465 lines) | ✅ | ✅ | Live two-thread demo dashboard, `--live` opt-in agent read |
| `analyze_video.py` (1267 lines) | ✅ | ✅ | Offline video analysis: Mode A/B, `--folder` batch, warm agent report |
| `test_live_agent.py` (105 lines) | ✅ | ✅ | Standalone single-live-call smoke test |

No missing or broken files. No `TODO`/`FIXME`/`XXX` markers found anywhere in the `.py` files (`grep -rn "TODO\|FIXME\|XXX" *.py` → no matches).

---

## B) Wiring / Consistency Checks

### B1 — V/A mapping is Decision 18 everywhere: PASS
- Exactly **one** definition of `map_to_valence_arousal` exists, in `stage1_step4_vectors.py:705`. Its body: `valence = tanh(z_es/2)`, `arousal = tanh(z_pd/2)` — no other vector feeds either.
- `stage3_demo_ui.py` imports it (`from stage1_step4_vectors import map_to_valence_arousal`, line 60) and calls it directly (line 1310) — not reimplemented.
- `analyze_video.py` imports the module (`import stage1_step4_vectors as s1`) and calls `s1.map_to_valence_arousal(...)` directly (line 269) for both Mode A and Mode B — not reimplemented.
- The retired `(z_es - z_bf)/2` formula (Decision 50) appears **only** in comments/docstrings explicitly marking it retired (`stage1_step4_vectors.py:63,723`, `stage2_personality_agent.py:18`) — zero occurrences in any executable expression (`grep` for the pattern in live code paths returns only these comment lines).

### B2 — V_bf and V_jc are logged-only everywhere: PASS
- `map_to_valence_arousal`'s return value only ever sets `valence`/`arousal` from `z_es`/`z_pd`; `z_bf` is placed only in the `components` sub-dict, explicitly documented as "logged-only, plays no part in the valence/arousal values."
- `WindowAccumulator` and `stage2_personality_agent.zscore_window` track V_bf's own deviation/z-score for display/logging purposes only — by hardcoded key list, never folded into a composite.
- `analyze_video.py`'s Mode A/B timeline entries carry `v_bf_z_logged_only` / `v_bf_relative_logged_only` as clearly-named, separate fields never used in `valence`/`arousal` aggregation.

### B3 — Honest framing intact on every output surface: PASS
- `grep -i "emotion\|clinical\|diagnos"` across all `.py` files: every hit is either (a) forbidding-rule text ("never say emotion", "never clinical language"), (b) an unrelated word ("diagnostic tool" in throwaway dev scripts), or (c) `stage1_step7_consent.py`'s own disclaimer ("never a claim of emotion... does not diagnose"). Zero instances where these words are used as an actual claim about a reading.
- Pain axis labeled "not measured" in the live V/A plane (`stage3_demo_ui.py:934`, `"No validated pain axis — not measured"`) and in the standalone overlay (`stage1_step4_vectors.py:1123`).
- Mode B (`analyze_video.py`) stamps `"validated": False` at the top level, on every timeline entry (verified present 47 times across the file, including in the two failure-path builders), and in the console banner/agent-report CONFIDENCE pointer.

### B4 — Two-thread architecture intact: PASS
- `stage3_demo_ui.py`: `t1 = threading.Thread(target=s1.capture_thread, ...)` (reused, capture-only), `t2 = threading.Thread(target=stage3_processing_thread, ...)` (all processing/UI-state/logging) — this pairing is unchanged from the original mandate.
- Two additional utility threads exist (`SoakFPSWatcher`, `AgentCallThread`) — neither touches `cv2.VideoCapture` or the webcam; both are documented as deliberate, narrowly-scoped exceptions (network/instrumentation only), not a change to the mandated T1/T2 vision split.

### B5 — API key read only from the environment: PASS
- No hardcoded key literal anywhere (`grep` for `sk-ant-[...]` pattern: no matches). No `api_key="..."` literal assignment anywhere.
- The only places `ANTHROPIC_API_KEY` and `sk-ant-...` appear as text are user-facing instructional strings telling a human what command to type (`test_live_agent.py`, `analyze_video.py`'s preflight warning) — never the actual key value being printed.
- `call_real_agent()` reads via bare `anthropic.Anthropic()` (SDK's own env resolution) — confirmed unchanged from prior verification.

### B6 — STUB is default everywhere; LIVE is opt-in: PASS
- No remaining hardcoded `use_stub=True` at any call site (`grep` confirms 0 matches in `stage3_demo_ui.py`/`analyze_video.py`) — both now pass `use_stub=not AGENT_LIVE`, and `AGENT_LIVE` defaults `False` in both files (`--live` flag or `STAGE3_LIVE` env var to opt in, same variable name shared by both tools).
- `soak_log.jsonl`'s 82 `soak_sample` records all show `"agent_is_stub": true` — the recorded 41-minute soak run spent zero tokens, confirming the default-stub guarantee held during the actual passed run, not just in theory.

### B7 — Video Mode A/B reuse the same pipeline: PASS
- Both `_run_mode_a` and `_run_mode_b` in `analyze_video.py` call the identical `s1.apply_clahe`, `s1.pose_normalize`, `s1.compute_v_bf/v_es/v_jc/v_pd`, `s1.WindowAccumulator`, and `s1.map_to_valence_arousal` — no parallel vector-math implementation exists.
- Mode B auto-selects when the first `CALIBRATION_SECONDS` doesn't yield a usable neutral (`_is_usable_calibration` returns `False`) or when `--uncalibrated` forces it directly.
- Mode B timeline fields are renamed (`valence_relative`, `arousal_relative`, `valence_peak_relative`, `arousal_peak_relative`, `v_bf_relative_logged_only`) — distinct from Mode A's `valence_z_es`/`arousal_z_pd` — so a consumer reading field names alone cannot mistake an approximate Mode B number for a validated z-score.

---

## C) Functional Smoke Tests

All four ran in this session, no camera, no tokens, using only existing files/logs.

1. **Stage 2 stub call** — built a prompt from `logs/session_09a3bc97-....jsonl` (an existing `window_summary` + `calibration_complete` pair), ran `call_agent(prompt, use_stub=True)` directly (no file writes — `call_agent` itself doesn't log), parsed via `parse_agent_pointers`. **Result: 4 pointers parsed correctly** (Pleasure/Arousal/Not-interpreted/Confidence), `is_stub: True`. **PASS.**

2. **`analyze_video.py` on an existing test clip, STUB** — ran against a clip from an earlier session in this conversation. **Result: exit 0, `mode: CALIBRATED (Mode A)`, `status: ok`, 3 windows analyzed, agent report printed in pointer+detail format, result file written.** **PASS.**

3. **`gate2_score.py` against the existing `logs/gate2_trials.jsonl`** — ran fresh (script only ever opens the file in read mode; `grep` confirms no `open(..., "w")` anywhere in it; file's mtime/MD5 confirmed unchanged before and after: `05b0f3f8184cd4a017f768184b43fc0f`).
   - **V_pd: 100.0% (5/5, all-people)** — matches CLAUDE.md exactly.
   - **V_es: 71.4% (5/7, all-people)** — matches CLAUDE.md exactly.
   - **V_bf: the script's own printed summary reports a *pooled* furrow+concentrate score (35.7%, 5/14)** — this is *not* the number CLAUDE.md quotes. Investigated: CLAUDE.md's "furrow 0/7, concentrate 71.4%" comes from the existing `GATE2_RESULT_REPORT.md` (generated by the companion `gate2_report.py`, which imports every scoring function from `gate2_score.py` rather than reimplementing them, then additionally splits the pooled V_bf score into furrow-only/concentrate-only for readability). Read (not regenerated, to stay read-only) `GATE2_RESULT_REPORT.md` §4: **furrow-only 0.0% (0/7), concentrate-only 71.4% (5/7)** — an **exact match** to CLAUDE.md.
   - **n = 6 distinct `person_label`s** (P01B–P06B) — reconciles with CLAUDE.md's "n≈5" because the audit table shows P01B contributing 2 trials of everything (furrow/concentrate/smile), consistent with CLAUDE.md's own stated caveat ("P01B spans 2 session_ids ⇒ one person double-weighted"). Not a contradiction — CLAUDE.md's own text already explains this.
   - Console output explicitly states: `"n=6 < 8 -- this is a PILOT-SCALE DIRECTIONAL RESULT (n=6), NOT a passed or failed Gate 2."` — matches CLAUDE.md's "pilot-scale, not a passed/failed Gate 2" framing.
   - **PASS** (all numbers reproduced exactly, once you know which of the two companion scripts each number comes from — see Part F for the one loose end this reveals).

4. **`soak_log.jsonl`** — read and independently recomputed (not just trusted the stored `soak_summary` record) from the raw 82 `soak_sample` records:
   - Runtime: 2486.9s (≈41.4 min). FPS: min 27.72, max 30.03, mean 29.35 — flat, well above the ≥15 bar, matches CLAUDE.md's "27.7–29.9" almost exactly.
   - Memory: 295.5 → 313.4 MB, peak 335.9 MB — matches CLAUDE.md's "295→313 MB, peak 336" exactly.
   - `any_thread_death: false`, `any_exception: false`, `clean_exit: true`, all 82 samples show `both_threads_alive: true`.
   - **PASS** — this is real, re-derived evidence, not a repeated claim.

---

## D) Single Live Check

**`ANTHROPIC_API_KEY` is NOT present in this session's environment** (checked via both bash and PowerShell). Per the task's hard constraint, I did not attempt to obtain or set it, and made no API call — zero calls were made in this entire report.

To complete this check, run it yourself in a terminal where the key is set:

```bash
python test_live_agent.py
```

It will pull one real `window_summary` + `calibration_complete` pair from an existing session log, make exactly one live call, print the response text/model, and confirm `is_stub: false` was logged. This means **CLAUDE.md's specific claim** ("Live path verified working on `claude-sonnet-5`... one real call returns a warm, honest, pointer+detail read") **could not be re-verified with fresh evidence in this session** — it rests on a prior run (either before this conversation or by the human, per instruction, after an earlier task in this conversation asked them to run it themselves). It was not fabricated or re-asserted here without caveat.

---

## E) POC Pass-Criteria Verdict

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Gate 1: FPS ≥15 with full processing | **MET** | `soak_log.jsonl`'s 82 real samples: FPS never drops below 27.72 across a 41-minute run *with* full CLAHE+landmark+vector processing — stronger evidence than a short isolated Gate-1-only test would give. |
| 2 | Gate 2: per-vector generalization result obtained, honest, pilot-scale | **MET** | `gate2_score.py` + `GATE2_RESULT_REPORT.md` reproduce V_pd 100%, V_es 71.4%, V_bf furrow 0%/concentrate 71.4% exactly, from the actual frozen dataset, this session. Explicitly and correctly labeled pilot-scale (n=6/n≈5, `n<8` printed by the script itself), not a passed/failed Gate 2 — matches Decision 15's honesty requirement. |
| 3 | Stability: 30–60 min soak clean (FPS flat, no leak, no deadlock) | **MET** | Same `soak_log.jsonl` evidence as #1: 41.4 min, memory bounded (not runaway), zero thread deaths, zero exceptions, clean exit. |
| 4 | Live demo on an unseen face works (live agent read) | **NOT RE-VERIFIED THIS SESSION** | No API key available in this environment (see Part D). CLAUDE.md's claim rests on a prior verification not reproduced here. This is the one pass-criterion I could not independently confirm with fresh evidence. |

**Overall: CLAUDE.md's "POC COMPLETE" claim is well-supported by re-derivable evidence for 3 of 4 criteria, gathered fresh in this session — not just repeated. The 4th (live demo) is plausible and previously claimed, but genuinely unverified here due to the missing key; it is not something this report can independently confirm as complete.**

---

## F) Open Items / Risks

1. **`GATE2_FINDINGS.md` does not exist.** CLAUDE.md's STATUS section calls it "authoritative" and links to it twice, but the actual file present in the repo is `GATE2_RESULT_REPORT.md` (confirmed: `ls GATE2_FINDINGS.md` → not found; `ls GATE2_RESULT_REPORT.md` → present and matches CLAUDE.md's numbers exactly, see Part C). This looks like a naming drift in CLAUDE.md, not a missing artifact — the content CLAUDE.md is pointing at does exist, just under a different filename. Worth a one-line fix to CLAUDE.md itself (not done here — read-only).

2. **`gate2_score.py`'s own console summary does not print the furrow/concentrate split CLAUDE.md quotes** — it only prints the pooled V_bf number (35.7%). A future reader who runs *only* `gate2_score.py` (as this task literally named) and stops at its own summary would see a number that doesn't match CLAUDE.md's headline claims, and would need to know to also check `gate2_report.py`/`GATE2_RESULT_REPORT.md` to find the split. Not a bug — `gate2_report.py`'s docstring explains this is intentional (pooled score is "the scored, frozen-rule number"; the split is "descriptive... for readability") — but it's a discoverability gap worth knowing about.

3. **`VECTOR_RELIABILITY["v_bf"] = "high"`** (`stage1_step4_vectors.py:689`) predates Gate 2 and measures calibration *repeatability*, not Gate-2 *generalization* (which failed). This was already flagged in an earlier verification pass this conversation and remains unchanged (correctly — no fixes were made per the read-only instruction each time). It doesn't affect any composite math (V_bf still never feeds Valence/Arousal), but the label reads oddly next to CLAUDE.md's "DEMOTED" framing.

4. **Multi-face "primary/largest face" safeguard — still open, as CLAUDE.md itself says.** Confirmed: the live capture pipeline (`stage1_step4_vectors.py`, `stage3_demo_ui.py`) uses `num_faces=1`, so MediaPipe's own opaque internal selection decides which face is "the" tracked one when multiple are visible — no explicit primary/largest-face logic exists to audit. `analyze_video.py`'s Mode B multi-face heuristic (built in a later step this conversation) is a *detection/reporting* feature (flags that a video looks multi-subject), not a *selection* safeguard — it was explicitly scoped to exclude that fix. CLAUDE.md's "not yet done" note is still accurate today.

5. **`analyze_video.py --live`'s warm report has not been human-verified with a real API call** (same root cause as Part D/E#4 — no key in this or prior sessions in this conversation). The stub path is thoroughly tested; the live path's actual model output has never been inspected.

6. **P01B double-session (n≈5, not 6)** — already documented by CLAUDE.md itself and independently confirmed against the raw audit table in `GATE2_RESULT_REPORT.md` this session (P01B contributes 2 trials of every expression vs. 1 for everyone else). Listed here for completeness, not as a new finding — it's accurately described already.

7. **No TODO/FIXME/XXX markers found** in any `.py` file — a genuinely clean signal, not a risk, included here so the absence is documented rather than assumed.
