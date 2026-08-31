# V_bf (Brow Furrow) Diagnostic Report

**Status:** Read-only investigation. No code, data, or formula was changed to produce this report.
**Scope:** Why V_bf scored low (25.0% all-people) in the pilot-scale Gate 2 run — sign/formula bug in code, or weak/inconsistent execution by the person being captured?
**Data source:** `logs/gate2_trials.jsonl` (unmodified), `stage1_step4_vectors.py` (read only), `stage1_step9_gate2_capture.py` (read only).

---

## 1. Furrow sign pattern across all people

| person_label | session (first 8) | furrow V_bf avg (z) | concentrate V_bf avg (z) | Result | Reason |
|---|---|---|---|---|---|
| P01 | b09ceab0 | -3.551 | +0.304 | furrow: fail | wrong sign (magnitude was large enough) |
| P05 | 4642e551 | -3.081 | +0.355 | furrow: fail | wrong sign |
| P06 | fc701cbf | -12.500 | +2.066 | furrow: fail (concentrate: **pass**) | furrow wrong sign |
| P06 | bc1856d0 | -1.687 | -1.399 | furrow: fail, concentrate: fail | furrow wrong sign; concentrate wrong sign too |
| p01 | 0636539a | -0.780 | -0.157 | furrow: fail, concentrate: fail | furrow wrong sign AND below \|z\|≥1.0; concentrate same |
| p02 | 768ea0c6 | **+4.215** | **+3.212** | furrow: **pass**, concentrate: **pass** | correct sign, strong magnitude, both blocks |

**Data-quality note surfaced during this investigation** (not the sign question, but material to reading the table above): the label **"P06" is reused across two separate sessions** (`fc701cbf...`, 2026-07-11, and `bc1856d0...`, 2026-07-12) — two independent capture runs, not one person's two attempts. "P01..P06" therefore names 6 independent sessions, not cleanly 6 different real participants; all of it is the developer's own face across a week of ad hoc tool-testing sessions (confirmed by session timestamps), not a dedicated multi-person Gate 2 sitting.

### Summary counts

- **Furrow: 5 of 6 sessions NEGATIVE (83%), 1 of 6 POSITIVE.** Magnitude range on the negative side: -0.780 to -12.500 (large — 4 of the 5 negative trials clear \|z\|≥1.0 easily). This is **not** "right sign but weak" and **not** random/mixed — it is a strong, consistent skew in the wrong direction.
- **Concentrate: 4 of 6 sessions POSITIVE (67%), 2 of 6 NEGATIVE.** Majority-correct, and the two negative cases are the same sessions where furrow also failed.
- **The critical fact:** furrow and concentrate both map to V_bf through the identical formula and identical expected sign, yet they skew in *opposite* directions in this data.

---

## 2. The core question — scorer vs. pipeline sign convention

**Scorer expects for furrow:** POSITIVE (`gate2_score.py`, `EXPECTED_SIGN_POSITIVE = {"v_es", "v_bf"}`).

**Pipeline's actual design**, from `stage1_step4_vectors.py::compute_v_bf` (current code, quoted verbatim):

```python
# both SHRINK when furrowing; flip sign so higher = more furrow
composite = -drop_ratio
```

where `drop_ratio` = inner-brow-to-glabella distance ÷ interocular distance. Both `drop_ratio` and the logged `convergence_ratio` **shrink** as brows lower/pull together; negating `drop_ratio` is a deliberate sign flip so the **composite increases (more positive) as furrow intensifies**. No further sign flip exists anywhere downstream — `NeutralCalibrator.deviation()` is a plain subtraction, `_z_score()` is a plain division, and the Gate 2 capture script's own z-scoring (`stage1_step9_gate2_capture.py`) does the same, unmodified.

**Match or mismatch:** **MATCH.** The code's documented design intent (genuine furrow → positive deviation → positive z) is exactly what the scorer expects. There is no scorer/pipeline sign mismatch in the code as written.

---

## 3. Cross-check against the developer's own earlier n=1 result

Stage 1 validation (documented in `compute_v_bf`'s own docstring, and in this project's history) already tested this *exact* formula (`composite = -drop_ratio`) under a careful, distance-controlled, maximal-effort furrow protocol and found it **moved correctly, 3 of 3 reps** (effect sizes 3.1-4.4σ).

**The current code has not changed since that validated result** — `compute_v_bf` still reads `composite = -drop_ratio`, unmodified. So: own-face furrow, executed the same deliberate, maximal, distance-controlled way as that earlier test, **would still pass under the current code** — nothing regressed. p02's furrow trial in the current Gate 2 data (+4.215, strong and correctly signed) is a live confirmation of this: the same code path, on a real capture, produced the expected result when the gesture was apparently performed well.

---

## 4. Conclusion

> **LIKELY WEAK EXECUTION / INSTRUCTION (people) — not a sign/convention bug.**

Justification: A code-level sign bug would make furrow *and* concentrate skew the same wrong way, since both route through the identical `compute_v_bf` formula and the identical expected sign — instead they skew in *opposite* directions (furrow mostly negative, concentrate mostly positive), which a shared-code bug cannot explain but session-to-session behavioral variation can. p02's furrow trial hits the correct sign at strong magnitude through this exact code path, and this exact formula was already independently validated 3/3 correct under careful maximal-effort testing in Stage 1 — together these rule out "the pipeline can't produce a correct furrow reading."

**Named uncertainty (not resolved by the numbers alone):** a plausible alternative to "weak effort" is that "furrow" was sometimes executed as an eyebrow *raise* (quizzical/surprised) rather than a lower-and-pull-together — a raise would increase `drop_ratio` and produce exactly this negative signature, and would not be "weak," just the wrong gesture for a verbal-only cue with no demonstrated example. The data cannot distinguish "tried but weakly" from "did the wrong motion" — that needs either a video review (not available) or a proper Gate 2 run where the furrow gesture is demonstrated before capture, done as dedicated data collection rather than incidental tool-testing.

---

## 5. Impact note — is the existing dataset salvageable if this *were* a code bug?

No code bug was found (see §4), so this question is currently moot for action — but answered directly, as requested, since it determines what a future fix would cost:

- **`calibration_baseline` in every Gate 2 trial record stores ONLY the composite-level neutral (mean/std/n for `v_bf`, `v_es`, `v_pd`)** — confirmed by reading `stage1_step9_gate2_capture.py::build_trial_record` (`"calibration_baseline": calibrator_reference["composite"]`) and by inspecting a real record.
- **`raw_components.v_bf` in every trial DOES store both sub-components' raw windowed values** (`convergence_ratio` and `drop_ratio`, each with avg/peak/variance/n) — confirmed directly from the stored data.
- **Consequence:** if the needed fix were "the composite's sign is backwards" (`composite = -drop_ratio` should be `+drop_ratio`), every existing trial is trivially re-scorable with no recapture: negating a value negates its deviation and its z-score uniformly, so `new_z = -old_z` for every stored `window_z.v_bf` value, using data already in `gate2_trials.jsonl`.
- **If instead the needed fix were a different composite** (e.g., switching to `convergence_ratio`, or some new combination), the existing data is **not** sufficient: `convergence_ratio`'s own per-person neutral baseline is computed internally by `NeutralCalibrator` (it lives in `COVARIATE_KEYS`) but is **never persisted** into the Gate 2 trial record — only the composite's baseline survives. Re-scoring under a materially different formula would need calibration re-run at minimum, and full recapture in practice, since the tool has no standalone calibration-replay path.

Since §4 concludes this is not a code bug, none of this is actionable right now — it's recorded so a future decision (if the sign question is ever reopened) doesn't have to re-derive it.
