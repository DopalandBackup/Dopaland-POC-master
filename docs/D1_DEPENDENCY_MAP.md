# D1 Dependency Map — Feature-Block Separation

**Audience:** an outside reviewer who did not write this code and will check
it against the repository directly (D0PA1 Batch 1, Step 3). Every claim
below names the file/function/test that backs it — check any of them
independently rather than taking this document's word for it.

**Governing rule (CLAUDE.md, D1):**

```
PERMITTED    X_core -> E_t          C_t -> E_t
FORBIDDEN    A_t -> X_core          A_t -> E_t
             U_t -> X_core          U_t -> E_t
```

- `X_core` — core behavioural/affective features (`features/x_core.py`)
- `E_t` — derived episode features, `E_t = h(X_core, ΔX_core, C_t)` (`features/episodes.py`)
- `A_t` — attention: ROI, orientation, dwell, persistence, switching,
  head-gaze coherence, and any downstream attention derivative
  (`features/attention.py`)
- `U_t` — audio (`features/audio.py`)
- `C_t` — context (`features/context.py`)
- `geometry` (`features/geometry.py`) — upstream of all five blocks, belongs
  to none of them. Any block may import it.

## 1. Flow diagram

```
                    +--------------+
                    |  geometry.py |  (upstream — raw landmark math + a
                    +------+-------+   handful of genuinely shared constants)
                           |
              -------------+-------------
              |            |            |
              v            v            v
       +-----------+ +-----------+ +-----------+      +-----------+ +-----------+
       |  x_core   | | episodes  | | attention |      |   audio   | |  context  |
       |  (X_core) | |   (E_t)   | |   (A_t)   |      |   (U_t)   | |   (C_t)   |
       +-----+-----+ +-----+-----+ +-----------+      +-----------+ +-----------+
             |             ^
             |  PERMITTED  |
             +-------------+
        (episodes.py may import x_core.py; today it does NOT need to —
         see section 2's per-feature source-variable trace)

       A_t -> X_core   FORBIDDEN  (no import exists; tests/test_feature_separation.py enforces)
       A_t -> E_t      FORBIDDEN  (no import exists; enforced)
       U_t -> X_core   FORBIDDEN  (audio.py has no content to leak; enforced)
       U_t -> E_t      FORBIDDEN  (audio.py has no content to leak; enforced)
       C_t -> E_t      PERMITTED, currently moot (context.py has no content)
```

One arrow that is **not a violation** and deliberately looks like one:
`BlinkDetector.update(aperture, ...)` inside `features/attention.py` takes
V_es's own aperture value as a **plain parameter** supplied by the caller
(`stage1_step4_vectors.py` / `stage3_demo_ui.py`). `attention.py` never
imports `features.x_core` to fetch that value itself. That is `X_core -> A_t`
— the reverse of the forbidden direction — and it is permitted.

## 2. E_t traced to source variables

`features/episodes.py` has exactly one feature-producing class,
`WindowAccumulator`, and one gate function, `classify_window_confidence`.

| E_t component | Source variables | Upstream blocks used |
|---|---|---|
| `window_summary.composite` (v_bf, v_es, v_pd: avg/peak/variance) | X_core `calibration_deviation` samples collected over one rolling 10s window, plus per-sample `detected`/`yaw_deg` | `x_core` (consumed as plain parameters passed into `add_sample` — `episodes.py` does not import `x_core.py`) |
| `window_summary.covariates` (v_jc, v_bf_convergence_ratio, v_es_cheek_raise: avg/peak/variance) | X_core covariate samples, same window | same as above |
| `window_summary.window_quality` | `detection_rate`, `yaw_variance_deg2` computed from the window's own samples | `geometry.YAW_VARIANCE_CEILING_DEG2` (imported directly by `episodes.py`) |
| `classify_window_confidence` gate | `detection_rate`, `yaw_variance_deg2` (parameters, not global reads) | `geometry.YAW_VARIANCE_CEILING_DEG2`, `episodes.DETECT_RATE_FLOOR` (own constant) |

`episodes.py` does not import `features.x_core` — every X_core value it
touches arrives as a plain function argument from the caller
(`stage1_step4_vectors.py`'s or `stage3_demo_ui.py`'s processing loop). The
`X_core -> E_t` direction is therefore exercised through the **caller**, not
through an inter-module import — both are permitted by the D1 rule, and this
repository happens to use the parameter-passing form. See
`features/manifests/episodes_v1.json` for the versioned, machine-readable
form of this table.

### Decision 2 disclosure — `episode_unit`

`WindowAccumulator` is classified as `E_t` (`E_t = h(X_core, ΔX_core, C_t)`,
and windowed avg/peak/variance is exactly that: a function of `X_core` over
time). **But** "episode" has a specific meaning in the D0PA1 study — a trial
in a controlled task environment — and that environment **does not yet
exist** (blocked on the client's D2 decision: prediction target, action
classes, horizon, tie/rapid-succession handling — see CLAUDE.md's BLOCKED
section). The current 10-second rolling window is an **aggregation window**,
not a study episode, and must never be presented as if it were.

- Disclosed as `"episode_unit": "rolling_10s_window"` in
  `features/manifests/episodes_v1.json`, with an explicit
  `episode_unit_status: "PROVISIONAL -- blocked on client D2 decision"`.
- **Deliberately NOT stamped into the runtime `window_summary` dict itself.**
  Adding a field there would change `window_summary`'s JSON schema and break
  the byte-exact regression net in `tests/test_refactor_snapshot.py`
  (verified: an earlier draft of this refactor added the field directly to
  the dict and the golden-file comparison failed immediately — reverted
  before any commit). The disclosure lives in the manifest and this document
  instead, per G5 (do not alter the validated path's output).
- When the study's real episode definition lands, it may differ from this
  window. That will require a manifest version bump and a documented
  decision, not a silent reinterpretation of what `window_summary` means.

## 3. Exclusion list

Every item CLAUDE.md's D1 section names by category, with where it is
excluded and which test enforces it:

| Excluded signal | Where it lives (never in X_core/E_t) | Enforced by |
|---|---|---|
| ROI | Nowhere — no code exists anywhere in this repository (verified by repo-wide search for `ROI`, `roi_id`, `salience`; see `features/manifests/context_v1.json`) | N/A — nothing to enforce against yet; `features/context.py` is an empty stub |
| Orientation (screen orientation, V_so) | `features/attention.py:compute_v_so`, `features/attention.py:AttentionWindowAccumulator` | `tests/test_feature_separation.py` checks 1–3 |
| Dwell | Nowhere — no code exists (same search as ROI) | N/A |
| Persistence | Nowhere — no code exists (same search) | N/A |
| Switching | Nowhere — no code exists (same search) | N/A |
| Head-gaze coherence | `features/attention.py:compute_v_so` blends head pose with `_gaze_centering_score`'s gaze-centering score — this IS the head-gaze coherence concept, and it lives entirely in `attention.py` | `tests/test_feature_separation.py` checks 1–3 |
| Gaze direction (L/R/C) | `features/attention.py:compute_gaze_direction` | `tests/test_feature_separation.py` checks 1–3 |
| Blink | `features/attention.py:BlinkDetector` | `tests/test_feature_separation.py` checks 1–3 |
| **Audio** | Nowhere — no microphone capture, no audio processing, no clock sync exists anywhere in this repository (`features/audio.py` is an empty stub; see `features/manifests/audio_v1.json`) | `tests/test_feature_separation.py` checks 1 and 3 confirm `x_core.py`/`episodes.py` never import or touch `features.audio`, even though it has no content to leak today — this holds the constraint in place for when the audio module is actually built |

**Both exclusions confirmed separately, as required** (the audio exclusion is
easy to lose because attention dominates the discussion — the client asked
for both by name):

- **Attention exclusion**: `tests/test_feature_separation.py` check 1 (static
  import graph, transitive) and check 2 (static call graph) both scan for
  any reference to `features.attention` or anything `attention.py` defines,
  inside `features/x_core.py` and `features/episodes.py`. Check 3 (runtime)
  poisons the real `features.attention` module object and re-runs real
  X_core/E_t computations (reusing `tests/test_refactor_snapshot.py`'s
  synthetic fixtures) to confirm neither module touches it in practice, not
  just in the static analysis. **Verified to actually fail** — see section 6.
- **Audio exclusion**: the exact same three checks run against
  `features.audio` in parallel (see `FORBIDDEN_EDGES` and the
  `for forbidden_mod in ("attention", "audio")` loop in
  `tests/test_feature_separation.py`). Because `audio.py` has no content
  yet, check 2 (call graph) is vacuous for it today (nothing to leak) — but
  checks 1 and 3 still run and still pass, and will catch the first line of
  code anyone ever adds to `audio.py` that gets imported into `x_core.py` or
  `episodes.py`.

## 4. `C_t` and `U_t`: nonexistent, not clean

`features/context.py` and `features/audio.py` are stated explicitly as
**"nonexistent"** in their own module docstrings and in
`features/manifests/context_v1.json` / `features/manifests/audio_v1.json` —
not "clean" or "compliant". Those two words read very differently to a
client reviewing this repository, and only "nonexistent" is true: a
repo-wide search for `context_id`, `"C_t"`, `ROI`, `roi_id`, and `salience`
terms, run before any code was moved in this task, found **zero matches**
anywhere in the repository. There is no context-feature code and no
audio-feature code to have been separated correctly or incorrectly — the
separation question does not yet apply to these two blocks in a meaningful
sense. Both modules deliberately expose nothing (no functions, no classes,
no constants, no placeholder return values) — an empty module is the honest
representation of "this does not exist yet," per G3. A stub function that
silently returned `None` or a hardcoded placeholder would invite a future
caller to treat the block as present when it isn't.

**Watch `C_t` specifically going forward**: `C_t -> E_t` is a *permitted*
direction (`E_t = h(X_core, ΔX_core, C_t)` by definition), which makes
context a legal-looking back door for attention into the core. Any future
context feature that is ROI-derived or gaze-derived (dwell time on a region,
fixation-weighted context, anything computed from `features/attention.py`)
would be an attention-class value smuggled in through a permitted channel,
and must be flagged loudly when it appears, not added quietly.

## 5. Pre-existing violations found (Step 2, restated)

Before any code was moved, a full trace of both loop implementations and the
agent path was performed to check for information-flow violations under the
**old**, single-file structure (everything living in one module,
`stage1_step4_vectors.py`).

**Method**: every call site constructing arguments to
`NeutralCalibrator.add_sample`, `WindowAccumulator.add_sample`, and
`map_to_valence_arousal` was traced by hand in both
`stage1_step4_vectors.py`'s `processing_thread()` and
`stage3_demo_ui.py`'s `stage3_processing_thread()`. Attention-class locals
(`head_yaw_deg`, `gaze_label`, `gaze_reliable`, `so_components`, `_v_so`)
were confirmed to be computed into separate local variables and never passed
into any X_core or E_t function call. The agent path was traced separately
(`_launch_agent_call -> run_stage2_on_window -> zscore_window`/
`build_prompt` in `stage2_personality_agent.py`): the `summary` argument is
the E_t window object, not the UI state dict, and no attention field is ever
read from it.

**Result: no information-flow violations were found.** The repository was
**structurally entangled** — one file, `stage1_step4_vectors.py`, held
X_core, E_t, and A_t code side by side — but nothing in that entangled file
actually violated the directional constraint at runtime. This task's job was
therefore to fix the *structure* (so the constraint becomes machine-checkable
and cannot be violated by accident in the future) without changing any
*behavior* — see section 6 for how that was verified, and
`tests/test_refactor_snapshot.py` for the byte-exact proof that it held.

## 6. Runtime isolation proof — verified to actually fail

A separation test that has never been observed to fail is worth nothing (the
same reasoning that required this repository's pre-commit media-guard hook
to be *proven*, not assumed — see `PROVENANCE.md`). This was verified
directly: `from features.attention import ATTENTION_ORIENTED_SCORE_THRESHOLD`
was temporarily added to the top of `features/x_core.py`, and
`tests/test_feature_separation.py` was re-run:

```
[1/3] STATIC IMPORT GRAPH
      x_core.py direct features.* imports: ['attention', 'geometry']
      FAIL:
        - x_core.py imports (transitively) attention.py -- path: x_core -> attention
[2/3] STATIC CALL GRAPH
      PASS -- no symbol defined in attention.py/audio.py is referenced by x_core.py/episodes.py.
[3/3] RUNTIME MONKEYPATCH (attention/audio raise on any attribute access)
      FAIL: ImportError: cannot import name 'ATTENTION_ORIENTED_SCORE_THRESHOLD' from '<unknown module name>' (unknown location)

FEATURE SEPARATION TEST: FAIL (2 violation(s))
```

(Check 2 stays green in this specific case because the violation is an
*unused* import — nothing in `x_core.py`'s code actually *references* the
name `ATTENTION_ORIENTED_SCORE_THRESHOLD` after importing it, so there is
no call-graph reference for check 2 to catch. Checks 1 and 3 catch it
regardless, which is exactly why the task specified three independent
checks rather than relying on any single one.)

The line was then reverted, and both `tests/test_feature_separation.py` and
`tests/test_refactor_snapshot.py` were re-run and confirmed passing (exit
code 0, SHA256 `4f9c0f1786c18e8dbe5e3048b8b6b6e280cf6c434b9c53b119344746fc31bcff`
unchanged) before any commit was made with the violation present.

## 7. Two parallel loop implementations — do they diverge?

**Yes — three ways, all confirmed by direct code inspection, none of them
in the validated X_core/E_t math itself.**

`stage1_step4_vectors.py`'s `processing_thread()` and
`stage3_demo_ui.py`'s `stage3_processing_thread()` both call the exact same
`features.x_core` functions (`compute_v_bf`, `compute_v_es`, `compute_v_jc`,
`compute_v_pd`, `NeutralCalibrator`, `map_to_valence_arousal`) — after this
refactor they are literally the same code objects, not just similar logic,
so the underlying affect-vector math cannot diverge between them. The
divergences are all in what each loop does *around* that shared core:

1. **v_jc z-scoring.** `stage3_demo_ui.py` additionally z-scores the
   logged-only `v_jc` covariate for its own live UI display
   (`z_scores["v_jc"] = x_core._z_score(jc_dev, jc_std)` in
   `stage3_processing_thread`). `stage1_step4_vectors.py`'s own
   `processing_thread()` never calls `_z_score` at all (confirmed: the only
   reference to `_z_score` in `stage1_step4_vectors.py` is the re-export
   import at the top of the file, not a call) — it only ever logs `v_jc`
   raw and windowed-raw, never z-scored.

2. **V_so usage.** `stage1_step4_vectors.py` uses `compute_v_so`'s **full**
   return value every cycle (score, `oriented`, `gaze_score`,
   `gaze_reliable`, `head_pose_only`) and windows it independently via
   `AttentionWindowAccumulator` every 10 seconds, logging both the per-sample
   detail and the window summary to its own JSONL log.
   `stage3_demo_ui.py` calls the same function but **discards everything
   except `yaw_deg`** (a deliberate, documented anti-overclaiming choice —
   see that file's own comment: pitch and the blended score would imply a
   validated orientation reading the pipeline cannot honestly make) and
   **never instantiates `AttentionWindowAccumulator` at all** (confirmed by
   search — zero references in `stage3_demo_ui.py`).

3. **Gaze/blink tracking.** `stage3_demo_ui.py` additionally runs
   `compute_gaze_direction` and `BlinkDetector` every cycle — code
   `stage1_step4_vectors.py`'s own loop never calls (confirmed: zero
   references to either name in `stage1_step4_vectors.py`) — and windows
   the result into its own `experimental_signals_window` records, written to
   a separate `logs/experimental_signals_log.jsonl` that
   `stage1_step4_vectors.py` never writes.

`analyze_video.py` is a **third**, further-diverging consumer (checked per
the task's instruction, not one of the "two parallel loops" but worth
recording here): it calls **zero** `features.attention` functions in either
of its two modes (confirmed by search — no `compute_v_so`, no
`compute_gaze_direction`, no `BlinkDetector` anywhere in the file). Mode A's
calibration runs against *video time* (`video_time`, derived from frame
index / fps) rather than wall-clock `time.perf_counter()`; Mode B skips
calibration entirely and uses a population-default baseline. Its output is
its own `_build_timeline_entry` record shape, not the `sample`/
`window_summary` JSONL schema the live pipeline writes.

**None of this was introduced by this task** — the divergence already
existed before Step 3 began (this refactor only moved *where* the shared
functions live, it did not touch any of the three loops' own
orchestration logic). It is recorded here because the task asked whether
the loops have diverged, independent of this task's own scope.

## 8. Call sites updated (Step 3.2)

All three consumers were updated to import the moved feature blocks by
module (`from features import geometry, x_core, episodes, attention`) rather
than transitively through `stage1_step4_vectors` — see the task's final
report for the exact per-symbol substitution counts. `stage1_step4_vectors.py`
itself re-exports every moved name it uses internally (so its own
`processing_thread()` body needed no call-site edits beyond the import
block) plus a small additional set of names (`classify_window_confidence`,
`_dist`, and the individual landmark-index constants) that historical
one-off diagnostic scripts (`orientation_capture.py`,
`stage1_step4_*_session.py`, `stage1_step8_calibration_repeat_test.py`,
`stage1_step9_gate2_capture.py`, and `tests/test_refactor_snapshot.py`
itself) import directly by name from `stage1_step4_vectors` — all confirmed
to still import and run cleanly after the move.
