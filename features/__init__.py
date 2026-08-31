"""
D0PA1 feature-block package (Batch 1, Step 3 -- feature-block separation).

Five blocks, per the D1 requirement in CLAUDE.md:
  x_core.py      X_core -- core behavioural/affective features
  episodes.py    E_t    -- derived episode features (E_t = h(X_core, dX_core, C_t))
  attention.py   A_t    -- attention (ROI, orientation, dwell, persistence,
                            switching, head-gaze coherence, and any downstream
                            attention derivative)
  audio.py       U_t    -- audio (STUB ONLY -- no microphone, no capture, no
                            clock sync exists; pending a keep-or-formally-
                            remove decision)
  context.py     C_t    -- context (STUB ONLY -- no context features exist yet)

geometry.py is UPSTREAM of all five blocks and belongs to none of them --
shared raw-landmark utilities and the small number of constants genuinely
used by more than one block. Any block may import it.

DIRECTIONAL CONSTRAINT (see docs/D1_DEPENDENCY_MAP.md for the full picture):
  PERMITTED   X_core -> E_t        C_t -> E_t
  FORBIDDEN   A_t -> X_core        A_t -> E_t
              U_t -> X_core        U_t -> E_t

x_core.py and episodes.py must never import attention.py or audio.py, directly
or transitively. tests/test_feature_separation.py enforces this both
statically (import-graph + call-graph inspection) and at runtime (monkeypatch
attention/audio to raise on any attribute access, then compute X_core/E_t and
confirm neither module was touched).
"""
