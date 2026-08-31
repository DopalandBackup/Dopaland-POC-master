# GATE 2 — OPERATOR RUN-SHEET (pre-registered; follow identically for every person)

> The capture tool (`stage1_step9_gate2_capture.py`) is deliberately DUMB — it
> records, it does not judge. **That means the rigor of Gate 2 lives HERE, in what
> you do as the operator.** Follow this sheet the same way for every person. It is
> written and dated BEFORE person 1 — treat it as frozen alongside the scoring rule.
> If you change the protocol mid-study, you restart from person 1.

---

## THE ONE RULE THAT MATTERS MOST (read every time)

**You accept or redo a block based ONLY on whether the person PERFORMED the
commanded expression and held it without disturbance — NEVER on whether the numbers
moved the "right" way.**

- Redo is for *fumbles*: they sneezed, laughed, talked, misunderstood, someone
  walked in, they broke the hold early.
- Redo is NOT for: "the smile didn't push V_es up much, let me get a better one."
  The moment you redo to chase a nicer signal, you have hand-tuned the result to
  what you wanted (Pitfall #5), and the ≥75% number becomes worthless. The whole
  point of Gate 2 is to find out whether the signal moves — you don't get to help it.

If a person performs the expression correctly and the vector doesn't move, that is
a REAL RESULT. Accept it. That is exactly the data Gate 2 exists to collect.

---

## WHO TO RECRUIT (sample = the experiment)

- **8 minimum, aim for 12** (Decision 7). Below 8, Gate 2 has no denominator.
- **VARIED people** — this tests generalization across faces. Deliberately vary skin
  tone, age, gender, facial hair, and glasses/no-glasses. Do NOT run 10 similar
  colleagues; that secretly re-tunes to one face type and defeats the point.
- Known-hard cases are WELCOME, not avoided (Pitfall #4): heavy beard (V_jc will be
  weak — expected), glasses (possible iris reflection). Run them and note it.

---

## PER-PERSON FLOW (identical every time)

### 1. Consent (their choice, no pressure)
- Let them read the consent screen themselves. Answer questions honestly.
- Frame it as an **opt-in wellness/validation tool**, never "monitoring" (Honest
  Framing). It reads geometry only, saves no video/audio, is local and
  session-only.
- If they decline (blank / no), the tool exits and records nothing. That's fine —
  thank them, move on. Do not persuade.

### 2. Participant code
- Assign the next sequential code: P01, P02, P03… Type it when prompted.
- **This is NOT a name.** Do not keep a name↔code list anywhere. The code is an
  anonymous join key, that's all (Decision 12).

### 3. Calibration — 25s neutral (the load-bearing step)
Say, calmly:
> **"Relax your face completely — as if you were sitting alone, resting. Look at
> the screen but don't react to it. Just let your face go slack for a few seconds."**

- This wording is not optional. A held slight-smile or held slight-frown during
  calibration silently biases every downstream reading, and the software CANNOT see
  a uniformly-held expression (Decision 45). **You are the control** — watch the
  live raw values and their face during the 25s.
- **Re-run policy (max 2×):** if there's an *obvious disturbance* (they talked,
  laughed, moved a lot) or a vector flags AND their face clearly wasn't neutral,
  re-run. **Maximum twice.**
- If a vector still flags after 2 re-runs but their face looked genuinely relaxed to
  you: **KEEP the person, note "P0X: [vector] low-confidence baseline," and move
  on** (frozen rule §4). Do NOT re-run a 3rd time to chase a green flag — that
  selects for lucky-clean baselines and can accept a stably-wrong one.
- Reminder: V_bf flag = investigate (it's the reliable vector). V_es flag = expected
  (it drifts ~0.02, known). V_pd flag = check they weren't shifting.

### 4. The five blocks — FIXED ORDER, never reordered
`smile → furrow → concentrate → sit-still → fidget`

Read each command clearly. You may demonstrate ONCE for clarity (identically for
everyone), but do NOT coach during the 10s hold and do NOT react to the numbers.

| Block | Say this (identical every person) |
|-------|-----------------------------------|
| **Smile** | "Give me a big, genuine smile — like you just saw someone you love. A real one. Hold it." *(genuine matters — V_es is the Duchenne/eye part, not a posed mouth-smile)* |
| **Furrow** | "Frown hard — pull your eyebrows down and together, like you're angry or in pain. Hold it." |
| **Concentrate** | "Now concentrate hard — like you're solving a difficult problem in your head. Hold that focus." |
| **Sit-still** | "Sit as completely still as you can — like you're posing for a photo. Don't move. Hold it." |
| **Fidget** | "Now shift and fidget — move in your seat, adjust yourself, like you're restless and can't get comfortable. Keep moving until I say stop." |

For each: get-ready prompt → 10s hold → review screen → **accept (A)** or **redo (R)**
per THE ONE RULE above.

### 5. Mid-session opt-out
- Tell them up front: "You can stop any time — just say so, or press Q."
- If they stop, the run discards everything for that person (nothing is written).
  Respect it immediately. Re-recruit is optional, never pressured.

---

## WHAT YOU MUST NOT DO

- Do NOT accept/redo based on the numbers (THE ONE RULE).
- Do NOT coach toward a bigger expression mid-hold.
- Do NOT skip or soften the calibration "relax completely" instruction.
- Do NOT reorder blocks or add/remove blocks.
- Do NOT run only easy/similar faces.
- Do NOT keep names, emails, or any identity tied to a P-code.

---

## OPERATOR NOTES LOG (keep alongside runs — findings, not PII)

For each person, jot only non-identifying, study-relevant facts. Examples:
- `P03: heavy beard — V_jc expected weak (fine).`
- `P05: glasses — watch for iris reflection.`
- `P07: V_es low-confidence baseline after 2 re-runs — kept per §4.`
- `P09: stopped mid-session, discarded.`

These notes feed the honest write-up (which faces were hard, how many flagged).
They are findings. They contain NO names.

---

## STOP CONDITION

Run until you have **≥8 completed people (aim 12)**. Then capture is done — hand the
`gate2_trials.jsonl` + this notes log to the SEPARATE scoring step (frozen rule,
applied by a human, never by the pipeline). Do not score as you go.
