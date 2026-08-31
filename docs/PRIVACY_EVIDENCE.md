# Privacy / Anonymization Evidence

Facts verified directly against this repository's contents, on **2026-08-24**,
by direct inspection and small purpose-built scripts (not summarized from
documentation or memory). Every number below was re-run fresh in the same
session this file was written, specifically so it could be cited precisely.
Where I was not fully certain of a value, that uncertainty is stated rather
than rounded away.

---

## 1. `person_id` is null in every sample record checked

**Method:** iterated every `logs/session_*.jsonl` file, parsed every line as
JSON, checked whether the `person_id` field was non-`null`.

**Files covered:** 19 files, matched by the glob `logs/session_*.jsonl`.

**Result:** **15,619 sample records checked. 0 had a non-null `person_id`.**
Face-ID (CLAUDE.md: "Face-ID stays OUT") is genuinely never populated in this
data, not merely documented as absent — confirmed against every record in
every session log that exists in this repository, not a sample.

This check covers only files matching the `session_*.jsonl` naming pattern.
It does not cover the differently-named diagnostic logs (`gate2_trials.jsonl`,
`orientation_trials.jsonl`, `browdiag_*`, `yawhold_*`, etc.) — those were
inspected separately (see §2) and use `person_label`/`participant_code`
instead of `person_id`, so the field this check targets does not apply to
them.

## 2. Every `person_label` / `participant_code` across every log is an anonymous code

**Method:** grep for the literal patterns `"person_label": "..."` and
`"participant_code": "..."` across every `.jsonl` file in `logs/`, deduplicated.

**Files covered:** 39 `.jsonl` files (all of `logs/*.jsonl`).

**Result — 32 distinct `person_label` values, all anonymous codes, no real
name in any of them:**
```
P01, P011B, P01B, P01C, P02, P02B, P03, P03B, P04, P04B, P05, P05B, P06,
P06B, P07B, P08B, P09B, P10B, P11B, SOAKTEST, T, TEST, TEST1, TEST2, TEST3,
TEST4, TEST5, TEST6, TEST_INTEGRATION, TEST_THREE_SIGNAL, p01, p02
```

**Result — 3 distinct `participant_code` values** (used by the Gate 2 /
orientation trial capture tools, a separate field from `person_label`):
```
TEST, TEST1, TEST2
```

## 3. `consent_log.jsonl` fields — corrected from an earlier, less precise pass

**Method:** parsed all 41 records, took the **union** of keys across every
record (not just the first one — the first record undercounts, see below).

**Result:** the field set is **not uniform across all 41 records.**
- All 41 records have: `event`, `session_id`, `ts_utc`, `ts_monotonic`.
- **40 of 41** records additionally have `person_label` (an anonymous code —
  values observed: `P01, P011B, P01B, P01C, P02, P02B, P03, P03B, P04, P04B,
  P05, P05B, P06, P06B, P07B, P08B, P09B, P10B, P11B, TEST, TEST1, TEST2,
  TEST3, TEST4, TEST5, TEST6, p01, p02`).
- **1 of 41** records lacks `person_label` entirely (the earliest record by
  file order — consistent with that field being added to the schema after
  the first consent event was logged).

No name field exists in any record, with or without `person_label`. Example
record:
```json
{"event": "consent_given", "session_id": "a1d2b506-babc-4191-9183-01645f3a7a47",
 "person_label": "P01", "ts_utc": "2026-07-04T15:30:12.786597+00:00",
 "ts_monotonic": 628508.0784129}
```

**Correction note:** a prior verbal summary of this file's schema (given in
a previous session) said its fields were "event, session_id, ts_utc,
ts_monotonic" only. That was incomplete — it described the first record, not
the union of all 41. This entry is the corrected version.

## 4. LLM prompt content: derived features only, verified against a real exchange

**Method:** read a full, non-truncated request/response pair directly from
`logs/agent_log.jsonl` where `is_stub` is `false` (a real API call, not the
stub path).

**Result — verbatim prompt sent to the model:**
```
You are a single-purpose assistant that describes ONLY a live, current 10-second behavioral-signal reading, in plain, honest language.

STRICT RULES (do not deviate):
- You are speaking about a SINGLE LIVE READING, right now. You have no memory of this person and no access to any past session. NEVER reference "last time", history, trends, progress, or repeated sessions. This is a first and only reading (L1).
- These are BEHAVIORAL SIGNALS / AFFECTIVE INDICATORS -- not emotions, not a diagnosis, not clinical or medical information. Never use the word "emotion" or any clinical/diagnostic language.
- State uncertainty honestly. If a signal is weak, unavailable, logged-only, or the window is low-confidence, say so plainly instead of ignoring it or reading it with unearned confidence.
- Do not predict what happens next and do not give advice. Describe only what the current signals show.

CURRENT SIGNALS (this person's own calibrated baseline, this window only, person_label=P04):
- Valence: z = +0.42 -- derived ONLY from V_es (eye-crinkle/aperture). This is a PLEASURE-SIDE-ONLY signal: it can indicate a positive-affect signal, but there is no validated signal for negative/pain-side valence in this system. A near-zero or negative value means "no detected pleasure signal", not "a detected negative signal".
- Arousal: z = +1.63 -- derived from V_pd (postural volatility) only.
- V_bf (brow furrow) and V_jc (jaw compression) are logged-only in this system and are NOT interpreted or included in this reading.

Write 2-4 plain-language sentences describing this single reading using "behavioral signals" or "affective indicators" language. Do not invent history. Do not diagnose. Do not predict.
```

**Result — verbatim response received:**
```
Right now, this single reading shows a mild positive-affect signal around the eyes (eye-crinkle/aperture), suggesting a modest pleasure-side indicator, though this is based on one narrow signal and shouldn't be read as a strong or definitive marker. Alongside that, postural volatility is notably elevated, pointing to a heightened arousal-type signal in this window. Brow furrow and jaw compression data were logged but aren't being interpreted here, so no statement can be made about those channels. Overall, this is a single 10-second snapshot with limited signal sources, not a full or certain picture.
```
Model: `claude-sonnet-5`.

**Content of the prompt:** fixed instructional text, two z-scores, and the
anonymous `person_label` code. No name, no raw landmark coordinates, no image
or video data, no free-text field a participant could have filled in.

**Scope limitation, stated honestly:** this is one representative exchange,
manually selected as the first `is_stub=false` record in the file, not an
exhaustive review of all 622 records in `agent_log.jsonl`. The prompt is
built from a fixed template (confirmed by reading `build_prompt()` in
`stage2_personality_agent.py`), so structural content should be consistent
across records, but I have not individually inspected every one of the 622
entries for this file.

## 5. Magic-byte media scan

**Method:** walked the entire working tree (excluding `.git`, which does not
yet exist at time of writing), read the first 64 bytes of **every file**,
checked against known signatures for JPEG, PNG, GIF, BMP, RIFF (WAV/AVI),
MP4/MOV (`ftyp` box), MKV/WEBM (EBML), MP3 (ID3), OGG, and FLAC — independent
of what the file's extension claims. Also checked every file's extension
against a broad media-extension list as a second, independent signal.

**Files covered:** 92 files (the entire tree as it existed at the moment this
check was run, after `.gitignore`, `.githooks/pre-commit`, `PROVENANCE.md`,
and `models/MODEL_MANIFEST.md` had already been added this session).

**Result:** **0 files flagged, by extension or by magic bytes.** No raw
media of any kind exists anywhere in this repository as of this commit.

This scan was re-run fresh at multiple points during this session; the count
of files checked differs between runs only because new (non-media) files
were being added to the tree in the course of doing this work — the flagged
count was 0 every time it was run.
