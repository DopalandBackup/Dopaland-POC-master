"""
Stage 2 — the single personality-agent call (Track A, Phase 4).

STUB ONLY. No API key required, no network call, no tokens spent. The
real Anthropic API is wired in behind call_agent() as a single,
isolated swap point (see call_agent's docstring) -- everything else
(prompt building, response parsing, logging) is written once and works
identically against the stub or the real API, because both return the
same shape: {"text": str, "model": str, "is_stub": bool}.

VECTOR INPUTS (Decision 17/18, per the Gate 2 result):
  Valence = z_es only (pleasure-side only -- no validated pain axis).
  Arousal = z_pd only.
  V_bf and V_jc are BOTH logged-only now (V_bf demoted alongside V_jc --
  its furrow direction did not generalize across faces at Gate 2). They
  are surfaced to the agent as "logged-only, not interpreted", never
  folded into Valence or any composite. The old
  Valence = (z_es - z_bf)/2 formula is retired -- it does not appear
  anywhere in this file.

This module does NOT touch capture, calibration, vector math, or
gate2_trials.jsonl. It consumes window_summary + calibration_complete
records the production pipeline (stage1_step4_vectors.py) already
writes to its session log, and reuses that file's own _z_score()
rather than reimplementing z-scoring.
"""

import json
import os
import uuid
from datetime import datetime, timezone

from features.x_core import _z_score

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
AGENT_LOG_PATH = os.path.join(LOG_DIR, "agent_log.jsonl")
SCHEMA_VERSION = "1.0"

STUB_MODEL_NAME = "stub-v1"
REAL_MODEL_NAME = "claude-sonnet-5"  # verified current against docs.claude.com models overview (Claude API ID + alias both "claude-sonnet-5") before wiring the live call below


def zscore_window(window_summary, calibration_reference):
    """Converts a production window_summary's composite avg/peak/variance
    (already DEVIATION from neutral -- see WindowAccumulator) into
    z-scored form using that person's calibration std. Reuses
    stage1_step4_vectors._z_score() for avg/peak (z = deviation/std);
    variance of a z-scored variable is variance(deviation)/std**2 (a
    unit conversion, not new vector math -- Var(X/std) = Var(X)/std**2).

    Returns None for any field where the input or the std is missing --
    never fabricates a number.
    """
    result = {}
    for vector in ("v_bf", "v_es", "v_pd"):
        dev_stats = window_summary.get("composite", {}).get(vector, {})
        std = calibration_reference.get("composite", {}).get(vector, {}).get("std")
        avg_dev = dev_stats.get("avg")
        peak_dev = dev_stats.get("peak")
        var_dev = dev_stats.get("variance")

        z_avg = _z_score(avg_dev, std)
        z_peak = _z_score(peak_dev, std)
        z_var = (var_dev / (std ** 2)) if (var_dev is not None and std is not None and std >= 1e-9) else None

        result[vector] = {"avg": z_avg, "peak": z_peak, "variance": z_var, "n": dev_stats.get("n")}
    return result


def build_prompt(window_z, person_label, low_confidence, quality_reasons):
    """Builds the prompt text for the agent call. Pure text construction
    -- no vector math happens here, window_z is already z-scored by
    zscore_window().

    Framing requirements (Pitfall #3 / Gap 2 L1, Honest Framing #10) --
    unchanged from the original dry version, just delivered in a warmer
    VOICE (Decision: warmth is a tone instruction, never a claims
    change):
      - L1 only: no session history, no "last time", no trends.
      - "behavioral signals / affective indicators", never "emotion",
        never clinical/diagnostic language.
      - Valence is explicitly labeled pleasure-side-only (z_es only).
      - Arousal from z_pd only.
      - V_bf and V_jc explicitly named as logged-only, not interpreted.
      - Low-confidence windows are surfaced, not hidden.

    Response shape (POINTERS + DETAIL, not one dense paragraph): the
    agent is instructed to return exactly 8 labeled lines --
    PLEASURE_HEADLINE/DETAIL, AROUSAL_HEADLINE/DETAIL,
    NOT_INTERPRETED_HEADLINE/DETAIL, CONFIDENCE_HEADLINE/DETAIL -- a
    fixed, parseable contract. parse_agent_pointers() below is the only
    other place that format is assumed; stub_agent_call() also returns
    it so the UI panel renders identically whether stubbed or live.
    """
    valence_z = window_z["v_es"]["avg"]
    arousal_z = window_z["v_pd"]["avg"]

    valence_str = "no reading available this window" if valence_z is None else f"z = {valence_z:+.2f}"
    arousal_str = "no reading available this window" if arousal_z is None else f"z = {arousal_z:+.2f}"

    confidence_note = ""
    if low_confidence:
        reasons_str = "; ".join(quality_reasons) if quality_reasons else "tracking was unstable this window"
        confidence_note = (
            f"\nWINDOW CONFIDENCE: LOW. Reason(s): {reasons_str}. "
            f"Treat the numbers above as unreliable this window -- say so plainly rather than reading them confidently."
        )
    stability_note = "flagged low-confidence this window" if low_confidence else "tracking looked stable this window"

    prompt = f"""You are a warm, conversational voice describing ONE live, current 10-second behavioral-signal reading directly TO the person wearing the sensor -- like a friendly, attentive observation, not a lab report. Natural and human, never dry or clinical.

STRICT RULES (do not deviate -- warmth belongs in the VOICE, never in the CLAIMS):
- Speak directly to the person ("you"/"you're"), about THIS moment only. You have no memory of them and no access to any past session -- never reference "last time", history, trends, progress, or repeated sessions. This is a first and only reading (L1).
- These are BEHAVIORAL SIGNALS / AFFECTIVE INDICATORS -- never say "emotion", never use clinical or diagnostic language, never diagnose, never predict what happens next, never give advice.
- State uncertainty honestly and plainly -- do not dress up a weak, missing, or low-confidence signal as more than it is.
- Never claim more than these two validated signals support: pleasure-side valence (from V_es only -- there is NO validated pain/negative-valence axis; a near-zero or negative value means "no detected pleasure signal", not "detected pain") and arousal (from V_pd, postural volatility, only).
- Brow furrow (V_bf) and jaw compression (V_jc) are logged-only in this system -- do not read, interpret, or describe what they show; only note that they're being logged and not interpreted.

CURRENT SIGNALS (this person's own calibrated baseline, this window only, person_label={person_label}):
- Pleasure-side valence: {valence_str} (from V_es, eye-crinkle/aperture, only)
- Arousal: {arousal_str} (from V_pd, postural volatility, only){confidence_note}

Return EXACTLY these 8 lines, one per line, in this exact order, with no extra commentary before or after:
PLEASURE_HEADLINE: <a short, warm headline, under 8 words, speaking to the person about the pleasure-side signal>
PLEASURE_DETAIL: <one warm, honest sentence grounding that headline in what was actually measured, stating uncertainty if the signal is weak or missing>
AROUSAL_HEADLINE: <a short, warm headline, under 8 words, speaking to the person about the arousal signal>
AROUSAL_DETAIL: <one warm, honest sentence grounding that headline in what was actually measured, stating uncertainty if the signal is weak or missing>
NOT_INTERPRETED_HEADLINE: <a short headline noting brow/jaw are not part of this reading>
NOT_INTERPRETED_DETAIL: <one sentence -- logged only, not read or interpreted>
CONFIDENCE_HEADLINE: <a short headline naming how much to trust this reading>
CONFIDENCE_DETAIL: <one sentence -- single 10-second window, only two validated signals, {stability_note}>"""

    return prompt


_POINTER_FIELDS = [
    ("PLEASURE_HEADLINE", "PLEASURE_DETAIL", "Pleasure signal"),
    ("AROUSAL_HEADLINE", "AROUSAL_DETAIL", "Arousal signal"),
    ("NOT_INTERPRETED_HEADLINE", "NOT_INTERPRETED_DETAIL", "Not interpreted"),
    ("CONFIDENCE_HEADLINE", "CONFIDENCE_DETAIL", "Reading confidence"),
]


def parse_agent_pointers(text):
    """Parses the LABEL: value lines build_prompt asks for into an
    ordered list of {"headline", "detail"} dicts for the UI's
    pointers-then-detail panel. This is the single place that response
    contract is read back out -- the UI never re-parses response text
    on its own.

    Never raises and never silently drops content: any recognized
    LABEL: line is used regardless of order or extra surrounding text;
    if nothing recognizable is found (a live response that ignored the
    format, or any other unexpected shape), the whole raw response is
    shown as one pointer rather than hidden -- honesty about what's
    measured must survive a formatting miss, not disappear behind it.
    """
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        value = value.strip()
        if value:
            fields[key] = value

    pointers = []
    for headline_key, detail_key, fallback_label in _POINTER_FIELDS:
        headline = fields.get(headline_key)
        detail = fields.get(detail_key)
        if headline or detail:
            pointers.append({"headline": headline or fallback_label, "detail": detail or ""})

    if not pointers:
        pointers = [{"headline": "Behavioral read", "detail": text.strip()}]
    return pointers


def stub_agent_call(prompt):
    """Canned, deterministic, local-only response -- same return shape
    the real API call will use, and the SAME 8-line POINTER format
    build_prompt asks a live agent for (see parse_agent_pointers), so
    the UI panel renders identically whether stubbed or live -- no
    special-casing in the display layer. Zero network, zero tokens.
    Picks a plausible-sounding response based on simple keyword
    presence in the prompt so the stub isn't a single fixed string,
    without ever parsing/trusting numbers out of the prompt text (that
    would be fragile and pointless for a stub).

    The first two branches below serve analyze_video.py's whole-video
    report (Step 3B), keyed off the dedicated "WHOLE-VIDEO summary" +
    "MODE=..." marker line that prompt always includes -- distinct from
    the live tool's own per-window branches further down, which key off
    that prompt's different phrasing and are otherwise untouched."""
    if "WHOLE-VIDEO summary" in prompt and "MODE=B_UNCALIBRATED" in prompt:
        text = (
            "PLEASURE_HEADLINE: Hard to say for sure without a real baseline\n"
            "PLEASURE_DETAIL: This video wasn't calibrated to you personally, so the pleasure-side reading is only a rough, approximate signal, not a validated one.\n"
            "AROUSAL_HEADLINE: Some movement, read loosely\n"
            "AROUSAL_DETAIL: The postural-volatility signal across the video is approximate too, for the same uncalibrated-baseline reason.\n"
            "NOT_INTERPRETED_HEADLINE: Brow and jaw, logged only\n"
            "NOT_INTERPRETED_DETAIL: Those are recorded but never read as part of this reading.\n"
            "CONFIDENCE_HEADLINE: Uncalibrated -- approximate only, not validated\n"
            "CONFIDENCE_DETAIL: This video had no usable neutral baseline (Mode B), so every number here is a relative approximation, not a validated per-person reading."
        )
    elif "WHOLE-VIDEO summary" in prompt and "MODE=A_CALIBRATED" in prompt:
        text = (
            "PLEASURE_HEADLINE: Mostly mild and positive-leaning overall\n"
            "PLEASURE_DETAIL: Across the video, the pleasure-side reading averaged mildly positive against your own calibrated baseline.\n"
            "AROUSAL_HEADLINE: Fairly steady, with brief movement\n"
            "AROUSAL_DETAIL: The postural-volatility signal stayed fairly settled across most of the video, with some short-lived movement.\n"
            "NOT_INTERPRETED_HEADLINE: Brow and jaw, logged only\n"
            "NOT_INTERPRETED_DETAIL: Those are recorded but never read as part of this reading.\n"
            "CONFIDENCE_HEADLINE: Calibrated -- validated for this video\n"
            "CONFIDENCE_DETAIL: This video had a usable neutral baseline (Mode A), so these readings are validated deviations from your own resting signals."
        )
    elif "WINDOW CONFIDENCE: LOW" in prompt:
        text = (
            "PLEASURE_HEADLINE: Hard to get a clear read on you right now\n"
            "PLEASURE_DETAIL: Tracking was unstable this window, so I'd rather flag that than guess at a pleasure-side reading.\n"
            "AROUSAL_HEADLINE: Same story on movement\n"
            "AROUSAL_DETAIL: The postural signal isn't reliable this window either -- not enough stable tracking to read it.\n"
            "NOT_INTERPRETED_HEADLINE: Brow and jaw, logged only\n"
            "NOT_INTERPRETED_DETAIL: Those are recorded but never read as part of this reading.\n"
            "CONFIDENCE_HEADLINE: Low confidence this window\n"
            "CONFIDENCE_DETAIL: Tracking was flagged unstable, so treat this window as a placeholder, not a real read."
        )
    elif "no reading available this window" in prompt:
        text = (
            "PLEASURE_HEADLINE: Only a partial picture right now\n"
            "PLEASURE_DETAIL: One or more signals didn't register this window, so this is only a partial read.\n"
            "AROUSAL_HEADLINE: Nothing standing out\n"
            "AROUSAL_DETAIL: What is registering doesn't show anything notably outside your resting baseline.\n"
            "NOT_INTERPRETED_HEADLINE: Brow and jaw, logged only\n"
            "NOT_INTERPRETED_DETAIL: Those are recorded but never read as part of this reading.\n"
            "CONFIDENCE_HEADLINE: Limited signal this window\n"
            "CONFIDENCE_DETAIL: A single 10-second window with a missing signal -- take this loosely."
        )
    else:
        text = (
            "PLEASURE_HEADLINE: You're leaning slightly positive\n"
            "PLEASURE_DETAIL: The eye-crinkle reading is mild and positive-leaning against your own calibrated baseline.\n"
            "AROUSAL_HEADLINE: Pretty settled physically\n"
            "AROUSAL_DETAIL: Your postural movement is fairly steady this window, nothing volatile.\n"
            "NOT_INTERPRETED_HEADLINE: Brow and jaw, logged only\n"
            "NOT_INTERPRETED_DETAIL: Those are recorded but never read as part of this reading.\n"
            "CONFIDENCE_HEADLINE: One snapshot, not a trend\n"
            "CONFIDENCE_DETAIL: Just this one 10-second window, from two validated signals -- not a broader read on you."
        )
    return {"text": text, "model": STUB_MODEL_NAME, "is_stub": True}


def call_real_agent(prompt):
    """The ONLY place a real Anthropic API call happens. NOT called by
    anything in this file's own pipeline by default (call_agent
    defaults use_stub=True) -- the Stage 3 UI also always passes
    use_stub=True explicitly, per its own "HARD CONSTRAINT, never flip
    this in this file" comment at its call site. This function exists
    so that swap point and test_live_agent.py have somewhere to call.

    Key handling: read from os.environ ONLY, via the SDK's own default
    credential resolution (bare `anthropic.Anthropic()`) -- never
    hardcoded, never logged, never put in a prompt, never written to
    any file. The explicit presence check below exists only to raise a
    clear, actionable error before the SDK's own (less specific) one,
    since a missing key is the most likely first-run failure.

    Lazy-imports the anthropic SDK so this module has zero hard
    dependency on it while stubbed.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Real agent calls require it in the environment (or a .env file "
            "loaded before this runs) -- never hardcode a key in source."
        )
    import anthropic  # lazy import: only required if this path is ever actually used

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment; key itself never touches this file
    response = client.messages.create(
        model=REAL_MODEL_NAME,
        max_tokens=300,
        # This is a short, single-purpose descriptive task (2-4 plain
        # sentences, rules already fully specified in the prompt) that
        # will eventually run every ~10s in the live UI -- not a
        # reasoning task. Claude Sonnet 5 runs WITH adaptive thinking by
        # default when `thinking` is omitted (a behavior change from
        # Sonnet 4.6, which ran thinking-off by omission) -- left
        # implicit, that silently adds thinking-token cost/latency to
        # every call for no benefit here. Disabled explicitly rather
        # than relying on the new default.
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((block.text for block in response.content if block.type == "text"), "")
    return {"text": text, "model": response.model, "is_stub": False}


def call_agent(prompt, use_stub=True):
    """Single isolated call-point. Swapping stub -> real API is exactly
    this: change `use_stub=True` to `use_stub=False` (or flip the
    default) -- nothing else in the pipeline (prompt building, response
    parsing, logging) needs to change, since both branches return the
    same {"text", "model", "is_stub"} shape."""
    if use_stub:
        return stub_agent_call(prompt)
    return call_real_agent(prompt)


def log_agent_exchange(prompt, response, session_id, person_label, window_summary):
    """Every request/response pair, logged to its own versioned JSONL
    record type -- this seeds Track B (master doc). is_stub is stamped
    on every record so stub-testing exchanges can never be silently
    mistaken for real agent output downstream."""
    os.makedirs(LOG_DIR, exist_ok=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "agent_exchange",
        "exchange_id": str(uuid.uuid4()),
        "session_id": session_id,
        "person_label": person_label,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "window_start_monotonic": window_summary.get("window_start_monotonic"),
        "window_end_monotonic": window_summary.get("window_end_monotonic"),
        "prompt": prompt,
        "response_text": response["text"],
        "model": response["model"],
        "is_stub": response["is_stub"],
    }
    with open(AGENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def run_stage2_on_window(window_summary, calibration_reference, session_id, person_label, use_stub=True):
    """Orchestrates the full Stage 2 pass for one window: z-score ->
    prompt -> agent call -> log -> return. This is the function
    everything else (a future live loop, or this file's own test) calls."""
    window_z = zscore_window(window_summary, calibration_reference)
    low_confidence = window_summary.get("window_quality", {}).get("low_confidence", False)
    quality_reasons = window_summary.get("window_quality", {}).get("reasons", [])

    prompt = build_prompt(window_z, person_label, low_confidence, quality_reasons)
    response = call_agent(prompt, use_stub=use_stub)
    logged_record = log_agent_exchange(prompt, response, session_id, person_label, window_summary)

    return {"window_z": window_z, "prompt": prompt, "response": response, "logged_record": logged_record}


if __name__ == "__main__":
    import glob

    # offline, zero-token self-test: pull one real window_summary +
    # calibration_complete pair from an existing production session log
    session_paths = sorted(glob.glob(os.path.join(LOG_DIR, "session_*.jsonl")), key=os.path.getmtime, reverse=True)
    chosen = None
    for path in session_paths:
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        types = set(r.get("record_type") for r in records)
        if "window_summary" in types and "calibration_complete" in types:
            chosen = (path, records)
            break

    if chosen is None:
        print("No existing session log with both window_summary and calibration_complete found -- nothing to test against.")
    else:
        path, records = chosen
        calibration_record = next(r for r in records if r["record_type"] == "calibration_complete")
        window_record = next(r for r in records if r["record_type"] == "window_summary")

        print(f"Testing against: {path}")
        result = run_stage2_on_window(
            window_summary=window_record,
            calibration_reference=calibration_record["reference"],
            session_id=window_record["session_id"],
            person_label=window_record.get("person_label"),
            use_stub=True,
        )

        print("\n--- window_z ---")
        print(json.dumps(result["window_z"], indent=2))
        print("\n--- prompt ---")
        print(result["prompt"])
        print("\n--- stub response ---")
        print(json.dumps(result["response"], indent=2))
        print(f"\nLogged to {AGENT_LOG_PATH}")
