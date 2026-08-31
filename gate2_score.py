"""
gate2_score.py — standalone, READ-ONLY Gate 2 scoring script.

Reads gate2_trials.jsonl, captured earlier by the dumb capture tool
(stage1_step9_gate2_capture.py, which computes no pass/fail itself),
and applies a FIXED, pre-registered scoring rule. This script is
written AFTER all capture is done, against data it never influenced --
that separation is the whole point (CLAUDE.md GATE 2: "never let the
pipeline score itself"). It does NOT import, call, or modify the
capture tool, and does NOT modify the input file.

FROZEN SCORING RULE (implemented exactly, nothing added/dropped):

  Unit = one ACCEPTED block attempt for one person (accepted == true
  only; rejected/superseded redo attempts are excluded and reported).

  Expression -> vector -> expected sign of the windowed z-scored avg,
  relative to that person's own calibration neutral:
    smile       -> V_es, expected POSITIVE
    furrow      -> V_bf, expected POSITIVE
    concentrate -> V_bf, expected POSITIVE

  Facial per-trial PASS (V_bf, V_es) requires BOTH:
    1. windowed avg has the expected sign
    2. |windowed avg| >= 1.0 (z-units)
  Otherwise FAIL. A low-confidence window makes the trial UNSCOREABLE
  (excluded from the denominator, not counted as fail).

  V_pd is a PER-PERSON CONTRAST, not per-trial: compare each person's
  accepted fidget window against their accepted sit-still window.
  PASSES if BOTH:
    1. v_pd(fidget avg) > v_pd(sit-still avg)
    2. v_pd(fidget avg) - v_pd(sit-still avg) >= 1.0
  Either window low-confidence (or missing) -> that person's V_pd is
  UNSCOREABLE.

  Pools (never blended -- reported strictly per-vector):
    V_bf = all accepted furrow trials + all accepted concentrate trials
    V_es = all accepted smile trials
    V_pd = per-person contrast, one result per person
    V_jc = NOT scored

  Flagged-baseline handling: a person flagged (in that trial's
  calibration_contamination_flags.drifted_vectors) for a given vector
  is "flagged" for that vector only. Every per-vector score is reported
  TWICE: all-people, and excluding people flagged for THAT vector. The
  gap between the two is a finding to report, not an error to resolve.

  Any missing field needed to score a trial -> that trial is
  unscoreable (reported), never guessed.
"""

import json
import os
from collections import defaultdict

TRIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "gate2_trials.jsonl")

# Study filter (NOT part of the scoring rule -- a selection of WHICH records
# are scored, applied before any rule logic runs). The original P01..P06
# sessions used an unclear furrow instruction (participants performed the
# wrong gesture; see VBF_DIAGNOSTIC_REPORT.md and its follow-up), so they were
# re-captured under a corrected instruction as the B-series. Both the pilot
# and the B-series stay in gate2_trials.jsonl -- only the B-series is scored
# as the study. Everything below this filter is the frozen rule, unchanged.
B_SERIES_LABELS = {"P01B", "P02B", "P03B", "P04B", "P05B", "P06B"}

Z_THRESHOLD = 1.0
GATE2_BAR_FRACTION = 0.75
GATE2_MIN_PEOPLE = 8

EXPRESSION_VECTOR = {
    "smile": "v_es",
    "furrow": "v_bf",
    "concentrate": "v_bf",
}
EXPECTED_SIGN_POSITIVE = {"v_es", "v_bf"}  # both expected POSITIVE per the frozen rule


def load_trials(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_flagged(trial, vector):
    if trial is None:
        return False
    flags = trial.get("calibration_contamination_flags")
    if not flags:
        return False
    return any(d.get("vector") == vector for d in flags.get("drifted_vectors", []))


def get_window_avg(trial, vector):
    wz = trial.get("window_z")
    if not wz or vector not in wz:
        return None
    return wz[vector].get("avg")


def get_low_confidence(trial):
    wc = trial.get("window_confidence")
    if wc is None:
        return None
    return wc.get("low_confidence")


def score_facial_trial(trial, vector):
    """Returns 'pass' / 'fail' / 'unscoreable_low_confidence' /
    'unscoreable_missing_field'. Never guesses on a missing field."""
    low_conf = get_low_confidence(trial)
    if low_conf is None:
        return "unscoreable_missing_field"
    if low_conf:
        return "unscoreable_low_confidence"

    avg = get_window_avg(trial, vector)
    if avg is None:
        return "unscoreable_missing_field"

    sign_ok = avg > 0  # expected sign is POSITIVE for both v_bf and v_es in this rule
    magnitude_ok = abs(avg) >= Z_THRESHOLD
    return "pass" if (sign_ok and magnitude_ok) else "fail"


def score_pd_person(fidget_t, still_t):
    flagged = is_flagged(fidget_t, "v_pd") or is_flagged(still_t, "v_pd")

    if fidget_t is None or still_t is None:
        return "unscoreable_missing_field", flagged, None, None

    low_fidget = get_low_confidence(fidget_t)
    low_still = get_low_confidence(still_t)
    if low_fidget is None or low_still is None:
        return "unscoreable_missing_field", flagged, None, None
    if low_fidget or low_still:
        return "unscoreable_low_confidence", flagged, None, None

    avg_fidget = get_window_avg(fidget_t, "v_pd")
    avg_still = get_window_avg(still_t, "v_pd")
    if avg_fidget is None or avg_still is None:
        return "unscoreable_missing_field", flagged, avg_fidget, avg_still

    diff = avg_fidget - avg_still
    passed = (avg_fidget > avg_still) and (diff >= Z_THRESHOLD)
    return ("pass" if passed else "fail"), flagged, avg_fidget, avg_still


def summarize(rows, exclude_flagged, unit_key):
    if exclude_flagged:
        rows = [r for r in rows if not r["flagged"]]
    passes = sum(1 for r in rows if r["result"] == "pass")
    fails = sum(1 for r in rows if r["result"] == "fail")
    excl_low_conf = sum(1 for r in rows if r["result"] == "unscoreable_low_confidence")
    excl_missing = sum(1 for r in rows if r["result"] == "unscoreable_missing_field")
    scoreable = passes + fails
    score = (passes / scoreable) if scoreable else None
    return {
        "passes": passes,
        "fails": fails,
        "scoreable": scoreable,
        "score": score,
        "excluded_low_confidence": excl_low_conf,
        "excluded_missing": excl_missing,
        unit_key: len(rows),
    }


def fmt_score(s):
    return "n/a (0 scoreable)" if s is None else f"{s * 100:.1f}%"


def main():
    all_trials = load_trials(TRIALS_PATH)
    print(f"Read {len(all_trials)} record(s) from {TRIALS_PATH}\n")

    # --- study filter: score ONLY the B-series (see B_SERIES_LABELS above) ---
    # Applied before any rule logic -- everything from here down is the
    # frozen scoring rule, completely unchanged, operating on a pre-filtered
    # record set rather than knowing about B-series at all.
    trials = [t for t in all_trials if t.get("person_label") in B_SERIES_LABELS]
    excluded_non_b = [t for t in all_trials if t.get("person_label") not in B_SERIES_LABELS]
    excluded_non_b_labels = sorted(set(t.get("person_label") for t in excluded_non_b))
    scored_labels_present = sorted(set(t.get("person_label") for t in trials))

    print("=" * 78)
    print("STUDY FILTER: scoring the B-series (corrected furrow instruction) only")
    print("=" * 78)
    print(f"B-series labels defined: {sorted(B_SERIES_LABELS)}")
    print(f"B-series labels present in data: {scored_labels_present}")
    print(f"B-series records scored: {len(trials)}")
    print(f"Excluded as non-B (original pilot sessions, not scored): {len(excluded_non_b)} record(s)")
    print(f"Excluded person_labels: {excluded_non_b_labels}\n")

    accepted = [t for t in trials if t.get("accepted") is True]
    excluded_not_accepted = len(trials) - len(accepted)

    people = sorted(set(t["person_label"] for t in trials if "person_label" in t))
    n_people = len(people)

    print(f"Excluded (accepted == false, redo/superseded attempts): {excluded_not_accepted}")
    print(f"Distinct people (person_label): n = {n_people}  -> {people}\n")

    # --- facial per-trial scoring: V_bf (furrow+concentrate), V_es (smile) ---
    facial_rows = {"v_bf": [], "v_es": []}
    for t in accepted:
        vector = EXPRESSION_VECTOR.get(t.get("commanded_label"))
        if vector is None:
            continue  # sit-still / fidget go to the V_pd contrast, not here
        facial_rows[vector].append(
            {
                "person": t.get("person_label"),
                "commanded_label": t.get("commanded_label"),
                "avg": get_window_avg(t, vector),
                "result": score_facial_trial(t, vector),
                "flagged": is_flagged(t, vector),
            }
        )

    # --- V_pd per-person contrast ---
    accepted_by_person_block = defaultdict(dict)
    for t in accepted:
        accepted_by_person_block[t.get("person_label")][t.get("commanded_label")] = t

    pd_rows = []
    for person in people:
        blocks = accepted_by_person_block.get(person, {})
        fidget_t = blocks.get("fidget")
        still_t = blocks.get("sit-still")
        result, flagged, avg_fidget, avg_still = score_pd_person(fidget_t, still_t)
        pd_rows.append(
            {"person": person, "result": result, "flagged": flagged, "avg_fidget": avg_fidget, "avg_still": avg_still}
        )

    # --- report ---
    print("=" * 78)
    print("PER-VECTOR SCORES (reported separately, never blended)")
    print("=" * 78)

    for vector, label in (("v_bf", "V_bf (furrow + concentrate trials)"), ("v_es", "V_es (smile trials)")):
        all_s = summarize(facial_rows[vector], exclude_flagged=False, unit_key="n_trials")
        excl_s = summarize(facial_rows[vector], exclude_flagged=True, unit_key="n_trials")
        print(f"\n{label}")
        print(
            f"  all-people:         {fmt_score(all_s['score'])}  "
            f"({all_s['passes']} pass / {all_s['scoreable']} scoreable)  "
            f"excluded: {all_s['excluded_low_confidence']} low-confidence, {all_s['excluded_missing']} missing-field"
        )
        print(
            f"  excluding-flagged:  {fmt_score(excl_s['score'])}  "
            f"({excl_s['passes']} pass / {excl_s['scoreable']} scoreable)  "
            f"excluded: {excl_s['excluded_low_confidence']} low-confidence, {excl_s['excluded_missing']} missing-field"
        )

    all_pd = summarize(pd_rows, exclude_flagged=False, unit_key="n_people")
    excl_pd = summarize(pd_rows, exclude_flagged=True, unit_key="n_people")
    print("\nV_pd (per-person fidget-vs-sit-still contrast)")
    print(
        f"  all-people:         {fmt_score(all_pd['score'])}  "
        f"({all_pd['passes']} pass / {all_pd['scoreable']} scoreable people)  "
        f"excluded: {all_pd['excluded_low_confidence']} low-confidence, {all_pd['excluded_missing']} missing-field"
    )
    print(
        f"  excluding-flagged:  {fmt_score(excl_pd['score'])}  "
        f"({excl_pd['passes']} pass / {excl_pd['scoreable']} scoreable people)  "
        f"excluded: {excl_pd['excluded_low_confidence']} low-confidence, {excl_pd['excluded_missing']} missing-field"
    )

    print("\nV_jc: NOT SCORED (logged-only vector, excluded from Gate 2 by design)")

    print("\n" + "=" * 78)
    print(f"Gate 2 bar = >={GATE2_BAR_FRACTION*100:.0f}% per vector, >={GATE2_MIN_PEOPLE} people. Current n = {n_people}.")
    if n_people < GATE2_MIN_PEOPLE:
        print(f"n={n_people} < {GATE2_MIN_PEOPLE} -- this is a PILOT-SCALE DIRECTIONAL RESULT (n={n_people}), NOT a passed or failed Gate 2.")
    print("=" * 78)

    # --- per-person, per-vector audit table ---
    print("\n" + "=" * 78)
    print("PER-PERSON, PER-VECTOR DETAIL (for hand audit)")
    print("=" * 78)
    furrow_by_person = {r["person"]: r for r in facial_rows["v_bf"] if r["commanded_label"] == "furrow"}
    concentrate_by_person = {r["person"]: r for r in facial_rows["v_bf"] if r["commanded_label"] == "concentrate"}
    smile_by_person = {r["person"]: r for r in facial_rows["v_es"]}
    pd_by_person = {r["person"]: r for r in pd_rows}

    for person in people:
        print(f"\n{person}:")
        for label, row_map in (("furrow (V_bf)", furrow_by_person), ("concentrate (V_bf)", concentrate_by_person), ("smile (V_es)", smile_by_person)):
            r = row_map.get(person)
            if r is None:
                print(f"  {label:20s} no accepted trial")
            else:
                avg_str = "n/a" if r["avg"] is None else f"{r['avg']:+.3f}"
                flag_str = " [FLAGGED]" if r["flagged"] else ""
                print(f"  {label:20s} avg={avg_str:>8s}  -> {r['result']}{flag_str}")
        pr = pd_by_person.get(person)
        if pr is None:
            print("  V_pd contrast        no data")
        else:
            fid = "n/a" if pr["avg_fidget"] is None else f"{pr['avg_fidget']:+.3f}"
            sti = "n/a" if pr["avg_still"] is None else f"{pr['avg_still']:+.3f}"
            flag_str = " [FLAGGED]" if pr["flagged"] else ""
            print(f"  {'V_pd contrast':20s} fidget={fid:>8s}  sit-still={sti:>8s}  -> {pr['result']}{flag_str}")


if __name__ == "__main__":
    main()
