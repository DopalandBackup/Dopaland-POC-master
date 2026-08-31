"""
gate2_report.py — writes the Gate 2 B-series scoring result to
GATE2_RESULT_REPORT.md. READ-ONLY over gate2_trials.jsonl.

Does NOT reimplement any scoring rule -- imports every scoring function
directly from gate2_score.py (load_trials, score_facial_trial,
score_pd_person, summarize, is_flagged, get_window_avg,
get_low_confidence, B_SERIES_LABELS, EXPRESSION_VECTOR, Z_THRESHOLD,
GATE2_BAR_FRACTION, GATE2_MIN_PEOPLE) and only adds: (1) writing the
result to a file instead of stdout, and (2) an additional descriptive
breakdown of the V_bf pool into furrow-only / concentrate-only, using
the same summarize() function on a sub-filtered set of the same rows
gate2_score.py already computes. The pooled V_bf score, the frozen
rule, and every threshold are untouched.
"""

import os
from datetime import datetime, timezone

from gate2_score import (
    TRIALS_PATH,
    B_SERIES_LABELS,
    EXPRESSION_VECTOR,
    Z_THRESHOLD,
    GATE2_BAR_FRACTION,
    GATE2_MIN_PEOPLE,
    load_trials,
    is_flagged,
    get_window_avg,
    get_low_confidence,
    score_facial_trial,
    score_pd_person,
    summarize,
    fmt_score,
)

REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GATE2_RESULT_REPORT.md")


def build_report():
    lines = []

    all_trials = load_trials(TRIALS_PATH)
    trials = [t for t in all_trials if t.get("person_label") in B_SERIES_LABELS]
    excluded_non_b = [t for t in all_trials if t.get("person_label") not in B_SERIES_LABELS]
    excluded_non_b_labels = sorted(set(t.get("person_label") for t in excluded_non_b))
    scored_labels_present = sorted(set(t.get("person_label") for t in trials))

    accepted = [t for t in trials if t.get("accepted") is True]
    excluded_not_accepted = len(trials) - len(accepted)
    people = sorted(set(t["person_label"] for t in trials if "person_label" in t))
    n_people = len(people)

    # --- facial per-trial scoring (identical loop to gate2_score.main()) ---
    facial_rows = {"v_bf": [], "v_es": []}
    for t in accepted:
        vector = EXPRESSION_VECTOR.get(t.get("commanded_label"))
        if vector is None:
            continue
        facial_rows[vector].append(
            {
                "person": t.get("person_label"),
                "commanded_label": t.get("commanded_label"),
                "avg": get_window_avg(t, vector),
                "result": score_facial_trial(t, vector),
                "flagged": is_flagged(t, vector),
            }
        )

    # --- V_pd per-person contrast (identical loop to gate2_score.main()) ---
    from collections import defaultdict

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

    # ======================= REPORT ASSEMBLY =======================
    lines.append("# Gate 2 Result Report (B-series)")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Source: `{TRIALS_PATH}` (read-only)")
    lines.append(f"Scorer: `gate2_score.py` (frozen rule, unchanged)")
    lines.append("")

    # --- 1. Header ---
    lines.append("## 1. Header")
    lines.append("")
    lines.append(f"- B-series labels defined: {sorted(B_SERIES_LABELS)}")
    lines.append(f"- B-series labels present in data: {scored_labels_present}")
    lines.append(f"- B-series records scored: {len(trials)}")
    lines.append(f"- Excluded as non-B (original pilot sessions, not scored): {len(excluded_non_b)} record(s)")
    lines.append(f"- Excluded person_labels: {excluded_non_b_labels}")
    lines.append(f"- Excluded (accepted == false, redo/superseded attempts): {excluded_not_accepted}")
    lines.append(f"- Distinct people (person_label): n = {n_people} -> {people}")
    lines.append("")
    lines.append(f"**Gate 2 bar = >={GATE2_BAR_FRACTION*100:.0f}% per vector, >={GATE2_MIN_PEOPLE} people. Current n = {n_people}.**")
    if n_people < GATE2_MIN_PEOPLE:
        lines.append(
            f"n={n_people} < {GATE2_MIN_PEOPLE} -- this is a **PILOT-SCALE DIRECTIONAL RESULT (n={n_people})**, "
            f"NOT a passed or failed Gate 2."
        )
    lines.append("")

    # --- 2. Per-vector results ---
    lines.append("## 2. Per-vector results")
    lines.append("")

    for vector, label in (("v_bf", "V_bf (furrow + concentrate pooled)"), ("v_es", "V_es (smile)")):
        all_s = summarize(facial_rows[vector], exclude_flagged=False, unit_key="n_trials")
        excl_s = summarize(facial_rows[vector], exclude_flagged=True, unit_key="n_trials")
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Reporting | Score | Passes/Scoreable | Excluded (low-confidence) | Excluded (missing-field) |")
        lines.append("|---|---|---|---|---|")
        lines.append(
            f"| all-people | {fmt_score(all_s['score'])} | {all_s['passes']}/{all_s['scoreable']} | "
            f"{all_s['excluded_low_confidence']} | {all_s['excluded_missing']} |"
        )
        lines.append(
            f"| excluding-flagged | {fmt_score(excl_s['score'])} | {excl_s['passes']}/{excl_s['scoreable']} | "
            f"{excl_s['excluded_low_confidence']} | {excl_s['excluded_missing']} |"
        )
        lines.append("")

    all_pd = summarize(pd_rows, exclude_flagged=False, unit_key="n_people")
    excl_pd = summarize(pd_rows, exclude_flagged=True, unit_key="n_people")
    lines.append("### V_pd (per-person fidget-vs-sit-still contrast)")
    lines.append("")
    lines.append("Denominator is PEOPLE, not trials.")
    lines.append("")
    lines.append("| Reporting | Score | Passes/Scoreable (people) | Excluded (low-confidence) | Excluded (missing-field) |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| all-people | {fmt_score(all_pd['score'])} | {all_pd['passes']}/{all_pd['scoreable']} | "
        f"{all_pd['excluded_low_confidence']} | {all_pd['excluded_missing']} |"
    )
    lines.append(
        f"| excluding-flagged | {fmt_score(excl_pd['score'])} | {excl_pd['passes']}/{excl_pd['scoreable']} | "
        f"{excl_pd['excluded_low_confidence']} | {excl_pd['excluded_missing']} |"
    )
    lines.append("")
    lines.append("### V_jc")
    lines.append("")
    lines.append("NOT SCORED (logged-only vector, excluded from Gate 2 by design).")
    lines.append("")

    # --- 3. Full audit table ---
    lines.append("## 3. Full audit table")
    lines.append("")
    lines.append("### Facial trials (V_bf, V_es)")
    lines.append("")
    lines.append("| person_label | commanded_label | vector | windowed z avg | pass/fail | reason | flagged |")
    lines.append("|---|---|---|---|---|---|---|")

    def reason_for(row):
        if row["result"] == "pass":
            return "correct sign, |z|>=1.0"
        if row["result"] == "unscoreable_low_confidence":
            return "low-confidence window"
        if row["result"] == "unscoreable_missing_field":
            return "missing field"
        # fail: distinguish wrong-sign vs below-threshold
        avg = row["avg"]
        if avg is None:
            return "missing field"
        if avg <= 0:
            return "wrong sign"
        return "below |z|>=1.0"

    audit_rows = []
    for vector in ("v_bf", "v_es"):
        for row in facial_rows[vector]:
            audit_rows.append((row["person"], row["commanded_label"], vector, row))
    audit_rows.sort(key=lambda r: (r[0], r[1], r[2]))

    for person, label, vector, row in audit_rows:
        avg_str = "n/a" if row["avg"] is None else f"{row['avg']:+.4f}"
        pf = "pass" if row["result"] == "pass" else ("fail" if row["result"] == "fail" else "unscoreable")
        flag_str = "yes" if row["flagged"] else ""
        lines.append(f"| {person} | {label} | {vector} | {avg_str} | {pf} | {reason_for(row)} | {flag_str} |")
    lines.append("")

    lines.append("### V_pd per-person contrast")
    lines.append("")
    lines.append("| person_label | fidget avg | sit-still avg | difference | pass/fail | reason | flagged |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in sorted(pd_rows, key=lambda r: r["person"]):
        fid = "n/a" if row["avg_fidget"] is None else f"{row['avg_fidget']:+.4f}"
        sti = "n/a" if row["avg_still"] is None else f"{row['avg_still']:+.4f}"
        if row["avg_fidget"] is not None and row["avg_still"] is not None:
            diff = row["avg_fidget"] - row["avg_still"]
            diff_str = f"{diff:+.4f}"
        else:
            diff_str = "n/a"
        pf = "pass" if row["result"] == "pass" else ("fail" if row["result"] == "fail" else "unscoreable")
        if row["result"] == "unscoreable_low_confidence":
            reason = "low-confidence window (fidget and/or sit-still)"
        elif row["result"] == "unscoreable_missing_field":
            reason = "missing fidget and/or sit-still trial"
        elif row["result"] == "pass":
            reason = "fidget > sit-still AND difference >= 1.0"
        else:
            reason = "fidget <= sit-still, or difference < 1.0"
        flag_str = "yes" if row["flagged"] else ""
        lines.append(f"| {row['person']} | {fid} | {sti} | {diff_str} | {pf} | {reason} | {flag_str} |")
    lines.append("")

    # --- 4. V_bf pool split (descriptive only) ---
    lines.append("## 4. V_bf pool split (descriptive breakdown, not a rule change)")
    lines.append("")
    lines.append("The pooled V_bf score above (furrow + concentrate combined) is the scored, frozen-rule number. "
                  "The two contributing expressions are additionally broken out separately below for readability; "
                  "this does not change the pooled score.")
    lines.append("")

    for expr in ("furrow", "concentrate"):
        sub_rows = [r for r in facial_rows["v_bf"] if r["commanded_label"] == expr]
        all_sub = summarize(sub_rows, exclude_flagged=False, unit_key="n_trials")
        excl_sub = summarize(sub_rows, exclude_flagged=True, unit_key="n_trials")
        lines.append(f"### V_bf -- {expr} only")
        lines.append("")
        lines.append("| Reporting | Score | Passes/Scoreable | Excluded (low-confidence) | Excluded (missing-field) |")
        lines.append("|---|---|---|---|---|")
        lines.append(
            f"| all-people | {fmt_score(all_sub['score'])} | {all_sub['passes']}/{all_sub['scoreable']} | "
            f"{all_sub['excluded_low_confidence']} | {all_sub['excluded_missing']} |"
        )
        lines.append(
            f"| excluding-flagged | {fmt_score(excl_sub['score'])} | {excl_sub['passes']}/{excl_sub['scoreable']} | "
            f"{excl_sub['excluded_low_confidence']} | {excl_sub['excluded_missing']} |"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    report_text = build_report()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Wrote {REPORT_PATH} ({len(report_text)} chars)")


if __name__ == "__main__":
    main()
