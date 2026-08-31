# Gate 2 Result Report (B-series)

Generated: 2026-07-15T15:45:56.567879+00:00
Source: `C:\Dopaland-POC\logs\gate2_trials.jsonl` (read-only)
Scorer: `gate2_score.py` (frozen rule, unchanged)

## 1. Header

- B-series labels defined: ['P01B', 'P02B', 'P03B', 'P04B', 'P05B', 'P06B']
- B-series labels present in data: ['P01B', 'P02B', 'P03B', 'P04B', 'P05B', 'P06B']
- B-series records scored: 35
- Excluded as non-B (original pilot sessions, not scored): 46 record(s)
- Excluded person_labels: ['P01', 'P01C', 'P05', 'P06', 'p01', 'p02']
- Excluded (accepted == false, redo/superseded attempts): 0
- Distinct people (person_label): n = 6 -> ['P01B', 'P02B', 'P03B', 'P04B', 'P05B', 'P06B']

**Gate 2 bar = >=75% per vector, >=8 people. Current n = 6.**
n=6 < 8 -- this is a **PILOT-SCALE DIRECTIONAL RESULT (n=6)**, NOT a passed or failed Gate 2.

## 2. Per-vector results

### V_bf (furrow + concentrate pooled)

| Reporting | Score | Passes/Scoreable | Excluded (low-confidence) | Excluded (missing-field) |
|---|---|---|---|---|
| all-people | 35.7% | 5/14 | 0 | 0 |
| excluding-flagged | 33.3% | 4/12 | 0 | 0 |

### V_es (smile)

| Reporting | Score | Passes/Scoreable | Excluded (low-confidence) | Excluded (missing-field) |
|---|---|---|---|---|
| all-people | 71.4% | 5/7 | 0 | 0 |
| excluding-flagged | 60.0% | 3/5 | 0 | 0 |

### V_pd (per-person fidget-vs-sit-still contrast)

Denominator is PEOPLE, not trials.

| Reporting | Score | Passes/Scoreable (people) | Excluded (low-confidence) | Excluded (missing-field) |
|---|---|---|---|---|
| all-people | 100.0% | 5/5 | 1 | 0 |
| excluding-flagged | 100.0% | 3/3 | 0 | 0 |

### V_jc

NOT SCORED (logged-only vector, excluded from Gate 2 by design).

## 3. Full audit table

### Facial trials (V_bf, V_es)

| person_label | commanded_label | vector | windowed z avg | pass/fail | reason | flagged |
|---|---|---|---|---|---|---|
| P01B | concentrate | v_bf | +2.9492 | pass | correct sign, |z|>=1.0 |  |
| P01B | concentrate | v_bf | +3.1900 | pass | correct sign, |z|>=1.0 |  |
| P01B | furrow | v_bf | -1.2884 | fail | wrong sign |  |
| P01B | furrow | v_bf | -3.9615 | fail | wrong sign |  |
| P01B | smile | v_es | +3.4673 | pass | correct sign, |z|>=1.0 | yes |
| P01B | smile | v_es | +4.9766 | pass | correct sign, |z|>=1.0 |  |
| P02B | concentrate | v_bf | +3.9037 | pass | correct sign, |z|>=1.0 |  |
| P02B | furrow | v_bf | -4.0155 | fail | wrong sign |  |
| P02B | smile | v_es | +6.6402 | pass | correct sign, |z|>=1.0 |  |
| P03B | concentrate | v_bf | -1.2209 | fail | wrong sign |  |
| P03B | furrow | v_bf | -6.2548 | fail | wrong sign |  |
| P03B | smile | v_es | +2.6760 | pass | correct sign, |z|>=1.0 |  |
| P04B | concentrate | v_bf | +1.0504 | pass | correct sign, |z|>=1.0 |  |
| P04B | furrow | v_bf | +0.4167 | fail | below |z|>=1.0 |  |
| P04B | smile | v_es | +0.9926 | fail | below |z|>=1.0 |  |
| P05B | concentrate | v_bf | +3.0194 | pass | correct sign, |z|>=1.0 | yes |
| P05B | furrow | v_bf | -1.4921 | fail | wrong sign | yes |
| P05B | smile | v_es | +7.1744 | pass | correct sign, |z|>=1.0 | yes |
| P06B | concentrate | v_bf | -0.3909 | fail | wrong sign |  |
| P06B | furrow | v_bf | -1.5992 | fail | wrong sign |  |
| P06B | smile | v_es | +0.8336 | fail | below |z|>=1.0 |  |

### V_pd per-person contrast

| person_label | fidget avg | sit-still avg | difference | pass/fail | reason | flagged |
|---|---|---|---|---|---|---|
| P01B | +42.3947 | +1.5109 | +40.8838 | pass | fidget > sit-still AND difference >= 1.0 |  |
| P02B | +35.6365 | -0.0547 | +35.6912 | pass | fidget > sit-still AND difference >= 1.0 | yes |
| P03B | +78.1543 | +2.7611 | +75.3932 | pass | fidget > sit-still AND difference >= 1.0 | yes |
| P04B | +6.9409 | +1.0470 | +5.8939 | pass | fidget > sit-still AND difference >= 1.0 |  |
| P05B | +23.9575 | +2.6917 | +21.2658 | pass | fidget > sit-still AND difference >= 1.0 |  |
| P06B | n/a | n/a | n/a | unscoreable | low-confidence window (fidget and/or sit-still) | yes |

## 4. V_bf pool split (descriptive breakdown, not a rule change)

The pooled V_bf score above (furrow + concentrate combined) is the scored, frozen-rule number. The two contributing expressions are additionally broken out separately below for readability; this does not change the pooled score.

### V_bf -- furrow only

| Reporting | Score | Passes/Scoreable | Excluded (low-confidence) | Excluded (missing-field) |
|---|---|---|---|---|
| all-people | 0.0% | 0/7 | 0 | 0 |
| excluding-flagged | 0.0% | 0/6 | 0 | 0 |

### V_bf -- concentrate only

| Reporting | Score | Passes/Scoreable | Excluded (low-confidence) | Excluded (missing-field) |
|---|---|---|---|---|
| all-people | 71.4% | 5/7 | 0 | 0 |
| excluding-flagged | 66.7% | 4/6 | 0 | 0 |

