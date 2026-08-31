"""
One-off data-hygiene script: relabel person_label in gate2_trials.jsonl
by session_id, per an explicit mapping. Does NOT touch the original file.
Writes gate2_trials_backup.jsonl (untouched copy) and
gate2_trials_relabeled.jsonl (only person_label changed, matched by
session_id). Not part of the shipped pipeline.
"""

import json
import shutil
from collections import defaultdict

SRC = "logs/gate2_trials.jsonl"
BACKUP = "logs/gate2_trials_backup.jsonl"
OUT = "logs/gate2_trials_relabeled.jsonl"

MAPPING = {
    "b09ceab0-882a-4917-96ba-90f119acf776": "P01",
    "4642e551-6630-4b2f-ab60-43572474e96c": "P02",
    "0636539a-6879-4b76-b799-516e0e7eecef": "P03",
    "768ea0c6-8333-44e8-b015-1bedf260849f": "P04",
    "fc701cbf-4dfd-4f47-87ca-313bb9577fda": "P05",
}

# 1. Untouched backup (byte-for-byte copy) -- done BEFORE any read/write of SRC.
shutil.copyfile(SRC, BACKUP)

with open(SRC, "r", encoding="utf-8") as f:
    original_lines = f.readlines()

records = [json.loads(line) for line in original_lines]

before_counts = defaultdict(int)
for r in records:
    before_counts[r["session_id"]] += 1

unmapped_session_ids = set()
relabeled_lines = []
after_counts = defaultdict(int)
field_diffs = []  # (line_index, differing_field_names) for any non-person_label diff

for idx, (line, rec) in enumerate(zip(original_lines, records)):
    sid = rec["session_id"]
    new_rec = dict(rec)  # shallow copy; only person_label is reassigned below

    if sid in MAPPING:
        new_rec["person_label"] = MAPPING[sid]
    else:
        unmapped_session_ids.add(sid)
        # leave unchanged, per instructions

    after_counts[new_rec["session_id"]] += 1

    # verify: every field except person_label must be byte-identical
    diffs = [k for k in rec if k != "person_label" and rec[k] != new_rec[k]]
    diffs += [k for k in new_rec if k not in rec]
    if diffs:
        field_diffs.append((idx, diffs))

    relabeled_lines.append(json.dumps(new_rec) + "\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.writelines(relabeled_lines)

# --- verification report ---
print(f"Original records: {len(records)}")
print(f"Backup written: {BACKUP}")
print(f"Relabeled output written: {OUT}")
print()

print("=== Per-session_id counts (before -> after) ===")
all_sids = set(before_counts) | set(after_counts)
counts_match = True
for sid in sorted(all_sids):
    b, a = before_counts.get(sid, 0), after_counts.get(sid, 0)
    ok = "OK" if b == a else "MISMATCH"
    if b != a:
        counts_match = False
    print(f"  {sid}  before={b}  after={a}  [{ok}]")
print(f"All counts identical: {counts_match}")
print()

print("=== Field-diff check (should be ZERO non-person_label diffs) ===")
if field_diffs:
    print(f"  FOUND {len(field_diffs)} record(s) with unexpected diffs:")
    for idx, diffs in field_diffs:
        print(f"    line {idx}: {diffs}")
else:
    print("  Zero non-person_label diffs across all records. Confirmed.")
print()

print("=== Final mapping report ===")
new_label_counts = defaultdict(lambda: [None, 0])
for new_rec_line in relabeled_lines:
    r = json.loads(new_rec_line)
    key = r["person_label"]
    new_label_counts[key][0] = r["session_id"]
    new_label_counts[key][1] += 1
for label in sorted(new_label_counts):
    sid, n = new_label_counts[label]
    print(f"  {label:6s} <- session_id {sid}  ({n} records)")
print()

print("=== Unmapped session_ids (left unchanged) ===")
if unmapped_session_ids:
    for sid in sorted(unmapped_session_ids):
        print(f"  {sid}  (n={before_counts[sid]} records) -- NOT relabeled, left as-is")
else:
    print("  None -- every session_id in the file was covered by the mapping.")
