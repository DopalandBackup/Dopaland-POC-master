"""
test_live_agent.py -- standalone, single-call verification of the live
Anthropic API path (stage2_personality_agent.call_real_agent).

Makes EXACTLY ONE real API call. This is a manual, human-run smoke
test for verifying the live path is wired correctly before live mode
is ever turned on anywhere else (that is a later, separate step --
the Stage 3 UI stays on the stub by default and is not touched here).
Never invoked by the UI, the processing pipeline, or any automated
path -- run it yourself, on purpose, when you want to check the live
path.

The API key comes ONLY from the ANTHROPIC_API_KEY environment
variable, via stage2_personality_agent.call_real_agent's own
os.environ read (this file never touches the key value itself -- it
only checks whether the variable is present, and never prints, logs,
or otherwise handles the key).

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...        (Windows cmd)
    python test_live_agent.py

    $env:ANTHROPIC_API_KEY = "sk-ant-..."   (PowerShell)
    python test_live_agent.py
"""

import glob
import json
import os
import sys

import stage2_personality_agent as stage2


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(">>> ANTHROPIC_API_KEY is not set in the environment.")
        print("    This test makes one real call to the Anthropic API and needs a key to do that.")
        print("    Set it and re-run, for example:")
        print('        set ANTHROPIC_API_KEY=sk-ant-...          (Windows cmd)')
        print('        $env:ANTHROPIC_API_KEY = "sk-ant-..."     (PowerShell)')
        print("    Exiting -- nothing was called, nothing was logged.")
        return  # clean exit, no crash, no traceback, no key leak

    # Same source stage2_personality_agent.py's own offline stub
    # self-test uses: pull one real window_summary + calibration_complete
    # pair from an existing production session log -- reusing real,
    # already-computed data rather than fabricating a synthetic window.
    session_paths = sorted(
        glob.glob(os.path.join(stage2.LOG_DIR, "session_*.jsonl")),
        key=os.path.getmtime,
        reverse=True,
    )
    chosen = None
    for path in session_paths:
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        types = {r.get("record_type") for r in records}
        if "window_summary" in types and "calibration_complete" in types:
            chosen = (path, records)
            break

    if chosen is None:
        print(">>> No existing session log with both window_summary and calibration_complete records found.")
        print("    Run stage1_step4_vectors.py (or the Stage 3 UI) through at least one calibration")
        print("    and one completed 10s window first, then re-run this test.")
        return

    path, records = chosen
    calibration_record = next(r for r in records if r["record_type"] == "calibration_complete")
    window_record = next(r for r in records if r["record_type"] == "window_summary")

    print(f"Pulling one window_summary from: {path}")
    print("Making ONE real call to the Anthropic API (use_stub=False) -- not free, not repeated.\n")

    try:
        result = stage2.run_stage2_on_window(
            window_summary=window_record,
            calibration_reference=calibration_record["reference"],
            session_id=window_record["session_id"],
            person_label=window_record.get("person_label"),
            use_stub=False,  # the ONE deliberate live call this script makes
        )
    except Exception as exc:
        print(f">>> Live call FAILED: {exc}")
        sys.exit(1)

    response = result["response"]

    print("--- response text ---")
    print(response["text"])
    print("\n--- details ---")
    print(f"model:   {response['model']}")
    print(f"is_stub: {response['is_stub']}")
    print(f"logged to: {stage2.AGENT_LOG_PATH}")

    if response["is_stub"]:
        print("\n>>> WARNING: is_stub is True -- this did NOT make a real API call. Something is wrong.")
        sys.exit(1)

    print("\n>>> SUCCESS: one real Anthropic API call completed, parsed, and logged with is_stub=false.")


if __name__ == "__main__":
    main()
