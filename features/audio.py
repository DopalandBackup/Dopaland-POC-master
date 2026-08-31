"""
U_t -- audio (D1 feature-block separation, CLAUDE.md). STUB ONLY.

There is no microphone capture, no audio processing, no clock sync between
an audio stream and the video pipeline anywhere in this repository. This is
not an oversight to be filled in casually -- CLAUDE.md's BLOCKED section
lists "U_t audio module" explicitly: "stub only. No microphone, no capture,
no clock sync exists. Pending a keep-or-formally-remove decision."

This module intentionally exposes NOTHING that could be mistaken for a
working signal: no functions, no classes, no constants, no placeholder
values. An empty module is the honest representation of "this does not
exist yet" (G3) -- a stub function that returns None or a hardcoded
placeholder would invite a caller to treat U_t as present when it isn't.

FORBIDDEN (D1): U_t -> X_core and U_t -> E_t are both forbidden directions.
Since this module has no content, that constraint is trivially satisfied
today -- but stays enforced by tests/test_feature_separation.py's
monkeypatch-to-raise check so it cannot be silently violated the moment
someone starts filling this file in.
"""
