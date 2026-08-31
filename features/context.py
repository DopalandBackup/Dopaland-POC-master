"""
C_t -- context (D1 feature-block separation, CLAUDE.md). STUB ONLY.

There is no context-feature code anywhere in this repository. A repo-wide
search for context_id, "C_t", ROI, roi_id, and salience terms (done as part
of this task's Step 2, before any code was moved) found zero matches. This
module records that as "nonexistent", not "clean" -- those read very
differently to a client, and only one of them is true here.

WATCH THIS MODULE if it is ever filled in: C_t -> E_t is a PERMITTED
direction (E_t = h(X_core, dX_core, C_t) by definition), which makes
context a legal-looking back door for attention into the core. Any future
context feature that is ROI-derived or gaze-derived (dwell time on a
region, fixation-weighted context, anything computed from attention.py)
would be an attention-class value smuggled in through a permitted channel,
and must be flagged loudly rather than added quietly. See
docs/D1_DEPENDENCY_MAP.md.

This module intentionally exposes NOTHING (no functions, no classes, no
constants, no placeholder values) -- an empty module is the honest
representation of "this does not exist yet" (G3).
"""
