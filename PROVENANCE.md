# Provenance

Version control was established on this repository on **2026-08-24**.

All work in this repository prior to that date was performed without version
control. Commit timestamps in this repository's history therefore record when
the repository was created and when each commit was made from that point
forward — they do **not** record when the underlying work was originally
done. No commit in this history has been backdated. No prior history has been
reconstructed, synthesised, or staged to appear as though it existed before
this date.

A hardened `.gitignore` and a media-blocking, secret-blocking pre-commit hook
(`.githooks/pre-commit`, `core.hooksPath=.githooks`) were both in place and
verified to actually fire before the first file was ever staged. No face,
audio, or other raw participant media has entered this repository's history
at any point — verified directly (see `docs/PRIVACY_EVIDENCE.md`), not merely
assumed from the `.gitignore` rules.

From this commit onward, every material change to this repository is a dated
commit.

## Pre-commit hook verification

Before the pre-commit hook was trusted, it was tested against two synthetic
files, both named `test_guard.txt`: one containing real PNG magic bytes
(`\x89PNG\r\n\x1a\n` followed by random binary padding), and one containing
an `ANTHROPIC_API_KEY` assignment with a synthetic, randomly-generated
high-entropy value in the same `sk-ant-` key format this project's real keys
use. Both files were created solely to exercise the hook and contained no
participant data, no biometric data, and no real credential of any kind.

(This section originally quoted that synthetic value directly. Doing so
tripped this repository's own pre-commit hook on the commit that added this
section — the hook does not distinguish a fake key quoted for documentation
from a real one, which is the correct, conservative failure mode. Rewritten
to describe the test without reproducing a string in that shape.)

The first attempt at the magic-bytes test did not trigger the hook. Before
that was diagnosed, `git commit` was run and it succeeded, creating one
commit — `178390c`, message "test: attempt to commit disguised PNG" —
containing only that synthetic test file. Diagnosis found the cause: git's
`autocrlf` line-ending normalization had altered the file's bytes on the way
into the index, so the staged content no longer matched a PNG signature; the
hook was reading correctly, the test file did not survive being staged
intact. That commit was removed via `git update-ref -d HEAD` before any
other commit existed in this repository — at the time of removal it was the
sole, unpublished, parentless commit on this branch, with no remote and
nothing built on top of it.

The test file was then rebuilt with random binary padding so it would
survive normalization intact, and both cases (magic bytes, key-like string)
were re-run and confirmed to correctly block a commit. Both test files were
deleted before any real content was staged.

The remaining object data from the removed commit stayed present but
unreachable in this repository's object store. `git reflog expire
--expire=now --all` followed by `git gc --prune=now` was run to remove it,
so a client running `git fsck` finds a clean object store with no dangling
objects. This removed only the unreachable synthetic-test objects; it did
not alter, rewrite, or remove any of the real commits in this repository's
history.

## Open item: raw data location

`logs/` is excluded from version control (`.gitignore`) and will never be
committed. That satisfies "raw data is not in git history." It does **not**
by itself satisfy the separate requirement that raw data live **outside the
repository directory, in a controlled location** — as of this commit,
`logs/` physically resides inside this repository's working directory on
disk, untracked but present. Being gitignored and being physically elsewhere
are two different guarantees; only the first is currently true.

This repository contains no raw media of any kind — no video, no audio, no
images — verified by a magic-byte content scan across every file in the
working tree, independent of file extension (method and result recorded in
`docs/PRIVACY_EVIDENCE.md`). What `logs/` currently holds is derived numeric
features and anonymous participant codes, not raw capture output.

The physical relocation of `logs/` to a location outside this repository is
an open decision, not yet made. This file records the gap; it does not
resolve it.
