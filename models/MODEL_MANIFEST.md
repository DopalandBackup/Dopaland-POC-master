# Model Bundle Manifest

`models/` is excluded from version control (`.gitignore`) — these are large binary
dependencies, not project source or output. This file is the **committed** record of
what belongs there and how to verify a copy is correct, so a clean-machine build can
reproduce the pipeline without those binaries ever entering git history.

Checksums below were computed directly against the files present in this working tree,
using two independent methods (`sha256sum` and Python's `hashlib.sha256`, chunked
read), which agreed exactly. They were not copied from any external source.

---

## face_landmarker.task

| Field | Value |
|---|---|
| Filename | `face_landmarker.task` |
| SHA256 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| Size | 3,758,596 bytes (3.6 MB) |
| Pairs with | `mediapipe==0.10.35` (this repo's pinned version — see `requirements.txt`) |
| Used by | `mp_vision.FaceLandmarker.create_from_options(...)`, all capture/analysis entry points |

## pose_landmarker_full.task

| Field | Value |
|---|---|
| Filename | `pose_landmarker_full.task` |
| SHA256 | `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1` |
| Size | 9,398,198 bytes (9.0 MB) |
| Pairs with | `mediapipe==0.10.35` |
| Used by | `mp_vision.PoseLandmarker.create_from_options(...)`, all capture/analysis entry points |

---

## Source

Both files match the naming convention of Google's official MediaPipe Tasks model
catalog, hosted at `storage.googleapis.com/mediapipe-models/`:

- Face Landmarker: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- Pose Landmarker (full): `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`

**Honest limitation, stated plainly:** these two URLs are the standard, publicly
documented locations for these model names in MediaPipe's own model catalog — they are
not fabricated. But I have **not** performed a live download-and-compare against them
in this session (no network verification was run), and there is **no download log,
fetch script, or provenance record anywhere in this repository** documenting the actual
original download event for the two files sitting in `models/` today. The SHA256 values
above are a real, freshly-computed fingerprint of the files as they exist in this
working tree right now — treat them as "this is what's here, verify against this,"
not as proof the files came from the URLs listed. If exact-source verification matters
for the client's reproduction test, someone should fetch fresh copies from the URLs
above and diff the checksums against this manifest.

## How to reproduce

```
sha256sum models/face_landmarker.task models/pose_landmarker_full.task
```

Compare against the two hashes above. A mismatch means a different model version/variant
is present and results are not expected to reproduce exactly.
