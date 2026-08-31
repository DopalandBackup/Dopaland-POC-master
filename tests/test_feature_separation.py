"""
D1 feature-block separation test (D0PA1 Batch 1, Step 3.4). Verifies the
directional constraint CLAUDE.md's D1 requirement sets:

    PERMITTED    X_core -> E_t          C_t -> E_t
    FORBIDDEN    A_t -> X_core          A_t -> E_t
                 U_t -> X_core          U_t -> E_t

Concretely: features/x_core.py and features/episodes.py must never import
features/attention.py or features/audio.py, directly or transitively, and
must never call anything defined in them -- statically OR at runtime.

Three independent checks, each of which must FAIL LOUDLY on a real
violation (see the bottom of this docstring for how that was proven):

  1. STATIC IMPORT GRAPH -- parses every features/*.py file with `ast`
     (not regex, so a violation can't hide behind unusual formatting or a
     comment that merely LOOKS like an import) and builds the transitive
     features.* dependency graph, including imports nested inside function
     bodies (x_core.py's compute_v_bf/v_es/v_jc import their landmark
     constants from features.geometry INSIDE the function, not at module
     level -- ast.walk() finds these too, a plain top-level scan would
     miss them).

  2. STATIC CALL GRAPH -- collects every top-level name features/attention.py
     and features/audio.py actually DEFINE (function/class/assignment,
     not names they merely import from elsewhere), then checks whether
     any such name is referenced anywhere in features/x_core.py or
     features/episodes.py's AST that is NOT also independently defined or
     imported by that file itself. This second exclusion matters: x_core.py
     and attention.py each define their OWN, independent PERSON_LABEL
     global (see features/x_core.py's and features/attention.py's own
     comments on why) -- a naive name-string match would flag that as a
     violation when it is not one; a symbol only counts as "reaching into"
     attention/audio if the referencing file has no other legitimate
     source for that name.

  3. RUNTIME MONKEYPATCH -- replaces sys.modules['features.attention'] and
     ['features.audio'] with an object that raises on ANY attribute
     access, force-reloads features.x_core and features.episodes fresh
     (catching a module-level "from features.attention import X" the
     moment the reload executes it), then runs the SAME kind of
     computations tests/test_refactor_snapshot.py's golden snapshot
     exercises (reusing its synthetic fixtures, not reinventing them) --
     compute_v_bf/es/jc/pd, NeutralCalibrator, map_to_valence_arousal,
     WindowAccumulator, classify_window_confidence, classify_calibration_
     quality. If anything in that real execution path ever touches the
     poisoned attention/audio module, the proxy raises and names exactly
     which attribute was touched.

PROOF THIS TEST CAN ACTUALLY FAIL: a separation test that has never been
observed to fail is worth nothing (same reasoning that required the
pre-commit media-guard hook to be proven, not assumed -- see PROVENANCE.md).
This was verified by temporarily adding
`from features.attention import ATTENTION_ORIENTED_SCORE_THRESHOLD` to
features/x_core.py, confirming check 1 caught it with a clear message
naming the violating edge, then reverting. See the task's final report for
the pasted failure output.
"""

import ast
import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(REPO_ROOT, "features")

for p in (REPO_ROOT, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

BLOCK_MODULES = ["geometry", "x_core", "episodes", "attention", "audio", "context"]
FORBIDDEN_EDGES = [
    ("x_core", "attention"),
    ("x_core", "audio"),
    ("episodes", "attention"),
    ("episodes", "audio"),
]


def _parse(module_name):
    path = os.path.join(FEATURES_DIR, module_name + ".py")
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path)


# ============================================================
# CHECK 1 -- static import graph (transitive, includes nested imports)
# ============================================================

def direct_feature_imports(module_name):
    """features.* module names imported ANYWHERE in module_name's AST
    (module level or nested inside a function/class body)."""
    tree = _parse(module_name)
    deps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "features":
                for alias in node.names:
                    if alias.name in BLOCK_MODULES:
                        deps.add(alias.name)
            elif mod.startswith("features."):
                sub = mod.split(".", 2)[1]
                if sub in BLOCK_MODULES:
                    deps.add(sub)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("features."):
                    sub = alias.name.split(".")[1]
                    if sub in BLOCK_MODULES:
                        deps.add(sub)
    return deps


def transitive_closure(graph, start):
    seen = set()
    stack = list(graph.get(start, ()))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(graph.get(cur, ()))
    return seen


def check_static_import_graph():
    graph = {name: direct_feature_imports(name) for name in BLOCK_MODULES}
    violations = []
    for src, forbidden in FORBIDDEN_EDGES:
        reachable = transitive_closure(graph, src)
        if forbidden in reachable:
            # Find a concrete path for a useful error message.
            path = _find_path(graph, src, forbidden)
            violations.append(f"{src}.py imports (transitively) {forbidden}.py -- path: {' -> '.join(path)}")
    return violations, graph


def _find_path(graph, start, target):
    from collections import deque
    q = deque([[start]])
    visited = {start}
    while q:
        path = q.popleft()
        node = path[-1]
        if node == target:
            return path
        for nxt in graph.get(node, ()):
            if nxt not in visited:
                visited.add(nxt)
                q.append(path + [nxt])
    return [start, "?", target]


# ============================================================
# CHECK 2 -- static call graph (name-level, shadow-aware)
# ============================================================

def top_level_defined_names(module_name):
    """Names module_name DEFINES itself at module scope (FunctionDef,
    ClassDef, Assign/AnnAssign targets) -- NOT names it merely imports.
    This is deliberately narrower than "every name in scope": a name this
    module imports from a permitted upstream (geometry) is not something
    it "defines", so it correctly stays out of this set."""
    tree = _parse(module_name)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def all_names_in_scope(module_name):
    """Every name module_name defines OR imports, anywhere (module level
    or nested) -- used to exclude a file's own legitimate bindings from
    being flagged as "reaching into" another module just because the
    other module happens to define a same-spelled symbol."""
    tree = _parse(module_name)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.arg,)):
            names.add(node.arg)
    return names


def referenced_names(module_name):
    tree = _parse(module_name)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def check_static_call_graph():
    violations = []
    for forbidden_mod in ("attention", "audio"):
        forbidden_defined = top_level_defined_names(forbidden_mod)
        if not forbidden_defined:
            continue  # audio.py is empty -- nothing to check
        for src in ("x_core", "episodes"):
            src_own = all_names_in_scope(src)
            src_refs = referenced_names(src)
            leaked = (src_refs & forbidden_defined) - src_own
            if leaked:
                violations.append(
                    f"{src}.py references name(s) {sorted(leaked)} defined in {forbidden_mod}.py, "
                    f"with no other legitimate source for that name in {src}.py"
                )
    return violations


# ============================================================
# CHECK 3 -- runtime monkeypatch (attention/audio raise on any attribute access)
# ============================================================

class _RaiseOnAnyAttribute:
    """Stands in for a features.* module. Any attribute access raises,
    naming exactly which attribute was touched -- this is what makes a
    violation impossible to miss silently."""

    def __init__(self, module_name):
        self._module_name = module_name

    def __getattr__(self, item):
        raise AttributeError(
            f"FEATURE SEPARATION VIOLATION: something touched "
            f"features.{self._module_name}.{item} while {self._module_name} "
            f"was poisoned -- X_core/E_t computation must never reach A_t/U_t."
        )


def check_runtime_isolation():
    """Returns (ok: bool, message: str)."""
    import features.attention as real_attention
    import features.audio as real_audio
    import features.geometry as geometry  # not poisoned -- upstream, permitted
    from collections import deque

    # Build fixtures and do the geometry pre-processing BEFORE poisoning.
    # Importing test_refactor_snapshot transitively imports stage1_step4_vectors,
    # which is an ORCHESTRATOR (not X_core/E_t) and legitimately imports
    # features.attention -- that import must complete against the REAL
    # module, not the poisoned stand-in, or this check would fail for the
    # wrong reason (an orchestrator's permitted import, not an X_core/E_t
    # leak). Only x_core/episodes are reloaded fresh and exercised while
    # attention/audio are poisoned, below.
    import test_refactor_snapshot as snap

    lms = snap.build_synthetic_face_landmarks()
    matrix = snap.build_synthetic_transform_matrix()
    normalized_pts = geometry.pose_normalize(lms, matrix, 640, 480)
    io_dist = geometry.interocular_distance(normalized_pts)
    pose_world_sequence = snap.build_synthetic_pose_world_sequence()
    cal_samples = snap.build_synthetic_calibration_samples()
    win_samples = snap.build_synthetic_window_samples()

    sys.modules["features.attention"] = _RaiseOnAnyAttribute("attention")
    sys.modules["features.audio"] = _RaiseOnAnyAttribute("audio")

    for mod_name in ("features.x_core", "features.episodes"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    try:
        x_core = importlib.import_module("features.x_core")
        episodes = importlib.import_module("features.episodes")

        v_bf, bf_components = x_core.compute_v_bf(normalized_pts, io_dist)
        v_es, es_components = x_core.compute_v_es(normalized_pts, io_dist)
        v_jc, jc_components = x_core.compute_v_jc(normalized_pts, io_dist)

        pd_buffer = deque()
        for t, nose, shoulder_mid in pose_world_sequence:
            x_core.compute_v_pd(pd_buffer, nose, shoulder_mid, t)

        calibrator = x_core.NeutralCalibrator()
        for t, composite, covariate, yaw_deg in cal_samples:
            calibrator.add_sample(t, composite, covariate, yaw_deg)
        reference = calibrator.complete(cal_samples[-1][0])
        dev = calibrator.deviation("v_es", -0.20)
        x_core.map_to_valence_arousal(dev, dev, dev, reference)

        window_acc = episodes.WindowAccumulator()
        for t, detected, yaw_deg, composite, covariate in win_samples:
            window_acc.add_sample(t, detected, yaw_deg, composite, covariate)
        window_acc.flush(win_samples[-1][0])

        return True, "computed X_core and E_t end-to-end with attention/audio poisoned -- neither was touched."
    except (AttributeError, ImportError) as e:
        # ImportError covers a module-level "from features.attention import X"
        # in x_core.py/episodes.py itself: Python's import machinery wraps the
        # poisoned __getattr__'s AttributeError into an ImportError at the
        # `from ... import` statement, with a less specific message -- still a
        # clear, loud failure, just from a different exception type.
        return False, f"{type(e).__name__}: {e}"
    finally:
        sys.modules["features.attention"] = real_attention
        sys.modules["features.audio"] = real_audio
        for mod_name in ("features.x_core", "features.episodes"):
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        importlib.import_module("features.x_core")
        importlib.import_module("features.episodes")


# ============================================================
# MAIN
# ============================================================

def run_all():
    failures = []

    static_import_violations, graph = check_static_import_graph()
    print("[1/3] STATIC IMPORT GRAPH")
    for name in BLOCK_MODULES:
        print(f"      {name}.py direct features.* imports: {sorted(graph[name]) or '(none)'}")
    if static_import_violations:
        print("      FAIL:")
        for v in static_import_violations:
            print(f"        - {v}")
        failures.extend(static_import_violations)
    else:
        print("      PASS -- x_core.py and episodes.py never import attention.py or audio.py, transitively.")

    static_call_violations = check_static_call_graph()
    print("[2/3] STATIC CALL GRAPH")
    if static_call_violations:
        print("      FAIL:")
        for v in static_call_violations:
            print(f"        - {v}")
        failures.extend(static_call_violations)
    else:
        print("      PASS -- no symbol defined in attention.py/audio.py is referenced by x_core.py/episodes.py.")

    runtime_ok, runtime_message = check_runtime_isolation()
    print("[3/3] RUNTIME MONKEYPATCH (attention/audio raise on any attribute access)")
    if runtime_ok:
        print(f"      PASS -- {runtime_message}")
    else:
        print(f"      FAIL: {runtime_message}")
        failures.append(runtime_message)

    return failures


if __name__ == "__main__":
    failures = run_all()
    print()
    if failures:
        print(f"FEATURE SEPARATION TEST: FAIL ({len(failures)} violation(s))")
        sys.exit(1)
    print("FEATURE SEPARATION TEST: PASS")
