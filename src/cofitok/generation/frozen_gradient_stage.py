"""Source-bound, non-training admission for the frozen selected-image assay.

Preparation is not permission. There is deliberately no stage-approval writer.
Every executable stage consumes the same exact preparation and physically
rechecks its sources. GPU allocation/model deserialization belong after admission.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from cofitok.generation_class_support_contingency import stable_file_identity

BASE = Path("/root/autodl-tmp/CoFiTok/checkpoints/generation")
CONTROL = BASE / ".frozen_loss_gradient_attribution_v1.control"
OUTPUT = BASE / "frozen_loss_gradient_attribution_v1"
CATALOG = CONTROL / "source_catalog.json"
CATALOG_SHA = "f6d1a1917db350a20d4d953f45b76e82050b51a7b2fe8ed0b7e2432bde5b929b"
ORIGINAL = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
COMPATIBLE = "5ac619228b051d12d87a2d790a3676d31c8d7b72"
METHODS = ("cofitok", "dense_identity")
TIMESTEPS = (100, 500, 700, 900)
ROLES = ("preparation", "authorization", "evaluator", "validator")
SCRIPT = "scripts/frozen_loss_gradient_diagnostic.py"
DENIED = {name: False for name in (
    "training_allowed", "optimizer_step_allowed", "ema_update_allowed", "sampling_allowed",
    "confirmation_allowed", "full_training_allowed", "full_300k_allowed", "promotion_allowed",
    "export_allowed", "release_allowed", "paper_integration_allowed",
)}
PROTOCOL = {
    "methods": list(METHODS), "image_count": 32, "timesteps": list(TIMESTEPS),
    "measurement_rows": 256, "weights": "model", "teacher": "original_checkpoint_ema",
    "checkpoint_step": 100000, "precision": "bf16", "device": "cuda:0",
    "microbatch": 1, "noise_seed_base": 20310000, "dropout_seed_base": 20320000,
    "seed_formula": "base + image_index*1000 + timestep; identical across methods",
    "augmentation": "fixed validation RGB resize256 ToTensor Normalize(0.5,0.5); no random flip",
    "unit": "single_selected_image_not_original_minibatch",
    "independent_units": 32, "timestep_rows_are_repeated_measurements": True,
    "max_invocations": 1, "max_wall_seconds": 7200,
    "minimum_free_storage_bytes": 10 * 1024**3, "gpu_must_be_idle_before_admission": True,
    "result_scope": "exploratory_loss_gradient_attribution_not_generation_quality",
    "nomination": {
        "cosine_at_most": -0.1, "norm_ratio_at_least": 0.1,
        "qualifying_timesteps_per_image": 3, "qualifying_images_per_method": 24,
        "groups": ["class_embedding", "conditioning_projections"],
        "terms": ["rollout", "ema_teacher"],
        "rule": "same term and same group must qualify in both methods; hypothesis only",
        "saturation_fraction_at_least": 0.5,
        "saturation_rule": "each unrolled step qualifies in at least 3/4 timesteps for 24/32 images per method",
        "authorizes_intervention": False, "proves_cause": False,
    },
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_bound(path, expected_sha=None):
    ident = stable_file_identity(path)
    if expected_sha is not None and ident["sha256"] != expected_sha:
        raise ValueError(f"bound SHA256 differs: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or stable_file_identity(path) != ident:
        raise ValueError(f"bound JSON changed or is not an object: {path}")
    return value, ident


def rehash(ident):
    if stable_file_identity(ident["path"]) != ident:
        raise ValueError(f"physical source differs: {ident['path']}")


def no_symlinks(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("output must be an exact absolute path")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlinked output path")


def exclusive_json(path, value):
    path = Path(path)
    no_symlinks(path)
    if path.parent not in (CONTROL, OUTPUT):
        raise ValueError("write outside the dedicated diagnostic roots")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    # Preserve partial evidence on failure; never overwrite/retry an invocation.
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return stable_file_identity(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def checkout(root):
    root = Path(root).resolve(strict=True)
    if git(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("diagnostic checkout is not completely clean")
    # Also defeat assume-unchanged/skip-worktree concealment for executable inputs.
    paths = git(root, "ls-files", "src/cofitok", SCRIPT).decode().splitlines()
    blobs = {}
    for path in paths:
        if not path.endswith(".py"):
            continue
        data = (root / path).read_bytes()
        if data != git(root, "show", f"HEAD:{path}"):
            raise ValueError(f"working file differs from exact blob: {path}")
        blobs[path] = hashlib.sha256(data).hexdigest()
    if SCRIPT not in blobs or "src/cofitok/generation/frozen_loss_gradients.py" not in blobs:
        raise ValueError("diagnostic executable/core is absent")
    return {"root": str(root), "revision": git(root, "rev-parse", "HEAD").decode().strip(),
            "tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "branch": git(root, "branch", "--show-current").decode().strip(),
            "tracked_dirty": False, "python_blobs": blobs}


def source_compatibility(root):
    paths = git(root, "ls-tree", "-r", "--name-only", ORIGINAL,
                "src/cofitok/models", "src/cofitok/diffusion").decode().splitlines()
    paths += ["src/cofitok/" + name for name in (
        "configs.py", "data/registry.py", "environment.py", "training/ema.py",
        "training/losses.py", "training/rollout.py", "training/runtime.py")]
    result = {}
    for path in sorted(paths):
        old, actual = git(root, "show", f"{ORIGINAL}:{path}"), (Path(root) / path).read_bytes()
        if actual != git(root, "show", f"{COMPATIBLE}:{path}"):
            raise ValueError(f"source differs from reviewed compatible implementation: {path}")
        equivalent_default = path in {"src/cofitok/configs.py", "src/cofitok/diffusion/schedule.py"}
        if old != actual and not equivalent_default:
            raise ValueError(f"frozen production source drift: {path}")
        result[path] = {"original_sha256": hashlib.sha256(old).hexdigest(),
                        "executable_sha256": hashlib.sha256(actual).hexdigest(),
                        "compatibility": "unchanged" if old == actual else "reviewed_default_endpoint_1_extension"}
    return result


def replay_catalog():
    catalog, ident = read_bound(CATALOG, CATALOG_SHA)
    builder = catalog["builder"]
    root = Path(builder["root"])
    if (git(root, "rev-parse", "HEAD").decode().strip() != builder["revision"]
            or git(root, "rev-parse", "HEAD^{tree}").decode().strip() != builder["tree"]):
        raise ValueError("catalog builder identity changed")
    rehash(builder["script"])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": str(root) + os.pathsep + str(root / "src")}
    subprocess.run([sys.executable, "-B", builder["script"]["path"], "validate", "--project-root",
                    str(root), "--catalog-sha256", CATALOG_SHA], cwd=root, env=env,
                   check=True, capture_output=True, text=True)
    rehash(ident)
    return catalog, ident


def build_preparation(roots):
    if set(roots) != set(ROLES):
        raise ValueError("all four exact stage checkouts are required")
    checkouts = {role: checkout(roots[role]) for role in ROLES}
    if len({c["root"] for c in checkouts.values()}) != 4:
        raise ValueError("stage roles require separate checkouts")
    if len({(c["revision"], c["tree"]) for c in checkouts.values()}) != 1:
        raise ValueError("stage implementation revisions differ")
    catalog, ident = replay_catalog()
    reports = {m: read_bound(catalog["methods"][m]["training_report"]["path"],
                             catalog["methods"][m]["training_report"]["sha256"])[0] for m in METHODS}
    runtime = reports[METHODS[0]]["runtime_environment"]
    if runtime != reports[METHODS[1]]["runtime_environment"]:
        raise ValueError("frozen method runtimes differ")
    for method in METHODS:
        cfg = catalog["methods"][method]["config"]
        if (cfg["runtime"]["precision"] != "bf16" or cfg["runtime"]["device"] != "cuda"
                or cfg["diffusion"].get("cosine_endpoint_fraction", 1.0) != 1.0
                or cfg["diffusion"]["prediction_target"] != "epsilon"):
            raise ValueError("frozen diagnostic runtime/target differs")
    return {"schema": "frozen_loss_gradient_preparation_v1", "catalog": ident,
            "checkouts": checkouts, "source_compatibility": source_compatibility(roots["preparation"]),
            "expected_runtime": runtime, "protocol": PROTOCOL, "output_root": str(OUTPUT),
            "permissions": DENIED, "execution_authorized": False}


def validate_preparation(path, sha):
    prepared, ident = read_bound(path, sha)
    if Path(path) != CONTROL / "preparation.json":
        raise ValueError("preparation path differs")
    roots = {role: row["root"] for role, row in prepared["checkouts"].items()}
    if prepared != build_preparation(roots):
        raise ValueError("preparation differs from physical source replay")
    rehash(ident)
    return prepared, ident


def validate_approval(approval, preparation_id, prepared):
    expected = {"schema": "user_created_frozen_loss_gradient_approval_v1",
                "preparation": preparation_id, "evaluator": prepared["checkouts"]["evaluator"],
                "protocol": PROTOCOL, "output_root": str(OUTPUT), "permissions": DENIED,
                "user_authorized_frozen_gradient_diagnostic": True}
    if set(approval) != set(expected) | {"user_instruction", "approved_at_utc"}:
        raise ValueError("distinct gradient stage approval fields differ")
    if any(canonical(approval[k]) != canonical(v) for k, v in expected.items()):
        raise ValueError("gradient stage approval does not bind this exact preparation")
    if not all(isinstance(approval[k], str) and approval[k].strip()
               for k in ("user_instruction", "approved_at_utc")):
        raise ValueError("explicit user instruction and timestamp required")
    # This is an artifact contract, not proof of consent. Operator must have the
    # actual distinct human approval; broad goals and previous stages do not count.


def authorization_payload(prepared, preparation_id, approval, approval_id):
    validate_approval(approval, preparation_id, prepared)
    return {"schema": "frozen_loss_gradient_execution_authorization_v1",
            "preparation": preparation_id, "stage_approval": approval_id,
            "evaluator": prepared["checkouts"]["evaluator"], "protocol": PROTOCOL,
            "output_root": str(OUTPUT), "permissions": DENIED,
            "frozen_gradient_execution_allowed": True}


def admission(preparation, preparation_sha, authorization, authorization_sha, *, project_root, role):
    prepared, pid = validate_preparation(preparation, preparation_sha)
    if role not in ("evaluator", "validator") or checkout(project_root) != prepared["checkouts"][role]:
        raise ValueError("caller is not the exact authorized stage checkout")
    auth, aid = read_bound(authorization, authorization_sha)
    if Path(authorization) != CONTROL / "execution_authorization.json":
        raise ValueError("execution authorization path differs")
    approval, sid = read_bound(auth["stage_approval"]["path"], auth["stage_approval"]["sha256"])
    if sid != auth["stage_approval"] or Path(sid["path"]) != CONTROL / "stage_approval.json":
        raise ValueError("stage approval identity/path differs")
    if auth != authorization_payload(prepared, pid, approval, sid):
        raise ValueError("execution authorization replay differs")
    return prepared, pid, aid, sid


def invocation_row(image_index, timestep):
    if type(image_index) is not int or not 0 <= image_index < 32 or timestep not in TIMESTEPS:
        raise ValueError("row outside bounded protocol")
    return {"image_index": image_index, "timestep": timestep,
            "noise_seed": PROTOCOL["noise_seed_base"] + image_index * 1000 + timestep,
            "dropout_seed": PROTOCOL["dropout_seed_base"] + image_index * 1000 + timestep}
