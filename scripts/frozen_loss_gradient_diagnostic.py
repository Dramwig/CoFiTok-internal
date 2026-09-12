"""Prepare/authorize/run/replay one separately approved frozen gradient assay.

No command writes a user approval. No optimizer, training, sampling or promotion
entry point is called. Partial outputs remain evidence and block automatic rerun.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from cofitok.generation.frozen_gradient_stage import (
    CATALOG, CATALOG_SHA, CONTROL, DENIED, METHODS, OUTPUT, PROTOCOL, ROLES, SCRIPT,
    TIMESTEPS, admission, authorization_payload, build_preparation, canonical,
    checkout, exclusive_json, invocation_row, no_symlinks, read_bound, rehash,
    validate_preparation,
)
from cofitok.generation.frozen_gradient_readout import readout
from cofitok.generation_class_support_contingency import stable_file_identity


def current_checkout(root):
    root = Path(root).resolve(strict=True)
    if Path(__file__).resolve() != root / SCRIPT:
        raise ValueError("running script is not the selected checkout script")
    # Prevent a clean working copy from attesting imports from another checkout.
    for name, module in tuple(sys.modules.items()):
        if name == "cofitok" or name.startswith("cofitok."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(root / "src/cofitok"):
                raise ValueError(f"imported module outside exact checkout: {name}")
    return checkout(root)


def gpu_ownership(*, allow_self=False, require_self=False):
    output = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"
    ], text=True)
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, uuid, memory = [p.strip() for p in line.split(",")]
        rows.append({"pid": int(pid), "gpu_uuid": uuid, "memory_mib": int(memory)})
    if any(row["pid"] != os.getpid() or not allow_self for row in rows):
        raise ValueError("GPU ownership conflict; no process will be signalled")
    if require_self and not any(row["pid"] == os.getpid() for row in rows):
        raise ValueError("own CUDA process was not physically observed")
    return rows


def runtime_check(prepared, project_root):
    import torch
    from cofitok.environment import capture_runtime_environment
    expected = prepared["expected_runtime"]
    # These are frozen-runtime settings, applied only within this authorized process.
    if os.environ.get("PYTORCH_ALLOC_CONF") != expected["environment_variables"]["PYTORCH_ALLOC_CONF"]:
        raise ValueError("launch allocator environment differs from frozen runtime")
    torch.backends.cuda.matmul.allow_tf32 = expected["torch"]["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = expected["torch"]["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = expected["torch"]["cudnn_benchmark"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exact single CUDA device required")
    actual = capture_runtime_environment(torch.device("cuda:0"), project_root=project_root)
    if actual != expected:
        from cofitok.environment import runtime_environment_mismatch_paths
        raise ValueError("frozen runtime differs: " + ", ".join(runtime_environment_mismatch_paths(expected, actual)))
    return actual


def load_frozen_pair_member(source, expected_runtime):
    """Called only after outer stage admission; check all bytes before torch.load."""
    import torch
    from cofitok.configs import config_from_dict
    from cofitok.environment import runtime_environment_sha256
    from cofitok.models import CoFiTokTiny
    from cofitok.training.authorization import validate_checkpoint_training_authorization
    from cofitok.training.checkpointing import verify_training_checkpoint

    for key in ("checkpoint", "sidecar", "training_report"):
        rehash(source[key])
    report, _ = read_bound(source["training_report"]["path"], source["training_report"]["sha256"])
    integrity = verify_training_checkpoint(source["checkpoint"]["path"])
    payload = torch.load(source["checkpoint"]["path"], map_location="cpu", weights_only=False)
    # No model allocation or CUDA copy until embedded provenance agrees.
    if payload["format_version"] != 1 or payload["step"] != 100000 or payload["config"] != source["config"]:
        raise ValueError("embedded checkpoint step/config differs")
    extra = payload["extra_state"]
    if (extra["git"] != report["git"] or extra["runtime_environment"] != expected_runtime
            or extra["runtime_environment_sha256"] != runtime_environment_sha256(expected_runtime)
            or extra["runtime_environment_sha256"] != integrity["runtime_environment_sha256"]
            or extra["dataset_provenance"] != report["dataset_provenance"]):
        raise ValueError("embedded checkpoint provenance differs")
    validate_checkpoint_training_authorization(payload, integrity)
    config = config_from_dict(payload["config"])
    ema = payload["ema"]
    if (ema["num_updates"] != 100000 or ema["decay"] != config.optimization.ema_decay
            or ema["warmup_steps"] != config.optimization.ema_warmup_steps):
        raise ValueError("original EMA metadata differs")
    if payload["model"].keys() != ema["shadow"].keys():
        raise ValueError("original EMA/model structures differ")
    for key, value in payload["model"].items():
        teacher = ema["shadow"][key]
        if (not isinstance(value, torch.Tensor) or not isinstance(teacher, torch.Tensor)
                or value.shape != teacher.shape or value.dtype != teacher.dtype
                or not torch.isfinite(value).all() or not torch.isfinite(teacher).all()):
            raise ValueError("nonfinite or incompatible frozen tensor")
    model = CoFiTokTiny(config.model)
    model.load_state_dict(payload["model"], strict=True)
    shadow = {k: v.detach() for k, v in ema["shadow"].items()}
    del payload
    # Close the deserialization TOCTOU before any model forward.
    for key in ("checkpoint", "sidecar", "training_report"):
        rehash(source[key])
    return model, shadow, config


def tensor_fingerprint(named):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(named):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _args_admission(args, role):
    current_checkout(args.project_root)
    return admission(args.preparation, args.preparation_sha256,
                     args.authorization, args.authorization_sha256,
                     project_root=args.project_root, role=role)


def run_assay(args):
    # No CUDA admission or torch.load precedes these full physical replays.
    prepared, pid, aid, sid = _args_admission(args, "evaluator")
    catalog, cid = read_bound(CATALOG, CATALOG_SHA)
    no_symlinks(OUTPUT)
    if OUTPUT.exists():
        raise ValueError("existing diagnostic output blocks rerun, including partial failures")
    if shutil.disk_usage(OUTPUT.parent).free < PROTOCOL["minimum_free_storage_bytes"]:
        raise ValueError("insufficient diagnostic disk reserve")
    gpu_ownership()
    OUTPUT.mkdir(exist_ok=False)  # Atomic single invocation claim; never removed on failure.
    start = time.monotonic()
    invocation = {"schema": "frozen_gradient_invocation_v1", "preparation": pid,
                  "authorization": aid, "stage_approval": sid,
                  "evaluator": prepared["checkouts"]["evaluator"],
                  "pid": os.getpid(), "hostname": socket.gethostname(),
                  "start_time_unix": time.time(), "permissions": DENIED}
    exclusive_json(OUTPUT / "invocation.json", invocation)
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline(signum, frame):
        del signum, frame
        raise TimeoutError("authorized diagnostic wall-time budget exhausted")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, PROTOCOL["max_wall_seconds"])
    try:
        import torch
        from PIL import Image
        from cofitok.data.registry import _image_transform
        from cofitok.generation.frozen_loss_gradients import measure_selected_example

        runtime = runtime_check(prepared, args.project_root)
        rows, row_ids, preserved = [], [], {}
        for method in METHODS:
            gpu_ownership(allow_self=True)
            model, shadow, config = load_frozen_pair_member(catalog["methods"][method], runtime)
            before = {"model": tensor_fingerprint(model.state_dict().items()),
                      "ema": tensor_fingerprint(shadow.items())}
            model = model.to("cuda:0").eval()
            shadow = {k: v.to("cuda:0") for k, v in shadow.items()}
            transform = _image_transform(config.data.image_size)
            for index, sample in enumerate(catalog["selection"]["samples"]):
                rehash(sample["image"])
                with Image.open(sample["image"]["path"]) as image:
                    clean = transform(image.convert("RGB")).unsqueeze(0).to("cuda:0")
                for timestep in TIMESTEPS:
                    if shutil.disk_usage(OUTPUT).free < PROTOCOL["minimum_free_storage_bytes"]:
                        raise ValueError("diagnostic disk reserve lost")
                    gpu = gpu_ownership(allow_self=True, require_self=True)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    row_start = time.monotonic()
                    seeds = invocation_row(index, timestep)
                    row = measure_selected_example(model, ema_state=shadow, config=config, clean=clean,
                        label=sample["label"], timestep=timestep, checkpoint_step=100000,
                        noise_seed=seeds["noise_seed"], dropout_seed=seeds["dropout_seed"])
                    torch.cuda.synchronize()
                    row.update({**seeds, "method": method, "image": sample["image"],
                                "elapsed_seconds": time.monotonic() - row_start,
                                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                                "observed_gpu_processes": gpu})
                    row_ids.append(exclusive_json(OUTPUT / f"measurement_{len(rows):06d}.json", row))
                    rows.append(row)
                    print(json.dumps({"completed_rows": len(rows), "total_rows": 256,
                                      "method": method, "image_index": index, "timestep": timestep}), flush=True)
                del clean
            after = {"model": tensor_fingerprint(model.state_dict().items()),
                     "ema": tensor_fingerprint(shadow.items())}
            if before != after or any(p.grad is not None for p in model.parameters()):
                raise ValueError("frozen tensors or gradient buffers changed")
            preserved[method] = {"before": before, "after": after, "grad_buffers_remained_none": True}
            del model, shadow
            gc.collect()
            torch.cuda.empty_cache()
        analysis = readout(rows, catalog)
        if runtime_check(prepared, args.project_root) != runtime:
            raise ValueError("runtime changed during measurement")
        if _args_admission(args, "evaluator") != (prepared, pid, aid, sid):
            raise ValueError("source/admission changed during measurement")
        rehash(cid)
        for ident in row_ids:
            rehash(ident)
        result = {"schema": "frozen_loss_gradient_result_v1", "status": "completed_diagnostic_only",
                  "preparation": pid, "authorization": aid, "stage_approval": sid,
                  "catalog": cid, "protocol": PROTOCOL, "permissions": DENIED,
                  "evaluator": prepared["checkouts"]["evaluator"], "runtime": runtime,
                  "invocation": stable_file_identity(OUTPUT / "invocation.json"),
                  "row_identities": row_ids, "tensor_preservation": preserved,
                  "analysis": analysis, "elapsed_seconds": time.monotonic() - start,
                  "postflight_gpu_processes": gpu_ownership(allow_self=True)}
        result_id = exclusive_json(OUTPUT / "result.json", result)
        print(json.dumps({"status": result["status"], "result": result_id, "training_allowed": False}))
    except BaseException as error:
        exclusive_json(OUTPUT / "failure.json", {"status": "failed_closed", "pid": os.getpid(),
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.monotonic() - start, "permissions": DENIED,
            "automatic_retry_allowed": False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def validate_result(args):
    prepared, pid, aid, sid = _args_admission(args, "validator")
    if (OUTPUT / "failure.json").exists():
        raise ValueError("failed invocation cannot receive a passing receipt")
    result, rid = read_bound(OUTPUT / "result.json", args.result_sha256)
    catalog, cid = read_bound(CATALOG, CATALOG_SHA)
    expected = {"schema": "frozen_loss_gradient_result_v1", "status": "completed_diagnostic_only",
                "preparation": pid, "authorization": aid, "stage_approval": sid,
                "catalog": cid, "protocol": PROTOCOL, "permissions": DENIED,
                "evaluator": prepared["checkouts"]["evaluator"], "runtime": prepared["expected_runtime"]}
    if any(canonical(result.get(k)) != canonical(v) for k, v in expected.items()):
        raise ValueError("result provenance/protocol differs")
    invocation, iid = read_bound(OUTPUT / "invocation.json", result["invocation"]["sha256"])
    if iid != result["invocation"] or any(invocation[k] != expected[k] for k in
        ("preparation", "authorization", "stage_approval", "evaluator", "permissions")):
        raise ValueError("invocation binding differs")
    if len(result["row_identities"]) != 256:
        raise ValueError("result requires all 256 rows")
    rows = []
    for index, ident in enumerate(result["row_identities"]):
        if Path(ident["path"]) != OUTPUT / f"measurement_{index:06d}.json":
            raise ValueError("row path/order differs")
        row, actual = read_bound(ident["path"], ident["sha256"])
        if actual != ident or not row["observed_gpu_processes"] or any(
            p["pid"] != invocation["pid"] for p in row["observed_gpu_processes"]):
            raise ValueError("row identity or recorded GPU ownership differs")
        rows.append(row)
    replayed = readout(rows, catalog)
    if replayed != result["analysis"]:
        raise ValueError("independent scalar/image-level replay differs")
    if set(result["tensor_preservation"]) != set(METHODS):
        raise ValueError("tensor-preservation method coverage differs")
    for row in result["tensor_preservation"].values():
        if row["before"] != row["after"] or row["grad_buffers_remained_none"] is not True:
            raise ValueError("frozen tensor preservation failed")
        for sha in row["before"].values():
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("tensor fingerprint malformed")
    if not 0 < result["elapsed_seconds"] <= PROTOCOL["max_wall_seconds"]:
        raise ValueError("result exceeded invocation budget")
    allowed = {"result.json", "invocation.json", "result.validation.json"} | {
        f"measurement_{i:06d}.json" for i in range(256)}
    if {p.name for p in OUTPUT.iterdir()} - allowed:
        raise ValueError("unexpected output artifact")
    if _args_admission(args, "validator") != (prepared, pid, aid, sid):
        raise ValueError("sources changed during independent validation")
    rehash(rid)
    rehash(iid)
    for ident in result["row_identities"]:
        rehash(ident)
    receipt = {"schema": "frozen_loss_gradient_validation_v1", "status": "pass",
               "result": rid, "preparation": pid, "authorization": aid,
               "validator": prepared["checkouts"]["validator"],
               "analysis_sha256": hashlib.sha256(canonical(replayed)).hexdigest(),
               "all_physical_rows_rehashed": True, "scalar_arithmetic_replayed": True,
               "gradient_vectors_independently_recomputed": False, "scientific_status": "diagnostic_only",
               "permissions": DENIED}
    path = OUTPUT / "result.validation.json"
    if path.exists():
        existing, receipt_id = read_bound(path)
        if existing != receipt:
            raise ValueError("existing immutable receipt differs")
    else:
        receipt_id = exclusive_json(path, receipt)
        persisted, _ = read_bound(path, receipt_id["sha256"])
        if persisted != receipt:
            raise ValueError("persisted receipt differs")
    print(json.dumps({"status": "pass", "receipt": receipt_id, "scientific_status": "diagnostic_only"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "replay-preparation", "authorize", "run", "validate"))
    parser.add_argument("--project-root", type=Path, required=True)
    for role in ROLES:
        parser.add_argument(f"--{role}-root", type=Path)
    parser.add_argument("--preparation", type=Path, default=CONTROL / "preparation.json")
    parser.add_argument("--preparation-sha256")
    parser.add_argument("--stage-approval-sha256")
    parser.add_argument("--authorization", type=Path, default=CONTROL / "execution_authorization.json")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--result-sha256")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise ValueError("real diagnostic stages are remote Linux only; local use is unit tests")
    current = current_checkout(args.project_root)
    if args.mode == "prepare":
        roots = {role: getattr(args, role + "_root") for role in ROLES}
        if any(root is None for root in roots.values()) or roots["preparation"].resolve() != args.project_root.resolve():
            raise ValueError("exact stage roots required for preparation")
        prepared = build_preparation(roots)
        no_symlinks(CONTROL)
        CONTROL.mkdir(exist_ok=True)
        ident = exclusive_json(CONTROL / "preparation.json", prepared)
        validate_preparation(ident["path"], ident["sha256"])
        print(json.dumps({"status": "prepared_not_authorized", "preparation": ident}))
        return
    if not args.preparation_sha256:
        raise ValueError("exact preparation SHA256 required")
    if args.mode in ("replay-preparation", "authorize"):
        prepared, pid = validate_preparation(args.preparation, args.preparation_sha256)
        if current != prepared["checkouts"]["authorization" if args.mode == "authorize" else "validator"]:
            raise ValueError("wrong stage checkout")
        if args.mode == "replay-preparation":
            print(json.dumps({"status": "pass", "preparation": pid, "execution_authorized": False}))
            return
        if not args.stage_approval_sha256:
            raise ValueError("distinct user-created stage approval required")
        approval, sid = read_bound(CONTROL / "stage_approval.json", args.stage_approval_sha256)
        auth = authorization_payload(prepared, pid, approval, sid)
        if OUTPUT.exists():
            raise ValueError("existing invocation forbids authorization reuse")
        rehash(pid)
        rehash(sid)
        print(json.dumps({"authorization": exclusive_json(CONTROL / "execution_authorization.json", auth)}))
        return
    if not args.authorization_sha256:
        raise ValueError("exact separate execution authorization required")
    if args.mode == "run":
        run_assay(args)
    else:
        if not args.result_sha256:
            raise ValueError("exact result SHA256 required")
        validate_result(args)


if __name__ == "__main__":
    main()
