from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report


PLAN_SCHEMA_VERSION = 1
PLAN_ROLE = "generation_deployment_pack_dedup_plan"
RESULT_SCHEMA_VERSION = 1
RESULT_ROLE = "generation_deployment_pack_dedup_result"
CHECKOUT_PATTERN = re.compile(r"checkout-([0-9a-f]{7})")


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "link_count": stat.st_nlink,
    }


def _receipt_path(deployment_root: Path, revision: str) -> Path:
    return deployment_root / "deployments" / revision / "deployment_receipt.json"


def _checkout_state(deployment_root: Path, checkout: Path) -> dict[str, Any]:
    if CHECKOUT_PATTERN.fullmatch(checkout.name) is None or not (checkout / ".git").is_dir():
        raise ValueError(f"invalid deployment checkout: {checkout}")
    revision = _git(checkout, "rev-parse", "HEAD")
    branch = _git(checkout, "branch", "--show-current")
    tracked_dirty = bool(_git(checkout, "status", "--porcelain", "--untracked-files=no"))
    receipt = _receipt_path(deployment_root, revision)
    packs = []
    for pack in sorted((checkout / ".git/objects/pack").glob("pack-*.pack")):
        index = pack.with_suffix(".idx")
        if not index.is_file():
            raise FileNotFoundError(f"Git pack index is missing: {index}")
        packs.append({"pack": _identity(pack), "index": _identity(index)})
    if not packs:
        raise ValueError(f"deployment checkout has no Git packs: {checkout}")
    return {
        "path": checkout.resolve().as_posix(),
        "revision": revision,
        "branch": branch,
        "tracked_dirty": tracked_dirty,
        "receipt": _identity(receipt) if receipt.is_file() else None,
        "packs": packs,
    }


def _process_ancestry(process_root: Path) -> set[int]:
    ancestry = {os.getpid()}
    current = os.getpid()
    while current > 1:
        status = process_root / str(current) / "status"
        try:
            parent_line = next(
                line for line in status.read_text(encoding="utf-8").splitlines()
                if line.startswith("PPid:")
            )
            parent = int(parent_line.split(":", 1)[1].strip())
        except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration, ValueError):
            break
        if parent < 1 or parent in ancestry:
            break
        ancestry.add(parent)
        current = parent
    return ancestry


def _active_checkout_paths(process_root: Path = Path("/proc")) -> set[str]:
    active: set[str] = set()
    if not process_root.is_dir():
        return active
    ancestry = _process_ancestry(process_root)
    for process in process_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            cwd = (process / "cwd").resolve(strict=True)
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        current = cwd
        while current != current.parent:
            if CHECKOUT_PATTERN.fullmatch(current.name):
                active.add(current.as_posix())
                break
            current = current.parent
        if int(process.name) in ancestry:
            continue
        for match in re.finditer(r"/[^\s]*/checkout-[0-9a-f]{7}(?:/[^\s]*)?", command):
            value = Path(match.group(0))
            while value != value.parent:
                if CHECKOUT_PATTERN.fullmatch(value.name):
                    active.add(value.as_posix())
                    break
                value = value.parent
    return active


def build_deployment_pack_dedup_plan(
    *,
    deployment_root: Path,
    canonical_checkout: Path,
    active_checkout_paths: set[str] | None = None,
) -> dict[str, Any]:
    root = deployment_root.resolve()
    canonical = canonical_checkout.resolve()
    if canonical.parent != root:
        raise ValueError("canonical checkout must be directly under deployment root")
    active = {
        Path(path).resolve().as_posix()
        for path in (
            _active_checkout_paths() if active_checkout_paths is None else active_checkout_paths
        )
    }
    checkouts = [
        _checkout_state(root, path)
        for path in sorted(root.glob("checkout-*"))
        if path.is_dir()
    ]
    canonical_state = next(
        (item for item in checkouts if item["path"] == canonical.as_posix()), None
    )
    if canonical_state is None:
        raise ValueError("canonical deployment checkout is missing")
    if canonical_state["tracked_dirty"] or canonical_state["receipt"] is None:
        raise ValueError("canonical deployment checkout is not receipt-bound and clean")

    canonical_files = {}
    for group in canonical_state["packs"]:
        for kind in ("pack", "index"):
            source = group[kind]
            canonical_files[(Path(source["path"]).name, source["bytes"], source["sha256"])] = source

    actions = []
    counts = {"eligible": 0, "already_linked": 0, "required": 0, "indeterminate": 0}
    potential_bytes = 0
    for checkout in checkouts:
        reasons = []
        if checkout["path"] == canonical.as_posix():
            reasons.append("canonical_checkout")
        if checkout["path"] in active:
            reasons.append("active_process_checkout")
        if checkout["tracked_dirty"]:
            reasons.append("tracked_dirty")
        if checkout["receipt"] is None:
            reasons.append("missing_deployment_receipt")
        for group in checkout["packs"]:
            for kind in ("pack", "index"):
                target = group[kind]
                key = (Path(target["path"]).name, target["bytes"], target["sha256"])
                source = canonical_files.get(key)
                item_reasons = list(reasons)
                if "canonical_checkout" in item_reasons or "active_process_checkout" in item_reasons:
                    disposition = "required"
                elif source is None:
                    item_reasons.append("no_identical_canonical_file")
                    disposition = "indeterminate"
                elif target["device"] != source["device"]:
                    item_reasons.append("cross_device")
                    disposition = "indeterminate"
                elif target["inode"] == source["inode"]:
                    item_reasons.append("already_hardlinked")
                    disposition = "already_linked"
                elif item_reasons:
                    disposition = "indeterminate"
                else:
                    disposition = "eligible"
                    potential_bytes += target["bytes"]
                counts[disposition] += 1
                actions.append(
                    {
                        "checkout": checkout["path"],
                        "revision": checkout["revision"],
                        "receipt": checkout["receipt"],
                        "kind": kind,
                        "source": source,
                        "target": target,
                        "disposition": disposition,
                        "reasons": sorted(set(item_reasons)),
                    }
                )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "role": PLAN_ROLE,
        "status": "pass",
        "read_only": True,
        "deployment_root": root.as_posix(),
        "canonical_checkout": canonical.as_posix(),
        "active_checkouts": sorted(active),
        "checkouts": checkouts,
        "actions": actions,
        "summary": {
            "checkout_count": len(checkouts),
            "counts_by_disposition": counts,
            "potential_physical_bytes_saved": potential_bytes,
            "currently_saved_bytes": 0,
        },
    }


def verify_deployment_pack_dedup_plan(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != PLAN_SCHEMA_VERSION or report.get("role") != PLAN_ROLE:
        raise ValueError("deployment pack dedup plan identity is invalid")
    replay = build_deployment_pack_dedup_plan(
        deployment_root=Path(report.get("deployment_root", "")),
        canonical_checkout=Path(report.get("canonical_checkout", "")),
        active_checkout_paths=set(report.get("active_checkouts", [])),
    )
    if replay != report:
        raise ValueError("deployment pack dedup plan replay changed")
    return report


def _replace_with_hardlink(source: Path, target: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".hardlink.tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        os.link(source, temporary)
        if os.name == "nt":
            os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        os.replace(temporary, target)
    finally:
        if temporary.exists() and os.name == "nt":
            os.chmod(temporary, stat.S_IREAD | stat.S_IWRITE)
        temporary.unlink(missing_ok=True)


def _replay_receipt(checkout: Path, receipt: dict[str, Any]) -> None:
    receipt_path = Path(receipt["path"])
    if _identity(receipt_path) != receipt:
        raise ValueError("deployment receipt changed after planning")
    validator = checkout / "scripts/validate_generation_large_capacity_deployment.py"
    if not validator.is_file():
        raise FileNotFoundError(f"deployment receipt validator is missing: {validator}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(checkout / "src")
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--receipt",
            str(receipt_path),
            "--expected-receipt-sha256",
            receipt["sha256"],
        ],
        cwd=checkout,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def apply_deployment_pack_dedup_plan(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    expected_plan_sha256: str,
    output_path: Path,
    allow_checkouts: set[str],
) -> dict[str, Any]:
    if file_sha256(plan_path) != expected_plan_sha256:
        raise ValueError("deployment pack dedup plan SHA256 differs")
    verify_deployment_pack_dedup_plan(plan)
    allowed = {Path(path).resolve().as_posix() for path in allow_checkouts}
    eligible = [item for item in plan["actions"] if item["disposition"] == "eligible"]
    eligible_checkouts = {item["checkout"] for item in eligible}
    if not allowed or allowed != eligible_checkouts:
        raise ValueError("explicit allow-checkout set must equal the eligible checkout set")
    active_now = _active_checkout_paths()
    newly_active = sorted(
        {item["checkout"] for item in eligible} & active_now
    )
    if newly_active:
        raise ValueError(
            "eligible deployment checkout became active after planning: "
            + ", ".join(newly_active)
        )
    if output_path.exists():
        raise FileExistsError(f"deployment pack dedup result already exists: {output_path}")
    checkout_states = {item["path"]: item for item in plan["checkouts"]}
    deployment_root = Path(plan["deployment_root"])
    receipts = {}
    for checkout in sorted(allowed):
        repository = Path(checkout)
        if _checkout_state(deployment_root, repository) != checkout_states[checkout]:
            raise ValueError("deployment checkout changed after planning")
        subprocess.run(["git", "fsck", "--strict"], cwd=repository, check=True)
        receipt = next(
            item["receipt"] for item in eligible if item["checkout"] == checkout
        )
        _replay_receipt(repository, receipt)
        receipts[checkout] = receipt
    for item in eligible:
        if _identity(Path(item["source"]["path"])) != item["source"] or _identity(
            Path(item["target"]["path"])
        ) != item["target"]:
            raise ValueError("deployment pack file changed after planning")
        if item["receipt"] is None or _identity(Path(item["receipt"]["path"])) != item[
            "receipt"
        ]:
            raise ValueError("deployment receipt changed after planning")
    before_free = shutil.disk_usage(Path(plan["deployment_root"])).free
    applied_pairs = []
    for item in eligible:
        source = Path(item["source"]["path"])
        target = Path(item["target"]["path"])
        _replace_with_hardlink(source, target)
        current = _identity(target)
        if current["sha256"] != item["target"]["sha256"] or not os.path.samefile(
            source, target
        ):
            raise ValueError("deployment pack hardlink verification failed")
        applied_pairs.append((source, target))
    checked = []
    for checkout in sorted(allowed):
        repository = Path(checkout)
        subprocess.run(["git", "fsck", "--strict"], cwd=repository, check=True)
        if _git(repository, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("deployment checkout became tracked-dirty after dedup")
        _replay_receipt(repository, receipts[checkout])
        checked.append(
            {
                "path": repository.as_posix(),
                "revision": _git(repository, "rev-parse", "HEAD"),
                "git_fsck": "pass",
                "deployment_receipt_replay": "pass",
            }
        )
    applied = [
        {"source": _identity(source), "target": _identity(target)}
        for source, target in applied_pairs
    ]
    after_free = shutil.disk_usage(Path(plan["deployment_root"])).free
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "role": RESULT_ROLE,
        "status": "pass",
        "read_only": False,
        "destructive_deletion_performed": False,
        "plan": {
            "path": plan_path.resolve().as_posix(),
            "bytes": plan_path.stat().st_size,
            "sha256": expected_plan_sha256,
        },
        "allowed_checkouts": sorted(allowed),
        "applied": applied,
        "verified_checkouts": checked,
        "filesystem_free_bytes_before": before_free,
        "filesystem_free_bytes_after": after_free,
        "observed_free_bytes_delta": after_free - before_free,
        "expected_physical_bytes_saved": plan["summary"][
            "potential_physical_bytes_saved"
        ],
    }
    write_json_report(output_path, result)
    return result


def verify_deployment_pack_dedup_result(
    report: dict[str, Any],
    *,
    result_path: Path,
    expected_result_sha256: str,
) -> dict[str, Any]:
    if file_sha256(result_path) != expected_result_sha256:
        raise ValueError("deployment pack dedup result SHA256 differs")
    if (
        report.get("schema_version") != RESULT_SCHEMA_VERSION
        or report.get("role") != RESULT_ROLE
        or report.get("status") != "pass"
        or report.get("destructive_deletion_performed") is not False
    ):
        raise ValueError("deployment pack dedup result identity is invalid")
    plan = report.get("plan", {})
    plan_path = Path(plan.get("path", ""))
    if (
        not plan_path.is_file()
        or plan_path.stat().st_size != plan.get("bytes")
        or file_sha256(plan_path) != plan.get("sha256")
    ):
        raise ValueError("deployment pack dedup source plan changed")
    for item in report.get("applied", []):
        source = Path(item["source"]["path"])
        target = Path(item["target"]["path"])
        current_source = _identity(source)
        current_target = _identity(target)
        if (
            current_source != item["source"]
            or current_target != item["target"]
            or not os.path.samefile(source, target)
        ):
            raise ValueError("deployment pack dedup hardlink changed")
    for checkout in report.get("verified_checkouts", []):
        repository = Path(checkout["path"])
        subprocess.run(["git", "fsck", "--strict"], cwd=repository, check=True)
        if _git(repository, "rev-parse", "HEAD") != checkout["revision"]:
            raise ValueError("deployment checkout revision changed after dedup")
        receipt = next(
            action["receipt"]
            for action in json.loads(plan_path.read_text(encoding="utf-8"))["actions"]
            if action["checkout"] == repository.as_posix()
            and action["disposition"] == "eligible"
        )
        _replay_receipt(repository, receipt)
    return report
