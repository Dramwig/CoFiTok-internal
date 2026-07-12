import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "baselines" / "clone_baseline_repos.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("clone_baseline_repos", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_clone_entries_defaults_to_marked_external_repos() -> None:
    module = load_script_module()
    registry = {
        "entries": [
            {
                "alias": "same_backbone_dense",
                "priority": "P0",
                "repo_type": "internal",
                "clone_on_setup": False,
            },
            {
                "alias": "edm",
                "priority": "P0",
                "repo_type": "external",
                "clone_on_setup": True,
            },
            {
                "alias": "selftok",
                "priority": "P2",
                "repo_type": "external",
                "clone_on_setup": False,
            },
        ]
    }

    selected = module.select_clone_entries(registry)

    assert [entry["alias"] for entry in selected] == ["edm"]


def test_dry_run_writes_manifest_without_git(tmp_path: Path) -> None:
    module = load_script_module()
    registry_path = tmp_path / "registry.json"
    manifest_path = tmp_path / "manifest.json"
    repo_root = tmp_path / "repos"
    registry = {
        "schema_version": 1,
        "project": "CoFiTok",
        "entries": [
            {
                "alias": "edm",
                "priority": "P0",
                "family": "pixel_diffusion",
                "method_name": "EDM",
                "repo_type": "external",
                "repo_url": "https://github.com/NVlabs/edm",
                "clone_on_setup": True,
            }
        ],
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    exit_code = module.main(
        [
            "--registry",
            str(registry_path),
            "--repo-root",
            str(repo_root),
            "--manifest-output",
            str(manifest_path),
            "--dry-run",
        ]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["failed_count"] == 0
    assert manifest["records"][0]["status"] == "dry_run"
    assert manifest["records"][0]["target_path"] == str(repo_root / "edm")
