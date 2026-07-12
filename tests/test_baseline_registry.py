import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "baselines" / "registry.json"


def load_registry() -> dict:
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_baseline_registry_has_required_methods() -> None:
    registry = load_registry()
    aliases = {entry["alias"] for entry in registry["entries"]}

    assert {
        "same_backbone_dense",
        "endpoint_only_factorized",
        "edm",
        "improved_diffusion",
        "d_ar",
        "ml_flextok",
        "mar",
        "titok_1d_tokenizer",
        "retok",
    }.issubset(aliases)


def test_baseline_registry_keeps_external_repos_remote() -> None:
    registry = load_registry()
    repo_root = registry["remote_repo_root"]

    assert repo_root == "/root/autodl-tmp/CoFiTok/baselines/repos"
    assert "CoFiTok-internal" not in repo_root

    seen: set[str] = set()
    for entry in registry["entries"]:
        alias = entry["alias"]
        assert re.fullmatch(r"[a-z0-9_]+", alias)
        assert alias not in seen
        seen.add(alias)

        if entry["repo_type"] == "external":
            assert entry["repo_url"].startswith("https://github.com/")
        else:
            assert entry["repo_url"] is None
            assert entry["clone_on_setup"] is False


def test_first_pass_clone_set_is_p0_p1_only() -> None:
    registry = load_registry()
    clone_entries = [entry for entry in registry["entries"] if entry["clone_on_setup"]]

    assert clone_entries
    assert {entry["priority"] for entry in clone_entries}.issubset({"P0", "P1"})
    assert all(entry["repo_type"] == "external" for entry in clone_entries)
    assert not any(entry["priority"] == "P2" for entry in clone_entries)
