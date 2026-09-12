"""Read-only, independent arithmetic audit of the frozen class-support readout.

This is an observer, not the qualification validator or an execution gate. It
does not import the contingency implementation, load a model, or write remotely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binom, hypergeom


EXPECTED = {
    "class_support_contingency_result.json":
        "9d1a5709a18f5beadf7a8654dd147906a269b35ce35ef0720a68ab2c484c0dc1",
    "class_support_contingency_result.validation.json":
        "6ad46a2cece75f17038ddb2c32fe08665db372909e728e956b34e02168855dae",
}


def identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError(f"file changed during read: {path}")
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def independent_summary(rows: list[dict], classes: int) -> dict:
    """Rebuild a dense contingency directly; hypergeom supplies the AMI null."""
    count = len(rows)
    if not count or count % classes:
        raise ValueError("nonempty balanced complete class occurrences required")
    table = np.zeros((classes, classes), dtype=np.int64)
    top1 = top5 = 0
    hashes: list[str] = []
    for index, row in enumerate(rows):
        label = row["requested_class"]
        pred = row["predicted_top1_class"]
        if (row["sample_index"] != index or label != index % classes
                or row["filename"] != f"{index:06d}.png"
                or not 0 <= pred < classes):
            raise ValueError("prediction index/label/filename contract differs")
        top = row["predicted_top5_classes"]
        if len(top) != 5 or len(set(top)) != 5 or top[0] != pred:
            raise ValueError("top-five class contract differs")
        if any(not 0 <= value < classes for value in top):
            raise ValueError("top-five class out of range")
        if not math.isfinite(row["requested_probability"]):
            raise ValueError("nonfinite requested probability")
        table[label, pred] += 1
        top1 += label == pred
        top5 += label in top
        hashes.append(row["image_sha256"])
    requested, predicted = table.sum(axis=1), table.sum(axis=0)
    entropy = lambda hist: -sum(float(n / count) * math.log(float(n / count))
                                for n in hist if n)
    hr, hp = entropy(requested), entropy(predicted)
    ri, pi = np.nonzero(table)
    cells = table[ri, pi].astype(np.float64)
    mi = float(np.sum(cells / count * np.log(
        cells * count / (requested[ri] * predicted[pi]))))
    # Sum over distinct margin counts, independent of the producer's combinatorics.
    expected_mi = 0.0
    for a, af in Counter(requested.tolist()).items():
        for b, bf in Counter(predicted.tolist()).items():
            if not a or not b:
                continue
            possible = np.arange(max(1, a + b - count), min(a, b) + 1)
            terms = possible / count * np.log(possible * count / (a * b))
            expected_mi += af * bf * float(np.sum(
                terms * hypergeom.pmf(possible, count, a, b)))
    denominator = (hr + hp) / 2 - expected_mi
    ami = (mi - expected_mi) / denominator if denominator > 1e-12 else 1.0
    offsets = [sum(int(table[y, (y + offset) % classes])
                   for y in range(classes)) for offset in range(classes)]
    return {
        "count": count, "top1_correct": top1, "top5_correct": top5,
        "top1_accuracy": top1 / count, "top5_accuracy": top5 / count,
        "top1_binomial_tail_iid_assumption": float(binom.sf(top1 - 1, count, 1 / classes)),
        "top5_binomial_tail_iid_assumption": float(binom.sf(top5 - 1, count, 5 / classes)),
        "mean_requested_probability": math.fsum(r["requested_probability"] for r in rows) / count,
        "adjusted_mutual_information": ami,
        "mutual_information_nats": mi,
        "expected_mutual_information_nats": expected_mi,
        "predicted_class_count": int(np.count_nonzero(predicted)),
        "predicted_entropy_nats": hp, "effective_class_count": math.exp(hp),
        "top_mode_class": int(np.argmax(predicted)),
        "top_mode_count": int(np.max(predicted)),
        "unique_image_hashes": len(set(hashes)),
        "best_cyclic_offset": int(np.argmax(offsets)),
        "best_cyclic_accuracy": max(offsets) / count,
    }


def close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-11):
        raise ValueError(f"independent calculation differs: {name}: {actual} != {expected}")


def audit(root: Path) -> dict:
    sources = {}
    frozen = {}
    for name, expected_sha in EXPECTED.items():
        path = root / name
        sources[name] = identity(path)
        if sources[name]["sha256"] != expected_sha:
            raise ValueError(f"frozen source hash differs: {name}")
        frozen[name] = json.loads(path.read_text())
    result = frozen["class_support_contingency_result.json"]
    receipt = frozen["class_support_contingency_result.validation.json"]
    if receipt["status"] != "pass" or receipt["result"] != sources[
        "class_support_contingency_result.json"
    ]:
        raise ValueError("independent validator receipt does not bind the result")
    summaries, predictions = {}, {}
    for method, bound in result["prediction_files"].items():
        path = Path(bound["path"])
        sources[method] = identity(path)
        if sources[method] != bound or bound != receipt["prediction_files"][method]:
            raise ValueError(f"prediction binding differs: {method}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 10000:
            raise ValueError("expected exactly 10000 predictions per method")
        predictions[method] = rows
        summary = independent_summary(rows, 1000)
        ref = result["analysis"]["methods"][method]
        for key in ("top1_correct", "top5_correct", "top1_accuracy", "top5_accuracy",
                    "mean_requested_probability"):
            close(summary[key], ref["direct_class_fidelity"][key], f"{method}.{key}")
        for key in ("adjusted_mutual_information", "mutual_information_nats",
                    "expected_mutual_information_nats"):
            close(summary[key], ref["adjusted_mutual_information"][key], f"{method}.{key}")
        for key in ("predicted_class_count", "effective_class_count", "top_mode_class",
                    "top_mode_count"):
            close(summary[key], ref["predicted_class_support"][key], f"{method}.{key}")
        cyclic = ref["best_cyclic_offset_alignment"]
        close(summary["best_cyclic_offset"], cyclic["best_offset"], "cyclic offset")
        close(summary["best_cyclic_accuracy"], cyclic["best_accuracy"], "cyclic accuracy")
        mapped_correct = total = 0
        for fold in ref["best_one_to_one_mapping"]["two_fold_occurrence_parity"]:
            training_parity = fold["training_occurrence_parity"]
            mapping = fold["mapping"]
            for row in rows:
                if (row["sample_index"] // 1000) % 2 != training_parity:
                    total += 1
                    mapped_correct += mapping[row["requested_class"]] == row["predicted_top1_class"]
        if total != 10000:
            raise ValueError("heldout fold count differs")
        summary["heldout_mapping_accuracy"] = mapped_correct / total
        close(summary["heldout_mapping_accuracy"], ref["best_one_to_one_mapping"][
            "cross_validated_accuracy"], "heldout mapping accuracy")
        summary["permutation_pvalue_from_validated_result"] = ref[
            "adjusted_mutual_information"]["deterministic_permutation_null"]["pvalue_greater"]
        summaries[method] = summary
    left, right = predictions["cofitok"], predictions["dense_identity"]
    histogram = lambda rows: np.bincount([r["predicted_top1_class"] for r in rows], minlength=1000) / len(rows)
    a, b = histogram(left), histogram(right)
    mean = (a + b) / 2
    kl = lambda p: float(np.sum(p[p > 0] * np.log2(p[p > 0] / mean[p > 0])))
    cross = {
        "histogram_overlap": float(np.minimum(a, b).sum()),
        "jensen_shannon_divergence_bits": (kl(a) + kl(b)) / 2,
        "same_index_top1_count": sum(x["predicted_top1_class"] == y["predicted_top1_class"]
                                     for x, y in zip(left, right)),
        "shared_image_hash_count": len({r["image_sha256"] for r in left}
                                       & {r["image_sha256"] for r in right}),
    }
    for key in ("histogram_overlap", "jensen_shannon_divergence_bits"):
        close(cross[key], result["analysis"]["cross_method"]["predicted_histograms"][key], key)
    close(cross["same_index_top1_count"], result["analysis"]["cross_method"][
        "same_index_top1_agreement"]["same_index_count"], "same index agreement")
    for source in sources.values():
        if identity(Path(source["path"])) != source:
            raise ValueError("bound source changed during observer audit")
    return {
        "schema_version": "cofitok_class_support_readout_observer_v1",
        "status": "pass", "sources": sources, "methods": summaries, "cross_method": cross,
        "scope": "independent arithmetic, not classifier reinference or a scientific release gate",
        "not_independently_recomputed": ["512-replicate permutation nulls", "classifier logits"],
        "claim": "weak detectable label association; no heldout relabeling rescue; shared support concentration",
        "caveats": [
            "Predeclared AMI practical floor 0.01 fails despite significant permutation association.",
            "The frozen label shared_unconditional_support_collapse is a decision label, not proof of independence.",
            "Binomial tails assume iid trials; balanced labels and correlated samples limit that interpretation.",
            "Same seed/index couples the two methods; agreement does not establish a training-time cause.",
            "Only 10 images per requested class; lack of mapping rescue does not rule out every possible mapping.",
        ],
        "common_cause_proven": False, "training_or_sampling_authorized": False,
        "confirmation_authorized": False, "full_300k_authorized": False,
        "release_or_paper_integration_authorized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    report["observer_source"] = identity(Path(__file__))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
