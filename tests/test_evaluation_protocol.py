from pathlib import Path

import pytest

from src.evaluation.protocol import (
    EvaluationProtocol,
    load_evaluation_protocol,
    validate_evaluation_protocol,
)


def _valid_split_meta() -> dict:
    return {
        "seed": 42,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "split_strategy": "stratified",
        "stratify_bins": 8,
    }


def test_validate_protocol_passes_for_matching_context():
    protocol = EvaluationProtocol()
    result = validate_evaluation_protocol(
        protocol,
        split_artifact_path="./splits/cd_split_seed42.pt",
        split_metadata=_valid_split_meta(),
        num_neg_eval=20,
        eval_hard_negative_ratio=0.0,
        runtime_seed=42,
        runtime_val_ratio=0.1,
        runtime_test_ratio=0.1,
        runtime_split_strategy="stratified",
        runtime_stratify_bins=8,
    )
    assert result.comparable is True
    assert result.violations == ()
    assert result.warnings == ()


def test_validate_protocol_fails_on_num_neg_eval_mismatch():
    protocol = EvaluationProtocol(num_neg_eval=20)
    result = validate_evaluation_protocol(
        protocol,
        split_artifact_path="./splits/cd_split_seed42.pt",
        split_metadata=_valid_split_meta(),
        num_neg_eval=50,
        eval_hard_negative_ratio=0.0,
    )
    assert result.comparable is False
    assert any("num_neg_eval" in msg for msg in result.violations)


def test_validate_protocol_fails_when_split_artifact_required_but_missing():
    protocol = EvaluationProtocol(require_split_artifact=True)
    result = validate_evaluation_protocol(
        protocol,
        split_artifact_path=None,
        split_metadata=_valid_split_meta(),
        num_neg_eval=20,
        eval_hard_negative_ratio=0.0,
    )
    assert result.comparable is False
    assert any("split_artifact_path" in msg for msg in result.violations)


def test_validate_protocol_fails_when_required_metadata_key_missing():
    protocol = EvaluationProtocol()
    meta = _valid_split_meta()
    meta.pop("stratify_bins")
    result = validate_evaluation_protocol(
        protocol,
        split_artifact_path="./splits/cd_split_seed42.pt",
        split_metadata=meta,
        num_neg_eval=20,
        eval_hard_negative_ratio=0.0,
    )
    assert result.comparable is False
    assert any("missing required keys" in msg for msg in result.violations)


def test_validate_protocol_allow_noncomparable_moves_violations_to_warnings():
    protocol = EvaluationProtocol()
    result = validate_evaluation_protocol(
        protocol,
        split_artifact_path=None,
        split_metadata=None,
        num_neg_eval=5,
        eval_hard_negative_ratio=0.3,
        allow_noncomparable=True,
    )
    assert result.comparable is False
    assert len(result.violations) > 0
    assert len(result.warnings) == len(result.violations)


def test_load_evaluation_protocol_from_yaml(tmp_path: Path):
    path = tmp_path / "eval_protocol.yaml"
    path.write_text(
        "\n".join(
            [
                "protocol:",
                "  version: 1",
                "  num_neg_eval: 20",
                "  eval_hard_negative_ratio: 0.0",
                "  require_split_artifact: true",
                "  required_split_metadata_keys: [seed, val_ratio, test_ratio, split_strategy, stratify_bins]",
                "  allowed_split_strategy: [stratified]",
            ]
        )
    )
    protocol = load_evaluation_protocol(path)
    assert protocol.num_neg_eval == 20
    assert protocol.require_split_artifact is True


def test_validate_protocol_detects_runtime_split_metadata_mismatch():
    protocol = EvaluationProtocol()
    result = validate_evaluation_protocol(
        protocol,
        split_artifact_path="./splits/cd_split_seed42.pt",
        split_metadata=_valid_split_meta(),
        num_neg_eval=20,
        eval_hard_negative_ratio=0.0,
        runtime_seed=7,
    )
    assert result.comparable is False
    assert any("runtime seed" in msg for msg in result.violations)

