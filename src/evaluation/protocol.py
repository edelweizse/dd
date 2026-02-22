"""Evaluation protocol loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


_DEFAULT_REQUIRED_META_KEYS: Tuple[str, ...] = (
    "seed",
    "val_ratio",
    "test_ratio",
    "split_strategy",
    "stratify_bins",
)


@dataclass(frozen=True)
class EvaluationProtocol:
    """Declarative protocol used to enforce comparable evaluation runs."""

    version: int = 1
    num_neg_eval: int = 20
    eval_hard_negative_ratio: float = 0.0
    require_split_artifact: bool = True
    required_split_metadata_keys: Tuple[str, ...] = _DEFAULT_REQUIRED_META_KEYS
    allowed_split_strategy: Tuple[str, ...] = ("stratified",)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvaluationProtocol":
        if not isinstance(data, Mapping):
            raise ValueError("Protocol config must be a mapping.")

        keys = set(cls.__dataclass_fields__.keys())
        clean: Dict[str, Any] = {}
        for key in keys:
            if key in data:
                clean[key] = data[key]

        if "required_split_metadata_keys" in clean:
            clean["required_split_metadata_keys"] = tuple(str(v) for v in clean["required_split_metadata_keys"])
        if "allowed_split_strategy" in clean:
            clean["allowed_split_strategy"] = tuple(str(v).strip().lower() for v in clean["allowed_split_strategy"])

        proto = cls(**clean)
        proto.validate()
        return proto

    def validate(self) -> None:
        if int(self.version) <= 0:
            raise ValueError(f"protocol.version must be > 0, got {self.version}")
        if int(self.num_neg_eval) <= 0:
            raise ValueError(f"protocol.num_neg_eval must be > 0, got {self.num_neg_eval}")
        hnr = float(self.eval_hard_negative_ratio)
        if hnr < 0.0 or hnr > 1.0:
            raise ValueError(
                "protocol.eval_hard_negative_ratio must be in [0,1], "
                f"got {self.eval_hard_negative_ratio}"
            )
        if not self.required_split_metadata_keys:
            raise ValueError("protocol.required_split_metadata_keys must be non-empty")
        if not self.allowed_split_strategy:
            raise ValueError("protocol.allowed_split_strategy must be non-empty")


@dataclass(frozen=True)
class ProtocolValidationResult:
    """Protocol validation result for logging and output payloads."""

    comparable: bool
    violations: Tuple[str, ...]
    warnings: Tuple[str, ...]
    checks: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparable": bool(self.comparable),
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "checks": self.checks,
        }


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for evaluation protocol config support. "
            "Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML protocol config must be a mapping at root: {path}")
    return data


def load_evaluation_protocol(config_path: str | Path) -> EvaluationProtocol:
    """Load an evaluation protocol from YAML path."""
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Evaluation protocol config not found: {path}")
    raw = _load_yaml(path)
    # Allow either flat root fields or nested under `protocol`.
    protocol_data = raw.get("protocol", raw)
    return EvaluationProtocol.from_mapping(protocol_data)


def _is_close_float(a: float, b: float, eps: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= float(eps)


def _meta_value(meta: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(meta, Mapping):
        return None
    return meta.get(key)


def _maybe_add_runtime_match_violation(
    violations: list[str],
    *,
    name: str,
    runtime_value: Any,
    metadata_value: Any,
) -> None:
    if runtime_value is None or metadata_value is None:
        return

    if isinstance(runtime_value, float) or isinstance(metadata_value, float):
        same = _is_close_float(float(runtime_value), float(metadata_value))
    elif isinstance(runtime_value, str) or isinstance(metadata_value, str):
        same = str(runtime_value).strip().lower() == str(metadata_value).strip().lower()
    else:
        same = runtime_value == metadata_value

    if not same:
        violations.append(
            f"runtime {name}={runtime_value!r} does not match split metadata {name}={metadata_value!r}"
        )


def validate_evaluation_protocol(
    protocol: EvaluationProtocol,
    *,
    split_artifact_path: str | None,
    split_metadata: Mapping[str, Any] | None,
    num_neg_eval: int,
    eval_hard_negative_ratio: float,
    runtime_seed: int | None = None,
    runtime_val_ratio: float | None = None,
    runtime_test_ratio: float | None = None,
    runtime_split_strategy: str | None = None,
    runtime_stratify_bins: int | None = None,
    allow_noncomparable: bool = False,
) -> ProtocolValidationResult:
    """Validate run-time evaluation context against required protocol."""
    protocol.validate()
    violations: list[str] = []
    warnings: list[str] = []

    if protocol.require_split_artifact and not split_artifact_path:
        violations.append("split_artifact_path is required by protocol but not provided")

    if int(num_neg_eval) != int(protocol.num_neg_eval):
        violations.append(
            f"num_neg_eval={num_neg_eval} does not match required {protocol.num_neg_eval}"
        )

    if not _is_close_float(float(eval_hard_negative_ratio), float(protocol.eval_hard_negative_ratio)):
        violations.append(
            f"eval_hard_negative_ratio={eval_hard_negative_ratio} does not match required "
            f"{protocol.eval_hard_negative_ratio}"
        )

    if protocol.required_split_metadata_keys:
        if not isinstance(split_metadata, Mapping):
            violations.append("split metadata is unavailable; required keys cannot be verified")
        else:
            missing = [k for k in protocol.required_split_metadata_keys if k not in split_metadata]
            if missing:
                violations.append(f"split metadata missing required keys: {missing}")

            meta_strategy = split_metadata.get("split_strategy")
            if meta_strategy is not None:
                norm = str(meta_strategy).strip().lower()
                if norm not in set(protocol.allowed_split_strategy):
                    violations.append(
                        "split metadata strategy "
                        f"{meta_strategy!r} is not in allowed set {list(protocol.allowed_split_strategy)}"
                    )

            _maybe_add_runtime_match_violation(
                violations,
                name="seed",
                runtime_value=runtime_seed,
                metadata_value=_meta_value(split_metadata, "seed"),
            )
            _maybe_add_runtime_match_violation(
                violations,
                name="val_ratio",
                runtime_value=runtime_val_ratio,
                metadata_value=_meta_value(split_metadata, "val_ratio"),
            )
            _maybe_add_runtime_match_violation(
                violations,
                name="test_ratio",
                runtime_value=runtime_test_ratio,
                metadata_value=_meta_value(split_metadata, "test_ratio"),
            )
            _maybe_add_runtime_match_violation(
                violations,
                name="split_strategy",
                runtime_value=runtime_split_strategy,
                metadata_value=_meta_value(split_metadata, "split_strategy"),
            )
            _maybe_add_runtime_match_violation(
                violations,
                name="stratify_bins",
                runtime_value=runtime_stratify_bins,
                metadata_value=_meta_value(split_metadata, "stratify_bins"),
            )

    if violations and bool(allow_noncomparable):
        warnings.extend(violations)

    comparable = not bool(violations)
    checks: Dict[str, Any] = {
        "required_num_neg_eval": int(protocol.num_neg_eval),
        "observed_num_neg_eval": int(num_neg_eval),
        "required_eval_hard_negative_ratio": float(protocol.eval_hard_negative_ratio),
        "observed_eval_hard_negative_ratio": float(eval_hard_negative_ratio),
        "require_split_artifact": bool(protocol.require_split_artifact),
        "observed_split_artifact_path": split_artifact_path,
        "allowed_split_strategy": list(protocol.allowed_split_strategy),
        "required_split_metadata_keys": list(protocol.required_split_metadata_keys),
    }
    return ProtocolValidationResult(
        comparable=comparable,
        violations=tuple(violations),
        warnings=tuple(warnings),
        checks=checks,
    )


def format_protocol_violations(violations: Sequence[str], *, prefix: str = "Protocol violation") -> str:
    """Human-readable multiline violation message."""
    if not violations:
        return ""
    lines = [prefix + ":"]
    lines.extend(f"  - {msg}" for msg in violations)
    return "\n".join(lines)

