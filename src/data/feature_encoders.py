"""
Feature encoders for converting metadata dictionaries into numeric tensors.

This module is intentionally model-agnostic and can be used to prepare
node/edge features for any downstream architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch


Record = Mapping[str, Any]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _field_get(record: Record, field: str, default: Any = None) -> Any:
    """Read field value with optional dot-path support."""
    value: Any = record
    for part in field.split('.'):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == '')


def _stable_hash_to_index(value: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    hashed = int.from_bytes(digest, byteorder="little", signed=False)
    return int(hashed % int(num_buckets))


def _tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


class BaseFieldEncoder:
    """Base interface for field-level encoders."""

    def fit(self, records: Sequence[Record]) -> "BaseFieldEncoder":
        return self

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, records: Sequence[Record]) -> np.ndarray:
        return self.fit(records).transform(records)

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def feature_names(self) -> List[str]:
        raise NotImplementedError


@dataclass
class NumericFieldEncoder(BaseFieldEncoder):
    """Encode scalar numeric values into one standardized column."""

    field: str
    fill_value: float = 0.0
    log1p: bool = False
    standardize: bool = False
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    mean_: float = 0.0
    std_: float = 1.0

    def _parse(self, value: Any) -> float:
        if _is_missing(value):
            x = float(self.fill_value)
        elif isinstance(value, bool):
            x = float(value)
        else:
            try:
                x = float(value)
            except (TypeError, ValueError):
                x = float(self.fill_value)

        if self.log1p:
            x = math.log1p(max(0.0, x))
        if self.clip_min is not None:
            x = max(float(self.clip_min), x)
        if self.clip_max is not None:
            x = min(float(self.clip_max), x)
        return x

    def fit(self, records: Sequence[Record]) -> "NumericFieldEncoder":
        if not self.standardize:
            return self
        vals = np.asarray([self._parse(_field_get(rec, self.field)) for rec in records], dtype=np.float32)
        if vals.size == 0:
            self.mean_ = 0.0
            self.std_ = 1.0
            return self
        self.mean_ = float(vals.mean())
        std = float(vals.std())
        self.std_ = std if std > 1e-12 else 1.0
        return self

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        vals = np.asarray([self._parse(_field_get(rec, self.field)) for rec in records], dtype=np.float32).reshape(-1, 1)
        if self.standardize:
            vals = (vals - self.mean_) / self.std_
        return vals

    @property
    def output_dim(self) -> int:
        return 1

    def feature_names(self) -> List[str]:
        name = f"{self.field}:numeric"
        if self.standardize:
            name += ":z"
        return [name]


@dataclass
class BooleanFieldEncoder(BaseFieldEncoder):
    """Encode bool-like scalar values into 0/1."""

    field: str
    true_values: Sequence[str] = ("1", "true", "yes", "y", "t")
    true_values_norm_: set[str] | None = None

    def __post_init__(self) -> None:
        self.true_values_norm_ = {v.lower() for v in self.true_values}

    def _to_bool(self, value: Any) -> float:
        if _is_missing(value):
            return 0.0
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value != 0)
        if isinstance(value, str):
            return float(value.strip().lower() in (self.true_values_norm_ or set()))
        return 0.0

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        vals = [self._to_bool(_field_get(rec, self.field)) for rec in records]
        return np.asarray(vals, dtype=np.float32).reshape(-1, 1)

    @property
    def output_dim(self) -> int:
        return 1

    def feature_names(self) -> List[str]:
        return [f"{self.field}:bool"]


@dataclass
class CategoryOneHotEncoder(BaseFieldEncoder):
    """One-hot encoder for scalar categorical values."""

    field: str
    min_freq: int = 1
    lowercase: bool = True
    add_unknown: bool = True

    vocab_: Dict[str, int] | None = None

    def _norm(self, value: Any) -> Optional[str]:
        if _is_missing(value):
            return None
        text = str(value).strip()
        return text.lower() if self.lowercase else text

    def fit(self, records: Sequence[Record]) -> "CategoryOneHotEncoder":
        counts: Dict[str, int] = {}
        for rec in records:
            token = self._norm(_field_get(rec, self.field))
            if token is None:
                continue
            counts[token] = counts.get(token, 0) + 1
        tokens = sorted([t for t, c in counts.items() if c >= int(self.min_freq)])
        self.vocab_ = {tok: i for i, tok in enumerate(tokens)}
        return self

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        if self.vocab_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before transform.")
        dim = self.output_dim
        out = np.zeros((len(records), dim), dtype=np.float32)
        unk_idx = len(self.vocab_) if self.add_unknown else None
        for i, rec in enumerate(records):
            token = self._norm(_field_get(rec, self.field))
            if token is None:
                continue
            idx = self.vocab_.get(token)
            if idx is not None:
                out[i, idx] = 1.0
            elif unk_idx is not None:
                out[i, unk_idx] = 1.0
        return out

    @property
    def output_dim(self) -> int:
        if self.vocab_ is None:
            raise RuntimeError(f"{self.__class__.__name__} has no fitted vocabulary.")
        return len(self.vocab_) + (1 if self.add_unknown else 0)

    def feature_names(self) -> List[str]:
        if self.vocab_ is None:
            raise RuntimeError(f"{self.__class__.__name__} has no fitted vocabulary.")
        names = [f"{self.field}:cat={tok}" for tok, _ in sorted(self.vocab_.items(), key=lambda kv: kv[1])]
        if self.add_unknown:
            names.append(f"{self.field}:cat=<unk>")
        return names


@dataclass
class MultiCategoryEncoder(BaseFieldEncoder):
    """Multi-hot (or count) encoder for list-like categorical fields."""

    field: str
    min_freq: int = 1
    lowercase: bool = True
    binary: bool = True
    add_unknown: bool = True
    delimiter: Optional[str] = None

    vocab_: Dict[str, int] | None = None

    def _items(self, value: Any) -> List[str]:
        if _is_missing(value):
            return []
        raw: List[Any]
        if isinstance(value, str) and self.delimiter is not None:
            raw = [v.strip() for v in value.split(self.delimiter)]
        elif isinstance(value, (list, tuple, set)):
            raw = list(value)
        else:
            raw = [value]
        out: List[str] = []
        for item in raw:
            if _is_missing(item):
                continue
            text = str(item).strip()
            if text == "":
                continue
            out.append(text.lower() if self.lowercase else text)
        return out

    def fit(self, records: Sequence[Record]) -> "MultiCategoryEncoder":
        counts: Dict[str, int] = {}
        for rec in records:
            for token in self._items(_field_get(rec, self.field)):
                counts[token] = counts.get(token, 0) + 1
        tokens = sorted([t for t, c in counts.items() if c >= int(self.min_freq)])
        self.vocab_ = {tok: i for i, tok in enumerate(tokens)}
        return self

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        if self.vocab_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before transform.")
        dim = self.output_dim
        out = np.zeros((len(records), dim), dtype=np.float32)
        unk_idx = len(self.vocab_) if self.add_unknown else None
        for i, rec in enumerate(records):
            for token in self._items(_field_get(rec, self.field)):
                idx = self.vocab_.get(token)
                if idx is None:
                    if unk_idx is not None:
                        if self.binary:
                            out[i, unk_idx] = 1.0
                        else:
                            out[i, unk_idx] += 1.0
                    continue
                if self.binary:
                    out[i, idx] = 1.0
                else:
                    out[i, idx] += 1.0
        return out

    @property
    def output_dim(self) -> int:
        if self.vocab_ is None:
            raise RuntimeError(f"{self.__class__.__name__} has no fitted vocabulary.")
        return len(self.vocab_) + (1 if self.add_unknown else 0)

    def feature_names(self) -> List[str]:
        if self.vocab_ is None:
            raise RuntimeError(f"{self.__class__.__name__} has no fitted vocabulary.")
        names = [f"{self.field}:token={tok}" for tok, _ in sorted(self.vocab_.items(), key=lambda kv: kv[1])]
        if self.add_unknown:
            names.append(f"{self.field}:token=<unk>")
        return names


@dataclass
class TextHashingEncoder(BaseFieldEncoder):
    """Hashing bag-of-words encoder for free text or list-of-text fields."""

    field: str
    n_features: int = 128
    lowercase: bool = True
    ngram_min: int = 1
    ngram_max: int = 2
    normalize: Optional[str] = "l2"

    def _text(self, value: Any) -> str:
        if _is_missing(value):
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return " ".join(str(v) for v in value if not _is_missing(v))
        return str(value)

    def _ngrams(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        out: List[str] = []
        lo = max(1, int(self.ngram_min))
        hi = max(lo, int(self.ngram_max))
        for n in range(lo, hi + 1):
            if n == 1:
                out.extend(tokens)
            elif len(tokens) >= n:
                for i in range(0, len(tokens) - n + 1):
                    out.append("_".join(tokens[i : i + n]))
        return out

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        out = np.zeros((len(records), int(self.n_features)), dtype=np.float32)
        for i, rec in enumerate(records):
            text = self._text(_field_get(rec, self.field))
            tokens = _tokenize(text, lowercase=self.lowercase)
            for gram in self._ngrams(tokens):
                idx = _stable_hash_to_index(gram, int(self.n_features))
                out[i, idx] += 1.0
            if self.normalize == "l2":
                denom = float(np.linalg.norm(out[i], ord=2))
                if denom > 0:
                    out[i] /= denom
        return out

    @property
    def output_dim(self) -> int:
        return int(self.n_features)

    def feature_names(self) -> List[str]:
        return [f"{self.field}:hash_{i}" for i in range(int(self.n_features))]


@dataclass
class ListHashingEncoder(BaseFieldEncoder):
    """Hashing encoder for list-like identifier fields (e.g. pipe-delimited IDs)."""

    field: str
    n_features: int = 128
    delimiter: Optional[str] = "|"
    lowercase: bool = True
    binary: bool = True
    normalize: Optional[str] = "l2"

    def _items(self, value: Any) -> List[str]:
        if _is_missing(value):
            return []
        raw: List[Any]
        if isinstance(value, str):
            if self.delimiter is not None:
                raw = [v.strip() for v in value.split(self.delimiter)]
            else:
                raw = [value]
        elif isinstance(value, (list, tuple, set)):
            raw = list(value)
        else:
            raw = [value]

        items: List[str] = []
        for v in raw:
            if _is_missing(v):
                continue
            s = str(v).strip()
            if s == "":
                continue
            items.append(s.lower() if self.lowercase else s)
        return items

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        out = np.zeros((len(records), int(self.n_features)), dtype=np.float32)
        for i, rec in enumerate(records):
            for token in self._items(_field_get(rec, self.field)):
                idx = _stable_hash_to_index(token, int(self.n_features))
                if self.binary:
                    out[i, idx] = 1.0
                else:
                    out[i, idx] += 1.0
            if self.normalize == "l2":
                denom = float(np.linalg.norm(out[i], ord=2))
                if denom > 0:
                    out[i] /= denom
        return out

    @property
    def output_dim(self) -> int:
        return int(self.n_features)

    def feature_names(self) -> List[str]:
        return [f"{self.field}:list_hash_{i}" for i in range(int(self.n_features))]


@dataclass
class UrlStatsEncoder(BaseFieldEncoder):
    """Encode URL fields into numeric stats + hashed domain bucket."""

    field: str
    domain_buckets: int = 16

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        dim = self.output_dim
        out = np.zeros((len(records), dim), dtype=np.float32)
        for i, rec in enumerate(records):
            value = _field_get(rec, self.field)
            if _is_missing(value):
                continue
            url = str(value).strip()
            if not url:
                continue
            parsed = urlparse(url)
            domain = (parsed.netloc or "").lower()
            path = parsed.path or ""
            query = parsed.query or ""

            out[i, 0] = 1.0
            out[i, 1] = 1.0 if parsed.scheme.lower() == "https" else 0.0
            out[i, 2] = 1.0 if query else 0.0
            out[i, 3] = 1.0 if parsed.fragment else 0.0
            out[i, 4] = float(path.count("/"))
            out[i, 5] = float(len(parse_qs(query)))
            out[i, 6] = float(min(len(url), 512) / 512.0)
            if int(self.domain_buckets) > 0 and domain:
                bucket = _stable_hash_to_index(domain, int(self.domain_buckets))
                out[i, 7 + bucket] = 1.0
        return out

    @property
    def output_dim(self) -> int:
        return 7 + int(self.domain_buckets)

    def feature_names(self) -> List[str]:
        names = [
            f"{self.field}:url_present",
            f"{self.field}:url_https",
            f"{self.field}:url_has_query",
            f"{self.field}:url_has_fragment",
            f"{self.field}:url_path_depth",
            f"{self.field}:url_query_params",
            f"{self.field}:url_len_norm",
        ]
        names.extend([f"{self.field}:url_domain_bucket={i}" for i in range(int(self.domain_buckets))])
        return names


@dataclass
class FieldPresenceEncoder(BaseFieldEncoder):
    """Encode key presence for arbitrary fields."""

    fields: Sequence[str]

    def transform(self, records: Sequence[Record]) -> np.ndarray:
        out = np.zeros((len(records), len(self.fields)), dtype=np.float32)
        for i, rec in enumerate(records):
            for j, field in enumerate(self.fields):
                val = _field_get(rec, field)
                out[i, j] = 0.0 if _is_missing(val) else 1.0
        return out

    @property
    def output_dim(self) -> int:
        return len(self.fields)

    def feature_names(self) -> List[str]:
        return [f"{field}:present" for field in self.fields]


@dataclass
class FeatureEncoderPipeline:
    """Composable encoder stack that concatenates field encodings."""

    encoders: Sequence[BaseFieldEncoder]
    output_dtype: np.dtype = np.float32

    def fit(self, records: Sequence[Record]) -> "FeatureEncoderPipeline":
        for enc in self.encoders:
            enc.fit(records)
        return self

    def transform_numpy(self, records: Sequence[Record]) -> np.ndarray:
        if len(self.encoders) == 0:
            return np.zeros((len(records), 0), dtype=self.output_dtype)
        blocks = [enc.transform(records).astype(self.output_dtype, copy=False) for enc in self.encoders]
        return np.concatenate(blocks, axis=1)

    def transform_tensor(self, records: Sequence[Record], *, device: Optional[torch.device] = None) -> torch.Tensor:
        arr = self.transform_numpy(records)
        t = torch.from_numpy(arr)
        if device is not None:
            t = t.to(device)
        return t

    def fit_transform_numpy(self, records: Sequence[Record]) -> np.ndarray:
        return self.fit(records).transform_numpy(records)

    def fit_transform_tensor(self, records: Sequence[Record], *, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.fit(records).transform_tensor(records, device=device)

    @property
    def output_dim(self) -> int:
        return int(sum(enc.output_dim for enc in self.encoders))

    def feature_names(self) -> List[str]:
        names: List[str] = []
        for enc in self.encoders:
            names.extend(enc.feature_names())
        return names


def build_default_metadata_encoder(
    *,
    synonyms_hash_dim: int = 256,
    id_hash_dim: int = 256,
    tree_hash_dim: int = 256,
    url_domain_buckets: int = 16,
) -> FeatureEncoderPipeline:
    """
    Build a pragmatic default encoder for CTD-like metadata dicts.

    Expected keys (all optional):
    - synonyms: list[str]
    - xrefs: list[str]
    - parentIDs: list[str]
    - parentTreeNumbers: list[str]
    - treeNumbers: list[str]
    - ctd: bool-like scalar
    - ctd_url: url string
    """
    return FeatureEncoderPipeline(
        encoders=[
            FieldPresenceEncoder(
                fields=[
                    "synonyms",
                    "xrefs",
                    "parentIDs",
                    "parentTreeNumbers",
                    "treeNumbers",
                    "ctd",
                    "ctd_url",
                ]
            ),
            TextHashingEncoder(field="synonyms", n_features=int(synonyms_hash_dim), ngram_min=1, ngram_max=2),
            TextHashingEncoder(field="xrefs", n_features=max(32, int(id_hash_dim // 2)), ngram_min=1, ngram_max=1),
            TextHashingEncoder(field="parentIDs", n_features=int(id_hash_dim), ngram_min=1, ngram_max=1),
            TextHashingEncoder(field="parentTreeNumbers", n_features=int(tree_hash_dim), ngram_min=1, ngram_max=2),
            TextHashingEncoder(field="treeNumbers", n_features=int(tree_hash_dim), ngram_min=1, ngram_max=2),
            BooleanFieldEncoder(field="ctd"),
            UrlStatsEncoder(field="ctd_url", domain_buckets=int(url_domain_buckets)),
        ]
    )


def build_current_kg_node_encoder(
    node_type: str,
    *,
    text_hash_dim: int = 256,
    id_hash_dim: int = 128,
    tree_hash_dim: int = 128,
    misc_hash_dim: int = 64,
) -> FeatureEncoderPipeline:
    """
    Build feature encoder pipelines for current KG node tables.

    Supported node types:
    - disease (columns like DS_NAME, DS_DEFINITION, DS_PARENT_IDS, ...)
    - chemical (columns like CHEM_NAME, CHEM_DEFINITION, CHEM_PARENT_IDS, ...)
    - gene (columns like GENE_SYMBOL, GENE_NAME, GENE_SYNONYMS, ...)
    """
    ntype = node_type.strip().lower()
    if ntype in {"disease", "diseases"}:
        return FeatureEncoderPipeline(
            encoders=[
                FieldPresenceEncoder(
                    fields=[
                        "DS_OMIM_MESH_ID",
                        "DS_NAME",
                        "DS_DEFINITION",
                        "DS_PARENT_IDS",
                        "DS_TREE_NUMBERS",
                        "DS_PARENT_TREE_NUMBERS",
                        "DS_SYNONYMS",
                        "DS_SLIM_MAPPINGS",
                    ]
                ),
                TextHashingEncoder(field="DS_NAME", n_features=max(32, text_hash_dim // 4), ngram_min=1, ngram_max=2),
                TextHashingEncoder(field="DS_DEFINITION", n_features=int(text_hash_dim), ngram_min=1, ngram_max=2),
                ListHashingEncoder(field="DS_OMIM_MESH_ID", n_features=max(32, id_hash_dim // 2), delimiter="|"),
                ListHashingEncoder(field="DS_PARENT_IDS", n_features=int(id_hash_dim), delimiter="|"),
                ListHashingEncoder(field="DS_TREE_NUMBERS", n_features=int(tree_hash_dim), delimiter="|"),
                ListHashingEncoder(field="DS_PARENT_TREE_NUMBERS", n_features=int(tree_hash_dim), delimiter="|"),
                ListHashingEncoder(field="DS_SYNONYMS", n_features=max(32, text_hash_dim // 2), delimiter="|"),
                ListHashingEncoder(field="DS_SLIM_MAPPINGS", n_features=max(32, int(misc_hash_dim)), delimiter="|"),
            ]
        )
    if ntype in {"chemical", "chemicals"}:
        return FeatureEncoderPipeline(
            encoders=[
                FieldPresenceEncoder(
                    fields=[
                        "CHEM_MESH_ID",
                        "CHEM_NAME",
                        "CHEM_DEFINITION",
                        "CHEM_PARENT_IDS",
                        "CHEM_TREE_NUMBERS",
                        "CHEM_PARENT_TREE_NUMBERS",
                        "CHEM_SYNONYMS",
                    ]
                ),
                TextHashingEncoder(field="CHEM_NAME", n_features=max(32, text_hash_dim // 4), ngram_min=1, ngram_max=2),
                TextHashingEncoder(field="CHEM_DEFINITION", n_features=int(text_hash_dim), ngram_min=1, ngram_max=2),
                ListHashingEncoder(field="CHEM_MESH_ID", n_features=max(32, id_hash_dim // 2), delimiter="|"),
                ListHashingEncoder(field="CHEM_PARENT_IDS", n_features=int(id_hash_dim), delimiter="|"),
                ListHashingEncoder(field="CHEM_TREE_NUMBERS", n_features=int(tree_hash_dim), delimiter="|"),
                ListHashingEncoder(field="CHEM_PARENT_TREE_NUMBERS", n_features=int(tree_hash_dim), delimiter="|"),
                ListHashingEncoder(field="CHEM_SYNONYMS", n_features=max(32, text_hash_dim // 2), delimiter="|"),
            ]
        )
    if ntype in {"gene", "genes"}:
        return FeatureEncoderPipeline(
            encoders=[
                FieldPresenceEncoder(
                    fields=[
                        "GENE_NCBI_ID",
                        "GENE_SYMBOL",
                        "GENE_NAME",
                        "GENE_BIOGRID_IDS",
                        "GENE_ALT_IDS",
                        "GENE_SYNONYMS",
                        "GENE_PHARMGKB_IDS",
                        "GENE_UNIPROT_IDS",
                    ]
                ),
                TextHashingEncoder(field="GENE_SYMBOL", n_features=max(32, text_hash_dim // 4), ngram_min=1, ngram_max=2),
                TextHashingEncoder(field="GENE_NAME", n_features=max(32, text_hash_dim // 2), ngram_min=1, ngram_max=2),
                ListHashingEncoder(field="GENE_NCBI_ID", n_features=max(32, id_hash_dim // 2), delimiter="|"),
                ListHashingEncoder(field="GENE_BIOGRID_IDS", n_features=max(32, misc_hash_dim), delimiter="|"),
                ListHashingEncoder(field="GENE_ALT_IDS", n_features=max(32, misc_hash_dim), delimiter="|"),
                ListHashingEncoder(field="GENE_SYNONYMS", n_features=max(32, text_hash_dim // 2), delimiter="|"),
                ListHashingEncoder(field="GENE_PHARMGKB_IDS", n_features=max(32, misc_hash_dim), delimiter="|"),
                ListHashingEncoder(field="GENE_UNIPROT_IDS", n_features=max(32, misc_hash_dim), delimiter="|"),
            ]
        )
    raise ValueError(
        f'Unsupported node_type "{node_type}". Expected one of: disease, chemical, gene.'
    )


__all__ = [
    "BaseFieldEncoder",
    "NumericFieldEncoder",
    "BooleanFieldEncoder",
    "CategoryOneHotEncoder",
    "MultiCategoryEncoder",
    "TextHashingEncoder",
    "ListHashingEncoder",
    "UrlStatsEncoder",
    "FieldPresenceEncoder",
    "FeatureEncoderPipeline",
    "build_default_metadata_encoder",
    "build_current_kg_node_encoder",
]
