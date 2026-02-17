#!/usr/bin/env python
"""
Interactive terminal TUI for the DD pipeline.

Run:
    python scripts/tui.py
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        RichLog,
        Select,
        Static,
        Switch,
    )
except ImportError as exc:
    raise SystemExit(
        "Textual >=0.40 is required.  Install with:  pip install 'textual>=0.40'"
    ) from exc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / ".dd_tui_profile.json"


# ---------------------------------------------------------------------------
# Data specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    """One form field that maps to a CLI flag."""

    key: str
    label: str
    kind: str = "str"  # str | int | optional_int | float | bool | choice | list_int
    flag: Optional[str] = None
    default: Any = ""
    placeholder: str = ""
    options: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModuleSpec:
    """One runnable module / pipeline step."""

    key: str
    title: str
    description: str
    script: Optional[str]
    fields: Tuple[FieldSpec, ...] = field(default_factory=tuple)


# A --config field reused in every script-backed module.
_CONFIG_FIELD = FieldSpec(
    "config", "Config YAML", "str", "--config", "", "path/to/config.yaml"
)

MODULE_SPECS: Tuple[ModuleSpec, ...] = (
    # ── Pipeline (virtual, no script) ──────────────────────────────
    ModuleSpec(
        key="pipeline",
        title="Run Full Pipeline",
        description=(
            "Chain: process_data → [build_node_features] → [train] "
            "→ [evaluate] → [cache_embeddings] → [streamlit]."
        ),
        script=None,
        fields=(
            FieldSpec("raw_dir", "Raw Data Dir", default="./data/raw"),
            FieldSpec("processed_dir", "Processed Dir", default="./data/processed"),
            FieldSpec("use_node_features", "Use Node Features", "bool", default=False),
            FieldSpec(
                "node_features_dir",
                "Node Features Dir",
                default="./data/processed/features",
            ),
            FieldSpec("train_model", "Train Model", "bool", default=True),
            FieldSpec("train_epochs", "Epochs", "int", default=50),
            FieldSpec("train_batch_size", "Batch Size", "int", default=4096),
            FieldSpec("run_evaluate", "Run Evaluate", "bool", default=True),
            FieldSpec(
                "evaluate_output_dir", "Evaluate Output", default="./evaluation_results"
            ),
            FieldSpec("build_cache", "Build Cache", "bool", default=True),
            FieldSpec("embeddings_dir", "Embeddings Dir", default="./embeddings"),
            FieldSpec("chunk_size", "Cache Chunk Size", "int", default=100000),
            FieldSpec(
                "checkpoint",
                "Checkpoint (fallback)",
                default="./checkpoints/best.pt",
            ),
            FieldSpec("start_streamlit", "Start Streamlit", "bool", default=False),
            FieldSpec("streamlit_port", "Streamlit Port", "int", default=8501),
            FieldSpec("streamlit_address", "Streamlit Address", default="0.0.0.0"),
            FieldSpec(
                "config",
                "Config YAML (shared)",
                placeholder="path/to/config.yaml",
            ),
        ),
    ),
    # ── Individual modules ─────────────────────────────────────────
    ModuleSpec(
        key="process_data",
        title="Process Data",
        description="Convert raw CTD files to processed parquet.",
        script="scripts/process_data.py",
        fields=(
            FieldSpec("raw_dir", "Raw Data Dir", "str", "--raw-dir", "./data/raw"),
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            _CONFIG_FIELD,
        ),
    ),
    ModuleSpec(
        key="build_node_features",
        title="Build Node Features",
        description="Build inductive node feature tables (text, FP, structure).",
        script="scripts/build_node_features.py",
        fields=(
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            FieldSpec("raw_dir", "Raw Data Dir", "str", "--raw-dir", "./data/raw"),
            FieldSpec(
                "output_dir",
                "Output Dir",
                "str",
                "--output-dir",
                "./data/processed/features",
            ),
            FieldSpec("text_dim", "Text Dim", "int", "--text-dim", 128),
            FieldSpec("chem_fp_bits", "Chem FP Bits", "int", "--chem-fp-bits", 1024),
            FieldSpec("no_pubchem", "Disable PubChem", "bool", "--no-pubchem", False),
            FieldSpec("no_uniprot", "Disable UniProt", "bool", "--no-uniprot", False),
            FieldSpec("use_umls", "Use UMLS", "bool", "--use-umls", False),
            FieldSpec(
                "disgenet_file", "DisGeNET File", "str", "--disgenet-file", ""
            ),
            FieldSpec(
                "max_pubchem_fetch",
                "Max PubChem Fetch",
                "optional_int",
                "--max-pubchem-fetch",
                "",
            ),
            FieldSpec(
                "max_uniprot_fetch",
                "Max UniProt Fetch",
                "optional_int",
                "--max-uniprot-fetch",
                "",
            ),
            FieldSpec(
                "umls_api_key", "UMLS API Key", "str", "--umls-api-key", ""
            ),
            FieldSpec(
                "max_umls_fetch",
                "Max UMLS Fetch",
                "optional_int",
                "--max-umls-fetch",
                "",
            ),
            FieldSpec(
                "request_timeout_s",
                "HTTP Timeout (s)",
                "int",
                "--request-timeout-s",
                20,
            ),
            FieldSpec(
                "sleep_s", "Sleep Between Reqs", "float", "--sleep-s", 0.02
            ),
            _CONFIG_FIELD,
        ),
    ),
    ModuleSpec(
        key="train",
        title="Train",
        description="Train the HGT link-prediction model.",
        script="scripts/train.py",
        fields=(
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            FieldSpec(
                "use_node_features",
                "Use Node Features",
                "bool",
                "--use-node-features",
                False,
            ),
            FieldSpec(
                "node_features_dir",
                "Node Features Dir",
                "str",
                "--node-features-dir",
                "./data/processed/features",
            ),
            FieldSpec("hidden_dim", "Hidden Dim", "int", "--hidden-dim", 128),
            FieldSpec("num_layers", "Num Layers", "int", "--num-layers", 2),
            FieldSpec("num_heads", "Num Heads", "int", "--num-heads", 4),
            FieldSpec("dropout", "Dropout", "float", "--dropout", 0.2),
            FieldSpec("epochs", "Epochs", "int", "--epochs", 50),
            FieldSpec("batch_size", "Batch Size", "int", "--batch-size", 4096),
            FieldSpec("lr", "Learning Rate", "float", "--lr", 0.0003),
            FieldSpec("weight_decay", "Weight Decay", "float", "--weight-decay", 0.0001),
            FieldSpec("grad_clip", "Grad Clip", "float", "--grad-clip", 1.0),
            FieldSpec(
                "num_neg_train", "Neg/Pos (train)", "int", "--num-neg-train", 5
            ),
            FieldSpec(
                "num_neg_eval", "Neg/Pos (eval)", "int", "--num-neg-eval", 20
            ),
            FieldSpec(
                "pos_weight",
                "Pos Weight",
                "optional_int",
                "--pos-weight",
                "",
                "leave blank for auto",
            ),
            FieldSpec(
                "focal_gamma", "Focal Gamma", "float", "--focal-gamma", 0.0
            ),
            FieldSpec(
                "hard_negative_ratio",
                "Hard Neg Ratio",
                "float",
                "--hard-negative-ratio",
                0.5,
            ),
            FieldSpec(
                "eval_hard_negative_ratio",
                "Eval Hard Neg Ratio",
                "float",
                "--eval-hard-negative-ratio",
                0.0,
            ),
            FieldSpec(
                "degree_alpha", "Degree Alpha", "float", "--degree-alpha", 0.75
            ),
            FieldSpec("val_ratio", "Val Ratio", "float", "--val-ratio", 0.1),
            FieldSpec("test_ratio", "Test Ratio", "float", "--test-ratio", 0.1),
            FieldSpec("seed", "Seed", "int", "--seed", 42),
            FieldSpec("patience", "LR Patience", "int", "--patience", 5),
            FieldSpec("factor", "LR Factor", "float", "--factor", 0.5),
            FieldSpec(
                "early_stopping",
                "Early Stopping",
                "int",
                "--early-stopping",
                10,
            ),
            FieldSpec(
                "monitor",
                "Monitor Metric",
                "choice",
                "--monitor",
                "auprc",
                options=(("auprc", "auprc"), ("auroc", "auroc"), ("f1", "f1")),
            ),
            FieldSpec(
                "num_neighbours",
                "Num Neighbours",
                "list_int",
                "--num-neighbours",
                "10 5",
                "e.g. 10 5",
            ),
            FieldSpec(
                "ckpt_dir", "Checkpoint Dir", "str", "--ckpt-dir", "./checkpoints"
            ),
            FieldSpec("run_name", "Run Name", "str", "--run-name", ""),
            FieldSpec(
                "experiment_name",
                "Experiment Name",
                "str",
                "--experiment-name",
                "HGT_linkpred",
            ),
            FieldSpec("no_amp", "Disable AMP", "bool", "--no-amp", False),
            _CONFIG_FIELD,
        ),
    ),
    ModuleSpec(
        key="evaluate",
        title="Evaluate",
        description="Evaluate a trained checkpoint and generate reports.",
        script="scripts/evaluate.py",
        fields=(
            FieldSpec(
                "checkpoint",
                "Checkpoint",
                "str",
                "--checkpoint",
                "./checkpoints/best.pt",
            ),
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            FieldSpec(
                "use_node_features",
                "Use Node Features",
                "bool",
                "--use-node-features",
                False,
            ),
            FieldSpec(
                "node_features_dir",
                "Node Features Dir",
                "str",
                "--node-features-dir",
                "./data/processed/features",
            ),
            FieldSpec(
                "output_dir",
                "Output Dir",
                "str",
                "--output-dir",
                "./evaluation_results",
            ),
            FieldSpec(
                "num_neg_eval", "Num Neg Eval", "int", "--num-neg-eval", 50
            ),
            FieldSpec("batch_size", "Batch Size", "int", "--batch-size", 1024),
            FieldSpec(
                "split",
                "Split",
                "choice",
                "--split",
                "test",
                options=(("test", "test"), ("val", "val")),
            ),
            FieldSpec("no_tsne", "Skip t-SNE", "bool", "--no-tsne", False),
            FieldSpec("seed", "Seed", "int", "--seed", 42),
            _CONFIG_FIELD,
        ),
    ),
    ModuleSpec(
        key="cache_embeddings_chunked",
        title="Cache Embeddings (Chunked)",
        description="Build prediction-cache tensors (chemical/disease embeddings + W_cd).",
        script="scripts/cache_embeddings_chunked.py",
        fields=(
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            FieldSpec(
                "checkpoint",
                "Checkpoint",
                "str",
                "--checkpoint",
                "./checkpoints/best.pt",
            ),
            FieldSpec(
                "output_dir", "Output Dir", "str", "--output-dir", "./embeddings"
            ),
            FieldSpec("chunk_size", "Chunk Size", "int", "--chunk-size", 100000),
            FieldSpec("cpu", "Force CPU", "bool", "--cpu", False),
            _CONFIG_FIELD,
        ),
    ),
    ModuleSpec(
        key="predict",
        title="Predict (Full Graph)",
        description="Run predictions using full graph inference.",
        script="scripts/predict.py",
        fields=(
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            FieldSpec(
                "use_node_features",
                "Use Node Features",
                "bool",
                "--use-node-features",
                False,
            ),
            FieldSpec(
                "node_features_dir",
                "Node Features Dir",
                "str",
                "--node-features-dir",
                "",
            ),
            FieldSpec(
                "checkpoint",
                "Checkpoint",
                "str",
                "--checkpoint",
                "./checkpoints/best.pt",
            ),
            FieldSpec("hidden_dim", "Hidden Dim", "int", "--hidden-dim", 128),
            FieldSpec("num_layers", "Num Layers", "int", "--num-layers", 2),
            FieldSpec("num_heads", "Num Heads", "int", "--num-heads", 4),
            FieldSpec("dropout", "Dropout", "float", "--dropout", 0.2),
            FieldSpec("disease", "Disease ID", "str", "--disease", ""),
            FieldSpec("chemical", "Chemical ID", "str", "--chemical", ""),
            FieldSpec("top_k", "Top-K", "int", "--top-k", 10),
            FieldSpec("threshold", "Threshold", "float", "--threshold", 0.5),
            FieldSpec(
                "include_known", "Include Known", "bool", "--include-known", False
            ),
            FieldSpec(
                "no_extended",
                "Disable Extended Graph",
                "bool",
                "--no-extended",
                False,
            ),
            _CONFIG_FIELD,
        ),
    ),
    ModuleSpec(
        key="predict_cached",
        title="Predict (Cached)",
        description="Run predictions from pre-cached embeddings.",
        script="scripts/predict_cached.py",
        fields=(
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            FieldSpec(
                "embeddings_dir",
                "Embeddings Dir",
                "str",
                "--embeddings-dir",
                "./embeddings",
            ),
            FieldSpec("disease", "Disease ID", "str", "--disease", ""),
            FieldSpec("chemical", "Chemical ID", "str", "--chemical", ""),
            FieldSpec("top_k", "Top-K", "int", "--top-k", 10),
            FieldSpec("threshold", "Threshold", "float", "--threshold", 0.5),
            FieldSpec(
                "exclude_known", "Exclude Known", "bool", "--exclude-known", False
            ),
            _CONFIG_FIELD,
        ),
    ),
    ModuleSpec(
        key="explain",
        title="Explain Prediction",
        description="Explain a chemical-disease prediction with ranked graph paths.",
        script="scripts/explain.py",
        fields=(
            FieldSpec("disease", "Disease ID", "str", "--disease", "",
                      "e.g. MESH:D014202"),
            FieldSpec("chemical", "Chemical ID", "str", "--chemical", "",
                      "e.g. C006901"),
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            FieldSpec(
                "cached", "Use Cached Embeddings", "bool", "--cached", False
            ),
            FieldSpec(
                "embeddings_dir",
                "Embeddings Dir",
                "str",
                "--embeddings-dir",
                "./embeddings",
            ),
            FieldSpec(
                "checkpoint",
                "Checkpoint",
                "str",
                "--checkpoint",
                "./checkpoints/best.pt",
            ),
            FieldSpec(
                "use_node_features",
                "Use Node Features",
                "bool",
                "--use-node-features",
                False,
            ),
            FieldSpec(
                "node_features_dir",
                "Node Features Dir",
                "str",
                "--node-features-dir",
                "",
            ),
            FieldSpec("hidden_dim", "Hidden Dim", "int", "--hidden-dim", 128),
            FieldSpec("num_layers", "Num Layers", "int", "--num-layers", 2),
            FieldSpec("num_heads", "Num Heads", "int", "--num-heads", 4),
            FieldSpec("dropout", "Dropout", "float", "--dropout", 0.2),
            FieldSpec(
                "no_extended",
                "Disable Extended Graph",
                "bool",
                "--no-extended",
                False,
            ),
            FieldSpec(
                "no_attention",
                "Skip Attention (Tier 1 only)",
                "bool",
                "--no-attention",
                False,
            ),
            FieldSpec("max_paths", "Max Paths", "int", "--max-paths", 10),
            FieldSpec(
                "max_paths_per_template",
                "Max Paths/Template",
                "int",
                "--max-paths-per-template",
                100,
            ),
            FieldSpec(
                "threshold", "Threshold", "float", "--threshold", 0.5
            ),
            _CONFIG_FIELD,
        ),
    ),
    ModuleSpec(
        key="tune",
        title="Tune (Optuna)",
        description="Run hyperparameter tuning with Optuna.",
        script="scripts/tune.py",
        fields=(
            FieldSpec(
                "processed_dir",
                "Processed Dir",
                "str",
                "--processed-dir",
                "./data/processed",
            ),
            FieldSpec(
                "use_node_features",
                "Use Node Features",
                "bool",
                "--use-node-features",
                False,
            ),
            FieldSpec(
                "node_features_dir",
                "Node Features Dir",
                "str",
                "--node-features-dir",
                "./data/processed/features",
            ),
            FieldSpec("timeout", "Timeout (s)", "int", "--timeout", 14400),
            FieldSpec(
                "n_trials",
                "Max Trials",
                "optional_int",
                "--n-trials",
                "",
                "blank = unlimited",
            ),
            FieldSpec("quick", "Quick Search", "bool", "--quick", False),
            FieldSpec(
                "study_name", "Study Name", "str", "--study-name", "hgt_cd_tuning"
            ),
            FieldSpec(
                "storage",
                "DB Storage URL",
                "str",
                "--storage",
                "sqlite:///optuna_hgt.db",
            ),
            FieldSpec(
                "experiment_name",
                "MLflow Experiment",
                "str",
                "--experiment-name",
                "HGT_tuning",
            ),
            FieldSpec(
                "epochs_per_trial",
                "Epochs / Trial",
                "int",
                "--epochs-per-trial",
                25,
            ),
            FieldSpec(
                "early_stopping",
                "Early Stopping",
                "int",
                "--early-stopping",
                7,
            ),
            FieldSpec(
                "output_dir",
                "Output Dir",
                "str",
                "--output-dir",
                "./tuning_results",
            ),
            _CONFIG_FIELD,
        ),
    ),
    # ── Streamlit (virtual, no script file) ────────────────────────
    ModuleSpec(
        key="streamlit_app",
        title="Start Streamlit",
        description="Launch the Streamlit prediction UI.",
        script=None,
        fields=(
            FieldSpec("port", "Port", "int", default=8501),
            FieldSpec("address", "Address", default="0.0.0.0"),
        ),
    ),
)

MODULE_BY_KEY: Dict[str, ModuleSpec] = {m.key: m for m in MODULE_SPECS}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_int(value: Any, key: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{key} must not be empty")
    return int(text)


def _parse_float(value: Any, key: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{key} must not be empty")
    return float(text)


def _parse_int_list(value: Any, key: str) -> List[int]:
    text = str(value).replace(",", " ").strip()
    if not text:
        return []
    try:
        return [int(p) for p in text.split()]
    except ValueError as exc:
        raise ValueError(f"{key} must be space-separated integers") from exc


def _build_streamlit_command(values: Dict[str, Any]) -> List[str]:
    port = _parse_int(values.get("port", 8501), "port")
    address = str(values.get("address", "0.0.0.0")).strip() or "0.0.0.0"
    return [
        "streamlit", "run", str(ROOT / "app.py"),
        "--server.port", str(port),
        "--server.address", address,
    ]


def _build_module_command(module_key: str, values: Dict[str, Any]) -> List[str]:
    """Build the CLI command list for *module_key* using *values*."""
    module = MODULE_BY_KEY[module_key]

    if module_key == "streamlit_app":
        return _build_streamlit_command(values)

    if module.script is None:
        raise ValueError(f"Module {module_key!r} has no script")

    cmd: List[str] = [sys.executable, str(ROOT / module.script)]
    for fs in module.fields:
        if not fs.flag:
            continue
        raw = values.get(fs.key, fs.default)

        if fs.kind == "bool":
            if bool(raw):
                cmd.append(fs.flag)
            continue

        if fs.kind == "int":
            cmd.extend([fs.flag, str(_parse_int(raw, fs.key))])
            continue

        if fs.kind == "optional_int":
            text = str(raw).strip()
            if text:
                cmd.extend([fs.flag, str(_parse_int(text, fs.key))])
            continue

        if fs.kind == "float":
            cmd.extend([fs.flag, str(_parse_float(raw, fs.key))])
            continue

        if fs.kind == "choice":
            text = str(raw).strip()
            if text:
                cmd.extend([fs.flag, text])
            continue

        if fs.kind == "list_int":
            items = _parse_int_list(raw, fs.key)
            if items:
                cmd.append(fs.flag)
                cmd.extend(str(v) for v in items)
            continue

        # str (default)
        text = str(raw).strip()
        if text:
            cmd.extend([fs.flag, text])

    extra = str(values.get("__extra_args", "")).strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class DDTUI(App[None]):
    """DD Pipeline — interactive terminal control center."""

    TITLE = "DD Pipeline TUI"
    CSS = """
    Screen {
        layout: vertical;
    }

    #outer {
        height: 1fr;
    }

    /* ── left sidebar ─────────────────────────────────── */
    #sidebar {
        width: 34;
        border: round #4f46e5;
        padding: 1 1;
    }
    #sidebar Button {
        width: 100%;
        margin-bottom: 1;
    }
    .sidebar-heading {
        color: #c084fc;
        text-style: bold;
        margin-bottom: 1;
    }
    #status-line {
        color: #f8fafc;
        margin-top: 1;
    }
    #ckpt-line {
        color: #93c5fd;
    }

    /* ── right workspace ──────────────────────────────── */
    #workspace {
        border: round #2563eb;
        padding: 0 1;
    }
    #module-info {
        height: auto;
        max-height: 3;
        color: #cbd5e1;
        margin: 1 0;
    }

    /* form */
    #form-scroll {
        height: 1fr;
        min-height: 6;
        border: round #334155;
        padding: 0 1;
    }
    .field-row {
        height: 3;
    }
    .field-label {
        width: 24;
        height: 3;
        content-align: left middle;
        color: #94a3b8;
    }

    /* preview */
    #cmd-preview {
        height: auto;
        max-height: 5;
        border: round #334155;
        padding: 0 1;
        color: #e2e8f0;
        margin: 1 0;
    }

    /* log */
    #log {
        height: 1fr;
        min-height: 8;
        border: round #0f766e;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("f5", "run_selected", "Run"),
        Binding("f6", "run_pipeline", "Pipeline"),
        Binding("escape", "stop_running", "Stop"),
        Binding("ctrl+l", "clear_log", "Clear log"),
    ]

    # ── lifecycle ──────────────────────────────────────────────────
    def __init__(self) -> None:
        super().__init__()
        self._active_module: str = "pipeline"
        # Per-module form values, keyed by module_key -> field_key -> value.
        self._values: Dict[str, Dict[str, Any]] = {}
        self._init_values()
        # Runtime state.
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stop_requested = False
        self._task_active = False
        self._last_ckpt: str = ""
        self._rebuilding = False  # guard against events during form rebuild
        # Load saved profile (data only, no widget access).
        self._load_profile_data()

    def _init_values(self) -> None:
        for m in MODULE_SPECS:
            self._values[m.key] = {fs.key: fs.default for fs in m.fields}
            self._values[m.key]["__extra_args"] = ""

    # ── compose ────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="outer"):
            with VerticalScroll(id="sidebar"):
                yield Static("DD Control Center", classes="sidebar-heading")
                yield Static("Module")
                yield Select(
                    options=[(m.title, m.key) for m in MODULE_SPECS],
                    value=self._active_module,
                    id="module-select",
                )
                yield Button("Run Selected  [F5]", id="btn-run", variant="success")
                yield Button("Run Pipeline  [F6]", id="btn-pipeline", variant="primary")
                yield Button("Start Streamlit", id="btn-streamlit", variant="warning")
                yield Button("Stop  [Esc]", id="btn-stop", variant="error", disabled=True)
                yield Button("Save Profile", id="btn-save")
                yield Button("Reload Profile", id="btn-reload")
                yield Button("Clear Log  [^L]", id="btn-clear")
                yield Static("idle", id="status-line")
                yield Static("", id="ckpt-line")
            with Vertical(id="workspace"):
                yield Static(id="module-info")
                yield VerticalScroll(id="form-scroll")
                yield Static(id="cmd-preview")
                yield RichLog(id="log", highlight=False, markup=False, wrap=True, max_lines=5000)
        yield Footer()

    async def on_mount(self) -> None:
        self._update_ckpt_display()
        await self._rebuild_form()
        self._refresh_preview()
        self._log("DD TUI ready.  Select a module and press F5 to run.")

    # ── profile persistence (data only — no widget access) ─────────
    def _load_profile_data(self) -> None:
        """Load saved profile into self._values / _active_module.
        Must be called BEFORE compose (no widget queries).
        """
        if not PROFILE_PATH.exists():
            return
        try:
            data = json.loads(PROFILE_PATH.read_text("utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        active = data.get("active_module")
        if isinstance(active, str) and active in MODULE_BY_KEY:
            self._active_module = active
        raw = data.get("values")
        if isinstance(raw, dict):
            for mk, mv in raw.items():
                if mk in self._values and isinstance(mv, dict):
                    self._values[mk].update(mv)
        ckpt = data.get("last_train_checkpoint")
        if isinstance(ckpt, str) and ckpt:
            self._last_ckpt = ckpt

    def _save_profile(self) -> None:
        payload = {
            "active_module": self._active_module,
            "values": self._values,
            "last_train_checkpoint": self._last_ckpt,
        }
        try:
            PROFILE_PATH.write_text(json.dumps(payload, indent=2), "utf-8")
        except Exception:
            pass

    # ── form construction ──────────────────────────────────────────
    async def _rebuild_form(self) -> None:
        self._rebuilding = True
        try:
            module = MODULE_BY_KEY[self._active_module]
            vals = self._values[self._active_module]

            # Update info label.
            self.query_one("#module-info", Static).update(
                f"[b]{module.title}[/b]  —  {module.description}"
            )

            form = self.query_one("#form-scroll", VerticalScroll)
            await form.remove_children()

            widgets_to_mount: List[Horizontal] = []

            for fs in module.fields:
                value = vals.get(fs.key, fs.default)
                wid = f"field-{fs.key}"

                if fs.kind == "bool":
                    widget: Any = Switch(value=bool(value), id=wid)
                elif fs.kind == "choice":
                    widget = Select(
                        options=list(fs.options),
                        value=str(value) if value else "",
                        id=wid,
                    )
                else:
                    widget = Input(
                        value=str(value),
                        placeholder=fs.placeholder,
                        id=wid,
                    )

                row = Horizontal(
                    Static(fs.label, classes="field-label"),
                    widget,
                    classes="field-row",
                )
                widgets_to_mount.append(row)

            # Extra args row (always last).
            widgets_to_mount.append(
                Horizontal(
                    Static("Extra Args", classes="field-label"),
                    Input(
                        value=str(vals.get("__extra_args", "")),
                        placeholder="raw CLI flags, e.g. --seed 7",
                        id="field-__extra_args",
                    ),
                    classes="field-row",
                )
            )

            # Mount all at once — avoids per-row async issues.
            await form.mount_all(widgets_to_mount)
        finally:
            self._rebuilding = False

    def _flush_form_to_values(self) -> None:
        """Read current widget values back into self._values."""
        module = MODULE_BY_KEY[self._active_module]
        vals = self._values[self._active_module]
        for fs in module.fields:
            wid = f"#field-{fs.key}"
            try:
                if fs.kind == "bool":
                    vals[fs.key] = self.query_one(wid, Switch).value
                elif fs.kind == "choice":
                    vals[fs.key] = str(self.query_one(wid, Select).value)
                else:
                    vals[fs.key] = self.query_one(wid, Input).value
            except Exception:
                pass
        try:
            vals["__extra_args"] = self.query_one("#field-__extra_args", Input).value
        except Exception:
            pass

    # ── command preview ────────────────────────────────────────────
    def _refresh_preview(self) -> None:
        try:
            preview = self.query_one("#cmd-preview", Static)
        except Exception:
            return  # widget not yet mounted
        try:
            if self._active_module == "pipeline":
                lines: List[str] = []
                for i, (name, cmd) in enumerate(
                    self._build_pipeline_commands(), 1
                ):
                    lines.append(f"{i}. {name}:  $ {shlex.join(cmd)}")
                preview.update("\n".join(lines) or "Pipeline has no active steps.")
            else:
                cmd = _build_module_command(
                    self._active_module,
                    self._values[self._active_module],
                )
                preview.update(f"$ {shlex.join(cmd)}")
        except Exception as exc:
            preview.update(f"(invalid params: {exc})")

    # ── event handlers ─────────────────────────────────────────────
    async def on_select_changed(self, event: Select.Changed) -> None:
        wid = event.select.id or ""
        if wid == "module-select":
            self._flush_form_to_values()
            selected = str(event.value)
            if selected in MODULE_BY_KEY:
                self._active_module = selected
                await self._rebuild_form()
                self._refresh_preview()
                self._save_profile()
            return

        # A choice field inside the form.
        if wid.startswith("field-") and not self._rebuilding:
            key = wid.removeprefix("field-")
            self._values[self._active_module][key] = str(event.value)
            self._refresh_preview()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._rebuilding:
            return
        wid = event.input.id or ""
        if not wid.startswith("field-"):
            return
        key = wid.removeprefix("field-")
        self._values[self._active_module][key] = event.value
        self._refresh_preview()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if self._rebuilding:
            return
        wid = event.switch.id or ""
        if not wid.startswith("field-"):
            return
        key = wid.removeprefix("field-")
        self._values[self._active_module][key] = bool(event.value)
        self._refresh_preview()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "btn-run":
            self.action_run_selected()
        elif bid == "btn-pipeline":
            self.action_run_pipeline()
        elif bid == "btn-streamlit":
            self._launch_module("streamlit_app")
        elif bid == "btn-stop":
            await self.action_stop_running()
        elif bid == "btn-save":
            self._flush_form_to_values()
            self._save_profile()
            self._set_status("profile saved")
        elif bid == "btn-reload":
            self._load_profile_data()
            await self._rebuild_form()
            self._refresh_preview()
            self._set_status("profile reloaded")
        elif bid == "btn-clear":
            self.action_clear_log()

    # ── actions ────────────────────────────────────────────────────
    def action_run_selected(self) -> None:
        self._flush_form_to_values()
        if self._active_module == "pipeline":
            self._launch_pipeline()
        else:
            self._launch_module(self._active_module)

    def action_run_pipeline(self) -> None:
        self._flush_form_to_values()
        self._launch_pipeline()

    async def action_stop_running(self) -> None:
        self._stop_requested = True
        self._set_status("stopping ...")
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            await asyncio.sleep(0.4)
            if self._process is not None and self._process.returncode is None:
                self._process.kill()

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    # ── subprocess execution ───────────────────────────────────────
    async def _exec(self, label: str, cmd: Sequence[str]) -> int:
        """Run *cmd* as a subprocess, streaming output to the log panel.

        Returns the exit code.
        """
        self._log(f"\n{'='*60}")
        self._log(f">>> {label}")
        self._log(f"$ {shlex.join(list(cmd))}")
        self._log(f"{'='*60}")
        self._set_status(f"running: {label}")

        try:
            import os
            env = os.environ.copy()
            # Ensure the project root is on PYTHONPATH so scripts can
            # import from `src.*`.
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(ROOT),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            self._log(f"ERROR: command not found — {exc}")
            self._set_status("error: command not found")
            return 127

        assert self._process.stdout is not None
        while True:
            line = await self._process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n\r")
            self._log(text)
            self._capture_hints(text)
            # Honour stop requests.
            if self._stop_requested and self._process.returncode is None:
                self._process.terminate()

        rc = await self._process.wait()
        self._process = None

        if rc == 0:
            self._log(f"<<< {label} finished OK (exit 0)")
        else:
            self._log(f"<<< {label} FAILED (exit {rc})")
        return rc

    # ── single module run ──────────────────────────────────────────
    def _launch_module(self, key: str) -> None:
        if self._task_active:
            self._log("A task is already running.")
            return
        self.run_worker(self._run_module(key), group="runner", exclusive=True)

    async def _run_module(self, key: str) -> None:
        self._task_active = True
        self._stop_requested = False
        self._toggle_buttons()
        name = MODULE_BY_KEY[key].title
        try:
            cmd = _build_module_command(key, self._values[key])
            rc = await self._exec(name, cmd)
            self._set_status(f"done: {name}" if rc == 0 else f"FAILED: {name}")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._set_status("error")
        finally:
            self._task_active = False
            self._process = None
            self._stop_requested = False
            self._toggle_buttons()

    # ── pipeline orchestration ─────────────────────────────────────
    def _build_pipeline_commands(self) -> List[Tuple[str, List[str]]]:
        v = self._values["pipeline"]
        steps: List[Tuple[str, List[str]]] = []

        raw = str(v.get("raw_dir", "./data/raw")).strip() or "./data/raw"
        pdir = str(v.get("processed_dir", "./data/processed")).strip() or "./data/processed"
        use_nf = bool(v.get("use_node_features", False))
        nf_dir = str(v.get("node_features_dir", "./data/processed/features")).strip()
        config = str(v.get("config", "")).strip()

        def _maybe_config(cmd: List[str]) -> List[str]:
            if config:
                cmd.extend(["--config", config])
            return cmd

        # 1. process_data
        steps.append((
            "Process Data",
            _maybe_config([
                sys.executable, str(ROOT / "scripts/process_data.py"),
                "--raw-dir", raw,
                "--processed-dir", pdir,
            ]),
        ))

        # 2. build_node_features (optional)
        if use_nf:
            steps.append((
                "Build Node Features",
                _maybe_config([
                    sys.executable, str(ROOT / "scripts/build_node_features.py"),
                    "--raw-dir", raw,
                    "--processed-dir", pdir,
                    "--output-dir", nf_dir,
                ]),
            ))

        # 3. train (optional)
        if bool(v.get("train_model", True)):
            train_cmd = [
                sys.executable, str(ROOT / "scripts/train.py"),
                "--processed-dir", pdir,
                "--epochs", str(_parse_int(v.get("train_epochs", 50), "epochs")),
                "--batch-size", str(_parse_int(v.get("train_batch_size", 4096), "batch_size")),
            ]
            if use_nf:
                train_cmd.extend(["--use-node-features", "--node-features-dir", nf_dir])
            steps.append(("Train", _maybe_config(train_cmd)))

        ckpt = str(v.get("checkpoint", "./checkpoints/best.pt")).strip() or "./checkpoints/best.pt"

        # 4. evaluate (optional)
        if bool(v.get("run_evaluate", True)):
            eval_cmd = [
                sys.executable, str(ROOT / "scripts/evaluate.py"),
                "--checkpoint", ckpt,
                "--processed-dir", pdir,
                "--output-dir",
                str(v.get("evaluate_output_dir", "./evaluation_results")).strip() or "./evaluation_results",
            ]
            if use_nf:
                eval_cmd.extend(["--use-node-features", "--node-features-dir", nf_dir])
            steps.append(("Evaluate", _maybe_config(eval_cmd)))

        # 5. cache embeddings (optional)
        if bool(v.get("build_cache", True)):
            cache_cmd = [
                sys.executable, str(ROOT / "scripts/cache_embeddings_chunked.py"),
                "--checkpoint", ckpt,
                "--processed-dir", pdir,
                "--output-dir",
                str(v.get("embeddings_dir", "./embeddings")).strip() or "./embeddings",
                "--chunk-size",
                str(_parse_int(v.get("chunk_size", 100000), "chunk_size")),
            ]
            steps.append(("Cache Embeddings", _maybe_config(cache_cmd)))

        # 6. streamlit (optional)
        if bool(v.get("start_streamlit", False)):
            steps.append((
                "Start Streamlit",
                _build_streamlit_command({
                    "port": v.get("streamlit_port", 8501),
                    "address": v.get("streamlit_address", "0.0.0.0"),
                }),
            ))

        return steps

    def _launch_pipeline(self) -> None:
        if self._task_active:
            self._log("A task is already running.")
            return
        self.run_worker(self._run_pipeline(), group="runner", exclusive=True)

    async def _run_pipeline(self) -> None:
        self._task_active = True
        self._stop_requested = False
        self._toggle_buttons()
        try:
            steps = self._build_pipeline_commands()
            if not steps:
                self._set_status("pipeline: nothing to run")
                return

            for name, cmd in steps:
                if self._stop_requested:
                    self._log("Pipeline stopped by user.")
                    self._set_status("stopped")
                    return

                # Inject discovered checkpoint for evaluate / cache steps.
                if name in {"Evaluate", "Cache Embeddings"} and "--checkpoint" in cmd:
                    effective_ckpt = self._last_ckpt or None
                    if effective_ckpt:
                        idx = cmd.index("--checkpoint") + 1
                        cmd[idx] = effective_ckpt

                rc = await self._exec(name, cmd)
                if rc != 0:
                    self._set_status(f"pipeline FAILED at: {name}")
                    return

            self._set_status("pipeline completed")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._set_status("pipeline error")
        finally:
            self._task_active = False
            self._process = None
            self._stop_requested = False
            self._toggle_buttons()

    # ── helpers ────────────────────────────────────────────────────
    def _capture_hints(self, line: str) -> None:
        """Parse subprocess stdout for runtime info (e.g. checkpoint path)."""
        if line.startswith("Checkpoint directory:"):
            ckpt_dir = line.split(":", 1)[1].strip()
            if ckpt_dir:
                self._last_ckpt = str(Path(ckpt_dir) / "best.pt")
                self._update_ckpt_display()

    def _update_ckpt_display(self) -> None:
        try:
            self.query_one("#ckpt-line", Static).update(
                f"ckpt: {self._last_ckpt}" if self._last_ckpt else ""
            )
        except Exception:
            pass

    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#status-line", Static).update(msg)
        except Exception:
            pass

    def _toggle_buttons(self) -> None:
        r = self._task_active
        for bid in ("btn-run", "btn-pipeline", "btn-streamlit"):
            try:
                self.query_one(f"#{bid}", Button).disabled = r
            except Exception:
                pass
        try:
            self.query_one("#btn-stop", Button).disabled = not r
        except Exception:
            pass

    def _log(self, text: str) -> None:
        try:
            self.query_one("#log", RichLog).write(text)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    DDTUI().run()


if __name__ == "__main__":
    main()
