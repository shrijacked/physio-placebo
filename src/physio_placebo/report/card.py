"""Week 10 SDS card.

The primary table is the filed macro-F1 SDS. A cell whose real-signal interval
does not sit strictly above the locked permutation floor prints
``undefined (validity gate)`` and no ratio. ``subject_perm_preserved`` is
``descriptive, not a null``.

Balanced accuracy, logprob AUROC, matched floors, Holm, the adjective vote,
and the input-matched ceilings are post-registration (addendum 2026-10-30).
Paired item statistics that need real-signal A or C rows print
``pending real rows`` until those rows are passed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from physio_placebo.baselines.adjective_vote import compare_to_model, eval_vote_frame
from physio_placebo.baselines.input_matched import hrv_feature_ceiling, prompt_stat_ceiling
from physio_placebo.features.frozen import feature_columns
from physio_placebo.paths import REPO_ROOT
from physio_placebo.prompts.ablation_metrics import chance_mean, paired_wilcoxon
from physio_placebo.stats.bootstrap import subject_bootstrap_ci
from physio_placebo.stats.metrics import macro_f1
from physio_placebo.stats.sds import paired_subject_bootstrap_delta, sds_or_undefined
from physio_placebo.stats.secondary import (
    auroc,
    balanced_accuracy,
    holm,
    item_variation_ok,
    margins,
    matched_permutation_floor,
)
from physio_placebo.surrogates.ladder import ALL_SURROGATES, DESCRIPTIVE, NULL_RUNGS

PENDING = "pending real rows"
UNDEFINED = "undefined (validity gate)"
NO_VARIATION = "undefined (no item-level variation)"
DESCRIPTIVE_LABEL = "descriptive, not a null"
IAAFT_LABEL = "keeps level and spectrum (see ledger)"
B_ROWS_NOTE = "real B rows = Week 9 re-score (i)"
MAUS_CAVEAT = (
    "Qwen MAUS variant (i) is 0.485 against a matched floor of 0.499 "
    "(p95 0.519). P2 holds on MAUS only under the filed classical floor."
)

EVAL_FILES = {
    "qwen3_8b": "eval_vllm_qwen3_8b.json",
    "llama31_8b": "eval_vllm_llama31_8b.json",
    "qwen25_vl": "eval_vllm_qwen25_vl.json",
}


def scheme_dir(scheme: str | None) -> str:
    return "none" if scheme in (None, "null") else str(scheme)


def format_sds(value: float | None, *, descriptive: bool) -> str:
    """Primary SDS cell text. A failed gate never becomes a number."""
    if descriptive:
        return DESCRIPTIVE_LABEL
    if value is None:
        return UNDEFINED
    return f"{value:.3f}"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows_ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("failure") and r.get("yhat") is not None]


def _align(real_rows: list[dict], surr_rows: list[dict]):
    by_surr = {r["window_id"]: r for r in _rows_ok(surr_rows)}
    subjects, y, yhat_real, yhat_surr = [], [], [], []
    for row in _rows_ok(real_rows):
        other = by_surr.get(row["window_id"])
        if other is None:
            continue
        subjects.append(str(row["subject"]))
        y.append(int(row["y"]))
        yhat_real.append(int(row["yhat"]))
        yhat_surr.append(int(other["yhat"]))
    return (
        np.asarray(subjects),
        np.asarray(y, dtype=int),
        np.asarray(yhat_real, dtype=int),
        np.asarray(yhat_surr, dtype=int),
    )


def _score_rows(rows: list[dict]) -> dict:
    kept = _rows_ok(rows)
    if not kept:
        return {"macro_f1": None, "ba": None, "auroc": None, "std_margin": None, "n": 0}
    y = np.array([int(r["y"]) for r in kept])
    yhat = np.array([int(r["yhat"]) for r in kept])
    margin = margins(kept)
    return {
        "macro_f1": macro_f1(y, yhat),
        "ba": balanced_accuracy(y, yhat),
        "auroc": auroc(y, margin),
        "std_margin": float(np.std(margin)),
        "n": len(kept),
        "y": y,
        "yhat": yhat,
        "subjects": np.array([str(r["subject"]) for r in kept]),
        "margin": margin,
    }


def _secondary_ratio(real: float, surr: float, chance: float, ci_lo: float) -> str:
    if not (ci_lo > chance):
        return UNDEFINED
    value = sds_or_undefined(real, surr, chance, ci_lo)
    if value is None:
        return UNDEFINED
    return f"{value:.3f}"


def _floors(root: Path) -> dict[str, dict]:
    raw = yaml.safe_load((root / "configs" / "locked_floors.yaml").read_text())
    return {dataset: _read(root / rel) for dataset, rel in raw.items()}


def _real_cells(root: Path) -> list[dict]:
    cells = []
    grid = root / "results" / "eval_grid"
    for model, name in EVAL_FILES.items():
        path = grid / name
        if not path.is_file():
            continue
        for cell in _read(path)["cells"]:
            if cell.get("model") in (None, model):
                cell = dict(cell)
                cell["model"] = model
                cells.append(cell)
    return cells


def _surrogate_path(root: Path, cell: dict, rung: str) -> Path:
    return (
        root
        / "results"
        / "surrogates"
        / "eval"
        / cell["model"]
        / cell["dataset"]
        / cell["paradigm"]
        / scheme_dir(cell.get("scheme"))
        / rung
        / "cell.json"
    )


def _load_real_item_rows(root: Path, cell: dict, real_rows_dir: Path | None) -> tuple[list[dict] | None, str]:
    if cell["paradigm"] == "B":
        path = (
            root
            / "results"
            / "verbalizer_ablation"
            / cell["model"]
            / cell["dataset"]
            / "i"
            / "cell.json"
        )
        if not path.is_file():
            return None, PENDING
        return _read(path)["cell"]["rows"], B_ROWS_NOTE
    if real_rows_dir is None:
        return None, PENDING
    name = EVAL_FILES.get(cell["model"])
    if name is None:
        return None, PENDING
    path = real_rows_dir / name
    if not path.is_file():
        return None, PENDING
    for rescored in _read(path)["cells"]:
        if (
            rescored["dataset"] == cell["dataset"]
            and rescored["paradigm"] == cell["paradigm"]
            and scheme_dir(rescored.get("scheme")) == scheme_dir(cell.get("scheme"))
        ):
            return rescored.get("rows") or [], f"real rows from {path}"
    return None, PENDING


def _classical_f1(floor: dict) -> float:
    return float(floor["results"]["models"]["logreg_l2"]["pooled_macro_f1"])


def build_card(root: Path, *, real_rows_dir: Path | None = None) -> dict:
    """Assemble the card from ``root``. Does not write files."""
    floors = _floors(root)
    chance = {dataset: chance_mean(floor) for dataset, floor in floors.items()}
    classical = {dataset: _classical_f1(floor) for dataset, floor in floors.items()}
    primary = []
    secondary = []
    determinism = []
    deleted_std: dict[tuple, float] = {}
    real_cache: dict[int, dict] = {}

    real_cells = _real_cells(root)
    loaded_surr: dict[tuple, dict] = {}
    for cell in real_cells:
        for rung in ALL_SURROGATES:
            path = _surrogate_path(root, cell, rung)
            if not path.is_file():
                continue
            payload = _read(path)["cell"]
            surr_key = (
                cell["model"],
                cell["dataset"],
                cell["paradigm"],
                scheme_dir(cell.get("scheme")),
                rung,
            )
            loaded_surr[surr_key] = payload
            if rung == "deleted" and payload.get("rows"):
                deleted_std[
                    (cell["model"], cell["dataset"], cell["paradigm"], scheme_dir(cell.get("scheme")))
                ] = float(np.std(margins(_rows_ok(payload["rows"]))))

    for cell in real_cells:
        key = (cell["model"], cell["dataset"], cell["paradigm"], scheme_dir(cell.get("scheme")))
        real_rows, rows_note = _load_real_item_rows(root, cell, real_rows_dir)
        real_scored = _score_rows(real_rows) if real_rows else None
        noise = deleted_std.get(key, float("nan"))
        m_real = float(cell["macro_f1"])
        ci_lo = float(cell["ci_lo"])
        m_chance = chance[cell["dataset"]]
        defined = ci_lo > m_chance
        determinism.append(_determinism_row(cell, real_scored, rows_note))
        for rung in ALL_SURROGATES:
            surr = loaded_surr.get((*key, rung))
            primary.append(
                _primary_row(cell, rung, surr, m_real, ci_lo, m_chance, defined, real_rows, real_scored)
            )
            secondary.append(
                _secondary_row(cell, rung, surr, real_scored, noise, rows_note, real_cache)
            )

    _apply_holm(primary)
    panel = _verbalizer_panel(root, real_cells)
    ceilings = _ceilings(root)
    return {
        "label": "post-registration (addendum 2026-10-30) applies to the secondary layer only",
        "coverage_note": (
            "Paradigm C is Qwen2.5-VL on WESAD and MAUS only. "
            "The grid is not a full 3 paradigms x 2 models x 3 datasets."
        ),
        "primary": primary,
        "secondary": secondary,
        "classical_delta": [
            {
                "model": cell["model"],
                "dataset": cell["dataset"],
                "paradigm": cell["paradigm"],
                "scheme": cell.get("scheme"),
                "m_real": float(cell["macro_f1"]),
                "logreg_macro_f1": classical[cell["dataset"]],
                "delta_vs_logreg": float(cell["macro_f1"]) - classical[cell["dataset"]],
            }
            for cell in real_cells
        ],
        "verbalizer": panel,
        "ceilings": ceilings,
        "determinism": determinism,
        "n_primary": len(primary),
        "n_sds_undefined": sum(1 for row in primary if row["sds"] == UNDEFINED),
        "n_sds_ratio": sum(1 for row in primary if _is_ratio(row["sds"])),
    }


def _is_ratio(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _determinism_row(cell: dict, real_scored: dict | None, rows_note: str) -> dict:
    base = {
        "model": cell["model"],
        "dataset": cell["dataset"],
        "paradigm": cell["paradigm"],
        "scheme": cell.get("scheme"),
        "locked_macro_f1": float(cell["macro_f1"]),
        "rows": rows_note,
    }
    if real_scored is None or real_scored["macro_f1"] is None:
        return {**base, "rescored_macro_f1": PENDING, "abs_delta": PENDING, "flag": PENDING}
    delta = abs(float(real_scored["macro_f1"]) - float(cell["macro_f1"]))
    return {
        **base,
        "rescored_macro_f1": float(real_scored["macro_f1"]),
        "abs_delta": delta,
        "flag": "flag" if delta > 0.02 else "ok",
    }


def _primary_row(cell, rung, surr, m_real, ci_lo, m_chance, defined, real_rows, real_scored) -> dict:
    descriptive = rung in DESCRIPTIVE
    row = {
        "model": cell["model"],
        "dataset": cell["dataset"],
        "paradigm": cell["paradigm"],
        "scheme": cell.get("scheme"),
        "rung": rung,
        "is_null": rung in NULL_RUNGS,
        "m_real": m_real,
        "ci_lo_real": ci_lo,
        "m_chance": m_chance,
        "gate_defined": defined,
        "iaaft_note": IAAFT_LABEL if rung == "iaaft" else None,
        "wilcoxon_p": None,
        "holm_p": None,
    }
    if surr is None:
        row.update(
            {
                "m_surr": None,
                "delta_point": None,
                "delta_ci": "missing surrogate cell",
                "sds": "missing surrogate cell",
            }
        )
        return row
    m_surr = float(surr["macro_f1"])
    row["m_surr"] = m_surr
    row["delta_point"] = m_real - m_surr
    row["wilcoxon_p"] = paired_wilcoxon(cell.get("per_subject") or {}, surr.get("per_subject") or {}).get(
        "pvalue"
    )
    if descriptive:
        row["sds"] = DESCRIPTIVE_LABEL
    else:
        row["sds"] = format_sds(
            sds_or_undefined(m_real, m_surr, m_chance, ci_lo) if defined else None,
            descriptive=False,
        )
    if real_rows is None or not surr.get("rows"):
        row["delta_ci"] = PENDING
        return row
    subjects, y, yhat_real, yhat_surr = _align(real_rows, surr["rows"])
    if len(y) == 0:
        row["delta_ci"] = PENDING
        return row
    ci = paired_subject_bootstrap_delta(subjects, y, yhat_real, yhat_surr)
    row["delta_ci"] = [ci.point, ci.lo, ci.hi]
    return row


def _real_side(cell, real_scored, deleted_std, cache: dict[int, dict]) -> dict:
    """Floor and intervals of the real rows. Shared by every rung of one cell."""
    key = id(real_scored)
    if key in cache:
        return cache[key]
    y, yhat, subjects = real_scored["y"], real_scored["yhat"], real_scored["subjects"]
    floor = matched_permutation_floor(subjects, y, yhat)
    ba_ci = subject_bootstrap_ci(subjects, y, yhat, metric=balanced_accuracy)
    real_var = item_variation_ok(real_scored["margin"], deleted_std)
    auc_ci = None
    if real_var and real_scored["auroc"] == real_scored["auroc"]:
        auc_ci = subject_bootstrap_ci(subjects, y, real_scored["margin"], metric=auroc)
    cache[key] = {"floor": floor, "ba_ci": ba_ci, "real_var": real_var, "auc_ci": auc_ci}
    return cache[key]


def _secondary_row(cell, rung, surr, real_scored, deleted_std, rows_note, real_cache) -> dict:
    row = {
        "model": cell["model"],
        "dataset": cell["dataset"],
        "paradigm": cell["paradigm"],
        "scheme": cell.get("scheme"),
        "rung": rung,
        "rows": rows_note,
        "layer": "post-registration (addendum 2026-10-30)",
    }
    if real_scored is None or surr is None or not surr.get("rows"):
        row.update(
            {
                "ba_real": PENDING,
                "auroc_real": PENDING,
                "matched_floor_mean": PENDING,
                "sds_ba": PENDING,
                "sds_auc": PENDING,
            }
        )
        return row
    surr_scored = _score_rows(surr["rows"])
    side = _real_side(cell, real_scored, deleted_std, real_cache)
    floor = side["floor"]
    ba_ci = side["ba_ci"]
    surr_var = item_variation_ok(surr_scored["margin"], deleted_std)
    auc_ci = side["auc_ci"]
    if side["real_var"] and surr_var and auc_ci is not None:
        sds_auc = _secondary_ratio(real_scored["auroc"], surr_scored["auroc"], 0.5, auc_ci.lo)
        auc_ci_text = [auc_ci.point, auc_ci.lo, auc_ci.hi]
    else:
        sds_auc = NO_VARIATION
        auc_ci_text = NO_VARIATION
    row.update(
        {
            "ba_real": real_scored["ba"],
            "ba_surr": surr_scored["ba"],
            "ba_ci": [ba_ci.point, ba_ci.lo, ba_ci.hi],
            "auroc_real": real_scored["auroc"],
            "auroc_surr": surr_scored["auroc"],
            "auroc_ci": auc_ci_text,
            "matched_floor_mean": floor["mean"],
            "matched_floor_p95": floor["p95"],
            "matched_gate": bool(float(cell["ci_lo"]) > floor["mean"]),
            "sds_ba": _secondary_ratio(real_scored["ba"], surr_scored["ba"], 0.5, ba_ci.lo),
            "sds_auc": sds_auc,
        }
    )
    return row


def _apply_holm(primary: list[dict]) -> None:
    indexes = [i for i, row in enumerate(primary) if row["is_null"]]
    adjusted = holm([primary[i]["wilcoxon_p"] for i in indexes])
    for i, value in zip(indexes, adjusted, strict=True):
        primary[i]["holm_p"] = value


def _verbalizer_panel(root: Path, real_cells: list[dict]) -> dict:
    path = root / "results" / "verbalizer_ablation" / "panel.json"
    if not path.is_file():
        return {"panel": [], "note": "no verbalizer panel in this tree"}
    panel = _read(path)["panel"]
    votes = []
    if root.resolve() == REPO_ROOT.resolve():
        votes = _vote_rows(root, real_cells)
    return {"panel": panel, "vote": votes, "maus_caveat": MAUS_CAVEAT}


def _vote_rows(root: Path, real_cells: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for cell in real_cells:
        if cell["paradigm"] != "B":
            continue
        key = (cell["model"], cell["dataset"])
        if key in seen:
            continue
        seen.add(key)
        frame_path = (
            root / "results" / "verbalizer_ablation" / cell["model"] / cell["dataset"] / "i" / "cell.json"
        )
        if not frame_path.is_file():
            continue
        rows = _read(frame_path)["cell"]["rows"]
        features = _feature_frame(cell["dataset"])
        if features is None:
            continue
        voted = eval_vote_frame(features, feature_columns(cell["dataset"]), cell["eval_subjects"], "i")
        summary = _score_rows(
            [
                {
                    "subject": str(r.subject),
                    "y": int(r.y),
                    "yhat": int(r.yhat),
                    "failure": False,
                    "logprobs": {"A": 0.0, "B": float(r.vote)},
                }
                for r in voted.itertuples(index=False)
            ]
        )
        compared = compare_to_model(voted, rows)
        out.append(
            {
                "model": cell["model"],
                "dataset": cell["dataset"],
                "variant": "i",
                "vote_macro_f1": summary["macro_f1"],
                "vote_ba": summary["ba"],
                "vote_auroc": summary["auroc"],
                "agreement": compared["agreement"],
                "spearman": compared["spearman"],
                "layer": "post-registration (addendum 2026-10-30)",
            }
        )
    return out


def _feature_frame(dataset: str) -> pd.DataFrame | None:
    from physio_placebo.features.matrix import load_locked_feature_matrix

    try:
        return load_locked_feature_matrix(dataset)
    except (FileNotFoundError, OSError):
        return None


def _ceilings(root: Path) -> list[dict]:
    if root.resolve() != REPO_ROOT.resolve():
        return []
    rows = []
    for dataset in ("wesad", "maus", "cogwear"):
        try:
            rows.append(hrv_feature_ceiling(dataset))
        except (FileNotFoundError, OSError, ValueError) as exc:
            rows.append({"dataset": dataset, "ceiling": "hrv", "error": str(exc)})
        for scheme in ("bin_mean", "uniform_stride"):
            try:
                rows.append(prompt_stat_ceiling(dataset, scheme))
            except (FileNotFoundError, OSError, ValueError) as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "scheme": scheme,
                        "ceiling": "prompt_stats",
                        "error": str(exc),
                    }
                )
    for row in rows:
        row["layer"] = "post-registration (addendum 2026-10-30)"
        row["note"] = "Evaluation-subject metrics are eval_macro_f1 and eval_auroc. all_* is the full roster."
    return rows


def render_markdown(card: dict) -> str:
    lines = [
        "# SDS report card",
        "",
        card["coverage_note"],
        "",
        "Primary numbers are the filed macro-F1 SDS. "
        "The secondary layer is post-registration (addendum 2026-10-30).",
        "",
        "## Primary SDS",
        "",
        "| Model | Dataset | View | Rung | M surrogate | Delta | Paired delta | Wilcoxon p | SDS |",
        "|---|---|---|---|---:|---:|---|---:|---|",
    ]
    for row in card["primary"]:
        view = f"{row['paradigm']} {scheme_dir(row.get('scheme'))}"
        delta = _num(row.get("delta_point"))
        paired = _ci(row.get("delta_ci"))
        note = f" {row['iaaft_note']}" if row.get("iaaft_note") else ""
        lines.append(
            f"| {row['model']} | {row['dataset']} | {view} | {row['rung']}{note} | "
            f"{_num(row.get('m_surr'))} | {delta} | {paired} | "
            f"{_num(row.get('wilcoxon_p'), 4)} | {row['sds']} |"
        )
    lines += [
        "",
        "## Classical delta",
        "",
        "| Model | Dataset | View | LLM | Logreg | Delta |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in card["classical_delta"]:
        view = f"{row['paradigm']} {scheme_dir(row.get('scheme'))}"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {view} | {_num(row['m_real'])} | "
            f"{_num(row['logreg_macro_f1'])} | {_num(row['delta_vs_logreg'])} |"
        )
    lines += ["", "## Verbalizer panel", ""]
    verbal = card["verbalizer"]
    if verbal.get("maus_caveat"):
        lines += [verbal["maus_caveat"], ""]
    lines += [
        "| Model | Dataset | As written | Words removed | Words flipped |",
        "|---|---|---:|---:|---:|",
    ]
    for row in verbal.get("panel") or []:
        f1 = row["macro_f1"]
        lines.append(
            f"| {row['model']} | {row['dataset']} | {_num(f1['i'])} | {_num(f1['ii'])} | {_num(f1['iii'])} |"
        )
    if verbal.get("vote"):
        lines += [
            "",
            "Adjective vote (post-registration, no language model).",
            "",
            "| Model | Dataset | Vote macro-F1 | Vote BA | Vote AUROC | Agreement | Spearman |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in verbal["vote"]:
            lines.append(
                f"| {row['model']} | {row['dataset']} | {_num(row['vote_macro_f1'])} | "
                f"{_num(row['vote_ba'])} | {_num(row['vote_auroc'])} | "
                f"{_num(row['agreement'])} | {_num(row['spearman'])} |"
            )
    lines += [
        "",
        "## Secondary layer",
        "",
        "post-registration (addendum 2026-10-30).",
        "",
        "| Model | Dataset | View | Rung | BA | AUROC | Floor | SDS_BA | SDS_AUC | Holm |",
        "|---|---|---|---|---:|---:|---:|---|---|---:|",
    ]
    holm_by = {
        (r["model"], r["dataset"], r["paradigm"], scheme_dir(r.get("scheme")), r["rung"]): r.get("holm_p")
        for r in card["primary"]
    }
    for row in card["secondary"]:
        view = f"{row['paradigm']} {scheme_dir(row.get('scheme'))}"
        holm_key = (
            row["model"],
            row["dataset"],
            row["paradigm"],
            scheme_dir(row.get("scheme")),
            row["rung"],
        )
        holm_p = holm_by[holm_key]
        lines.append(
            f"| {row['model']} | {row['dataset']} | {view} | {row['rung']} | {_num(row.get('ba_real'))} | "
            f"{_num(row.get('auroc_real'))} | {_num(row.get('matched_floor_mean'))} | {row.get('sds_ba')} | "
            f"{row.get('sds_auc')} | {_num(holm_p, 4)} |"
        )
    lines += [
        "",
        "## Input-matched ceilings",
        "",
        "post-registration (addendum 2026-10-30). Surrogate windows are not re-fit.",
        "",
    ]
    if not card["ceilings"]:
        lines.append("Not computed for this tree.")
    else:
        lines += [
            "| Dataset | Ceiling | Scheme | Eval macro-F1 | Eval AUROC | All macro-F1 | All AUROC |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for row in card["ceilings"]:
            if row.get("error"):
                lines.append(
                    f"| {row['dataset']} | {row.get('ceiling')} | {row.get('scheme', '')} | "
                    f"{row['error']} | | | |"
                )
                continue
            lines.append(
                f"| {row['dataset']} | {row['ceiling']} | {row.get('scheme', '')} | "
                f"{_num(row.get('eval_macro_f1'))} | {_num(row.get('eval_auroc'))} | "
                f"{_num(row.get('all_macro_f1'))} | {_num(row.get('all_auroc'))} |"
            )
    lines += [
        "",
        "## Determinism",
        "",
        "| Model | Dataset | View | Locked | Re-score | Abs delta | Flag |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in card["determinism"]:
        view = f"{row['paradigm']} {scheme_dir(row.get('scheme'))}"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {view} | {_num(row['locked_macro_f1'])} | "
            f"{_num(row['rescored_macro_f1'])} | {_num(row['abs_delta'])} | {row['flag']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _num(value, digits: int = 3) -> str:
    if value is None or value == PENDING or isinstance(value, str):
        return PENDING if value == PENDING else ("" if value is None else str(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return ""
    return f"{number:.{digits}f}"


def _ci(value) -> str:
    if value == PENDING or isinstance(value, str):
        return str(value)
    if not isinstance(value, list) or len(value) != 3:
        return ""
    return f"{value[0]:.3f} [{value[1]:.3f}, {value[2]:.3f}]"


def write_figures(card: dict, dest_dir: Path) -> None:
    """Heatmap of filed SDS and AUROC against macro-F1. Same palette as the supervisor report."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dest_dir.mkdir(parents=True, exist_ok=True)
    accent = "#0F4C5C"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.edgecolor": accent,
            "axes.labelcolor": accent,
            "xtick.color": accent,
            "ytick.color": accent,
            "text.color": accent,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )
    _heatmap(card, dest_dir / "fig_sds_heatmap.pdf")
    _scatter(card, dest_dir / "fig_auroc_vs_f1.pdf")
    plt.close("all")


def _heatmap(card: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    rungs = list(NULL_RUNGS)
    rows = []
    labels = []
    for row in card["primary"]:
        if row["rung"] not in rungs:
            continue
        label = f"{row['model']} {row['dataset']} {row['paradigm']} {scheme_dir(row.get('scheme'))}"
        if label not in labels:
            labels.append(label)
            rows.append([np.nan] * len(rungs))
        value = row["sds"]
        if _is_ratio(value):
            rows[labels.index(label)][rungs.index(row["rung"])] = float(value)
    data = np.array(rows, dtype=float) if rows else np.zeros((1, len(rungs)))
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.28 * max(len(labels), 1))))
    im = ax.imshow(data, aspect="auto", cmap="cividis", vmin=-0.5, vmax=1.5)
    ax.set_xticks(range(len(rungs)), rungs, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Filed SDS. Blank cells are undefined (validity gate).")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _scatter(card: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"qwen3_8b": "#0F4C5C", "llama31_8b": "#C46B2C", "qwen25_vl": "#5C3D7A"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    seen = set()
    for row in card["secondary"]:
        if row["rung"] != "gaussian":
            continue
        # Macro-F1 is the filed real score. AUROC is the post-registration margin score.
        f1 = next(
            (
                p["m_real"]
                for p in card["primary"]
                if p["model"] == row["model"]
                and p["dataset"] == row["dataset"]
                and p["paradigm"] == row["paradigm"]
                and scheme_dir(p.get("scheme")) == scheme_dir(row.get("scheme"))
                and p["rung"] == row["rung"]
            ),
            None,
        )
        if not isinstance(row.get("auroc_real"), float) or f1 is None:
            continue
        if row["auroc_real"] != row["auroc_real"]:
            continue
        label = row["model"] if row["model"] not in seen else None
        seen.add(row["model"])
        ax.scatter(f1, row["auroc_real"], color=colors.get(row["model"], "#0F4C5C"), s=18, label=label)
    ax.axhline(0.5, color="#8A9397", lw=0.8, ls="--")
    ax.set_xlabel("macro-F1 (filed real, or surrogate point on the y of the pair)")
    ax.set_ylabel("AUROC of the logprob margin")
    ax.set_title("post-registration (addendum 2026-10-30)")
    if seen:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
