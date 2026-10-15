"""Figures for reports/supervisor_master_report.tex from locked eval JSONs."""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"

ACCENT = "#0F4C5C"
SOFT = "#1B6B7A"
QWEN = "#0F4C5C"
LLAMA = "#C46B2C"
CLASSICAL = "#1F6B4A"
MAJORITY = "#8A9397"
C_COLOR = "#5C3D7A"
PAPER = "#7A2E0E"

FLOORS = {
    "wesad": {"classical": 0.9181, "majority": 0.4123, "label": "WESAD"},
    "maus": {"classical": 0.4815, "majority": 0.4000, "label": "MAUS"},
    "cogwear": {"classical": 0.5692, "majority": 0.3968, "label": "CogWear"},
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": ACCENT,
            "axes.labelcolor": ACCENT,
            "xtick.color": ACCENT,
            "ytick.color": ACCENT,
            "text.color": ACCENT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def _load_cells(name: str) -> list[dict]:
    return json.loads((ROOT / "results" / "eval_grid" / name).read_text())["cells"]


def _cell_label(c: dict) -> str:
    if c["paradigm"] == "A" and c["scheme"] == "uniform_stride":
        return "every nth"
    if c["paradigm"] == "A" and c["scheme"] == "bin_mean":
        return "128 averages"
    if c["paradigm"] == "B":
        return "sentences"
    if c["paradigm"] == "C":
        return "plot"
    return f"{c['paradigm']}"


def _yerr(c: dict) -> np.ndarray:
    f1, lo, hi = c["macro_f1"], c["ci_lo"], c["ci_hi"]
    return np.array([[f1 - lo], [hi - f1]])


def fig_floors_vs_llm(qwen: list[dict], llama: list[dict]) -> None:
    datasets = ("wesad", "maus", "cogwear")
    labels = [FLOORS[d]["label"] for d in datasets]
    classical = [FLOORS[d]["classical"] for d in datasets]
    majority = [FLOORS[d]["majority"] for d in datasets]
    q_best = [max(c["macro_f1"] for c in qwen if c["dataset"] == d) for d in datasets]
    l_best = [max(c["macro_f1"] for c in llama if c["dataset"] == d) for d in datasets]

    x = np.arange(len(datasets))
    w = 0.18
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    ax.bar(x - 1.5 * w, majority, w, color=MAJORITY, label="Guessing")
    ax.bar(x - 0.5 * w, classical, w, color=CLASSICAL, label="Best simple model")
    ax.bar(x + 0.5 * w, q_best, w, color=QWEN, label="Best Qwen")
    ax.bar(x + 1.5 * w, l_best, w, color=LLAMA, label="Best Llama")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_floors_vs_llm.pdf")
    plt.close(fig)


def fig_ab_grid(qwen: list[dict], llama: list[dict]) -> None:
    order = [
        ("wesad", "A", "uniform_stride"),
        ("wesad", "A", "bin_mean"),
        ("wesad", "B", None),
        ("maus", "A", "uniform_stride"),
        ("maus", "A", "bin_mean"),
        ("maus", "B", None),
        ("cogwear", "A", "uniform_stride"),
        ("cogwear", "A", "bin_mean"),
        ("cogwear", "B", None),
    ]

    def pick(rows: list[dict], key: tuple) -> dict:
        ds, par, sch = key
        for c in rows:
            if c["dataset"] == ds and c["paradigm"] == par and c["scheme"] == sch:
                return c
        raise KeyError(key)

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3.2), sharey=True)
    for ax, ds in zip(axes, ("wesad", "maus", "cogwear"), strict=True):
        keys = [k for k in order if k[0] == ds]
        xs = np.arange(len(keys))
        w = 0.34
        q_cells = [pick(qwen, k) for k in keys]
        l_cells = [pick(llama, k) for k in keys]
        q_y = [c["macro_f1"] for c in q_cells]
        l_y = [c["macro_f1"] for c in l_cells]
        q_e = np.hstack([_yerr(c) for c in q_cells])
        l_e = np.hstack([_yerr(c) for c in l_cells])
        ax.bar(xs - w / 2, q_y, w, yerr=q_e, color=QWEN, capsize=2, error_kw={"elinewidth": 0.8})
        ax.bar(xs + w / 2, l_y, w, yerr=l_e, color=LLAMA, capsize=2, error_kw={"elinewidth": 0.8})
        ax.axhline(FLOORS[ds]["majority"], color=MAJORITY, ls="--", lw=1.0)
        ax.set_xticks(xs, [_cell_label(c) for c in q_cells], rotation=20, ha="right")
        ax.set_title(FLOORS[ds]["label"])
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("macro-F1")
    axes[0].bar([], [], color=QWEN, label="Qwen3-8B")
    axes[0].bar([], [], color=LLAMA, label="Llama-3.1-8B")
    axes[0].plot([], [], color=MAJORITY, ls="--", label="Guessing")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ab_grid.pdf")
    plt.close(fig)


def fig_paradigm_c(qwen: list[dict], vl: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.05), sharey=True)
    for ax, ds in zip(axes, ("wesad", "maus"), strict=True):
        maj = FLOORS[ds]["majority"]
        a_cells = [c for c in qwen if c["dataset"] == ds and c["paradigm"] == "A"]
        b = next(c for c in qwen if c["dataset"] == ds and c["paradigm"] == "B")
        a = max(a_cells, key=lambda c: c["macro_f1"])
        c = next(cell for cell in vl if cell["dataset"] == ds)
        items = [
            ("Guessing", maj, None, MAJORITY),
            ("Qwen, numbers", a["macro_f1"], _yerr(a), QWEN),
            ("Qwen, sentences", b["macro_f1"], _yerr(b), SOFT),
            ("Vision model", c["macro_f1"], _yerr(c), C_COLOR),
        ]
        xs = np.arange(len(items))
        for i, (_name, y, err, color) in enumerate(items):
            ax.bar(i, y, 0.62, color=color, yerr=err, capsize=2, error_kw={"elinewidth": 0.8})
        ax.set_xticks(xs, [n for n, *_ in items], rotation=22, ha="right")
        ax.set_title(FLOORS[ds]["label"])
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("macro-F1")
    fig.tight_layout()
    fig.savefig(OUT / "fig_paradigm_c.pdf")
    plt.close(fig)


def fig_methods() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 1.65))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    labels = [
        "60 s\nwindow",
        "A / B / C\nbody",
        "logprob\nA vs B",
        "freeze on\nprompt-dev",
        "eval +\nbootstrap CI",
    ]
    xs = [0.22, 2.22, 4.22, 6.22, 8.22]
    for x, lab in zip(xs, labels, strict=True):
        box = FancyBboxPatch(
            (x, 0.38),
            1.56,
            1.24,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor="#E8F1F3",
            edgecolor=ACCENT,
            linewidth=1.15,
        )
        ax.add_patch(box)
        ax.text(x + 0.78, 1.00, lab, ha="center", va="center", fontsize=8, color=ACCENT)
    for x in xs[:-1]:
        ax.add_patch(
            FancyArrowPatch(
                (x + 1.58, 1.00),
                (x + 1.98, 1.00),
                arrowstyle="-|>",
                mutation_scale=9,
                color=ACCENT,
                lw=1.05,
            )
        )
    fig.tight_layout()
    fig.savefig(OUT / "fig_methods.pdf")
    plt.close(fig)


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def _clip_series(body: str, n_head: int = 8, n_tail: int = 3) -> str:
    prefix, _, rest = body.partition(": ")
    vals = rest.split()
    mid = f"{len(vals) - n_head - n_tail} values omitted"
    shown = (
        " ".join(vals[:n_head])
        + f"\n... {mid} ...\n"
        + " ".join(vals[-n_tail:])
    )
    return f"{prefix}:\n{shown}"


def _b_excerpt(body: str, want: int = 4) -> str:
    sentences = [s.strip().rstrip(".") + "." for s in body.split(". ") if s.strip()]
    prefer = ("eda_tonic_mean", "hrv_mean_hr", "hrv_sdnn", "hrv_rmssd")
    picked: list[str] = []
    for key in prefer:
        for s in sentences:
            if s.startswith(key + " ") and s not in picked:
                picked.append(s)
                break
        if len(picked) >= want:
            break
    if len(picked) < want:
        for s in sentences:
            if s not in picked:
                picked.append(s)
            if len(picked) >= want:
                break
    return "\n".join(textwrap.fill(s, width=70) for s in picked)


def fig_inputs() -> None:
    from physio_placebo.data.loso import prompt_dev_split
    from physio_placebo.features.frozen import feature_columns
    from physio_placebo.features.matrix import load_locked_feature_matrix
    from physio_placebo.paradigms.plot import plot_body
    from physio_placebo.paradigms.series import load_windows, series_body
    from physio_placebo.prompts.verbalizer import train_medians, verbalize_row

    df = load_locked_feature_matrix("wesad")
    subjects = sorted(df["subject"].astype(str).unique())
    _dev, ev = prompt_dev_split(subjects, n_dev=2, seed=1337)
    held = ev[0]
    rows = df[df["subject"].astype(str) == held].sort_values(["t0", "seg_name"])
    row = rows.iloc[0]
    train = [s for s in subjects if s != held]
    medians = train_medians(df[df["subject"].astype(str).isin(train)], feature_columns("wesad"))
    sw = load_windows("wesad", held)
    a_body = series_body(
        sw, t0=float(row.t0), seg_name=str(row.seg_name), scheme="bin_mean", dataset="wesad"
    )
    b_body = verbalize_row(row, medians, feature_columns("wesad"))
    _caption, png = plot_body(sw, t0=float(row.t0), seg_name=str(row.seg_name), dataset="wesad")
    Image.open(io.BytesIO(png)).save(OUT / "fig_input_c.png")
    a_lines = [ln for ln in _clip_series(a_body).splitlines() if ln]
    b_lines = _b_excerpt(b_body).splitlines()
    tex = (
        r"\noindent\textit{Numbers --- the ECG averaged into 128 bins}" + "\n"
        r"\begin{quote}\ttfamily\footnotesize" + "\n"
        + r"\\".join(_tex_escape(line) for line in a_lines) + "\n"
        r"\end{quote}" + "\n"
        r"\noindent\textit{Sentences --- 4 of the 18 feature lines}" + "\n"
        r"\begin{quote}\ttfamily\footnotesize" + "\n"
        + r"\\".join(_tex_escape(line) for line in b_lines) + "\n"
        r"\end{quote}" + "\n"
    )
    (OUT / "input_excerpts.tex").write_text(tex)


def fig_waveforms() -> None:
    """Eight seconds of one WESAD ECG window, real and three replacements."""
    import sys

    root = Path("/tmp/wesad-wave")
    if not (root / "real" / "S10.npz").is_file():
        print("skip waveforms: /tmp/wesad-wave missing", file=sys.stderr)
        return
    sys.path.insert(0, str(ROOT / "src"))
    from physio_placebo.data.windows import load_subject_windows

    real = load_subject_windows(root / "real" / "S10.npz")
    stress = [i for i, y in enumerate(real.y) if int(y) == 1]
    idx = stress[0]
    fs = float(real.fs["ecg"])
    n = int(8 * fs)
    series = {
        "Real ECG": real.channels["ecg"][idx][:n],
        "Phase-randomized": load_subject_windows(root / "iaaft" / "S10.npz").channels["ecg"][idx][:n],
        "Gaussian noise": load_subject_windows(root / "gaussian" / "S10.npz").channels["ecg"][idx][:n],
        "Flat line": load_subject_windows(root / "flatline" / "S10.npz").channels["ecg"][idx][:n],
    }
    t = np.arange(n) / fs
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 4.8), sharex=True, sharey=True)
    colors = [ACCENT, SOFT, LLAMA, MAJORITY]
    short = ["Real ECG", "Phase-randomized", "Gaussian noise", "Flat line"]
    for ax, title, y, color in zip(axes, short, series.values(), colors, strict=True):
        ax.plot(t, y, color=color, lw=0.55)
        ax.set_ylabel(title, fontsize=8)
        ax.set_xlim(0, 8)
    axes[-1].set_xlabel("Seconds")
    fig.tight_layout()
    fig.savefig(OUT / "fig_waveforms.pdf")
    plt.close(fig)


def fig_surrogate_qwen() -> None:
    """Qwen3-8B on WESAD, real recording versus each replacement."""
    root = ROOT / "results" / "surrogates" / "eval" / "qwen3_8b" / "wesad"
    order = [
        ("Real", None),
        ("Noise", "gaussian"),
        ("Phase\nrand.", "iaaft"),
        ("Wrong\ncondition", "subject_perm_broken"),
        ("Flat line", "flatline"),
        ("Empty", "deleted"),
        ("Same\ncondition", "subject_perm_preserved"),
    ]
    real = {"A": 0.595, "B": 0.584}
    real_err = {"A": (0.595 - 0.550, 0.647 - 0.595), "B": (0.584 - 0.462, 0.699 - 0.584)}

    def load(paradigm: str, scheme: str, kind: str) -> tuple[float, tuple[float, float]]:
        path = root / paradigm / scheme / kind / "cell.json"
        cell = json.loads(path.read_text())["cell"]
        f1 = float(cell["macro_f1"])
        lo = float(cell["ci_lo"])
        hi = float(cell["ci_hi"])
        return f1, (f1 - lo, hi - f1)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.15), sharey=True)
    for ax, paradigm, scheme, title in (
        (axes[0], "A", "bin_mean", "Raw ECG, 128 bins"),
        (axes[1], "B", "none", "Feature sentences"),
    ):
        ys, errs, colors = [], [], []
        for i, (label, kind) in enumerate(order):
            if kind is None:
                ys.append(real[paradigm])
                errs.append(real_err[paradigm])
                colors.append(CLASSICAL)
            else:
                f1, err = load(paradigm, scheme, kind)
                ys.append(f1)
                errs.append(err)
                colors.append(PAPER if kind == "subject_perm_preserved" else QWEN)
        xs = np.arange(len(order))
        yerr = np.array(errs).T
        ax.bar(xs, ys, color=colors, yerr=yerr, capsize=2, error_kw={"elinewidth": 0.7})
        ax.axhline(0.412, color=MAJORITY, ls="--", lw=0.9)
        ax.set_xticks(xs, [lab for lab, _ in order], fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("macro-F1")
    axes[0].bar([], [], color=CLASSICAL, label="Real recording")
    axes[0].bar([], [], color=QWEN, label="Null replacement")
    axes[0].bar([], [], color=PAPER, label="Same condition, other person")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_surrogate_qwen.pdf")
    plt.close(fig)


def fig_ledger() -> None:
    labels = ["Noise", "Phase\nrandomized", "Wrong\ncondition", "Flat line", "Same\ncondition"]
    wesad = [0.412, 0.880, 0.411, 0.412, 0.954]
    maus = [0.400, 0.405, 0.405, np.nan, 0.494]
    cog = [0.486, 0.458, 0.380, 0.397, 0.773]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.6, 3.15))
    ax.bar(x - w, wesad, w, color=QWEN, label="WESAD")
    ax.bar(x, maus, w, color=LLAMA, label="MAUS")
    ax.bar(x + w, cog, w, color=CLASSICAL, label="CogWear")
    ax.axhline(0.412, color=MAJORITY, ls="--", lw=0.9, label="WESAD guessing")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Logistic regression macro-F1")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ledger.pdf")
    plt.close(fig)


def fig_anchor() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 2.85))
    ax.bar([0], [0.76], 0.55, color=PAPER, yerr=[[0.1], [0.1]], capsize=3, label="Paper")
    ax.bar([1], [2.689], 0.55, color=ACCENT, yerr=[[0.038], [0.038]], capsize=3, label="Ours")
    ax.set_xticks([0, 1], ["Health-LLM\npaper", "This run\n(A6000)"])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_ylim(0, 3.1)
    ax.set_title("PMData stress, zero-shot MedAlpaca-7b")
    fig.tight_layout()
    fig.savefig(OUT / "fig_anchor_mae.pdf")
    plt.close(fig)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    qwen = _load_cells("eval_vllm_qwen3_8b.json")
    llama = _load_cells("eval_vllm_llama31_8b.json")
    vl = _load_cells("eval_vllm_qwen25_vl.json")
    fig_methods()
    try:
        fig_inputs()
    except FileNotFoundError:
        print("skip input excerpts: processed windows are not on this machine")
    fig_floors_vs_llm(qwen, llama)
    fig_ab_grid(qwen, llama)
    fig_paradigm_c(qwen, vl)
    fig_anchor()
    fig_waveforms()
    fig_surrogate_qwen()
    fig_ledger()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
