"""
Generate all manuscript figures from cached results JSON.
Run AFTER all pipeline steps are complete.
Exports: SVG (vector, primary), PDF (vector, backup), PNG (300 DPI, raster).
All figures: grayscale-safe, publication-grade typography, clean layout.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_loader import load_config

cfg = load_config()
RESULTS_DIR = cfg["RESULTS_DIR"]
FIGURES_DIR = cfg["FIGURES_DIR"]
MODELS = ["LR", "RF", "GB"]

# ---------------------------------------------------------------------------
# Publication color palette — navy / crimson / forest-green per-model
# Grayscale-safe: distinct markers + line styles used alongside color
# ---------------------------------------------------------------------------
PALETTE = {
    "LR": {"color": "#1f3c6e", "ls": "-",  "marker": "o", "hatch": ""},
    "RF": {"color": "#b5121b", "ls": "--", "marker": "s", "hatch": "///"},
    "GB": {"color": "#2a6f41", "ls": "-.", "marker": "^", "hatch": "..."},
}
ACCENT   = "#c97829"   # copper/amber — thresholds, annotations
DIAG     = "#888888"   # gray diagonal
GRID_C   = "#eeeeee"   # grid lines
BIN_ALPHA = 0.18        # transparency for bin histograms

THEME = {
    "lw":  1.6,
    "cap": 3,
    "ms":  5.0,
}

plt.rcParams.update({
    "font.family":          "DejaVu Serif",
    "font.size":            12,
    "axes.labelsize":       11,
    "axes.titlesize":       12,
    "axes.titleweight":     "normal",
    "axes.labelcolor":      "0.15",
    "axes.edgecolor":       "0.55",
    "axes.linewidth":       0.9,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "xtick.color":          "0.4",
    "ytick.color":          "0.4",
    "xtick.major.size":     4.0,
    "ytick.major.size":     4.0,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "legend.fontsize":      10,
    "legend.framealpha":    0.93,
    "legend.edgecolor":     "0.72",
    "legend.borderpad":     0.5,
    "figure.dpi":           150,
    "savefig.dpi":          600,
    "axes.spines.right":    False,
    "axes.spines.top":      False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
})


def _source_note(fig, text: str, y: float = -0.04):
    """Consistent footnote placement across all figures."""
    fig.text(0.5, y, text, ha="center", fontsize=9.0,
             color="0.50", style="italic")


def savefig(name: str):
    """Save SVG, PDF, PNG (600 DPI), and TIFF (600 DPI) for JEI submission."""
    svg_dir  = os.path.join(FIGURES_DIR, "svg")
    pdf_dir  = os.path.join(FIGURES_DIR, "pdf")
    tiff_dir = os.path.join(FIGURES_DIR, "tiff")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(svg_dir,     exist_ok=True)
    os.makedirs(pdf_dir,     exist_ok=True)
    os.makedirs(tiff_dir,    exist_ok=True)
    outputs = [
        (os.path.join(svg_dir,     f"{name}.svg"),  {}),
        (os.path.join(pdf_dir,     f"{name}.pdf"),  {}),
        (os.path.join(FIGURES_DIR, f"{name}.png"),  {"dpi": 600}),
        (os.path.join(tiff_dir,    f"{name}.tiff"), {"dpi": 600}),
    ]
    for path, kw in outputs:
        plt.savefig(path, bbox_inches="tight", **kw)
    plt.close()
    print(f"  Saved: svg/{name}.svg  |  pdf/{name}.pdf  |  {name}.png  |  tiff/{name}.tiff")


def figure1_reliability_diagrams():
    """Fig 1: 1×3 reliability diagrams — best-calibrated probabilities per model."""
    with open(os.path.join(RESULTS_DIR, "calibration_results.json")) as f:
        calib = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), sharey=True, sharex=True)
    fig.suptitle("Reliability Diagrams — Best-Calibrated Probabilities",
                 fontsize=12, y=1.02)

    for col, name in enumerate(MODELS):
        ax   = axes[col]
        pal  = PALETTE[name]
        best_m = calib[name]["best_method"]
        bins   = calib[name][best_m]["reliability_bins"]

        centers = np.array(bins["bin_centers"])
        accs    = np.array([a if (a is not None and not (
                    isinstance(a, float) and np.isnan(a))) else np.nan
                    for a in bins["bin_accuracies"]])
        confs   = np.array(bins["bin_confidences"])
        counts  = np.array(bins["bin_counts"])

        # Bin-fraction histogram bars — tinted with model color
        total = max(counts.sum(), 1)
        import matplotlib.colors as mcolors
        rgba  = list(mcolors.to_rgba(pal["color"]))
        rgba[3] = BIN_ALPHA
        ax.bar(centers, counts / total, width=0.088,
               color=rgba, edgecolor=pal["color"], linewidth=0.4,
               align="center", zorder=1, label="_nolegend_")

        # Calibration curve
        valid = ~np.isnan(accs)
        if valid.any():
            ax.plot(confs[valid], accs[valid],
                    marker=pal["marker"], ls="-",
                    color=pal["color"], lw=THEME["lw"], ms=THEME["ms"],
                    zorder=3, clip_on=False)

        # Perfect-calibration diagonal
        ax.plot([0, 1], [0, 1], "--", color=DIAG, lw=0.9, zorder=2)

        # Annotation box
        ece_cal = calib[name]["ece_calibrated"]
        ece_raw = calib[name]["raw"]["ece"]
        label   = best_m if best_m != "raw" else "raw (no gain)"
        ax.text(0.04, 0.97,
                f"ECE = {ece_cal:.3f}  [{label}]\nECE\u2080 = {ece_raw:.3f}",
                transform=ax.transAxes, fontsize=9, va="top", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=pal["color"], alpha=0.95, linewidth=0.8))

        ax.set_title(name, fontsize=12, pad=5, color=pal["color"], fontweight="bold")
        ax.set_xlabel("Mean predicted probability", fontsize=11)
        if col == 0:
            ax.set_ylabel("Observed positive fraction", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.yaxis.grid(True, color=GRID_C, lw=0.5, zorder=0)

    plt.tight_layout(rect=[0, 0, 1, 1])
    _source_note(fig, "IBM HR Analytics Employee Attrition (N=1,470; test N=294). "
                 "Dashed diagonal = perfect calibration. Shaded bars = bin sample fraction.", y=-0.05)
    savefig("fig1_reliability_diagrams")


def figure2_coverage_accuracy():
    """Fig 2: Coverage-accuracy curves with per-model Gov-M threshold markers."""
    with open(os.path.join(RESULTS_DIR, "selective_prediction_results.json")) as f:
        sp = json.load(f)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    ymin, ymax = 0.70, 1.01

    for name in MODELS:
        pal = PALETTE[name]
        c   = sp[name]["coverage_accuracy_curve"]
        cov = np.array(c["coverages"])
        acc = np.array(c["accuracies"])
        mask = cov >= 0.50
        ax.plot(cov[mask], acc[mask], pal["ls"],
                color=pal["color"], lw=THEME["lw"],
                marker=pal["marker"], markevery=12, ms=4.5,
                label=name, zorder=3)

    # Gov-M vertical ticks per model
    gov_m = {m: 1.0 - sp[m]["threshold_analysis"]["gov_moderate"]["abstention_rate"]
             for m in MODELS}
    label_offsets = {"LR": -0.010, "RF": +0.010, "GB": 0.0}
    for name in MODELS:
        cov_val = gov_m[name]
        ax.axvline(cov_val, color=PALETTE[name]["color"],
                   linestyle=":", lw=0.9, alpha=0.55, zorder=2)
        ax.text(cov_val + label_offsets[name], ymax - 0.008,
                f"{name}\n{cov_val:.2f}",
                fontsize=8, color=PALETTE[name]["color"],
                va="top", ha="center", alpha=0.80,
                linespacing=1.3)

    ax.set_xlabel("Coverage (fraction of predictions retained)", fontsize=11)
    ax.set_ylabel("Accuracy on retained predictions", fontsize=11)
    ax.set_xlim(0.50, 1.01)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax.yaxis.grid(True, color=GRID_C, lw=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Coverage–Accuracy Tradeoff Under Selective Prediction", fontsize=12)

    plt.tight_layout()
    _source_note(fig, "IBM HR Analytics Employee Attrition (N=1,470; test N=294). "
                 "Dotted verticals = Gov-Moderate threshold (conf. < 0.70) per model.", y=-0.05)
    savefig("fig2_coverage_accuracy")


def figure3_shap_stability():
    """Fig 3: Paired bars — per-sample Spearman ρ (±1 SD) by confidence zone."""
    with open(os.path.join(RESULTS_DIR, "shap_stability_results.json")) as f:
        shap_data = json.load(f)

    def _get(m, key, fallback=0.0):
        v = shap_data[m].get(key)
        return fallback if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    ssi_d   = [_get(m, "ssi_zone_d")     for m in MODELS]
    ssi_a   = [_get(m, "ssi_zone_a")     for m in MODELS]
    std_d   = [_get(m, "ssi_zone_d_std") for m in MODELS]
    std_a   = [_get(m, "ssi_zone_a_std") for m in MODELS]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.arange(len(MODELS))
    w = 0.30

    import matplotlib.colors as mcolors
    for i, name in enumerate(MODELS):
        pal  = PALETTE[name]
        rgba = list(mcolors.to_rgba(pal["color"]))

        ax.bar(x[i] + w / 2, ssi_d[i], w,
               color=pal["color"], edgecolor=pal["color"], lw=0.7,
               yerr=std_d[i], capsize=THEME["cap"],
               error_kw={"lw": 0.9, "ecolor": "0.25"}, zorder=3)

        rgba_light = rgba[:]; rgba_light[3] = 0.25
        ax.bar(x[i] - w / 2, ssi_a[i], w,
               color=rgba_light, edgecolor=pal["color"], lw=0.9,
               hatch=pal["hatch"] if pal["hatch"] else "///",
               yerr=std_a[i], capsize=THEME["cap"],
               error_kw={"lw": 0.9, "ecolor": "0.25"}, zorder=3)

    # Null-result annotation: all SSI values 0.69–0.75; no zone differentiation
    ax.annotate("",
                xy=(len(MODELS) - 0.6, 0.765), xytext=(0 - 0.3, 0.765),
                arrowprops=dict(arrowstyle="<->", color="0.50", lw=0.9,
                                connectionstyle="arc3,rad=0.0"))
    ax.text((len(MODELS) - 1) / 2, 0.79,
            "No zone-level differentiation (SSI 0.69–0.75 across all models)",
            ha="center", fontsize=9, color="0.38", style="italic")

    ax.yaxis.grid(True, color=GRID_C, lw=0.5, zorder=0)

    ylo = min(min(ssi_a), min(ssi_d)) - 0.10
    ax.set_ylim(min(ylo, 0.60), 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#1f3c6e", edgecolor="#1f3c6e", label="Zone D (high conf.)"),
        Patch(facecolor=(0.12, 0.23, 0.43, 0.25), edgecolor="#1f3c6e",
              hatch="///", label="Zone A (low conf.)"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=10)

    ax.set_ylabel("Mean per-sample Spearman ρ\n(B = 50 bootstrap resamples)", fontsize=11)
    ax.set_title("SHAP Explanation Stability by Confidence Zone", fontsize=12)

    plt.tight_layout()
    _source_note(fig, "IBM HR Analytics Employee Attrition (N=1,470; test N=294). "
                 "Error bars = ±1 SD across cases. ρ = 1 → perfectly stable ranking.", y=-0.05)
    savefig("fig3_shap_stability")


def figure4_gadrs():
    """Fig 4: GADRS readiness scores — per-model colors with sensitivity range."""
    with open(os.path.join(RESULTS_DIR, "gadrs_results.json")) as f:
        gadrs_data = json.load(f)

    primary   = [gadrs_data["gadrs"]["primary"][m]["gadrs_readiness"]     for m in MODELS]
    equal     = [gadrs_data["gadrs"]["equal_weights"][m]["gadrs_readiness"] for m in MODELS]
    ece_heavy = [gadrs_data["gadrs"]["ece_heavy"][m]["gadrs_readiness"]    for m in MODELS]

    all_specs = list(zip(primary, equal, ece_heavy))
    err_low  = [max(0.0, p - min(p, e, h)) for p, e, h in all_specs]
    err_high = [max(0.0, max(p, e, h) - p) for p, e, h in all_specs]
    top_err  = max(err_high) if err_high else 0.02

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    x = np.arange(len(MODELS))
    colors = [PALETTE[m]["color"] for m in MODELS]

    ax.bar(x, primary, color=colors, edgecolor="0.25", lw=0.7,
           width=0.50, zorder=3, label="Primary weights")
    ax.errorbar(x, primary, yerr=[err_low, err_high], fmt="none",
                color="0.20", capsize=5, lw=1.4, zorder=4,
                label="Sensitivity range (3 specs)")

    # Value labels above each bar
    for i, (p, name) in enumerate(zip(primary, MODELS)):
        ax.text(i, p + top_err + 0.018, f"{p:.3f}",
                ha="center", fontsize=10, color=PALETTE[name]["color"],
                fontweight="bold")

    ax.yaxis.grid(True, color=GRID_C, lw=0.5, zorder=0)
    ax.set_xticks(x)
    xlabels = []
    for name in MODELS:
        xlabels.append(name)
    ax.set_xticklabels(xlabels)
    for tick, name in zip(ax.get_xticklabels(), MODELS):
        tick.set_color(PALETTE[name]["color"])
        tick.set_fontweight("bold")
    ax.set_ylabel("GADRS readiness score  (0 = low, 1 = high)", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_title("Governance-Aware Deployment Readiness (GADRS)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    _source_note(fig, "IBM HR Analytics Employee Attrition (N=1,470). "
                 "Error bars = range over three weight specifications (primary, equal, ECE-heavy).", y=-0.06)
    savefig("fig4_gadrs_comparison")


def figure5_shift():
    """Fig 5: ECE and AURC_delta before/after covariate shift — 2-panel comparison."""
    with open(os.path.join(RESULTS_DIR, "gadrs_results.json")) as f:
        gadrs_data = json.load(f)

    sub   = gadrs_data["sub_scores"]
    shift = gadrs_data["covariate_shift"]

    ece_base  = [sub[m]["ece_calibrated"] for m in MODELS]
    aurc_base = [sub[m]["aurc_delta"]     for m in MODELS]

    ece_shift, aurc_shift = [], []
    for m in MODELS:
        s = shift.get(m, {})
        ece_shift.append( s.get("ece_calibrated_shifted", sub[m]["ece_calibrated"]))
        aurc_shift.append(s.get("aurc_delta_shifted",     sub[m]["aurc_delta"]))

    import matplotlib.colors as mcolors
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    x   = np.arange(len(MODELS))
    w   = 0.30
    pad = 0.003
    model_colors = [PALETTE[m]["color"] for m in MODELS]

    def _paired_bars(ax, base, after, ylabel, title):
        for i, name in enumerate(MODELS):
            c    = model_colors[i]
            rgba = list(mcolors.to_rgba(c)); rgba[3] = 0.25
            ax.bar(x[i] - w / 2, base[i],  w, label="Baseline" if i == 0 else "_nl",
                   color=rgba, edgecolor=c, lw=0.9, zorder=3)
            ax.bar(x[i] + w / 2, after[i], w, label="After shift" if i == 0 else "_nl",
                   color=c,    edgecolor=c, lw=0.9, zorder=3)
            delta = after[i] - base[i]
            sign  = "+" if delta >= 0 else ""
            ax.text(x[i] + w / 2, after[i] + pad,
                    f"{sign}{delta:.3f}", ha="center", fontsize=9,
                    color=c, va="bottom", fontweight="bold")
        ymax = max(max(base), max(after))
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        for tick, name in zip(ax.get_xticklabels(), MODELS):
            tick.set_color(PALETTE[name]["color"])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, ymax * 1.35)
        ax.yaxis.grid(True, color=GRID_C, lw=0.5, zorder=0)
        ax.legend(fontsize=10, loc="upper right")

    _paired_bars(axes[0], ece_base,  ece_shift,
                 "Expected Calibration Error (lower = better)",
                 "A.  Calibration Error (ECE)")
    _paired_bars(axes[1], aurc_base, aurc_shift,
                 "AURC\u0394  (higher = better)",
                 "B.  Selective Prediction Advantage (AURC\u0394)")

    fig.suptitle("Metric Change Under Simulated Covariate Shift", fontsize=12, y=1.02)
    plt.tight_layout()
    _source_note(fig, "IBM HR Analytics Employee Attrition (N=1,470; test N=294). "
                 "Shift: Gaussian noise (\u03c3 = 0.5\u00d7feature SD) on top-3 SHAP features per model.", y=-0.05)
    savefig("fig5_distribution_shift")


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    steps = [
        ("Figure 1: Reliability Diagrams",    figure1_reliability_diagrams),
        ("Figure 2: Coverage-Accuracy Curves", figure2_coverage_accuracy),
        ("Figure 3: SHAP Stability",           figure3_shap_stability),
        ("Figure 4: GADRS Comparison",         figure4_gadrs),
        ("Figure 5: Covariate Shift",          figure5_shift),
    ]
    for label, fn in steps:
        print(f"Generating {label}...")
        fn()
    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print("Formats: SVG (primary), PDF (backup), PNG (300 dpi)")


if __name__ == "__main__":
    run()
