# B6 - diagnostic plots. One hue (blue sequential ramp / single series), direct value labels,
# light chart surface, text in ink colours (never the series colour).
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
BAR = "#2a78d6"


def plot_confusion(cm_df: pd.DataFrame, title: str, path: str) -> None:
    """cm_df: pooled out-of-fold counts (rows = true). Colour = row-normalised share, label = count."""
    counts = cm_df.values
    share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(share, cmap=CMAP, vmin=0, vmax=1)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            ax.text(j, i, f"{counts[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.5 else INK)
    ax.set_xticks(range(counts.shape[1]), [c.replace("pred_", "") for c in cm_df.columns], color=INK2)
    ax.set_yticks(range(counts.shape[0]), [c.replace("true_", "") for c in cm_df.index], color=INK2)
    ax.set_xlabel("predicted", color=INK2); ax.set_ylabel("true", color=INK2)
    ax.set_title(textwrap.fill(title, 38), color=INK, fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)


def plot_importance(imp_series: pd.Series, title: str, path: str, top: int = 12) -> None:
    """imp_series: mean permutation importance (drop in macro-F1) per raw column."""
    s = imp_series.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.34 * len(s) + 1.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(s.index, s.values, color=BAR, height=0.55)
    for y, v in enumerate(s.values):
        ax.text(max(v, 0), y, f" {v:.3f}", va="center", fontsize=9, color=INK2)
    fig.suptitle(textwrap.fill(title, 55), x=0.02, ha="left", color=INK, fontsize=11)
    ax.set_xlabel("mean drop in macro-F1 when the column is shuffled", color=INK2)
    ax.tick_params(colors=INK2, length=0)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d3")
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(path, dpi=160, facecolor=SURFACE); plt.close(fig)
