"""Render results/robustness.png.

Two rows, because the interesting effect is invisible in the metric everyone
watches. Top row: injections still caught after perturbation -- adversarial
training barely moves it, because the clean-trained model is already near
ceiling. Bottom row: safe prompts wrongly flagged after the *same* harmless
perturbation -- this is where adversarial training earns its keep.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from attacks import ATTACKS, HELD_OUT

SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASE = "#2a78d6"    # categorical slot 1
ADV = "#eb6834"     # categorical slot 2

TIERS = [("SLM · Qwen2.5-0.5B", "LoRA SLM (0.5B)", "LoRA SLM + AdvTrain"),
         ("LLM · Qwen2.5-7B", "LoRA LLM (7B)", "LoRA LLM + AdvTrain")]
ROWS = [("robust_recall", "Injections still caught (%)", "higher is better"),
        ("benign_flip", "Safe prompts wrongly flagged (%)", "lower is better")]


def main(out="results"):
    adv = pd.read_csv(f"{out}/adversarial_raw.csv")
    have = set(adv["model"])
    tiers = [t for t in TIERS if t[1] in have and t[2] in have]
    if not tiers:
        print("no complete tier pairs yet; skipping figure")
        return

    atts = list(ATTACKS)
    x = np.arange(len(atts))
    bw = 0.34

    fig, axes = plt.subplots(len(ROWS), len(tiers),
                             figsize=(6.4 * len(tiers), 4.1 * len(ROWS)),
                             dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ri, (metric, ylab, hint) in enumerate(ROWS):
        top = max(
            (adv[adv["model"] == m][metric].mean() * 100
             for _, a, b in tiers for m in (a, b)), default=100)
        ylim = 105 if metric == "robust_recall" else max(12, top * 3.2)
        for ci, (title, m_base, m_adv) in enumerate(tiers):
            ax = axes[ri][ci]
            ax.set_facecolor(SURFACE)
            for j, (model, color) in enumerate([(m_base, BASE), (m_adv, ADV)]):
                sub = adv[adv["model"] == model]
                mean = [sub[sub["attack"] == a][metric].mean() * 100 for a in atts]
                std = [sub[sub["attack"] == a][metric].std(ddof=1) * 100 for a in atts]
                ax.bar(x + (j - 0.5) * (bw + 0.02), mean, bw, yerr=std, capsize=3,
                       color=color, linewidth=0, zorder=3,
                       error_kw=dict(ecolor=INK_2, elinewidth=1, capthick=1))
            ax.set_xticks(x)
            ax.set_xticklabels(
                [a + "\n(held out)" if a in HELD_OUT else a for a in atts],
                fontsize=8.5, color=INK, rotation=20, ha="right")
            ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
            ax.set_axisbelow(True)
            for side in ["top", "right", "left"]:
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="both", length=0, labelsize=9, colors=INK_2)
            ax.set_ylim(0, ylim)
            if ri == 0:
                ax.set_title(title, fontsize=11.5, color=INK,
                             fontweight="bold", loc="left", pad=10)
            if ci == 0:
                ax.set_ylabel(f"{ylab}\n({hint})", fontsize=10,
                              color=INK_2, labelpad=9)

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=BASE),
               plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=ADV)]
    leg = fig.legend(handles, ["LoRA (clean training)", "LoRA + adversarial training"],
                     loc="lower center", ncol=2, frameon=False, fontsize=10.5,
                     bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle("Adversarial training barely changes detection — it removes the "
                 "false-positive shortcut (mean ± std, 3 seeds)",
                 fontsize=12.5, color=INK, fontweight="bold", x=0.02, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{out}/robustness.png", facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}/robustness.png")


if __name__ == "__main__":
    main()
