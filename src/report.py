"""Turn the raw per-seed CSVs into the tables and figure used in the README."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from attacks import ATTACKS, HELD_OUT

ORDER = ["Always-INJECTION", "TF-IDF + LR", "Zero-shot SLM (0.5B)", "Zero-shot LLM (7B)",
         "LoRA SLM (0.5B)", "LoRA SLM + AdvTrain", "LoRA LLM (7B)", "LoRA LLM + AdvTrain"]


def pm(g):
    g = g.dropna()
    if g.empty:
        return "–"
    if len(g) == 1:
        return f"{g.iloc[0]*100:.1f}"
    return f"{g.mean()*100:.1f} ± {g.std(ddof=1)*100:.1f}"


def present(df):
    return [m for m in ORDER if m in set(df["model"])]


def main(out="results"):
    clean = pd.read_csv(f"{out}/clean_raw.csv")
    adv = pd.read_csv(f"{out}/adversarial_raw.csv")
    order = present(clean)

    cols = ["accuracy", "balanced_accuracy", "mcc", "f1", "pred_pos_rate"]
    clean_tbl = clean.groupby("model")[cols].agg(pm).reindex(order)
    clean_tbl.columns = ["Accuracy", "Balanced acc.", "MCC", "F1", "Predicted-positive rate"]

    atts = [a for a in ATTACKS]
    rr = (adv.groupby(["model", "attack"])["robust_recall"].agg(pm)
          .unstack("attack").reindex(order)[atts])
    rr["mean"] = (adv.groupby(["model", "seed"])["robust_recall"].mean()
                  .groupby("model").agg(pm).reindex(order))
    rr.columns = [f"{c} (held out)" if c in HELD_OUT else c for c in rr.columns]

    asr = (adv.groupby(["model", "attack"])["asr_conditional"].agg(pm)
           .unstack("attack").reindex(order)[atts])
    asr["mean"] = (adv.groupby(["model", "seed"])["asr_conditional"].mean()
                   .groupby("model").agg(pm).reindex(order))
    asr.columns = [f"{c} (held out)" if c in HELD_OUT else c for c in asr.columns]

    flip = (adv.groupby(["model", "seed"])["benign_flip"].mean()
            .groupby("model").agg(pm).reindex(order).rename("Mean benign flip rate"))
    ncond = (adv.groupby("model")["n_conditional"].mean().reindex(order)
             .round(1).rename("mean n (conditional ASR)"))

    md = []
    md.append("### Clean test performance (%), mean ± std over 3 seeds\n")
    md.append(clean_tbl.to_markdown() + "\n")
    md.append("\n### Robust recall (%) — injections still caught after perturbation. "
              "Higher is better; same denominator for every model.\n")
    md.append(rr.to_markdown() + "\n")
    md.append("\n### Conditional ASR (%) — evasion among injections each model caught "
              "on clean text. Lower is better; denominator differs per model.\n")
    md.append(asr.to_markdown() + "\n")
    md.append("\n### Benign flip rate (%) and conditional-ASR denominators\n")
    md.append(pd.concat([flip, ncond], axis=1).to_markdown() + "\n")
    text = "\n".join(md)
    open(f"{out}/tables.md", "w").write(text)
    print(text)
    return clean, adv, order


if __name__ == "__main__":
    main()
