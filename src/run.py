"""Run the full study: clean performance + robustness under six perturbations.

Model tiers
  TF-IDF + LR        classical baseline -- is a language model needed at all?
  Zero-shot SLM/LLM  no fine-tuning
  LoRA SLM/LLM       fine-tuned on clean data
  + AdvTrain         fine-tuned on clean data plus one perturbed copy of every
                     example (both classes, so "perturbed" is not a shortcut
                     for "injection"); `homoglyph` is held out of training
"""
import argparse
import gc
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from attacks import ATTACKS, HELD_OUT, TRAIN_ATTACKS
from metrics import attack_metrics, clean_metrics
import models as M

SLM = "Qwen/Qwen2.5-0.5B-Instruct"
LLM = "Qwen/Qwen2.5-7B-Instruct"
DATASET = "deepset/prompt-injections"
SEEDS = [42, 43, 44]


def free():
    """Reclaim GPU memory.

    Deleting names inside a helper does NOT release the caller's references,
    and a predict closure keeps its model alive anyway -- so every model is
    built inside `run_one` and released when that frame returns, and this only
    has to collect and empty the cache.
    """
    gc.collect()
    torch.cuda.empty_cache()


def augment(texts, labels, seed):
    """One perturbed copy of every example, drawn from the training attacks."""
    rng = random.Random(seed + 999)
    names = list(TRAIN_ATTACKS)
    aug_t, aug_y = list(texts), list(labels)
    for t, y in zip(texts, labels):
        aug_t.append(TRAIN_ATTACKS[rng.choice(names)](t, rng))
        aug_y.append(y)
    return aug_t, aug_y


def evaluate(name, predict, test_texts, test_labels, seed, clean_rows, adv_rows):
    y = np.asarray(test_labels)
    clean_pred = np.asarray(predict(test_texts))
    row = clean_metrics(y, clean_pred)
    row.update(model=name, seed=seed)
    clean_rows.append(row)

    inj_pos = np.where(y == 1)[0]                     # every injection
    ben_ok = np.where((y == 0) & (clean_pred == 0))[0]  # safe prompts got right
    detected_mask = clean_pred[inj_pos] == 1

    for k, (att, fn) in enumerate(ATTACKS.items()):
        rng = random.Random(seed * 100 + k)
        adv_inj = [fn(test_texts[i], rng) for i in inj_pos]
        adv_ben = [fn(test_texts[i], rng) for i in ben_ok]
        preds = np.asarray(predict(adv_inj + adv_ben))
        m = attack_metrics(preds[:len(adv_inj)], detected_mask,
                           preds[len(adv_inj):], seed=seed * 10 + k)
        m.update(model=name, seed=seed, attack=att, held_out=att in HELD_OUT)
        adv_rows.append(m)

    mean_rr = np.mean([r["robust_recall"] for r in adv_rows
                       if r["model"] == name and r["seed"] == seed])
    print(f"[{name} | seed {seed}] clean F1={row['f1']:.3f}  "
          f"mean robust recall={mean_rr:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+",
                    default=["constant", "tfidf", "zs_slm", "zs_llm", "lora_slm",
                             "lora_slm_adv", "lora_llm", "lora_llm_adv"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--adapter_out", default="adapters")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(DATASET)
    train_texts, train_labels = list(ds["train"]["text"]), list(ds["train"]["label"])
    test_texts, test_labels = list(ds["test"]["text"]), list(ds["test"]["label"])
    print(f"train n={len(train_labels)} (inj={sum(train_labels)})  "
          f"test n={len(test_labels)} (inj={sum(test_labels)})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    clean_rows, adv_rows, meta = [], [], {}
    done = set()
    if args.resume and os.path.exists(f"{args.out}/clean_raw.csv"):
        prev_c = pd.read_csv(f"{args.out}/clean_raw.csv")
        prev_a = pd.read_csv(f"{args.out}/adversarial_raw.csv")
        clean_rows.extend(prev_c.to_dict("records"))
        adv_rows.extend(prev_a.to_dict("records"))
        done = {(r["model"], int(r["seed"])) for _, r in prev_c.iterrows()}
        if os.path.exists(f"{args.out}/meta.json"):
            meta.update(json.load(open(f"{args.out}/meta.json")))
        print(f"resuming: {len(done)} (model, seed) cells already done", flush=True)

    def dump():
        pd.DataFrame(clean_rows).to_csv(f"{args.out}/clean_raw.csv", index=False)
        pd.DataFrame(adv_rows).to_csv(f"{args.out}/adversarial_raw.csv", index=False)
        json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)

    specs = [
        ("constant", "Always-INJECTION", None, False),
        ("tfidf", "TF-IDF + LR", None, False),
        ("zs_slm", "Zero-shot SLM (0.5B)", SLM, False),
        ("zs_llm", "Zero-shot LLM (7B)", LLM, False),
        ("lora_slm", "LoRA SLM (0.5B)", SLM, False),
        ("lora_slm_adv", "LoRA SLM + AdvTrain", SLM, True),
        ("lora_llm", "LoRA LLM (7B)", LLM, False),
        ("lora_llm_adv", "LoRA LLM + AdvTrain", LLM, True),
    ]

    def run_one(key, name, model_id, adv, seed):
        """Build, evaluate and drop one model. Everything stays local to this
        frame so the weights are released as soon as it returns."""
        if key == "constant":
            evaluate(name, M.constant_predictor(1), test_texts, test_labels,
                     seed, clean_rows, adv_rows)
        elif key == "tfidf":
            evaluate(name, M.tfidf_predictor(train_texts, train_labels, seed),
                     test_texts, test_labels, seed, clean_rows, adv_rows)
        elif key.startswith("zs_"):
            predict, _model = M.zeroshot_predictor(model_id)
            evaluate(name, predict, test_texts, test_labels, seed, clean_rows, adv_rows)
        else:
            tx, ty = augment(train_texts, train_labels, seed) if adv \
                else (train_texts, train_labels)
            grad_ckpt = "7B" in name
            predict, model, tok, info = M.train_lora(
                model_id, tx, ty, seed, out_dir=f"outputs/{key}_{seed}",
                batch_size=4 if grad_ckpt else 8, grad_checkpoint=grad_ckpt)
            meta[name] = info
            evaluate(name, predict, test_texts, test_labels, seed, clean_rows, adv_rows)
            if seed == SEEDS[-1]:
                d = os.path.join(args.adapter_out, key)
                os.makedirs(d, exist_ok=True)
                model.save_pretrained(d)
                tok.save_pretrained(d)

    for key, name, model_id, adv in specs:
        if key not in args.tiers:
            continue
        print(f"\n===== {name} =====", flush=True)
        for seed in SEEDS:
            if (name, seed) in done:
                print(f"  skip {name} seed {seed} (already in results)", flush=True)
                continue
            run_one(key, name, model_id, adv, seed)
            free()
            dump()

    dump()
    print(f"\nwrote {args.out}/clean_raw.csv and {args.out}/adversarial_raw.csv", flush=True)


if __name__ == "__main__":
    main()
