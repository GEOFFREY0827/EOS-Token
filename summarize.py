#!/usr/bin/env python
"""Turn results_step1/summary.jsonl into tables and one figure per task."""
import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results_step1")
    args = ap.parse_args()
    path = os.path.join(args.out_dir, "summary.jsonl")
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    df = df.drop_duplicates(subset=["task", "model", "filler", "attn", "n_tail", "k", "steps", "eot"], keep="last")

    for task, g in df.groupby("task"):
        base = g[g.n_tail == 0]["acc"]
        base_acc = float(base.iloc[0]) if len(base) else float("nan")
        print(f"\n=== {task}  (baseline n_tail=0: {base_acc:.3f}) ===")

        full = g[(g.attn == "full") & (g.n_tail > 0)]
        if len(full):
            print("\nFiller x tail length (attn=full):")
            print(full.pivot_table(index="filler", columns="n_tail", values="acc").round(3).to_string())

        abl = g[g.attn != "full"]
        if len(abl):
            print("\nAttention-path ablations (accuracy; 'full' column for reference):")
            ref = full.set_index(["filler", "n_tail"])["acc"]
            t = abl.pivot_table(index=["filler", "n_tail"], columns="attn", values="acc")
            t.insert(0, "full", [ref.get(i, float("nan")) for i in t.index])
            t["baseline_n0"] = base_acc
            print(t.round(3).to_string())

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for fl, h in full.groupby("filler"):
                h = h.sort_values("n_tail")
                ax.plot([0] + list(h.n_tail), [base_acc] + list(h.acc), marker="o", label=fl)
            ax.set_xlabel("tail positions after the answer")
            ax.set_ylabel("accuracy")
            ax.set_title(task)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"{task}_filler_vs_tail.png"), dpi=150)
            plt.close(fig)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
