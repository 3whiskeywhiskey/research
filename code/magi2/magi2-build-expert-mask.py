#!/usr/bin/env python3
"""Build expert keep-masks for MAGI2_EXPERT_MASK from route_stats dumps.

  python magi2-build-expert-mask.py --inputs <route_stats dir> ... --out <dir> --keep 128 96 --rules modality global

Rules (experts per head are ranked within each (layer, head) and the top E' kept, so every
head keeps the same count, which is what the grouped-GEMM kernel wants):
  global    rank by total selections over all tokens.
  modality  rank by max over modalities of the expert's share of that modality's selections,
            so audio's and text's cores survive although they are ~1 % of the tokens.
Writes <out>/<rule>-<E'>.pt holding {"keep": bool[L, H, E], "rule", "keep_per_head", "coverage"}.
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import torch


def load_total_mod(dirs):
    total = None
    for d in dirs:
        by_sample = defaultdict(dict)
        for f in sorted(glob.glob(os.path.join(d, "sample_*_rank*.pt"))):
            m = re.match(r"(sample_\d+)_rank(\d+)\.pt", os.path.basename(f))
            by_sample[m.group(1)][int(m.group(2))] = f
        for name, ranks in sorted(by_sample.items()):
            parts = [torch.load(ranks[r], map_location="cpu") for r in sorted(ranks)]
            c = torch.cat([p["counts"].to(torch.int64) for p in parts], dim=3)  # [L, S, 8, H, E]
            L, S, C, H, E = c.shape
            cm = c.view(L, S, 2, 4, H, E)[:, :, :, :3].sum(dim=(1, 2))  # [L, 3, H, E]
            total = cm if total is None else total + cm
            print(f"loaded {d}/{name}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", nargs="+", type=int, default=[128, 96])
    ap.add_argument("--rules", nargs="+", default=["modality", "global"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tm = load_total_mod(args.inputs).float()  # [L, 3, H, E]
    L, _, H, E = tm.shape
    tot = tm.sum(1)  # [L, H, E]
    share = tm / tm.sum(-1, keepdim=True).clamp_min(1)  # share within (layer, mod, head)
    scores = {"global": tot, "modality": share.amax(1)}
    for rule in args.rules:
        sc = scores[rule]
        for k in args.keep:
            idx = torch.topk(sc, k, dim=-1).indices
            keep = torch.zeros(L, H, E, dtype=torch.bool).scatter_(-1, idx, True)
            cov = {m: (tm[:, i][keep].sum() / tm[:, i].sum()).item() for i, m in enumerate(("video", "audio", "text"))}
            cov["all"] = (tot[keep].sum() / tot.sum()).item()
            path = os.path.join(args.out, f"{rule}-{k}.pt")
            torch.save({"keep": keep, "rule": rule, "keep_per_head": k, "coverage": cov}, path)
            print(f"{path}: keep {k}/{E} per head; coverage of corpus selections "
                  + ", ".join(f"{m} {v:.3f}" for m, v in cov.items()))


if __name__ == "__main__":
    main()
