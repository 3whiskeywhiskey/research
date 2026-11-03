#!/usr/bin/env python3
"""Analyse MAGI-2 Multi-Head MoE routing statistics written by inference/utils/route_stats.py.

Usage (inside the magi image, or anywhere with torch + matplotlib):
  python magi2-route-analysis.py --inputs <route_stats dir> [<dir> ...] --out <dir> \
      [--grid T,H,W] [--moe-layer-offset 2]

Each input dir holds sample_NNN_rank<r>.pt files; ranks are merged along the head axis
(rank r holds heads [r*Hl, (r+1)*Hl)). Writes PNG figures, summary.json and summary.md.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

MOD_NAMES = ["video", "audio", "text"]


def load_samples(dirs):
    """-> dict name -> dict(counts [L,S,8,H,E] int64, mass [L,S,8,H,E] f32, margin [L,S,64], bias [L,H,E], raw, meta)."""
    samples = {}
    for d in dirs:
        tag = os.path.basename(os.path.normpath(d))
        if tag == "route_stats":  # <run>/route_stats: name samples after the run
            tag = os.path.basename(os.path.dirname(os.path.normpath(d)))
        by_sample = defaultdict(dict)
        for f in sorted(glob.glob(os.path.join(d, "sample_*_rank*.pt"))):
            m = re.match(r"(sample_\d+)_rank(\d+)\.pt", os.path.basename(f))
            by_sample[m.group(1)][int(m.group(2))] = f
        for name, ranks in sorted(by_sample.items()):
            parts = [torch.load(ranks[r], map_location="cpu") for r in sorted(ranks)]
            counts = torch.cat([p["counts"].to(torch.int64) for p in parts], dim=3)
            mass = torch.cat([p["mass"] for p in parts], dim=3)
            margin = sum(p["margin_hist"] for p in parts)
            Hl, E = parts[0]["local_heads"], parts[0]["n_experts"]
            layers = parts[0]["layers"]
            bias = torch.stack(
                [torch.cat([p["expert_bias"][l].view(Hl, E) for p in parts], dim=0) for l in layers]
            )
            raw = {}
            for key, v in parts[0]["raw_topk"].items():
                raw[key] = torch.cat([p["raw_topk"][key] for p in parts], dim=0)  # [H, S, K]
            samples[f"{tag}/{name}"] = dict(
                counts=counts, mass=mass, margin=margin, bias=bias, raw=raw,
                labels=parts[0]["labels"], pos=parts[0]["pos"], layers=layers,
                n_steps=parts[0]["n_steps"], margin_edges=parts[0]["margin_bin_edges_log10"],
            )
            print(f"loaded {tag}/{name}: counts {tuple(counts.shape)}, steps {parts[0]['n_steps']}, raw {list(raw)}")
    return samples


def jaccard(a, b):
    a = a.bool(); b = b.bool()
    return ((a & b).sum().float() / (a | b).sum().float().clamp_min(1)).item()


def cosine(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12).item()


def eff_experts(util):
    """Effective number of experts (exp of entropy) along the last axis."""
    p = util.float() / util.float().sum(-1, keepdim=True).clamp_min(1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    return torch.exp(ent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="32,32,56", help="video latent grid T,H,W (tokens are t-major)")
    ap.add_argument("--moe-layer-offset", type=int, default=2, help="moe_key 0 is transformer layer 2")
    ap.add_argument("--step-bins", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = load_samples(args.inputs)
    names = list(samples)
    first = samples[names[0]]
    L, S_steps, C, H, E = first["counts"].shape
    layer_ids = [l + args.moe_layer_offset for l in first["layers"]]
    summary = {"samples": names, "n_layers": L, "heads": H, "experts": E, "layers": layer_ids}

    # ---- utilisation tensors --------------------------------------------------------------
    # per sample: util_mod [L, 3, H, E] (steps and cfg halves summed), util_cfg [L, 2, H, E]
    util_mod = {}
    util_cfg = {}
    util_step = {}
    for n, s in samples.items():
        c = s["counts"]
        cm = c.view(L, c.shape[1], 2, 4, H, E)
        util_mod[n] = cm[:, :, :, :3].sum(dim=(1, 2))  # [L, 3, H, E]
        util_cfg[n] = cm[:, :, :, :3].sum(dim=(1, 3))  # [L, 2, H, E]
        util_step[n] = cm[:, :, :, :3].sum(dim=(2, 3))  # [L, steps, H, E]
    total = sum(u.sum(1) for u in util_mod.values())  # [L, H, E]
    total_mod = sum(util_mod.values())  # [L, 3, H, E]
    n_sel_layer = total.sum(dim=(1, 2))  # selections per layer

    # ---- dead / near-dead ------------------------------------------------------------------
    dead_total = (total == 0)  # dead across the whole corpus (intersection of per-sample dead)
    dead_any = torch.stack([(u.sum(1) == 0) for u in util_mod.values()]).any(0)  # dead in >=1 sample
    share = total.float() / n_sel_layer.view(L, 1, 1).float()
    near = {thr: (share < thr) for thr in (1e-5, 1e-4, 1e-3)}
    summary["dead_pairs_per_layer_corpus"] = dead_total.sum(dim=(1, 2)).tolist()
    summary["dead_pairs_per_layer_any_sample"] = dead_any.sum(dim=(1, 2)).tolist()
    summary["near_dead_pairs_per_layer"] = {str(t): v.sum(dim=(1, 2)).tolist() for t, v in near.items()}
    summary["dead_pairs_total"] = int(dead_total.sum())
    summary["pairs_total"] = int(L * H * E)
    summary["dead_per_sample"] = {n: int((u.sum(1) == 0).sum()) for n, u in util_mod.items()}
    # per-head: how many experts per head are dead (uniform-E' pruning needs the per-head max)
    summary["dead_per_head_min_over_heads_per_layer"] = dead_total.sum(-1).min(-1).values.tolist()
    summary["dead_per_head_max_over_heads_per_layer"] = dead_total.sum(-1).max(-1).values.tolist()

    # ---- modality overlap -------------------------------------------------------------------
    jac = {k: [] for k in ("video_audio", "video_text", "audio_text")}
    jac90 = {k: [] for k in jac}
    active_per_mod = {m: [] for m in MOD_NAMES}
    for l in range(L):
        act = [(total_mod[l, m] > 0) for m in range(3)]
        for m in range(3):
            active_per_mod[MOD_NAMES[m]].append(int(act[m].sum()))
        # sets covering 90 % of each modality's selection mass
        top = []
        for m in range(3):
            u = total_mod[l, m].flatten().float()
            order = torch.argsort(u, descending=True)
            cum = torch.cumsum(u[order], 0) / u.sum().clamp_min(1)
            k = int((cum < 0.9).sum()) + 1
            mask = torch.zeros_like(u, dtype=torch.bool); mask[order[:k]] = True
            top.append(mask)
        for key, (a, b) in zip(jac, ((0, 1), (0, 2), (1, 2))):
            jac[key].append(jaccard(act[a], act[b]))
            jac90[key].append(jaccard(top[a], top[b]))
    summary["active_pairs_per_mod_per_layer"] = active_per_mod
    summary["jaccard_active_sets"] = jac
    summary["jaccard_90pct_mass_sets"] = jac90
    # audio-exclusive experts: used by audio, never by video
    audio_only = ((total_mod[:, 1] > 0) & (total_mod[:, 0] == 0)).sum(dim=(1, 2)).tolist()
    text_only = ((total_mod[:, 2] > 0) & (total_mod[:, 0] == 0)).sum(dim=(1, 2)).tolist()
    summary["audio_only_pairs_per_layer"] = audio_only
    summary["text_only_pairs_per_layer"] = text_only

    # ---- timestep drift ----------------------------------------------------------------------
    steps = first["n_steps"]
    nb = min(args.step_bins, steps)
    edges = np.linspace(0, steps, nb + 1).astype(int)
    us = sum(util_step.values())  # [L, steps, H, E]
    binned = torch.stack([us[:, edges[i]:edges[i + 1]].sum(1) for i in range(nb)], dim=1)  # [L, nb, H, E]
    drift_first = [[cosine(binned[l, 0], binned[l, i]) for i in range(nb)] for l in range(L)]
    drift_prev = [[cosine(binned[l, i - 1], binned[l, i]) for i in range(1, nb)] for l in range(L)]
    summary["step_bins"] = edges.tolist()
    summary["drift_cos_vs_first_bin"] = drift_first
    summary["drift_cos_vs_prev_bin"] = drift_prev
    # expert sets that only appear in the high-noise bins vs only low-noise
    hi = binned[:, : max(1, nb // 3)].sum(1) > 0
    lo = binned[:, -max(1, nb // 3):].sum(1) > 0
    summary["pairs_only_high_noise_per_layer"] = (hi & ~lo).sum(dim=(1, 2)).tolist()
    summary["pairs_only_low_noise_per_layer"] = (lo & ~hi).sum(dim=(1, 2)).tolist()

    # ---- cond vs uncond ------------------------------------------------------------------------
    uc = sum(util_cfg.values())  # [L, 2, H, E]
    summary["cfg_cos_cond_uncond"] = [cosine(uc[l, 0], uc[l, 1]) for l in range(L)]
    summary["cfg_jaccard_active"] = [jaccard(uc[l, 0] > 0, uc[l, 1] > 0) for l in range(L)]

    # ---- concentration ---------------------------------------------------------------------------
    eff = eff_experts(total)  # [L, H]
    eff_mod = eff_experts(total_mod)  # [L, 3, H]
    summary["effective_experts_per_head_mean"] = eff.mean(1).tolist()
    summary["effective_experts_per_mod_mean"] = {MOD_NAMES[m]: eff_mod[:, m].mean(1).tolist() for m in range(3)}

    # ---- coverage by the top-k experts per head (uniform-E' pruning) ---------------------------------
    cov = {}
    for m_name, u in (("all", total), ("video", total_mod[:, 0]), ("audio", total_mod[:, 1]), ("text", total_mod[:, 2])):
        srt = torch.sort(u.float(), dim=-1, descending=True).values  # [L, H, E]
        cum = torch.cumsum(srt, -1) / srt.sum(-1, keepdim=True).clamp_min(1)
        cov[m_name] = {str(k): cum[:, :, k - 1].mean().item() for k in (32, 64, 96, 128, 160, 192, 224)}
    # the same, but ranking experts by ALL-token utilisation and measuring each modality's coverage
    order_all = torch.argsort(total.float(), dim=-1, descending=True)
    cov_by_all = {}
    for m in range(3):
        g = torch.gather(total_mod[:, m].float(), -1, order_all)
        cum = torch.cumsum(g, -1) / g.sum(-1, keepdim=True).clamp_min(1)
        cov_by_all[MOD_NAMES[m]] = {str(k): cum[:, :, k - 1].mean().item() for k in (64, 128, 192)}
    summary["coverage_topk_per_head_own_ranking"] = cov
    summary["coverage_topk_per_head_ranked_by_all_tokens"] = cov_by_all

    # ---- per-sample dead-set generalisation --------------------------------------------------------
    per_sample_dead = {n: (u.sum(1) == 0) for n, u in util_mod.items()}
    inter = torch.stack(list(per_sample_dead.values())).all(0)
    union = torch.stack(list(per_sample_dead.values())).any(0)
    summary["dead_intersection_total"] = int(inter.sum())
    summary["dead_union_total"] = int(union.sum())
    # leave-one-out: pairs dead on all other samples but alive on this one
    loo = {}
    for n in names:
        others = torch.stack([v for k, v in per_sample_dead.items() if k != n]).all(0) if len(names) > 1 else inter
        loo[n] = int((others & ~per_sample_dead[n]).sum())
    summary["dead_on_all_others_but_alive_here"] = loo

    # ---- margins ----------------------------------------------------------------------------------
    mh = sum(s["margin"] for s in samples.values()).float()  # [L, steps, 64]
    edges_log = np.array(first["margin_edges"])
    cdf = torch.cumsum(mh.sum(1), dim=-1) / mh.sum(dim=(1, 2)).clamp_min(1).view(L, 1)
    def frac_below(thr):
        b = int(np.searchsorted(edges_log, np.log10(thr)))
        return cdf[:, max(b - 1, 0)].tolist()
    summary["margin_frac_below_1e-3"] = frac_below(1e-3)
    summary["margin_frac_below_1e-2"] = frac_below(1e-2)
    summary["margin_frac_below_1e-1"] = frac_below(1e-1)

    # ---- bias ------------------------------------------------------------------------------------
    bias = first["bias"]  # [L, H, E]
    summary["bias_mean_abs_per_layer"] = bias.abs().mean(dim=(1, 2)).tolist()
    summary["bias_min_per_layer"] = bias.amin(dim=(1, 2)).tolist()
    summary["bias_max_per_layer"] = bias.amax(dim=(1, 2)).tolist()
    bias_util_corr = []
    for l in range(L):
        b = bias[l].flatten(); u = torch.log1p(total[l].flatten().float())
        bias_util_corr.append(float(np.corrcoef(b.numpy(), u.numpy())[0, 1]))
    summary["corr_bias_vs_log_util_per_layer"] = bias_util_corr
    summary["dead_pairs_mean_bias"] = float(bias[dead_total].mean()) if dead_total.any() else None
    summary["alive_pairs_mean_bias"] = float(bias[~dead_total].mean())

    # ---- figures ----------------------------------------------------------------------------------
    def heat(mat, title, fname, ylabel="MoE layer", xlabel="expert", log=True):
        fig, ax = plt.subplots(figsize=(14, 7))
        m = mat.float().numpy()
        if log:
            m = np.log10(m + 1)
        im = ax.imshow(m, aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if m.shape[0] == len(layer_ids):
            ax.set_yticks(range(len(layer_ids))); ax.set_yticklabels(layer_ids, fontsize=6)
        fig.colorbar(im, ax=ax, label="log10(selections+1)" if log else "")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, fname), dpi=130); plt.close(fig)

    for m in range(3):
        heat(total_mod[:, m].sum(1), f"{MOD_NAMES[m]} tokens: selections per (layer, expert), heads summed",
             f"util_{MOD_NAMES[m]}_layer_x_expert.png")
    heat(total.permute(0, 1, 2).reshape(L, H * E), "all tokens: selections per (layer, head*256+expert)",
         "util_all_layer_x_headexpert.png", xlabel="head*256 + expert")
    # per-head view of a middle layer
    lm = L // 2
    heat(total[lm], f"layer {layer_ids[lm]}: selections per (head, expert)", f"util_layer{layer_ids[lm]}_head_x_expert.png",
         ylabel="head")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(layer_ids, summary["dead_pairs_per_layer_corpus"], label="dead over corpus")
    ax.plot(layer_ids, summary["near_dead_pairs_per_layer"]["0.0001"], "o-", color="C1", label="share < 1e-4")
    ax.plot(layer_ids, summary["near_dead_pairs_per_layer"]["0.001"], "s-", color="C3", label="share < 1e-3")
    ax.set_xlabel("layer"); ax.set_ylabel(f"(head, expert) pairs of {H * E}"); ax.legend(); ax.set_title("dead and near-dead experts per layer")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "dead_per_layer.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    for k, v in jac.items():
        ax.plot(layer_ids, v, "o-", label=f"{k} (active sets)")
    for k, v in jac90.items():
        ax.plot(layer_ids, v, "--", label=f"{k} (90% mass sets)")
    ax.set_ylim(0, 1); ax.set_xlabel("layer"); ax.set_ylabel("Jaccard"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("expert-set overlap between modalities, per layer")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "modality_overlap.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(np.array(drift_first), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(L)); ax.set_yticklabels(layer_ids, fontsize=6); ax.set_xlabel(f"step bin ({steps} steps / {nb})"); ax.set_ylabel("layer")
    ax.set_title("cosine(utilisation at bin, utilisation at first bin)"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "timestep_drift.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(layer_ids, summary["cfg_cos_cond_uncond"], "o-", label="cosine cond vs uncond")
    ax.plot(layer_ids, summary["cfg_jaccard_active"], "s-", label="Jaccard active sets")
    ax.set_ylim(0, 1.02); ax.legend(); ax.set_xlabel("layer"); ax.set_title("cond vs uncond routing")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "cfg_cond_vs_uncond.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(layer_ids, summary["effective_experts_per_head_mean"], "o-", label="all tokens")
    for m in range(3):
        ax.plot(layer_ids, summary["effective_experts_per_mod_mean"][MOD_NAMES[m]], "--", label=MOD_NAMES[m])
    ax.set_xlabel("layer"); ax.set_ylabel("effective experts per head (of 256)"); ax.legend(); ax.set_title("routing concentration")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "effective_experts.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    for l in (0, L // 4, L // 2, 3 * L // 4, L - 1):
        ax.plot(edges_log[1:], cdf[l].numpy(), label=f"layer {layer_ids[l]}")
    ax.set_xlim(-6, 0.5); ax.set_xlabel("log10 selection margin (6th - 7th score)"); ax.set_ylabel("CDF"); ax.legend()
    ax.set_title("selection-margin CDF (flip-proneness)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "margin_cdf.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(layer_ids, summary["bias_mean_abs_per_layer"], "o-", label="mean |bias|")
    ax.plot(layer_ids, summary["bias_min_per_layer"], "v-", label="min bias")
    ax.plot(layer_ids, summary["corr_bias_vs_log_util_per_layer"], "s--", label="corr(bias, log util)")
    ax.set_xlabel("layer"); ax.legend(); ax.set_title("checkpoint aux-free expert bias")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "expert_bias.png"), dpi=130); plt.close(fig)

    # ---- spatial maps from the raw dump (first sample, middle layer) -----------------------------------
    T, Hg, Wg = (int(x) for x in args.grid.split(","))
    s0 = samples[names[0]]
    if s0["raw"]:
        key = sorted(s0["raw"], key=lambda k: (int(k.split("_")[0]) != lm, k))[0]
        l_raw, step_raw = (int(x) for x in key.split("_"))
        raw = s0["raw"][key].to(torch.int64)  # [H, S, K]
        labels = s0["labels"]; pos = s0["pos"].to(torch.int64)
        vid = (labels == 0).nonzero().flatten()  # cond video tokens
        if vid.numel() == T * Hg * Wg:
            head = 0
            sel = raw[head, vid]  # [Nv, K]
            p = pos[vid]  # global index, video tokens of item 0 start at 0
            counts_e = torch.bincount(sel.flatten(), minlength=E)
            top_e = torch.topk(counts_e, 6).indices.tolist()
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for ax, e in zip(axes.flat, top_e):
                hit = (sel == e).any(1).float()
                grid = torch.zeros(T * Hg * Wg); grid[p] = hit
                ax.imshow(grid.view(T, Hg, Wg).mean(0).numpy(), cmap="magma", vmin=0, vmax=1)
                ax.set_title(f"layer {l_raw + args.moe_layer_offset} head {head} expert {e}: P(selected) per (y,x), t-avg")
                ax.axis("off")
            fig.suptitle(f"{names[0]} step {step_raw}: spatial footprint of the 6 most-used experts")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, "spatial_footprints.png"), dpi=110); plt.close(fig)
            # temporal profile
            fig, ax = plt.subplots(figsize=(12, 4))
            for e in top_e:
                hit = (sel == e).any(1).float(); grid = torch.zeros(T * Hg * Wg); grid[p] = hit
                ax.plot(grid.view(T, -1).mean(1).numpy(), label=f"expert {e}")
            ax.set_xlabel("latent frame"); ax.set_ylabel("P(selected)"); ax.legend(fontsize=7)
            ax.set_title(f"temporal profile, layer {l_raw + args.moe_layer_offset} head {head}")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, "temporal_profiles.png"), dpi=130); plt.close(fig)
            # co-activation: how many distinct experts does a token's 6 heads x 6 picks touch? (head-level diversity)
            summary["spatial_maps"] = {"layer": l_raw + args.moe_layer_offset, "step": step_raw, "head": head, "experts": top_e}
        else:
            print(f"grid {T}x{Hg}x{Wg} != cond video tokens {vid.numel()}; skipping spatial maps")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---- markdown summary ----------------------------------------------------------------------------
    md = []
    md.append(f"# MAGI-2 routing dissection: {len(names)} samples, {L} MoE layers, {H} heads x {E} experts, {steps} steps\n")
    md.append(f"Selections per layer per sample: {int(n_sel_layer[0] / len(names))} (tokens x heads x top-6).\n")
    md.append(f"- Dead (head, expert) pairs over the corpus: {summary['dead_pairs_total']} of {summary['pairs_total']} "
              f"({100 * summary['dead_pairs_total'] / summary['pairs_total']:.2f} %). Dead in at least one sample: "
              f"{summary['dead_union_total']}. Per-sample dead: min {min(summary['dead_per_sample'].values())}, "
              f"max {max(summary['dead_per_sample'].values())}.")
    md.append(f"- Near-dead (< 1e-4 of a layer's selections): {sum(summary['near_dead_pairs_per_layer']['0.0001'])}; "
              f"(< 1e-3): {sum(summary['near_dead_pairs_per_layer']['0.001'])}.")
    md.append(f"- Leave-one-out: pairs dead on every other sample but alive on this one: {summary['dead_on_all_others_but_alive_here']}")
    md.append(f"- Audio-only pairs (used by audio, never video), total over layers: {sum(audio_only)}; text-only: {sum(text_only)}.")
    md.append(f"- Jaccard video/audio active sets, mean over layers: {np.mean(jac['video_audio']):.3f} "
              f"(90 %-mass sets {np.mean(jac90['video_audio']):.3f}); video/text {np.mean(jac['video_text']):.3f} "
              f"({np.mean(jac90['video_text']):.3f}).")
    md.append(f"- Effective experts per head, mean over layers: all {np.mean(summary['effective_experts_per_head_mean']):.1f}, "
              + ", ".join(f"{m} {np.mean(summary['effective_experts_per_mod_mean'][m]):.1f}" for m in MOD_NAMES) + ".")
    md.append(f"- Timestep drift: cosine(last bin, first bin) mean over layers {np.mean([d[-1] for d in drift_first]):.3f}; "
              f"min {np.min([d[-1] for d in drift_first]):.3f}.")
    md.append(f"- cond vs uncond cosine mean {np.mean(summary['cfg_cos_cond_uncond']):.3f}, min {np.min(summary['cfg_cos_cond_uncond']):.3f}.")
    md.append(f"- Selection margin: fraction of picks with 6th-7th score gap < 1e-2: mean {np.mean(summary['margin_frac_below_1e-2']):.3f}; "
              f"< 1e-3: {np.mean(summary['margin_frac_below_1e-3']):.3f}.")
    md.append(f"- Checkpoint bias: mean |bias| {np.mean(summary['bias_mean_abs_per_layer']):.3f}, min {np.min(summary['bias_min_per_layer']):.3f}, "
              f"corr(bias, log util) mean {np.mean(bias_util_corr):.3f}; dead pairs mean bias {summary['dead_pairs_mean_bias']}, "
              f"alive {summary['alive_pairs_mean_bias']:.3f}.")
    md.append("- Coverage of selections by the top-k experts per head (own ranking), mean over layers and heads: "
              + "; ".join(f"{m}: " + ", ".join(f"k={k} {v:.3f}" for k, v in c.items() if k in ("64", "128", "192")) for m, c in cov.items()) + ".")
    md.append("- Coverage when experts are ranked by all-token utilisation: "
              + "; ".join(f"{m}: " + ", ".join(f"k={k} {v:.3f}" for k, v in c.items()) for m, c in cov_by_all.items()) + ".")
    md.append("\n| layer | dead | <1e-4 | audio-only | J(v,a) | J(v,t) | eff/head | drift(last) | cfg cos | margin<1e-2 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, l in enumerate(layer_ids):
        md.append(f"| {l} | {summary['dead_pairs_per_layer_corpus'][i]} | {summary['near_dead_pairs_per_layer']['0.0001'][i]} | "
                  f"{audio_only[i]} | {jac['video_audio'][i]:.2f} | {jac['video_text'][i]:.2f} | "
                  f"{summary['effective_experts_per_head_mean'][i]:.0f} | {drift_first[i][-1]:.2f} | "
                  f"{summary['cfg_cos_cond_uncond'][i]:.2f} | {summary['margin_frac_below_1e-2'][i]:.3f} |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
