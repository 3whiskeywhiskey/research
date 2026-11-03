#!/usr/bin/env python3
"""Teacher-forced one-step error of MAGI-2 runs vs the teacher (baseline) run.

  python magi2-teacher-score.py --base <teacher run dir> --runs name=<run dir> ... --out <dir>

Every run dumped latents after every step (MAGI2_TRAJ_EVERY=1) and, except the teacher, was
fed the teacher's post-step latents before each step (MAGI2_TEACHER_TRAJ). For step k the
candidate's post-step latent c[k] and the teacher's t[k] share the input t[k-1], so
c[k]-t[k] is this step's prediction error alone. Reported per step: the relative update
error ||c[k]-t[k]|| / ||t[k]-t[k-1]|| and the latent SNR 10*log10(var(t[k]) / mse), for video
and audio latents; then means over steps and samples, and per-step curves.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch


def load_run(d, sample):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "traj", f"sample_{sample:03d}", "step_*.pt"))):
        x = torch.load(f, map_location="cpu")
        out[int(x["step"])] = (x["latent"].float(), x["audio_latent"].float())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = dict(r.split("=", 1) for r in args.runs)
    samples = sorted(int(os.path.basename(p)[7:10]) for p in glob.glob(os.path.join(args.base, "traj", "sample_*")))
    teacher = {s: load_run(args.base, s) for s in samples}
    results = {}
    for name, d in runs.items():
        per = {}
        for s in samples:
            cand = load_run(d, s)
            steps = sorted(k for k in cand if k in teacher[s] and (k - 1) in teacher[s])
            rel_v, rel_a, snr_v, snr_a = [], [], [], []
            for k in steps:
                for i, (rel, snr) in enumerate(((rel_v, snr_v), (rel_a, snr_a))):
                    t, tp, c = teacher[s][k][i], teacher[s][k - 1][i], cand[k][i]
                    err = (c - t).pow(2).mean().item()
                    upd = (t - tp).pow(2).mean().item()
                    rel.append(float(np.sqrt(err / max(upd, 1e-20))))
                    snr.append(float(10 * np.log10(t.var().item() / max(err, 1e-20))))
            per[s] = {"steps": steps, "rel_update_err_video": rel_v, "rel_update_err_audio": rel_a,
                      "snr_video_db": snr_v, "snr_audio_db": snr_a}
        agg = {}
        for key in ("rel_update_err_video", "rel_update_err_audio", "snr_video_db", "snr_audio_db"):
            vals = [np.mean(p[key]) for p in per.values() if p[key]]
            agg[key + "_mean"] = float(np.mean(vals))
            med = [np.median(p[key]) for p in per.values() if p[key]]
            agg[key + "_median"] = float(np.mean(med))
        results[name] = {"per_sample": per, "mean": agg}
        print(name, json.dumps({k: round(v, 3) for k, v in agg.items()}))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"base": args.base, "samples": samples, "runs": results}, f, indent=1)

    md = ["# Teacher-forced one-step error vs the dense model (mean over steps and held-out samples)\n",
          "| run | video rel. update error | audio rel. update error | video SNR | audio SNR |", "|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["mean"]
        md.append(f"| {name} | {a['rel_update_err_video_mean']:.3f} (median {a['rel_update_err_video_median']:.3f}) | "
                  f"{a['rel_update_err_audio_mean']:.3f} ({a['rel_update_err_audio_median']:.3f}) | "
                  f"{a['snr_video_db_mean']:.1f} dB | {a['snr_audio_db_mean']:.1f} dB |")
    first = next(iter(results.values()))["per_sample"][samples[0]]["steps"]
    pick = [k for k in first if k % 10 == 0]
    md.append(f"\nVideo relative update error by step, sample 0 (steps {pick[0]}..{pick[-1]}):\n")
    md.append("| run | " + " | ".join(str(k) for k in pick) + " |")
    md.append("|---|" + "---|" * len(pick))
    for name, r in results.items():
        p = r["per_sample"][samples[0]]
        idx = {k: i for i, k in enumerate(p["steps"])}
        md.append(f"| {name} | " + " | ".join(f"{p['rel_update_err_video'][idx[k]]:.3f}" for k in pick if k in idx) + " |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, r in results.items():
        v = np.mean([r["per_sample"][s]["rel_update_err_video"] for s in samples], axis=0)
        a = np.mean([r["per_sample"][s]["rel_update_err_audio"] for s in samples], axis=0)
        st = r["per_sample"][samples[0]]["steps"]
        axes[0].plot(st, v, label=name); axes[1].plot(st, a, label=name)
    for ax, t in zip(axes, ("video", "audio")):
        ax.set_yscale("log"); ax.set_xlabel("denoise step"); ax.set_ylabel("||c[k]-t[k]|| / ||t[k]-t[k-1]||")
        ax.set_title(f"teacher-forced one-step error, {t} latent (mean of samples)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "one_step_error.png"), dpi=130)


if __name__ == "__main__":
    main()
