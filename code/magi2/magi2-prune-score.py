#!/usr/bin/env python3
"""Score pruned / perturbed MAGI-2 runs against a baseline run (Phase B of the dissection).

  python magi2-prune-score.py --base <run dir> --runs name=<run dir> ... --out <dir>

Each run dir is the launcher's output/<RUN>: sample_NNN.mp4 plus traj/sample_NNN/step_NNN.pt
(MAGI2_TRAJ_DUMP). Per sample and run: latent SNR (dB) of the video and audio latents vs the
baseline at every dumped step, decoded-frame PSNR and SSIM (grey, global) vs the baseline
clip, and audio log-mel spectral distance (dB RMS). Writes summary.json, summary.md and a
quality-curve figure. Runs inside the magi image (torch, av, librosa, numpy).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch


def snr_db(ref, x):
    ref = ref.float(); x = x.float()
    mse = ((ref - x) ** 2).mean().item()
    return 10 * np.log10(ref.var().item() / max(mse, 1e-20)), mse


def load_traj(run_dir, sample):
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "traj", f"sample_{sample:03d}", "step_*.pt"))):
        d = torch.load(f, map_location="cpu")
        out[int(d["step"])] = (d["latent"], d["audio_latent"])
    return out


def read_video(path, max_frames=None):
    import av
    frames = []
    with av.open(path) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format="gray").astype(np.float32))
            if max_frames and len(frames) >= max_frames:
                break
    return np.stack(frames)


def read_audio(path, sr=16000):
    import av
    import librosa
    with av.open(path) as c:
        s = c.streams.audio[0]
        chunks = []
        for fr in c.decode(s):
            chunks.append(fr.to_ndarray().astype(np.float32))
        rate = s.rate
    a = np.concatenate(chunks, axis=-1)
    if a.ndim == 2:
        a = a.mean(0)
    if rate != sr:
        a = librosa.resample(a, orig_sr=rate, target_sr=sr)
    return a


def ssim_global(a, b):
    """Global SSIM per frame on grey frames (no window; good enough for a ranking metric)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    out = []
    for x, y in zip(a, b):
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mx) * (y - my)).mean()
        out.append(((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import librosa
    runs = dict(r.split("=", 1) for r in args.runs)
    samples = sorted(int(os.path.basename(p)[7:10]) for p in glob.glob(os.path.join(args.base, "sample_*.mp4")))
    base_traj = {s: load_traj(args.base, s) for s in samples}
    base_vid = {s: read_video(os.path.join(args.base, f"sample_{s:03d}.mp4")) for s in samples}
    base_aud = {s: read_audio(os.path.join(args.base, f"sample_{s:03d}.mp4")) for s in samples}
    base_mel = {s: librosa.power_to_db(librosa.feature.melspectrogram(y=base_aud[s], sr=16000, n_mels=64)) for s in samples}

    results = {}
    for name, d in runs.items():
        per = {}
        for s in samples:
            r = {}
            tr = load_traj(d, s)
            steps = sorted(set(tr) & set(base_traj[s]))
            r["steps"] = steps
            r["video_latent_snr_db"] = [snr_db(base_traj[s][k][0], tr[k][0])[0] for k in steps]
            r["audio_latent_snr_db"] = [snr_db(base_traj[s][k][1], tr[k][1])[0] for k in steps]
            mp4 = os.path.join(d, f"sample_{s:03d}.mp4")
            if os.path.exists(mp4):
                v = read_video(mp4)
                n = min(len(v), len(base_vid[s]))
                mse = ((v[:n] - base_vid[s][:n]) ** 2).mean(axis=(1, 2))
                r["frame_psnr_db"] = float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6))))
                r["frame_psnr_min_db"] = float(np.min(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6))))
                r["ssim"] = ssim_global(v[:n], base_vid[s][:n])
                a = read_audio(mp4)
                m = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=16000, n_mels=64))
                t = min(m.shape[1], base_mel[s].shape[1])
                r["audio_logmel_rms_db"] = float(np.sqrt(((m[:, :t] - base_mel[s][:, :t]) ** 2).mean()))
            per[s] = r
        agg = {}
        for key in ("frame_psnr_db", "frame_psnr_min_db", "ssim", "audio_logmel_rms_db"):
            vals = [p[key] for p in per.values() if key in p]
            agg[key] = float(np.mean(vals)) if vals else None
        agg["video_latent_snr_final_db"] = float(np.mean([p["video_latent_snr_db"][-1] for p in per.values() if p["video_latent_snr_db"]]))
        agg["audio_latent_snr_final_db"] = float(np.mean([p["audio_latent_snr_db"][-1] for p in per.values() if p["audio_latent_snr_db"]]))
        results[name] = {"per_sample": per, "mean": agg}
        print(name, json.dumps(agg))

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"base": args.base, "samples": samples, "runs": results}, f, indent=1)

    md = ["# Phase B scores vs baseline (mean over held-out samples)\n",
          "| run | video latent SNR (final) | audio latent SNR (final) | frame PSNR | frame PSNR min | SSIM | audio log-mel RMS (dB) |",
          "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a = r["mean"]
        md.append(f"| {name} | {a['video_latent_snr_final_db']:.1f} dB | {a['audio_latent_snr_final_db']:.1f} dB | "
                  f"{a['frame_psnr_db']:.1f} dB | {a['frame_psnr_min_db']:.1f} dB | {a['ssim']:.3f} | {a['audio_logmel_rms_db']:.1f} |")
    md.append("\nPer-step video latent SNR (dB), sample 0:\n")
    md.append("| run | " + " | ".join(str(k) for k in results[next(iter(results))]["per_sample"][samples[0]]["steps"]) + " |")
    md.append("|---|" + "---|" * len(results[next(iter(results))]["per_sample"][samples[0]]["steps"]))
    for name, r in results.items():
        md.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in r["per_sample"][samples[0]]["video_latent_snr_db"]) + " |")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, r in results.items():
        p = r["per_sample"][samples[0]]
        axes[0].plot(p["steps"], p["video_latent_snr_db"], "o-", label=name)
        axes[1].plot(p["steps"], p["audio_latent_snr_db"], "o-", label=name)
    axes[0].set_title("video latent SNR vs baseline, sample 0"); axes[0].set_xlabel("denoise step"); axes[0].set_ylabel("dB"); axes[0].legend(fontsize=8)
    axes[1].set_title("audio latent SNR vs baseline, sample 0"); axes[1].set_xlabel("denoise step"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "trajectory_snr.png"), dpi=130)


if __name__ == "__main__":
    main()
