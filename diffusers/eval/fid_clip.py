"""FID and CLIP score of generated COCO images, aggregated over seeds.

Every setting is evaluated on the same number of images: the smallest image count found in any
`seed_*/lam*` directory being compared. Real statistics are computed only over the COCO image ids
that were generated.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cleanfid import fid
from cleanfid.fid import frechet_distance, get_files_features
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")


def parse_image_id(path):
    return int(path.stem.split("_")[-1])


def list_images(folder, n=None, target_ids=None):
    files = sorted({f for pattern in IMAGE_PATTERNS for f in Path(folder).glob(pattern)})
    if target_ids is not None:
        files = [f for f in files if _safe_id(f) in target_ids]
    return files[:n]


def _safe_id(path):
    try:
        return parse_image_id(path)
    except (ValueError, IndexError):
        return None


def collect_image_ids(*dirs):
    ids = set()
    for base in dirs:
        if base is None or not Path(base).exists():
            continue
        for pattern in IMAGE_PATTERNS:
            ids.update(i for i in map(_safe_id, Path(base).rglob(pattern)) if i is not None)
    return ids


def discover(base, prefix, parse):
    pairs = []
    for d in sorted(Path(base).glob(f"{prefix}*")):
        if d.is_dir():
            try:
                pairs.append((parse(d.name[len(prefix):]), d))
            except ValueError:
                continue
    return sorted(pairs, key=lambda p: p[0])


def lam_dirs(seed_dir, target_lams=None):
    pairs = discover(seed_dir, "lam", lambda s: float(s.replace("p", ".")))
    if target_lams is None:
        return pairs
    return [(lam, d) for lam, d in pairs if any(abs(lam - t) < 1e-6 for t in target_lams)]


def seed_dirs(base):
    pairs = discover(base, "seed_", int)
    if not pairs and any(Path(base).glob("lam*")):
        return [(None, Path(base))]
    return pairs


def common_sample_count(base_dirs, target_ids, target_lams):
    counts = []
    for base in base_dirs:
        if not Path(base).exists():
            continue
        for _, seed_dir in seed_dirs(base):
            for _, d in lam_dirs(seed_dir, target_lams):
                n = len(list_images(d, target_ids=target_ids))
                if n > 0:
                    counts.append(n)
    if not counts:
        raise ValueError(f"No generated images found under {[str(d) for d in base_dirs]}")
    return min(counts)


def load_prompts(path):
    df = pd.read_csv(path)
    df["image_id"] = df["file_name"].map(lambda fn: int(Path(fn).stem.split("_")[-1]))
    return dict(zip(df["image_id"], df["text"]))


def feature_stats(folder, model, device, n, target_ids):
    files = [str(f) for f in list_images(folder, n=n, target_ids=target_ids)]
    feats = get_files_features(files, model, device=torch.device(device))
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


@torch.no_grad()
def clip_score(folder, prompts, clip_model, clip_processor, device, n):
    scores = []
    for path in tqdm(list_images(folder, n=n), desc=f"CLIP {folder.name}"):
        image_id = _safe_id(path)
        if image_id not in prompts:
            continue
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(text=[prompts[image_id]], images=image, return_tensors="pt",
                                padding=True, truncation=True, max_length=75).to(device)
        scores.append(clip_model(**inputs).logits_per_image[0, 0].item())
    return np.mean(scores)


def evaluate(base, tag, models, real_stats, prompts, device, n, target_ids, target_lams):
    feat_model, clip_model, clip_processor = models
    per_lam = {}
    for seed, seed_dir in seed_dirs(base):
        for lam, d in lam_dirs(seed_dir, target_lams):
            if not list_images(d, n=n, target_ids=target_ids):
                continue
            mu, sigma = feature_stats(d, feat_model, device, n, target_ids)
            fid_val = frechet_distance(*real_stats, mu, sigma)
            clip_val = clip_score(d, prompts, clip_model, clip_processor, device, n)
            per_lam.setdefault(lam, []).append((fid_val, clip_val))
            print(f"[{tag} seed={seed}] lam={lam:.3f} FID={fid_val:.4f} CLIP={clip_val:.4f}")

    results = {}
    for lam, vals in per_lam.items():
        fids, clips = np.array(vals).T
        ddof = 1 if len(vals) > 1 else 0
        results[lam] = dict(fid_mean=fids.mean(), fid_std=fids.std(ddof=ddof),
                            clip_mean=clips.mean(), clip_std=clips.std(ddof=ddof), n_seeds=len(vals))
        print(f"[{tag}] lam={lam:.3f} FID={fids.mean():.4f}±{results[lam]['fid_std']:.4f} "
              f"CLIP={clips.mean():.4f}±{results[lam]['clip_std']:.4f} ({len(vals)} seeds)")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_dir", type=Path, required=True, help="COCO val2014 images.")
    parser.add_argument("--steer_dir", type=Path, required=True)
    parser.add_argument("--tsr_dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).parents[1] / "data" / "coco_10k_prompts.csv")
    parser.add_argument("--n_replicas", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--lams", type=float, nargs="+", default=None, help="Only evaluate these lam_start values.")
    parser.add_argument("--out", type=Path, default=Path("fid_clip.csv"))
    parser.add_argument("--figure", type=Path, default=Path("fid_vs_clip.png"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    steer_dirs = [args.steer_dir / f"steer_{n}" for n in args.n_replicas]
    target_ids = collect_image_ids(args.tsr_dir, args.steer_dir)
    n = common_sample_count([args.tsr_dir, *steer_dirs], target_ids, args.lams)
    print(f"Evaluating {n} images per setting over {len(target_ids)} COCO ids")

    feat_model = fid.build_feature_extractor("clean", device=torch.device(args.device))
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    models = (feat_model, clip_model, clip_processor)
    real_stats = feature_stats(args.real_dir, feat_model, args.device, n, target_ids)
    prompts = load_prompts(args.prompts)

    settings = [("tsr", args.tsr_dir)] + [(f"steer_{k}", d) for k, d in zip(args.n_replicas, steer_dirs)]
    rows = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for tag, base in settings:
        results = evaluate(base, tag, models, real_stats, prompts, args.device, n, target_ids, args.lams)
        if not results:
            continue
        lams = sorted(results, reverse=True)
        clips = [results[lam]["clip_mean"] for lam in lams]
        fids = [results[lam]["fid_mean"] for lam in lams]
        ax.plot(clips, fids, marker="o", linewidth=2, label=tag)
        for lam, c, f in zip(lams, clips, fids):
            ax.annotate(f"{lam:.2f}", (c, f), textcoords="offset points", xytext=(6, 0), fontsize=8)
        rows += [dict(setting=tag, lam_start=lam, n_images=n, **results[lam]) for lam in lams]

    pd.DataFrame(rows).to_csv(args.out, index=False)
    ax.set_xlabel("CLIP")
    ax.set_ylabel("FID")
    ax.set_title(f"FID vs. CLIP ({n} images per setting)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)


if __name__ == "__main__":
    main()
