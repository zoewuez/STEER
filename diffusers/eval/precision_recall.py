"""Improved precision and recall (Kynkäänniemi et al., 2019) of generated image folders.

Every directory under `--root` that directly contains images, other than `--real_dir`, is evaluated
against the real images using VGG16 features and k-nearest-neighbour manifold radii.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImageFolderFlat(Dataset):
    def __init__(self, folder, transform, limit=None, seed=0):
        paths = sorted(p for p in Path(folder).iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 0)
        if limit is not None and len(paths) > limit:
            idx = np.sort(np.random.RandomState(seed).choice(len(paths), size=limit, replace=False))
            paths = [paths[i] for i in idx]
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            return self.transform(Image.open(self.paths[idx]).convert("RGB"))
        except OSError:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


def find_image_dirs(root, real_dir):
    root = Path(root)
    real_path = (root / real_dir).resolve()
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        if dirpath.resolve() == real_path:
            dirnames[:] = []
            continue
        if any(Path(f).suffix.lower() in IMAGE_EXTS for f in filenames):
            found[str(dirpath.relative_to(root))] = dirpath
    return found


def build_feature_extractor(device):
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])
    return vgg.eval().requires_grad_(False).to(device)


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_features(folder, model, device, batch_size=64, limit=None, num_workers=4, cache_path=None, seed=0):
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)

    dataset = ImageFolderFlat(folder, TRANSFORM, limit=limit, seed=seed)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {folder}")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=device.type == "cuda", collate_fn=collate_skip_none)
    feats = [model(batch.to(device, non_blocking=True)).cpu().numpy() for batch in loader if batch is not None]
    if not feats:
        raise RuntimeError(f"No decodable images found in {folder}")
    feats = np.concatenate(feats).astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, feats)
    return feats


def pairwise_distances(x, y, device, batch_size=2048):
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    out = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size]
        d = (xb ** 2).sum(dim=1).view(-1, 1) + y_norm - 2.0 * xb @ y.t()
        out.append(d.clamp(min=0.0).sqrt().cpu().numpy())
    return np.concatenate(out)


def kth_nn_radii(feats, k, device, batch_size=2048):
    d = pairwise_distances(feats, feats, device, batch_size)
    d.sort(axis=1)
    return d[:, k]


def precision_recall(real_feats, fake_feats, device, k=3, batch_size=2048, real_radii=None):
    if real_radii is None:
        real_radii = kth_nn_radii(real_feats, k, device, batch_size)
    fake_radii = kth_nn_radii(fake_feats, k, device, batch_size)
    d_fr = pairwise_distances(fake_feats, real_feats, device, batch_size)
    precision = np.mean(np.any(d_fr <= real_radii[None, :], axis=1))
    d_rf = pairwise_distances(real_feats, fake_feats, device, batch_size)
    recall = np.mean(np.any(d_rf <= fake_radii[None, :], axis=1))
    return float(precision), float(recall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--real_dir", default="coco_10k_real", help="Real-image folder, relative to --root.")
    parser.add_argument("--only", nargs="*", default=None, help="Only evaluate folders containing one of these substrings.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_real", type=int, default=10000)
    parser.add_argument("--num_fake", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dist_batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("precision_recall.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Skip folders already present in --out.")
    args = parser.parse_args()
    device = torch.device(args.device)

    real_dir = args.root / args.real_dir
    if not real_dir.is_dir():
        raise SystemExit(f"Real image directory not found: {real_dir}")
    fake_dirs = find_image_dirs(args.root, args.real_dir)
    if args.only:
        fake_dirs = {label: p for label, p in fake_dirs.items() if any(s in label for s in args.only)}

    results = []
    if args.resume and args.out.exists():
        with open(args.out, newline="") as f:
            results = list(csv.DictReader(f))
        done = {r["folder"] for r in results}
        fake_dirs = {label: p for label, p in fake_dirs.items() if label not in done}
    if not fake_dirs:
        raise SystemExit("No generated-image folders left to evaluate.")

    def cache_path(label):
        if args.cache_dir is None:
            return None
        digest = hashlib.md5(label.encode()).hexdigest()[:10]
        return args.cache_dir / f"{label.replace('/', '__')}_{digest}.npy"

    model = build_feature_extractor(device)
    feature_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    real_feats = extract_features(real_dir, model, device, limit=args.num_real,
                                  cache_path=cache_path(f"__real__{args.real_dir}"), **feature_kwargs)
    real_radii = kth_nn_radii(real_feats, args.k, device, args.dist_batch_size)

    for label in sorted(fake_dirs):
        try:
            fake_feats = extract_features(fake_dirs[label], model, device, limit=args.num_fake,
                                          cache_path=cache_path(label), **feature_kwargs)
        except RuntimeError as e:
            print(f"Skipping {label}: {e}")
            continue
        precision, recall = precision_recall(real_feats, fake_feats, device, k=args.k,
                                             batch_size=args.dist_batch_size, real_radii=real_radii)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        print(f"{label}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
        results.append(dict(folder=label, n_real=len(real_feats), n_fake=len(fake_feats), k=args.k,
                            precision=precision, recall=recall, f1=f1))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "n_real", "n_fake", "k", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
