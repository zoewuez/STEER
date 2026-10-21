"""Generate COCO-prompted SD3 images with STEER and a matched TSR baseline.

For every prompt, one STEER ladder is sampled and its `lam_start` image is saved. A TSR image is then
sampled at the same temperature from the exact initial noise that the saved STEER image started from.

Output layout:
    {steer_dir}/steer_{n_replicas}/seed_{seed}/lam{lam_start}/COCO_val2014_{image_id:012d}.png
    {tsr_dir}/seed_{seed}/lam{lam_start}/COCO_val2014_{image_id:012d}.png
"""

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from pipeline_steer_sd3 import StableDiffusion3SteerPipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).parent / "data" / "coco_10k_prompts.csv")
    parser.add_argument("--steer_dir", type=Path, default=Path("images/steer"))
    parser.add_argument("--tsr_dir", type=Path, default=Path("images/tsr"))
    parser.add_argument("--lam_start", type=float, default=1.0)
    parser.add_argument("--lam_end", type=float, default=None,
                        help="Temperature of the last rung (default: 1 / (1 / lam_start - 0.2)).")
    parser.add_argument("--n_replicas", type=int, nargs="+", default=[4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--start", type=int, default=0, help="Index of the first prompt to generate.")
    parser.add_argument("--end", type=int, default=15, help="Index one past the last prompt to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    return parser.parse_args()


def lam_dirname(lam):
    return f"lam{lam:.3f}".replace(".", "p")


def coco_filename(image_id):
    return f"COCO_val2014_{image_id:012d}.png"


def load_prompts(path, start, end):
    df = pd.read_csv(path, usecols=["file_name", "text"])
    df["image_id"] = df["file_name"].map(lambda fn: int(Path(fn).stem.split("_")[-1]))
    return df.sort_values("image_id").reset_index(drop=True).iloc[start:end]


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    lam_end = args.lam_end if args.lam_end is not None else 1 / (1 / args.lam_start - 0.2)
    print(f"lam_start={args.lam_start}, lam_end={lam_end}")

    pipe = StableDiffusion3SteerPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    prompts = load_prompts(args.prompts, args.start, args.end)
    sample_kwargs = dict(
        negative_prompt="",
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    for n_replicas in args.n_replicas:
        for seed in args.seeds:
            steer_dir = args.steer_dir / f"steer_{n_replicas}" / f"seed_{seed}" / lam_dirname(args.lam_start)
            tsr_dir = args.tsr_dir / f"seed_{seed}" / lam_dirname(args.lam_start)
            steer_dir.mkdir(parents=True, exist_ok=True)
            tsr_dir.mkdir(parents=True, exist_ok=True)

            for row in prompts.itertuples():
                steer_path = steer_dir / coco_filename(row.image_id)
                tsr_path = tsr_dir / coco_filename(row.image_id)
                if steer_path.exists() and tsr_path.exists():
                    continue

                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(
                    row.text,
                    sampler="steer",
                    lam_start=args.lam_start,
                    lam_end=lam_end,
                    n_replicas=n_replicas,
                    generator=generator,
                    **sample_kwargs,
                ).images[0]
                if not steer_path.exists():
                    image.save(steer_path)
                initial_latents = pipe.last_initial_latents[:1].clone()
                free_cuda()

                if not tsr_path.exists():
                    image = pipe(
                        row.text,
                        sampler="tsr",
                        lam_start=args.lam_start,
                        latents=initial_latents,
                        **sample_kwargs,
                    ).images[0]
                    image.save(tsr_path)
                free_cuda()

            print(f"Finished n_replicas={n_replicas}, seed={seed}")


if __name__ == "__main__":
    main()
