# STEER for Stable Diffusion 3

Text-to-image sampling with [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers), comparing STEER against temporal score rescaling (TSR) on COCO captions.

`pipeline_steer_sd3.py` defines `StableDiffusion3SteerPipeline`, a drop-in subclass of the `diffusers` SD3 pipeline that adds three samplers:

| `sampler` | Behaviour |
|---|---|
| `vanilla` | Unmodified SD3 sampling. |
| `tsr` | Every replica's noise prediction is rescaled for its rung of the temperature ladder. |
| `steer` | TSR plus a replica-exchange swap between neighbouring rungs at every step. |

SD3 is a flow-matching model, so the pipeline converts the velocity prediction to a noise prediction with $\alpha_t = 1 - \sigma_t$, applies STEER from the shared [`steer`](../steer) package, and converts back before the scheduler step.

## Installation

```bash
pip install -e ..                 # the shared steer package
pip install -r requirements.txt
```

SD3 Medium is gated on the Hugging Face Hub: accept the license on the model page and run `huggingface-cli login`, or pass a local copy with `--model`.

## Usage

```python
import torch
from pipeline_steer_sd3 import StableDiffusion3SteerPipeline

pipe = StableDiffusion3SteerPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
).to("cuda")

images = pipe(
    "A man in a tie and a fake moustache",
    negative_prompt="",
    num_inference_steps=30,
    guidance_scale=7.5,
    sampler="steer",
    lam_start=0.95,
    lam_end=1.17,
    n_replicas=6,
    generator=torch.Generator("cuda").manual_seed(42),
).images

images[0].save("steer.png")  # the lam_start replica
```

Images come back ordered by rung, so `images[r * n_prompts + p]` is prompt `p` at rung `r` and `images[0]` is the `lam_start` sample. The initial noise of each returned image, in the same order, is stored in `pipe.last_initial_latents`.

| Argument | Default | Description |
|---|---|---|
| `sampler` | `"vanilla"` | `vanilla`, `tsr` or `steer`. |
| `lam_start` | `1.0` | Temperature of the first rung, which is the rung that is read out. Values below 1 sharpen the distribution. |
| `lam_end` | `lam_start` | Temperature of the last rung. Rungs are linearly spaced. |
| `n_replicas` | `1` | Number of rungs, i.e. latents sampled per prompt. |
| `max_sequence_length` | `77` | T5 prompt length. The paper runs used 77; upstream `diffusers` defaults to 256. |

All other arguments behave as in `StableDiffusion3Pipeline`.

## Reproducing the experiments

**1. Generate images.** For every prompt, `generate.py` samples one STEER ladder and saves its `lam_start` image. It then samples a TSR image at the same temperature, starting from the same initial noise. By default `lam_end = 1 / (1 / lam_start - 0.2)`.

```bash
for lam in 0.95 0.98 1.02 1.05 1.1; do
  python generate.py \
      --lam_start $lam \
      --n_replicas 2 4 6 8 \
      --seeds 42 \
      --start 0 --end 10000 \
      --steer_dir images/steer \
      --tsr_dir images/tsr
done
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `stabilityai/stable-diffusion-3-medium-diffusers` | Hub id or local path. |
| `--prompts` | `data/coco_10k_prompts.csv` | 10k COCO val2014 captions (`text`, `file_name`). |
| `--lam_start` | `1.0` | Temperature that is read out. |
| `--lam_end` | `1 / (1 / lam_start - 0.2)` | Temperature of the last rung. |
| `--n_replicas` | `4` | One or more ladder sizes to sweep. |
| `--seeds` | `42` | One or more seeds. |
| `--start`, `--end` | `0`, `15` | Range of prompts, ordered by COCO image id. |
| `--num_inference_steps` | `30` | |
| `--guidance_scale` | `7.5` | |

Existing images are skipped, so interrupted runs can be resumed. Outputs use COCO file names so they can be matched against the reference images:

```
images/
├── steer/steer_{n_replicas}/seed_{seed}/lam{lam_start}/COCO_val2014_{image_id:012d}.png
└── tsr/seed_{seed}/lam{lam_start}/COCO_val2014_{image_id:012d}.png
```

**2. FID and CLIP score.** These are computed against the COCO val2014 images with the same ids. Every setting is evaluated on the same number of images.

```bash
python eval/fid_clip.py \
    --real_dir /path/to/coco/val2014 \
    --steer_dir images/steer \
    --tsr_dir images/tsr \
    --n_replicas 2 4 6 8 \
    --lams 0.95 \
    --out fid_clip.csv --figure fid_vs_clip.png
```

**3. Precision and recall.** This uses improved precision/recall with VGG16 features. Every image folder under `--root` is evaluated against `--root/--real_dir`.

```bash
python eval/precision_recall.py \
    --root images \
    --real_dir coco_10k_real \
    --k 3 --num_real 10000 --num_fake 10000 \
    --cache_dir feat_cache \
    --out precision_recall.csv
```
