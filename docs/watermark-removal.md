# Full-frame watermark removal (ProPainter)

The pipeline's built-in branding bands (`brand_top_band` / `brand_bottom_band`)
only *hide* a watermark sitting on the top/bottom edge. They cannot recover a
watermark that is alpha-blended across the whole frame ("đánh chìm"):

```
observed = alpha * watermark + (1 - alpha) * original
```

For that case the optional **`watermark`** stage reconstructs the frame with
[ProPainter](https://github.com/sczhou/ProPainter), a video-inpainting model
that fills the masked region using neighbouring frames (temporally consistent,
no flicker). It runs right after `ingest` and writes `source_clean.mp4`, which
every downstream stage (`scenes`, `render`) then reads instead of the original.

The stage is **disabled by default** and **self-skips** when ProPainter is not
installed, so nothing changes for existing jobs.

## 1. Install ProPainter (one-time, needs a GPU)

```bash
git clone https://github.com/sczhou/ProPainter
cd ProPainter
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# weights download automatically on first run into ./weights
```

## 2. Point the pipeline at it

| Variable | Purpose | Default |
| --- | --- | --- |
| `MRF_PROPAINTER_DIR` | clone dir containing `inference_propainter.py` (**required**) | – |
| `MRF_PROPAINTER_PYTHON` | interpreter for ProPainter's venv | current python |
| `MRF_PROPAINTER_DEVICE` | `cuda` or `cpu` (`cpu` disables `--fp16`) | `cuda` |
| `MRF_PROPAINTER_FP16` | half precision `1`/`0` | `1` |
| `MRF_PROPAINTER_MASK_DILATION` | grow the mask by N px | `8` |
| `MRF_PROPAINTER_RESIZE_RATIO` | downscale before inpaint (speed/VRAM) | `1.0` |
| `MRF_PROPAINTER_SUBVIDEO_LENGTH` / `_NEIGHBOR_LENGTH` / `_REF_STRIDE` | window tuning | `80` / `10` / `10` |

```bash
# Windows PowerShell
$env:MRF_PROPAINTER_DIR = "D:\tools\ProPainter"
$env:MRF_PROPAINTER_PYTHON = "D:\tools\ProPainter\.venv\Scripts\python.exe"
```

## 3. Enable it on a job

Set `watermark_removal` in the job config. Give it **either** an explicit mask
PNG (white = remove, black = keep) **or** let a rectangular mask be generated
from bands / boxes using the frame size detected at ingest.

```python
from movie_review_factory.models import JobConfig, WatermarkRemoval

cfg = JobConfig(
    job_id="demo",
    source_video="input.mp4",
    watermark_removal=WatermarkRemoval(
        enabled=True,
        mask="watermark_mask.png",   # OR use boxes / bands below
        # boxes=[[0.35, 0.4, 0.3, 0.2]],           # [x, y, w, h] as fractions
        # top_band=0.12, bottom_band=0.12,          # mirrors the branding bands
    ),
)
```

Artifacts written by the stage:

- `source_clean.mp4` – the watermark-free source used downstream.
- `watermark_mask.png` – the generated mask (only when no explicit mask given).
- `watermark.json` – provenance (source, mask, ProPainter home).

## Auto-detect the mask (no hand-drawing)

Instead of `mask` / `boxes` / bands, set `detect` to have the stage build a
per-frame mask folder (`watermark_masks/`, `00000.png` …) straight from the
video. Three methods:

```python
watermark_removal=WatermarkRemoval(
    enabled=True,
    detect=WatermarkDetect(
        method="color",              # threshold a colour per frame (moving marks)
        color=[255, 255, 255],       # target RGB (white/grey overlay)
        tolerance=30, dilation=4,
        # method="temporal", threshold=12,   # flag pixels that barely change
        # method="external", external_cmd="python detect.py --in {video} --out {out}",
    ),
)
```

| Method | Best for | Needs |
| --- | --- | --- |
| `color` | semi-transparent white/grey overlay, moving or static | FFmpeg |
| `temporal` | a fixed overlay over moving content | FFmpeg |
| `external` | a smart detector (Florence-2 / SAM) you already have | `MRF_MASK_DETECTOR_CMD` or `external_cmd` (`{video}`/`{out}` placeholders) |

`color`/`temporal` decode frames through an FFmpeg pipe and process them incrementally; they do not materialize a full-resolution frame scratch directory on disk. A machine without FFmpeg (or without the external detector) **skips** the stage rather than failing.

### External detector: Florence-2 sample

A ready-to-use detector ships at
[`scripts/detectors/florence2_watermark_masks.py`](../scripts/detectors/florence2_watermark_masks.py).
It decodes every frame and writes numbered masks (`00000.png` …) using
Florence-2 open-vocabulary grounding (or polygon segmentation with `--task
segment`). Its deps (torch/transformers/opencv) are **not** bundled — install
them in a separate environment:

```bash
python -m venv florence-env
florence-env\Scripts\activate        # Windows (Linux/macOS: source florence-env/bin/activate)
pip install --upgrade pip
# CUDA users: install the matching torch wheel first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r scripts/detectors/requirements-florence2.txt
```

Point the pipeline at that interpreter (prefer paths without spaces — the
command is split with shlex):

```powershell
$env:MRF_MASK_DETECTOR_CMD = "D:\envs\florence-env\Scripts\python.exe " +
    "D:\LacViet\movie-review-factory\scripts\detectors\florence2_watermark_masks.py " +
    "--in {video} --out {out} --prompt watermark --scene-threshold 0.08 --dilation 8"
```

Then enable `WatermarkRemoval(enabled=True, detect=WatermarkDetect(method="external"))`.

Both detector scripts **only re-run detection when the scene changes** (mean
frame difference vs the last detected frame ≥ `--scene-threshold`, default
`0.08`) and reuse the mask in between — a static/slow watermark barely moves, so
this runs the model a handful of times instead of once per frame. Use
`--scene-threshold 0` to force detection on every frame, or `--every N` to also
force a refresh at least every N frames.

### Grounded-SAM sample (pixel-tight masks)

The Florence-2 script above produces **box** masks. For **pixel-accurate** masks
(tight around an irregular logo), use
[`scripts/detectors/grounded_sam_watermark_masks.py`](../scripts/detectors/grounded_sam_watermark_masks.py):
a detector (Florence-2 grounding or GroundingDINO) finds boxes, then **SAM**
segments inside each box. All via 🤗 transformers — nothing to compile.

```bash
pip install -r scripts/detectors/requirements-grounded-sam.txt   # in a separate env
```

```powershell
$env:MRF_MASK_DETECTOR_CMD = "D:\envs\grounded-sam-env\Scripts\python.exe " +
    "D:\LacViet\movie-review-factory\scripts\detectors\grounded_sam_watermark_masks.py " +
    "--in {video} --out {out} --prompt watermark --detector groundingdino --sam-model facebook/sam-vit-huge --scene-threshold 0.08"
```

Pick a detector with `--detector florence2` (default) or `--detector groundingdino`
(needs `--box-threshold` / `--text-threshold`), and SAM size with `--sam-model`
(`sam-vit-base` = light, `sam-vit-huge` = best).

**Choosing:** box masks (`florence2_watermark_masks.py`) are faster and fine for
rectangular badges; Grounded-SAM is slower but hugs irregular shapes. Both honour
the same contract — accept `--in {video} --out {out}` and write zero-padded PNG
masks — so any custom detector can be dropped in the same way.

## Choosing the mask

- **Static logo/badge** → one PNG (or a box) covering the region is enough.
- **Bands on the edges** → `top_band` / `bottom_band` (fractions of height).
- **Moving/animated mark** → set `mask` to a **folder** of per-frame masks
  (zero-padded PNGs: `00000.png`, `00001.png`, … in frame order, white = remove).
  ProPainter matches them to frames by sorted filename and reuses the last mask
  if there are fewer masks than frames. A detector (e.g. Florence-2) can generate
  these frame masks. `watermark.json` then records `mask_kind: "per-frame"` and
  the `mask_frames` count.

## Notes & alternatives

- ProPainter *inpaints* (repaints) the masked region. For a faint, uniform
  overlay you may get sharper results from an **un-blending / decomposition**
  approach (e.g. WDNet) that solves the alpha equation instead of repainting.
- Keep only footage you are authorised to process.
