#!/usr/bin/env python3
"""Florence-2 watermark mask detector for the ``external`` mask method.

Contract expected by ``movie_review_factory.mask_detection`` (method="external"):

    <python> florence2_watermark_masks.py --in {video} --out {out}

It decodes every frame of ``--in`` and writes one zero-padded single-channel
PNG mask per frame into ``--out`` (``00000.png``, ``00001.png`` ...), white (255)
= "remove this pixel". ProPainter then matches masks to frames by sorted
filename, so the numbering must stay in decode order.

This script lives OUTSIDE the packaged runtime on purpose: Florence-2 pulls in
torch/transformers, which the pipeline never bundles. Install its dependencies
in a separate environment (see scripts/detectors/requirements-florence2.txt) and
point the pipeline at that interpreter via ``MRF_MASK_DETECTOR_CMD``.

Wiring example (PowerShell):

    $env:MRF_MASK_DETECTOR_CMD = "D:\\envs\\florence\\Scripts\\python.exe " +
        "D:\\repo\\scripts\\detectors\\florence2_watermark_masks.py " +
        "--in {video} --out {out} --prompt watermark --every 5 --dilation 8"

then run a job with ``WatermarkRemoval(enabled=True, detect=WatermarkDetect(method="external"))``.
(Prefer paths without spaces: the pipeline splits the command with shlex.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Florence-2 task tokens.
_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
_SEGMENT = "<REFERRING_EXPRESSION_SEGMENTATION>"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write per-frame watermark masks with Florence-2.")
    parser.add_argument("--in", dest="input", required=True, help="source video path")
    parser.add_argument("--out", dest="out", required=True, help="output mask directory")
    parser.add_argument("--prompt", default="watermark", help="text describing the watermark region")
    parser.add_argument("--task", choices=["grounding", "segment"], default="grounding",
                        help="grounding = bounding-box mask (robust); segment = polygon mask (tighter)")
    parser.add_argument("--model", default="microsoft/Florence-2-large", help="Florence-2 model id")
    parser.add_argument("--device", default="", help="cuda / cpu (default: auto)")
    parser.add_argument("--dilation", type=int, default=6, help="grow the mask by N px (0 = off)")
    parser.add_argument("--scene-threshold", dest="scene_threshold", type=float, default=0.08,
                        help="re-detect when the mean frame difference (0-1) vs the last detected "
                             "frame exceeds this (scene change); 0 = detect every frame")
    parser.add_argument("--every", type=int, default=0,
                        help="also force a re-detect at least every N frames (0 = only on scene change)")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=0, help="cap frames (0 = all)")
    parser.add_argument("--no-fallback-last", dest="fallback_last", action="store_false",
                        help="do NOT reuse the previous mask when a frame detects nothing")
    parser.set_defaults(fallback_last=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    every = max(0, args.every)

    # Heavy deps are imported lazily so --help and dependency errors stay clean.
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:  # pragma: no cover - depends on the external env
        print(
            f"[florence2] missing dependency: {exc}\n"
            "Install into a separate env:\n"
            "  pip install -r scripts/detectors/requirements-florence2.txt",
            file=sys.stderr,
        )
        return 2

    source = Path(args.input)
    out_dir = Path(args.out)
    if not source.is_file():
        print(f"[florence2] source video not found: {source}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[florence2] loading {args.model} on {device} ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    task = _SEGMENT if args.task == "segment" else _GROUNDING
    kernel = None
    if args.dilation > 0:
        kernel = np.ones((2 * args.dilation + 1, 2 * args.dilation + 1), np.uint8)

    def detect(image, width: int, height: int):
        """Return an HxW uint8 mask (255 = watermark) for one frame."""
        inputs = processor(text=task + args.prompt, images=image, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(text, task=task, image_size=(width, height))
        result = parsed.get(task, {}) or {}

        mask = np.zeros((height, width), np.uint8)
        if args.task == "segment":
            for instance in result.get("polygons", []):
                for polygon in instance:
                    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], 255)
        else:
            for box in result.get("bboxes", []):
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        return mask

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"[florence2] cannot open video: {source}", file=sys.stderr)
        return 2

    def thumbnail(frame):
        """Small grayscale frame for a cheap scene-change metric."""
        return cv2.cvtColor(cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    def scene_changed(reference, current) -> bool:
        if reference is None or args.scene_threshold <= 0:
            return True  # first frame, or "detect every frame" mode
        return float(np.mean(cv2.absdiff(reference, current))) / 255.0 >= args.scene_threshold

    index = 0
    written = 0
    detections = 0
    last_mask = None
    reference_thumb = None
    since_detect = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if args.max_frames and index >= args.max_frames:
                break
            height, width = frame_bgr.shape[:2]
            thumb = thumbnail(frame_bgr)

            forced = every > 0 and since_detect >= every
            if last_mask is None or scene_changed(reference_thumb, thumb) or forced:
                image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                mask = detect(image, width, height)
                if int(mask.max()) == 0 and args.fallback_last and last_mask is not None:
                    mask = last_mask  # detection missed this frame -> keep the last hit
                else:
                    if kernel is not None and int(mask.max()) > 0:
                        mask = cv2.dilate(mask, kernel)
                    last_mask = mask
                reference_thumb = thumb
                since_detect = 0
                detections += 1
            else:
                mask = last_mask
                since_detect += 1

            cv2.imwrite(str(out_dir / f"{index:05d}.png"), mask)
            index += 1
            written += 1
            if index % 100 == 0:
                print(f"[florence2] {index} frames ({detections} detections) ...", file=sys.stderr)
    finally:
        capture.release()

    if written == 0:
        print("[florence2] no frames decoded - nothing written", file=sys.stderr)
        return 1
    print(f"[florence2] wrote {written} masks to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
