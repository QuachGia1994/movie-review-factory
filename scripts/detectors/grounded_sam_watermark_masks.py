#!/usr/bin/env python3
"""Grounded-SAM watermark mask detector for the ``external`` mask method.

Two-stage, for pixel-tight masks (vs. the box-only Florence-2 script):

  1. a text-prompted detector (Florence-2 grounding **or** GroundingDINO) finds
     bounding boxes for the watermark, then
  2. SAM segments inside each box to a precise per-pixel mask.

Everything runs through 🤗 transformers (GroundingDino / Sam / Florence-2), so
there is nothing to compile from the original repos.

Contract expected by ``movie_review_factory.mask_detection`` (method="external"):

    <python> grounded_sam_watermark_masks.py --in {video} --out {out}

Decodes every frame of ``--in`` and writes one zero-padded single-channel PNG
mask per frame into ``--out`` (``00000.png`` …), white (255) = "remove". Masks
stay in decode order so ProPainter matches them to frames by sorted filename.

Heavy deps (torch/transformers/opencv) are NOT bundled with the pipeline —
install them separately (scripts/detectors/requirements-grounded-sam.txt) and
point ``MRF_MASK_DETECTOR_CMD`` at that interpreter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_FLORENCE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write per-frame watermark masks with Grounded-SAM.")
    parser.add_argument("--in", dest="input", required=True, help="source video path")
    parser.add_argument("--out", dest="out", required=True, help="output mask directory")
    parser.add_argument("--prompt", default="watermark", help="text describing the watermark region")
    parser.add_argument("--detector", choices=["florence2", "groundingdino"], default="florence2",
                        help="box detector backend")
    parser.add_argument("--det-model", dest="det_model", default="",
                        help="detector model id (default: Florence-2-large / grounding-dino-base)")
    parser.add_argument("--sam-model", dest="sam_model", default="facebook/sam-vit-base",
                        help="SAM model id (sam-vit-huge = best, sam-vit-base = light)")
    parser.add_argument("--box-threshold", dest="box_threshold", type=float, default=0.3,
                        help="GroundingDINO box confidence threshold")
    parser.add_argument("--text-threshold", dest="text_threshold", type=float, default=0.25,
                        help="GroundingDINO text confidence threshold")
    parser.add_argument("--device", default="", help="cuda / cpu (default: auto)")
    parser.add_argument("--fp16", action="store_true", help="fp16 for the detector on CUDA (SAM stays fp32)")
    parser.add_argument("--dilation", type=int, default=2, help="grow the mask by N px (0 = off)")
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

    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor, SamModel, SamProcessor
    except ImportError as exc:  # pragma: no cover - depends on the external env
        print(
            f"[grounded-sam] missing dependency: {exc}\n"
            "Install into a separate env:\n"
            "  pip install -r scripts/detectors/requirements-grounded-sam.txt",
            file=sys.stderr,
        )
        return 2

    source = Path(args.input)
    out_dir = Path(args.out)
    if not source.is_file():
        print(f"[grounded-sam] source video not found: {source}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    det_dtype = torch.float16 if (device == "cuda" and args.fp16) else torch.float32

    # --- load SAM (segmentation) --------------------------------------------
    print(f"[grounded-sam] loading SAM {args.sam_model} on {device} ...", file=sys.stderr)
    sam_model = SamModel.from_pretrained(args.sam_model).to(device)
    sam_model.eval()
    sam_processor = SamProcessor.from_pretrained(args.sam_model)

    # --- load the box detector ----------------------------------------------
    if args.detector == "florence2":
        det_id = args.det_model or "microsoft/Florence-2-large"
        print(f"[grounded-sam] loading Florence-2 {det_id} ...", file=sys.stderr)
        fl_model = AutoModelForCausalLM.from_pretrained(
            det_id, torch_dtype=det_dtype, trust_remote_code=True
        ).to(device)
        fl_model.eval()
        fl_processor = AutoProcessor.from_pretrained(det_id, trust_remote_code=True)

        def detect_boxes(image, width: int, height: int):
            inputs = fl_processor(
                text=_FLORENCE_GROUNDING + args.prompt, images=image, return_tensors="pt"
            ).to(device, det_dtype)
            with torch.inference_mode():
                generated = fl_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
            text = fl_processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = fl_processor.post_process_generation(
                text, task=_FLORENCE_GROUNDING, image_size=(width, height)
            )
            return [[float(v) for v in box] for box in parsed.get(_FLORENCE_GROUNDING, {}).get("bboxes", [])]
    else:
        from transformers import GroundingDinoForObjectDetection

        det_id = args.det_model or "IDEA-Research/grounding-dino-base"
        print(f"[grounded-sam] loading GroundingDINO {det_id} ...", file=sys.stderr)
        gd_processor = AutoProcessor.from_pretrained(det_id)
        gd_model = GroundingDinoForObjectDetection.from_pretrained(det_id).to(device)
        gd_model.eval()
        # GroundingDINO expects a lowercase phrase ending with a period.
        gd_text = args.prompt.strip().lower()
        if not gd_text.endswith("."):
            gd_text += " ."

        def detect_boxes(image, width: int, height: int):
            inputs = gd_processor(images=image, text=gd_text, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = gd_model(**inputs)
            results = gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(height, width)],
            )
            return [[float(v) for v in box] for box in results[0]["boxes"].tolist()]

    kernel = None
    if args.dilation > 0:
        kernel = np.ones((2 * args.dilation + 1, 2 * args.dilation + 1), np.uint8)

    def segment(image, boxes, width: int, height: int):
        mask = np.zeros((height, width), np.uint8)
        if not boxes:
            return mask
        inputs = sam_processor(image, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = sam_model(**inputs)
        per_box = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # tensor: (num_boxes, num_predicted, H, W)
        scores = outputs.iou_scores.cpu()[0]  # (num_boxes, num_predicted)
        for i in range(per_box.shape[0]):
            best = int(scores[i].argmax())
            mask[per_box[i, best].numpy().astype(bool)] = 255
        if kernel is not None and int(mask.max()) > 0:
            mask = cv2.dilate(mask, kernel)
        return mask

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"[grounded-sam] cannot open video: {source}", file=sys.stderr)
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
                boxes = detect_boxes(image, width, height)
                mask = segment(image, boxes, width, height)
                if int(mask.max()) == 0 and args.fallback_last and last_mask is not None:
                    mask = last_mask
                else:
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
                print(f"[grounded-sam] {index} frames ({detections} detections) ...", file=sys.stderr)
    finally:
        capture.release()

    if written == 0:
        print("[grounded-sam] no frames decoded - nothing written", file=sys.stderr)
        return 1
    print(f"[grounded-sam] wrote {written} masks to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
