#!/usr/bin/env python3
# ------------------------------------------------------------
#  Coral-TPU websocket client:
#    • receives base-64 images over a JSON websocket frame
#    • performs YOLO-v8 object detection on the Edge-TPU
#    • returns a JSON list of detections
# ------------------------------------------------------------
import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite   # pycoral runtime

# ------------------------------------------------------------------
# ── YOLO-v8 helpers ────────────────────────────────────────────────
# ------------------------------------------------------------------
def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k highest‐scoring entries (unsorted)."""
    if scores.size <= k:
        return np.arange(scores.size)
    return np.argpartition(-scores, k)[:k]


def nms(boxes: np.ndarray,
        scores: np.ndarray,
        iou_thr: float = 0.5,
        max_det: int = 10) -> np.ndarray:
    """
    Classic Non-Max-Suppression (vectorised except for the main loop).
    Returns the indices of the boxes to keep (≤ max_det).
    """
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]                # descending
    keep  = []

    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.clip(xx2 - xx1, 0.0, None)
        h = np.clip(yy2 - yy1, 0.0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]

    return np.array(keep, dtype=np.int32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def postprocess_yolo(output:       np.ndarray,
                     img_wh:       Tuple[int, int],
                     conf_thr:     float = 0.4,
                     iou_thr:      float = 0.5,
                     max_det:      int = 10,
                     prefilter_k:  int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    • Converts raw YOLO-v8 tensor to absolute-pixel xyxy boxes
    • Performs confidence filtering + top-K pre-selection + NMS
    • Returns ≤ max_det boxes / class_ids / scores  (all np.ndarrays)
    """
    W, H = img_wh

    out  = output.squeeze().T        # [N, 84]
    boxes, obj_conf, cls_conf = out[:, :4], sigmoid(out[:, 4]), sigmoid(out[:, 5:])
    cls_ids = np.argmax(cls_conf, axis=1)
    cls_scores = cls_conf[np.arange(len(cls_conf)), cls_ids]
    scores = obj_conf * cls_scores

    # 1️⃣ confidence filter
    mask = scores > conf_thr
    if not mask.any():
        return np.empty((0, 4), dtype=np.int16), np.empty(0, dtype=np.int16), np.empty(0)

    boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]

    # 2️⃣ keep only the top-K highest scores before NMS (cuts work ~10×)
    idx = top_k(scores, prefilter_k)
    boxes, scores, cls_ids = boxes[idx], scores[idx], cls_ids[idx]

    # cx,cy,w,h ➜ x1,y1,x2,y2  (absolute pixels)
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
    boxes_xyxy = boxes_xyxy.astype(np.float32)

    # 3️⃣ Non-Max-Suppression (global — keeps max_det)
    keep = nms(boxes_xyxy, scores, iou_thr=iou_thr, max_det=max_det)

    return boxes_xyxy[keep], cls_ids[keep], scores[keep]


def set_input_tensor(interpreter, image_tensor):
    """Write an already-prepared input tensor into the Edge-TPU interpreter."""
    index = interpreter.get_input_details()[0]['index']
    interpreter.tensor(index)()[0][:] = image_tensor


def preprocess_pil(img_pil: Image.Image, input_shape):
    """
    Resize a PIL image to the model’s expected H × W and reformat
    to uint8 NHWC.
    """
    h, w = input_shape[2], input_shape[3]  # input_shape = [1, H, W, C]
    img_resized = img_pil.convert("RGB").resize((w, h))
    arr = np.asarray(img_resized, np.uint8)
    img_transposed = np.transpose(arr , (2,0,1))    
    tensor = np.expand_dims(img_transposed, 0)
    #tensor.transpose(0,3,1,2)
    return tensor


# ------------------------------------------------------------------
# ── Websocket handler ──────────────────────────────────────────────
# ------------------------------------------------------------------
async def inference_loop(ws_url: str,
                         interpreter: tflite.Interpreter,
                         labels: Dict[int, str],
                         conf_threshold: float):
    """
    Opens the websocket, then processes each inbound JSON message with key
    "payload" = base-64 JPEG/PNG.  Sends back:
        {"detections": [ {label, class_id, score, bbox:[x1,y1,x2,y2]} ]}
    where bbox coordinates are **pixel values** w.r.t. the original image.
    """
    # Prepare some fixed metadata
    in_h, in_w = interpreter.get_input_details()[0]['shape'][1:3]

    async with websockets.connect(ws_url) as ws:
        print(f"✅ Connected to {ws_url}")
        async for msg in ws:
            start_time = time.perf_counter()
            try:
                data = json.loads(msg)
                b64_image = data["payload"]
            except (json.JSONDecodeError, KeyError):
                print("⚠️  Received malformed message – skipping")
                continue

            # --- decode & preprocess -------------------------------------------------
            decode_time = time.perf_counter()
            img_bytes = base64.b64decode(b64_image)
            pil_img   = Image.open(io.BytesIO(img_bytes))
            orig_w, orig_h = pil_img.size
            
            preprocess_time = time.perf_counter()
            input_tensor = preprocess_pil(pil_img,
                                          interpreter.get_input_details()[0]['shape'])
            set_input_tensor(interpreter, input_tensor)

            # --- inference -----------------------------------------------------------
            inference_time = time.perf_counter()
            interpreter.invoke()

            output_parse_time = time.perf_counter()
            output_details = interpreter.get_output_details()
            output = interpreter.get_tensor(output_details[0]['index'])

            postprocess_time = time.perf_counter()
            # boxes, classes, scores = postprocess_yolo(output,
            #                                         img_wh=(orig_w, orig_h),
            #                                         conf_thr=conf_threshold,
            #                                         iou_thr=0.5,
            #                                         max_det=10)

            # detections = [
            #     {
            #         "label": labels.get(int(c), f"id_{int(c)}"),
            #         "class_id": int(c),
            #         "score": float(f"{s:.4f}"),
            #         "bbox": [int(x1), int(y1), int(x2), int(y2)]
            #     }
            #     for (x1, y1, x2, y2), c, s in zip(boxes, classes, scores)
            # ]

            # --- send results ---------------------------------------------------------
            send_time = time.perf_counter()
            print(output_details[0].keys())
            print(type(output_details[0]))
            print(output[0])
            await ws.send(json.dumps({"detections": float(output[0][0]),
                                      # "inference_ms": round(inf_ms, 2)
                                      }))
            finish_time = time.perf_counter()
            
            rec_ms = (decode_time - start_time)*1000
            dec_ms = (preprocess_time - decode_time)*1000
            pre_ms = (inference_time - preprocess_time)*1000
            inf_ms = (output_parse_time - inference_time)*1000
            out_ms = (postprocess_time - output_parse_time)*1000
            pps_ms = (send_time - postprocess_time)*1000
            snd_ms = (finish_time - send_time)*1000
            print(f"🖼 dets – rec: {rec_ms}, dec: {dec_ms}, pre: {pre_ms}, inf: {inf_ms}, out: {out_ms}, pps: {pps_ms}, snd: {snd_ms} (ms)")
            # print(f"🖼  {len(detections)} dets – rec: {rec_ms}, dec: {dec_ms}, pre: {pre_ms}, inf: {inf_ms}, out: {out_ms}, pps: {pps_ms}, snd: {snd_ms} (ms)")

# ------------------------------------------------------------------
# ── Command-line entry - point ─────────────────────────────────────
# ------------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-m", "--model",
                   required=True,
                   help="Edge-TPU-compiled .tflite model path")
    p.add_argument("-l", "--labels",
                   required=True,
                   help="Label file; one class name per line")
    p.add_argument("--host", default="192.168.43.8",
                   help="Websocket server IP or hostname")
    p.add_argument("--port", type=int, default=8125,
                   help="Websocket server port")
    p.add_argument("-t", "--threshold", type=float, default=0.4,
                   help="Confidence threshold")
    return p.parse_args()


def load_labels(path: str) -> Dict[int, str]:
    return {i: line.strip() for i, line in enumerate(Path(path).read_text().splitlines())}


async def main_async():
    args = parse_cli()

    # --- Edge-TPU interpreter -----------------------------------------------------
    interpreter = tflite.Interpreter(model_path=args.model,
                                     experimental_delegates=[
                                         tflite.load_delegate("libedgetpu.so.1")
                                     ])
    interpreter.allocate_tensors()
    print("✅ Model loaded and tensors allocated")

    labels = load_labels(args.labels)
    ws_url = f"ws://{args.host}:{args.port}"
    await inference_loop(ws_url, interpreter, labels, args.threshold)


if __name__ == "__main__":
    import websockets          # deferred import, raises fast if missing
    asyncio.run(main_async())
