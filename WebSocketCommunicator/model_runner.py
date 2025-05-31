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
from typing import Dict

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite   # pycoral runtime

# ------------------------------------------------------------------
# ── YOLO-v8 helpers ────────────────────────────────────────────────
# ------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def postprocess_yolo(output,
                     conf_threshold: float = 0.4,
                     iou_threshold: float = 0.4):
    """
    YOLO-v8 nano (n) post-processing for a single output tensor shaped
    [1, 84, N].  Returns boxes in (x1, y1, x2, y2) *relative* coords
    plus class IDs and confidences.
    """
    output = np.squeeze(output)           # [84, N]
    output = output.T                     # [N, 84]
    boxes  = output[:, :4]                # cx, cy, w, h
    objectness = sigmoid(output[:, 4])
    class_probs = sigmoid(output[:, 5:])  # [N, 80]
    class_ids   = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(len(class_probs)), class_ids]

    conf = objectness * class_scores
    keep = conf > conf_threshold
    if not np.any(keep):
        return [], [], []

    boxes, conf, class_ids = boxes[keep], conf[keep], class_ids[keep]

    # cx,cy,w,h ➜ x1,y1,x2,y2  (relative 0-1 range)
    boxes_xyxy         = np.empty_like(boxes)
    boxes_xyxy[:, 0]   = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1]   = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2]   = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3]   = boxes[:, 1] + boxes[:, 3] / 2.0

    return boxes_xyxy, class_ids, conf


def set_input_tensor(interpreter, image_tensor):
    """Write an already-prepared input tensor into the Edge-TPU interpreter."""
    index = interpreter.get_input_details()[0]['index']
    interpreter.tensor(index)()[0][:] = image_tensor


def preprocess_pil(img_pil: Image.Image, input_shape):
    """
    Resize a PIL image to the model’s expected H × W and reformat
    to uint8 NHWC.
    """
    h, w = input_shape[1], input_shape[2]  # input_shape = [1, H, W, C]
    img_resized = img_pil.convert("RGB").resize((w, h))
    tensor = np.expand_dims(np.asarray(img_resized, dtype=np.uint8), 0)
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
            start = time.perf_counter()
            try:
                data = json.loads(msg)
                b64_image = data["payload"]
            except (json.JSONDecodeError, KeyError):
                print("⚠️  Received malformed message – skipping")
                continue

            # --- decode & preprocess -------------------------------------------------
            img_bytes = base64.b64decode(b64_image)
            pil_img   = Image.open(io.BytesIO(img_bytes))
            orig_w, orig_h = pil_img.size

            input_tensor = preprocess_pil(pil_img,
                                          interpreter.get_input_details()[0]['shape'])
            set_input_tensor(interpreter, input_tensor)

            # --- inference -----------------------------------------------------------
            interpreter.invoke()

            output_details = interpreter.get_output_details()
            output = interpreter.get_tensor(output_details[0]['index'])

            boxes_r, class_ids, scores = postprocess_yolo(output, conf_threshold)

            # Map relative boxes → absolute integer pixel coords
            detections = []
            for (x1, y1, x2, y2), cid, score in zip(boxes_r, class_ids, scores):
                detections.append({
                    "label"    : labels.get(int(cid), f"id_{cid}"),
                    "class_id" : int(cid),
                    "score"    : float(f"{score:.4f}"),
                    "bbox"     : [int(x1 * orig_w), int(y1 * orig_h),
                                  int(x2 * orig_w), int(y2 * orig_h)]
                })

            # --- send results ---------------------------------------------------------
            await ws.send(json.dumps({"detections": detections,
                                      # "inference_ms": round(inf_ms, 2)
                                      }))
            inf_ms = (time.perf_counter() - start) * 1000.0
            print(f"🖼️  {len(detections)} dets – {inf_ms:5.1f} ms sent")

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
