import argparse
import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import asyncio
import websockets
import json
from PIL import Image
import io
import base64
from WebSocketCommunicator import WebSocketWrapper

def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')
def parseArges():
  parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file')
  parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects')
  args = parser.parse_args()
  return args

async def main():
    args = parseArges()

    labels = read_label_file(args.labels) if args.labels else {}
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    ws = WebSocketWrapper("192.168.43.8", 8125)
    await ws.connect()
    while (1):

        response = await ws.receive_json()
        image = response["payload"]
        image_bytes = base64.b64decode(image)
        image = Image.open(io.BytesIO(image_bytes))
        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

        print('----INFERENCE TIME----')
        print('Note: The first inference is slow because it includes',
              'loading the model into Edge TPU memory.')
        
        start = time.perf_counter()
        interpreter.invoke()
        objs = detect.get_objects(interpreter, args.threshold, scale)
        inference_time = time.perf_counter() - start
        print('%.2f ms' % (inference_time * 1000))

        print('-------RESULTS--------')
        if not objs:
          print('No objects detected')
        masage = ""
        for obj in objs:
          masage+=str(labels.get(obj.id, obj.id))
          masage+=str('  id:    ', obj.id)
          masage+=str('  score: ', obj.score)
          masage+=str('  bbox:  ', obj.bbox)
        ws.send(masage)



if __name__ == '__main__':
    asyncio.run(main())
