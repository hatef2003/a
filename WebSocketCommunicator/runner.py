import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
from PIL import Image, ImageDraw

# Specify the model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = "./y8.tflite"
label_file = "Y.txt"
image_file = "./bus.jpg"

# Initialize the interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Load and preprocess the image
size = common.input_size(interpreter)
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Run inference
common.set_input(interpreter, image)
interpreter.invoke()

# Get detection results
threshold = 0.1
objects = detect.get_objects(interpreter, score_threshold=threshold)

# Draw results
# draw = ImageDraw.Draw(image)
labels = dataset.read_label_file(label_file)

for obj in objects:
    bbox = obj.bbox
    draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], outline='red', width=2)
    label = labels.get(obj.id, obj.id)
    print(f'{label}: {obj.score:.2f}')
    draw.text((bbox.xmin + 5, bbox.ymin + 5), f'{label} {obj.score:.2f}', fill='red')

# Optionally show the image
image.show()
