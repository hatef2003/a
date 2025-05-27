import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time

MODEL_PATH = "240_yolov8n_full_integer_quant_edgetpu.tflite"
LABEL_PATH = "Y.txt"
IMAGE_PATH = "bus.jpg"  # your test image

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def preprocess(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    return np.expand_dims(np.array(image, dtype=np.uint8), axis=0), image

def run_inference(interpreter, image):
    set_input_tensor(interpreter, image)
    
    start = time.time()

    interpreter.invoke()
    end = time.time()
    print(f"Inference time: {end - start:.2f} seconds")

    output_details = interpreter.get_output_details()
    print("Number of output tensors:", len(output_details))
    for i, out in enumerate(output_details):
        print(f"Output {i}: shape={out['shape']}, dtype={out['dtype']}")
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    return 

def main():
    labels = load_labels(LABEL_PATH)

    # Load Edge TPU-compatible TFLite model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    image_input, raw_image = preprocess(IMAGE_PATH, input_shape)

    run_inference(interpreter, image_input)

    # for i in range(len(scores)):
    #     if scores[i] > 0.5:
    #         print(f"Detected {labels[int(classes[i])]} with confidence {scores[i]:.2f}")

if __name__ == '__main__':
    main()
