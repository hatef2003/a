import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time

MODEL_PATH = "../model_s_int8_static_edgetpu.tflite"
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
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess_yolov9(output, conf_threshold=0.5, iou_threshold=0.4):
    output = np.squeeze(output)  # shape: [84, 1344]
    output = np.transpose(output)  # shape: [1344, 84]

    boxes = output[:, :4]  # x, y, w, h
    scores = sigmoid(output[:, 4])  # objectness score
    class_probs = sigmoid(output[:, 5:])  # shape [1344, 80]
    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(len(class_probs)), class_ids]

    # Final confidence = objectness * class probability
    final_scores = scores * class_scores

    # Filter by confidence threshold
    mask = final_scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = final_scores[mask]
    filtered_classes = class_ids[mask]

    # Convert from (x, y, w, h) to (x1, y1, x2, y2)
    boxes_xyxy = np.zeros_like(filtered_boxes)
    boxes_xyxy[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
    boxes_xyxy[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
    boxes_xyxy[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
    boxes_xyxy[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2

    return boxes_xyxy, filtered_classes, filtered_scores

def run_inference(interpreter, image):
    set_input_tensor(interpreter, image)
    for i in range(10):
        start = time.time()

        interpreter.invoke()
        end = time.time()
        print(f"Inference time: {end - start:.2f} seconds")

    # output_details = interpreter.get_output_details()
    # output_details = interpreter.get_output_details()
    # print(output_details) # shape [1, 84, 1344]
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

    # run_inference(interpreter, image_input)
    output = run_inference(interpreter, image_input)
    boxes, classes, scores = postprocess_yolov9(output)

    for i in range(len(scores)):
        print(f"Detected class {classes[i]} with confidence {scores[i]:.2f} at {boxes[i]}")

    # for i in range(len(scores)):
    #     if scores[i] > 0.5:
    #         print(f"Detected {labels[int(classes[i])]} with confidence {scores[i]:.2f}")

if __name__ == '__main__':
    main()
