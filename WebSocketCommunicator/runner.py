import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import time

# === Load the Edge TPU model ===
model_path = 'y5.tflite'
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

# === Get model input/output details ===
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debug info
print("Input details:", input_details)
print("Output details:", output_details)
print("------------------------------------------------------")

# === Load and preprocess the image ===
image_path = 'pets.jpg'
input_shape = input_details[0]['shape']      # [1, height, width, 3]
height, width = input_shape[1], input_shape[2]

# Load image and convert to RGB
image = Image.open(image_path).convert('RGB').resize((width, height))
image_np = np.asarray(image, dtype=np.float32)

# Get quantization parameters for input tensor
scale, zero_point = input_details[0]['quantization']  # e.g., (0.017, 128)

# Quantize image to int8
input_data = image_np / scale + zero_point
input_data = np.round(input_data).astype(np.int8)
input_data = np.expand_dims(input_data, axis=0)  # Shape: [1, H, W, 3]


# Confirm dtype
for i in range(10):
    print("input_data dtype:", input_data.dtype)

    # === Set input tensor and run inference ===
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # === Get and print the output ===
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("Inference Time: {:.2f} ms".format((end_time - start_time) * 1000))
    # print("Raw Output:", output_data)

# === Optional: post-process if classification ===
if len(output_data.shape) == 2 or len(output_data.shape) == 1:
    predicted_label = np.argmax(output_data)
    confidence = np.max(output_data)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.2f}")
