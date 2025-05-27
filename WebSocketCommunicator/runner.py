import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import time

# Load Edge TPU model
model_path = 'trueY8.tflite'
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
print("------------------------------------------------------")

# Load and preprocess image
image_path = 'pets.jpg'
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]
# Preprocess image
image = Image.open(image_path).convert('RGB').resize((width, height))
input_data = np.asarray(image)
input_data = np.expand_dims(input_data, axis=0).astype(np.int8)
print("Expected dtype:", input_details[0]['dtype'])  # should show <class 'numpy.int8'>



# Check if input type is quantized (uint8)
if input_details[0]['dtype'] == np.uint8:
    input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)
else:
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    # Optional: normalize if required by model
    # input_data = (input_data - 127.5) / 127.5

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
start_time = time.time()
interpreter.invoke()
end_time = time.time()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print inference result and time
print("Inference Time: {:.2f} ms".format((end_time - start_time) * 1000))
print("Raw output:", output_data)

# (Optional) Post-processing - depends on your model
# If it's classification:
predicted_label = np.argmax(output_data)
confidence = np.max(output_data)
print(f"Predicted label: {predicted_label}, Confidence: {confidence:.2f}")
