import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite
import time

model_path = 'trueY8.tflite'
interpreter = tflite.Interpreter(model_path=model_path,experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print('----------------------------------------------------------------------')
print(output_details)