import tensorflow as tf
import numpy as np

saved_model_dir = "saved_model" # SavedModel Path

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] #Option

# type option
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

# convert
tflite_model = converter.convert()
with open('converted_model.tflite', 'wb') as f:
  f.write(tflite_model)
