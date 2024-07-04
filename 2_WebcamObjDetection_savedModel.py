# Type A
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

model_path = 'path/to/your/saved_model'
model = tf.saved_model.load(model_path)

def load_image_into_numpy_array(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(image_rgb)

def process_frame(frame):
    image_np = load_image_into_numpy_array(frame)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]  # Batch dimension 추가

    detections = model(input_tensor)

    num_detections = int(detections['num_detections'])

    detection_scores = detections['detection_scores'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0].astype(int)

    for i in range(num_detections):
        score = detection_scores[i]
        if score > 0.6: 
            class_id = detection_classes[i]
            bbox = detections['detection_boxes'][0][i].numpy()
            ymin, xmin, ymax, xmax = bbox
            (left, right, top, bottom) = (int(xmin * frame.shape[1]), int(xmax * frame.shape[1]),
                                          int(ymin * frame.shape[0]), int(ymax * frame.shape[0]))

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {class_id}, Score: {score:.2f}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while(True):
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Object Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# Type B
"""
import tensorflow as tf
import cv2
import numpy as np

model_path = 'C:/path/to/your/saved_model'
model = tf.saved_model.load(model_path)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

threshold = 0.5  

while(True):
    ret, frame = cap.read()

    if not ret:
        print("check your webcam.")
        break

    resized_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA) 

    # RGB
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)  # uint8 
    input_tensor = input_tensor[tf.newaxis, ...]

    # GPU
    with tf.device('/GPU:0'):  
        detections = model(input_tensor)

    detections = model(input_tensor)

    for i in range(int(detections['num_detections'])):
        score = detections['detection_scores'][0][i].numpy()

        if score > threshold:
            ymin, xmin, ymax, xmax = detections['detection_boxes'][0][i].numpy()
            class_id = int(detections['detection_classes'][0][i].numpy())

            left, right, top, bottom = (int(xmin * frame.shape[1]), int(xmax * frame.shape[1]),
                                          int(ymin * frame.shape[0]), int(ymax * frame.shape[0]))

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {class_id}, Score: {score:.2f}', (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
