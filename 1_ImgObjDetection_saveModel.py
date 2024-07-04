# Type A

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model_path = 'path/to/your/saved_model'
model = tf.saved_model.load(model_path)

def load_image_into_numpy_array(path):
    image = Image.open(path).resize((320, 320))  
    return np.array(image)

image_path = 'TestImage.jpg'
image_np = load_image_into_numpy_array(image_path)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]  

detections = model(input_tensor)

num_detections = int(detections['num_detections'])
print("Number of detections:", num_detections)

print("Detection Scores:", detections['detection_scores'].numpy())
print("Detection Classes:", detections['detection_classes'].numpy())

max_score_index = np.argmax(detections['detection_scores'])
max_score = detections['detection_scores'][0][max_score_index].numpy()
max_score_class = int(detections['detection_classes'][0][max_score_index].numpy()) 

print(f"max score class: {max_score_class}, score: {max_score:.2f}")

def display_detections(image_np, detections, threshold=0.3):  
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    ax = plt.gca()

    for i in range(num_detections):
        score = detections['detection_scores'][0][i].numpy() 
        if score < threshold:  
            continue

        bbox = detections['detection_boxes'][0][i].numpy() 
        class_id = int(detections['detection_classes'][0][i].numpy())  

        ymin, xmin, ymax, xmax = bbox
        (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
                                      ymin * image_np.shape[0], ymax * image_np.shape[0])

        rect = plt.Rectangle((left, top), right - left, bottom - top,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        plt.text(left, top, f'Class: {class_id}, Score: {score:.2f}',
                 fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()

display_detections(image_np, detections, threshold=0.3) 


# Type B
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model_path = 'path/to/your/saved_model'
model = tf.saved_model.load(model_path)

def load_image_into_numpy_array(path):
    image = Image.open(path).resize((320, 320))  
    return np.array(image)

image_paths = ['TestImage1.jpg', 'TestImage2.jpg', 'TestImage3.jpg', 'TestImage4.jpg', 'TestImage5.jpg', 'TestImage6.jpg', 'TestImage7.jpg']  

def display_detections(image_np, detections, threshold=0.6):  
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    ax = plt.gca()

    for i in range(num_detections):
        score = detections['detection_scores'][0][i].numpy() 
        if score < threshold:  
            continue

        bbox = detections['detection_boxes'][0][i].numpy() 
        class_id = int(detections['detection_classes'][0][i].numpy())  

        ymin, xmin, ymax, xmax = bbox
        (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
                                      ymin * image_np.shape[0], ymax * image_np.shape[0])

        rect = plt.Rectangle((left, top), right - left, bottom - top,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        plt.text(left, top, f'Class: {class_id}, Score: {score:.2f}',
                 fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()

for image_path in image_paths:
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]  

    detections = model(input_tensor)

    num_detections = int(detections['num_detections'])
    print(f"이미지: {image_path}")
    print("Number of detections:", num_detections)

    print("Detection Scores:", detections['detection_scores'].numpy())
    print("Detection Classes:", detections['detection_classes'].numpy())

    max_score_index = np.argmax(detections['detection_scores'])
    max_score = detections['detection_scores'][0][max_score_index].numpy()
    max_score_class = int(detections['detection_classes'][0][max_score_index].numpy()) 

    print(f"max score class: {max_score_class}, score: {max_score:.2f}")

    display_detections(image_np, detections, threshold=0.4)
"""
