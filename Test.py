import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# load
model_path = 'PATH/TO/SAVED/MODEL'
model = tf.saved_model.load(model_path)

def load_image_into_numpy_array(path):
    image = Image.open(path).resize((320, 320))  # resizing
    return np.array(image)

image_path = 'PATH/TO/TEST/IMAGE.jpg'
image_np = load_image_into_numpy_array(image_path)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]  # Batch dimension

detections = model(input_tensor)

# number of detected classes
num_detections = int(detections['num_detections'])
print("Number of detections:", num_detections)
print("Detection Scores:", detections['detection_scores'].numpy())
print("Detection Classes:", detections['detection_classes'].numpy())

max_score_index = np.argmax(detections['detection_scores'])
max_score = detections['detection_scores'][0][max_score_index].numpy()
max_score_class = int(detections['detection_classes'][0][max_score_index].numpy())  # [0] 추가

print(f"가장 높은 점수를 가진 클래스: {max_score_class}, 점수: {max_score:.2f}")

# visualization
def display_detections(image_np, detections, threshold=0.3):  # 임계값 기본값 0.3
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    ax = plt.gca()

    for i in range(num_detections):
        score = detections['detection_scores'][0][i].numpy()  # i번째 점수 선택
        if score < threshold:  # 임계값 적용
            continue

        bbox = detections['detection_boxes'][0][i].numpy()  # i번째 bbox 선택
        class_id = int(detections['detection_classes'][0][i].numpy())  # i번째 클래스 ID 선택

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

display_detections(image_np, detections, threshold=0.3)  # 임계값 설정