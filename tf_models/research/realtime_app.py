import cv2
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import os
import time

def load_image_into_numpy_array(image):
    # BW -> RGB
    last_axis = -1
    image = np.expand_dims(image, last_axis)
    dim_to_repeat = 2
    repeats = 3
    image = np.repeat(image, repeats, dim_to_repeat)
    return image


# recover our saved model
pipeline_file = "trained_pb/essentialDet/imp/pipeline_file.config"
pipeline_config = pipeline_file
# generally you want to put the last ckpt from training in here
model_dir = 'trained_pb/essentialDet/ckpt-6'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
    model=detection_model)
ckpt.restore(os.path.join('trained_pb/essentialDet/ckpt-6'))


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


size = (640, 640, 3)


def resize(img):
    height, width = img.shape[:2]

    blank_image = np.zeros(size, np.uint8)
    blank_image[:, :] = (255, 255, 255)

    l_img = blank_image.copy()  # (600, 900, 3)

    x_offset = y_offset = 0
    # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
    l_img[y_offset:y_offset + height, x_offset:x_offset + width] = img.copy()
    return l_img


detect_fn = get_model_detection_function(detection_model)

# map labels for inference decoding
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

# run detector on test image
# it takes a little longer on the first run and then runs at normal speed.


cap = cv2.VideoCapture(0)
while True:
    ret, image_np = cap.read()
    start = time.time()
    image_np = cv2.resize(image_np, (640,640))
    #image_np = resize(image_np)
    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    classes = detections['detection_classes'][0].numpy().astype(int) + label_id_offset
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.2,
        agnostic_mode=False,
    )
    # Scale to pixel co-ordinates
    # boxes[:, (0, 2)] *= 640
    # boxes[:, (1, 3)] *= 640

    # cond = (classes == 1) & (scores >= .2)
    # selected_base = boxes[cond, :][0]
    # based_pose = [210.89705, 376.74963, 308.44348, 426.44995]  # todo auto
    # # POSITION
    # shift_x = based_pose[0] - selected_base[0]
    # shift_y = based_pose[1] - selected_base[1]
    end = time.time()
    print("fps=", end-start)
    cv2.imshow('frame', image_np_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
