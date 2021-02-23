import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'trained_pb' + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/object-detection.pbtxt'

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:


def load_image_into_numpy_array(image):
    # BW -> RGB
    last_axis = -1
    image = np.expand_dims(image, last_axis)
    dim_to_repeat = 2
    repeats = 3
    image = np.repeat(image, repeats, dim_to_repeat)
    return image

    # (im_width, im_height) = image.size
    # return np.array(image.getdata()).reshape(
    #     (im_height, im_width, 3)).astype(np.uint8)




def empty(area):
    pass


# Create Control Bar
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("threshold_red", "Parameters", 248, 255, empty)
cv2.createTrackbar("threshold_green", "Parameters", 255, 255, empty)
cv2.createTrackbar("threshold_blue", "Parameters", 255, 255, empty)
cv2.createTrackbar("t_light", "Parameters", 80, 255, empty)
cv2.createTrackbar("t_contrast", "Parameters", 90, 255, empty)

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


# # Detection

# In[9]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'images/test'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Image{}.jpg'.format(i)) for i in range(1, 13)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # for i, image_path in enumerate(TEST_IMAGE_PATHS):
        while True:
            # image = Image.open(image_path)
            # # the array based representation of the image will be used later in order to prepare the
            # # result image with boxes and labels on it.
            # image_np = load_image_into_numpy_array(image)



            # Capture frame-by-frame
            ret, frame = cap.read()

            threshold_red = cv2.getTrackbarPos("threshold_red", "Parameters")
            threshold_green = cv2.getTrackbarPos("threshold_green", "Parameters")
            threshold_blue = cv2.getTrackbarPos("threshold_blue", "Parameters")
            t_contrast = cv2.getTrackbarPos("t_contrast", "Parameters")
            t_light = cv2.getTrackbarPos("t_light", "Parameters")

            # Convert BGR to HSV
            hsv = apply_brightness_contrast(frame, t_light, t_contrast)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY)

            # define range of a color in HSV
            lower_hue = np.array([0, 0, 0])
            upper_hue = np.array([threshold_red, threshold_green, threshold_blue])

            # Threshold the HSV image to get only blue colors
            # mask = cv2.inRange(hsv, lower_hue, upper_hue)

            image_np = cv2.resize(hsv, (640, 640), interpolation=cv2.INTER_AREA)
            image_np = load_image_into_numpy_array(image_np)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            image_np = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                boxes=np.squeeze(boxes),
                classes=np.squeeze(classes).astype(np.int32),
                scores=np.squeeze(scores),
                category_index=category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.1)
            cv2.imshow('frame', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # plt.figure(figsize=IMAGE_SIZE)
            # plt.imshow(image_np)
            # plt.imsave('%dimage.png' % i, image_np)
            # plt.figure(figsize=IMAGE_SIZE)
            # plt.imshow(image_np)
            # plt.imsave('/root/models/0image.png',image_np )
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()