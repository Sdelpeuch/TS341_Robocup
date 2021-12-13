"""
Object Detection From TF2 Saved Model
=====================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Saved Model Format <https://www.tensorflow.org/guide/saved_model>`__ to load the model.

import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

## Adapted by Jean-Luc Charles (JLC) from the work of Evan Juras.
###
### JLC v1.0 2020/07/11 - add argparse to allow options in teh command line.
###                     - using PIL.Image do load image into a numpy array
###                     - using matplotlib instead of cv2 to display image.
###

# Import packages
import sys, os, argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
matplotlib.use('TkAgg')

parser = argparse.ArgumentParser(description="Uses a trained network to detect object in images")
parser.add_argument('-p', '--project', type=str, required=True,
                    help='project name.')
parser.add_argument('-s', '--path_to_saved_model', type=str, required=True,
                    help='path to the "saved_model" directory.')
parser.add_argument('-i', '--images', type=str, required=True,
                    help='path of the image to process.>')
parser.add_argument('-n', '--nb_max_object', type=int, required=True,
                    help='number max of object to detect.')
parser.add_argument('-t', '--threshold', type=int, required=False, default=50,
                    help='Detection theshold (percent) to display bounding boxe.')
args = parser.parse_args()

# Name of the directory containing the object detection module we're using
PATH_TO_SAVED_MODEL = args.path_to_saved_model
PROJECT = args.project
if os.path.isfile(args.images):
    IMAGE_PATHS = [args.images]
elif os.path.isdir(args.images):
    IMAGE_PATHS = [os.path.join(args.images, f) for f in os.listdir(args.images) if f.lower().endswith("png") or f.lower().endswith("jpg")]

THRESHOLD  = args.threshold/100
NB_MAX_OBJ = args.nb_max_object

# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
print('Loading model...', end='')
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Done! Took {elapsed_time:.2f} seconds')


# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.

PATH_TO_LABELS = os.path.join(PROJECT, './training', 'label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.

#import warnings
#warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

import xml.etree.ElementTree as ET
import cv2

med = 0
moy = 0 
i = 0

for image_path in IMAGE_PATHS:

    tree = ET.parse(image_path[0:len(image_path)-4]+'.xml')
    root = tree.getroot()
    child = root.find('object')
    box = child.find('bndbox')

    xmin = int(box.find('xmin').text)*300/800
    ymin = int(box.find('ymin').text)*172/448
    xmax = int(box.find('xmax').text)*300/800
    ymax = int(box.find('ymax').text)*172/448

    print('Running inference for {}... '.format(image_path), end='')

    #image_expanded = np.expand_dims(image_rgb, axis=0)
    image_np = load_image_into_numpy_array(image_path)
    image_np = cv2.resize(image_np, (300,172))

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    # input_tensor = np.expand_dims(image_np, 0)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    if num_detections > NB_MAX_OBJ: num_detections = NB_MAX_OBJ
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    print(detections['detection_classes'])
    print(detections['detection_scores'])
    print(detections['detection_boxes'])

    image_np_with_detections = image_np.copy()

    vis_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          line_thickness=3,
          use_normalized_coordinates=True,
          max_boxes_to_draw=4,
          min_score_thresh=THRESHOLD,
          agnostic_mode=False)

    img = Image.fromarray(image_np_with_detections)
    draw = ImageDraw.Draw(img)
    draw.line((xmin, ymin, xmin, ymax))
    draw.line((xmin, ymax, xmax, ymax))
    draw.line((xmax, ymin, xmax, ymax))
    draw.line((xmin, ymin, xmax, ymin))
    box1 = (xmin, ymin, xmax, ymax)
    box2 = (detections['detection_boxes'][0][1]*300, detections['detection_boxes'][0][0]*172, detections['detection_boxes'][0][3]*300, detections['detection_boxes'][0][2]*172)
    iou = bb_intersection_over_union(box1, box2)
    draw.text((5, 5), 'IoU = ' + str(iou))
    image_np_with_detections = np.asarray(img)

    med = med + iou
    moy = moy + detections['detection_scores']
    i = i + 1

    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.show()

moyenne_IoU = med/i
moyenne_detection = moy/i
print('moyenne IoU = ',moyenne_IoU)
print('moyenne detection = ',moyenne_detection)