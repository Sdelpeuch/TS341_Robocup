import os

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class DeepLearningProcessing:
    def __init__(self, data):
        self.data = data
        plt.show()

    def predict(self):
        PATH_TO_SAVED_MODEL = "tod_tf2/object_detection/training/ssd_mobilnet/saved_model2/saved_model"
        PROJECT = "object_detection"
        NB_MAX_OBJ = 2

        detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

        PATH_TO_LABELS = os.path.join('tod_tf2/object_detection/training/', 'label_map.pbtxt')
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)

        # self.data.image.working_image = cv2.resize(self.data.image.working_image, (300, 172))
        input_tensor = tf.convert_to_tensor(np.array(self.data.image.working_image))
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        if num_detections > NB_MAX_OBJ:
            num_detections = NB_MAX_OBJ
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        x_list = detections['detection_boxes'][:, 1]
        indexes = np.argsort(x_list)
        list_label_sorted = detections['detection_classes'][indexes]
        list_boxe_sorted = detections['detection_boxes'][indexes]
        list_score_sorted = detections['detection_scores'][indexes]
        self.center_boxes(list_boxe_sorted, self.data.image.working_image)

        image_copy = self.data.image.working_image.copy()
        vis_utils.visualize_boxes_and_labels_on_image_array(
            image_copy,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            line_thickness=2,
            use_normalized_coordinates=True,
            max_boxes_to_draw=2,
            min_score_thresh=0.1,
            agnostic_mode=False)
        self.data.image.predict_image = image_copy
        plt.imshow(image_copy)
        plt.show()
        # self.data.image.working_image = cv2.resize(self.data.image.working_image,
        #                                            (self.data.image.width, self.data.image.height))
        # self.data.image.base_goal_1 = (
        #     int(self.data.image.base_goal_1[0] * self.data.image.working_image.shape[1] / 300),
        #     int(self.data.image.base_goal_1[1] * self.data.image.working_image.shape[0] / 172))
        # self.data.image.base_goal_2 = (
        #     int(self.data.image.base_goal_2[0] * self.data.image.working_image.shape[1] / 300),
        #     int(self.data.image.base_goal_2[1] * self.data.image.working_image.shape[0] / 172))

    def center_boxes(self, list_boxe_sorted, image_np):
        if len(list_boxe_sorted) == 0:
            return
        elif len(list_boxe_sorted) >= 1:
            x_min = list_boxe_sorted[0][1] * self.data.image.working_image.shape[1]
            y_min = list_boxe_sorted[0][0] * self.data.image.working_image.shape[0]
            x_max = list_boxe_sorted[0][3] * self.data.image.working_image.shape[1]
            y_max = list_boxe_sorted[0][2] * self.data.image.working_image.shape[0]
            self.data.image.base_goal_1 = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        if len(list_boxe_sorted) == 2:
            x_min = list_boxe_sorted[1][1] * self.data.image.working_image.shape[1]
            y_min = list_boxe_sorted[1][0] * self.data.image.working_image.shape[0]
            x_max = list_boxe_sorted[1][3] * self.data.image.working_image.shape[1]
            y_max = list_boxe_sorted[1][2] * self.data.image.working_image.shape[0]
            self.data.image.base_goal_2 = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        return
