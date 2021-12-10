"""
This file performs a prediction on an image via the network whose training results are available in the object_detection
folder.
"""

import os

import matplotlib
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class DeepLearningProcessing:
    """
    This class performs a prediction on an image via the network whose training results are available in the
    object_detection folder.
    """

    def __init__(self, data):
        self.data = data
        plt.show()

    def predict(self):
        """
        This function performs a prediction on an image via the network whose training results are available in the
        object_detection folder.
        :return: update self.data.image.base_goal_1 and self.data.image.base_goal_2 with the coordinates of the base and
        the goal detected by the network. Update self.data.image.predict_image with the image of the prediction. Finally
        ,update self.data.image.working_image with the base image cropped according to the coordinates of the base and
        the goal detected by the network.
        """
        PATH_TO_SAVED_MODEL = "tod_tf2/object_detection/training/ssd_mobilnet/saved_model3/saved_model"
        NB_MAX_OBJ = 2

        detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

        PATH_TO_LABELS = os.path.join('tod_tf2/object_detection/training/', 'label_map.pbtxt')
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

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
        list_boxe_sorted = detections['detection_boxes'][indexes]
        list_score_sorted = detections['detection_scores'][indexes]

        boxe1, boxe2 = [], []
        if list_score_sorted[0] > 0.4:
            boxe1 = list_boxe_sorted[0]
        if list_score_sorted[1] > 0.4:
            boxe2 = list_boxe_sorted[1]

        self.center_boxes((boxe1, boxe2), self.data.image.working_image)

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
            min_score_thresh=0.5,
            agnostic_mode=False)
        self.data.image.predict_image = image_copy
        self.crop_image()

        # if resize is needed
        # self.data.image.working_image = cv2.resize(self.data.image.working_image,
        #                                            (self.data.image.width, self.data.image.height))
        # self.data.image.base_goal_1 = (
        #     int(self.data.image.base_goal_1[0] * self.data.image.working_image.shape[1] / 300),
        #     int(self.data.image.base_goal_1[1] * self.data.image.working_image.shape[0] / 172))
        # self.data.image.base_goal_2 = (
        #     int(self.data.image.base_goal_2[0] * self.data.image.working_image.shape[1] / 300),
        #     int(self.data.image.base_goal_2[1] * self.data.image.working_image.shape[0] / 172))

    def center_boxes(self, boxes, image_np):
        """
        This function find the center of the boxes and update the coordinates of the base and the goal.
        :param boxes: the boxes detected by the network.
        :param image_np: the image on which the boxes are detected.
        :return: update self.data.image.base_goal_1 and self.data.image.base_goal_2 with the coordinates of the base and
        the goal detected by the network.
        """
        boxe1 = boxes[0]
        boxe2 = boxes[1]
        if boxe1 == [] and boxe2 == []:
            return
        elif boxe1 != [] and boxe2 == []:
            self.data.image.base_goal_1 = self.unormalize(boxe1)
        elif boxe1 == [] and boxe2 != []:
            self.data.image.base_goal_1 = self.unormalize(boxe2)
        elif boxe1 != [] and boxe2 != []:
            self.data.image.base_goal_1 = self.unormalize(boxe1)
            self.data.image.base_goal_2 = self.unormalize(boxe2)
        return

    def unormalize(self, boxe):
        """
        This function unormalize the coordinates of the boxes.
        :param boxe: the boxe detected by the network.
        :return: the coordinates of the boxe in the original image.
        """
        x_min = boxe[1] * self.data.image.working_image.shape[1]
        y_min = boxe[0] * self.data.image.working_image.shape[0]
        x_max = boxe[3] * self.data.image.working_image.shape[1]
        y_max = boxe[2] * self.data.image.working_image.shape[0]
        return int((x_min + x_max) / 2), int((y_min + y_max) / 2)

    def crop_image(self):
        """
        This function crop the image to the base and the goal.
        :return: update self.data.image.working_image with the cropped image.
        """
        cte_x = int(0.05 * self.data.image.working_image.shape[1])
        cte_y = int(0.05 * self.data.image.working_image.shape[0])

        if self.data.image.base_goal_1 != (-1, -1) and self.data.image.base_goal_2 != (-1, -1):
            height = max(self.data.image.base_goal_1[1], self.data.image.base_goal_2[1]) + cte_y

            left_width = min(self.data.image.base_goal_1[0], self.data.image.base_goal_2[0]) - cte_x \
                if min(self.data.image.base_goal_1[0], self.data.image.base_goal_2[0]) - cte_x > 0 \
                else 0

            right_width = max(self.data.image.base_goal_1[0], self.data.image.base_goal_2[0]) + cte_x \
                if max(self.data.image.base_goal_1[0], self.data.image.base_goal_2[0]) + cte_x < \
                   self.data.image.working_image.shape[1] \
                else self.data.image.working_image.shape[1]

            self.data.image.working_image = self.data.image.working_image[0:height, left_width:right_width]
            self.data.image.cropped_coordinates = (left_width, height)

        elif self.data.image.base_goal_1 != (-1, -1) and self.data.image.base_goal_2 == (-1, -1):
            height = self.data.image.base_goal_1[1] + cte_y

            left_width = self.data.image.base_goal_1[0] - cte_x \
                if self.data.image.base_goal_1[0] - cte_x > 0 \
                else 0

            right_width = self.data.image.base_goal_1[0] + cte_x \
                if self.data.image.base_goal_1[0] + cte_x < self.data.image.working_image.shape[1] \
                else self.data.image.working_image.shape[1]

            self.data.image.working_image = self.data.image.working_image[0:height, left_width:right_width]
            self.data.image.cropped_coordinates = (left_width, height)

        elif self.data.image.base_goal_1 == (-1, -1) and self.data.image.base_goal_2 != (-1, -1):
            height = self.data.image.base_goal_2[1] + cte_y

            left_width = self.data.image.base_goal_2[0] - cte_x \
                if self.data.image.base_goal_2[0] - cte_x > 0 \
                else 0
            right_width = self.data.image.base_goal_2[0] + cte_x \
                if self.data.image.base_goal_2[0] + cte_x < self.data.image.working_image.shape[1] \
                else self.data.image.working_image.shape[1]
            self.data.image.working_image = self.data.image.working_image[0:height, left_width:right_width]
            self.data.image.cropped_coordinates = (left_width, height)
