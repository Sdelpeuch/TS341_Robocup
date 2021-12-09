from cv2 import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cv2.cv2 import mean

from data import Data

matplotlib.use('TkAgg')


def _max_area(num_labels, stats, number):
    area_list = []
    for i in range(num_labels):
        area_list.append(stats[i][4])
    area_list.sort()
    if number == 1:
        try:
            return 1, area_list[-2], -1
        except IndexError:
            return 1, area_list[-1], -1
    elif number == 2:
        try:
            if area_list[-2] - area_list[-3] > 600:
                return 3, area_list[-2], -1
            else:
                return 2, area_list[-2], area_list[-3]
        except IndexError:
            if area_list[-1] - area_list[-2] > 600:
                return 3, area_list[-1], -1
            else:
                return 2, area_list[-1], area_list[-2]


class DataProcessing:
    """
    This class is used to process the data.
    """

    def __init__(self, path_folder, path_image, debug=False):
        self.image = Data(path_folder, path_image)
        self.debug = debug

    def show_image(self):
        """
        Show the image.
        """
        if self.debug:
            plt.imshow(self.image.working_image, cmap='gray')
            plt.show()

    def _gray_scale(self, threshold):
        """
        Convert the image to gray scale.
        """
        self.image.working_image = cv2.cvtColor(self.image.working_image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(self.image.working_image, cv2.COLOR_BGR2GRAY)
        ret, self.image.working_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        self.show_image()

    def _median(self):
        """
        Apply median filter to the image with a k size of 5.
        """
        self.image.working_image = cv2.medianBlur(self.image.working_image, 5)
        self.show_image()

    def _erode(self):
        """
        Apply erosion to the image with a kernel (2,2).
        """
        kernel = np.ones((2, 2), np.uint8)
        self.image.working_image = cv2.erode(self.image.working_image, kernel)
        self.show_image()

    def _dilate(self):
        """
        Apply dilation to the image with a kernel (2,2).
        """
        kernel = np.ones((5, 5), np.uint8)
        self.image.working_image = cv2.dilate(self.image.working_image, kernel)
        self.show_image()

    def _contours(self):
        """
        Detect vertical lines
        """
        self.image.working_image = cv2.Canny(self.image.working_image, 100, 200)
        self.show_image()

    def _max_area_components(self, number):
        output = cv2.connectedComponentsWithStats(self.image.working_image)
        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        number, area1, area2 = _max_area(num_labels, stats, number)
        if number == 1 or number == 3:
            for label in range(1, num_labels):
                if stats[label][4] == area1:
                    mask = np.array(labels, dtype=np.uint8)
                    mask[labels == label] = 255
                    self.image.working_image = mask
                    self.show_image()
                    return number
        if number == 2:
            mask1, mask2 = None, None
            for label in range(1, num_labels):
                if stats[label][4] == area1:
                    mask1 = np.array(labels, dtype=np.uint8)
                    mask1[labels == label] = 255
                if stats[label][4] == area2:
                    mask2 = np.array(labels, dtype=np.uint8)
                    mask2[labels == label] = 255
            self.image.working_image = cv2.bitwise_or(mask1, mask2)
            self.show_image()
            return number

    def _line_detection(self):
        """
        Detect vertical lines
        """
        y_position = 0
        for line in range(self.image.working_image.shape[0]):
            if mean(self.image.working_image[line])[0] >= 2 * 255 / self.image.working_image.shape[1]:
                return line
        return y_position

    def _column_detection(self, y_position):
        x_left, x_right = 0, 0
        for column in range(self.image.working_image.shape[1] - 1):
            if self.image.working_image[y_position][column] == 255 \
                    and self.image.working_image[y_position][column + 1] == 255:
                x_left = column
                break
        for column in range(self.image.working_image.shape[1] - 1, 0, -1):
            if self.image.working_image[y_position][column] == 255 \
                    and self.image.working_image[y_position][column - 1] == 255:
                x_right = column
                break
        return x_left, x_right

    def _reconstruct_one(self, number):
        ret, self.image.working_image = cv2.threshold(self.image.working_image, 240, 255, cv2.THRESH_BINARY)
        y_position = self._line_detection()
        x_left, x_right = self._column_detection(y_position)
        x_position = int((x_left + x_right) / 2)
        if number == 1:
            cv2.line(self.image.working_image, self.image.base_goal_1, (x_position, y_position), 255, 2)
        elif number == 2:
            cv2.line(self.image.working_image, self.image.base_goal_2, (x_position, y_position), 255, 2)

    def _reconstruct_two(self):
        # Cut the image in two
        sub_image_1 = self.image.working_image[0: self.image.working_image.shape[0],
                      0: self.image.working_image.shape[1] // 2]
        sub_image_2 = self.image.working_image[0: self.image.working_image.shape[0],
                      self.image.working_image.shape[1] // 2: self.image.working_image.shape[1]]
        self.image.working_image = sub_image_1
        self._reconstruct_one(1)
        sub_image_1 = self.image.working_image
        self.image.working_image = sub_image_2
        self.image.base_goal_2 = (self.image.base_goal_2[0] - self.image.width // 2, self.image.base_goal_2[1])
        self._reconstruct_one(2)
        sub_image_2 = self.image.working_image
        self.image.base_goal_2 = (self.image.base_goal_2[0] + self.image.width // 2, self.image.base_goal_2[1])
        self.image.working_image = np.zeros((self.image.height, self.image.width))
        self.image.working_image[0: sub_image_1.shape[0], 0: sub_image_1.shape[1]] = sub_image_1
        self.image.working_image[0: sub_image_2.shape[0],
        sub_image_1.shape[1]: sub_image_1.shape[1] + sub_image_2.shape[1]] = sub_image_2
        self.show_image()

    def _reconstruct_goal(self):
        """
        Reconstruct the goal position.
        """
        self._reconstruct_two()
        y_position = self._line_detection()
        x_left, x_right = self._column_detection(y_position)
        cv2.line(self.image.working_image, (self.image.base_goal_1[0], y_position),
                 (self.image.base_goal_2[0], y_position), 255, 2)

    def _superpose(self):
        """
        Superpose the image with the working image.
        """
        ret, mask = cv2.threshold(self.image.working_image, 240, 255, cv2.THRESH_BINARY)
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        new_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)
        new_mask[mask >= 180] = [0, 255, 0]
        self.image.working_image = cv2.addWeighted(self.image.base_image, 0.5, new_mask, 0.5, 0)
        self.image.working_image = cv2.addWeighted(self.image.working_image, 0.5, self.image.predict_image, 0.5, 0)
        self.show_image()

    def segmentation_post(self):
        """
        Segment one post
        """
        self._gray_scale(130)
        self._erode()
        self._erode()
        self._median()
        self._contours()
        self._max_area_components(1)
        self._dilate()
        self._reconstruct_one(1)
        self._superpose()

    def segmentation_goal(self):
        """
        Segment the goal.
        """
        self._gray_scale(175)
        self._erode()
        self._median()
        self._contours()
        number = self._max_area_components(2)
        self._dilate()
        if number == 2:
            self._reconstruct_two()
        elif number == 3:
            self._reconstruct_goal()
        self._superpose()
