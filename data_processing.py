"""Ce fichier comporte tous les traitements nécessaires à la partie de détection et segmentation des buts intervenant
après la reconnaissance de la base des poteaux grâce au réseau de neurones. """
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from cv2.cv2 import mean

from data import Data

matplotlib.use('TkAgg')


def _max_area(num_labels, stats, number):
    """
    Find in stats the biggest area of the number
    :param num_labels: the number of labels
    :param stats: the output of cv2.connectedComponentsWithStats
    :param number: the number of the area we want to find
    :return: and index (1: 1 post, 2: 2 post, 3: goal) and the area(s)
    """
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
    This class is used to process the data with classical image treatment.
    """

    def __init__(self, path_folder, path_image, debug=False):
        self.image = Data(path_folder, path_image)
        self.debug = debug

    def show_image(self):
        """
        Show the image if debug is enable
        """
        if self.debug:
            plt.imshow(self.image.working_image, cmap='gray')
            plt.show()

    def _gray_scale(self, threshold):
        """
        Convert the image to gray scale then binarize it with a threshold.
        :return: update self.image.working_image with the binarized image
        """
        self.image.working_image = cv2.cvtColor(self.image.working_image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(self.image.working_image, cv2.COLOR_BGR2GRAY)
        ret, self.image.working_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        self.show_image()

    def _median(self):
        """
        Apply median filter to the image with a k size of 5.
        :return:  update self.image.working_image with the median filtered image
        """
        self.image.working_image = cv2.medianBlur(self.image.working_image, 5)
        self.show_image()

    def _erode(self):
        """
        Apply erosion to the image with a kernel (2,2).
        :return: update self.image.working_image with the eroded image
        """
        kernel = np.ones((2, 2), np.uint8)
        self.image.working_image = cv2.erode(self.image.working_image, kernel)
        self.show_image()

    def _dilate(self):
        """
        Apply dilation to the image with a kernel (2,2).
        :return: update self.image.working_image with the dilated image
        """
        kernel = np.ones((5, 5), np.uint8)
        self.image.working_image = cv2.dilate(self.image.working_image, kernel)
        self.show_image()

    def _contours(self):
        """
        Find edges in the image with the Canny algorithm.
        :return: update self.image.working_image with the edges image
        """
        self.image.working_image = cv2.Canny(self.image.working_image, 100, 200)
        self.show_image()

    def _max_area_components(self, number):
        """
        Take an image with edges and find the biggest area with cv2.connectedComponentsWithStats.
        :param number: the number of the area we want to find
        :return: update self.image.working_image with the biggest area(s) and return the number of the area(s)
        """
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

    def _line_detection(self, threshold):
        """
        Detect first horizontal line according to the threshold.
        :param threshold: limit for the detection of the line
        :return: return the y position of the line
        """

        y_position = 0
        for line in range(self.image.working_image.shape[0]):
            if mean(self.image.working_image[line])[0] >= threshold * 255 / self.image.working_image.shape[1]:
                return line
        return y_position

    def _column_detection(self, y_position):
        """
        Detect the left and right column of a line
        :param y_position: the position of the line
        :return: the left and right column of the line
        """
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

    def _reconstruct_one(self, number, threshold):
        """
        With the original image, the position of post and the working image, reconstruct one posts of this image
        :param number: 1: left post, 2: right post
        :return: update self.image.working_image with the reconstructed post
        """
        ret, self.image.working_image = cv2.threshold(self.image.working_image, 240, 255, cv2.THRESH_BINARY)
        y_position = self._line_detection(threshold)
        x_left, x_right = self._column_detection(y_position)
        x_position = int((x_left + x_right) / 2)
        width = 40

        if number == 1:
            if abs(x_position - self.image.base_goal_1[1]) < 0.2 * self.image.width:
                cv2.line(self.image.working_image, self.image.base_goal_1, (x_position, y_position), 255, int(width))
        elif number == 2:
            if abs(x_position - self.image.base_goal_2[1]) < 0.2 * self.image.width:
                cv2.line(self.image.working_image, self.image.base_goal_2, (x_position, y_position), 255, int(width))

    def _reconstruct_two(self):
        """
        With the working image as a mask, find the two posts then reconstruct the goal of this image
        :return: update self.image.working_image with the reconstructed posts
        """
        sub_image_1 = self.image.working_image[0: self.image.working_image.shape[0],
                      0: self.image.working_image.shape[1] // 2]
        sub_image_2 = self.image.working_image[0: self.image.working_image.shape[0],
                      self.image.working_image.shape[1] // 2: self.image.working_image.shape[1]]
        self.image.working_image = sub_image_1
        self._reconstruct_one(1, 2)
        sub_image_1 = self.image.working_image
        self.image.working_image = sub_image_2
        self.image.base_goal_2 = (self.image.base_goal_2[0] - self.image.width // 2, self.image.base_goal_2[1])
        self._reconstruct_one(2, 2)
        sub_image_2 = self.image.working_image
        self.image.base_goal_2 = (self.image.base_goal_2[0] + self.image.width // 2, self.image.base_goal_2[1])
        self.image.working_image = np.zeros((self.image.working_image.shape[0], self.image.working_image.shape[1] * 2))
        self.image.working_image[0: sub_image_1.shape[0], 0: sub_image_1.shape[1]] = sub_image_1
        self.image.working_image[0: sub_image_2.shape[0],
        sub_image_1.shape[1]: sub_image_1.shape[1] + sub_image_2.shape[1]] = sub_image_2
        self.show_image()

    def _reconstruct_goal(self):
        """
        With the working image as a mask, find the highest line and the two posts then reconstruct the goal of this image
        :return: update self.image.working_image with the reconstructed goal
        """
        self._reconstruct_two()
        y_position = self._line_detection(50)
        cv2.line(self.image.working_image, (self.image.base_goal_1[0], y_position),
                 (self.image.base_goal_2[0], y_position), 255, int(0.01 * self.image.width))

    def _superpose(self):
        """
        Take the original image, the predict image and the working image and superpose them
        :return: update self.image.working_image with the superposed image
        """
        new_mask = np.zeros(self.image.working_image.shape, dtype=np.uint8)
        new_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2RGB)
        new_mask[self.image.working_image >= 180] = [0, 255, 0]
        self.image.working_image = cv2.addWeighted(self.image.base_image, 0.5, new_mask, 0.5, 0)
        self.image.working_image = cv2.addWeighted(self.image.working_image, 0.5, self.image.predict_image, 0.5, 0)
        self.image.working_image = cv2.cvtColor(self.image.working_image, cv2.COLOR_BGR2RGB)
        self.show_image()

    def _uncrop(self):
        """
        Take the working image as a mask (with the biggest area and the reconstruct the goal of this image) and the coordinate of the crop in the base image and create a new image at the same size of the base image where the mask is superposed at the right position.
        :return:
        """
        ret, mask = cv2.threshold(self.image.working_image, 240, 255, cv2.THRESH_BINARY)
        empty_image = np.zeros((self.image.height, self.image.width), dtype=np.uint8)
        try:
            empty_image[0: self.image.cropped_coordinates[1],
            self.image.cropped_coordinates[0]: self.image.cropped_coordinates[0] + mask.shape[1]] = mask
        except ValueError:
            empty_image[0: self.image.cropped_coordinates[1],
            self.image.cropped_coordinates[0] - 1: self.image.cropped_coordinates[0] + mask.shape[1]] = mask
        self.image.working_image = empty_image

    def segmentation_post(self):
        """
        Entire treatment for one post
        """
        self._gray_scale(130)
        self._erode()
        self._erode()
        self._median()
        self._contours()
        self._max_area_components(1)
        self._dilate()
        self._uncrop()
        self._reconstruct_one(1, 2)
        self._superpose()

    def segmentation_goal(self):
        """
        Entire treatment for two posts or the goal
        """
        self._gray_scale(150)
        self._erode()
        self._median()
        self._contours()
        number = self._max_area_components(2)
        self._dilate()
        self._uncrop()
        if number == 2:
            self._reconstruct_two()
        elif number == 3:
            self._reconstruct_goal()
        self._superpose()
