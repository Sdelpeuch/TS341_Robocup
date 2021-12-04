import cv2
import matplotlib.pyplot as plt
import numpy as np

from data import Data


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

    def _gray_scale(self):
        """
        Convert the image to gray scale.
        """
        gray = cv2.cvtColor(self.image.working_image, cv2.COLOR_BGR2GRAY)
        ret, self.image.working_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
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
        kernel = np.ones((2, 2), np.uint8)
        self.image.working_image = cv2.dilate(self.image.working_image, kernel)
        self.show_image()

    def _contours(self):
        """
        Detect vertical lines
        """
        self.image.working_image = cv2.Canny(self.image.working_image, 100, 200)
        self.show_image()

    def _max_area(self, num_labels, stats):
        list = []
        for i in range(num_labels):
            list.append(stats[i][4])
        list.sort()
        try:
            return list[-2]
        except IndexError:
            return list[-1]

    def _max_area_components(self):
        output = cv2.connectedComponentsWithStats(self.image.working_image)
        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        area = self._max_area(num_labels, stats)
        for label in range(1, num_labels):
            if stats[label][4] == area:
                mask = np.array(labels, dtype=np.uint8)
                mask[labels == label] = 255
                # [intX, intY, intW, intH] = cv2.boundingRect(mask)
                # print(intX, intY, intW, intH)
                # cv2.rectangle(mask,
                #               (intX, intY),  # upper left corner
                #               (intX + intW, intY + intH),  # lower right corner
                #               255,  # red
                #               0)
                self.image.working_image = mask
                self.show_image()
                return;

    def _superpose(self):
        """
        Superpose the image with the working image.
        """
        ret, mask = cv2.threshold(self.image.working_image, 240, 255, cv2.THRESH_BINARY)
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        new_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)
        new_mask[mask >= 180] = [0, 255, 0]
        self.image.working_image = cv2.addWeighted(self.image.base_image, 0.5, new_mask, 0.5, 0)
        self.show_image()

    def segmentation_goal(self):
        self._gray_scale()
        self._erode()
        self._erode()
        self._median()
        # self._dilate()
        self._contours()
        self._max_area_components()
        self._dilate()
        self._superpose()
