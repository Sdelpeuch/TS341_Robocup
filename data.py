import cv2


class Data:
    """
    Data class for storing data.
    """
    def __init__(self, path_folder, path_image):
        self._path_folder = path_folder
        self._path_image = path_image
        self.base_image = cv2.imread(self._path_folder + self._path_image)
        self._width = self.base_image.shape[1]
        self._height = self.base_image.shape[0]
        self.working_image = self.base_image.copy()
        self.base_goal_1 = (0, 0)
        self.base_goal_2 = (0, 0)

    def save(self):
        """
        Saves the current working image in process folder
        """
        cv2.imwrite("process/" + self._path_image, self.working_image)
