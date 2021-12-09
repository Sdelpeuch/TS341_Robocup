import os

import cv2
import matplotlib.pyplot as plt

from DeepLearningProcessing import DeepLearningProcessing
from Process import Process
from data_processing import DataProcessing


def folder_process(folder):
    """
    Process all images in a folder
    """
    for filename in os.listdir(folder):
        try:
            print(filename)
            data = DataProcessing(folder, filename, False)
            deepLearning = DeepLearningProcessing(data.image)
            deepLearning.predict((150, 300), (-1, -1))
            process = Process(data)
            process.process()
            data.image.save()
        except cv2.error:
            pass
        except AttributeError:
            pass


def video(folder):
    """
    Create a video from a folder of png
    """
    images = []
    for filename in os.listdir(folder):
        images.append(cv2.imread(folder + filename))
    height, width, layers = images[0].shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
    for image in images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    data = DataProcessing("data/", "test.png", False)
    deepLearning = DeepLearningProcessing(data)
    deepLearning.predict()
    print(data.image.base_goal_1, data.image.base_goal_2)
    process = Process(data)
    process.process()
    plt.imshow(data.image.working_image)
    plt.show()
    data.image.save()
    # folder_process("data/")
    # video("process/")
