import os
import sys
from datetime import datetime

from cv2 import cv2

from data_processing import DataProcessing
from deep_learning_processing import DeepLearningProcessing


def process(data_processing):
    """
    Process the data with the classical image processing
    :param data_processing: a data_processing object
    :return: nothing
    """
    if data_processing.image.base_goal_1 == (-1, -1) and data_processing.image.base_goal_2 == (-1, -1):
        return
    elif (data_processing.image.base_goal_1 != (-1, -1) and data_processing.image.base_goal_2 == (-1, -1)) or (
            data_processing.image.base_goal_1 == (-1, -1) and data_processing.image.base_goal_2 != (-1, -1)):
        data_processing.segmentation_post()
    elif data_processing.image.base_goal_1 != (-1, -1) and data_processing.image.base_goal_2 != (-1, -1):
        data_processing.segmentation_goal()


def folder_process(folder):
    """
    Process all images in a folder
    """
    total = 0
    error = 0
    general_begin = datetime.now()
    time = []
    for filename in os.listdir(folder):
        try:
            begin = datetime.now()
            total += 1
            print("Process : " + filename)
            data = DataProcessing(folder, filename)
            deepLearning = DeepLearningProcessing(data)
            deepLearning.predict()
            process(data)
            data.image.save()
            time.append(datetime.now() - begin)
        except Exception as e:
            print("\033[91m" + filename + str(e) + "\033[0m")
            error += 1
            pass

    print("\033[92m" + "Total : " + str(total) + "\033[0m")
    print("\033[91m" + "Error : " + str(error) + "\033[0m")
    print("\033[92m" + "Time : " + str(datetime.now() - general_begin) + "\033[0m")
    print("\033[92m" + "Average time : " + str(sum(time) / len(time)) + "\033[0m")


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
    if sys.argv[1] == "photo":
        data = DataProcessing("data/", sys.argv[2], True)
        deepLearning = DeepLearningProcessing(data)
        deepLearning.predict()
        process(data)
        data.image.save()
    if sys.argv[1] == "folder":
        folder_process("data/")
