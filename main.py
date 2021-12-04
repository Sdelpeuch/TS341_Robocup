import os

import cv2
from data import Data
from data_processing import DataProcessing

def folder_process(folder):
    """
    Process all images in a folder
    """
    for filename in os.listdir(folder):
        try:
            print(filename)
            data = DataProcessing(folder, filename)
            data.segmentation_goal()
            data.image.save()
        except cv2.error:
            pass
        except AttributeError:
            pass

if __name__ == '__main__':
    data = DataProcessing("data/", "42-rgb.png", True)
    data.segmentation_goal()
    data.image.save()
    # folder_process("data/")