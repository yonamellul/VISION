import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from utils import *
from middlebury import *
from lucas import find_best_window_size

from lucas import *


def lucas_mysine():

    # Step 1: Load the data
    I1_path = '../data/mysine/mysine9.png'
    I2_path = '../data/mysine/mysine10.png'
    gt_flow_path = '../data/mysine/correct_mysine.flo'

    #I1 = rgb2gray(imread(I1_path))
    #I2 = rgb2gray(imread(I2_path))

    I1 = imread(I1_path)
    I2 = imread(I2_path)
    gt_flow = readflo(gt_flow_path)

    # Parameters for optimization
    smallest_window = 5
    biggest_window = 40
    step = 2
    std = 1.5  # Assuming a fixed standard deviation for Gaussian kernel

    # Step 2: Find the best window size
    min_error, best_flow, best_window_size, errors = find_best_window_size(I1, I2, gt_flow, smallest_window, biggest_window, step, std)


    showResults(gt_flow, smallest_window, biggest_window, step, min_error, best_flow, best_window_size, errors)

if __name__ == "__main__":
    lucas_mysine()