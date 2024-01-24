import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from utils import *
from middlebury import *
from lucas import find_best_window_size

from lucas import *


def lucas_rubic():

    # Step 1: Load the data
    I1_path = '../data/rubic/rubic9.png'
    I2_path = '../data/rubic/rubic10.png'
    I3_path = '../data/rubic/rubic_lucas.png'


    I1 = imread(I1_path)
    I2 = imread(I2_path)
    I3 = imread(I3_path)
    fig, axs = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True, sharex=True, sharey=True)
    
    i = 0
    for n in range(5, 25, 2): 
        w = lucas_kanade(I1, I2, n)
        step = 5

        # Calculate row and column for current subplot
        row, col = divmod(i, 5)
        
        ax = axs[row, col]  # Correct way to access the subplot
        #X, Y = np.meshgrid(np.arange(0, w.shape[1], step), np.arange(0, w.shape[0], step))
        #ax.quiver(X, Y, w[Y, X, 0], -w[Y, X, 1], color='r', alpha = 0.2,scale = 7)
        
        # Show the flow visualization
        ax.imshow(computeColor(w, True))
        ax.set_title(f'n = {n}')
        i += 1

    plt.show()


if __name__ == "__main__":
    lucas_rubic()

    
    