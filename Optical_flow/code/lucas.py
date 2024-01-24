import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gradhorn import gradhorn
from middlebury import readflo
from utils import angular_error, endPoint_error, relative_norm_error
from skimage.color import rgb2gray


def lucas_kanade(I1,I2,n):
    Ix, Iy, It = gradhorn(I1,I2)
    W=np.zeros((I1.shape[0],I1.shape[1],2))
    
    half_window = n // 2

    for i in range(half_window, I1.shape[0] - half_window):
      for j in range(half_window, I1.shape[1] - half_window):
          Ix_window = Ix[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
          Iy_window = Iy[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
          It_window = It[i-half_window:i+half_window+1, j-half_window:j+half_window+1].flatten()
    
          A = np.vstack((Ix_window, Iy_window)).T
          b = -It_window
    
          if np.linalg.det(A.T @ A) != 0:
              nu = np.linalg.pinv(A.T @ A) @ A.T @ b
              W[i,j,0] = nu[0]
              W[i,j,1] = nu[1]
    
    return W

def find_best_window_size(I1, I2, gt_flow, smallest_window, biggest_window, step, std):
    
    best_window_size = {}
    errors = {
        "mean": {"Angular error": [], "EndPoint error": [], "Relative norm error": []},
        "std": {"Angular error": [], "EndPoint error": [], "Relative norm error": []},
    }
    min_error = {}
    best_flow = {}  
    for window_size in np.arange(smallest_window, biggest_window+1, step):
            print(f"Processing window size: {window_size}")
            #flow = lucas_gaussian(I1, I2, window_size, std)
            flow = lucas_kanade(I1, I2, window_size)

            ang_error = angular_error(gt_flow, flow)
            epe_error = endPoint_error(gt_flow, flow)
            norm_error = relative_norm_error(gt_flow, flow)
            
            mean_ang_error = np.mean(ang_error)
            mean_epe_error = np.mean(epe_error)
            mean_norm_error = np.mean(norm_error)
            
            errors["mean"]["Angular error"].append(mean_ang_error)
            errors["std"]["Angular error"].append(np.std(ang_error))
            errors["mean"]["EndPoint error"].append(mean_epe_error)
            errors["std"]["EndPoint error"].append(np.std(epe_error))
            errors["mean"]["Relative norm error"].append(mean_norm_error)
            errors["std"]["Relative norm error"].append(np.std(norm_error))

            if window_size == smallest_window:
                min_error["Angular error"] = mean_ang_error
                best_flow["Angular error"] = flow
                best_window_size["Angular error"] = window_size

                min_error["EndPoint error"] = mean_epe_error
                best_flow["EndPoint error"] = flow
                best_window_size["EndPoint error"] = window_size

                min_error["Relative norm error"] = np.abs(mean_norm_error)
                best_flow["Relative norm error"] = flow
                best_window_size["Relative norm error"] = window_size
            else:
                if mean_ang_error < min_error["Angular error"]:
                    min_error["Angular error"] = mean_ang_error
                    best_flow["Angular error"] = flow
                    best_window_size["Angular error"] = window_size

                if mean_epe_error < min_error["EndPoint error"]:
                    min_error["EndPoint error"] = mean_epe_error
                    best_flow["EndPoint error"] = flow
                    best_window_size["EndPoint error"] = window_size

                if np.abs(mean_norm_error) < min_error["Relative norm error"]:
                    min_error["Relative norm error"] = np.abs(mean_norm_error)
                    best_flow["Relative norm error"] = flow
                    best_window_size["Relative norm error"] = window_size
    
    return min_error, best_flow, best_window_size, errors

