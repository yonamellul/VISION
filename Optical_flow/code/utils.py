import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from middlebury import *


def load_image(image_path):
    return np.array(Image.open(image_path).convert('L'),dtype=np.float32)


def relative_norm_error(I_gt, I_hat, eps=1e-10):
  return (np.linalg.norm(I_gt) - np.linalg.norm(I_hat)) / (np.linalg.norm(I_gt) + eps)

def endPoint_error(I_gt, I_hat):
  return np.linalg.norm(I_gt - I_hat)

def relative_endPoint_error(I_gt, I_hat, eps=1e-10):
  return (np.linalg.norm(I_gt - I_hat)) / (np.linalg.norm(I_gt) + eps)

def angular_error(I_gt, I_hat):
    # Flatten the flow arrays to shape (num_pixels, 2)
    I_gt_flat = I_gt.reshape(-1, 2)
    I_hat_flat = I_hat.reshape(-1, 2)
    
    # Compute the dot product and norms for each vector
    dot_product = np.sum(I_gt_flat * I_hat_flat, axis=1)
    norm_I_gt = np.linalg.norm(I_gt_flat, axis=1)
    norm_I_hat = np.linalg.norm(I_hat_flat, axis=1)
    
    # Compute the cosine of the angle between vectors
    cos_angle = (dot_product + 1) / (norm_I_gt * norm_I_hat + 1)
    # Ensure the cosine values are within the valid range [-1, 1] for arccos
    cos_angle = np.clip(cos_angle, -1, 1)
    
    # Return the angular error in radians for each vector
    return np.arccos(cos_angle)



def quiver(flow,title,scale,step=5,eps = 0.0000000001):
    plt.figure(figsize=(10,5))
    
    plt.title(title)
    
    if(scale):
        norm = np.sqrt(flow[:,:,0]**2 + (flow[:,:,1])**2)
        flow[:,:,0] /= (norm + eps)
        flow[:,:,1] /= (norm + eps)
    
    plt.quiver(np.arange(0,flow.shape[1],step), 
               np.arange(flow.shape[0], 0,-step), 
               flow[::step,::step, 0], 
               -flow[::step,::step, 1])
    
    #plt.show()

def showResults(gt_flow, smallest_window, biggest_window, step, min_error, best_flow, best_window_size, errors ):
        
        # Step 3: Plot the evolution of each error according to the window size
        window_sizes = np.arange(smallest_window, biggest_window+1, step)
        for error_type in errors["mean"]:
            plt.figure(figsize=(10, 6))
            plt.plot(window_sizes, errors["mean"][error_type], label=f"Mean {error_type}")
            plt.fill_between(window_sizes, 
                            np.array(errors["mean"][error_type]) - np.array(errors["std"][error_type]),
                            np.array(errors["mean"][error_type]) + np.array(errors["std"][error_type]),
                            alpha=0.2, label=f"Std {error_type}")
            plt.xlabel('Window Size')
            plt.ylabel('Error Value')
            plt.title(f'{error_type} vs. Window Size')
            plt.legend()
            #plt.show()

        # Step 4: Compute optical flow with the best window size found and display the velocity map
        for error_type, flow in best_flow.items():
            print(f"Best window size for minimizing {error_type}: {best_window_size[error_type]}")
            color_map = computeColor(flow)
            plt.figure(figsize=(10, 6))
            quiver(flow,f"Computed Flow using best window size for {error_type}", scale = False )
            plt.imshow(color_map, cmap='seismic')
            plt.title(f"Computed Flow using best window size for {error_type}")
            #plt.show()

        # Display the ground truth velocity map
        gt_color_map = computeColor(gt_flow)
        plt.figure(figsize=(10, 6))
        quiver(gt_flow,"Ground Truth Flow", scale=False )
        plt.imshow(gt_color_map, cmap='seismic')
        plt.title("Ground Truth Flow")
        plt.show()


