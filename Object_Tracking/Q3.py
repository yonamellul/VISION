import numpy as np
import cv2

cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')
thresh = 100


def compute_orientation_and_magnitude(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients along the x and y axis
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180  # Convert radians to degrees
    
    return magnitude, orientation

def mask_orientation_by_magnitude(magnitude, orientation, threshold=thresh):
    # Mask where magnitude is below the threshold
    mask = np.where(magnitude > threshold, 1, 0)
    
    # Create an image to display the result
    display_img = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    
    # Apply the mask to the orientation
    # Map the orientation to a colormap for visualization
    orientation_colored = cv2.applyColorMap((orientation * (255/180)).astype(np.uint8), cv2.COLORMAP_HSV)
    
    # Use the mask to select which pixels to display
    display_img[mask == 1] = orientation_colored[mask == 1]
    
    # Mark masked pixels as red
    display_img[mask == 0] = [0, 0, 255]  # BGR format
    
    return display_img



while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    magnitude, orientation = compute_orientation_and_magnitude(frame)
    orientation_display = mask_orientation_by_magnitude(magnitude, orientation)
    
    # Normalize magnitude for display
    magnitude_display = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_display = magnitude_display.astype(np.uint8)
    magnitude_colored = cv2.applyColorMap(magnitude_display, cv2.COLORMAP_JET)
    
    # Display the results
    cv2.imshow('Original Video', frame)
    cv2.imshow('Orientation Display', orientation_display)
    cv2.imshow('Gradient Magnitude', magnitude_colored)
    #cv2.imshow('Gradient Orientation', cv2.applyColorMap((orientation * (255/180)).astype(np.uint8), cv2.COLORMAP_HSV))
    cv2.imshow('Gradient Orientation', (orientation * (255/180)).astype(np.uint8))

    k = cv2.waitKey(60) & 0xff
    if k == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()
cap.release()
