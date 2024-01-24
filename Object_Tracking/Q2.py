#Dynamic Histogram Update Mechanism: After every update_rate frames, 
#the script updates the histogram model of the object. It does this 
#by recalculating the histogram for the current RoI and then blending 
#this new histogram with the original histogram to accommodate appearance 
#changes. 

#This dynamic updating approach helps the tracking algorithm stay adaptive
# to changes in the object's appearance, potentially improving tracking 
#accuracy over time. However, it's important to balance the update frequency
# and the blending weights to avoid rapid fluctuations in the model histogram 
#that could destabilize tracking, especially in noisy or dynamic environments.

# histogram update
import numpy as np
import cv2

roi_defined = False
update_interval = 10  # Define an interval (in frames) for histogram update
#cap = cv2.VideoCapture('Test-Videos/Antoine_Mug.mp4')
cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')

 
def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2-r)
        w = abs(c2-c)
        r = min(r, r2)
        c = min(c, c2)  
        roi_defined = True

ret, frame = cap.read()
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

while True:
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    if roi_defined:
        cv2.rectangle(frame, (r, c), (r+h, c+w), (0, 255, 0), 2)
    else:
        frame = clone.copy()
    if key == ord("q"):
        break

track_window = (r, c, h, w)
roi = frame[c:c+w, r:r+h]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 30., 20.)), np.array((180., 255., 235.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

frame_count = 0  # Initialize a frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_image = hsv[:, :, 0]  # Extract the hue image
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    r, c, h, w = track_window
    frame_tracked = cv2.rectangle(frame, (r, c), (r+h, c+w), (255, 0, 0), 2)

    # Update the model histogram at the specified interval
    if frame_count % update_interval == 0:
        roi_update = frame[c:c+h, r:r+w]
        hsv_roi_update = cv2.cvtColor(roi_update, cv2.COLOR_BGR2HSV)
        mask_update = cv2.inRange(hsv_roi_update, np.array((0., 30., 20.)), np.array((180., 255., 235.)))
        roi_hist_update = cv2.calcHist([hsv_roi_update], [0], mask_update, [180], [0, 180])
        cv2.normalize(roi_hist_update, roi_hist_update, 0, 255, cv2.NORM_MINMAX)
        roi_hist = roi_hist_update  # Update the histogram model

    cv2.imshow('Sequence', frame_tracked)
    cv2.imshow('Hue Component', hue_image)  # Display the hue component
    cv2.imshow('Back Projection', dst)  # Display the weight image (back-projection)
    
    k = cv2.waitKey(60) & 0xff
    if k == 27:  # ESC key to exit
        break

    frame_count += 1  # Increment the frame counter

cv2.destroyAllWindows()
cap.release()
