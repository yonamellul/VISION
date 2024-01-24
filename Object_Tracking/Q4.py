import cv2
import numpy as np

roi_defined = False
#cap = cv2.VideoCapture('Test-Videos/Antoine_Mug.mp4')
cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')


def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True
          

def build_r_table(image, roi):
    r, c, w, h = roi
    roi_image = image[c:c+h, r:r+w]
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 100, 200)

    grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    M = cv2.moments(edges)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    r_table = {}
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] != 0:
                angle = int(orientation[y, x])
                vector = (cx - x, cy - y)
                if angle not in r_table:
                    r_table[angle] = []
                r_table[angle].append(vector)

    return r_table, (cx + r, cy + c)

def hough_transform(image, r_table, ref_point):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    accumulator = np.zeros(image.shape[:2])
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] != 0:
                angle = int(orientation[y, x])
                if angle in r_table:
                    for vector in r_table[angle]:
                        vote_x, vote_y = x - vector[0], y - vector[1]
                        if 0 <= vote_x < accumulator.shape[1] and 0 <= vote_y < accumulator.shape[0]:
                            accumulator[vote_y, vote_x] += 1

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(accumulator)
    detected_center = max_loc

    return detected_center, accumulator

ret, frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r,c,h,w)
roi = (r,c,w,h)

r_table, ref_point = build_r_table(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_center, accumulator = hough_transform(frame, r_table, ref_point)
    cv2.circle(frame, detected_center, 5, (0, 255, 0), -1)

    cv2.imshow('Detected', frame)
    cv2.imshow('Accumulator', np.uint8(255 * accumulator / np.max(accumulator)))

    if cv2.waitKey(30) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
