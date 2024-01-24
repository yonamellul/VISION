import cv2
import numpy as np

cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')

# Initialize variables for position and velocity
last_position = None
velocity = (0, 0)
threshold = 50

roi_defined = False
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
                        vote_x, vote_y = int(x - vector[0]), int(y - vector[1])
                        if 0 <= vote_x < accumulator.shape[1] and 0 <= vote_y < accumulator.shape[0]:
                            accumulator[vote_y, vote_x] += 1

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(accumulator)
    detected_center = max_loc

    return detected_center, accumulator


def predict_next_position(last_position, velocity):
    if last_position is not None:
        predicted_position = (last_position[0] + velocity[0], last_position[1] + velocity[1])
        return predicted_position
    return None

def update_model_based_on_detection(r_table, image, detected_center, ref_point, update_radius=20):
    """
    Updates the R-Table based on the current detection.
    
    Args:
    - r_table: The current R-Table.
    - image: The current frame image.
    - detected_center: The detected center point of the object in the current frame.
    - ref_point: The reference point used in the R-Table.
    - update_radius: The radius around the detected center to consider for updates.
    """
    # Define the ROI around the detected center
    x_start = max(0, detected_center[0] - update_radius)
    y_start = max(0, detected_center[1] - update_radius)
    x_end = min(image.shape[1], detected_center[0] + update_radius)
    y_end = min(image.shape[0], detected_center[1] + update_radius)
    roi = image[y_start:y_end, x_start:x_end]

    # Process the ROI similar to initial R-Table construction
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 100, 200)
    grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # Update the R-Table entries based on edges within the ROI
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] != 0:
                global_x, global_y = x + x_start, y + y_start
                angle = int(orientation[y, x])
                vector = (ref_point[0] - global_x, ref_point[1] - global_y)
                
                
                
                if angle not in r_table:
                    r_table[angle] = [vector]
                else:
                    # Average vectors for the given angle
                    existing_vectors = np.array(r_table[angle])
                    new_average_vector = np.mean(np.vstack((existing_vectors, vector)), axis=0)
                    r_table[angle] = [new_average_vector.tolist()]  # Update with the new averaged vector


    return r_table

ret, frame = cap.read()

# Define ROI manually for the first frame (this part remains unchanged)
# Assuming the ROI has been defined and the R-Table constructed

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

    # Perform object detection to find 'detected_center'
    detected_center, _ = hough_transform(frame, r_table, ref_point)

    # Prediction based on last known position and velocity
    # Prediction based on last known position and velocity
    predicted_position = predict_next_position(last_position, velocity)


    # Visualization of predicted position (for demonstration)
    if predicted_position is not None:
        cv2.circle(frame, predicted_position, 5, (0, 0, 255), -1)  # Red circle for predicted position


    # Update the position and velocity if the center is detected
    if detected_center is not None:
        if last_position is not None:
            velocity = (detected_center[0] - last_position[0], detected_center[1] - last_position[1])
        last_position = detected_center

        # Use 'predicted_position' to decide whether to update the model
        # This is a simple check; you might want to add more conditions for robustness
        if predicted_position is None or np.linalg.norm(np.array(detected_center) - np.array(predicted_position)) < threshold:
            r_table = update_model_based_on_detection(r_table, frame, detected_center, ref_point, update_radius=20)

    # Visualization and interaction code here...
    if detected_center:
        cv2.circle(frame, detected_center, 5, (0, 255, 0), -1)
    cv2.imshow('Detected', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()