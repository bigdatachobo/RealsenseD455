import cv2
import numpy as np

def focus_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def auto_focus(frame):
    best_focus_score = -1
    best_frame = frame.copy()
    
    for focus_distance in range(100, 1000, 100):  # Adjust the range and step size as needed
        # Adjust the camera focus distance here using the RealSense SDK or OpenCV
        # (Note: RealSense SDK does not support focus adjustment, you may need to use a different camera API)
        
        # Capture a new frame after adjusting the focus
        ret, new_frame = cap.read()
        
        score = focus_score(new_frame)
        if score > best_focus_score:
            best_focus_score = score
            best_frame = new_frame.copy()

    return best_frame

# Replace this with the RealSense D455 capture initialization
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    focused_frame = auto_focus(frame)

    cv2.imshow('Focused Frame', focused_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
