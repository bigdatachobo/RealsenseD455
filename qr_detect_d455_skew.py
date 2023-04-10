import cv2
import numpy as np
from pyzbar import pyzbar
import pyrealsense2 as rs


# def draw_qr_codes(image, qr_codes):
#     """Draw bounding boxes around detected QR codes."""
#     for qr in qr_codes:
#         polygon = np.array(qr.polygon, dtype=np.int32)

#         # Get the 4 points of the QR code polygon
#         points = sorted(polygon, key=lambda p: (p[1], p[0]))
#         top_left, bottom_left, bottom_right, top_right = points

#         # Draw bounding box
#         box = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
#         cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

#         # Draw text
#         cv2.putText(image, qr.data.decode("utf-8"), tuple(box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

def draw_qr_codes(image, qr_codes):
    """Draw bounding boxes around detected QR codes."""
    for qr in qr_codes:
        polygon = np.array(qr.polygon, dtype=np.int32)

        # Get the 4 points of the QR code polygon and sort them by x-coordinate
        points = sorted(polygon, key=lambda p: p[0])

        # Separate the points into two groups by y-coordinate
        left_points = sorted(points[:2], key=lambda p: p[1])
        right_points = sorted(points[2:], key=lambda p: p[1])

        # Reconstruct the box points
        top_left, bottom_left = left_points
        top_right, bottom_right = right_points

        # Draw bounding box
        box = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Draw text
        cv2.putText(image, qr.data.decode("utf-8"), tuple(box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)



def detect_qr_codes(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to enhance the QR code pattern
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Detect QR codes in the thresholded image
    qr_codes = pyzbar.decode(threshold)

    return qr_codes




# Initialize camera capture object
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming with the configuration
pipeline.start(config)

try:
    # QR code detection loop
    while True:
        # Wait for a coherent color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert color frame to BGR format
        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect QR codes in the frame
        # qr_codes = pyzbar.decode(frame)
        qr_codes = detect_qr_codes(frame)

        # Draw bounding boxes and text on the frame
        draw_qr_codes(frame, qr_codes)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show the frame
        cv2.imshow("QR Code Detection", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

    # Close all windows
    cv2.destroyAllWindows()

