import cv2
import numpy as np
from pyzbar import pyzbar
import pyrealsense2 as rs


def rotate_box(box, angle, center):
    """Rotate a box by a given angle around a given center."""
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_box = cv2.transform(np.array([box]), M)[0]
    return rotated_box


def draw_qr_codes(image, qr_codes):
    """Draw bounding boxes around detected QR codes."""
    for qr in qr_codes:
        polygon = np.array(qr.polygon, dtype=np.int32)
        rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Draw bounding box
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Draw text
        cv2.putText(image, qr.data.decode("utf-8"), tuple(box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)  


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
        qr_codes = pyzbar.decode(frame)

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
