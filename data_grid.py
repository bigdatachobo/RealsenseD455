import cv2
import pyrealsense2 as rs
import numpy as np
from pyzbar.pyzbar import decode

# Define the number of rows and columns in the grid
num_rows, num_cols = 3, 6

# Define the size of the cells in the grid
cell_width, cell_height = None, None

# Define the coordinates of the top-left QR code
qr_x1, qr_y1 = 0, 0

# Define the coordinates of the bottom-right QR code
qr_x2, qr_y2 = 0, 0

def detect_qr_codes(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find QR codes in the image
    qr_codes = decode(gray)
    
    return qr_codes

def draw_grid(image, qr_codes):
    global cell_width, cell_height, qr_x1, qr_y1, qr_x2, qr_y2

    # Compute the size of the cells in the grid
    cell_width = image.shape[1] // num_cols
    cell_height = image.shape[0] // num_rows

    # Find the coordinates of the top-left and bottom-right QR codes
    for qr in qr_codes:
        qr_x, qr_y, qr_width, qr_height = qr.rect
        if qr_x < qr_x1 or qr_y < qr_y1:
            qr_x1, qr_y1 = qr_x, qr_y
        if qr_x + qr_width > qr_x2 or qr_y + qr_height > qr_y2:
            qr_x2, qr_y2 = qr_x + qr_width, qr_y + qr_height

    # Draw rectangle around the grid
    cv2.rectangle(image, (qr_x1, qr_y1), (qr_x2, qr_y2), (0, 0, 255), 2)

    # Draw grid on the image
    for row in range(num_rows):
        for col in range(num_cols):
            # Compute coordinates of the cell
            x1, y1 = col * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height

            # Draw rectangle around the cell
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw grid line on the image
            if col > 0:
                cv2.line(image, (x1, y1), (x1, y2), (0, 255, 0), 1)
            if row > 0:
                cv2.line(image, (x1, y1), (x2, y1), (0, 255, 0), 1)

    # Draw text on the image
    cv2.putText(image, f"Cell width: {cell_width}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f"Cell height: {cell_height}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f"Grid size: {num_rows}x{num_cols}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply the grid and QR code detection
        qr_codes = detect_qr_codes(color_image)
        draw_grid(color_image, qr_codes)

        # Show the images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', color_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
