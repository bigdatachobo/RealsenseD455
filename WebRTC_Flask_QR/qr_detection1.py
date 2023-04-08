import cv2
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
