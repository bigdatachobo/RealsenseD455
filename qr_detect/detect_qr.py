import cv2
import numpy as np


def detect_qr_candidates(image):
    """Detect QR code candidates in an image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and aspect ratio
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            rect = cv2.minAreaRect(contour)
            width = rect[1][0]
            height = rect[1][1]
            if 0.8 <= min(width, height) / max(width, height) <= 1.2:
                candidates.append(contour)

    return candidates


def decode_qr_code(image, contour):
    """Decode a QR code within a contour in an image."""
    # Create a mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Apply mask to the grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)

    # Detect and decode the QR code
    decoded, points, _ = cv2.QRCodeDetector().detectAndDecode(masked)
    if decoded:
        return decoded, points


