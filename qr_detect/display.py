import cv2


def draw_qr_codes(image, candidates):
    """
    Draw bounding boxes around detected QR codes.

    Args:
        image (numpy.ndarray): Input color image.
        candidates (List): Detected QR code candidates.
    """

    for candidate in candidates:
        rect = cv2.minAreaRect(candidate)
        box = cv2.boxPoints(rect)
        box = box.astype('int')
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
