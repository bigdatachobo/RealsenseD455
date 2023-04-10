import cv2
import numpy as np
from pyzbar import pyzbar
import pyrealsense2 as rs

def draw_qr_codes(image, qr_codes):
    for qr in qr_codes:
        polygon = np.array(qr.polygon, dtype=np.int32)
        points = sorted(polygon, key=lambda p: p[0])
        left_points = sorted(points[:2], key=lambda p: p[1])
        right_points = sorted(points[2:], key=lambda p: p[1])
        top_left, bottom_left = left_points
        top_right, bottom_right = right_points

        box = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        cv2.putText(image, qr.data.decode("utf-8"), tuple(box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

def detect_qr_codes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    qr_codes = pyzbar.decode(threshold)
    return qr_codes

def depth_focus(depth_frame, color_frame, threshold=500):
    depth_data = np.asanyarray(depth_frame.get_data())
    color_data = np.asanyarray(color_frame.get_data())

    focused_area = cv2.inRange(depth_data, 0, threshold)
    focused_area = cv2.dilate(focused_area, np.ones((5, 5), np.uint8), iterations=2)
    focused_area = cv2.erode(focused_area, np.ones((5, 5), np.uint8), iterations=1)

    focused_color_image = cv2.bitwise_and(color_data, color_data, mask=focused_area)
    return focused_color_image

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        focused_image = depth_focus(depth_frame, color_frame)
        qr_codes = detect_qr_codes(focused_image)

        draw_qr_codes(focused_image, qr_codes)
        # focused_image = cv2.cvtColor(focused_image, cv2.COLOR_BGR2RGB)

        cv2.imshow("QR Code Detection", focused_image)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows
