import cv2
import numpy as np
import pyrealsense2 as rs

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

        focused_image = depth_focus(depth_frame, color_frame)

        cv2.imshow('Focused Image', focused_image)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
