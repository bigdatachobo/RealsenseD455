import cv2
import numpy as np
import pyrealsense2 as rs

# 설정하려는 거리 (미터 단위)
min_depth = 0.3  # 최소 깊이 (m)
max_depth = 1.0  # 최대 깊이 (m)

# 카메라 파이프 라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 파이프 라인 시작
pipeline.start(config)

try:
    while True:
        # 프레임 받기
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # 깊이 프레임을 numpy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())

        # 스케일링된 깊이 데이터 계산
        scaled_depth_data = np.clip(depth_image, min_depth * 1000, max_depth * 1000)

        # OpenCV를 사용하여 깊이 맵을 표시
        depth_frame_colored = cv2.normalize(scaled_depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 빨간색에서 파란색으로 변환하는 컬러 맵 생성
        depth_frame_colored = cv2.applyColorMap(255 - depth_frame_colored, cv2.COLORMAP_JET)

        # min_depth 밑의 값을 흰색, max_depth 밖의 값을 회색으로 표시하는 마스크 생성
        too_close_mask = depth_image < min_depth * 1000
        too_far_mask = depth_image > max_depth * 1000

        # 마스크를 깊이 맵에 적용
        depth_frame_colored[too_close_mask] = [255, 255, 255]  # 흰색
        depth_frame_colored[too_far_mask] = [128, 128, 128]    # 회색

        cv2.imshow("Depth", depth_frame_colored)

        if cv2.waitKey(1) == ord("q"):
            break
finally:
    # 파이프 라인 정리
    pipeline.stop()
    cv2.destroyAllWindows()
