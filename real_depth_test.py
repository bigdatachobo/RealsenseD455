import cv2
import numpy as np
import pyrealsense2 as rs

# 설정하려는 거리 (미터 단위)
target_distance = 1.0

# 거리 허용 오차 (미터 단위)
distance_tolerance = 0.05

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

        # 타겟 거리에 있는 물체를 찾기 위해 깊이 이미지를 이진화
        target_mask = cv2.inRange(depth_image, (target_distance - distance_tolerance) * 1000, (target_distance + distance_tolerance) * 1000)

        # 결과를 보여주기 위해 이미지를 8비트로 변환하고 컬러 맵 적용
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        target_colormap = cv2.applyColorMap(cv2.convertScaleAbs(target_mask, alpha=0.03), cv2.COLORMAP_JET)

        # 결과 이미지를 화면에 표시
        cv2.imshow("Depth Image", depth_colormap)
        cv2.imshow("Target Objects", target_colormap)

        # 'q'키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 파이프 라인 정리
    pipeline.stop()
    cv2.destroyAllWindows()
