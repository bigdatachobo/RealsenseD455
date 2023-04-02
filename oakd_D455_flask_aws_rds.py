import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

load_dotenv()

# 설정하려는 거리 (미터 단위)
min_depth = 0.3  # 최소 깊이 (m)
max_depth = 1.0  # 최대 깊이 (m)

app = Flask(__name__)

db_user = os.environ.get('DB_USERNAME')
db_pass = os.environ.get('DB_PASSWORD')
db_host = os.environ.get('DB_HOST')
db_port = os.environ.get('DB_PORT')
db_name = os.environ.get('DB_NAME')

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# DB model for parking spaces
class ParkingSpace(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    is_empty = db.Column(db.Boolean, nullable=False)

    def __init__(self, is_empty):
        self.is_empty = is_empty

db.create_all()

# 카메라 파이프 라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 파이프 라인 시작
pipeline.start(config)

# 주차 공간의 위치를 정의합니다. 예를 들어, 주차 공간의 네모난 바운딩 박스를 정의합니다.
parking_space_positions = [
    # (x, y, width, height)
    # ...
]

def process_parking_spaces(depth_image, parking_space_positions):
    for idx, position in enumerate(parking_space_positions):
        x, y, width, height = position

        img_crop = depth_image[y:y + height, x:x + width]
        count = cv2.countNonZero(img_crop)

        # 여기에서 주차 공간의 비어있음 여부를 결정합니다.
        is_empty = count < 900

        # DB 업데이트
        parking_space = ParkingSpace.query.get(idx)
        if parking_space:
            parking_space.is_empty = is_empty
            db.session.commit()
        else:
            new_parking_space = ParkingSpace(is_empty)
            db.session.add(new_parking_space)
            db.session.commit()

try:
    while True:
        # 프레임 받기
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # 깊이 프레임을 numpy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())

        # 주차 공간을 처리하고 데이터베이스를 업데이트합니다.
        process_parking_spaces(depth_image, parking_space_positions)

        # 나머지 코드는 영상 처리 및 시각화를 위한 코드입니다.
        # 이 부분은 필요에 따라 수정할 수 있습니다.

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

        for idx, position in enumerate(parking_space_positions):
            x, y, width, height = position
            parking_space = ParkingSpace.query.get(idx)
            is_empty = parking_space.is_empty

            if is_empty:
                color = (0, 255, 0)
                thickness = 5
            else:
                color = (0, 0, 255)
                thickness = 2

            # 바운딩 박스와 주차 공간 상태 표시
            cv2.rectangle(depth_frame_colored, (x, y), (x + width, y + height), color, thickness)
            cv2.putText(depth_frame_colored, f"ID: {idx}, Empty: {is_empty}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 깊이 맵 표시
        cv2.imshow("Depth", depth_frame_colored)

        if cv2.waitKey(1) == ord("q"):
            break
finally:
    # 파이프 라인 정리
    pipeline.stop()
    cv2.destroyAllWindows()
