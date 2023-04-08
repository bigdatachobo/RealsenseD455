from flask import Flask, render_template, Response
import cv2
import pyrealsense2 as rs
import numpy as np
from qr_detection1 import detect_qr_codes, draw_grid

app = Flask(__name__)

def gen_frames():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Apply the grid and QR code detection
        qr_codes = detect_qr_codes(color_image)
        draw_grid(color_image, qr_codes)

        # Encode image to jpg
        ret, buffer = cv2.imencode('.jpg', color_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
