from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from qr_detection1 import detect_qr_codes, draw_grid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Initialize variables
camera = None
num_rows, num_cols = 3, 6
cell_width, cell_height = None, None
qr_x1, qr_y1, qr_x2, qr_y2 = 0, 0, 0, 0

def gen_frames():
    global camera, num_rows, num_cols, cell_width, cell_height, qr_x1, qr_y1, qr_x2, qr_y2
    while True:
        # Read frame from camera
        success, frame = camera.read()
        if not success:
            break
        else:
            # Apply the grid and QR code detection
            qr_codes = detect_qr_codes(frame)
            draw_grid(frame, qr_codes, num_rows, num_cols, cell_width, cell_height, qr_x1, qr_y1, qr_x2, qr_y2)

            # Encode the frame as an image buffer
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Emit the frame to the client
            socketio.emit('video_frame', {'image_data': frame})

def start_camera():
    global camera, num_rows, num_cols, cell_width, cell_height, qr_x1, qr_y1, qr_x2, qr_y2
    # Initialize the camera
    camera = cv2.VideoCapture(0)

    # Wait for the camera to warm up
    for i in range(30):
        camera.read()

    # Get the size of the camera image
    success, frame = camera.read()
    if success:
        height, width, channels = frame.shape
    else:
        height, width = 480, 640

    # Compute the size of the cells in the grid
    cell_width = width // num_cols
    cell_height = height // num_rows

    # Set the coordinates of the top-left and bottom-right QR codes
    qr_x1, qr_y1 = 0, 0
    qr_x2, qr_y2 = width, height

def stop_camera():
    global camera
    if camera is not None:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def on_connect():
    print('Client connected')
    start_camera()
    emit('video_params', {'rows': num_rows, 'cols': num_cols})

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')
    stop_camera()

@socketio.on('start_video')
def on_start_video():
    print('Starting video')
    global camera
    if camera is not None:
        socketio.start_background_task(gen_frames)

@socketio.on('stop_video')
def on_stop_video():
    print('Stopping video')

if __name__ == '__main__':
    socketio.run(app, debug=True ,port=3000)
