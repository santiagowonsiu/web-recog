from flask import Blueprint, render_template, request, jsonify, redirect, url_for, Response
import random
import numpy as np
import cv2

#### SETUP
views = Blueprint(__name__, "views")

@views.route('/') # return html
def home():
    return render_template('index.html', name = "Santiago", age= 20)

@views.route("go-to-home") # redirect to another route
def go_to_home():
    return redirect(url_for('views.home')) # redirect to home route

#### 3D OBJECTS

@views.route('/objects3d', methods=["GET"])
def generate_objects(num_objects=1):
    objects = []
    for _ in range(num_objects):
        shape = np.random.choice(["sphere", "cube"])
        size = np.random.uniform(0.1, 1.0)
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        z = np.random.uniform(-1.0, 1.0)
        color = np.random.rand(3) * 255
        obj = {
            "shape": shape,
            "size": size,
            "x": x,
            "y": y,
            "z": z,
            "color": color.tolist()
        }
        objects.append(obj)
    return jsonify(objects)

#### POSE CLASSIFICATION

@views.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for web camera

    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    camera.release()