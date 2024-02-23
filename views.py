from flask import Blueprint, render_template, request, jsonify, redirect, url_for, Response
import random
import numpy as np
import cv2
import mediapipe as mp
import pickle

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

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for web camera

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = camera.read()  # Read the camera frame
            if not success:
                break
            else:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    # Draw face landmarks
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                            )

                    # Right hand
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                            )

                    # Left Hand
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            )

                    # Pose Detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            )
                except:
                    pass

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    camera.release()