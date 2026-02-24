from flask import Flask, render_template, Response
import cv2
from detection import detect_violations
import os
import time

app = Flask(__name__)

# Camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera not accessible")
    exit()

# Evidence folder
EVIDENCE_PATH = "evidence/violations"
os.makedirs(EVIDENCE_PATH, exist_ok=True)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        violation = detect_violations(frame)

        # Save evidence if violation
        if violation != "TC-000 : No Violation":
            filename = f"{EVIDENCE_PATH}/{violation.replace(':','')}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)

        cv2.putText(frame, violation, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Starting RoadSafe AI...")
    app.run(debug=True)
