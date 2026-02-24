from ultralytics import YOLO
import cv2
import numpy as np

# ===== LOAD YOLO MODEL (THIS FIXES YOUR ERROR) =====
model = YOLO("yolov8n.pt")

# ===== SEATBELT DETECTION =====
def seatbelt_detected(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if 20 < angle < 70:
                return True
    return False


# ===== MAIN DETECTION FUNCTION =====
def detect_violations(frame):
    results = model(frame, stream=True)

    person = False
    helmet = False
    phone = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person = True
            elif label == "helmet":
                helmet = True
            elif label == "cell phone":
                phone = True

    if phone:
        return "TC-102 : Mobile Phone Usage"

    if person and not helmet:
        return "TC-101 : Helmet Not Worn"

    if person and not seatbelt_detected(frame):
        return "TC-103 : Seatbelt Not Worn"

    return "TC-000 : No Violation"
