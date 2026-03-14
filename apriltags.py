import cv2
import numpy as np
from pupil_apriltags import Detector

# ---------------- SETTINGS ----------------

TARGET_ID = 12
TAG_SIZE = 0.204 # meters (measure your real printed tag!)

# Camera intrinsics (approximate — calibrate for better accuracy)
fx = 1000
fy = 1000
cx = 640
cy = 360
camera_params = (fx, fy, cx, cy)
# ------------------------------------------

cap = cv2.VideoCapture(0)

detector = Detector(families="tag36h11")

while True:
    ret, frame = cap.read()
    if not ret:
        break

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=TAG_SIZE)
found = False


for detection in detections:
    if detection.tag_id == TARGET_ID:
        found = True

# Get translation vector
pose = detection.pose_t
x = pose[0][0]
z = pose[2][0] # forward distance

# Robot position relative to tag (tag at 0,0)
robot_x = -x
robot_y = -z

print(f"Robot position (meters):")
print(f"X: {robot_x:.3f}")
print(f"Y: {robot_y:.3f}")
print("---------------------")


# Draw box
corners = detection.corners.astype(int)
for i in range(4):
    pt1 = tuple(corners[i])
    pt2 = tuple(corners[(i + 1) % 4])
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)


    # Draw ID text
    center = tuple(detection.center.astype(int))
    cv2.putText(frame, "ID 31", (center[0] - 20, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not found:
        print("Tag 31 not detected")
        cv2.imshow("AprilTag 31 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
