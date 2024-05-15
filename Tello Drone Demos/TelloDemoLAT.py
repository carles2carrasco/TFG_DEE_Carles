import time

from ultralytics import YOLO
import cv2
from djitellopy import Tello
import numpy

xError = 0
yError = 0
pid_output = 0
lat = 0
pError = [xError, yError]

w, h = 640, 480

# PID controller parameters

Kp = 0.2
Ki = 0.0
Kd = 0.0
# Initialize PID variables
integral = 0
last_error = 0


# Initialize Tello drone
tello = Tello()
tello.connect()
#tello.streamoff()
tello.streamon()
time.sleep(2)

# OpenCV window
#cv2.namedWindow("Tello Stream", cv2.WINDOW_NORMAL)

# Capture frame from Tello camera


model_path = "yolo-Weights/yolov8n.pt"
model = YOLO(model_path)

tello.send_rc_control(0, 0, 0,0)
tello.takeoff()
while True:

    img = tello.get_frame_read().frame
    img = cv2.resize(img, (640, 480))
    results = model(img, stream=True)

    xError = 0
    yError = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
                xError = ((x1 + x2) / 2) - w/2
                yError = ((y1 + y2) / 2) - h/2


                #print(pError[0], pError[1])

    # Hover if not detecting any
    if pError[0] == 0:
        print("                                                            quieto")
        pError = [0, 0]
    else:
        integral = integral + xError
        derivative = xError - pError[0]

        pid_output = Kp * xError + Ki * integral + Kd * derivative
        #pid_output = int(pid_output)

    print(pid_output)
    lat= numpy.clip(pid_output, -100, 100)
    lat = int(lat)

    # Use tello.send_rc_control() to control the Tello drone
    tello.send_rc_control(lat, 0, 0, 0)

    pError = [xError, yError]

    cv2.imshow('Tello', img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
