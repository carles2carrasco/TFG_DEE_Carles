import time
from ultralytics import YOLO
import cv2
import numpy
from Dron import Dron
from pymavlink import mavutil


xError = 0
yError = 0
pid_output = 0
lat = 0
pError = [xError, yError]

w, h = 640, 480

# PID controller parameters

Kp = 0.005
Ki = 0.0
Kd = 0.0
# Initialize PID variables
integral = 0
last_error = 0

# Capture frame from camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
model_path = "yolo-Weights/yolov8n.pt"
model = YOLO(model_path)

# Initialize drone in Guided mode
dron = Dron()
connection_string = 'tcp:127.0.0.1:5763'
baud = 115200
print('voy a conectarme')
dron.connect(connection_string, baud, id = 1)
print('conectado')
dron.arm()
print('armado')
dron.takeOff(3)
dron.startGo()
dron.fixHeading()

while True:

    success, img = cap.read()
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
    #lat= numpy.clip(pid_output, -10, 10)
    lat = pid_output
    lat = int(lat)

    # Use to control the drone

    if lat>0:
        dron.direction="Right"
        #dron.go("Right")
    else:
        dron.direction="Left"
        #dron.go("Left")

    print(dron.direction)

    '''''''''
    #0b0000111111111000
    msg = mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111111000,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        0, lat, 0,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    dron.cmd=msg
    dron.vehicle.mav.send(dron.cmd)
    '''''''''

    dron.changeNavSpeed(abs(lat))
    #dron.navSpeed= abs(lat)

    pError = [xError, yError]

    cv2.imshow('Dron', img)
    if cv2.waitKey(1) == ord('q'):
        break
dron.stopGo()
dron.Land()
cv2.destroyAllWindows()
