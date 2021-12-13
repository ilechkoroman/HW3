import cv2
import numpy as np
from util import get_cursor, get_speed, F, B, H, Q, R
from kalman_filter import CursorSystem

"""
For some reason I was not able to run the video capture from jup notebook, that why I have created
the main file which demonstrate the work of Kalman filter
"""

# for reading the video the base code was taken from
# https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
# and after refactored a bit

cap = cv2.VideoCapture('example.avi')
if not cap.isOpened():
    print("Error opening video stream or file")
previous_sensor_data, speed = None, None
frame_idx = 0
cs = CursorSystem(F, B, H, Q, R)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    th, noise, cursor_x, cursor_y = get_cursor(frame, 13, 15)

    cursor_coord = np.array([[cursor_y, cursor_x]])
    sensor_data = np.vstack((noise, cursor_coord))
    idx = np.random.choice(sensor_data.shape[0], 1)
    sensor_data = sensor_data[idx]

    speed = get_speed(previous_sensor_data, sensor_data)
    previous_sensor_data = sensor_data

    display_frame = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    if speed is None:
        # if it is first frame we skip
        frame_idx += 1
        continue

    x = np.vstack((sensor_data, speed)).reshape((4, 1))
    # first we need to init the values
    if frame_idx == 1:
        cs.init_vars(x)
    else:
        # make prediction and correct with the sensor data
        cs.prediction_step()
        cs.correction_step(x.reshape((4, 1)))
        res = cs.get_result()
        # draw results
        display_frame = cv2.circle(display_frame, (res[1, 0], res[0, 0]), radius=0,
                                   color=(0, 0, 255), thickness=5)

    cv2.imshow('Frame', display_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()

cv2.destroyAllWindows()
