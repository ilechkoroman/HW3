import cv2
import numpy as np

# state transition model
F = np.array([[1, 0, 0.2, 0],
              [0, 1, 0, 0.2],
              [0, 0, 1,   0],
              [0, 0, 0,   1]])
B = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# observation model
H = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0]])
# uncertainty matrix
Q = np.array([[0, 0, 0,   0],
              [0, 0, 0,   0],
              [0, 0, 0.1, 0],
              [0, 0, 0, 0.1]])
# sensor noise
R = np.array([[0.1, 0, 0, 0],
              [0, 0.1, 0, 0],
              [0, 0, 0.1, 0],
              [0, 0, 0, 0.1]])


def get_cursor(img, sigma, noise_count):
    # as input take the recorded video and find the coords of cursor
    if img is not None:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cursor_y, cursor_x = np.unravel_index(np.argmax(th), th.shape)

    # add some noise
    noise = np.random.normal(0, sigma, (noise_count, 2))
    noise = noise + np.array([cursor_y, cursor_x])
    noise = noise.astype(int)
    for n in noise:
        try:
            th[n[0], n[1]] = 255
        except IndexError:
            pass

    return th, noise, cursor_x, cursor_y


def get_speed(previous_sensor_data, sensor_data):
    if previous_sensor_data is None:
        return None
    # as a time unit I take one iteration
    return previous_sensor_data - sensor_data
