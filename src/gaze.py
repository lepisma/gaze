"""
The main script
The script creates a trace of gaze points
"""

import cv2
import numpy as np

# There are two classifiers in res folder
eye_cascade = cv2.CascadeClassifier("../res/haarcascade_eye_tree_eyeglasses.xml")

vc = cv2.VideoCapture(0)

ret, frame = vc.read()

while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Values for scaleFactor = 1.2 and for minNeighbors = 1
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 1)

    if len(eyes) != 0:
        eyes = [eyes[0]]

    kernel = np.ones((3, 3), np.uint8)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # --------------------------------------------
        # Detecting pupil using thresholding
        #roi_gray = (255 - roi_gray)
        #ret_val, roi_gray = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #roi_gray = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        #roi_gray = cv2.erode(roi_gray, kernel, iterations = 1)
        #contours, hierarchy = cv2.findContours(roi_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #cv2.drawContours(roi_color, contours, -1, (0, 255, 0), 3)

        # --------------------------------------------
        
        frame[0 : h, 0 : w] = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        # --------------------------------------------
        # Detecting pupil using Hough circle transform
        # Have to improve this to a better technique
        # Detecting pupil only in window of the eye
        # pupils = cv2.HoughCircles(roi_gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 75, param1 = 50, param2 = 13, minRadius = 0, maxRadius = 0)
        
        # if pupils is not None:
        #     for pupil in pupils[0, :]:
        #         cv2.circle(roi_color, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 2)
        
        #         cv2.circle(roi_color, (pupil[0], pupil[1]), 2, (0, 0, 255), 3)
        # ---------------------------------------------

    # Print some message
    #cv2.putText(frame, "Gaze vector :", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))
    #cv2.putText(frame, "Feature not added yet", (10, 50), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255))

    cv2.imshow("preview", frame)

    ret, frame = vc.read()

    # Exits on Esc
    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()


def find_gaze(x, y, w, h, a, b):
    """
    Calculates the gaze position using the given positions of eye and pupil
    x, y, w, h define the bounding rectangle
    a, b define the center of pupil
    """

    center = [x + 0.5 * w, y + 0.5 * h]

    return [center[0] - a, center[1] - b]
