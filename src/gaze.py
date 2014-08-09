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

    # Values for scaleFactor = 1.2 and for minNeighbors = 2
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 1)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # --------------------------------------------
        # Detecting pupil using thresholding
        #roi_gray = (255 - roi_gray)
        #roi_gray = cv2.threshold

        # --------------------------------------------
        
        # --------------------------------------------
        # Detecting pupil using Hough circle transform
        # Have to improve this to a better technique
        # Detecting pupil only in window of the eye
        pupils = cv2.HoughCircles(roi_gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 75, param1 = 50, param2 = 13, minRadius = 0, maxRadius = 0)
        
        if pupils is not None:
            for pupil in pupils[0, :]:
                cv2.circle(roi_color, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 2)

                cv2.circle(roi_color, (pupil[0], pupil[1]), 2, (0, 0, 255), 3)
        # ---------------------------------------------

    # Print some message
    cv2.putText(frame, "Gaze vector :", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))
    cv2.putText(frame, "Feature not added yet", (10, 50), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255))

    cv2.imshow("preview", frame)

    ret, frame = vc.read()

    # Exits on Esc
    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()


def find_gaze():
    """
    Calculates the gaze position using the given positions of eye and pupil
    """

    pass
