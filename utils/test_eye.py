import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf


window_width = 1920
window_height = 1080

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 1)
    for x, y, w, h in faces:  
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 2:  
           
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            x_min = min(ex1, ex2) + x - 60
            y_min = min(ey1, ey2) + y - 40
            x_max = max(ex1 + ew1, ex2 + ew2) + x + 60
            y_max = max(ey1 + eh1, ey2 + eh2) + y + 40

            black_background = np.zeros_like(frame)
            black_background[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]

            cv2.imshow("Eye Area", black_background)          
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2,
            )
              
    cv2.imshow("Eye Detection", frame)
  
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
