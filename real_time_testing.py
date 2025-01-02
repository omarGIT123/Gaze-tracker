import cv2
import numpy as np
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch import nn


window_width = 1920
window_height = 1080
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)
model.classifier = nn.Linear(
    model.config.hidden_size, 2
)  # Adjust classifier for (x, y)
model.load_state_dict(torch.load("best_model.pth"))  
model.eval()  # Set to evaluation mode


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

black_screen = np.zeros((window_height, window_width, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 2:  
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            x_min = min(ex1, ex2) + x - 60
            y_min = min(ey1, ey2) + y - 40
            x_max = max(ex1 + ew1, ex2 + ew2) + x + 60
            y_max = max(ey1 + eh1, ex2 + eh2) + y + 40

            cropped_eyes = frame[y_min:y_max, x_min:x_max]
            cropped_eyes_resized = cv2.resize(cropped_eyes, (224, 224))

            cropped_eyes_rgb = cv2.cvtColor(cropped_eyes_resized, cv2.COLOR_BGR2RGB)
            pixel_values = feature_extractor(
                images=cropped_eyes_rgb, return_tensors="pt"
            ).pixel_values

            with torch.no_grad():
                outputs = model(pixel_values)
                gaze_position = outputs.logits.squeeze(
                    0
                ).numpy() 

            # Normalize gaze prediction to screen coordinates
            x_gaze = int(gaze_position[0] * window_width)
            y_gaze = int(gaze_position[1] * window_height)

            black_screen = np.zeros(
                (window_height, window_width, 3), dtype=np.uint8
            ) 
            cv2.circle(black_screen, (x_gaze, y_gaze), 10, (0, 255, 0), -1)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            black_background = np.zeros_like(frame)
            black_background[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]
            cv2.imshow("Eye Area", black_background)

    cv2.imshow("Gaze Position", black_screen)
    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
