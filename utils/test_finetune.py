import cv2
import numpy as np
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch import nn

model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)


model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)
model.classifier = nn.Linear(
    model.config.hidden_size, 2
)  # Adjust classifier for (x, y)
model.load_state_dict(torch.load("epoch_15_model.pth")) 
model.eval()

test = {
    "relative_x": 0.4427083333333333,
    "relative_y": 0.4648148148148148,
    "head_image": "cropped/0/head_images_8/eye_48.png",
    "full_image": "full/full_images_8/full_48.png",
    "eye_behaviour": 0,
}

image_path = test["head_image"]
label = [test["relative_x"], test["relative_y"]]  

cropped_eyes = cv2.imread(image_path)
if cropped_eyes is None:
    print(f"Failed to load image from {image_path}")
    exit()

cropped_eyes_resized = cv2.resize(cropped_eyes, (224, 224))

cropped_eyes_rgb = cv2.cvtColor(cropped_eyes_resized, cv2.COLOR_BGR2RGB)

pixel_values = feature_extractor(
    images=cropped_eyes_rgb, return_tensors="pt"
).pixel_values

with torch.no_grad():
    outputs = model(pixel_values)
    gaze_position = outputs.logits.squeeze(0).numpy()  

window_width = 1920
window_height = 1080

x_gaze = int(gaze_position[0] * window_width)
y_gaze = int(gaze_position[1] * window_height)  

black_screen = np.zeros(
    (window_height, window_width, 3), dtype=np.uint8
) 

cv2.circle(black_screen, (x_gaze, y_gaze), 10, (0, 255, 0), -1)
cv2.circle(
    black_screen,
    (int(label[0] * window_width), int(label[1] * window_height)),
    10,
    (255, 255, 0),
    -1,
) 

cv2.imshow("Gaze Position", black_screen)

cv2.waitKey(0)
cv2.destroyAllWindows()
