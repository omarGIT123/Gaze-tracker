import json
import cv2
import numpy as np
import os

window_width = 1920
window_height = 1080

eye_behaviour = 0  # 0: normal, 1: scope hand, 2: wide open

# Dark area dimensions (proportional to webcam resolution)
dark_area_width = 1920
dark_area_height = 1080

collected_data = []
box_data = []
index = 0
with open("index.txt", "r") as f:
    index = int(f.read())
# Ensure output directories exist
output_dir_head = f"cropped/{eye_behaviour}/head_images_{index}"
output_dir_coor = f"cropped/{eye_behaviour}/click_coor_{index}"
output_dir_full = f"full/{eye_behaviour}/full_images_{index}"
output_dir_full_box = f"full/{eye_behaviour}/full_box_{index}"
os.makedirs(output_dir_head, exist_ok=True)
os.makedirs(output_dir_coor, exist_ok=True)
os.makedirs(output_dir_full, exist_ok=True)
os.makedirs(output_dir_full_box, exist_ok=True)


def mouse_callback(event, x, y, flags, param):
    global collected_data, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate relative coordinates
        relative_x = x / dark_area_width
        relative_y = y / dark_area_height
        
        roi = cv2.selectROI(
            "Crop Eye Region", frame, fromCenter=False, showCrosshair=True
        )
        cv2.destroyWindow("Crop Eye Region")

        # Crop the eye region from the frame
        eye_region = frame[
            int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
        ]

        # Create a black background with the same size as the eye region
        black_background = np.zeros_like(frame)

        # Replace the eye region in the black background with the cropped eye region
        black_background[
            int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
        ] = eye_region

        head_filename = f"{output_dir_head}/eye_{len(collected_data)}.png"
        full_image = f"{output_dir_full}/full_{len(collected_data)}.png"
        cv2.imwrite(head_filename, black_background)
        cv2.imwrite(full_image, frame)

        box_data.append(
            {
                "x_min": int(roi[0]),
                "y_min": int(roi[1]),
                "x_max": int(roi[0] + roi[2]),
                "y_max": int(roi[1] + roi[3]),
                "full_image": full_image,
            }
        )
        collected_data.append(
            {
                "relative_x": relative_x,
                "relative_y": relative_y,
                "head_image": head_filename,
                "full_image": full_image,
                "eye_behaviour": eye_behaviour,
            }
        )

        print(f"Saved head image: {head_filename}")
        print(f"Coordinates: ({relative_x:.2f}, {relative_y:.2f}) saved.")


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

cv2.namedWindow("Dark Area")
cv2.setMouseCallback("Dark Area", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    dark_area = np.zeros((dark_area_height, dark_area_width, 3), dtype=np.uint8)

    cv2.imshow("Dark Area", dark_area)
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


for box in box_data:
    with open(f"{output_dir_full_box}/full_box_{len(collected_data)}.jsonl", "a") as f:
        f.write(json.dumps(box) + "\n")
    print(f"Box Data: {box}")
for data in collected_data:
    with open(f"{output_dir_coor}/coor_{len(collected_data)}.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")
    print(f"Coordinates: ({data['relative_x']:.2f}, {data['relative_y']:.2f})")
    print(f"Head Image Path: {data['head_image']}")

index += 1

with open("index.txt", "w") as f:
    f.write(str(index))
