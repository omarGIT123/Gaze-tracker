import cv2
import numpy as np

# Test data
test = {
    "head_image": "./cropped/0/head_images_8/eye_48.png",
}

img = cv2.imread(test["head_image"])

if img is None:
    print("Error: Image not found. Check the file path:", test["head_image"])
else:
    # Increase Brightness
    brightness = 20
    img_brighter = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
    for i in range(3):  
        channel = img[:, :, i]
        # Increase brightness only for non-black pixels
        channel_adjusted = np.where(channel > 0, np.clip(channel + brightness, 0, 255), 0)
        img_brighter[:, :, i] = channel_adjusted
    # Increase Contrast
    contrast = 1.5
    img_higher_contrast = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

    # Increase Sharpness
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img_sharper = cv2.filter2D(img, -1, kernel)

    cv2.imshow("Original Image", img)
    cv2.imshow("Brighter Image", img_brighter)
    cv2.imshow("Higher Contrast Image", img_higher_contrast)
    cv2.imshow("Sharper Image", img_sharper)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
