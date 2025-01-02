# Eye Tracking and Gaze Estimation Project

This repository is a personal project focused on eye tracking and gaze estimation using computer vision and deep learning techniques. The system involves data generation, model training, and real-time testing to predict gaze positions based on eye regions. While this project demonstrates functional capabilities, it is a work in progress and may require further optimization and improvements. 🚀

---

## ✨ Description of Components

### 1. **Data Generation** (`eye_tracking_data.py`)
This script captures images and annotations from a webcam to create a labeled dataset for gaze estimation. Key functionalities include:
- 🎥 Webcam feed displays two windows: 
  - **Black Screen**: For capturing mouse clicks and gaze coordinates.
  - **Webcam Feed**: For visualizing the live video and selecting the eye region.
- 🖱️ Interactive eye region cropping with mouse clicks.
- 📂 Organized saving of images and annotations in directories.
- 🖼️ Augmentation of images with brightness, contrast, and sharpness adjustments.

#### 🔄 General Workflow for Data Generation:
1. **Initialization**: 
   - Directories are set up for storing data.
   - Webcam parameters (ex: resolution) are configured.
2. **Data Collection**:
   - The user clicks on the black screen to mark gaze points.
   - Eye regions are manually cropped using an interactive tool.
3. **Augmentation**:
   - Cropped images are enhanced with brightness, contrast, and sharpness variations.
   - Multiple versions of each image are saved for robustness.
4. **Storage**:
   - All data is saved in structured directories (cropped images, full images, annotations).

---

### 2. **Model Training** (`gaze_train.py`)
This script trains a ViT-based model for gaze estimation. Key highlights include:
- **Model Architecture**: 
  - Utilizes the Vision Transformer (`google/vit-base-patch16-224-in21k`).
  - The classifier is modified for predicting gaze coordinates (x, y).
- **Dataset Preparation**:
  - Loads and preprocesses images with normalization.
  - Incorporates augmented data for diversity.
- **Training Process**:
  1. 📊 Splits the dataset into training and validation sets.
  2. 🎯 Uses Mean Squared Error (MSE) loss for regression tasks.
  3. 📈 Implements learning rate scheduling and early stopping for optimal results.
- **Frameworks Used**: 
  - Built with `transformers`, `torch`, `torchvision` and  `scikit-learn`.

---

### 3. **Real-Time Testing** (`real_time_testing.py`)
This script performs real-time gaze estimation using the trained model. Features include:
- **Eye Detection**: 
  - Uses Haar cascades to detect facial and eye regions in the webcam feed.
- **Gaze Prediction**:
  1. Extracts eye regions and preprocesses them for model input.
  2. Predicts normalized gaze coordinates using the trained model.
  3. Maps predictions to screen coordinates.
- **Visualization**: 
  - Displays the predicted gaze point on a black background.
  - Highlights detected eye regions with bounding boxes.

---

## 📝 Notes
This project reflects ongoing exploration and learning in the field of computer vision. Feedback and contributions are welcome to help refine the system further. 💡
