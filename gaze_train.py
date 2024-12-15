import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import json
from torch import nn
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm

model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.config.hidden_size, 2)
)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

normalize = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=0.5, std=0.5)
])

class EyeGazeDataset(Dataset):
    def __init__(self, image_paths, gaze_positions, transforms=None):
        self.image_paths = image_paths
        self.gaze_positions = gaze_positions
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        if self.transforms:
            img = self.transforms(img)
        gaze_position = self.gaze_positions[idx]
        return img, torch.tensor(gaze_position, dtype=torch.float)

def load_data_from_jsonl(jsonl_file):
    images, labels = [], []
    with open(jsonl_file, "r") as f:
        for line in f:
            sample = json.loads(line.strip())
            img_path = sample["head_image"]
            img = cv2.imread(img_path)
            brightness = 20
            img_brighter = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
            contrast = 1.5
            img_higher_contrast = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img_sharper = cv2.filter2D(img, -1, kernel)
            brighter_path = f"augmented/img_brighter/{img_path}"
            contrast_path = f"augmented/img_higher_contrast/{img_path}"
            sharper_path = f"augmented/img_sharper/{img_path}"
            os.makedirs(os.path.dirname(brighter_path), exist_ok=True)
            os.makedirs(os.path.dirname(contrast_path), exist_ok=True)
            os.makedirs(os.path.dirname(sharper_path), exist_ok=True)
            cv2.imwrite(brighter_path, img_brighter)
            cv2.imwrite(contrast_path, img_higher_contrast)
            cv2.imwrite(sharper_path, img_sharper)
            relative_x, relative_y = sample["relative_x"], sample["relative_y"]
            normalized_coords = [relative_x, relative_y]
            images.extend([img_path, brighter_path, contrast_path, sharper_path])
            labels.extend([normalized_coords] * 4)
    return np.array(images), np.array(labels)

images_normal, normal_positions, images_gaze, gaze_positions = [], [], [], []
normal_directory = "cropped/0"
gaze_directory = "cropped/1"
for directory in [normal_directory, gaze_directory]:
    for subdir in os.listdir(directory):
        for file in os.listdir(f"{directory}/{subdir}"):
            if file.endswith(".jsonl"):
                file_path = os.path.join(f"{directory}/{subdir}", file)
                images, labels = load_data_from_jsonl(file_path)
                if directory == normal_directory:
                    images_normal.extend(images)
                    normal_positions.extend(labels)
                else:
                    images_gaze.extend(images)
                    gaze_positions.extend(labels)

all_images = images_normal + images_gaze
all_labels = normal_positions + gaze_positions

X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2)

train_dataset = EyeGazeDataset(X_train, y_train, transforms=normalize)
val_dataset = EyeGazeDataset(X_val, y_val, transforms=normalize)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
criterion = nn.MSELoss()

def compute_accuracy(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 15
patience, best_val_loss, patience_counter = 5, float("inf"), 0
for epoch in range(epochs):
    model.train()
    running_loss, total_samples = 0.0, 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for images, gaze_positions in pbar:
            images, gaze_positions = images.to(device), gaze_positions.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, gaze_positions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(images)
            total_samples += len(images)
            pbar.set_postfix(loss=running_loss / total_samples)

    avg_loss = running_loss / total_samples
    model.eval()
    val_loss, val_samples = 0.0, 0
    with torch.no_grad():
        for images, gaze_positions in val_loader:
            images, gaze_positions = images.to(device), gaze_positions.to(device)
            outputs = model(images).logits
            val_loss += criterion(outputs, gaze_positions).item() * len(images)
            val_samples += len(images)
    avg_val_loss = val_loss / val_samples
    print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

torch.save(model.state_dict(), f"epoch_{epoch+1}_model.pth")
