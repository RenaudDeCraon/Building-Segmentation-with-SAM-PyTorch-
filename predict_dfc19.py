# predict_dfc19_v3.py
import torch
import cv2
import numpy as np
import os
import json
from torchvision import transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A

# --- Config ---
MODEL_PATH = "checkpoints/unet_dfc19_sam_masks.pth"
IMAGE_PATH = "test/test-image/OMA_374_011_RGB.tif"  # Tahmin yapılacak görüntü
OUTPUT_MASK_PATH = "test/result-mask/pred_mask.png"
OUTPUT_JSON_PATH = "test/result-json/pred_mask.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Transform ---
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

# --- Load image ---
orig_image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
orig_size = (orig_image.shape[1], orig_image.shape[0])
aug = transform(image=image_rgb)
input_tensor = aug["image"].unsqueeze(0).to(DEVICE)

# --- Load model ---
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# --- Predict ---
with torch.no_grad():
    pred = model(input_tensor)
    pred = torch.sigmoid(pred)
    pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

# --- Resize to original size ---
pred_mask_resized = cv2.resize(pred_mask, orig_size, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(OUTPUT_MASK_PATH, pred_mask_resized * 255)
print(f"✅ Tahmin maskesi kaydedildi: {OUTPUT_MASK_PATH}")

# --- Export to LabelMe JSON ---
contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

labelme_data = {
    "version": "5.2.1",
    "flags": {},
    "shapes": [],
    "imagePath": IMAGE_PATH,
    "imageData": None,
    "imageHeight": pred_mask_resized.shape[0],
    "imageWidth": pred_mask_resized.shape[1],
}

for contour in contours:
    if cv2.contourArea(contour) < 30:
        continue
    points = contour.squeeze().tolist()
    points = [[float(x), float(y)] for [x, y] in points]

    if len(points) >= 3:
        labelme_data["shapes"].append({
            "label": "building",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(labelme_data, f, indent=4)

print(f"✅ LabelMe JSON dosyası oluşturuldu: {OUTPUT_JSON_PATH}")
