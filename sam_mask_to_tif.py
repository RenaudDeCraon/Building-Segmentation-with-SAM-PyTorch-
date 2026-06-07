# sam_mask_to_tif.py (ÇOKLU BİNA TIKLAMA DESTEKLİ)
import torch
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor

# --- Config ---
CHECKPOINT_PATH = "segment-anything/models/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "DFC19/train/images/JAX_004_006_RGB.tif"  # Değiştir
SAVE_MASK_PATH = "DFC19/train/masks/JAX_004_006_CLS.tif"  # Değiştir

# --- Load image ---
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- Load SAM model ---
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# --- Kullanıcıdan çoklu tıklama al ---
print("\n🖱️ Binaların üzerine tıkla (birden fazla). Bittiğinde pencereyi kapatmak için q bas.")
clicked = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        print(f"📍 Nokta: ({x}, {y})")

cv2.namedWindow("Görsel")
cv2.setMouseCallback("Görsel", mouse_callback)

while True:
    show = image_bgr.copy()
    for pt in clicked:
        cv2.circle(show, pt, 5, (0, 255, 0), -1)
    cv2.imshow("Görsel", show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()

if not clicked:
    print("❗ Hiç tıklama algılanmadı, çıkılıyor.")
    exit()

# --- Her nokta için maske üret ve birleştir ---
combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

for point in clicked:
    input_point = np.array([point])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    mask = masks[0].astype(np.uint8)
    combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

# --- Maske kaydet ---
cv2.imwrite(SAVE_MASK_PATH, combined_mask * 255)
print(f"\n✅ Maske kaydedildi: {SAVE_MASK_PATH}")