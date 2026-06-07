# convert_masks_to_json.py
import os
import cv2
import json

# --- Config ---
MASKS_DIR = "DFC19/train/masks"
OUTPUT_DIR = "DFC19/train/json_labels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- İşlem Başlat ---
all_masks = [f for f in os.listdir(MASKS_DIR) if f.endswith(".tif") or f.endswith(".png")]
print(f"Toplam {len(all_masks)} maske bulundu. İşleniyor...\n")

for mask_file in all_masks:
    mask_path = os.path.join(MASKS_DIR, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 127).astype("uint8")

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    json_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": mask_file.replace("_CLS.tif", "_RGB.tif"),
        "imageData": None,
        "imageHeight": mask.shape[0],
        "imageWidth": mask.shape[1]
    }

    for contour in contours:
        if cv2.contourArea(contour) < 30:
            continue
        points = contour.squeeze().tolist()
        if isinstance(points[0], int):  # Tek noktaysa
            continue
        points = [[float(x), float(y)] for [x, y] in points]
        json_data["shapes"].append({
            "label": "building",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

    json_name = mask_file.replace("_CLS.tif", ".json").replace(".png", ".json")
    with open(os.path.join(OUTPUT_DIR, json_name), "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"✅ {json_name} oluşturuldu")

print("\n🎉 Tüm maskeler başarıyla JSON formatına dönüştürüldü!")
