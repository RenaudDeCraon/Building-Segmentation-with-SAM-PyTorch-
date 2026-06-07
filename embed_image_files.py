# embed_image_base64.py
import os
import json
import base64

from PIL import Image
from io import BytesIO

# === Ayarlar ===
JSON_DIR = "DFC19/train/json_labels"
IMAGE_DIR = "DFC19/train/images"

all_jsons = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
print(f"🔍 {len(all_jsons)} adet JSON bulundu.\n")

for filename in all_jsons:
    json_path = os.path.join(JSON_DIR, filename)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_filename = data.get("imagePath")
    if not image_filename:
        print(f"⚠️  imagePath boş: {filename}")
        continue

    image_path = os.path.join(IMAGE_DIR, image_filename)
    if not os.path.exists(image_path):
        print(f"❌ Görsel bulunamadı: {image_path}")
        continue

    try:
        # Görseli oku ve base64'e çevir
        with open(image_path, "rb") as img_f:
            img_bytes = img_f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        data["imageData"] = img_b64

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"✅ {filename} dosyasına imageData eklendi.")

    except Exception as e:
        print(f"❌ HATA: {filename} - {str(e)}")

print("\n🎉 Tüm JSON dosyalarına imageData başarıyla gömüldü!")
