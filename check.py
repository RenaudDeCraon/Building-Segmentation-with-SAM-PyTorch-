import cv2
import matplotlib.pyplot as plt

mask = cv2.imread("DFC19/train/masks/JAX_004_006_CLS.tif", cv2.IMREAD_GRAYSCALE)
plt.imshow(mask, cmap='gray')
plt.title("Kaydedilen Maske")
plt.show()