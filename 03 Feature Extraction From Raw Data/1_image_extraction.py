import numpy as np
import pandas as pd
import cv2

print("LESSON 1: IMAGE FEATURE EXTRACTION (JPG)")
print("=" * 70)

image_path = 'sample.jpg'

# Load image
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Get dimensions
height, width, channels = img_rgb.shape

# Color channel features (RGB order)
red_channel = img_rgb[:, :, 0]
green_channel = img_rgb[:, :, 1]
blue_channel = img_rgb[:, :, 2]

# Brightness and contrast
brightness = img_gray.mean()
contrast = img_gray.std()

# Edge detection
# edges = cv2.Canny(img_gray, 100, 200)
# edge_density = np.count_nonzero(edges) / (height * width)

# Create feature vector
image_features = pd.DataFrame([{
    'Width': width,
    'Height': height,
    'Aspect_Ratio': width / height if height != 0 else 0,
    'Brightness': brightness,
    'Contrast': contrast,
    'Red_Mean': red_channel.mean(),
    'Green_Mean': green_channel.mean(),
    'Blue_Mean': blue_channel.mean(),
    # 'Edge_Density': edge_density,
}])

print(image_features)

print("=" * 70)
