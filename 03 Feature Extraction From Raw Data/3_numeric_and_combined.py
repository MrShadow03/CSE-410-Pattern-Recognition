import numpy as np
import pandas as pd
import os

# ============================================================
# LESSON 4: NUMERIC DATA FEATURE ENGINEERING
# ============================================================
print("LESSON 4: NUMERIC DATA FEATURES")

raw_data = [22.5, 23.1, 22.8, 23.5, 24.2, 23.9, 24.5, 25.1, 24.8, 25.3]

numeric_features = {
    'Mean': np.mean(raw_data),
    'Std_Dev': np.std(raw_data),
    'Min': np.min(raw_data),
    'Max': np.max(raw_data),
    'Median': np.median(raw_data),
    'Range': np.max(raw_data) - np.min(raw_data),
}

print(pd.DataFrame([numeric_features]))

# ============================================================
# LESSON 5: COMBINING ALL FEATURES
# ============================================================
print("\nLESSON 5: COMBINED FEATURES")

all_features = {'Sample_ID': 1}
all_features.update(numeric_features)


import cv2
img = cv2.imread('sample.jpg')
if img is not None:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_features.update({
        'Image_Width': img.shape[1],
        'Image_Height': img.shape[0],
        'Image_Brightness': img_gray.mean(),
    })



import librosa
y, sr = librosa.load('sample.mp3')
all_features.update({
    'Audio_Duration': len(y) / sr,
    'Audio_RMS_Energy': np.sqrt(np.mean(y**2)),
})



import cv2
cap = cv2.VideoCapture('sample.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
all_features.update({
    'Video_Frame_Count': frame_count,
    'Video_FPS': fps,
    'Video_Duration': frame_count / fps if fps > 0 else 0,
})
cap.release()


combined_df = pd.DataFrame([all_features])
print(combined_df)
