import numpy as np
import pandas as pd
import os

# ============================================================
# LESSON 2: AUDIO FEATURE EXTRACTION
# ============================================================
print("LESSON 2: AUDIO FEATURES")
import librosa

audio_path = 'sample.mp3'

y, sr = librosa.load(audio_path)
duration = len(y) / sr

rms_energy = np.sqrt(np.mean(y**2))
zcr = np.mean(librosa.feature.zero_crossing_rate(y))
spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

audio_features = pd.DataFrame([{
    'Duration': duration,
    'RMS_Energy': rms_energy,
    'Zero_Crossing_Rate': zcr,
    'Spectral_Centroid': spectral_centroid,
}])
print(audio_features)

# ============================================================
# LESSON 3: VIDEO FEATURE EXTRACTION
# ============================================================
# print("\nLESSON 3: VIDEO FEATURES")
import cv2

video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

video_features = pd.DataFrame([{
    'Frame_Count': frame_count,
    'FPS': fps,
    'Duration': frame_count / fps if fps > 0 else 0,
    'Width': width,
    'Height': height,
}])
print(video_features)
cap.release()


#============================================================
# Visualization
#===========================================================
import matplotlib.pyplot as plt

# Audio waveform
plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title('Audio Waveform')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
