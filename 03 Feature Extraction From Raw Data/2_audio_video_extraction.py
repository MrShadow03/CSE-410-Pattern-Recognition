# ============================================================
# LESSON 2 & 3: AUDIO AND VIDEO FEATURE EXTRACTION
# ============================================================
# Goal: Extract numerical features from audio (MP3) and video (MP4) files
# Outcome: Convert media files into usable feature vectors

import numpy as np
import pandas as pd
import os

print("=" * 70)
print("LESSON 2: AUDIO FEATURE EXTRACTION (MP3)")
print("=" * 70)

import librosa

audio_path = 'sample.mp3'

if os.path.exists(audio_path):
    print(f"✓ Found: {audio_path}\n")
    
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    print(f"Sampling Rate: {sr} Hz")
    print(f"Duration: {len(y) / sr:.2f} seconds")
    print(f"Total Samples: {len(y)}\n")
    
    # Extract features
    rms_energy = np.sqrt(np.mean(y**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    print("FEATURES:")
    print(f"  RMS Energy: {rms_energy:.4f}")
    print(f"  Zero Crossing Rate: {zcr:.4f}")
    print(f"  Spectral Centroid Mean: {np.mean(spectral_centroid):.2f} Hz")
    print(f"  MFCC Coefficients: {np.mean(mfcc, axis=1)}\n")
    
    audio_features = {
        'Duration': len(y) / sr,
        'Sampling_Rate': sr,
        'RMS_Energy': rms_energy,
        'Zero_Crossing_Rate': zcr,
        'Spectral_Centroid_Mean': np.mean(spectral_centroid),
        'Spectral_Centroid_Std': np.std(spectral_centroid),
        'MFCC_Mean': np.mean(mfcc)
    }
    
    audio_df = pd.DataFrame([audio_features])
    print("AUDIO FEATURE VECTOR:")
    print(audio_df)

print("\n" + "=" * 70)
print("LESSON 3: VIDEO FEATURE EXTRACTION (MP4)")
print("=" * 70)

import cv2

video_path = '../sample.mp4'

if os.path.exists(video_path):
    print(f"✓ Found: {video_path}\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Frames: {frame_count} | FPS: {fps:.2f} | Duration: {duration:.2f}s")
    print(f"Resolution: {width}x{height}\n")
    
    # Extract features from sample frames
    frame_indices = [0, frame_count // 2, frame_count - 1]
    frame_labels = ["First", "Middle", "Last"]
    frame_features_list = []
    
    for idx, label in zip(frame_indices, frame_labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_b, frame_g, frame_r = cv2.split(frame)
            
            frame_feature = {
                f'Frame_{label}_Brightness': np.mean(frame_gray),
                f'Frame_{label}_Red_Mean': np.mean(frame_r),
                f'Frame_{label}_Green_Mean': np.mean(frame_g),
                f'Frame_{label}_Blue_Mean': np.mean(frame_b),
            }
            frame_features_list.append(frame_feature)
            
            print(f"{label} Frame - Brightness: {frame_feature[f'Frame_{label}_Brightness']:.2f}")
    
    print()
    
    video_features = {
        'Frame_Count': frame_count,
        'FPS': fps,
        'Duration': duration,
        'Width': width,
        'Height': height,
        'Resolution': width * height,
        'Aspect_Ratio': width / height if height != 0 else 0
    }
    
    # Add frame features
    for frame_dict in frame_features_list:
        video_features.update(frame_dict)
    
    video_df = pd.DataFrame([video_features])
    print("VIDEO FEATURE VECTOR:")
    print(video_df)
    
    cap.release()

print("\n" + "=" * 70)
