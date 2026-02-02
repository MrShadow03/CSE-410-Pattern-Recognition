# ============================================================
# LESSON 4 & 5: NUMERIC DATA AND COMBINED FEATURE EXTRACTION
# ============================================================
# Goal: Extract features from numeric data and combine all features
# Outcome: Create complete feature vectors for pattern recognition

import numpy as np
import pandas as pd
import os

print("=" * 70)
print("LESSON 4: NUMERIC DATA FEATURE ENGINEERING")
print("=" * 70)

print("\n4.1 Creating Features from Numeric Data:")
print("-" * 70)

# Example: Raw sensor data
raw_data = [22.5, 23.1, 22.8, 23.5, 24.2, 23.9, 24.5, 25.1, 24.8, 25.3]
print(f"Raw Sensor Data: {raw_data}")

# Derive features
numeric_features = {
    'Mean': np.mean(raw_data),
    'Std_Dev': np.std(raw_data),
    'Min': np.min(raw_data),
    'Max': np.max(raw_data),
    'Range': np.max(raw_data) - np.min(raw_data),
    'Median': np.median(raw_data),
    'Variance': np.var(raw_data),
    'Skewness': pd.Series(raw_data).skew(),
    'Kurtosis': pd.Series(raw_data).kurtosis()
}

print("\nDerived Numeric Features:")
for feature, value in numeric_features.items():
    print(f"  - {feature}: {value:.4f}")

print("\n4.2 Extracting Time-Series Features:")
print("-" * 70)

# Calculate rate of change
differences = np.diff(raw_data)
time_series_features = {
    'Rate_of_Change_Mean': np.mean(differences),
    'Rate_of_Change_Std': np.std(differences),
    'Max_Increase': np.max(differences),
    'Max_Decrease': np.min(differences),
    'Volatility': np.std(differences)
}

print(f"Differences: {differences}")
print("\nTime-Series Features:")
for feature, value in time_series_features.items():
    print(f"  - {feature}: {value:.4f}")

print("\n" + "=" * 70)
print("NUMERIC EXTRACTION SUMMARY")
print("=" * 70)
print("""
KEY CONCEPTS:
✓ Statistical Features: Mean, Std Dev, Min, Max, Median, Variance
✓ Distribution Features: Skewness, Kurtosis
✓ Time-Series Features: Rate of change, Volatility, Trends
✓ Domain Features: Features specific to your problem domain
✓ Aggregation: Combine multiple statistics into feature vector

USE CASES:
- Sensor data analysis (temperature, pressure)
- Stock price prediction (financial data)
- Heart rate monitoring (health data)
- System performance monitoring
""")

print("\n" + "=" * 70)
print("LESSON 5: COMBINING ALL FEATURES INTO A DATASET")
print("=" * 70)

print("\n5.1 Creating a Complete Feature Dataset:")
print("-" * 70)

# Example combining multiple data types
sample_id = 1

# Check if sample files exist
image_path = '../sample.jpg'
audio_path = '../sample.mp3'
video_path = '../sample.mp4'

# Combine all extracted features
all_features = {
    'Sample_ID': sample_id,
    'Data_Type': 'Mixed',
}

# Add image features (if available)
if os.path.exists(image_path):
    print("✓ Image data available")
    try:
        from PIL import Image
        img = Image.open(image_path)
        img_array = np.array(img.convert('RGB'))
        img_gray = np.array(img.convert('L'))
        
        image_data = {
            'Image_Width': img.size[0],
            'Image_Height': img.size[1],
            'Image_Aspect_Ratio': img.size[0] / img.size[1] if img.size[1] != 0 else 0,
            'Image_Brightness': img_gray.mean(),
            'Image_Contrast': img_gray.std(),
            'Image_Red_Mean': img_array[:, :, 0].mean(),
            'Image_Green_Mean': img_array[:, :, 1].mean(),
            'Image_Blue_Mean': img_array[:, :, 2].mean(),
        }
        all_features.update(image_data)
    except Exception as e:
        print(f"  Note: Could not process image - {e}")
else:
    print("✗ Image data not found")

# Add audio features (if available and librosa is installed)
if os.path.exists(audio_path):
    print("✓ Audio data available")
    try:
        import librosa
        y, sr = librosa.load(audio_path)
        
        audio_data = {
            'Audio_Duration': len(y) / sr,
            'Audio_Sampling_Rate': sr,
            'Audio_RMS_Energy': np.sqrt(np.mean(y**2)),
            'Audio_Zero_Crossing_Rate': np.mean(librosa.feature.zero_crossing_rate(y)),
            'Audio_Spectral_Centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        }
        all_features.update(audio_data)
    except ImportError:
        print("  Note: librosa not installed. Install with: pip install librosa")
    except Exception as e:
        print(f"  Note: Could not process audio - {e}")
else:
    print("✗ Audio data not found")

# Add video features (if available and cv2 is installed)
if os.path.exists(video_path):
    print("✓ Video data available")
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_data = {
            'Video_Frame_Count': frame_count,
            'Video_FPS': fps,
            'Video_Duration': frame_count / fps if fps > 0 else 0,
            'Video_Width': width,
            'Video_Height': height,
            'Video_Resolution': width * height,
            'Video_Aspect_Ratio': width / height if height != 0 else 0,
        }
        all_features.update(video_data)
        cap.release()
    except ImportError:
        print("  Note: opencv-python not installed. Install with: pip install opencv-python")
    except Exception as e:
        print(f"  Note: Could not process video - {e}")
else:
    print("✗ Video data not found")

# Add numeric features (always available)
print("✓ Numeric data available")
all_features.update(numeric_features)
all_features.update({'Numeric_' + k: v for k, v in time_series_features.items()})

combined_df = pd.DataFrame([all_features])
print("\n5.2 Combined Feature DataFrame:")
print("-" * 70)
print(combined_df.T)  # Transpose for better readability

print("\n5.3 Feature Scaling and Normalization:")
print("-" * 70)

print("""
Before using features in machine learning models, normalize them:

Option 1: Standardization (Z-score normalization)
  formula: (x - mean) / std_dev
  result: mean = 0, std_dev = 1

Option 2: Min-Max Scaling
  formula: (x - min) / (max - min)
  result: values between 0 and 1

Option 3: Mean Normalization
  formula: (x - mean) / (max - min)
  result: values between -1 and 1
""")

# Example normalization
print("\nExample: Normalizing numeric features")
df_numeric = pd.DataFrame(numeric_features, index=[0])
print("\nOriginal features:")
print(df_numeric)

df_normalized = (df_numeric - df_numeric.mean()) / df_numeric.std()
print("\nNormalized features (Z-score):")
print(df_normalized)

print("\n" + "=" * 70)
print("COMPLETE FEATURE EXTRACTION WORKFLOW")
print("=" * 70)

print("""
STEP-BY-STEP PROCESS:

1. DATA COLLECTION
   └─ Gather raw data (images, audio, video, sensors, etc.)

2. FEATURE EXTRACTION
   ├─ Image: Color, brightness, contrast, shape
   ├─ Audio: Energy, spectral, MFCC
   ├─ Video: Properties, frame samples
   └─ Numeric: Statistics, time-series

3. FEATURE ENGINEERING
   └─ Combine features, create derived features

4. FEATURE NORMALIZATION
   └─ Scale features to comparable ranges

5. FEATURE SELECTION
   └─ Remove redundant/irrelevant features

6. MACHINE LEARNING
   └─ Use feature vectors for classification/regression

KEY PRINCIPLES:
✓ Extract meaningful features specific to your domain
✓ Remove correlated features to reduce dimensionality
✓ Normalize features before training models
✓ Document what each feature represents
✓ Validate features improve model performance
""")

print("\n" + "=" * 70)
print("SUMMARY: FEATURE EXTRACTION FOR PATTERN RECOGNITION")
print("=" * 70)

print("""
✓ IMAGE FEATURES:
  Brightness, Contrast, Color (RGB), Shape

✓ AUDIO FEATURES:
  Energy, Zero Crossing Rate, Spectral Centroid, MFCC

✓ VIDEO FEATURES:
  FPS, Duration, Resolution, Frame Features

✓ NUMERIC FEATURES:
  Statistical (mean, std), Temporal (rate of change)

✓ COMBINED:
  Merge all features into single feature vector

→ Use normalized feature vectors for machine learning models!
""")

print("\n" + "=" * 70)
print("INSTALLATION TIPS:")
print("=" * 70)
print("""
Required packages:
  pip install numpy pandas pillow

Optional packages for advanced features:
  pip install librosa           # For audio feature extraction
  pip install opencv-python    # For video feature extraction
  pip install scikit-learn      # For machine learning models
""")

print("\n" + "=" * 70)
