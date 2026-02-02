# ============================================================
# FEATURE EXTRACTION FROM RAW DATA
# ============================================================
# Goal: Extract numerical features from raw data (images, audio, video)
# Outcome: Convert raw data into usable feature values

import numpy as np
import pandas as pd
from PIL import Image
import os

print("=" * 70)
print("FEATURE EXTRACTION FROM RAW DATA - BEGINNER'S GUIDE")
print("=" * 70)

print("\n" + "=" * 70)
print("LESSON 1: IMAGE FEATURE EXTRACTION (JPG)")
print("=" * 70)

print("\n1.1 Loading and Understanding Images:")
print("-" * 70)

# Path to sample image
image_path = '../sample.jpg'

if os.path.exists(image_path):
    print(f"✓ Found image: {image_path}\n")
    
    # Load image using PIL
    img = Image.open(image_path)
    print(f"Image size (width, height): {img.size}")
    print(f"Image format: {img.format}")
    print(f"Image mode: {img.mode}")  # RGB, RGBA, etc.
    
    # Convert to NumPy array for analysis
    img_array = np.array(img)
    print(f"\nImage shape: {img_array.shape}")
    if len(img_array.shape) == 3:
        height, width, channels = img_array.shape
        print(f"Height: {height}, Width: {width}, Channels: {channels}")
    else:
        print(f"Grayscale image with shape: {img_array.shape}")
    
    print("\n1.2 Extracting Color Features:")
    print("-" * 70)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
    
    # Extract color channel features
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]
    
    print(f"Red channel - Mean: {red_channel.mean():.2f}, Std: {red_channel.std():.2f}")
    print(f"Green channel - Mean: {green_channel.mean():.2f}, Std: {green_channel.std():.2f}")
    print(f"Blue channel - Mean: {blue_channel.mean():.2f}, Std: {blue_channel.std():.2f}")
    
    # Create color feature vector
    color_features = np.array([
        red_channel.mean(), red_channel.std(),
        green_channel.mean(), green_channel.std(),
        blue_channel.mean(), blue_channel.std()
    ])
    print(f"\nColor Feature Vector: {color_features}")
    
    print("\n1.3 Extracting Brightness and Contrast:")
    print("-" * 70)
    
    # Convert to grayscale for brightness analysis
    img_gray = img.convert('L')
    img_gray_array = np.array(img_gray)
    
    brightness = img_gray_array.mean()
    contrast = img_gray_array.std()
    min_intensity = img_gray_array.min()
    max_intensity = img_gray_array.max()
    
    print(f"Brightness (Mean): {brightness:.2f}")
    print(f"Contrast (Std Dev): {contrast:.2f}")
    print(f"Min Intensity: {min_intensity}")
    print(f"Max Intensity: {max_intensity}")
    print(f"Dynamic Range: {max_intensity - min_intensity}")
    
    print("\n1.4 Extracting Shape Features:")
    print("-" * 70)
    
    # Image dimensions
    height, width = img_gray_array.shape
    aspect_ratio = width / height if height != 0 else 0
    area = height * width
    
    print(f"Image Height: {height}")
    print(f"Image Width: {width}")
    print(f"Aspect Ratio (width/height): {aspect_ratio:.2f}")
    print(f"Total Pixels (Area): {area}")
    
    print("\n1.5 Creating Complete Image Feature Vector:")
    print("-" * 70)
    
    image_features = {
        'Width': width,
        'Height': height,
        'Aspect_Ratio': aspect_ratio,
        'Brightness': brightness,
        'Contrast': contrast,
        'Min_Intensity': min_intensity,
        'Max_Intensity': max_intensity,
        'Red_Mean': red_channel.mean(),
        'Red_Std': red_channel.std(),
        'Green_Mean': green_channel.mean(),
        'Green_Std': green_channel.std(),
        'Blue_Mean': blue_channel.mean(),
        'Blue_Std': blue_channel.std()
    }
    
    image_df = pd.DataFrame([image_features])
    print("\nImage Feature DataFrame:")
    print(image_df)
    
else:
    print(f"✗ Image not found at: {image_path}")
    print("Please ensure sample.jpg is in the parent directory\n")

print("\n" + "=" * 70)
print("LESSON 2: AUDIO FEATURE EXTRACTION (MP3)")
print("=" * 70)

print("\n2.1 Understanding Audio Files:")
print("-" * 70)

audio_path = '../sample.mp3'

try:
    # Try to import audio libraries
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Note: librosa not installed. Install with: pip install librosa")

if os.path.exists(audio_path) and LIBROSA_AVAILABLE:
    print(f"✓ Found audio file: {audio_path}\n")
    
    # Load audio file
    y, sr = librosa.load(audio_path)  # y = audio time series, sr = sampling rate
    
    print(f"Sampling Rate: {sr} Hz")
    print(f"Duration: {len(y) / sr:.2f} seconds")
    print(f"Total samples: {len(y)}")
    
    print("\n2.2 Extracting Audio Features:")
    print("-" * 70)
    
    # Temporal features
    print("Temporal Features:")
    print(f"  - Energy (RMS): {np.sqrt(np.mean(y**2)):.4f}")
    print(f"  - Zero Crossing Rate: {np.mean(librosa.feature.zero_crossing_rate(y)):.4f}")
    
    # Spectral features
    print("\nSpectral Features:")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    print(f"  - Spectral Centroid (mean): {np.mean(spectral_centroid):.2f} Hz")
    print(f"  - Spectral Centroid (std): {np.std(spectral_centroid):.2f} Hz")
    
    # MFCC (Mel-Frequency Cepstral Coefficients) - standard audio features
    print("\nMFCC (Mel-Frequency Cepstral Coefficients):")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"  - MFCC shape: {mfcc.shape}")
    print(f"  - Mean MFCC coefficients: {np.mean(mfcc, axis=1)}")
    
    print("\n2.3 Creating Audio Feature Vector:")
    print("-" * 70)
    
    audio_features = {
        'Duration': len(y) / sr,
        'Sampling_Rate': sr,
        'RMS_Energy': np.sqrt(np.mean(y**2)),
        'Zero_Crossing_Rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        'Spectral_Centroid_Mean': np.mean(spectral_centroid),
        'Spectral_Centroid_Std': np.std(spectral_centroid),
        'MFCC_1': np.mean(mfcc[0]),
        'MFCC_2': np.mean(mfcc[1]),
        'MFCC_3': np.mean(mfcc[2]),
        'MFCC_Mean': np.mean(mfcc)
    }
    
    audio_df = pd.DataFrame([audio_features])
    print("\nAudio Feature DataFrame:")
    print(audio_df)
    
elif os.path.exists(audio_path) and not LIBROSA_AVAILABLE:
    print(f"✓ Found audio file: {audio_path}")
    print("Install librosa to extract audio features: pip install librosa")
    
else:
    print(f"✗ Audio file not found at: {audio_path}")
    print("Please ensure sample.mp3 is in the parent directory\n")

print("\n" + "=" * 70)
print("LESSON 3: VIDEO FEATURE EXTRACTION (MP4)")
print("=" * 70)

print("\n3.1 Understanding Video Files:")
print("-" * 70)

video_path = '../sample.mp4'

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Note: opencv-python not installed. Install with: pip install opencv-python")

if os.path.exists(video_path) and CV2_AVAILABLE:
    print(f"✓ Found video file: {video_path}\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Frame Count: {frame_count}")
    print(f"FPS (Frames Per Second): {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Resolution: {width}x{height}")
    
    print("\n3.2 Extracting Frame Features:")
    print("-" * 70)
    
    # Extract features from first, middle, and last frames
    frame_indices = [0, frame_count // 2, frame_count - 1]
    frame_labels = ["First", "Middle", "Last"]
    frame_features_list = []
    
    for idx, label in zip(frame_indices, frame_labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract color features from this frame
            frame_b, frame_g, frame_r = cv2.split(frame)
            
            frame_feature = {
                f'Frame_{label}_Brightness': np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
                f'Frame_{label}_Red_Mean': np.mean(frame_r),
                f'Frame_{label}_Green_Mean': np.mean(frame_g),
                f'Frame_{label}_Blue_Mean': np.mean(frame_b),
            }
            frame_features_list.append(frame_feature)
            
            print(f"{label} Frame:")
            print(f"  - Brightness: {frame_feature[f'Frame_{label}_Brightness']:.2f}")
            print(f"  - Red Mean: {frame_feature[f'Frame_{label}_Red_Mean']:.2f}")
            print(f"  - Green Mean: {frame_feature[f'Frame_{label}_Green_Mean']:.2f}")
            print(f"  - Blue Mean: {frame_feature[f'Frame_{label}_Blue_Mean']:.2f}")
    
    print("\n3.3 Creating Video Feature Vector:")
    print("-" * 70)
    
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
    print("\nVideo Feature DataFrame:")
    print(video_df)
    
    cap.release()
    
elif os.path.exists(video_path) and not CV2_AVAILABLE:
    print(f"✓ Found video file: {video_path}")
    print("Install opencv-python to extract video features: pip install opencv-python")
    
else:
    print(f"✗ Video file not found at: {video_path}")
    print("Please ensure sample.mp4 is in the parent directory\n")

print("\n" + "=" * 70)
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
print("LESSON 5: COMBINING ALL FEATURES INTO A DATASET")
print("=" * 70)

print("\n5.1 Creating a Complete Feature Dataset:")
print("-" * 70)

# Combine all extracted features
all_features = {
    'Sample_ID': 1,
    'Data_Type': 'Mixed',
}

# Add image features (if available)
if os.path.exists(image_path):
    all_features.update({
        'Image_Width': image_features['Width'],
        'Image_Height': image_features['Height'],
        'Image_Brightness': image_features['Brightness'],
        'Image_Contrast': image_features['Contrast']
    })

# Add audio features (if available)
if os.path.exists(audio_path) and LIBROSA_AVAILABLE:
    all_features.update({
        'Audio_Duration': audio_features['Duration'],
        'Audio_RMS_Energy': audio_features['RMS_Energy'],
        'Audio_Zero_Crossing_Rate': audio_features['Zero_Crossing_Rate']
    })

# Add video features (if available)
if os.path.exists(video_path) and CV2_AVAILABLE:
    all_features.update({
        'Video_Duration': video_features['Duration'],
        'Video_FPS': video_features['FPS'],
        'Video_Resolution': video_features['Resolution']
    })

# Add numeric features
all_features.update(numeric_features)

combined_df = pd.DataFrame([all_features])
print("\nCombined Feature DataFrame:")
print(combined_df.T)  # Transpose for better readability

print("\n" + "=" * 70)
print("SUMMARY: KEY TAKEAWAYS")
print("=" * 70)

print("""
✓ IMAGE FEATURES:
  - Pixel intensity (brightness, contrast)
  - Color channels (RGB means and standard deviations)
  - Shape features (width, height, aspect ratio)

✓ AUDIO FEATURES:
  - Temporal: Energy, Zero Crossing Rate
  - Spectral: Spectral Centroid, MFCC
  - Duration and Sampling Rate

✓ VIDEO FEATURES:
  - Frame count, FPS, Duration
  - Resolution and Aspect Ratio
  - Color features from sample frames

✓ NUMERIC DATA:
  - Statistical features: Mean, Std Dev, Min, Max
  - Derived features: Range, Variance, Skewness
  - Temporal features: Rate of change, Volatility

✓ WORKFLOW FOR FEATURE EXTRACTION:
  1. Load raw data (image, audio, video, or numeric)
  2. Analyze data properties (size, format, duration)
  3. Extract relevant numerical features
  4. Normalize/scale features if needed
  5. Combine features into feature vectors
  6. Use features for pattern recognition models

➜ Next Step: Use these features to train machine learning models!
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
