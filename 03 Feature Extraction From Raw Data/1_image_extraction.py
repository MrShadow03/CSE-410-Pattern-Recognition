# ============================================================
# LESSON 1: IMAGE FEATURE EXTRACTION FROM JPG
# ============================================================
# Goal: Extract numerical features from images using OpenCV
# Outcome: Convert images into usable feature vectors

import numpy as np
import pandas as pd
import cv2
import os

print("=" * 70)
print("LESSON 1: IMAGE FEATURE EXTRACTION (JPG)")
print("=" * 70)

print("\n1.1 Loading and Understanding Images:")
print("-" * 70)

# Path to sample image
image_path = 'sample.jpg'

if os.path.exists(image_path):
    print(f"✓ Found image: {image_path}\n")
    
    # Load image using OpenCV
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Get image properties
    height, width, channels = img_rgb.shape
    print(f"Image size (width, height): ({width}, {height})")
    print(f"Image channels: {channels}")
    
    print(f"\nImage shape: {img_rgb.shape}")
    print(f"Height: {height}, Width: {width}, Channels: {channels}")
    
    print("\n1.2 Extracting Color Features:")
    print("-" * 70)
    
    # Extract color channel features (RGB order)
    red_channel = img_rgb[:, :, 0]
    green_channel = img_rgb[:, :, 1]
    blue_channel = img_rgb[:, :, 2]
    
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
    
    # Convert to grayscale for brightness analysis using OpenCV
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    brightness = img_gray.mean()
    contrast = img_gray.std()
    min_intensity = img_gray.min()
    max_intensity = img_gray.max()
    
    print(f"Brightness (Mean): {brightness:.2f}")
    print(f"Contrast (Std Dev): {contrast:.2f}")
    print(f"Min Intensity: {min_intensity}")
    print(f"Max Intensity: {max_intensity}")
    print(f"Dynamic Range: {max_intensity - min_intensity}")
    
    print("\n1.4 Extracting Shape Features:")
    print("-" * 70)
    
    # Image dimensions
    height, width = img_gray.shape
    aspect_ratio = width / height if height != 0 else 0
    area = height * width
    
    print(f"Image Height: {height}")
    print(f"Image Width: {width}")
    print(f"Aspect Ratio (width/height): {aspect_ratio:.2f}")
    print(f"Total Pixels (Area): {area}")
    
    print("\n1.5 Additional OpenCV Features:")
    print("-" * 70)
    
    # Edge detection using Canny
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = np.count_nonzero(edges) / (height * width)
    print(f"Edge Density: {edge_density:.4f}")
    
    # Image histogram analysis
    hist_blue = cv2.calcHist([img_bgr], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([img_bgr], [1], None, [256], [0, 256])
    hist_red = cv2.calcHist([img_bgr], [2], None, [256], [0, 256])
    
    print(f"Histogram Blue Mean: {np.mean(hist_blue):.2f}")
    print(f"Histogram Green Mean: {np.mean(hist_green):.2f}")
    print(f"Histogram Red Mean: {np.mean(hist_red):.2f}")
    
    print("\n1.6 Creating Complete Image Feature Vector:")
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
        'Blue_Std': blue_channel.std(),
        'Edge_Density': edge_density,
        'Histogram_Blue_Mean': np.mean(hist_blue),
        'Histogram_Green_Mean': np.mean(hist_green),
        'Histogram_Red_Mean': np.mean(hist_red)
    }
    
    image_df = pd.DataFrame([image_features])
    print("\nImage Feature DataFrame:")
    print(image_df)
    
    print("\n" + "=" * 70)
    print("IMAGE EXTRACTION SUMMARY (Using OpenCV)")
    print("=" * 70)
    print("""
KEY CONCEPTS:
✓ Color Features: Extract mean and std dev from RGB channels
✓ Brightness: Average pixel intensity in grayscale
✓ Contrast: Standard deviation of pixel intensities
✓ Shape Features: Width, height, aspect ratio
✓ Edge Density: Proportion of edges detected (Canny edge detection)
✓ Histogram Analysis: Distribution of pixel intensities per channel
✓ Feature Vector: Combine all features into a single vector

OPENCV ADVANTAGES:
✓ Faster image processing than PIL
✓ Better support for computer vision operations
✓ Edge detection, contour finding, image transformations
✓ Histogram analysis and statistical functions
✓ Widely used in industry for image analysis

USE CASES:
- Image classification (cat vs dog)
- Object detection (size and color analysis)
- Medical imaging (tissue analysis)
- Quality control (detecting defects in products)
- Edge detection and shape analysis
""")
    
else:
    print(f"✗ Image not found at: {image_path}")
    print("Please ensure sample.jpg is in the current directory\n")

print("\n" + "=" * 70)
