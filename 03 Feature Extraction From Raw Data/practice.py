import cv2
import numpy as np
import pandas as pd

img = cv2.imread("cliff.jpg")
print(type(img))

h, w, c = img.shape

red_channel = img[:,:,2]
green_channel = img[:,:,1]
blue_channel = img[:,:,0]

gray = cv2.imread("cliff.jpg", 0)

brightness = gray.mean()
contrast = gray.std()






img_data = [{
    "height" : h,
    "width" : w,
    "red_mean" : red_channel.mean(),
    "green_mean" : green_channel.mean(),
    "blue_mean" : blue_channel.mean(),
    "brightness" : brightness,
    "contrast" : contrast,
    "label" : "cat"
}]

df = pd.DataFrame(img_data)

cap = cv2.VideoCapture("sample.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
f_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
f_hight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
f_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

print(f"duration: {f_count/fps}")