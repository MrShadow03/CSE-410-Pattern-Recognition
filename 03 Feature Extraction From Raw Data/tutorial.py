import cv2

cap = cv2.VideoCapture('sample.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("Total Frames: ", frame_count)
duration = frame_count / fps
print("Duration (seconds): ", duration)

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)


# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break;

#     cv2.imshow("Video Stream", frame)

#     if cv2.waitKey(100) == ord('q'):
#         break;