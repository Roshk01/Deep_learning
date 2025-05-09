from ultralytics import YOLO
import cv2

model = YOLO("models/yolo11m.pt")
result = model("images/7.jpg", show=True)

cv2.waitKey(0)

