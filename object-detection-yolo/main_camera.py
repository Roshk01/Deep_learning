from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4,720)
cap = cv2.VideoCapture('video/2.mp4')
model = YOLO("models/yolo11n.pt")

while True:
    success, img = cap.read()
    result = model(img)

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2-x1), (y2-y1)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            cvzone.cornerRect(img,(x1,y1,w,h), l=50, t=5, rt=3,
               colorR=(0, 0, 255), colorC=(255, 0, 0))
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            label = model.names[int(box.cls)]
            cvzone.putTextRect(img,f'{label} {conf}',(max(0,x1),max(40,y1)), scale=1.5, thickness=2)

    img_resized = cv2.resize(img, None, fx=0.5, fy=0.4)  # width x height 50% reduce
    cv2.imshow("Image", img_resized)
    cv2.waitKey(1)
