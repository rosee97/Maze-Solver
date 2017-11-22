import numpy as np
import cv2
import time


'''
img = cv2.imread('/home/rose/Pictures/transcripts1.png')
img = cv2.resize(img,(100, 100))
img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
image_data = np.asarray(img)

for i in range(len(image_data)):
    for j in range(len(image_data[0])):
        print(image_data[i][j])  #

cv2.namedWindow('web',cv2.WINDOW_NORMAL)
cv2.resizeWindow('web', 100,100)
while True:
  cv2.imshow("web", img)
  if cv2.waitKey(1)==ord('e'):
    break
'''



cam = cv2.VideoCapture("")
cv2.namedWindow('web',cv2.WINDOW_NORMAL)
cv2.resizeWindow('web', 400, 400)
while True:
  ret, frame = cam.read()
  if ret==True:
    cv2.imshow("web", frame)
  if cv2.waitKey(1)==ord('e'):
    break




'''

video_capture = cv2.VideoCapture(0)

ret = True;

while True:

    ret, frame = video_capture.read()

    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        print("hello")
        if cv2.waitKey(1)==ord('e'):
            break
'''
