import os
import cv2

images = 'video2img/img_output/'
videos = os.listdir('video2img/video')
step = 10

c = 0
for i in videos:
    path = 'video2img/video/' + i
    cap = cv2.VideoCapture(path)
    while 1:
        for j in range(step):
            success, frame = cap.read()
        if success:
            img = cv2.imwrite(images + str(c) + '.jpg', frame)
            c = c + 1
            print(c)
        else:
            break
    cap.release()
