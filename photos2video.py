import cv2
import os

photos = os.listdir(r'photos2video')
photo = cv2.imread('photos2video\\' + photos[0])
size = (photo.shape[1], photo.shape[0])
videowrite = cv2.VideoWriter(r'photos2video.mp4', -1, 30, size)
i=0
for filename in photos:
    path = 'photos2video\\' + filename
    img = cv2.imread(path)
    if img is None:
        print(filename + " is error!")
        continue
    videowrite.write(img)
    i += 1
    print(i)
videowrite.release()
print('finished')