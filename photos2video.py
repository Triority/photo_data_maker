import cv2
import os

dir_path = r'photos2video'
photos = os.listdir(dir_path)
path = dir_path + '\\' + photos[0]
photo = cv2.imread(path)
size = (photo.shape[1], photo.shape[0])
videowrite = cv2.VideoWriter(r'photos2video.mp4', -1, 30, size)
i=0
for filename in photos:
    path = dir_path + '\\' + filename
    img = cv2.imread(path)
    if img is None:
        print(filename + " is error!")
        continue
    videowrite.write(img)
    i += 1
    print(i)
videowrite.release()
print('finished')
