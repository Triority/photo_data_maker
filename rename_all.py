import os

path = r'F:\auto_marking\images_marked'
objects = os.listdir(path)
print(objects)
for i in objects:
    a = 1
    imgs = os.listdir(os.path.join(path, i))
    imgs.sort(key=str.lower)
    print(imgs)
    for j in imgs:
        os.rename(os.path.join(path, i, j), os.path.join(path, i, 'img' + str(a) + '.jpg'))
        a = a + 1
