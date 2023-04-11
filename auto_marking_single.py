from funcs import *
from txt_output import *
import os
import multiprocessing
import time

m = 0
total = 100


def data_marker(img, img_marked, back):
    # 亮度
    img = random_brightness(img)
    back = random_brightness(back)
    # 缩放
    r = random.randint(5, 25) / 10
    img = cv2.resize(img, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    img_marked = cv2.resize(img_marked, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    # 透视变换
    img, points = random_perspective(img, 0.2, 0, 0)
    # 模糊
    img = random_blur(img)
    # copy透视处理
    xc, yc, wc, hc = points_perspective(img_marked, points)
    # 叠加
    back, x, y = overlay(img, back)
    xmin, ymin = (x + xc, y + yc)
    xmax, ymax = (x + xc + wc, y + yc + hc)
    # cv2.rectangle(back, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return back, xmin, ymin, xmax, ymax


def data_maker(a):
    global m, total
    objs = os.listdir('auto_marking/images')
    backs = os.listdir('auto_marking/backs')
    data_format = 'yolo'
    for k in objs:
        imgs = os.listdir('auto_marking/images/' + k)
        for i in imgs:
            img = cv2.imread('auto_marking/images/' + k + '/' + i)
            img_marked = cv2.imread('auto_marking/images_marked/' + k + '/' + i)
            for j in backs:
                # lock.acquire()
                m += 1
                # lock.release()
                s = str(m) + str(a)
                back = cv2.imread('auto_marking/backs/' + j)
                data_output, xmin, ymin, xmax, ymax = data_marker(img, img_marked, back)
                cv2.imwrite("auto_marking/output/images/" + k + '/' + s + '.jpg', data_output)
                # cv2.imwrite("auto_marking/output/images/" + k + '/' + s + '.png', data_output)
                cv2.rectangle(data_output, (xmin, ymin), (xmax, ymax), (0, 255, 1), 2)
                cv2.imwrite("auto_marking/output/images_marked/" + k + '/' + s + '.jpg', data_output)
                # cv2.imwrite("auto_marking/output/images_marked/" + k + '/' + s + '.png', data_output)
                if data_format == 'voc':
                    picture_width = back.shape[1]
                    picture_height = back.shape[0]
                    txt = voc_xml_maker(s + '.jpg', xmin, ymin, xmax, ymax, k, picture_width, picture_height)
                    # txt = voc_xml_maker(s + '.png', xmin, ymin, xmax, ymax, k, picture_width, picture_height)
                    label_name = s + '.xml'
                elif data_format == 'yolo':
                    y, x, n = data_output.shape
                    txt = yolo_txt_maker(objs.index(k), xmin, ymin, xmax, ymax, x, y)
                    label_name = s + '.txt'
                else:
                    raise Exception('wrong label_name')
                path = 'auto_marking/output/labels/' + k + '/' + label_name
                fw = open(path, 'w')
                fw.write(txt)
                fw.close()
                print(s)


if __name__ == "__main__":
    data_maker('a')

