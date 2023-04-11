import traceback
from funcs import *
from txt_output import *
import os
import multiprocessing

total = 100
processes = 16


def data_marker(img, img_marked, back):
    # 亮度，可自定义参数
    img = random_brightness(img)
    back = random_brightness(back)
    # 缩放，可自定义参数
    r = random.randint(5, 25) / 100
    img = cv2.resize(img, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    img_marked = cv2.resize(img_marked, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    # 透视变换，可自定义参数
    img, points = random_perspective(img, 0.25, 0, 0)
    # 模糊
    # img = random_blur(img)
    # copy透视处理
    xc, yc, wc, hc = points_perspective(img_marked, points)
    # 叠加
    back, x, y = overlay(img, back)
    xmin, ymin = (x + xc, y + yc)
    xmax, ymax = (x + xc + wc, y + yc + hc)
    # cv2.rectangle(back, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return back, xmin, ymin, xmax, ymax


def data_maker(a, pro):
    try:
        m = 0
        objs = os.listdir('auto_marking/images')
        backs = os.listdir('auto_marking/backs')
        data_format = 'yolo'
        while m < pro:
            for k in objs:
                imgs = os.listdir('auto_marking/images/'+k)
                for i in imgs:
                    img = cv2.imread('auto_marking/images/' + k + '/' + i)
                    img_marked = cv2.imread('auto_marking/images_marked/' + k + '/' + i)
                    for j in backs:
                        m += 1
                        print(k+' '+i+' '+j+' '+str(m))
                        s = str(m) + str(a)
                        back = cv2.imread('auto_marking/backs/' + j)
                        data_output, xmin, ymin, xmax, ymax = data_marker(img, img_marked, back)
                        cv2.imwrite("auto_marking/output/images/" + k + '/' + s + '.jpg', data_output)
                        cv2.rectangle(data_output, (xmin, ymin), (xmax, ymax), (0, 255, 1), 2)
                        cv2.imwrite("auto_marking/output/images_marked/" + k + '/' + s + '.jpg', data_output)
                        if data_format == 'voc':
                            picture_width = back.shape[1]
                            picture_height = back.shape[0]
                            txt = voc_xml_maker(s + '.jpg', xmin, ymin, xmax, ymax, k, picture_width, picture_height)
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
                        #print(s)
    except (Exception, BaseException) as e:
        exstr = traceback.format_exc()
        print(exstr)


if __name__ == "__main__":
    pl = multiprocessing.Manager().Lock()
    pool = multiprocessing.Pool(processes)
    per = int(total/processes)
    for i in range(processes):
        pool.apply_async(data_maker, args=(str(i), per))
    pool.close()
    pool.join()
    print("Sub-process(es) done.")
