import random
import time
from funcs import *
from txt_output import *
import os
import multiprocessing
import yaml


time_start = time.time()
time_this = time.time()

i_t = 0
yamlPath = 'config.yaml'
with open(yamlPath, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

total = config['total']
processes = multiprocessing.cpu_count() - config['cpu_free']
backs_dir_path = config['backs_dir_path']
images_dir_path = config['images_dir_path']
marked_dir_path = config['marked_dir_path']
output_dir_path = config['output_dir_path']


def data_marker(img, img_marked, back):
    img_h, img_w = img.shape[:2]
    back_h, back_w = back.shape[:2]
    # 初始图像翻转
    img, ret = random_flip(img, config['flip']['mode'])
    img_marked = cv2.flip(img_marked, ret)
    back, ret = random_flip(back, config['flip']['mode'])
    # 亮度随机变化
    img = random_brightness(img, a=config['brightness']['a'], b=config['brightness']['b']
                            , bright_min=config['brightness']['min'], bright_max=config['brightness']['max'])
    back = random_brightness(back, a=config['brightness']['a'], b=config['brightness']['b']
                             , bright_min=config['brightness']['min'], bright_max=config['brightness']['max'])
    # 缩放
    r = random.randint(config['size']['min'], config['size']['max']) / 100 * min(back_h / img_h, back_w / img_w)
    img = cv2.resize(img, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    img_marked = cv2.resize(img_marked, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    # 旋转，报错，在改
    # center = (img_w / 2, img_h / 2)
    # rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=0, scale=1)
    # img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(img_w, img_h))
    # img_marked = cv2.warpAffine(src=img_marked, M=rotate_matrix, dsize=(img_w, img_h))
    # 透视变换
    img, points = random_perspective(img, config['perspective']['range']
                                     , config['perspective']['mode'], config['perspective']['direction'])
    # 模糊
    img = random_blur(img, config['blur']['r'])
    # copy透视
    xc, yc, wc, hc = points_perspective(img_marked, points)
    # 叠加
    back, x, y = overlay(img, back)
    xmin, ymin = (x + xc, y + yc)
    xmax, ymax = (x + xc + wc, y + yc + hc)
    return back, xmin, ymin, xmax, ymax


def data_maker(a):
    global time_this, i_t
    total_cpu = int(total / processes)
    objs = os.listdir(images_dir_path)
    backs = os.listdir(backs_dir_path)
    data_format = config['format']
    for k in objs:
        o = 0
        imgs = os.listdir(images_dir_path + '\\' + k)
        if not os.path.exists(output_dir_path + "\\images\\" + k):
            print('mkdir:' + output_dir_path + "\\images\\" + k)
            os.makedirs(output_dir_path + "\\images\\" + k)
        if not os.path.exists(output_dir_path + "\\images_marked\\" + k):
            print('mkdir:' + output_dir_path + "\\images_marked\\" + k)
            os.makedirs(output_dir_path + "\\images_marked\\" + k)
        if not os.path.exists(output_dir_path + "\\labels\\" + k):
            print('mkdir:' + output_dir_path + "\\labels\\" + k)
            os.makedirs(output_dir_path + "\\labels\\" + k)
        for i in imgs:
            m = 0
            while m <= total_cpu:
                o = o + 1
                if str(a) == '0':
                    print('----------time now: ' + str(round(time.time()-time_start, 2)) + 's')
                    print('----------time this: ' + str(round(time.time()-time_this, 2)) + 's')
                    print('----------img total(about): ' + str(i_t*processes))
                    time_this = time.time()
                    i_t = i_t + 1
                img = cv2.imread(images_dir_path + '\\' + k + '\\' + i)
                img_marked = cv2.imread(marked_dir_path + '\\' + k + '\\' + i)
                j = random.choice(backs)
                m += 1
                print(k + ' ' + i + ' ' + str(a) + ' ' + str(m) + ' ' + str(o))
                s = str(a) + str(o)
                back = cv2.imread(backs_dir_path + '\\' + j)
                data_output, xmin, ymin, xmax, ymax = data_marker(img, img_marked, back)
                cv2.imwrite(output_dir_path + "\\images\\" + k + '\\' + s + '.jpg', data_output)
                if random.randint(0, 100) <= config['save_marked']:
                    cv2.rectangle(data_output, (xmin, ymin), (xmax, ymax), (0, 255, 1), 2)
                    cv2.imwrite(output_dir_path + "\\images_marked\\" + k + '\\' + s + '.jpg', data_output)
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
                    raise Exception('wrong label_type')
                path = output_dir_path + '\\labels\\' + k + '\\' + label_name
                fw = open(path, 'w')
                fw.write(txt)
                fw.close()
                # print(s)


if __name__ == "__main__":
    pl = multiprocessing.Manager().Lock()
    pool = multiprocessing.Pool(processes)
    for i in range(processes):
        pool.apply_async(data_maker, args=(str(i),))
    pool.close()
    pool.join()
    print("Sub-process(es) done.")
    print('Total time = ' + str(round(time.time()-time_start, 2)) + 's')
