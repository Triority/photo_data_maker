import time
from funcs import *
from txt_output import *
import os
import multiprocessing
import yaml
import sys
from tqdm import tqdm

time_start = time.time()
time_this = time.time()
i_t = 0
yamlPath = 'config.yaml'
with open(yamlPath, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

total = config['total']
processes = multiprocessing.cpu_count() - config['cpu_free']
if config['dir_path']['back_in_root']:
    backs_dir_path = config['dir_path']['root_dir_path'] + config['dir_path']['backs_dir_path']
else:
    backs_dir_path = config['dir_path']['backs_dir_path']
images_dir_path = config['dir_path']['root_dir_path'] + config['dir_path']['images_dir_path']
marked_dir_path = config['dir_path']['root_dir_path'] + config['dir_path']['marked_dir_path']
output_dir_path = config['output_dir_path']


def data_marker(img, img_marked, back):
    img_h, img_w = img.shape[:2]
    back_h, back_w = back.shape[:2]
    # 初始图像翻转
    img, ret = random_flip(img, config['flip']['mode'])
    img_marked = cv2.flip(img_marked, ret)
    back, ret = random_flip(back, config['flip']['mode'])
    # 亮度随机变化
    img = random_brightness(img, a=config['brightness']['a'], b=config['brightness']['b'], c=config['brightness']['c']
                            , bright_min=config['brightness']['min'], bright_max=config['brightness']['max'])
    back = random_brightness(back, a=config['brightness']['a'], b=config['brightness']['b'], c=config['brightness']['c']
                             , bright_min=config['brightness']['min'], bright_max=config['brightness']['max'])
    # RGB变化
    img = random_channel_gain(img, random_range=config['brightness']['RGB'])
    back = random_channel_gain(back, random_range=config['brightness']['RGB'])
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


def img_maker(a):
    global time_this, i_t
    total_cpu = int(total / processes)
    objs = os.listdir(images_dir_path)
    backs = os.listdir(backs_dir_path)
    data_format = config['format']
    output_classification = config['output_classification']
    with tqdm(range(total_cpu), total=total_cpu, unit='img', file=sys.stdout, desc=a) as pbar:
        for k in objs:
            o = 0
            if output_classification:
                if not os.path.exists(output_dir_path + "\\train\\images\\" + k):
                    print('mkdir:' + output_dir_path + "\\train\\images\\" + k)
                    os.makedirs(output_dir_path + "\\train\\images\\" + k)
                if config['save_marked'] and not os.path.exists(output_dir_path + "\\train\\images_marked\\" + k):
                    print('mkdir:' + output_dir_path + "\\train\\images_marked\\" + k)
                    os.makedirs(output_dir_path + "\\train\\images_marked\\" + k)
                if not os.path.exists(output_dir_path + "\\train\\labels\\" + k):
                    print('mkdir:' + output_dir_path + "\\train\\labels\\" + k)
                    os.makedirs(output_dir_path + "\\train\\labels\\" + k)
                if config['val_total']:
                    if not os.path.exists(output_dir_path + "\\val\\images\\" + k):
                        print('mkdir:' + output_dir_path + "\\val\\images\\" + k)
                        os.makedirs(output_dir_path + "\\val\\images\\" + k)
                    if not os.path.exists(output_dir_path + "\\val\\labels\\" + k):
                        print('mkdir:' + output_dir_path + "\\val\\labels\\" + k)
                        os.makedirs(output_dir_path + "\\val\\labels\\" + k)
            else:
                if not os.path.exists(output_dir_path + "\\JPEGImages"):
                    print('mkdir:' + output_dir_path + "\\JPEGImages")
                    os.makedirs(output_dir_path + "\\JPEGImages")
                if not os.path.exists(output_dir_path + '\\Annotations'):
                    print('mkdir:' + output_dir_path + '\\Annotations')
                    os.makedirs(output_dir_path + '\\Annotations')
                if config['save_marked'] and not os.path.exists(output_dir_path + "\\images_marked\\" + k):
                    print('mkdir:' + output_dir_path + "\\images_marked\\" + k)
                    os.makedirs(output_dir_path + "\\images_marked\\" + k)
            imgs = os.listdir(images_dir_path + '\\' + k)
            for i in imgs:
                pbar.reset()
                m = 0
                while m < total_cpu:
                    o = o + 1
                    if str(a) == '0':
                        time_now = round(time.time() - time_start, 2)
                        print('----------time now: {} s'.format(time_now))
                        print('----------time this: {} s'.format(round(time.time() - time_this, 2)))
                        print('----------img total(about): {}'.format(i_t * processes))
                        try:
                            print('----------time left(about): {} min(without val)'.format(round(total*len(objs)/(i_t*processes/time_now)/60, 2)))
                        except:
                            pass
                        time_this = time.time()
                        i_t = i_t + 1
                    img = cv2.imread(images_dir_path + '\\' + k + '\\' + i)
                    img_marked = cv2.imread(marked_dir_path + '\\' + k + '\\' + i)
                    j = random.choice(backs)
                    m += 1
                    s = str(k) + "-" + str(i) + "-" + str(a).rjust(3, '0') + str(o).rjust(5, '0')
                    back = cv2.imread(backs_dir_path + '\\' + j)
                    data_output, xmin, ymin, xmax, ymax = data_marker(img, img_marked, back)
                    if output_classification:
                        cv2.imwrite(output_dir_path + "\\train\\images\\" + k + '\\' + s + config['sign'] + '.jpg',
                                    data_output)
                    else:
                        cv2.imwrite(output_dir_path + "\\JPEGImages\\" + s + config['sign'] + '.jpg',
                                    data_output)
                    if random.randint(1, 100) <= config['save_marked']:
                        cv2.rectangle(data_output, (xmin, ymin), (xmax, ymax), (0, 255, 1), 2)
                        if output_classification:
                            cv2.imwrite(output_dir_path + "\\train\\images_marked\\" + k + '\\' + s + config['sign'] + '.jpg', data_output)
                        else:
                            cv2.imwrite(
                                output_dir_path + "\\images_marked\\" + k + '\\' + s + config['sign'] + '.jpg',
                                data_output)
                    if data_format == 'voc':
                        picture_width = back.shape[1]
                        picture_height = back.shape[0]
                        txt = voc_xml_maker(s + '.jpg', xmin, ymin, xmax, ymax, k, picture_width, picture_height)
                        label_name = s + config['sign'] + '.xml'
                    elif data_format == 'yolo':
                        y, x, n = data_output.shape
                        txt = yolo_txt_maker(objs.index(k), xmin, ymin, xmax, ymax, x, y)
                        label_name = s + config['sign'] + '.txt'
                    else:
                        raise Exception('wrong label_type')
                    if output_classification:
                        path = output_dir_path + '\\train\\labels\\' + k + '\\' + label_name
                    else:
                        path = output_dir_path + '\\Annotations\\' + label_name
                    fw = open(path, 'w')
                    fw.write(txt)
                    fw.close()
                    pbar.update(1)
                    print('')
                m = 0
                while m < int(config['val_total'] / processes) and output_classification:
                    o = o + 1
                    print('processes' + str(a) + ': making val......')
                    img = cv2.imread(images_dir_path + '\\' + k + '\\' + i)
                    img_marked = cv2.imread(marked_dir_path + '\\' + k + '\\' + i)
                    j = random.choice(backs)
                    m += 1
                    s = str(a).rjust(3, '0') + str(o).rjust(5, '0')
                    back = cv2.imread(backs_dir_path + '\\' + j)
                    data_output, xmin, ymin, xmax, ymax = data_marker(img, img_marked, back)
                    cv2.imwrite(output_dir_path + "\\val\\images\\" + k + '\\' + s + config['sign'] + '.jpg', data_output)
                    if data_format == 'voc':
                        picture_width = back.shape[1]
                        picture_height = back.shape[0]
                        txt = voc_xml_maker(s + '.jpg', xmin, ymin, xmax, ymax, k, picture_width, picture_height)
                        label_name = s + config['sign'] + '.xml'
                    elif data_format == 'yolo':
                        y, x, n = data_output.shape
                        txt = yolo_txt_maker(objs.index(k), xmin, ymin, xmax, ymax, x, y)
                        label_name = s + config['sign'] + '.txt'
                    else:
                        raise Exception('wrong label_type')
                    path = output_dir_path + '\\val\\labels\\' + k + '\\' + label_name
                    fw = open(path, 'w')
                    fw.write(txt)
                    fw.close()



if __name__ == "__main__":
    if not len(output_dir_path) == 0:
        print('此操作将删除{}文件夹并重新生成，请确保你已经把数据在其他位置备份或决定丢弃'.format(output_dir_path))
        if not input('输入y回车确认') == 'y':
            exit()
    for root, dirs, files in os.walk(output_dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    pl = multiprocessing.Manager().Lock()
    pool = multiprocessing.Pool(processes)
    for i in range(processes):
        pool.apply_async(img_maker, args=(str(i),))
    pool.close()
    pool.join()
    print("Sub-process(es) done.")
    print('Total time = ' + str(round(time.time() - time_start, 2)) + 's')
