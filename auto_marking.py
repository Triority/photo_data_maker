import traceback
from funcs import *
from txt_output import *
import os
import multiprocessing
import yaml

yamlPath = 'config.yaml'
with open(yamlPath, 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

# 每张图片与背景组合生成的图片总数
# 如果设置数量过小(if total < processes * len(backs))将导致舍弃部分背景图片
# 生成的图片总量 = imgs * processes * total
total = 1
# 使用的进程数，不大于cpu核心数，这里保留一个核心空闲避免电脑过于卡顿
processes = multiprocessing.cpu_count() - 1
# 图片路径设置
backs_dir_path = r'C:\auto_marking\backs'
images_dir_path = r'C:\auto_marking\images'
marked_dir_path = r'C:\auto_marking\images_marked'
output_dir_path = r'C:\auto_marking\output'


def data_marker(img, img_marked, back):
    # 亮度，可自定义参数
    img = random_brightness(img, a=config['brightness']['a'], b=config['brightness']['b']
                            , bright_min=config['brightness']['min'], bright_max=config['brightness']['max'])
    back = random_brightness(back, a=config['brightness']['a'], b=config['brightness']['b']
                             , bright_min=config['brightness']['min'], bright_max=config['brightness']['max'])
    # 缩放，可自定义参数
    r = random.randint(config['size']['min'], config['size']['max']) / 100 * min(back.shape[0] / img.shape[0]
                                                                                 , back.shape[1] / img.shape[1])
    img = cv2.resize(img, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    img_marked = cv2.resize(img_marked, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    # 透视变换，可自定义参数
    img, points = random_perspective(img, config['perspective']['range']
                                     , config['perspective']['mode'], config['perspective']['direction'])
    # 模糊
    img = random_blur(img, config['blur']['r'])
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
        objs = os.listdir(images_dir_path)
        backs = os.listdir(backs_dir_path)
        data_format = config['format']
        for k in objs:
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
                while m < total:
                    img = cv2.imread(images_dir_path + '\\' + k + '\\' + i)
                    img_marked = cv2.imread(marked_dir_path + '\\' + k + '\\' + i)
                    for j in backs:
                        if m > total:
                            break
                        m += 1
                        print(k + ' ' + i + ' ' + j + ' ' + str(m))
                        s = str(m) + str(a)
                        back = cv2.imread(backs_dir_path + '\\' + j)
                        data_output, xmin, ymin, xmax, ymax = data_marker(img, img_marked, back)
                        cv2.imwrite(output_dir_path + "\\images\\" + k + '\\' + s + '.jpg', data_output)
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
                            raise Exception('wrong label_name')
                        path = output_dir_path + '\\labels\\' + k + '\\' + label_name
                        fw = open(path, 'w')
                        fw.write(txt)
                        fw.close()
                        # print(s)
    except (Exception, BaseException) as e:
        exstr = traceback.format_exc()
        print(exstr)


if __name__ == "__main__":
    pl = multiprocessing.Manager().Lock()
    pool = multiprocessing.Pool(processes)
    per = int(total / processes)
    for i in range(processes):
        pool.apply_async(data_maker, args=(str(i), per))
    pool.close()
    pool.join()
    print("Sub-process(es) done.")
