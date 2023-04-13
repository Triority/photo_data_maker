import random
import cv2
import numpy as np


def n_random(loc, scale, min, max):
    """
    生成指定范围内的随机正态分布数
    :param loc: 平均值
    :param scale: 标准差
    :param min: 范围最小值
    :param max: 范围最大值
    :return: 得到的随机值
    """
    while True:
        n_r = np.random.normal(loc, scale, size=(1, 1))[1][1]
        if min < n_r < max:
            break
    return n_r

def random_flip(img, mode):
    if mode == 0:
        return img, None
    elif mode == 1:
        if random.choice([0, 1]):
            img = cv2.flip(img, 0)
        return img, 0
    elif mode == 2:
        if random.choice([0, 1]):
            img = cv2.flip(img, 1)
        return img, 1
    elif mode == 3:
        i = random.choice([0, 1, 2, 3])
        if i:
            if i == 1:
                img = cv2.flip(img, 0)
                return img, 0
            elif i == 2:
                img = cv2.flip(img, 1)
                return img, 1
            elif i == 3:
                img = cv2.flip(img, -1)
                return img, -1
        else:
            return img, None
    else:
        raise Exception('wrong flip mode')


def random_brightness(img, a=30, b=30, bright_min=2, bright_max=254):
    """
    亮度随机变化
    :param img: 传入图像
    :param a: 亮度变化倍率最大范围（0-100），默认30
    :param b: 亮度变化大小最大范围（0-255），默认30
    :param bright_min: 返回图像最小亮度，默认2
    :param bright_max: 返回图像最大亮度，默认254
    :return: 返回图像
    """
    alpha = 0.01 * random.randint(-a, a) + 1
    beta = random.randint(-b, b)
    return np.uint8(np.clip((alpha * img + beta), bright_min, bright_max))


def random_blur(img, r=3):
    """
    随机大小高斯滤波
    :param img: 传入图像
    :param r: 高斯滤波范围，模糊处理边长为2r+1，默认3
    :return: 返回图像
    """
    n = random.randint(0, r)
    return cv2.blur(img, (2 * n + 1, 2 * n + 1))


def random_perspective(img, random_range=0.3, symmetry_mode=0, symmetry_direction=0):
    """
    随机透视变换
    :param img:传入图像
    :param random_range:随机范围，默认为0.3，指每个角位置变化距离占边长百分比，应介于0-0.5
    :param symmetry_mode:对称模式，默认为0，即不对称，1为左右对称，2为上下对称。
    以正方形为例，对称将只会把正方形透视变换为等腰梯形，左右对称则左右两边长度相等上下两边平行，上下对称则上下两边长度相等左右两边跑平行。
    :param symmetry_direction:对称方向指定，仅当开启对称情况下有效，默认为0，即不指定。
    若前一参数对称模式为1即左右对称，此参数为1表示等腰梯形上窄下宽，此参数为2反之.
    若前一参数对称模式为2即上下对称，此参数为1表示等腰梯形左宽右窄，此参数为2反之.
    :return:透视后的图像，随机生成透视变换四个点的xy坐标的二维数组
    """
    while True:
        h, w, p = img.shape
        w2 = int(w / 2)
        h2 = int(h / 2)
        if symmetry_mode == 0:
            x1 = random.randint(int(random_range * w), w2)
            x2 = random.randint(int((1 - random_range) * w), w)
            x3 = random.randint(int(random_range * w), w2)
            x4 = random.randint(int((1 - random_range) * w), w)
            y1 = random.randint(int(random_range * h), h2)
            y2 = random.randint(int(random_range * h), h2)
            y3 = random.randint(int((1 - random_range) * h), h)
            y4 = random.randint(int((1 - random_range) * h), h)
            points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        elif symmetry_mode == 1:
            x1 = random.randint(int(random_range * w), w2)
            x2 = w - 1 - x1
            x3 = 0
            x4 = w - 1
            y1 = random.randint(int(random_range * h), h2)
            y2 = y1
            y3 = h - 1
            y4 = y3
            if symmetry_direction == 1:
                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            elif symmetry_direction == 2:
                points = [[x3, y1], [x4, y2], [x1, y3], [x2, y4]]
            elif symmetry_direction == 0:
                if random.choice([1, 2]) == 1:
                    points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                else:
                    points = [[x3, y1], [x4, y2], [x1, y3], [x2, y4]]
            else:
                raise Exception('symmetry_direction只应该是0/1/2')
        elif symmetry_mode == 2:
            x1 = 0
            x2 = random.randint(int((1 - random_range) * w), w)
            x3 = 0
            x4 = x2
            y1 = 0
            y2 = random.randint(int(random_range * h), h2)
            y3 = h - 1
            y4 = h - 1 - y2
            if symmetry_direction == 1:
                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            elif symmetry_direction == 2:
                points = [[x1, y2], [x2, y1], [x3, y4], [x4, y3]]
            elif symmetry_direction == 0:
                if random.choice([1, 2]) == 1:
                    points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                else:
                    points = [[x1, y2], [x2, y1], [x3, y4], [x4, y3]]
            else:
                raise Exception('symmetry_direction只应该是0/1/2')
        else:
            raise Exception('symmetry_mode只应该是0/1/2')

        # 凸四边形验证，不确定修改后是否依然需要，反正留着不会出错
        if ((x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)) * ((x3 - x1) * (y4 - y1) - (y3 - y1) * (x4 - x1)) < 0 and (
                (x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2)) * ((x4 - x2) * (y3 - y2) - (y4 - y2) * (x3 - x2)) < 0:
            break
        else:
            print("warning" + str(points))
    pts3_d1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  # 原图点
    pts3_d2 = np.float32(points)  # 随机得到的四个点
    m = cv2.getPerspectiveTransform(pts3_d1, pts3_d2)  # 矩阵计算
    return cv2.warpPerspective(img, m, (w, h)), points


def points_perspective(img_copy, points):
    """
    进行指定参数的透视变换求纯绿色的最小外接矩形
    :param img_copy: 带绿色标注的图像
    :param points: 透视变换参数点
    :return: 所得矩形的[xc, yc, wc, hc]
    """
    h, w, p = img_copy.shape
    img_copy = cv2.warpPerspective(img_copy, cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [0, h], [w, h]]),
                                                                         np.float32(points)), (w, h))
    cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    img_copy = cv2.inRange(img_copy, np.array([0, 250, 0]), np.array([5, 255, 255]))
    xc, yc, wc, hc = cv2.boundingRect(img_copy)
    # cv2.rectangle(img_copy, (xc, yc), (xc + wc, yc + hc), 255, 2)
    return xc, yc, wc, hc


def overlay(img, back):
    """
    透视变换后的图像叠加到背景图
    :param img: 透视变换后的图像
    :param back: 背景图
    :return: 输出图像，输入图像被放置的位置（左上xy坐标）
    """
    h, w, p = back.shape
    rows, cols, channels = img.shape
    x = random.randint(0, round(w - cols))
    y = random.randint(0, round(h - rows))
    roi = back[y:rows + y, x:cols + x]
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img, img, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    back[y:rows + y, x:cols + x] = dst
    return back, x, y
