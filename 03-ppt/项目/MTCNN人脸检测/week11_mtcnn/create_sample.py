import os
import random

import cv2
import numpy as np
from PIL import Image

from utils import utils, trans_util

# 数据样本
img_src = r"E:\data_seg\train"
# 标签
anno_src = r"E:\data_seg\train\trainImageList.txt"
# 样本（正样本、负样本、部分样本）路径
save_path = r"C:\lfw_out_add4"
# 按照PRO三层网路的要求，生成12,24,48尺寸的样本
# 正样本、负样本、部分样本
for face_size in [48]:
    print(f"生成{face_size}的数据")
    # 样本路径：正样本positive、负样本negative、部分样本part
    # 文件夹 -- 文本文件.txt
    positive_image_dir = f"{save_path}/{str(face_size)}/positive"
    negative_image_dir = f"{save_path}/{str(face_size)}/negative"
    part_image_dir = f"{save_path}/{str(face_size)}/part"
    # 自动创建
    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    # 创建标签文件
    positive_anno_filename = f"{save_path}/{str(face_size)}/positive.txt"
    negative_anno_filename = f"{save_path}/{str(face_size)}/negative.txt"
    part_anno_filename = f"{save_path}/{str(face_size)}/part.txt"
    #  正样本positive:负样本negative:部分样本part = 1: 3: 1
    # 计数器
    positive_count = 0
    negative_count = 0
    part_count = 0


    # 写文档、抠样本
    positive_anno_file = open(positive_anno_filename, "w")
    negative_anno_file = open(negative_anno_filename, "w")
    part_anno_file = open(part_anno_filename, "w")
    try:
        lines = open(anno_src, "r").readlines()
        # '000001.jpg 136 147 293 338 165 184 244 176 196 249 194 271 266 260\n'
        for i in range(len(lines)):

            print(f'img {i} adding...')

            strs = lines[i].split()
            strs[2], strs[3] = strs[3], strs[2]
            # ['000001.jpg', '136', '147', '293', '338', '165', '184', '244', '176', '196', '249', '194', '271', '266', '260']
            image_filename = strs[0].strip()
            # 图片完整路径
            image_file = f"{img_src}/{image_filename}"

            # 三种img：原，水平反转，旋转
            # 抠图
            for mode in range(1, 4):
                with Image.open(image_file) as img:

                    img_w, img_h = img.size
                    if mode == 1:
                        img, strs = trans_util.rotate_image_and_keypoints(np.asarray(img), np.array(strs[1:]),
                                                                          random.randint(-7, 7))
                        strs = np.insert(strs, 0, 0)


                    elif mode == 2:
                        img, strs = trans_util.flip_image_and_keypoints(np.asarray(img), np.asarray(strs[1:]))
                        strs = np.insert(strs, 0, 0)
                        strs[1], strs[3] = strs[3], strs[1]

                    # img = cv2.rectangle(img, (int(strs[1]), int(strs[2])), (int(strs[3]), int(strs[4])),
                    #                     color=(0, 255, 0))
                    # img = cv2.circle(img, (int(strs[5]), int(strs[6])), color=(0, 255, 0), radius=3)
                    # img = cv2.circle(img, (int(strs[7]), int(strs[8])), color=(0, 255, 0), radius=3)
                    # img = cv2.circle(img, (int(strs[9]), int(strs[10])), color=(0, 255, 0), radius=3)
                    # img = cv2.circle(img, (int(strs[11]), int(strs[12])), color=(0, 255, 0), radius=3)
                    # img = cv2.circle(img, (int(strs[13]), int(strs[14])), color=(0, 255, 0), radius=3)
                    # cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # cv2.waitKey()
                    if mode != 3:
                        img = Image.fromarray(img)

                    # 建议框的读取
                    x1 = float(strs[1])
                    y1 = float(strs[2])
                    x2 = float(strs[3])
                    y2 = float(strs[4])

                    # 关键点的读取
                    px1 = float(strs[5])
                    py1 = float(strs[6])
                    px2 = float(strs[7])
                    py2 = float(strs[8])
                    px3 = float(strs[9])
                    py3 = float(strs[10])
                    px4 = float(strs[11])
                    py4 = float(strs[12])
                    px5 = float(strs[13])
                    py5 = float(strs[14])
                    # 半张脸
                    print(x1, y1, x2, y2)
                    if x1 < 0 or y1 < 0 or (x2 - x1) < 0 or (y2 - y1) < 0:
                        continue
                    # 坐标
                    boxes = [[x1, y1, x2, y2]]
                    # 中心
                    # 建议框：位置(中心点2、左上角右下角坐标4)、形状（4，2）
                    # （x1,y1,x2,y2）（cx1,cy1,width,height）
                    cx = int(x1 + (x2 - x1) / 2)
                    cy = int(y1 + (y2 - y1) / 2)
                    w = x2 - x1
                    h = y2 - y1
                    # 随机位置，抠样本
                    for _ in range(12):
                        try:
                            w_ = np.random.randint(-w * 0.2, w * 0.2 + 1)
                            h_ = np.random.randint(-h * 0.2, h * 0.2 + 1)
                        except:
                            continue
                        # 随机之后的中心点
                        cx_ = cx + w_
                        cy_ = cy + h_
                        # 抠图为正方形
                        side_len = np.random.randint(int(min(w, h) * 0.7), int(max(w, h) * 1.2))
                        # 偏移量
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len
                        crop_box = np.array([x1_, y1_, x2_, y2_])
                        # 偏移量

                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len
                        # 五官关键点的偏移量
                        offset_px1 = (px1 - x1_) / side_len
                        offset_py1 = (py1 - y1_) / side_len
                        offset_px2 = (px2 - x1_) / side_len
                        offset_py2 = (py2 - y1_) / side_len
                        offset_px3 = (px3 - x1_) / side_len
                        offset_py3 = (py3 - y1_) / side_len
                        offset_px4 = (px4 - x1_) / side_len
                        offset_py4 = (py4 - y1_) / side_len
                        offset_px5 = (px5 - x1_) / side_len
                        offset_py5 = (py5 - y1_) / side_len
                        # 抠图
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))
                        # 计算iou，判断样本类型
                        iou = utils.IOU(crop_box, np.array(boxes))[0]
                        if iou > 0.65:  # 正样本
                            # 写文件
                            # 文件名，置信度：1，建议框坐标的偏移量，关键点坐标的偏移量
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2,
                                    offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4,
                                    offset_py4, offset_px5, offset_py5
                                ))
                            positive_anno_file.flush()
                            # 保存抠图  celaba/celeba_output/12/positive
                            face_resize.save(f"{positive_image_dir}/{positive_count}.jpg")
                            positive_count += 1
                        elif 0.6 > iou > 0.40:
                            # part样本置信度：2
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2,
                                    offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4,
                                    offset_py4, offset_px5, offset_py5
                                ))
                            part_anno_file.flush()
                            # 保存抠图  celaba/celeba_output/12/positive
                            face_resize.save(f"{part_image_dir}/{part_count}.jpg")
                            part_count += 1
                        elif iou < 0.001:
                            # 负样本置信度：0
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                    negative_count, 0, offset_x1, offset_y1, offset_x2, offset_y2,
                                    offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4,
                                    offset_py4, offset_px5, offset_py5
                                ))
                            negative_anno_file.flush()
                            # 保存抠图  celaba/celeba_output/12/positive
                            face_resize.save(f"{negative_image_dir}/{negative_count}.jpg")
                            negative_count += 1
                    # 负样本不足
                    for i in range(100):
                        # np.random.randint(int(min(w, h) * 0.5), int(max(w, h) * 1.1))
                        try:
                            side_len = np.random.randint(face_size, min(img_w, img_h) / 2)

                            x_ = np.random.randint(0, img_w - side_len)
                            y_ = np.random.randint(0, img_h - side_len)
                            crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])
                            iou = utils.IOU(crop_box, np.array(boxes))[0]
                            if iou < 0.001:
                                face_crop = img.crop(crop_box)
                                face_resize = face_crop.resize((face_size, face_size))
                                # 写文件
                                negative_anno_file.write(
                                    "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                        negative_count, 0, offset_x1, offset_y1, offset_x2, offset_y2,
                                        offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3,
                                        offset_px4,
                                        offset_py4, offset_px5, offset_py5
                                    ))
                                negative_anno_file.flush()
                                # 保存抠图  celaba/celeba_output/12/positive
                                face_resize.save(f"{negative_image_dir}/{negative_count}.jpg")
                                negative_count += 1
                        except:
                            continue

    finally:
        # 关闭
        positive_anno_file.close()
        part_anno_file.close()
        negative_anno_file.close()
