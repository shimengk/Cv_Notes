import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw, ImageFont
from torchvision import transforms

from mtcnn import PNet, RNet, ONet
from week11_mtcnn.utils import utils

# 网络调参
# P网络:
p_cls = 0.6  # 原为0.6
p_nms = 0.1  # 原为0.5
# R网络：
r_cls = 0.6  # 原为0.6
r_nms = 0.5  # 原为0.5
# O网络：
o_cls = 0.95  # 原为0.97
o_nms = 0.5  # 原为0.7


# 侦测器
class Detector:
    # 初始化时加载三个网络的权重(训练好的)，cuda默认设为True
    def __init__(self, pnet_param=r"params_best/p_net.pt", rnet_param="params_best/r_net.pt",
                 onet_param="params_best/o_net.pt", isCuda=True):
        self.isCuda = isCuda
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        # 把预训练权重加载到网络
        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))
        self.pnet.eval(), self.rnet.eval(), self.onet.eval()
        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # 检测图片，传入原图
    def detect(self, image):

        # P网络检测
        pnet_boxes = self.__pnet_detect(image)
        pnet_boxes = np.array(pnet_boxes)
        # 没有人脸，返回空数组
        if pnet_boxes.shape[0] == 0:
            return np.array([[], [], []])

        # R网络检测：传入原图和P网络的筛选框
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        rnet_boxes = np.array(rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([pnet_boxes, [], []])

        # O网络检测：传入原图和R网络的筛选框
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        onet_boxes = np.array(onet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([pnet_boxes, rnet_boxes, []])

        return pnet_boxes, rnet_boxes, onet_boxes

    # P网络检测函数
    def __pnet_detect(self, image):
        boxes = []
        img = image
        w, h = img.size
        # 图片的最小边长
        min_side_len = min(w, h)
        # 缩放比例（为1时不缩放）:得到不同分辨率的图片
        scale = 1
        # 缩到小于等于12时停止
        while min_side_len > 12:
            # 将图片数组转成张量，CHW格式。
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            # 升维（CHW --> NCHW）
            img_data.unsqueeze_(0)
            # 返回多个置信度和偏移量
            _cls, _offset, _ = self.pnet(img_data)
            # NCHW --> HW  分组卷积特征图的尺寸
            cls = _cls[0][0].cpu().data
            # NCHW --> CHW  分组卷积特征图的通道、尺寸:C,W,H
            offset = _offset[0].cpu().data
            # 返回True（cls > 0.6）对应的索引值
            # ★置信度大于0.6的框索引；把P网络输出，看有没没框到的人脸，若没框到人脸，说明网络没训练好；或者置信度给高了、调低。
            idxs = torch.nonzero(torch.gt(cls, p_cls))
            for idx in idxs:
                # 根据索引，依次添加符合条件的框；cls[idx[0], idx[1]]在置信度中取值：idx[0]行索引，idx[1]列索引
                boxes.append(
                    # ★调用框反算函数_box（把特征图上的框，反算到原图上去），把大于0.6的框留下来；
                    self.__box(idx, offset, cls[idx[0], idx[1]], scale))
            scale *= 0.7  # 缩放图片：循环控制条件
            _w = int(w * scale)  # 新的宽度
            _h = int(h * scale)
            img = img.resize((_w, _h))  # 根据缩放后的宽和高，对图片进行缩放
            min_side_len = min(_w, _h)  # 重新获取最小宽高

        return utils.py_nms(np.array(boxes), p_nms)  # 返回框框，原阈值给p_nms=0.5（iou为0.5），尽可能保留IOU小于0.5的一些框下来，若网络训练的好，值可以给低些

    # 特征反算：将回归量还原到原图上去，根据特征图反算得到原图建议框
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):  # p网络池化步长为2
        # HW
        _x1 = (start_index[1].float() * stride) / scale  # 索引乘以步长，除以缩放比例；★特征反算时“行索引，索引互换”，原为[0]
        _y1 = (start_index[0].float() * stride) / scale
        _x2 = (start_index[1].float() * stride + side_len - 1) / scale
        _y2 = (start_index[0].float() * stride + side_len - 1) / scale

        ow = _x2 - _x1  # 人脸所在区域建议框的宽和高
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]  # 根据idxs行索引与列索引，找到对应偏移量△δ:[x1,y1,x2,y2]
        x1 = _x1 + ow * _offset[0]  # 根据偏移量算实际框的位置，x1=x1_+w*△δ；生样时为:△δ=x1-x1_/w
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]  # 正式框：返回4个坐标点和1个偏移量

    # R网络检测函数
    def __rnet_detect(self, image, pnet_boxes):
        _img_dataset = []  # 创建空列表，存放抠图
        pnet_boxes = np.array(pnet_boxes)
        _pnet_boxes = utils.convert_to_square(pnet_boxes)  # ★给p网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”，再抠图
        for _box in _pnet_boxes:  # ★遍历每个框，每个框返回框4个坐标点，抠图，放缩，数据类型转换，添加列表
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))  # 根据4个坐标点抠图
            img = img.resize((24, 24))  # 放缩在固尺寸
            img_data = self.__image_transform(img)  # 将图片数组转成张量
            _img_dataset.append(img_data)
        img_dataset = torch.stack(_img_dataset)  # stack堆叠(默认在0轴)，此处相当数据类型转换，见例子2★
        if self.isCuda:
            img_dataset = img_dataset.cuda()  # 给图片数据采用cuda加速
        _cls, _offset, _ = self.rnet(img_dataset)  # ★★将24*24的图片传入网络再进行一次筛选
        cls = _cls.cpu().data.numpy()  # 将gpu上的数据放到cpu上去，在转成numpy数组
        offset = _offset.cpu().data.numpy()
        # print("r_cls:",cls.shape)  # (11, 1):P网络生成了11个框
        # print("r_offset:", offset.shape)  # (11, 4)
        boxes = []  # R 网络要留下来的框，存到boxes里
        idxs, _ = np.where(
            cls > r_cls)  # 原置信度0.6是偏低的，时候很多框并没有用(可打印出来观察)，可以适当调高些；idxs置信度框大于0.6的索引；★返回idxs:0轴上索引[0,1]，_:1轴上索引[0,0]，共同决定元素位置，见例子3
        for idx in idxs:  # 根据索引，遍历符合条件的框；1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            ow = _x2 - _x1  # 基准框的宽
            oh = _y2 - _y1
            x1 = _x1 + ow * offset[idx][0]  # 实际框的坐标点
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            boxes.append([x1, y1, x2, y2, cls[idx][0]])  # 返回4个坐标点和置信度
        return utils.py_nms(np.array(boxes), r_nms)  # 原r_nms为0.5（0.5要往小调），上面的0.6要往大调;小于0.5的框被保留下来

    # O网络检测函数
    def __onet_detect(self, image, rnet_boxes):
        _img_dataset = []  # 创建列表，存放抠图r
        _rnet_boxes = utils.convert_to_square(rnet_boxes)  # 给r网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”
        for _box in _rnet_boxes:  # 遍历R网络筛选出来的框，计算坐标，抠图，缩放，数据类型转换，添加列表，堆叠
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))  # 根据坐标点“抠图”
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)  # 将抠出的图转成张量
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)  # 堆叠，此处相当数据格式转换，见例子2
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset, _landmark_offset = self.onet(img_dataset)
        cls = _cls.cpu().data.numpy()  # (1, 1)
        offset = _offset.cpu().data.numpy()  # (1, 4)
        landmark_offset = _landmark_offset.cpu().data.numpy()  # (1, 10)

        boxes = []  # 存放o网络的计算结果
        idxs, _ = np.where(
            cls > o_cls)  # 原o_cls为0.97是偏低的，最后要达到标准置信度要达到0.99999，这里可以写成0.99998，这样的话出来就全是人脸;留下置信度大于0.97的框；★返回idxs:0轴上索引[0]，_:1轴上索引[0]，共同决定元素位置，见例子3
        for idx in idxs:  # 根据索引，遍历符合条件的框；1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）
            _box = _rnet_boxes[idx]  # 以R网络做为基准框
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1  # 框的基准宽，框是“方”的，ow=oh
            oh = _y2 - _y1
            # O网络最终生成的框的坐标；生样，偏移量△δ=x1-_x1/w*side_len
            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            lx1 = _x1 + ow * landmark_offset[idx][0]
            ly1 = _y1 + oh * landmark_offset[idx][1]
            lx2 = _x1 + ow * landmark_offset[idx][2]
            ly2 = _y1 + oh * landmark_offset[idx][3]
            lx3 = _x1 + ow * landmark_offset[idx][4]
            ly3 = _y1 + oh * landmark_offset[idx][5]
            lx4 = _x1 + ow * landmark_offset[idx][6]
            ly4 = _y1 + oh * landmark_offset[idx][7]
            lx5 = _x1 + ow * landmark_offset[idx][8]
            ly5 = _y1 + oh * landmark_offset[idx][9]
            # 返回4个坐标点和1个置信度
            boxes.append([x1, y1, x2, y2, cls[idx][0], lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, lx5, ly5])
        # 用最小面积的IOU；原o_nms(IOU)为小于0.7的框被保留下来
        return utils.py_nms(np.array(boxes), o_nms, "Minimum")


if __name__ == '__main__':

    mode = 2

    if mode == 1:

        # 测试 test_images 下的图像
        font = ImageFont.truetype("font/arial.ttf", size=23)
        image_path = r"test_images"
        for idx, i in enumerate(os.listdir(image_path)):
            detector = Detector()
            # 画框
            with Image.open(os.path.join(image_path, i)) as im:
                try:
                    _,_,boxes = detector.detect(im)
                except:
                    continue
                print("图像尺寸：", im.size)
                imDraw = ImageDraw.Draw(im)
                for box in boxes:
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    print("坐标：", (x1, y1, x2, y2), "\t置信度：", box[4])
                    imDraw.rectangle((x1, y1, x2, y2), outline='red')
                    imDraw.text((x1, y1), "{:.4f}".format(box[4]), font=font, fill=(255, 0, 255))
                    imDraw.ellipse(xy=(box[5] - 2, box[6] - 2, box[5] + 2, box[6] + 2), fill=(0, 255, 0))
                    imDraw.ellipse(xy=(box[7] - 2, box[8] - 2, box[7] + 2, box[8] + 2), fill=(0, 255, 0))
                    imDraw.ellipse(xy=(box[9] - 2, box[10] - 2, box[9] + 2, box[10] + 2), fill=(0, 255, 0))
                    imDraw.ellipse(xy=(box[11] - 2, box[12] - 2, box[11] + 2, box[12] + 2), fill=(0, 255, 0))
                    imDraw.ellipse(xy=(box[13] - 2, box[14] - 2, box[13] + 2, box[14] + 2), fill=(0, 255, 0))

                im.save(f'test_results/img{idx}.jpg')

    else:
        # 视频检测
        with torch.no_grad() as grad:
            detector = Detector()
            # 摄像头
            cap = cv2.VideoCapture(r"C:\Users\Administrator\Downloads\dd.mp4")
            # 视频文件
            # cap = cv2.VideoCapture(r"test2.mp4")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的width
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的height
            out = cv2.VideoWriter('portrait6.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (w, h))

            print(w, h)
            while True:
                x = time.time()
                ret, photo = cap.read()
                if ret:
                    b, g, r = cv2.split(photo)
                    img = cv2.merge([r, g, b])
                else:
                    break
                im = Image.fromarray(img, "RGB")
                _,_,boxes = detector.detect(im)
                for i, box in enumerate(boxes):
                    box = list(map(int, box))
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    cv2.rectangle(photo, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BG
                    circle_radius = 5
                    color = (0, 255, 0)
                    cv2.circle(photo, (box[5], box[6]), radius=circle_radius, color=color, thickness=-1)
                    cv2.circle(photo, (box[7], box[8]), radius=circle_radius, color=color, thickness=-1)
                    cv2.circle(photo, (box[9], box[10]), radius=circle_radius, color=color, thickness=-1)
                    cv2.circle(photo, (box[11], box[12]), radius=circle_radius, color=color, thickness=-1)
                    cv2.circle(photo, (box[13], box[14]), radius=circle_radius, color=color, thickness=-1)

                cv2.imshow("capture", photo)
                out.write(photo)
                y = time.time()
                # timelag = y - xq
                # print(timelag)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()  # 关闭视频
            cv2.destroyAllWindows()  # 关闭窗口
