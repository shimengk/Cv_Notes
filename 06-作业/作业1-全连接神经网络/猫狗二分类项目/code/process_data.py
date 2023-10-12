import os
import random
import shutil

from PIL import Image
from torchvision import transforms


def pre_process_data(src_dir, tgt_dir, size, is_gray):
    """
    将以下文件夹的文件resize，并存储到目标目录，目录结构：
        src_dir
        ├─ cat
        └─ dog
    resize to
        tgt_dir
        ├─ cat
        └─ dog
    :param src_dir: 源目录
    :param tgt_dir: 目标目录
    :param size: 目标图像的尺寸
    :param is_gray: 是否单通道
    :return:
    """

    def cut_img(img, size):
        h, w = img.size
        base_trans = transforms.Compose([
            transforms.CenterCrop(min(h, w)),
            transforms.Resize(size)
        ])
        return base_trans(img)

    labels = os.listdir(src_dir)
    for label in labels:
        label_dir = os.path.join(src_dir, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            try:
                img = Image.open(file_path)
                # 图片中心裁剪为指定尺寸
                img = cut_img(img, size)
                # resize不同三个尺寸到文件夹中，用于做数据的对比
                if is_gray:
                    img = transforms.Compose([transforms.Grayscale()])(img)

                tgt_path = os.path.join(tgt_dir, label)
                if not os.path.exists(tgt_path):
                    os.makedirs(tgt_path)
                # 写入
                print(os.path.join(tgt_path, file_name))
                img.save(os.path.join(tgt_path, file_name), 'JPEG')
            except:
                print(f'{file_name} save failed')


def split_train_or_test(data_dir, categories):
    """
    划分测试机和数据集,例如将：
        data_dir
        ├─ cat
        └─ dog
    划分为：
        data_dir
        ├─ train
        │    ├─ cat
        │    ├─ dog
        ├─ test
        │    ├─ cat
        │    ├─ dog
    :param data_dir: 数据路径
    :param categories: 分类类别
    :return:
    """
    # 定义数据路径

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 定义训练集和测试集比例
    train_ratio = 0.8  # 训练集比例
    test_ratio = 0.2  # 测试集比例

    # 创建训练集和测试集文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历每个类别
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)

        # 获取类别下的所有图像文件
        image_files = os.listdir(category_dir)

        # 随机打乱图像文件列表
        random.shuffle(image_files)

        # 计算训练集和测试集的分割点
        split_index = int(len(image_files) * train_ratio)

        # 将图像文件复制到训练集和测试集文件夹中
        for i, image_file in enumerate(image_files):
            src_path = os.path.join(category_dir, image_file)
            if i < split_index:
                dst_path = os.path.join(train_category_dir, image_file)
            else:
                dst_path = os.path.join(test_category_dir, image_file)
            shutil.copyfile(src_path, dst_path)


# 数据增样到指定数目
def add_data(path, max_num):
    def random_add_trans(img, size):
        h, w = img.size
        # 随机增样
        random_add_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            transforms.RandomRotation(degrees=10),  # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 调整亮度、对比度、饱和度和色调
            transforms.CenterCrop(min(h, w)),
            transforms.Resize(size)
        ])
        return random_add_trans(img)

    label_dir_path = path
    data_sum = len(os.listdir(label_dir_path))
    while data_sum < max_num:
        data_sum += 1
        # 随机选择一张：
        img_name = os.listdir(label_dir_path)[random.randint(0, data_sum - 2)]
        img_path = os.path.join(label_dir_path, img_name)
        print(img_path)
        img = Image.open(img_path)
        add_img = random_add_trans(img, 64)
        add_img.save(os.path.join(label_dir_path, f'a{data_sum}.jpg'))
        print(f'a{data_sum}.jpg save success')


if __name__ == '__main__':
    # 100*100*1的数据
    # pre_process_data(r'E:\kagglecatsanddogs_5340\PetImages\Dog', r'E:\data\cat_dog\100-100-1', 'dog', 100, True)
    # pre_process_data(r'E:\kagglecatsanddogs_5340\PetImages\Cat', r'E:\data\cat_dog\100-100-1', 'cat', 100, True)
    # # 64*64*1的数据
    # pre_process_data(r'E:\kagglecatsanddogs_5340\PetImages\Dog', r'E:\data\cat_dog\64-64-1', 'dog', 64, True)
    # pre_process_data(r'E:\kagglecatsanddogs_5340\PetImages\Cat', r'E:\data\cat_dog\64-64-1', 'cat', 64, True)
    # # 32*32*3的数据
    # pre_process_data(r'E:\kagglecatsanddogs_5340\PetImages\Dog', r'E:\data\cat_dog\32-32-3', 'dog', 32, False)
    # pre_process_data(r'E:\kagglecatsanddogs_5340\PetImages\Cat', r'E:\data\cat_dog\32-32-3', 'cat', 32, False)

    # 将该目录下的文件划分为train  test
    # 参数1：源路径，参数2：分类
    # split_train_or_test(r'E:\data\cat_dog\100-100-1', ["cat", "dog"])
    # split_train_or_test(r'E:\data\cat_dog\64-64-1', ["cat", "dog"])
    # split_train_or_test(r'E:\data\cat_dog\32-32-3', ["cat", "dog"])

    # pre_process_data(r'E:\kagglecatsanddogs_5340\PetImages', r'E:\data\cat_dog\100-100-3', 100, False)
    # split_train_or_test(r'E:\data\cat_dog\100-100-3', ["cat", "dog"])
    # 数据增样
    # add_data(r'E:\data\cat_dog\64-64-1\dog', 20000)
    # add_data(r'E:\data\cat_dog\64-64-1\cat', 20000)

    # 添加64*64*3的数据
    # pre_process_data(r'E:\kagglecatsanddogs_5340\PetImages', r'E:\data\cat_dog\64-64-3', 64, False)
    # 增样
    # add_data(r'E:\data\cat_dog\64-64-3\dog', 20000)
    # add_data(r'E:\data\cat_dog\64-64-3\cat', 20000)
    # # 分隔
    # split_train_or_test(r'E:\data\cat_dog\64-64-3', ["cat", "dog"])

    # pre_process_data(r'E:\data\Animals-10\raw-img', r'E:\data\animals10\64-64-3', 64, False)
    #
    tgt_dir = r'E:\data\animals10\64-64-3'
    # for label in os.listdir(tgt_dir):
    #     add_data(os.path.join(tgt_dir, label), 12500)
    # # 分隔
    split_train_or_test(r'E:\data\animals10\64-64-3', os.listdir(tgt_dir))
