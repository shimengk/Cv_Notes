import os.path
import pathlib
import time

import numpy as np
import torch.cuda
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from week08_plane.net import vgg, convnet


class PlaneDataset(Dataset):

    def __init__(self, root, isTrain=True):
        super().__init__()
        self.dataset = []

        path = os.path.join(root, 'train' if isTrain else 'test')

        for label in os.listdir(path):
            for img_name in os.listdir(os.path.join(path, label)):
                img_path = os.path.join(path, label, img_name)
                self.dataset.append((img_path, label))
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 调整图片尺寸为模型所需尺寸
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对应红色、绿色和蓝色通道的均值和标准差
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        img = self.transform(Image.open(data[0]).convert('RGB'))

        # 标签进行one-hot
        ont_hot = np.zeros(9)
        ont_hot[int(data[1])] = 1
        return img, np.float32(ont_hot)


class Trainer:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        root = r'E:\data\plane2'

        train_dataset = PlaneDataset(root=root)
        test_dataset = PlaneDataset(root=root, isTrain=False)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
        self.classification = ['滑翔伞', '客机', '前掠翼', '三角翼', '双翼机', '水上飞机', '四旋翼', '战斗机', '直升飞机']
        # 定义网络
        # self.net = convnet.resnet18(classes_num=9, version=3)
        self.net = vgg.VGG(vgg_name='VGG11')

        self.net.to(self.DEVICE)
        # 预加载权重
        # self.net.load_state_dict(torch.load(r'E:\pyprojects\zhenshu_caokj\week08_plane\params3\plane_vgg.pt'))
        # 损失函数
        self.loss = nn.CrossEntropyLoss()
        # 优化器
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.005)

        self.log_dir_path = r'logs/log1'
        self.param_path = r'params1'
        pathlib.Path(self.log_dir_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.param_path).mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir_path)

    def train_test(self):
        for epoch in range(1, 1000):
            sum_loss = 0.
            start_time = time.time()
            for i, (img, target) in enumerate(self.train_loader):
                self.net.train()
                img, target = img.to(self.DEVICE), target.to(self.DEVICE)
                # 网络计算
                out = self.net(img)
                train_loss = self.loss(out, target)

                self.optim.zero_grad()
                train_loss.backward()
                self.optim.step()

                sum_loss += train_loss.item()

            avg_loss = sum_loss / len(self.train_loader)
            print(f'Epoch: {epoch}, Loss: {avg_loss}, time: {time.time() - start_time}')

            self.writer.add_scalar('loss', avg_loss, epoch)
            # 写入权重文件
            torch.save(self.net.state_dict(), os.path.join(self.param_path, 'plane_vgg.pt'))

            # 测试
            if epoch % 5 == 0:
                self.test(epoch)

    def test(self, epoch):
        self.net.load_state_dict(torch.load(r'E:\pyprojects\zhenshu_caokj\week08_plane\params4_pool\plane_vgg.pt'))

        sum_score = 0.
        for i, (img, target) in enumerate(self.test_loader):
            self.net.eval()
            img, target = img.to(self.DEVICE), target.to(self.DEVICE)

            out = self.net(img)

            out_inx = torch.argmax(out, dim=1)
            target = torch.argmax(target, dim=1)
            score = torch.mean(torch.eq(out_inx, target).float())
            sum_score += score.item()

        avg_score = sum_score / len(self.test_loader)

        # self.writer.add_scalar('score', avg_score, epoch)

        print(f"Test Score: {avg_score}")

    def predict_image(self, image_path):
        self.net.eval()
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 调整图片尺寸为模型所需尺寸
            transforms.ToTensor(),  # 转换为张量
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(self.DEVICE)  # 添加batch维度并将图像移动到设备上
        self.net.load_state_dict(torch.load(r'E:\pyprojects\zhenshu_caokj\week08_plane\params9_vgg\plane_vgg.pt'))
        with torch.no_grad():
            output = self.net(image)

        probabilities = torch.softmax(output, dim=1)  # 对输出进行softmax处理
        _, predicted_class = torch.max(probabilities, 1)  # 获取预测的类别索引

        print(f'预测的是  {predicted_class.item()}')

    def test_each_classification(self):
        self.net.load_state_dict(torch.load(r'E:\pyprojects\zhenshu_caokj\week08_plane\params9_vgg\plane_vgg.pt'))

        class_correct = [0] * 9
        class_total = [0] * 9

        for i, (img, target) in enumerate(self.test_loader):
            self.net.eval()
            img, target = img.to(self.DEVICE), target.to(self.DEVICE)

            out = self.net(img)

            out_inx = torch.argmax(out, dim=1)
            target_inx = torch.argmax(target, dim=1)
            score = torch.mean(torch.eq(out_inx, target_inx).float())

            # Update class-wise accuracy
            for j in range(len(target_inx)):
                label = target_inx[j]
                class_correct[label] += torch.eq(out_inx[j], label).item()
                class_total[label] += 1

        class_accuracy = []
        for i in range(9):
            accuracy = 100 * class_correct[i] / class_total[i]
            class_accuracy.append(accuracy)

        print("Test Accuracy per Class:")
        for i in range(9):
            print(f"{self.classification[i]}: {class_accuracy[i]:.2f}%")

        avg_score = sum(class_accuracy) / len(class_accuracy)
        print(f"Average Test Score: {avg_score:.2f}%")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train_test()
    # trainer.test_each_classification()
