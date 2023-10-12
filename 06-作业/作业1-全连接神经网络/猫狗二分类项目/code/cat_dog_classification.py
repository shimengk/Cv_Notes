import os

import cv2
import numpy as np
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import fc_net


class Cat_Dog_Dataset(Dataset):
    def __init__(self, root=r'E:\data\cat_dog\64-64-3', isTrain=True, isGray=False):
        """
        :param root: 数据集根目录
        :param isTrain: 是否是训练集
        :param isGray: 是否转灰度图
        """
        super().__init__()
        self.dataset = []
        self.isGray = isGray
        # 读文件
        path = os.path.join(root, 'train' if isTrain else 'test')
        for label in os.listdir(path):
            for img_name in os.listdir(os.path.join(path, label)):
                img_path = os.path.join(path, label, img_name)
                self.dataset.append((img_path, 0 if label == 'cat' else 1))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        data = self.dataset[idx]
        try:
            if self.isGray:
                img = cv2.imread(data[0], 0)
            else:
                img = cv2.imread(data[0])
            # 由于全连接是NV结构，需要对数据降维
            img = img.reshape(-1)
        except:
            print(f'{data[0]}读取失败')
            img = np.full((100, 100), fill_value=220, dtype=np.uint8)
        # 归一化数据
        img = img / 255
        # 标签进行one-hot
        ont_hot = np.zeros(2)
        ont_hot[int(data[1])] = 1
        return np.float32(img), np.float32(ont_hot)


class Trainer:
    # 定义数据集
    root_path = r'E:\data\cat_dog'
    # 不同尺寸图片的目录
    data_path = {
        "32": os.path.join(root_path, '32-32-3'),
        "64": os.path.join(root_path, '64-64-3'),
        "100": os.path.join(root_path, '100-100-3'),
    }

    def __init__(self):

        train_dataset = Cat_Dog_Dataset(root=self.data_path['64'], isGray=False)
        test_dataset = Cat_Dog_Dataset(root=self.data_path['64'], isGray=False, isTrain=False)
        # 定义数据加载器
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

        # 当前设备环境
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 定义网络
        self.net = fc_net.FcNet3()
        self.net.to(self.device)
        self.loss = torch.nn.MSELoss()

        # 定义优化器
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.001)

        self.writer = SummaryWriter('results')

    def train(self):
        for epoch in range(1, 1000):
            sum_loss = 0.
            for i, (img, label) in enumerate(self.train_loader):
                # 网络开启训练模式
                self.net.train()
                # 数据放到device
                img, label = img.to(self.device), label.to(self.device)

                # 前向计算
                out = self.net(img)

                # 计算损失
                loss = self.loss(out, label)
                # loss = torch.mean((label - y) ** 2)

                # 清空过往梯度
                self.optim.zero_grad()
                # 反向传播 计算梯度，并将梯度值存储在每个参数的 grad属性中。
                loss.backward()
                # 更新模型参数：优化器根据梯度值和指定的学习率更新模型的参数值
                self.optim.step()

                sum_loss += loss

            avg_loss = sum_loss / len(self.train_loader)
            print(f'Train Epoch: {epoch},  Loss: {avg_loss}')
            # if epoch % 10 == 0:
            #     self.test(1)

            # 使用TensorBoard图形化显示
            # 命令行启动board：tensorboard --logdir=result --port=8899
            self.writer.add_scalar('loss', avg_loss, epoch)
            # 写入权重文件
            torch.save(self.net.state_dict(), f'params6//{epoch}.pt')

    def test(self, test_epoch):
        # 把最优的训练效果进行测试
        # self.net.load_state_dict(torch.load(r'params2//' + os.listdir(r'params2')[-1]))
        self.net.load_state_dict(torch.load(r'params6/222.pt'))
        for epoch in range(test_epoch):
            sum_score = 0
            for i, (img, label) in enumerate(self.test_loader):
                # 测试模式
                self.net.eval()
                img, label = img.to(self.device), label.to(self.device)

                h = self.net(img)

                # h为网络计算答案,label标准答案
                a = torch.argmax(h, dim=1)
                b = torch.argmax(label, dim=1)
                # 当前批次得分
                score = torch.mean(torch.eq(a, b).float())
                sum_score += score.item()
            avg_score = sum_score / len(self.test_loader)
            print(f'Test Score: {avg_score}')
            self.writer.add_scalar('score', avg_score, epoch)

        pass


if __name__ == '__main__':
    # 数据预处理
    # mnist_to_img()
    # 实例化训练器
    trainer = Trainer()
    trainer.train()
    # trainer.test(20)
