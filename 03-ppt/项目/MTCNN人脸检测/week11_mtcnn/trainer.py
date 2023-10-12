import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from celeba_dataset import CelebaDataset


# 创建训练器
class Trainer:
    # 网络，参数保存路径，训练数据路径，cuda加速为True
    def __init__(self, net, save_path, dataset_path, batch_size):
        print(f'{net._get_name()}初始化...')
        self.batch_size = batch_size
        self.net = net.cuda()
        self.save_path = save_path
        self.dataset_path = dataset_path
        # 置信度损失
        # nn.BCELoss()：二分类交叉熵损失函数，使用之前用sigmoid()激活
        self.conf_loss_fc = nn.BCELoss()
        # 偏移量损失
        self.offset_loss_fc = nn.MSELoss()
        self.landmark_loss_fc = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))
            print(f'{net._get_name()}成功加载权重...')

    # 训练
    def train(self):
        faceDataset = CelebaDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=self.batch_size, shuffle=True)
        print(f'{self.net._get_name}的数据集数量：{len(dataloader)}')
        for epoch in range(1, 10000):
            for i, (img, conf, offset, landmark) in enumerate(dataloader):
                # 样本输出，置信度，偏移量
                img, conf, offset, landmark = img.cuda(), conf.cuda(), offset.cuda(), landmark.cuda()
                # 输出置信度，偏移量
                out_conf, out_offset, out_landmark = self.net(img)
                out_conf = out_conf.reshape(-1, 1)
                out_offset = out_offset.reshape(-1, 4)
                out_landmark = out_landmark.reshape(-1, 10)
                # 求置信度损失
                # 置信度掩码：求置信度只考虑正、负样本，不考虑部分样本。
                conf_mask = torch.lt(conf, 2)
                # 标签：根据置信度掩码，筛选出置信度为0、1的正、负样本。
                conf_mask_value = torch.masked_select(conf, conf_mask)
                # 网络输出值：预测的“标签”进掩码，返回符合条件的结果
                out_conf_value = torch.masked_select(out_conf, conf_mask)
                # 对置信度做损失
                conf_loss = self.conf_loss_fc(out_conf_value, conf_mask_value)

                # 求偏移量损失：不考虑负样本，只考虑正、部分样本
                offset_mask = torch.gt(conf, 0.)
                # 对置信度大于0的标签，进行掩码；★负样本不参与计算，负样本没偏移量
                offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引
                offset = offset[offset_index]  # 标签里偏移量
                out_offset = out_offset[offset_index]  # 输出的偏移量
                offset_loss = self.offset_loss_fc(out_offset, offset)  # 偏移量损失

                # landmark的损失
                landmark = landmark[offset_index]
                out_landmark = out_landmark[offset_index]
                landmark_loss = self.landmark_loss_fc(out_landmark, landmark)
                # 总损失
                loss = 0.1 * conf_loss + 0.1 * offset_loss + 0.8 * landmark_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print(
                        f"{self.net._get_name()}第{epoch}轮{i}次的总损失：{loss.item()}，置信度损失：{conf_loss.item()}，"
                        f"偏移量损失：{offset_loss.item()}，关键点损失：{landmark_loss.item()}")
                    torch.save(self.net.state_dict(), self.save_path)
