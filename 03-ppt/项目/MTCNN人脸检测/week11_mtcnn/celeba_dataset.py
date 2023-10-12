import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 12, 24 48
class CelebaDataset(Dataset):
    def generate_lines(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield line

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.dataset = []

        self.dataset.extend(open(f"{self.path}/positive.txt").readlines())
        self.dataset.extend(open(f"{self.path}/negative.txt").readlines())
        self.dataset.extend(open(f"{self.path}/part.txt").readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].split()
        # 置信度，偏移量
        conf = torch.Tensor([int(strs[1])])
        offset = torch.Tensor(
            [
                float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])
            ]
        )
        landmark = torch.Tensor([
            float(strs[6]), float(strs[7]), float(strs[8]), float(strs[9]),
            float(strs[10]), float(strs[11]), float(strs[12]), float(strs[13]),
            float(strs[14]), float(strs[15])

        ])
        # 返回：数据，置信度，建议框的偏移量
        # 图片路径
        img_path = f"{self.path}/{strs[0]}"
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255 - 0.5)
        # shape : HWC --> CHW
        img_data = img_data.permute(2, 0, 1)
        return img_data, conf, offset, landmark
