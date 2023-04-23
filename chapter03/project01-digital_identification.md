# 手写数字识别

手写数字识别相当于ML中的"hello world"，新手可以使用MNIST数据集进行练习。

## 环境准备

```
pytorch
torchvision
numpy
```

## 数据集MNIST

MNIST数据集（Mixed National Institute of Standards and Technology database）是一个用来训练各种图像处理系统的二进制图像数据集，广泛应用于机器学习中的训练和测试。

它包含60000个训练样本集和10000个测试样本集，图像的存储形式是28*28像素单通道。



1. 下载数据集

```python
from torchvision import datasets, transforms

# 训练集
train_dataset = datasets.MNIST(root='datafiles/', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
train_dataset = datasets.MNIST(root='datafiles/', train=False, transform=transforms.ToTensor(), download=True)


```

datasets.MNIST()是torchvision下的内置函数，可以导入数据集

train=True 表示读入的数据作为训练集

transform是读取数据预处理操作

download=True表示当前根目录如果没有数据集时，自动下载



运行后下载的文件目录结构如下：

![image-20230423150421472](assets/image-20230423150421472.png)

2. 加载数据集

DataLoader为torch内部函数，可以这样使用

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
```





