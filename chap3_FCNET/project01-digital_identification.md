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
test_dataset = datasets.MNIST(root='datafiles/', train=False, transform=transforms.ToTensor(), download=True)


```

datasets.MNIST()是torchvision下的内置函数，可以导入数据集

train=True 表示读入的数据作为训练集

transform是读取数据预处理操作

download=True表示当前根目录如果没有数据集时，自动下载



运行后下载的文件目录结构如下：

![目录结构](E:\pyprojects\Cv_Notes\chap3_FCNET\assets\cca3818c44a740e39be259d4119924f4.png)


此时的数据集是二进制形式，需要使用加载器进行解析


2. 加载数据集

DataLoader为torch内部函数，可以这样使用

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
```

此时train_dataset.data的形状为 ` torch.Size([60000, 28, 28]) `，即60000张28*28像素的单通道图片。一张图片的存储格式如下：

```python
tensor([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,  51, 159, 253, 159,  50,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 48, 238, 252, 252, 252, 237,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  54, 227, 253, 252, 239, 233, 252,  57,   6,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  10,  60, 224, 252, 253, 252, 202,  84, 252, 253, 122,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 163, 252, 252, 252, 253, 252, 252,  96, 189, 253, 167,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  51, 238, 253, 253, 190, 114, 253, 228,  47,  79, 255, 168,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,  48, 238, 252, 252, 179, 12,  75, 121,  21,   0,   0, 253, 243,  50,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,  38, 165, 253, 233, 208,  84, 0,   0,   0,   0,   0,   0, 253, 252, 165,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   7, 178, 252, 240,  71,  19,  28, 0,   0,   0,   0,   0,   0, 253, 252, 195,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,  57, 252, 252,  63,   0,   0,   0, 0,   0,   0,   0,   0,   0, 253, 252, 195,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0, 198, 253, 190,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0, 255, 253, 196,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  76, 246, 252, 112,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0, 253, 252, 148,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  85, 252, 230,  25,   0,   0,   0,   0, 0,   0,   0,   0,   7, 135, 253, 186,  12,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  85, 252, 223,   0,   0,   0,   0,   0, 0,   0,   0,   7, 131, 252, 225,  71,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  85, 252, 145,   0,   0,   0,   0,   0, 0,   0,  48, 165, 252, 173,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  86, 253, 225,   0,   0,   0,   0,   0, 0, 114, 238, 253, 162,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  85, 252, 249, 146,  48,  29,  85, 178, 225, 253, 223, 167,  56,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  85, 252, 252, 252, 229, 215, 252, 252, 252, 196, 130,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,  28, 199, 252, 252, 253, 252, 252, 233, 145,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,  25, 128, 252, 253, 252, 141,  37, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],
       dtype=torch.uint8)

```
以上代码即为下图的存储数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/99014cc7915447a8b46f35b41016c72e.png)

## 定义网络
自定义网络需要继承nn.Module
全连接的结构如下
![在这里插入图片描述](E:\pyprojects\Cv_Notes\chap3_FCNET\assets\adee5c2528a34a78addeebdfa36ecffc.png)
![在这里插入图片描述](E:\pyprojects\Cv_Notes\chap3_FCNET\assets\88896770d6c34b1791f69de03e29d925.png)

我们使用`nn.Linear(in_features: int, out_features: int, bias: bool = True)`定义全连接的隐藏层。Linear的内部核心是一个大小为`out_features X in_features`的权重矩阵。

例如我们网络的第一层：`nn.Linear(784, 512)`
我们输入的图像最后变成了784个向量值，输入到了Linear中，Linear会输出个数为512的向量。那么根据$$ Y=w\cdot X  + b$$公式：
权重矩阵w应该是512 * 784的矩阵。
$$
\begin{bmatrix}    w_{11}  & w_{13} & \cdots & w \\    w_{21} & w_{22} & \cdots & w \\    \vdots & \vdots & \ddots & \vdots \\    w & w & \cdots & w_{512,784}  \end{bmatrix} \cdot \begin{bmatrix} x_{1}\\  x_{2}\\ x_{3}\\  \dots \\ x_{784}\end{bmatrix}  =\begin{bmatrix} y_{1}\\ y_{2}\\ \cdots\\ y_{512}\end{bmatrix}
$$

在Linear后，使用ReLU进行激活。激活函数的作用：加入非线性因素，增强模型的表达能力。
> 关于激活函数，详细可查看[这篇文章]( https://zhuanlan.zhihu.com/p/427541517)


因为MNIST数据集是做十分类，因此最后一层应该输出个数为10的向量，并且需要用softmax函数进行归一化操作。
softmax的主要作用是：
1. 将预测结果转化为非负数
2. 各种预测结果概率之和等于1

最终得到0-9十个数字的对应概率，且这些概率的和为1

```python
# 定义全连接网络
class FC_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            # 图像为1*28*28，因此输入参数为784
            nn.Linear(784, 512),
            # 激活函数使用relu
            nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
            # 最后使用sigmoid激活
            nn.Sigmoid(dim=1)
        )
        pass

    def forward(self, x):
        return self.fc_layer(x)
```

## 定义训练器
训练的流程如下：
1. 一张数据放入网络计算
2. 求损失
3. 清空过往梯度   optimizer.zero_grad()
4. 反向传播，计算当前梯度  loss.backward()
5. 根据梯度更新网络参数   optimizer.step()

循环进行以上操作。所以，每次计算都会对权值w和偏置b进行更新操作
> 关于optimizer.zero_grad(), loss.backward(), optimizer.step()的作用及原理可以参考[这篇文章](https://blog.csdn.net/PanYHHH/article/details/107361827)

>梯度下降是一种优化算法，用于最小化损失函数。在神经网络的训练中，我们希望寻找最优的参数组合，以使模型能够更准确地预测目标值。梯度下降算法的基本思想是不断调整当前参数的值，以使损失函数的值不断降低。
>具体来说，梯度下降的过程中，我们首先随机初始化一组参数，并计算其对应的损失函数值。然后根据损失函数对参数的导数（即梯度），计算出下降的方向，并以一定的步长移动参数的值，直至达到损失函数的最小值或达到指定的迭代次数。

可以查看[这篇文章](https://blog.csdn.net/m0_63167598/article/details/123535339)

>梯度清零是指在训练神经网络时，由于网络层数过多或激活函数过于复杂等原因，导致梯度值变得非常小或非常大，出现梯度消失或梯度爆炸的现象。梯度清零的目的是通过某些方法调整梯度值的大小和方向，以避免梯度消失或梯度爆炸的问题，从而更好地训练神经网络。
>具体来说，梯度清零的方法包括梯度剪裁、归一化梯度、自适应方法等。梯度剪裁是指限制梯度的大小，以避免梯度爆炸的问题；归一化梯度是指将梯度值等比例缩小，以避免梯度消失的问题；自适应方法是指根据梯度值的大小和方向来自适应地调整学习率的大小，以更好地平衡速度和精度的问题。

```python
class Trainer:
    def __init__(self):
        # 网络模型
        self.net = FC_Net()
        # 装载训练集和测试集
        # 定义优化器
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        pass

    def train(self):
        for epoch in range(10000):
            # 总损失
            sum_loss = 0
            for i, (data, label) in enumerate(self.train_loader):
            	# label 进行one-hot处理
                label = torch.nn.functional.one_hot(label)
                # 模型训练，打开训练模式
                self.net.train()
                # 前向计算
                h = self.net(data.reshape(-1, 784))
                # 求损失 使用均方差公式
                loss = torch.mean((h - label) ** 2)
                # 清空过往梯度
                self.opt.zero_grad()
                # 反向传播，计算当前梯度
                loss.backward()
                # 根据梯度更新网络参数
                self.opt.step()

                sum_loss += loss
                # 将训练好的权重文件保存
                torch.save(self.net.state_dict(), f'params//{i}.pth')

            avg_loss = sum_loss / len(self.train_loader)

            print('Train Epoch: {} [{}/{} ({:.2f}%)]   Loss: {:.6f}'.format(
                epoch, epoch, self.train_epoch, 100. * epoch / self.train_epoch, avg_loss.item()))

    
```

# 完整代码
```python
# %%
import os
import torch.cuda
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

# 训练集
train_dataset = datasets.MNIST(root='datafiles/', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
test_dataset = datasets.MNIST(root='datafiles/', train=False, transform=transforms.ToTensor(), download=True)

# 自定义全连接网络
class FC_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layer = nn.Sequential(
            # 图像为1*28*28，因此输入参数为784
            nn.Linear(784, 512),
            # 激活函数使用relu
            nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
            # 最后使用Softmax激活
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc_layer(x)


class Trainer:
    def __init__(self):
        # 定义训练参数
        # 学习率
        self.lr = 0.001
        # 训练集批次
        self.train_batch_size = 512
        # 测试集批次
        self.test_batch_size = 256
        # 训练迭代次数
        self.train_epoch = 10000
        # 测试迭代次数
        self.test_epoch = 1000

        # 判断是否有cuda，如果有，将net放到gpu进行训练
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 网络模型
        self.net = FC_Net()
        # 将网络放入cuda或cpu
        self.net.to(self.device)

        # 装载训练集和测试集
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=True)

        # 定义优化器
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        pass

    def train(self):
        train_epoch = self.train_epoch
        for epoch in range(train_epoch):
            # 总损失
            sum_loss = 0
            for i, (data, label) in enumerate(self.train_loader):
                # 模型训练，打开训练模式
                self.net.train()
                # 数据放入cuda
                data, label = data.to(self.device), label.to(self.device)
                data = data.reshape(-1, 784)
                # print(data.shape)
                # label 进行one-hot处理
                label = torch.nn.functional.one_hot(label)
                self.net.train()
                # 前向计算
                h = self.net(data)
                # 求损失 使用均方差公式
                loss = torch.mean((h - label) ** 2)
                # 清空过往梯度
                self.opt.zero_grad()
                # 反向传播，计算当前梯度
                loss.backward()
                # 根据梯度更新网络参数
                self.opt.step()

                sum_loss += loss
                # 将训练好的权重文件保存
                torch.save(self.net.state_dict(), f'params//{i}.pth')

            avg_loss = sum_loss / len(self.train_loader)

            print('Train Epoch: {} [{}/{} ({:.2f}%)]   Loss: {:.6f}'.format(
                epoch, epoch, self.train_epoch, 100. * epoch / self.train_epoch, avg_loss.item()))

    def test(self):
        # 载入最优的训练结果，进行测试
        self.net.load_state_dict(torch.load(r'params//' + os.listdir(r'params')[-1]))

        for epoch in range(self.test_epoch):
            sum_score = 0
            for i, (img, label) in enumerate(self.test_loader):
                # 测试模式
                self.net.eval()
                img, label = img.to(self.device), label.to(self.device)
                img = img.reshape(-1, 784)

                # 网络计算答案
                h = self.net(img)
                a = torch.argmax(h, dim=1)
                # 标准答案
                label = torch.nn.functional.one_hot(label)
                b = torch.argmax(label, dim=1)

                # 计算当前批次的得分
                score = torch.mean(torch.eq(a, b).float())
                sum_score += score
            avg_score = sum_score / len(self.test_loader)
            print('Test Epoch: {} [{}/{} ({:.2f}%)]   Score: {:.6f}'.format(
                epoch, epoch, self.test_epoch, 100. * epoch / self.test_epoch, avg_score.item()))

        pass


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.train()
    trainer.test()

```
