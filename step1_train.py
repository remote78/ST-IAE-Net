# -*-coding:utf-8 -*-

"""
#-------------------------------
# @Author : 肆十二
# @QQ : 3045834499 可定制毕设
#-------------------------------
# @File : step1_train.py
# @Description: 模型训练
# @Software : PyCharm
# @Time : 2024/2/14 10:48
#-------------------------------
"""

from model.unet_model import UNet
from utils.dataset import Dateset_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


# 定义 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1.
        pred = torch.sigmoid(pred)  # 将 logits 转换为概率
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice


# 定义 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = nn.BCEWithLogitsLoss()(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss


# 定义 Tversky Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        smooth = 1.
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        fps = (pred * (1 - target)).sum()
        fns = ((1 - pred) * target).sum()
        tversky = (intersection + smooth) / (intersection + self.alpha * fps + self.beta * fns + smooth)
        return 1 - tversky


# 定义融合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss()

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        return self.alpha * bce + self.beta * dice + self.gamma * focal + self.delta * tversky


def train_net(net, device, data_path, epochs=1200, batch_size=1, lr=0.0001):
    '''
    :param net: 语义分割网络
    :param device: 网络训练所使用的设备
    :param data_path: 数据集的路径
    :param epochs: 训练的轮数
    :param batch_size: 批次大小
    :param lr: 学习率
    :return:
    '''
    # 加载数据集
    dataset = Dateset_Loader(data_path)
    per_epoch_num = len(dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08, amsgrad=False)
    # 定义融合损失函数
    criterion = CombinedLoss(alpha=0.4, beta=0.3, gamma=0.2, delta=0.1)
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 开始训练
    loss_record = []
    with tqdm(total=epochs * per_epoch_num) as pbar:
        for epoch in range(epochs):
            # 训练模式
            net.train()
            # 按照batch_size开始训练
            for image, label in train_loader:
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                pbar.set_description("Processing Epoch: {} Loss: {}".format(epoch + 1, loss))
                # 如果当前的损失比最好的损失小，则保存当前轮次的模型
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model1.pth')
                loss.backward()
                optimizer.step()
                pbar.update(1)
            # 记录每轮的损失
            loss_record.append(loss.item())

    # 绘制loss折线图
    plt.figure()
    # 绘制折线图
    plt.plot([i + 1 for i in range(0, len(loss_record))], loss_record)
    # 添加标题和轴标签
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/training_loss.png')


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置输出的通道和输出的类别数目，这里的1表示执行的是二分类的任务
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到device中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "../DRIVE-SEG-DATA"  # todo 或者使用相对路径也是可以的
    print("进度条出现卡着不动不是程序问题，是他正在计算，请耐心等待")
    time.sleep(1)
    train_net(net, device, data_path, epochs=1200, batch_size=1)  # 开始训练