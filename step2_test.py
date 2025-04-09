# -*-coding:utf-8 -*-

"""
#-------------------------------
# @Author : 肆十二
# @QQ : 3045834499 可定制毕设
#-------------------------------
# @File : step2_test.py
# @Description: 模型测试，并输出测试结果在test目录下
# @Software : PyCharm
# @Time : 2024/2/14 10:48
#-------------------------------
"""

import os
import time

from tqdm import tqdm
from utils.utils_metrics import compute_mIoU_gray, show_results
import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet


def cal_miou(test_dir="../DRIVE-SEG-DATA/Test_Images",
             pred_dir="../DRIVE-SEG-DATA/results", gt_dir="../DRIVE-SEG-DATA/Test_Labels",
             model_path='best_model1.pth'):
    name_classes = ["background", "vein"]
    num_classes = len(name_classes)
    # 加载模型
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    print("---------------------------------------------------------------------------------------")
    print("加载训练好的模型,模型位于{}".format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load(model_path, map_location=device))
    # 测试模式
    net.eval()
    print("模型加载成功！")

    img_names = os.listdir(test_dir)
    image_ids = [image_name.split(".")[0] for image_name in img_names]

    print("---------------------------------------------------------------------------------------")
    print("对测试集进行批量推理")
    time.sleep(1)
    for image_id in tqdm(image_ids):
        image_path = os.path.join(test_dir, image_id + ".png")
        img = cv2.imread(image_path)
        origin_shape = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)
    print("测试集批量推理结束")
    print("开始计算MIOU等测试指标")
    hist, IoUs, PA_Recall, Precision = compute_mIoU_gray(gt_dir, pred_dir, image_ids, num_classes,
                                                         name_classes)  # 执行计算mIoU的函数
    print("测试指标计算成功，测试结果已经保存在results目录下")
    miou_out_path = "results/"
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


if __name__ == '__main__':
    cal_miou(test_dir="../DRIVE-SEG-DATA/Test_Images",  # 测试集路径
             pred_dir="../DRIVE-SEG-DATA/results",  # 测试集推理结果保存路径
             gt_dir="../DRIVE-SEG-DATA/Test_Labels",  # 测试标签路径
             model_path='best_model1.pth')  # 训练好的模型路径
