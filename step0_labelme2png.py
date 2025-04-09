# -*-coding:utf-8 -*-

"""
#-------------------------------
# @Author : 肆十二
# @QQ : 3045834499 可定制毕设
#-------------------------------
# @File : aa.py
# @Description: 将标注好的json文件转换为图像标签文件
# @Software : PyCharm
# @Time : 2024/2/14 10:48
#-------------------------------
"""


from __future__ import print_function
import argparse
import glob
import math
import json
import os
import os.path as osp
import shutil
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2


def json2png(json_folder, png_save_folder):
    print("准备开始json文件转转化 注意json文件不用有中文")
    if osp.isdir(png_save_folder):
        shutil.rmtree(png_save_folder)
    os.makedirs(png_save_folder)
    print("数据保存目录创建成功！")
    # 遍历文件夹，把文件夹中得json文件先删除
    for json_file in os.listdir(json_folder):
        file_path = osp.join(json_folder, json_file)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print("文件夹清除成功！")
    print("开始批量生成json文件")
    for json_file in os.listdir(json_folder):
        json_path = osp.join(json_folder, json_file)
        os.system("labelme_json_to_dataset {}".format(json_path))
        label_path = osp.join(json_folder, json_file.split(".")[0] + "_json/label.png")
        png_save_path = osp.join(png_save_folder, json_file.split(".")[0] + ".png")
        label_png = cv2.imread(label_path, 0)
        label_png[label_png > 0] = 255
        cv2.imwrite(png_save_path, label_png)
        # shutil.copy(label_path, png_save_path)
        # break
    print("标签文件已保存在目录：{}".format(png_save_folder))



if __name__ == '__main__':
    # !!!!你的json文件夹下只能有json文件不能有其他文件
    json2png(json_folder="testdata/jsons/", png_save_folder="testdata/labels/")
