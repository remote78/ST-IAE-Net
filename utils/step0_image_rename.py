# -*-coding:utf-8 -*-

"""
#-------------------------------
# @Author : 肆十二
# @QQ : 3045834499 可定制毕设
#-------------------------------
# @File : step0_image_rename.py
# @Description: 文件描述
# @Software : PyCharm
# @Time : 2024/2/14 13:11
#-------------------------------
"""
import os
import os.path as osp

def folder_rename(folder_path):
    file_names = os.listdir(folder_path)
    for filename in file_names:
        print(filename)
        src_name= osp.join(folder_path, filename)
        target_name = osp.join(folder_path, filename.replace("_manual1", ""))
        os.rename(src_name, target_name)

if __name__ == '__main__':
    folder_rename(r"G:\AAA-projects\ING\unet-drive\DRIVE-SEG-DATA\Training_Labels")
    # folder_rename(r"G:\AAA-projects\ING\unet-drive\DRIVE-SEG-DATA\Test_Labels")