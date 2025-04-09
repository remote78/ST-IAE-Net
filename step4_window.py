# -*-coding:utf-8 -*-

import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from model.unet_model import UNet
import numpy as np
import time
import threading
from sklearn.metrics import accuracy_score, precision_score, f1_score, cohen_kappa_score

# 窗口主类
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MainWindow(QMainWindow):  # 继承自 QMainWindow
    def __init__(self):
        super().__init__()
        self.setWindowTitle('基于改进UNet模型的小麦种植面积提取软件')
        self.resize(1600, 1200)  # 调整窗口大小
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        self.output_size = 600  # 调整输出图像的大小
        self.img2predict = ""
        self.origin_shape = ()
        self.model = None
        self.label_img = None
        self.initUI()

    def initUI(self):
        font_title = QFont('楷体', 24)  # 增加标题字体大小
        font_main = QFont('楷体', 28)  # 增加按钮字体大小

        # 主标题
        main_title = QLabel("基于改进UNet模型的小麦种植面积提取软件")
        main_title.setAlignment(Qt.AlignCenter)
        main_title.setFont(QFont('楷体', 36))  # 设置更大的字体
        main_title.setStyleSheet("QLabel { color: #4CAF50; }")  # 设置字体颜色

        # 遥感数据分割界面
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()  # 使用垂直布局

        # 将左侧按钮和右侧图像区域添加到主布局
        img_detection_layout = QHBoxLayout()  # 使用水平布局

        # 左侧按钮区域
        button_widget = QWidget()
        button_layout = QVBoxLayout()
        img_detection_title = QLabel("功能选择区")
        img_detection_title.setAlignment(Qt.AlignCenter)
        img_detection_title.setFont(font_title)

        load_data_button = QPushButton("载入遥感数据")
        load_model_button = QPushButton("载入分割模型")
        det_img_button = QPushButton("开始分割")
        area_extract_button = QPushButton("面积提取")
        load_label_button = QPushButton("载入标签")
        accuracy_eval_button = QPushButton("精度评估")
        reset_button = QPushButton("软件重置")
        exit_button = QPushButton("软件退出")

        load_data_button.clicked.connect(self.load_data)
        load_model_button.clicked.connect(self.load_model)
        det_img_button.clicked.connect(self.detect_img)
        area_extract_button.clicked.connect(self.area_extract)
        load_label_button.clicked.connect(self.load_label)
        accuracy_eval_button.clicked.connect(self.accuracy_eval)
        reset_button.clicked.connect(self.reset)
        exit_button.clicked.connect(self.close)

        for button in [load_data_button, load_model_button, det_img_button, area_extract_button, load_label_button,
                       accuracy_eval_button, reset_button, exit_button]:
            button.setFont(font_main)
            button.setStyleSheet(
                "QPushButton{color:white; background-color: #4CAF50; border: none; padding: 15px 30px; text-align: center; text-decoration: none; display: inline-block; font-size: 28px; margin: 6px 3px; cursor: pointer; border-radius: 10px;}"
                "QPushButton:hover{background-color: #45a049;}")
            button.setMinimumSize(250, 70)  # 调大按钮的最小宽度和高度

        button_layout.addWidget(img_detection_title)
        button_layout.addWidget(load_data_button)
        button_layout.addWidget(load_model_button)
        button_layout.addWidget(det_img_button)
        button_layout.addWidget(area_extract_button)
        button_layout.addWidget(load_label_button)
        button_layout.addWidget(accuracy_eval_button)
        button_layout.addWidget(reset_button)
        button_layout.addWidget(exit_button)
        button_layout.addStretch()  # 添加弹性空间，使按钮靠上
        button_widget.setLayout(button_layout)

        # 右侧图像显示区域
        image_widget = QWidget()
        image_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        self.left_img.setScaledContents(True)
        self.right_img.setScaledContents(True)

        # 设置图像显示的最大尺寸
        self.left_img.setFixedSize(500, 500)  # 设置左侧图像的最大尺寸
        self.right_img.setFixedSize(500, 500)  # 设置右侧图像的最大尺寸

        image_layout.addWidget(self.left_img)
        image_layout.addWidget(self.right_img)
        image_widget.setLayout(image_layout)

        # 将左侧按钮和右侧图像区域添加到主布局
        img_detection_layout.addWidget(button_widget, stretch=1)  # 左侧按钮区域占1份
        img_detection_layout.addWidget(image_widget, stretch=4)  # 右侧图像区域占4份
        img_detection_widget.setLayout(img_detection_layout)

        # 创建底部标签用于显示精度评估结果
        self.result_label = QLabel("精度评估结果将显示在这里")
        self.result_label.setFont(QFont('楷体', 24))  # 设置字体大小
        self.result_label.setStyleSheet("QLabel { color: #4CAF50; }")  # 设置字体颜色
        self.result_label.setAlignment(Qt.AlignCenter)  # 居中对齐

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_title)
        main_layout.addWidget(img_detection_widget)
        main_layout.addSpacing(20)  # 添加间距，使结果标签不紧贴图像
        main_layout.addWidget(self.result_label)  # 将结果标签添加到主布局中
        main_layout.addSpacing(50)  # 添加更大的间距，使结果标签不紧贴底部

        # 设置主窗口的布局
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)  # 使用 QMainWindow 的 setCentralWidget 方法

    def load_data(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选择遥感数据', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
            QMessageBox.information(self, "提示", "遥感数据加载成功！")

    def load_model(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选择分割模型', '', '*.pth')
        if fileName:
            net = UNet(n_channels=1, n_classes=1)
            net.to(device=device)
            try:
                net.load_state_dict(torch.load(fileName, map_location=device))
                net.eval()
                self.model = net
                QMessageBox.information(self, "提示", "模型加载成功！")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"模型加载失败: {str(e)}")

    def load_label(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选择标签', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            self.label_img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            self.right_img.setPixmap(QPixmap(fileName))
            QMessageBox.information(self, "提示", "标签加载成功！")

    def detect_img(self):
        if self.img2predict == "":
            QMessageBox.warning(self, "警告", "请先载入遥感数据！")
            return
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先载入分割模型！")
            return

        try:
            img = cv2.imread(self.img2predict, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512, 512))
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            img_tensor = torch.from_numpy(img).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = self.model(img_tensor)
            pred = pred.cpu().numpy()[0][0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            im0 = cv2.resize(pred, self.origin_shape)
            cv2.imwrite("images/tmp/single_result.jpg", im0)
            self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            QMessageBox.information(self, "提示", "图像分割完成！")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"图像分割失败: {str(e)}")

    def area_extract(self):
        if self.img2predict == "":
            QMessageBox.warning(self, "警告", "请先载入遥感数据！")
            return
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先载入分割模型！")
            return

        try:
            img = cv2.imread(self.img2predict, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512, 512))
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            img_tensor = torch.from_numpy(img).to(device=device, dtype=torch.float32)
            with torch.no_grad():  # 修正为 torch.no_grad()
                pred = self.model(img_tensor)
            pred = pred.cpu().numpy()[0][0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            area = np.sum(pred == 255)
            QMessageBox.information(self, "面积提取", f"提取的面积为：{area} 像素")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"面积提取失败: {str(e)}")

    def accuracy_eval(self):
        if self.img2predict == "":
            QMessageBox.warning(self, "警告", "请先载入遥感数据！")
            return
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先载入分割模型！")
            return
        if self.label_img is None:
            QMessageBox.warning(self, "警告", "请先载入标签！")
            return

        try:
            img = cv2.imread(self.img2predict, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512, 512))
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            img_tensor = torch.from_numpy(img).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = self.model(img_tensor)
            pred = pred.cpu().numpy()[0][0]
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            label = cv2.resize(self.label_img, (512, 512))
            label[label >= 0.5] = 1
            label[label < 0.5] = 0

            oa = accuracy_score(label.flatten(), pred.flatten())
            precision = precision_score(label.flatten(), pred.flatten())
            f1 = f1_score(label.flatten(), pred.flatten())
            kappa = cohen_kappa_score(label.flatten(), pred.flatten())

            # 在底部标签中显示精度评估结果，横向排列
            self.result_label.setText(f"OA: {oa:.4f} | Precision: {precision:.4f} | F1-Score: {f1:.4f} | Kappa: {kappa:.4f}")

        except Exception as e:
            QMessageBox.warning(self, "警告", f"精度评估失败: {str(e)}")

    def reset(self):
        self.img2predict = ""
        self.model = None
        self.label_img = None
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.result_label.setText("精度评估结果将显示在这里")  # 重置结果标签
        QMessageBox.information(self, "提示", "软件已重置！")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出', "确定要退出吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())