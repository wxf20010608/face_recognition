from login import LoginWindow
import sys
import os
import cv2
import requests
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
    QLineEdit, QComboBox, QGroupBox, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen
from concurrent.futures import ThreadPoolExecutor
import io
from PIL import Image
import threading
import time
import subprocess
import socket

BACKEND_URL = "http://127.0.0.1:8002"

class FaceApp(QWidget):
    show_db_panel_signal = pyqtSignal()
    update_recog_result_signal = pyqtSignal(str, object)
    update_image_with_box_signal = pyqtSignal(QImage, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别与管理系统")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(800, 600)
        self.setMaximumSize(1200, 900)
        self.cap = None
        self.timer = None
        self.recognizing = False
        self.frame_count = 0
        self.last_box = None
        self.last_result = "识别结果："
        self.backend_url = BACKEND_URL
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.recog_img = None
        self.collecting_mode = False
        self.img_path = None  # 初始化图片路径
        self.init_ui()
        self.check_backend_status()

        # 主窗口背景渐变+圆角
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #e0eafc, stop:1 #cfdef3);
                border-radius: 16px;
            }
        """)
        self.update_recog_result_signal.connect(self.update_recog_result)
        self.update_image_with_box_signal.connect(self.update_image_display)

    def check_backend_status(self):
        try:
            response = requests.get(f"{self.backend_url}/list_faces/")
            if response.status_code == 200:
                # QMessageBox.information(self, "连接成功", "已成功连接到后端服务") # Removed: No more success message
                pass # Do nothing on success
            else:
                QMessageBox.warning(self, "连接失败", f"无法连接到后端服务，状态码: {response.status_code}")
        except Exception as e:
            QMessageBox.critical(self, "连接错误", f"连接后端服务时发生错误: {str(e)}")

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)

        # 左侧面板布局
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(30)

        self.switch_fatigue_btn = QPushButton("切换疲劳监测系统")
        self.switch_fatigue_btn.setStyleSheet(
            "background:#ff9800; color:white; border-radius:10px; font-size:15px; padding:8px 24px;"
        )
        self.switch_fatigue_btn.clicked.connect(self.switch_to_fatigue)
        left_panel_layout.addWidget(self.switch_fatigue_btn)

        # 在init_ui方法左侧面板布局添加
        self.back_login_btn = QPushButton("返回登录")
        self.back_login_btn.setStyleSheet(
            "background:#b0b0b0; color:white; border-radius:10px; font-size:15px; padding:8px 24px;"
        )
        self.back_login_btn.clicked.connect(self.back_to_login)
        left_panel_layout.addWidget(self.back_login_btn)

        # 1. 人脸信息录入组框
        group_entry = QGroupBox("人脸信息录入")
        entry_layout = QGridLayout()
        entry_layout.setVerticalSpacing(18)
        entry_layout.setHorizontalSpacing(12)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("请输入姓名")
        self.age_edit = QLineEdit()
        self.age_edit.setPlaceholderText("请输入年龄")
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["男", "女"])

        # 录入图片按钮水平排列
        entry_btn_hbox = QHBoxLayout()
        self.choose_img_btn = QPushButton("选择图片")
        self.choose_img_btn.setMinimumHeight(30)
        self.cam_img_btn = QPushButton("摄像头采集")
        self.cam_img_btn.setMinimumHeight(30)
        self.choose_img_btn.clicked.connect(self.choose_img)
        self.cam_img_btn.clicked.connect(self.capture_from_cam)
        entry_btn_hbox.addWidget(self.choose_img_btn)
        entry_btn_hbox.addWidget(self.cam_img_btn)

        # 录入人脸按钮单独一行
        self.add_btn = QPushButton("录入人脸")
        self.add_btn.setMinimumHeight(30)
        self.add_btn.clicked.connect(self.add_face)
        add_btn_hbox = QHBoxLayout()
        add_btn_hbox.addWidget(self.add_btn)

        # 总体垂直布局
        entry_img_vbox = QVBoxLayout()
        entry_img_vbox.addLayout(entry_btn_hbox)
        entry_img_vbox.addLayout(add_btn_hbox)
        entry_img_vbox.setSpacing(12)
        entry_img_vbox.setContentsMargins(0, 0, 0, 0)

        # 放到表单布局
        entry_layout.addWidget(QLabel("姓名："), 0, 0)
        entry_layout.addWidget(self.name_edit, 0, 1)
        entry_layout.addWidget(QLabel("年龄："), 1, 0)
        entry_layout.addWidget(self.age_edit, 1, 1)
        entry_layout.addWidget(QLabel("性别："), 2, 0)
        entry_layout.addWidget(self.gender_combo, 2, 1)
        entry_layout.addLayout(entry_img_vbox, 3, 0, 1, 2)
        group_entry.setLayout(entry_layout)
        left_panel_layout.addWidget(group_entry)

        # 2. 人脸识别操作组框
        group_recog_controls = QGroupBox("人脸识别操作")
        recog_controls_layout = QVBoxLayout()
        recog_controls_layout.setSpacing(18)
        self.open_cam_btn = QPushButton("开始识别")
        self.open_cam_btn.clicked.connect(self.toggle_camera)
        self.recog_result_label = QLabel("识别结果：")
        self.recog_result_label.setFont(QFont("微软雅黑", 15))
        self.recog_result_label.setWordWrap(True)
        recog_controls_layout.addWidget(self.open_cam_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        recog_controls_layout.addWidget(self.recog_result_label, alignment=Qt.AlignmentFlag.AlignCenter)
        group_recog_controls.setLayout(recog_controls_layout)
        left_panel_layout.addWidget(group_recog_controls)

        # 3. 数据库管理按钮
        self.manage_db_btn = QPushButton("管理数据库")
        self.manage_db_btn.clicked.connect(self.launch_db_manager)
        left_panel_layout.addWidget(self.manage_db_btn)

        self.btn_exit_app = QPushButton("退出系统")
        self.btn_exit_app.setStyleSheet(
            "background:#f44336; color:white; border-radius:10px; font-size:15px; padding:8px 24px;"
        )
        self.btn_exit_app.setMinimumHeight(40)
        left_panel_layout.addWidget(self.btn_exit_app)
        left_panel_layout.addStretch(1)

        main_layout.addLayout(left_panel_layout, 1)

        # 右侧摄像头显示区域和按钮
        self.recog_img_display_label = QLabel("摄像头画面")
        self.recog_img_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recog_img_display_label.setMinimumSize(640, 480)
        self.recog_img_display_label.setMaximumSize(640, 480)
        self.recog_img_display_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # 拍照和关闭摄像头按钮
        self.capture_btn = QPushButton("拍照")
        self.capture_btn.clicked.connect(self.capture_photo)
        self.capture_btn.setEnabled(False)

        self.close_cam_btn = QPushButton("关闭摄像头")
        self.close_cam_btn.clicked.connect(self.stop_camera)
        self.close_cam_btn.setEnabled(False)

        self.btn_exit_app.clicked.connect(self._exit_app)

        right_panel_layout = QVBoxLayout()
        right_panel_layout.addStretch(1)
        right_panel_layout.addWidget(self.recog_img_display_label, alignment=Qt.AlignmentFlag.AlignCenter)
        # 按钮水平排列
        btn_hbox = QHBoxLayout()
        btn_hbox.addStretch(1)
        btn_hbox.addWidget(self.capture_btn)
        btn_hbox.addSpacing(20)
        btn_hbox.addWidget(self.close_cam_btn)
        btn_hbox.addStretch(1)
        right_panel_layout.addLayout(btn_hbox)
        right_panel_layout.addStretch(2)
        main_layout.addLayout(right_panel_layout, 3)

        self.setLayout(main_layout)

        # --------- 美化样式 ---------
        group_entry.setStyleSheet("""
            QGroupBox {
                font-size: 17px;
                font-weight: bold;
                color: #0078d7;
                border: 2px solid #b0b0b0;
                border-radius: 10px;
                margin-top: 8px;
                background: #f7fbff;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        group_recog_controls.setStyleSheet(group_entry.styleSheet())

        for btn in [self.manage_db_btn, self.add_btn, self.choose_img_btn, self.cam_img_btn, self.open_cam_btn, self.capture_btn, self.close_cam_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                                stop:0 #43e97b, stop:1 #38f9d7);
                    color: white;
                    border-radius: 10px;
                    font-size: 15px;
                    padding: 8px 24px;
                }
                QPushButton:hover {
                    background: #0078d7;
                }
            """)

        for edit in [self.name_edit, self.age_edit]:
            edit.setStyleSheet("""
                QLineEdit {
                    border: 1.5px solid #b0b0b0;
                    border-radius: 6px;
                    padding: 4px 8px;
                    font-size: 15px;
                    background: #f7fbff;
                }
            """)
        self.gender_combo.setStyleSheet("""
            QComboBox {
                border: 1.5px solid #b0b0b0;
                border-radius: 6px；
                padding: 4px 8px;
                font-size: 15px;
                background: #f7fbff;
            }
        """)
        self.recog_img_display_label.setStyleSheet("""
            QLabel {
                border: 3px solid #0078d7;
                background: #f4f8fb;
                border-radius: 16px;
                font-size: 18px;
                color: #0078d7;
            }
        """)
        self.recog_result_label.setStyleSheet("""
            QLabel {
                color: #0078d7;
                font-weight: bold;
                font-size: 18px;
            }
        """)

    def choose_img(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.png *.jpeg)")
        if path:
            self.img_path = path
            self._display_static_image(path)
        else:
            self.img_path = None
            self.choose_img_btn.setText("选择图片")

    def _display_static_image(self, image_path):
        """Helper to display a static image on the recog_img_display_label."""
        try:
            pil_image = Image.open(image_path).convert('RGB')
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            h, w, ch = cv_image.shape
            bytes_per_line = ch * w
            qimg = QImage(cv_image.data.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimg)
            self.recog_img_display_label.setPixmap(
                pixmap.scaled(
                    self.recog_img_display_label.width(),
                    self.recog_img_display_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
            self.recog_img_display_label.setText("") # Clear text when image is displayed
        except Exception as e:
            QMessageBox.warning(self, "错误", f"显示图片失败: {str(e)}")
            self.recog_img_display_label.setText("加载图片失败")

    def display_selected_image(self, image_path):
        """显示选中的图片并进行人脸检测"""
        try:
            # 读取图片
            pil_image = Image.open(image_path).convert('RGB')
            # 转换为OpenCV格式
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 进行人脸识别
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            files = {"file": ("temp_recog.jpg", img_byte_arr, "image/jpeg")}
            
            # 发送到后端进行识别
            r = requests.post(f"{BACKEND_URL}/recognize/", files=files, timeout=10)
            if r.ok:
                result = r.json()
                if result.get("result"):
                    # 有识别结果，显示带框的图片
                    name = result['result']['name']
                    age = result['result']['age']
                    gender = result['result']['gender']
                    similarity = result['result']['similarity']
                    box = result['box']
                    
                    # 在图片上画框
                    x1, y1, x2, y2 = box
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"{name} {age}岁 {gender}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 显示识别结果
                    self.recog_result_label.setText(f"识别结果：姓名 {name}，年龄 {age}，性别 {gender} (相似度: {similarity:.2f})")
                else:
                    # 没有识别结果，但可能有检测框
                    box = result.get("box")
                    if box:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(cv_image, "未识别", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        self.recog_result_label.setText("识别结果：未识别到已知人脸")
                    else:
                        self.recog_result_label.setText("识别结果：未检测到人脸")
            else:
                self.recog_result_label.setText("识别结果：后端连接失败")
            
            # 显示图片
            h, w, ch = cv_image.shape
            bytes_per_line = ch * w
            qimg = QImage(cv_image.data.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888) # Ensure data is contiguous
            pixmap = QPixmap.fromImage(qimg)
            self.recog_img_display_label.setPixmap(
                pixmap.scaled(
                    self.recog_img_display_label.width(),
                    self.recog_img_display_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"显示图片失败: {str(e)}")
            self.recog_result_label.setText("识别结果：显示图片失败")

    def reset_image_selection(self):
        """重置图片选择状态"""
        self.img_path = None
        self.choose_img_btn.setText("选择图片")
        self.recog_img_display_label.setText("摄像头画面")
        self.recog_result_label.setText("识别结果：")

    def switch_to_fatigue(self):
        QMessageBox.information(self, "系统切换", "正在尝试切换至疲劳监测系统...")
        
        # 1. 停止当前人脸识别系统的摄像头和定时器
        self.stop_camera() # This method already handles stopping timer and releasing cap

        # 2. 关闭当前窗口
        self.close()

        # Helper function to check if a port is in use
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("127.0.0.1", port)) == 0

        # 3. 启动疲劳监测系统的后端服务 (fatigue_monitor_update/backend/app.py)
        # 使用绝对路径确保找到脚本，并后台运行
        fatigue_backend_port = 8000 # Fatigue monitoring backend port
        if not is_port_in_use(fatigue_backend_port):
            fatigue_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fatigue_monitor_update/backend/app.py")) # Corrected path
            
            # Define log file paths for the backend
            # Project root is D:\AI_Learning\python\01_Learning\Target_Detection
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
            fatigue_output_log = os.path.join(project_root, "fatigue_backend_output.log")
            fatigue_error_log = os.path.join(project_root, "fatigue_backend_error.log")

            try:
                with open(fatigue_output_log, "w") as fout, open(fatigue_error_log, "w") as ferr:
                    subprocess.Popen([sys.executable, fatigue_backend_path],
                                     creationflags=subprocess.DETACHED_PROCESS,
                                     stdout=fout, # Redirect stdout to file
                                     stderr=ferr  # Redirect stderr to file
                                     )
                print(f"疲劳监测后端启动命令已执行，输出到 {fatigue_output_log} 和 {fatigue_error_log}")
            except Exception as e:
                print(f"启动疲劳监测后端失败: {e}")
        else:
            print(f"疲劳监测后端 (端口 {fatigue_backend_port}) 已经在运行，跳过启动。")

        # 4. 启动疲劳监测系统的前端应用程序 (fatigue_monitor_update/frontend/main.py)
        fatigue_frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../fatigue_monitor_update/frontend/main.py")) # Corrected path
        try:
            subprocess.Popen([sys.executable, fatigue_frontend_path]) # Removed DETACHED_PROCESS and DEVNULL for debugging
            print(f"疲劳监测前端已启动: {fatigue_frontend_path}")
        except Exception as e:
            print(f"启动疲劳监测前端失败: {e}")

    def back_to_login(self):
        from login import LoginWindow
        self.login_win = LoginWindow()
        self.login_win.show()
        self.close()

    def capture_from_cam(self):
        if self.collecting_mode:
            # 采集模式下再次点击，关闭摄像头
            self.collecting_mode = False
            if hasattr(self, "collect_timer") and self.collect_timer:
                self.collect_timer.stop()
            if hasattr(self, "collect_cap") and self.collect_cap:
                self.collect_cap.release()
                self.collect_cap = None
            self.cam_img_btn.setText("摄像头采集")
            self.capture_btn.setEnabled(False)
            self.recog_img_display_label.setText("摄像头画面")
        else:
            # 进入采集模式，打开摄像头，仅显示画面不做识别
            self.collect_cap = cv2.VideoCapture(0)
            if not self.collect_cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头")
                self.collect_cap = None
                return
            self.collecting_mode = True
            self.cam_img_btn.setText("停止采集")
            self.capture_btn.setEnabled(True)
            self.collect_timer = QTimer(self)
            self.collect_timer.timeout.connect(self.update_collect_display)
            self.collect_timer.start(30)

    def update_collect_display(self):
        if self.collect_cap and self.collect_cap.isOpened():
            ret, frame = self.collect_cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_image.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888) # Ensure data is contiguous
                pixmap = QPixmap.fromImage(qimg)
                self.recog_img_display_label.setPixmap(pixmap.scaled(
                    self.recog_img_display_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
                # 保存当前帧用于拍照
                self.collect_frame = frame.copy()
            else:
                if self.collect_timer:
                    self.collect_timer.stop()
                if self.collect_cap:
                    self.collect_cap.release()
                self.collect_cap = None
                self.collecting_mode = False
                self.cam_img_btn.setText("摄像头采集")
                self.capture_btn.setEnabled(False)
                self.recog_img_display_label.setText("摄像头画面")

    def capture_photo(self):
        if not self.collecting_mode or not hasattr(self, "collect_frame"):
            return
        frame = self.collect_frame
        save_dir = os.path.join(os.path.dirname(__file__), "face_images")
        os.makedirs(save_dir, exist_ok=True)
        self.img_path = os.path.join(save_dir, "temp_capture.jpg")
        cv2.imwrite(self.img_path, frame)
        QMessageBox.information(self, "成功", "照片已采集并保存！")
        # 采集完成后不再自动关闭摄像头
        # self.collect_timer.stop()
        # self.collect_cap.release()
        # self.collect_cap = None
        # self.collecting_mode = False
        # self.cam_img_btn.setText("摄像头采集")
        # self.capture_btn.setEnabled(False)
        # self.recog_img_display_label.setText("摄像头画面")

    def add_face(self):
        name = self.name_edit.text().strip()
        age = self.age_edit.text().strip()
        gender = self.gender_combo.currentText()

        # 更详细的验证
        missing_fields = []
        if not name:
            missing_fields.append("姓名")
        if not age:
            missing_fields.append("年龄")
        if not self.img_path:
            missing_fields.append("图片")
        
        if missing_fields:
            QMessageBox.warning(self, "警告", f"请填写以下信息：{', '.join(missing_fields)}")
            return
        
        # 验证年龄是否为数字
        try:
            age_int = int(age)
            if age_int <= 0 or age_int > 150:
                QMessageBox.warning(self, "警告", "请输入有效的年龄（1-150）")
                return
        except ValueError:
            QMessageBox.warning(self, "警告", "年龄必须是数字")
            return
        
        try:
            files = {"file": open(self.img_path, "rb")}
            data = {"name": name, "age": age_int, "gender": gender}
            r = requests.post(f"{BACKEND_URL}/add_face/", files=files, data=data)
            if r.ok:
                QMessageBox.information(self, "成功", "录入成功")
                # 清空输入框和图片显示
                self.name_edit.clear()
                self.age_edit.clear()
                self.reset_image_selection()
                # 录入成功后关闭采集模式的摄像头
                if hasattr(self, "collect_timer") and self.collect_timer:
                    self.collect_timer.stop()
                    self.collect_timer = None
                if hasattr(self, "collect_cap") and self.collect_cap:
                    self.collect_cap.release()
                    self.collect_cap = None
                self.collecting_mode = False
                self.cam_img_btn.setText("摄像头采集")
                self.capture_btn.setEnabled(False)
                self.recog_img_display_label.setText("摄像头画面")
            else:
                QMessageBox.critical(self, "失败", f"录入失败: {r.text}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"发生异常: {str(e)}")

    def start_camera_thread(self):
        self.camera_running = True
        self.camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.camera_thread.start()

    def camera_worker(self):
        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame_count += 1
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888) # Ensure data is contiguous
            self.update_image_with_box_signal.emit(qimg, self.last_box)
            if self.frame_count % 5 == 0 and not self.recognizing:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.recognizing = True
                self.executor.submit(self.async_recognize, pil_image, frame.copy())
            time.sleep(0.015)  # 控制帧率，约66fps

    def toggle_camera(self):
        if self.cap is None: # Camera is currently OFF
            if self.img_path is not None: # If a static image is chosen, process it
                self.display_selected_image(self.img_path)
                self.open_cam_btn.setText("停止识别")
                self.close_cam_btn.setEnabled(True) # Enable close button for static image display
                self.capture_btn.setEnabled(False) # Disable capture button for static image
            else: # No static image, try to open live camera
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    QMessageBox.warning(self, "错误", "无法打开摄像头")
                    self.cap = None
                    return
                self.open_cam_btn.setText("停止识别")
                self.capture_btn.setEnabled(True)
                self.close_cam_btn.setEnabled(True)
                self.start_camera_thread()  # Start live camera recognition
        else: # Camera is currently ON, so stop it
            self.stop_camera()

    def camera_loop(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                self.stop_camera()
                return
            ret, frame = self.cap.read()
            if not ret:
                return
            self.frame_count += 1

            # 实时显示+用最近一次识别结果画框
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888) # Ensure data is contiguous
            self.update_image_display(qimg, self.last_box)  # 关键：每帧都用last_box

            # 降低识别频率，且只在空闲时识别
            if self.frame_count % 5 == 0 and not self.recognizing:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.recognizing = True
                self.executor.submit(self.async_recognize, pil_image, frame.copy())
        except Exception as e:
            import traceback
            traceback.print_exc()

    def async_recognize(self, pil_image, original_frame):
        try:
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            files = {"file": ("temp_recog.jpg", img_byte_arr, "image/jpeg")}
            
            # 设置超时时间，避免长时间无响应
            r = requests.post(f"{BACKEND_URL}/recognize/", files=files, timeout=10)
            r.raise_for_status()
            result = r.json()
            
            if result.get("result"):
                name = result['result']['name']
                age = result['result']['age']
                gender = result['result']['gender']
                similarity = result['result']['similarity']
                box = result['box']
                self.last_box = box
                self.last_result = f"识别结果：姓名 {name}，年龄 {age}，性别 {gender} (相似度: {similarity:.2f})"
                self.update_recog_result_signal.emit(self.last_result, box)
                self.update_image_with_box_signal.emit(QImage(original_frame.data.tobytes(), original_frame.shape[1], original_frame.shape[0], original_frame.shape[2] * original_frame.shape[1], QImage.Format_BGR888), box)
            else:
                msg = result.get("msg", "未知错误")
                self.last_box = None
                self.last_result = f"识别结果：{msg}"
                self.update_recog_result_signal.emit(self.last_result, None)
                self.update_image_with_box_signal.emit(QImage(original_frame.data.tobytes(), original_frame.shape[1], original_frame.shape[0], original_frame.shape[2] * original_frame.shape[1], QImage.Format_BGR888), None)
        except requests.exceptions.ConnectionError:
            self.update_recog_result_signal.emit("识别结果：无法连接到后端", None)
            self.update_image_with_box_signal.emit(QImage(original_frame.data.tobytes(), original_frame.shape[1], original_frame.shape[0], original_frame.shape[2] * original_frame.shape[1], QImage.Format_BGR888), None)
        except requests.exceptions.HTTPError as e:
            self.update_recog_result_signal.emit(f"识别结果：后端错误 {e.response.status_code}", None)
            self.update_image_with_box_signal.emit(QImage(original_frame.data.tobytes(), original_frame.shape[1], original_frame.shape[0], original_frame.shape[2] * original_frame.shape[1], QImage.Format_BGR888), None)
        except requests.exceptions.Timeout:
            self.update_recog_result_signal.emit("识别结果：请求超时", None)
            self.update_image_with_box_signal.emit(QImage(original_frame.data.tobytes(), original_frame.shape[1], original_frame.shape[0], original_frame.shape[2] * original_frame.shape[1], QImage.Format_BGR888), None)
        except Exception as e:
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            self.update_recog_result_signal.emit(f"识别结果：发生错误 {e}", None)
            self.update_image_with_box_signal.emit(QImage(original_frame.data.tobytes(), original_frame.shape[1], original_frame.shape[0], original_frame.shape[2] * original_frame.shape[1], QImage.Format_BGR888), None)
        finally:
            self.recognizing = False

    def update_recog_result(self, text, box=None):
        self.recog_result_label.setText(text)

    def update_image_display(self, qimage, box=None):
        try:
            if not qimage.isNull():
                pixmap = QPixmap.fromImage(qimage)
                if box and isinstance(box, (list, tuple)) and len(box) == 4:
                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.GlobalColor.green, 4))
                    x1, y1, x2, y2 = box
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                    painter.setPen(QPen(Qt.GlobalColor.white, 2))
                    painter.setFont(QFont("Arial", 14))
                    painter.fillRect(x1, max(0, y1 - 30), x2 - x1, 30, Qt.GlobalColor.green)
                    painter.drawText(x1 + 5, max(0, y1 - 8), self.last_result)
                    painter.end()
                # 无论是否有框都刷新画面
                self.recog_img_display_label.setPixmap(
                    pixmap.scaled(
                        self.recog_img_display_label.width(),
                        self.recog_img_display_label.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )
            else:
                self.recog_img_display_label.setText("摄像头画面")
        except Exception as e:
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        self.stop_camera()
        # 优雅地关闭线程池，等待所有任务完成
        if self.executor:
            self.executor.shutdown(wait=True)
        event.accept()

    def stop_camera(self):
        # If no cameras are active, do nothing
        if self.cap is None and (not hasattr(self, "collect_cap") or self.collect_cap is None):
            return

        # 停止实时识别定时器
        self.camera_running = False
        self.recognizing = False # Ensure recognition stops immediately
        if hasattr(self, "camera_thread") and self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join() 
        if self.timer:
            self.timer.stop()
            self.timer = None
        if self.cap:
            self.cap.release()
            self.cap = None

        # 停止采集照片定时器
        if hasattr(self, "collect_timer") and self.collect_timer:
            self.collect_timer.stop()
            self.collect_timer = None
        if hasattr(self, "collect_cap") and self.collect_cap:
            self.collect_cap.release()
            self.collect_cap = None

        self.collecting_mode = False
        self.open_cam_btn.setText("开始识别")
        self.capture_btn.setEnabled(False)
        self.close_cam_btn.setEnabled(False)
        self.recog_img_display_label.clear() # Clear any pixmap
        self.recog_img_display_label.setText("摄像头画面")
        self.recog_result_label.setText("识别结果：")

    def launch_db_manager(self):
        # 延迟导入避免循环依赖
        from face_db_manager import FaceDBManager
        self.db_manager_win = FaceDBManager()
        self.db_manager_win.back_to_face_main_signal.connect(self.show_again)
        self.db_manager_win.show()
        self.close()

    def show_again(self):
        self.show()
        self.db_manager_win.close()

    def _exit_app(self):
        """Handles the application exit."""
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FaceApp()
    win.show()
    sys.exit(app.exec_())