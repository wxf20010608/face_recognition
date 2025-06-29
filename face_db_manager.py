import sys
import os
import sqlite3
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QLineEdit, QComboBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal


DB_PATH = os.path.join(os.path.dirname(__file__), "face_db.sqlite")
DB_PATH = r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\face_db.sqlite"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT id, embedding FROM faces")
to_delete = []
for fid, emb_bytes in c.fetchall():
    emb = np.frombuffer(emb_bytes, dtype=np.float32)
    if emb.shape != (512,):
        print(f"人脸ID {fid} 的特征维度为 {emb.shape}，将被删除")
        to_delete.append((fid,))
if to_delete:
    c.executemany("DELETE FROM faces WHERE id=?", to_delete)
    conn.commit()
    print(f"已删除 {len(to_delete)} 条无效人脸数据")
else:
    print("所有人脸特征均有效")
conn.close()

class FaceDBManager(QWidget):
    back_to_face_main_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸数据库管理")
        self.setGeometry(200, 100, 950, 650)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #e0eafc, stop:1 #cfdef3);
            }
        """)
        self.init_ui()
        self.refresh_table()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(18)

        self.back_btn = QPushButton("返回人脸识别")
        self.back_btn.clicked.connect(self.back_to_face_main_signal.emit)

         # 返回按钮
        self.back_btn = QPushButton("返回人脸识别")
        self.back_btn.setStyleSheet("QPushButton {background:#0078d7; color:white; border-radius:8px; font-size:16px; padding:8px 20px;} QPushButton:hover{background:#4facfe;}")
        self.back_btn.clicked.connect(self.back_to_face_main_signal.emit)
        layout.addWidget(self.back_btn, alignment=Qt.AlignLeft)  # 添加到布局

        # 顶部标题
        title = QLabel("人脸数据库管理系统")
        title.setFont(QFont("微软雅黑", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0078d7; margin-bottom: 8px;")
        layout.addWidget(title)

        # 顶部操作区
        op_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setStyleSheet("QPushButton {background:#4facfe; color:white; border-radius:8px; font-size:16px; padding:8px 20px;} QPushButton:hover{background:#00f2fe;}")
        self.refresh_btn.clicked.connect(self.refresh_table)
        self.delete_btn = QPushButton("删除选中")
        self.delete_btn.setStyleSheet("QPushButton {background:#ff5858; color:white; border-radius:8px; font-size:16px; padding:8px 20px;} QPushButton:hover{background:#ff7b7b;}")
        self.delete_btn.clicked.connect(self.delete_selected)
        op_layout.addWidget(self.refresh_btn)
        op_layout.addWidget(self.delete_btn)
        op_layout.addStretch(1)
        layout.addLayout(op_layout)

        # 数据表
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "姓名", "年龄", "性别", "图片"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.MultiSelection)  # 支持多选
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)  # 允许编辑
        self.table.setMinimumHeight(420)
        self.table.setStyleSheet("""
            QTableWidget {
                background: rgba(255,255,255,0.95);
                border-radius: 12px;
                font-size: 15px;
            }
            QHeaderView::section {
                background: #0078d7;
                color: white;
                font-weight: bold;
                font-size: 16px;
                border: none;
                height: 32px;
            }
        """)
        layout.addWidget(self.table)

        self.save_btn = QPushButton("保存修改")
        self.save_btn.setStyleSheet("QPushButton {background:#ffb347; color:white; border-radius:8px; font-size:16px; padding:8px 20px;} QPushButton:hover{background:#ffd580;}")
        self.save_btn.clicked.connect(self.save_changes)
        op_layout.addWidget(self.save_btn)

        # 新增人脸信息区
        add_box = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("姓名")
        self.name_edit.setStyleSheet("border-radius:8px; padding:6px; font-size:15px;")
        self.age_edit = QLineEdit()
        self.age_edit.setPlaceholderText("年龄")
        self.age_edit.setStyleSheet("border-radius:8px; padding:6px; font-size:15px;")
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["男", "女"])
        self.gender_combo.setStyleSheet("border-radius:8px; padding:6px; font-size:15px;")
        self.img_path_edit = QLineEdit()
        self.img_path_edit.setPlaceholderText("图片路径")
        self.img_path_edit.setStyleSheet("border-radius:8px; padding:6px; font-size:15px;")
        self.add_btn = QPushButton("手动添加")
        self.add_btn.setStyleSheet("QPushButton {background:#43e97b; color:white; border-radius:8px; font-size:16px; padding:8px 20px;} QPushButton:hover{background:#38f9d7;}")
        self.add_btn.clicked.connect(self.add_face)
        add_box.addWidget(self.name_edit)
        add_box.addWidget(self.age_edit)
        add_box.addWidget(self.gender_combo)
        add_box.addWidget(self.img_path_edit)
        add_box.addWidget(self.add_btn)
        layout.addLayout(add_box)

    def refresh_table(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, age, gender, image_path FROM faces")
        faces = c.fetchall()
        conn.close()
        self.table.setRowCount(len(faces))
        for row, (fid, name, age, gender, img_path) in enumerate(faces):
            id_item = QTableWidgetItem(str(fid))
            id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)  # ID不可编辑
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, QTableWidgetItem(name))
            self.table.setItem(row, 2, QTableWidgetItem(str(age)))
            self.table.setItem(row, 3, QTableWidgetItem(gender))
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border:1px solid #b0b0b0; border-radius:6px; background:#f4f8fb;")
            if os.path.exists(img_path):
                pix = QPixmap(img_path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(pix)
            else:
                img_label.setText("无图片")
            self.table.setCellWidget(row, 4, img_label)
        QMessageBox.information(self, "刷新成功", "数据已刷新！")

    def delete_selected(self):
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            QMessageBox.warning(self, "提示", "请先选择要删除的行")
            return
        # 收集所有选中行的ID
        ids = set()
        for rng in selected_ranges:
            for row in range(rng.topRow(), rng.bottomRow() + 1):
                id_item = self.table.item(row, 0)
                if id_item:
                    ids.add(int(id_item.text()))
        if not ids:
            QMessageBox.warning(self, "提示", "未选中有效数据")
            return
        reply = QMessageBox.question(self, "确认", f"确定要删除选中的{len(ids)}条人脸信息吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.executemany("DELETE FROM faces WHERE id=?", [(i,) for i in ids])
            conn.commit()
            conn.close()
            self.refresh_table()

    def save_changes(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        updated = 0
        for row in range(self.table.rowCount()):
            id_item = self.table.item(row, 0)
            name_item = self.table.item(row, 1)
            age_item = self.table.item(row, 2)
            gender_item = self.table.item(row, 3)
            if not (id_item and name_item and age_item and gender_item):
                continue
            fid = int(id_item.text())
            name = name_item.text().strip()
            try:
                age = int(age_item.text().strip())
            except Exception:
                continue
            gender = gender_item.text().strip()
            c.execute("UPDATE faces SET name=?, age=?, gender=? WHERE id=?", (name, age, gender, fid))
            updated += 1
        conn.commit()
        conn.close()
        QMessageBox.information(self, "保存成功", f"已保存{updated}条修改！")
        self.refresh_table()

    def add_face(self):
        name = self.name_edit.text().strip()
        age = self.age_edit.text().strip()
        gender = self.gender_combo.currentText()
        img_path = self.img_path_edit.text().strip()
        if not name or not age or not img_path:
            QMessageBox.warning(self, "提示", "姓名、年龄、图片路径不能为空")
            return
        try:
            age = int(age)
        except ValueError:
            QMessageBox.warning(self, "提示", "年龄必须为数字")
            return
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO faces (name, age, gender, image_path, embedding) VALUES (?, ?, ?, ?, ?)",
                  (name, age, gender, img_path, b""))
        conn.commit()
        conn.close()
        self.refresh_table()
        self.name_edit.clear()
        self.age_edit.clear()
        self.img_path_edit.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FaceDBManager()
    win.show()
    sys.exit(app.exec_())