import sys
import os
import sqlite3
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt

DB_PATH = "user.db"

def init_user_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("登录")
        self.setFixedSize(560, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(26)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("人脸识别系统登录")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #0078d7;")
        layout.addWidget(title)

        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("请输入账号")
        self.user_edit.setFixedHeight(38)
        self.user_edit.setStyleSheet("font-size: 17px; border-radius: 8px; border: 1.5px solid #b0b0b0;")
        layout.addWidget(self.user_edit)

        self.pwd_edit = QLineEdit()
        self.pwd_edit.setPlaceholderText("请输入密码")
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setFixedHeight(38)
        self.pwd_edit.setStyleSheet("font-size: 17px; border-radius: 8px; border: 1.5px solid #b0b0b0;")
        layout.addWidget(self.pwd_edit)

        # 记住密码复选框
        self.remember_cb = QCheckBox("记住密码")
        layout.addWidget(self.remember_cb)

        btn_layout = QHBoxLayout()
        self.login_btn = QPushButton("登录")
        self.login_btn.clicked.connect(self.try_login)
        self.login_btn.setStyleSheet("background:#43e97b; color:white; border-radius:8px; font-size:16px; padding:8px 28px;")
        btn_layout.addWidget(self.login_btn)

        self.reg_btn = QPushButton("注册")
        self.reg_btn.clicked.connect(self.register)
        self.reg_btn.setStyleSheet("background:#0078d7; color:white; border-radius:8px; font-size:16px; padding:8px 28px;")
        btn_layout.addWidget(self.reg_btn)

        layout.addLayout(btn_layout)

        # # 新增用户管理按钮
        # self.user_mgr_btn = QPushButton("用户管理")
        # self.user_mgr_btn.setStyleSheet("background:#ffb300; color:white; border-radius:8px; font-size:15px; padding:6px 18px;")
        # self.user_mgr_btn.clicked.connect(self.open_user_mgr)
        # layout.addWidget(self.user_mgr_btn, alignment=Qt.AlignRight)

        self.setLayout(layout)
        self.load_remembered()

    def save_remembered(self, username, password):
        with open("remember.conf", "w", encoding="utf-8") as f:
            f.write(f"{username}\n{password}")

    def load_remembered(self):
        try:
            with open("remember.conf", "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    self.user_edit.setText(lines[0].strip())
                    self.pwd_edit.setText(lines[1].strip())
                    self.remember_cb.setChecked(True)
        except Exception:
            pass

    def try_login(self):
        username = self.user_edit.text().strip()
        password = self.pwd_edit.text().strip()
        if not username or not password:
            QMessageBox.warning(self, "提示", "请输入账号和密码")
            return
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            if self.remember_cb.isChecked():
                self.save_remembered(username, password)
            else:
                if os.path.exists("remember.conf"):
                    os.remove("remember.conf")
            QMessageBox.information(self, "登录成功", "欢迎，%s！" % username)
            self.open_main()
        else:
            QMessageBox.warning(self, "登录失败", "账号或密码错误")

    def register(self):
        username = self.user_edit.text().strip()
        password = self.pwd_edit.text().strip()
        if not username or not password:
            QMessageBox.warning(self, "提示", "请输入账号和密码")
            return
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            QMessageBox.information(self, "注册成功", "注册成功，请登录")
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "注册失败", "该账号已存在")
        finally:
            conn.close()

    def open_main(self):
        from app import FaceApp
        self.main_win = FaceApp()
        self.main_win.show()
        self.close()
    
if __name__ == "__main__":
    init_user_db()
    app = QApplication(sys.argv)
    win = LoginWindow()
    win.show()
    sys.exit(app.exec_())