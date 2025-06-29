import sys
import sqlite3
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QTableWidget, QTableWidgetItem,
    QMessageBox, QApplication
)
from PyQt5.QtCore import Qt

DB_PATH = r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\user.db"

class UserManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("用户管理")
        self.setMinimumSize(520, 400)
        self.init_ui()
        self.load_users()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(18)

        # 表格美化
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["ID", "账号", "密码"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget {
                background: #f4f8fb;
                font-size: 15px;
                border-radius: 8px;
            }
            QHeaderView::section {
                background-color: #43e97b;
                color: white;
                font-size: 15px;
                border: none;
                height: 32px;
            }
            QTableWidget::item:selected {
                background: #b2f7ef;
                color: #0078d7;
            }
        """)
        layout.addWidget(self.table)

        # 输入区美化
        form_layout = QHBoxLayout()
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("账号")
        self.user_edit.setStyleSheet("""
            QLineEdit {
                border: 1.5px solid #b0b0b0;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 15px;
                background: #f7fbff;
            }
        """)
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setPlaceholderText("密码")
        self.pwd_edit.setStyleSheet(self.user_edit.styleSheet())
        form_layout.addWidget(QLabel("账号:"))
        form_layout.addWidget(self.user_edit)
        form_layout.addWidget(QLabel("密码:"))
        form_layout.addWidget(self.pwd_edit)
        layout.addLayout(form_layout)

        # 按钮区美化
        btn_layout = QHBoxLayout()
        btn_style = """
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
        """
        self.add_btn = QPushButton("添加")
        self.add_btn.setStyleSheet(btn_style)
        self.add_btn.clicked.connect(self.add_user)
        btn_layout.addWidget(self.add_btn)

        self.update_btn = QPushButton("修改")
        self.update_btn.setStyleSheet(btn_style)
        self.update_btn.clicked.connect(self.update_user)
        btn_layout.addWidget(self.update_btn)

        self.delete_btn = QPushButton("删除")
        self.delete_btn.setStyleSheet(btn_style)
        self.delete_btn.clicked.connect(self.delete_user)
        btn_layout.addWidget(self.delete_btn)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setStyleSheet(btn_style)
        self.refresh_btn.clicked.connect(self.load_users)
        btn_layout.addWidget(self.refresh_btn)

        layout.addLayout(btn_layout)

        # 返回按钮美化
        self.back_btn = QPushButton("返回登录")
        self.back_btn.setStyleSheet("""
            QPushButton {
                background:#b0b0b0; color:white; border-radius:10px; font-size:15px; padding:8px 24px;
            }
            QPushButton:hover {
                background: #0078d7;
            }
        """)
        self.back_btn.clicked.connect(self.back_to_login)
        layout.addWidget(self.back_btn, alignment=Qt.AlignRight)

        self.setLayout(layout)
        self.table.cellClicked.connect(self.on_table_click)

    def load_users(self):
        self.table.setRowCount(0)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, username, password FROM users")
        for row_idx, (uid, username, password) in enumerate(c.fetchall()):
            self.table.insertRow(row_idx)
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(uid)))
            self.table.setItem(row_idx, 1, QTableWidgetItem(username))
            self.table.setItem(row_idx, 2, QTableWidgetItem(password))
        conn.close()

    def add_user(self):
        username = self.user_edit.text().strip()
        password = self.pwd_edit.text().strip()
        if not username or not password:
            QMessageBox.warning(self, "提示", "账号和密码不能为空")
            return
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            QMessageBox.information(self, "成功", "添加成功")
            self.load_users()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "失败", "账号已存在")
        finally:
            conn.close()

    def update_user(self):
        selected = self.table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "提示", "请先选择要修改的用户")
            return
        uid = int(self.table.item(selected, 0).text())
        username = self.user_edit.text().strip()
        password = self.pwd_edit.text().strip()
        if not username or not password:
            QMessageBox.warning(self, "提示", "账号和密码不能为空")
            return
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("UPDATE users SET username=?, password=? WHERE id=?", (username, password, uid))
            conn.commit()
            QMessageBox.information(self, "成功", "修改成功")
            self.load_users()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "失败", "账号已存在")
        finally:
            conn.close()

    def delete_user(self):
        selected = self.table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "提示", "请先选择要删除的用户")
            return
        uid = int(self.table.item(selected, 0).text())
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id=?", (uid,))
        conn.commit()
        conn.close()
        QMessageBox.information(self, "成功", "删除成功")
        self.load_users()

    def on_table_click(self, row, col):
        self.user_edit.setText(self.table.item(row, 1).text())
        self.pwd_edit.setText(self.table.item(row, 2).text())

    def back_to_login(self):
        from login import LoginWindow
        self.login_win = LoginWindow()
        self.login_win.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UserManager()
    win.show()
    sys.exit(app.exec_())