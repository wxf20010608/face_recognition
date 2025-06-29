import os
import sqlite3
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import io
import torch
import uvicorn
import logging
from ultralytics import YOLO
import cv2
from PyQt5.QtWidgets import QMessageBox
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\face_db.sqlite"
IMAGE_DIR = r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\face_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# 相似度阈值
SIMILARITY_THRESHOLD = 0.5

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
try:
    facenet_model = InceptionResnetV1(pretrained='casia-webface').eval()
    logger.info("FaceNet模型加载成功")
except Exception as e:
    logger.error(f"FaceNet模型加载失败: {e}")
    raise

try:
    yolo_model_path = r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\runs\detect\myexp2\weights\best.pt"
    yolo_model = YOLO(yolo_model_path)
    logger.info("YOLOv8模型加载成功")
except Exception as e:
    logger.error(f"YOLOv8模型加载失败: {e}")
    raise

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        image_path TEXT,
        embedding BLOB
    )''')
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")

init_db()

def extract_embedding(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((160, 160))
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            emb = facenet_model(img_tensor).squeeze().numpy()
        emb = emb / np.linalg.norm(emb)
        emb = emb.astype(np.float32)
        return emb
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        raise

def extract_embedding_from_img(img):
    try:
        img = img.resize((160, 160))
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255
        img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            emb = facenet_model(img_tensor).squeeze().numpy()
        emb = emb / np.linalg.norm(emb)
        emb = emb.astype(np.float32)
        return emb
    except Exception as e:
        logger.error(f"图像特征提取失败: {e}")
        raise

@app.post("/add_face/")
async def add_face(
    file: UploadFile,
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
):
    try:
        image_bytes = await file.read()
        logger.info(f"Received data for adding face: Name='{name}', Age={age}, Gender='{gender}'")
        logger.info(f"开始为 {name} 提取特征")
        emb = extract_embedding(image_bytes)
        image_path = os.path.join(IMAGE_DIR, f"{name}_{file.filename}")
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        conn = get_conn()
        c = conn.cursor()
        c.execute("INSERT INTO faces (name, age, gender, image_path, embedding) VALUES (?, ?, ?, ?, ?)",
                  (name, age, gender, image_path, emb.tobytes()))
        conn.commit()
        conn.close()
        logger.info(f"成功添加人脸: {name}")
        return {"msg": "success"}
    except Exception as e:
        logger.error(f"添加人脸失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加人脸失败: {str(e)}")

@app.post("/recognize/")
async def recognize(file: UploadFile):
    try:
        logger.info("接收到识别请求")
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 使用YOLOv8检测人脸，降低置信度阈值提高检测率
        results = yolo_model.predict(img, verbose=False, conf=0.3)
        
        # 检查YOLOv8返回结果
        if not results or not hasattr(results[0], "boxes") or results[0].boxes is None or len(results[0].boxes) == 0:
            logger.info("未检测到人脸")
            return {"result": None, "msg": "no face detected"}

        # 获取检测框
        boxes = results[0].boxes
        logger.info(f"检测到 {len(boxes)} 个人脸")
        
        # 取置信度最高的人脸
        confidences = boxes.conf.cpu().numpy()
        max_idx = np.argmax(confidences)
        box_tensor = boxes.xyxy[max_idx]
        
        box = box_tensor.cpu().numpy().astype(int).tolist()
        
        # 防止box越界
        w, h = img.size
        x1, y1, x2, y2 = [max(0, min(v, w if i%2==0 else h)) for i, v in enumerate(box)]
        if x2 <= x1 or y2 <= y1:
            logger.warning("人脸框尺寸无效")
            return {"result": None, "msg": "box value error"}
        
        
        face_img = img.crop((x1, y1, x2, y2))
        
        # 提取人脸特征
        logger.info("开始提取人脸特征")
        emb = extract_embedding_from_img(face_img)
        emb = extract_embedding(image_bytes)
        if emb.shape != (512,):
            logger.error(f"添加人脸失败: 特征维度异常 {emb.shape}")
            raise HTTPException(status_code=500, detail="特征提取失败")
        
        # 从数据库获取所有人脸特征
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT id, name, age, gender, image_path, embedding FROM faces")
        faces = c.fetchall()
        conn.close()
        
        if not faces:
            logger.info("数据库中没有人脸数据")
            return {"msg": "no faces in db"}
        
        # 计算相似度
        max_sim = -1
        best = None
        for fid, name, age, gender, img_path, emb_bytes in faces:
            db_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if db_emb.shape != (512,):
                logger.warning(f"数据库人脸ID {fid} 特征维度异常: {db_emb.shape}，已跳过")
                continue
            sim = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))
            if sim > max_sim:
                max_sim = sim
                best = {"id": fid, "name": name, "age": age, "gender": gender, "image_path": img_path, "similarity": float(sim)}
        
        # 应用相似度阈值
        if max_sim >= SIMILARITY_THRESHOLD:
            logger.info(f"识别成功: {best['name']}, 相似度: {max_sim:.4f}")
            return {"result": best, "box": [x1, y1, x2, y2], "msg": "success"}
        else:
            logger.info(f"识别失败: 最高相似度 {max_sim:.4f} 低于阈值 {SIMILARITY_THRESHOLD}")
            return {"result": None, "box": [x1, y1, x2, y2], "msg": "no match found"}
            
    except Exception as e:
        logger.error(f"识别过程发生错误: {e}", exc_info=True)
        return {"result": None, "msg": f"server error: {str(e)}"}

@app.get("/list_faces/")
def list_faces():
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT id, name, age, gender, image_path FROM faces")
        faces = [{"id": fid, "name": name, "age": age, "gender": gender, "image_path": img_path} for fid, name, age, gender, img_path in c.fetchall()]
        conn.close()
        logger.info(f"返回 {len(faces)} 条人脸数据")
        return {"faces": faces}
    except Exception as e:
        logger.error(f"获取人脸列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取人脸列表失败: {str(e)}")

@app.delete("/delete_face/{face_id}")
def delete_face(face_id: int):
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE id=?", (face_id,))
        conn.commit()
        conn.close()
        logger.info(f"删除人脸ID: {face_id}")
        return {"msg": "deleted"}
    except Exception as e:
        logger.error(f"删除人脸失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除人脸失败: {str(e)}")

@app.post("/refresh_db/")
def refresh_db():
    return list_faces()

def stop_camera(self):
    self.camera_running = False
    if hasattr(self, "camera_thread") and self.camera_thread and self.camera_thread.is_alive():
        self.camera_thread.join()
        self.camera_thread = None
    if self.timer:
        self.timer.stop()
        self.timer = None
    if self.cap:
        self.cap.release()
        self.cap = None
    self.open_cam_btn.setText("开始识别")
    self.capture_btn.setEnabled(False)
    self.close_cam_btn.setEnabled(False)
    self.recog_img_display_label.clear()
    self.recog_img_display_label.setText("摄像头画面")
    self.recog_result_label.setText("识别结果：")

def toggle_camera(self):
    if self.cap is None:
        # 只有在"开始识别"时才检查
        if not self.img_path:
            QMessageBox.information(self, "提示", "请先打开摄像头或选择图片")
            return
        # ...打开摄像头或识别图片...
        # 设置 self.cap 不为 None
    else:
        # 只做停止，不弹提示
        self.stop_camera()

def closeEvent(self, event):
    self.stop_camera() # 这一行是正常的，它会先停止摄像头
    # 优雅地关闭线程池，等待所有任务完成
    if self.executor:
        self.executor.shutdown(wait=True)
    event.accept() # 这一行允许窗口关闭

if __name__ == "__main__":
    logger.info("启动人脸识别服务...")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False, log_level="info")