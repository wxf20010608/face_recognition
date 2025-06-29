from ultralytics import YOLO
# import torch
# torch.backends.cudnn.enabled = False  # 不使用cudnn加速

def main():
    # 加载预训练的YOLOv8模型
    model = YOLO(r"D:\AI_Learning\python\01_Learning\Target_Detection\yolov8n.pt")  # 使用预训练模型

    # 开始训练，设置数据文件路径、训练周期、图像尺寸和批量大小
    model.train(data=r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\data.yaml",
                epochs=50, 
                batch=16, 
                val=True,
                project=r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\runs\detect",
                name="myexp"
            )
    
    # #2. 单GPU训练
    # model.train(data="./dataset/mydata.yaml", epochs=50，save= true，save_period=10)  # 训练模型
    
    # #3. 多GPU训练
    # # results = model.train(data="coco8.yaml", epochs=30, imgsz=640, device=[0, 1])
    
    #4. 在验证集上进行验证模型
    metrics = model.val(data=r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\data.yaml")
    results = model(r"D:\AI_Learning\python\01_Learning\Target_Detection\face_recoginition_myself\dataset\valid\images\Anthony_LaPaglia_0002.jpg")  # 对图像进行预测
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen

if __name__ == '__main__':
    main()