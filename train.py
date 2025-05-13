from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-seg-dwz.yaml")  # build a new model from scratch
# model = YOLO("samv8n-seg.yaml")  # build a new model from scratch
model = YOLO("/home/bob/experiment/dwz/yolov8s-seg-img1024.pt")  # load a pretrained model (recommended for training)

# Use the model
# results = model.train(data="/home/bob/code/git_repository/ultralytics-8.0.4/ultralytics/yolo/data/datasets/dwz-seg.yaml", epochs=60, imgsz=1024)  # train the model
# results = model.val(data="/home/bob/code/git_repository/ultralytics-8.0.4/ultralytics/yolo/data/datasets/dwz-seg.yaml", imgsz=1024)  # evaluate model performance on the validation set
results = model(
    source="/home/bob/experiment/nkthesis/predict/D8_S20241103103000_E20241103113040_1900.png",
    # source="/home/bob/experiment/nkthesis/predict/D9_S20241018070001_E20241018081401_16640.png",
    save=True,          # 自动保存带标注的图片
    save_txt=False,     # 是否保存检测结果的 txt 文件
    save_conf=False,    # 是否在 txt 中保存置信度
    project="/home/bob/experiment/nkthesis/predict/",
      name="predict"  # 指定保存目录
)  # predict on an image

for result in results:
    print(f"检测到 {len(result.boxes)} 个目标")
    if len(result.boxes) == 0:
        print("⚠️ 未检测到任何目标，请检查模型阈值！")
    else:
        result.save("manual_save.jpg")  # 

# success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format

# results = model("https://ultralytics.com/images/bus.jpg")