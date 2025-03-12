from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-seg-dwz.yaml")  # build a new model from scratch
model = YOLO("samv8n-seg.yaml")  # build a new model from scratch
# model = YOLO("/root/code/DWZ/runs/segment/train/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="/root/code/DWZ/ultralytics/yolo/data/datasets/all-seg.yaml", epochs=60, imgsz=1024)  # train the model
results = model.val(data="/root/code/DWZ/ultralytics/yolo/data/datasets/all-seg.yaml", imgsz=1024)  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format


# results = model("https://ultralytics.com/images/bus.jpg")