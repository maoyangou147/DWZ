from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/bob/experiment/dwz/yolov8s-seg.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="/home/bob/code/git_repository/ultralytics-8.0.4/ultralytics/yolo/data/datasets/dwz-seg.yaml", epochs=100, imgsz=640)  # train the model
results = model.val(data="/home/bob/code/git_repository/ultralytics-8.0.4/ultralytics/yolo/data/datasets/dwz-seg.yaml")  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format


# results = model("https://ultralytics.com/images/bus.jpg")