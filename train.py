from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n-seg-dwz.yaml")  # build a new model from scratch
model = YOLO("/home/bob/dataset/dwz_test/model/last.pt")  # build a new model from scratch
# model = YOLO("/home/bob/experiment/dwz/yolov8n-seg.pt")  # load a pretrained model (recommended for training)

# Use the model
# results = model.train(data="/home/bob/code/git_repository/ultralytics-8.0.4/ultralytics/yolo/data/datasets/dwz-seg.yaml", epochs=5, imgsz=1024)  # train the model
# results = model.val(data="/home/bob/code/git_repository/ultralytics-8.0.4/ultralytics/yolo/data/datasets/dwz-seg.yaml", imgsz=1024)  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format


results = model(
    source="/home/bob/dataset/dwz_test/data/dwz_test3.mp4",
    save=True,
    save_txt=False,
    save_conf=False,
    project='/home/bob/dataset/dwz_test/data_res',
    name='predict'
    )
