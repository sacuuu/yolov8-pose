from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Load a model
model = YOLO('yolov8m-pose.yaml')  # build a new model from YAML
model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8m-pose.yaml').load('yolov8m-pose.pt')  # build from YAML and transfer weights


# Train the model
results = model.train(data='coco-pose.yaml', workers=0, epochs=1, imgsz=640, batch=8)
