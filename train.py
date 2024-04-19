from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Load a model
model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='open_thermal_pose.yaml', workers=0, epochs=100, imgsz=640, batch=32, val=True, cache=True)
