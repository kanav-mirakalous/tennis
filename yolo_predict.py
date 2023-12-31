from ultralytics import YOLO

# Load a model
model = YOLO('yolo/train3/weights/best.onnx')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('images/AO.RodLaverArena1.mp4', stream=True, show=True, save=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs