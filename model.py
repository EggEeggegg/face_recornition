import cv2
import torch
from ultralytics import YOLO



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov8s.pt").to(device)

def predict(frame, input_size):

    resized = cv2.resize(frame, input_size)

    # เปลี่ยนภาพเป็น tensor และส่งไปที่ GPU
    frame_tensor = torch.from_numpy(resized).permute(2, 0, 1).float()  # HWC -> CHW
    frame_tensor /= 255.0  # Normalize
    frame_tensor = frame_tensor.unsqueeze(0).to(device)

    results = model(frame_tensor, conf=0.7, classes=[0], verbose=False)

    return results
