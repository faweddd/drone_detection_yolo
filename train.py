from ultralytics import YOLO
import torch

if __name__ == '__main__':

    model = YOLO('yolo8s.pt')

    model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    name='drone_detection'
    )
