from ultralytics import YOLO
import os

model = YOLO('runs/detect/drone_detection3/weights/best.pt')

test_images_dir = 'test_images'
save_dir = 'predictions'
os.makedirs(save_dir, exist_ok=True)

image_files = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]

for image_file in image_files:
    image_path = os.path.join(test_images_dir, image_file)

    results = model.predict(source=image_path, conf=0.3, save=True, save_txt=False, save_crop=False, project=save_dir, name='.')
