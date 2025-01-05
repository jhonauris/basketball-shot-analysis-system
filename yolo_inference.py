from ultralytics import YOLO

model = YOLO('yolov8x')

model.predict("input_videos/left_side_frame1054.jpg", save=True)