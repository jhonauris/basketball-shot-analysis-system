from ultralytics import YOLO

model = YOLO('yolo11x.pt')

model.predict("input_videos/left_side_frame1054.jpg", save=True)