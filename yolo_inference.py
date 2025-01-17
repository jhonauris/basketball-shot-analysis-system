from ultralytics import YOLO

model = YOLO('yolo11x.pt')

# 
result = model.predict("input_videos/IMG_6151.mp4", save=True)

print("*****RESULTS START HEAR*****",result)

print("boxes: ")

for box in result[0].boxes:
    print(box)