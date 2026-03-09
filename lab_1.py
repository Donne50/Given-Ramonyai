from ultralytics import YOLO

# This tells the AI what to look at. 
# 1. "Asian market.jpg" (Image)
# 2. "Cars Moving On Road Footage.mp4" (Video)
# 3. 0 (Webcam)
source = "Asian market.jpg"

# We load the model and tell it to use "cpu" to avoid the DLL error
model = YOLO("yolov8n.pt") 

print(f"Starting AI on {source}...")

try:
    if source == 0 or (isinstance(source, str) and not source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))):
        # Tracking mode for Video/Webcam
        model.track(source=source, tracker="bytetrack.yaml", conf=0.35, show=True, save=True, device='cpu')
    else:
        # Prediction mode for Images
        model.predict(source=source, conf=0.35, show=True, save=True, device='cpu')
    
    print("Done! Check the 'runs' folder for your results.")
except Exception as e:
    print(f"An error occurred: {e}")