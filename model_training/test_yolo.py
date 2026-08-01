from ultralytics import YOLO
import glob

# Load trained model
model = YOLO("runs/detect/train-5/weights/best.pt")

# Get all images from train + val folders
train_images = glob.glob("dataset/images/train/*.jpg")
val_images = glob.glob("dataset/images/val/*.jpg")

# Combine both datasets
images = train_images + val_images

print(f"Total train images: {len(train_images)}")
print(f"Total val images: {len(val_images)}")
print(f"Total images to test: {len(images)}")

# Run prediction on all images
results = model(images, save=True)

# Print detections
for r in results:
    print(r.boxes)

print("✅ Prediction completed on TRAIN + VAL datasets")