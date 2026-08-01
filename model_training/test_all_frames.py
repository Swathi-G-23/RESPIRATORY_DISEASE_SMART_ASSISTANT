from ultralytics import YOLO
import os
import cv2

# =====================================
# LOAD MODEL
# =====================================

model = YOLO("best.pt")

# =====================================
# FOLDER PATH
# =====================================

folder_path = "dataset/images/val"

# =====================================
# LOOP THROUGH IMAGES
# =====================================

for image_name in os.listdir(folder_path):

    image_path = os.path.join(
        folder_path,
        image_name
    )

    # Skip non-images
    if not image_name.endswith((
        ".jpg",
        ".png",
        ".jpeg"
    )):
        continue

    # Predict
    results = model.predict(
        source=image_path,
        conf=0.10
    )

    # Read image
    img = cv2.imread(image_path)

    predicted_class = "No Detection"

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        confidence = float(box.conf[0])

        predicted_class = model.names[
            cls_id
        ]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Draw box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = (
            f"{predicted_class} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # =====================================
    # SHOW RESULT
    # =====================================

    print(
        f"{image_name} --> {predicted_class}"
    )

    cv2.imshow(
        "Prediction",
        img
    )

    key = cv2.waitKey(500)

    # Press q to quit
    if key == ord('q'):
        break

cv2.destroyAllWindows()