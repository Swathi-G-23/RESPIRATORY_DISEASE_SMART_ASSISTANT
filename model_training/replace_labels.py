import os
import json

# =========================
# INPUT JSON FOLDER
# =========================
json_folder = "dataset/annotations"

# =========================
# OUTPUT TXT FOLDER
# =========================
txt_folder = "dataset/labels"

os.makedirs(txt_folder, exist_ok=True)

# =========================
# START & END FRAME
# =========================
start_frame = 874
end_frame = 1398

# =========================
# CLASS MAP
# =========================
label_map = {
    "asthma": 0,
    "respiratory infection": 1,
    "pneumonia": 2,
    "normal": 3
}

# =========================
# CONVERT JSON TO TXT
# =========================
for file in os.listdir(json_folder):

    if file.endswith(".json"):

        # Extract frame number
        frame_number = int(
            file.replace("frame_", "").replace(".json", "")
        )

        # Skip outside range
        if frame_number < start_frame or frame_number > end_frame:
            continue

        json_path = os.path.join(json_folder, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        txt_name = file.replace(".json", ".txt")

        txt_path = os.path.join(txt_folder, txt_name)

        with open(txt_path, "w") as txt_file:

            for shape in data["shapes"]:

                label = shape["label"]

                class_id = label_map[label]

                points = shape["points"]

                x1 = points[0][0]
                y1 = points[0][1]

                x2 = points[1][0]
                y2 = points[1][1]

                # YOLO FORMAT
                x_center = ((x1 + x2) / 2) / image_width
                y_center = ((y1 + y2) / 2) / image_height

                width = abs(x2 - x1) / image_width
                height = abs(y2 - y1) / image_height

                txt_file.write(
                    f"{class_id} "
                    f"{x_center} "
                    f"{y_center} "
                    f"{width} "
                    f"{height}\n"
                )

        print(f"Converted: {txt_name}")

print("\nConversion Completed")