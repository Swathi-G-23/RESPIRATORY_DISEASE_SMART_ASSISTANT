import os
import json

# =========================================
# JSON ANNOTATION FOLDER
# =========================================

json_folder = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\annotations"

# =========================================
# OUTPUT LABEL FOLDER
# =========================================

output_folder = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\labels"

# CREATE LABELS FOLDER
os.makedirs(output_folder, exist_ok=True)

# =========================================
# START FRAME NUMBER
# =========================================

frame_number = 1394

# =========================================
# CLASS IDS
# =========================================

classes = {
    "asthma": 0,
    "pneumonia": 1,
    "respiratory infection": 2,
    "normal": 3
}

# =========================================
# SORT JSON FILES
# =========================================

json_files = sorted(os.listdir(json_folder))

# =========================================
# LOOP THROUGH JSON FILES
# =========================================

for file in json_files:

    if file.endswith(".json"):

        json_path = os.path.join(
            json_folder,
            file
        )

        with open(json_path, "r") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        # =========================================
        # OUTPUT TXT NAME
        # =========================================

        txt_filename = f"frame_{frame_number}.txt"

        txt_path = os.path.join(
            output_folder,
            txt_filename
        )

        with open(txt_path, "w") as txt_file:

            for shape in data["shapes"]:

                label = shape["label"].lower()

                if label not in classes:
                    continue

                class_id = classes[label]

                points = shape["points"]

                # =========================================
                # GET BOUNDING BOX
                # =========================================

                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)

                # =========================================
                # CONVERT TO YOLO FORMAT
                # =========================================

                x_center = (
                    (x_min + x_max) / 2
                ) / image_width

                y_center = (
                    (y_min + y_max) / 2
                ) / image_height

                width = (
                    x_max - x_min
                ) / image_width

                height = (
                    y_max - y_min
                ) / image_height

                txt_file.write(
                    f"{class_id} "
                    f"{x_center} "
                    f"{y_center} "
                    f"{width} "
                    f"{height}\n"
                )

        print(
            f"✅ Saved: {txt_filename}"
        )

        frame_number += 1

print("\n🎉 JSON to YOLO TXT Conversion Completed!")