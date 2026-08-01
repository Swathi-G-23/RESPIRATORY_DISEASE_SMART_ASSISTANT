import os
import random
import shutil

# =========================================
# PATHS
# =========================================

image_dir = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\images"

label_dir = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\labels"

# =========================================
# OUTPUT TRAIN / VAL PATHS
# =========================================

train_img = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\images\train"

val_img = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\images\val"

train_lbl = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\labels\train"

val_lbl = r"C:\Users\swath\OneDrive\Desktop\Smart Healthcare\dataset\labels\val"

# =========================================
# CREATE FOLDERS
# =========================================

os.makedirs(train_img, exist_ok=True)
os.makedirs(val_img, exist_ok=True)

os.makedirs(train_lbl, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

# =========================================
# GET ONLY IMAGE FILES
# =========================================

images = [

    f for f in os.listdir(image_dir)

    if f.lower().endswith((
        ".jpg",
        ".jpeg",
        ".png"
    ))
]

# =========================================
# RANDOM SHUFFLE
# =========================================

random.shuffle(images)

# =========================================
# TRAIN / VAL SPLIT
# =========================================

split_ratio = 0.8

split_index = int(
    len(images) * split_ratio
)

train_files = images[:split_index]

val_files = images[split_index:]

# =========================================
# COPY FUNCTION
# =========================================

def copy_files(file_list, img_dest, lbl_dest):

    for file in file_list:

        # IMAGE SOURCE
        img_src = os.path.join(
            image_dir,
            file
        )

        # LABEL SOURCE
        txt_name = os.path.splitext(file)[0] + ".txt"

        lbl_src = os.path.join(
            label_dir,
            txt_name
        )

        # =====================================
        # COPY IMAGE
        # =====================================

        shutil.copy(
            img_src,
            os.path.join(img_dest, file)
        )

        # =====================================
        # COPY LABEL
        # =====================================

        if os.path.exists(lbl_src):

            shutil.copy(
                lbl_src,
                os.path.join(lbl_dest, txt_name)
            )

# =========================================
# COPY TRAIN FILES
# =========================================

copy_files(
    train_files,
    train_img,
    train_lbl
)

# =========================================
# COPY VAL FILES
# =========================================

copy_files(
    val_files,
    val_img,
    val_lbl
)

# =========================================
# DONE
# =========================================

print("\n✅ Dataset Split Completed!")

print(
    f"\n📁 Train Images: {len(train_files)}"
)

print(
    f"📁 Validation Images: {len(val_files)}"
)