#Import Libraries and Load Annotations

import os
import json

MOUNT_FOLDER = "/tmp/dataset"

ANNOTATIONS_PATH = "annotations/office/waste-object-tracking/real/all/latest.json"

# Load the annotation JSON file
with open(os.path.join(MOUNT_FOLDER, ANNOTATIONS_PATH)) as f:
    annotations = json.load(f)

img_map = {
    i["id"]: {
        "path": os.path.join(MOUNT_FOLDER, "images", i["file_name"]), 
        "annotations": []
    } for i in annotations["images"]
}

# Build a category mapping (e.g., {1: "PAPER CUP", 2: "PAPER SHEET", ...})
cat_map = {c["id"]: c["name"] for c in annotations["categories"]}

# Associate each annotation with the corresponding image entry in img_map
for a in annotations["annotations"]:
    if a["image_id"] in img_map:  # Safety check
        img_map[a["image_id"]]["annotations"].append(a)

print(f"Loaded annotations with {len(annotations['images'])} images and {len(annotations['annotations'])} polygons")


#Filtering Images
from PIL import Image
import os

def filter_bad_images(img_map):
    good_img_map = {}
    bad_files = []

    for img_id, entry in img_map.items():
        path = entry["path"]
        try:
            with Image.open(path) as img:
                img.verify()  # Verify will raise an exception if image is corrupted
            good_img_map[img_id] = entry  # Only keep valid images
        except Exception as e:
            print(f"Bad image detected: {path} | Error: {e}")
            bad_files.append(path)

    print(f"\nFiltering complete: {len(bad_files)} bad images found and excluded.")
    return good_img_map, bad_files

# Usage:
img_map, bad_files = filter_bad_images(img_map)


# Cell 0: Split data

image_ids = sorted(list(img_map.keys()))  # maintain natural order (sorted by id or filename)


train_cut = int(len(image_ids) * 0.8)
val_cut = int(len(image_ids) * 0.95)

train_ids = set(image_ids[:train_cut])
val_ids   = set(image_ids[train_cut:val_cut])
test_ids  = set(image_ids[val_cut:])

split_map = {
    **{i: "train" for i in train_ids},
    **{i: "val"   for i in val_ids},
    **{i: "test"  for i in test_ids},
}

with open("split_map.json", "w") as f:
    json.dump(split_map, f)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")


#Load Dataset
import json
from pathlib import Path

in_path = Path("split_map1.json")   #split_map.json
with in_path.open("r") as f:
    split_map = {int(k): v for k, v in json.load(f).items()}

print(f"Loaded split_map with {len(split_map)} entries")
print("Train:", sum(1 for v in split_map.values() if v == "train"),
      "Val:", sum(1 for v in split_map.values() if v == "val"),
      "Test:", sum(1 for v in split_map.values() if v == "test"))


################################################################################
#Pre-trained models
### RESNET50 

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Custom layer to replace Lambda
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def build_trainable_embedding_model(
    embedding_dim=128,
    input_shape=(224, 224, 3),
    dropout_rate=0.2
):
    # 1) Load ResNet50 backbone
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 2) Phase 1: freeze entire backbone
    base_model.trainable = False

    # 3) Build embedding head
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="projection_dropout")(x)
    x = layers.Dense(embedding_dim, name="projection_dense")(x)
    
    # Use custom L2 normalization
    outputs = L2Normalization(name="l2_normalization")(x)

    model = models.Model(inputs, outputs, name="trainable_embedding_model")
    return model

# Build the model (backbone frozen)
embedding_model1 = build_trainable_embedding_model()
embedding_model1.summary()


### ResNet101
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras import layers, models

def build_embedding_model_resnet101(embedding_dim=128, input_shape=(224, 224, 3)):
    base_model = ResNet101(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze all layers

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = preprocess_input(inputs)  # Preprocessing for ResNet101
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim, name="dense_projection")(x)  # 128-dim output
    outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),
                            name="l2_normalization")(x)

    model = models.Model(inputs, outputs, name="embedding_model_resnet101")
    return model

embedding_model1 = build_embedding_model_resnet101()
embedding_model1.summary()
embedding_model1.save("saved_models/embedding_model_resnet101.keras")




###MobileNetV2

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

def build_embedding_model_mobilenet(embedding_dim=128, input_shape=(224, 224, 3)):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze MobileNet

    inputs = layers.Input(shape=input_shape)
    x = preprocess_input(inputs)  # Preprocessing for MobileNetV2
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim)(x)
    outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2_normalization")(x)

    model = models.Model(inputs, outputs, name="embedding_model_mobilenetv2")
    return model

embedding_model4 = build_embedding_model_mobilenet()
embedding_model4.summary()
embedding_model4.save("saved_models/embedding_model_mobilenetv2.keras")


###Xception 
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

# Custom L2 normalization layer for embedding output
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def build_embedding_model_xception(input_shape=(224, 224, 3), embedding_dim=128, dropout_rate=0.2):
    # Load pretrained Xception backbone (exclude top)
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze for Phase 1

    inputs = Input(shape=input_shape, name="image_input")
    x = preprocess_input(inputs)  # Normalize as Xception expects
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="projection_dropout")(x)
    x = layers.Dense(embedding_dim, name="projection_dense")(x)
    outputs = L2Normalization(name="l2_normalization")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="embedding_model_xception")

embedding_model7 = build_embedding_model_xception()
embedding_model7.summary()




