#Cell 0: Import Libraries and Load Annotations

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


#cell 1: Filtering Images
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


# Cell 2: Split data

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


#cell 3: Load Dataset
import json
from pathlib import Path

in_path = Path("split_map1.json")   #split_map.json
with in_path.open("r") as f:
    split_map = {int(k): v for k, v in json.load(f).items()}

print(f"Loaded split_map with {len(split_map)} entries")
print("Train:", sum(1 for v in split_map.values() if v == "train"),
      "Val:", sum(1 for v in split_map.values() if v == "val"),
      "Test:", sum(1 for v in split_map.values() if v == "test"))

#Cell 4: Load and Crop Objects

def load_and_crop_objects(img_entry, img_size=(224, 224)):
    img_path = img_entry["path"]
    annotations = img_entry["annotations"]

    # Try to load the image safely
    try:
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_raw, channels=3).numpy()
    except:
        print(f" Skipped corrupt or unreadable image: {img_path}")
        return [], []

    h, w, _ = img.shape
    crops = []
    meta_infos = []

    for ann in annotations:
        seg = ann.get("segmentation", [])
        if not seg or len(seg[0]) < 6:      #at leasset 3 point for seg
            continue

        poly = np.array(seg[0], dtype=np.int32).reshape((-1, 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], color=1)     #creates a binary mask of the polygon-shaped object

        masked = img * np.expand_dims(mask, axis=-1)

        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        tight_crop = masked[y_min:y_max+1, x_min:x_max+1]

        # Try to resize safely
        try:
            padded = tf.image.resize_with_pad(tight_crop, img_size[0], img_size[1])
        except:
            continue  # skip problematic crops

        crops.append(tf.cast(padded, tf.uint8))

        meta_infos.append({
            "category_id": ann["category_id"],
            "is_new": ann.get("attributes", {}).get("new", "unknown") == "yes",
            "is_occluded": ann.get("attributes", {}).get("occluded", False)
        })

    return crops, meta_infos


#Test
import random

# Pick random train entry
random_entry = random.choice([v for k, v in img_map.items() if k in split_map and split_map[k] == "train"])

crops, metas = load_and_crop_objects(random_entry, (224, 224))

print("Crops:", len(crops))
print("Metas:", metas)

################################################################################
#Pre-trained models
###cell 5(1): RESNET50 

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


###cell 5(2): ResNet101
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




###cell 5(3): MobileNetV2

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


###cell 5(4): Xception 
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


#######################################################################################

# Cell 6: Using Embeddings for Cropped Objects
import random
import numpy as np
import cv2

def generate_embeddings(crops, embedding_model):
    """
    Given a list of cropped object and compute their embeddings

    crops: list of tf.Tensor (shape: (224, 224, 3)) 

    Returns:
        embeddings: np.array of shape (num_objects, embedding_dim)
    """
    if len(crops) == 0:
        return np.array([])


    crop_batch = tf.stack(crops)         #(N, 224, 224, 3)     N=number of crops in image

    # Pass through model
    embeddings = embedding_model(crop_batch, training=False)

    return embeddings.numpy()    #convert to numpy for easier use

random_img_entry = random.choice(list(img_map.values()))
crops, metas = load_and_crop_objects(random_img_entry)

# Generate embeddings
embeddings = generate_embeddings(crops, embedding_model)

print(f"Generated embeddings shape: {embeddings.shape}")  #(num_objects, 128)



# Cell 7: compute_cosine_similarity function

import numpy as np
from scipy.spatial.distance import cdist
import tensorflow as tf

def compute_cosine_similarity(embeddings_a, embeddings_b):
    """
    Compute cosine similarity between two sets of embeddings
    
        embeddings_a: np.array of shape (num_objects_a, embedding_dim)
        embeddings_b: np.array of shape (num_objects_b, embedding_dim)

    Returns:
        similarity_matrix: np.array of shape (num_objects_a, num_objects_b)
    """

    if embeddings_a.shape[0] == 0 or embeddings_b.shape[0] == 0:
        return np.zeros((embeddings_a.shape[0], embeddings_b.shape[0]))

    # Normalize embeddings
    embeddings_a = tf.math.l2_normalize(embeddings_a, axis=-1).numpy()
    embeddings_b = tf.math.l2_normalize(embeddings_b, axis=-1).numpy()

    # Compute cosine similarity (1 - cosine distance)
    similarity_matrix = 1.0 - cdist(embeddings_a, embeddings_b, metric='cosine')
    return similarity_matrix

    #In matrix: Objects of A in Row , Objects of B in column and Elements are similarity



# Cell 8: match_objects(with category_check)

THRESHOLD = 0.66
def match_objects_with_category_check(embeddings_a, embeddings_b, metas_a, metas_b, threshold=THRESHOLD):
    """
    Matches objects between two images based on cosine similarity and category consistency

    Args:
        embeddings_a: np.array or tf.Tensor, shape (num_objects_a, embedding_dim)
        embeddings_b: np.array or tf.Tensor, shape (num_objects_b, embedding_dim)
        metas_a: list of dicts with 'category_id' for each object in A
        metas_b: list of dicts with 'category_id' for each object in B
        threshold: float, minimum cosine similarity to accept a match

    Returns:
        matches: list of (index_a, index_b)
        unmatched_a: list of indices in A not matched
        unmatched_b: list of indices in B not matched
    """
    similarity = compute_cosine_similarity(embeddings_a, embeddings_b)

    matches = []
    unmatched_a = list(range(len(embeddings_a)))
    unmatched_b = list(range(len(embeddings_b)))

    used_b = set()  # Track already matched B objects

    for i in range(len(embeddings_a)):
        best_sim = -1.0
        best_j = None
        for j in range(len(embeddings_b)):
            if j in used_b:
                continue  # Skip already matched B indices

            if metas_a[i]['category_id'] != metas_b[j]['category_id']:
                continue
            sim = similarity[i, j]
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_j = j

        if best_j is not None:
            matches.append((i, best_j))
            if i in unmatched_a:
                unmatched_a.remove(i)
            if best_j in unmatched_b:
                unmatched_b.remove(best_j)
            used_b.add(best_j)  # Mark this B index as used

    return matches, unmatched_a, unmatched_b


# Cell 5.1: Pick, Match & Visualize 

import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

#----------------------------------# Version A — Use ALL images (no split)-------------------------

'''
entries = list(img_map.values())           #111, 189, 587, 234, 649, 400, 3479
entries.sort(key=lambda x: x['path'])  # Sort by path (time order)

# Pick two consecutive images
idx = 590
img_entry_a = entries[idx]
img_entry_b = entries[idx + 1]

print("Viewing Image A:", img_entry_a['path'])
print("Viewing Image B:", img_entry_b['path'])
'''

#----------------------------------# Version B — Use only one split (train)-------------------------

split_name = "test"

# Step 1: Extract (img_idx, entry) only for selected split
entries_with_idx = [(idx, entry) for idx, entry in img_map.items() if split_map.get(idx) == split_name]

# Step 2: Sort by path
entries_with_idx.sort(key=lambda x: x[1]['path'])

# Step 3: Select a sample index
idx = 205   #642
img_idx_a, img_entry_a = entries_with_idx[idx]
img_idx_b, img_entry_b = entries_with_idx[idx + 1]

print(f"Split: {split_name}")
print(f"Image A (idx={img_idx_a}): {img_entry_a['path']}")
print(f"Image B (idx={img_idx_b}): {img_entry_b['path']}")

#----------------------------------------------------------------------------------------------------------------------------------

crops_a, metas_a = load_and_crop_objects(img_entry_a)
crops_b, metas_b = load_and_crop_objects(img_entry_b)

embeddings_a = generate_embeddings(crops_a, embedding_model2)
embeddings_b = generate_embeddings(crops_b, embedding_model2)


matches, unmatched_a, unmatched_b = match_objects_with_category_check(
    embeddings_a, embeddings_b, metas_a, metas_b, threshold=THRESHOLD)



def visualize_matches(crops_a, crops_b, unmatched_a, unmatched_b, max_objects=10):
    num_a = min(len(crops_a), max_objects)
    num_b = min(len(crops_b), max_objects)
    cols = max(num_a, num_b)

    fig, axes = plt.subplots(2, cols, figsize=(3*cols, 6))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Row 0: A
    for i in range(cols):
        ax = axes[0, i]
        if i < num_a:
            img = crops_a[i].numpy().astype('uint8')
            ax.imshow(img)
            title = f"A{i}"
            if i in unmatched_a:
                title += " (Unmatched)"
            ax.set_title(title)
        ax.axis('off')

    # Row 1: B
    for i in range(cols):
        ax = axes[1, i]
        if i < num_b:
            img = crops_b[i].numpy().astype('uint8')
            ax.imshow(img)
            title = f"B{i}"
            if i in unmatched_b:
                title += " (Unmatched)"
            ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

visualize_matches(crops_a, crops_b, unmatched_a, unmatched_b, max_objects=10)

#print(f"Matches: {matches}")
#print(f"Unmatched in A: {unmatched_a}")
#print(f"Unmatched in B: {unmatched_b}")

# NEW: Print total object counts
print(f"Total objects detected in Image A: {len(crops_a)}")
print(f"Total objects detected in Image B: {len(crops_b)}")


# Compute cosine similarity matrix (reuse the function from before)
similarity_matrix = compute_cosine_similarity(embeddings_a, embeddings_b)


plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="viridis", 
            xticklabels=[f"B{i}" for i in range(similarity_matrix.shape[1])],
            yticklabels=[f"A{i}" for i in range(similarity_matrix.shape[0])])

plt.xlabel("Objects in Image B")
plt.ylabel("Objects in Image A")
plt.title("Cosine Similarity Matrix with Values")
plt.show()



#Cell 10: Calculation of added objects

import os
import matplotlib.pyplot as plt
import tensorflow as tf


#Count formula
num_changes = len(unmatched_b)
print(f"Total Changes (added in B): {num_changes}")  #for B

num_removed  = len(unmatched_a)  # Objects from Image A that disappeared
#print(f"Total Changes (dissapeared in A): {num_removed}")  #for B

#total_changes = num_added + num_removed   #A+B

# Load full-resolution images
img_a = tf.io.read_file(img_entry_a["path"])
img_a = tf.image.decode_jpeg(img_a, channels=3).numpy()

img_b = tf.io.read_file(img_entry_b["path"])
img_b = tf.image.decode_jpeg(img_b, channels=3).numpy()

# Extract filenames
fname_a = os.path.basename(img_entry_a["path"])
fname_b = os.path.basename(img_entry_b["path"])

# Show side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_a)
axes[0].set_title(f"Image A:\n{fname_a}")
axes[0].axis('off')

axes[1].imshow(img_b)
axes[1].set_title(f"Image B:\n{fname_b}")
axes[1].axis('off')

plt.tight_layout()
plt.show()






