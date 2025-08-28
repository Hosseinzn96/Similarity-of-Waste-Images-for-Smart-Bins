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


# Cell 9: Pick, Match & Visualize 

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


# Cell 11: Triplet Loss Function
import tensorflow as tf

def triplet_loss(anchor, positive, negative, margin=0.3):  
    """
    Computes triplet loss using cosine distance.

    Args:
        anchor, positive, negative: Embedding vectors (B, D)
        margin: float, enforced margin between positive and negative distances

    Returns:
        Tensor loss value
    """
    # Normalize for cosine similarity
    anchor = tf.math.l2_normalize(anchor, axis=-1)
    positive = tf.math.l2_normalize(positive, axis=-1)
    negative = tf.math.l2_normalize(negative, axis=-1)

    # Cosine similarity → convert to distance
    sim_ap = tf.reduce_sum(anchor * positive, axis=-1)  # similarity(anchor, pos)
    sim_an = tf.reduce_sum(anchor * negative, axis=-1)  # similarity(anchor, neg)

    loss = tf.maximum(0.0, margin + sim_an - sim_ap)
    return tf.reduce_mean(loss)



#Cell 12: generate triplets_loss 

import random
import numpy as np

# ----------------------------- #
# OPTIONAL BIN FILTERING HELPER #
# ----------------------------- #

def should_generate_triplets(similarity_matrix, bin_threshold=0.75):
    """
    Decide if this image pair should generate triplets based on object similarity.

    Args:
        similarity_matrix: numpy array, cosine similarity matrix between objects A and B
        bin_threshold: float, threshold for deciding same bin (default 0.75)

    Returns:
        bool → True if triplets should be generated (same bin), False to skip (different bin)
    """
    return np.any(similarity_matrix > bin_threshold)


# ----------------------------- #
# TRIPLET GENERATION FUNCTION   #
# ----------------------------- #

def generate_triplets_loss(img_map, embedding_model, split_name="train",
                           threshold_pos=0.67, threshold_neg=0.48,        #pos=73, neg=0.44
                           img_size=(224, 224), 
                           max_per_image=20, 
                           num_passes=1,
                           enable_bin_filtering=False):  # CONTROL BIN FILTERING HERE
    """
    Generate triplets with multiple passes through dataset for higher coverage.

    Args:
        enable_bin_filtering: bool → Set to True to enable bin filtering (skip unrelated image pairs).
    """

    anchors, positives, negatives = [], [], []
    entries = [v for k, v in img_map.items() if split_map.get(k) == split_name]
    entries.sort(key=lambda x: x['path'])  # Consistent order

    for pass_num in range(num_passes):
        print(f"Triplet generation pass {pass_num + 1}/{num_passes}")

        for idx in range(len(entries) - 1):
            img_a = entries[idx]
            img_b = entries[idx + 1]

            crops_a, metas_a = load_and_crop_objects(img_a, img_size)
            crops_b, metas_b = load_and_crop_objects(img_b, img_size)

            if not crops_a or not crops_b:
                continue

            embeddings_a = generate_embeddings(crops_a, embedding_model)
            embeddings_b = generate_embeddings(crops_b, embedding_model)

            similarity = compute_cosine_similarity(embeddings_a, embeddings_b)

            # --------------------------------------- #
            # OPTIONAL BIN FILTERING (SAFE TO COMMENT) #
            # --------------------------------------- #
            
            if enable_bin_filtering:
                if not should_generate_triplets(similarity):
                    # Skip this pair → likely different bin → do not generate triplets
                    continue

            triplet_count = 0
            used_b = set()  # To ensure each B object is used only once as positive

            for i, meta_a in enumerate(metas_a):
                cat_a = meta_a['category_id']

                # Positive candidates (same category + above threshold + not used yet)
                candidates_pos = [
                    j for j, meta_b in enumerate(metas_b)
                    if meta_b['category_id'] == cat_a and similarity[i, j] >= threshold_pos and j not in used_b
                ]

                if not candidates_pos:
                    continue

                # Choose best positive (highest similarity)
                j = max(candidates_pos, key=lambda idx: similarity[i, idx])
                used_b.add(j)

                anchors.append(crops_a[i])
                positives.append(crops_b[j])

                # Negative selection (same category, low similarity preferred)
                neg = None
                for m_idx, meta_b in enumerate(metas_b):
                    if meta_b['category_id'] == cat_a and similarity[i, m_idx] <= threshold_neg:
                        neg = crops_b[m_idx]
                        break

                # Fallback negative (random object from other image)
                if neg is None:
                    while True:
                        neg_entry = random.choice(entries)
                        if neg_entry in [img_a, img_b]:
                            continue
                        neg_crops, neg_metas = load_and_crop_objects(neg_entry, img_size)
                        if not neg_crops:
                            continue
                        k = random.randint(0, len(neg_crops) - 1)
                        neg = neg_crops[k]
                        break

                negatives.append(neg)

                triplet_count += 1
                if triplet_count >= max_per_image:
                    break

    print(f"Generated total {len(anchors)} triplets.")
    return anchors, positives, negatives



#Cell 13(1): generate validation triplets
import pickle
import os
import tensorflow as tf

# ---- CONFIG ----
MAX_VAL_TRIPLETS = 5500  # Maximum triplets to generate and save

# ---- GENERATE ----
anchors_val, positives_val, negatives_val = generate_triplets_loss(
    img_map,
    embedding_model,
    split_name="val"  # Use val split
)

# Check total generated
total_val_triplets = len(anchors_val)
print("Total generated validation triplets:", total_val_triplets)

# ---- TRIM IF NEEDED ----
if total_val_triplets > MAX_VAL_TRIPLETS:
    print(f"Trimming validation triplets to {MAX_VAL_TRIPLETS}")
    anchors_val = anchors_val[:MAX_VAL_TRIPLETS]
    positives_val = positives_val[:MAX_VAL_TRIPLETS]
    negatives_val = negatives_val[:MAX_VAL_TRIPLETS]

# ---- CONVERT TO NUMPY ----
anchors_val_np = tf.stack(anchors_val).numpy()
positives_val_np = tf.stack(positives_val).numpy()
negatives_val_np = tf.stack(negatives_val).numpy()

# ---- SAVE ----
save_path = "saved_triplets/val_triplets_Xception.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "wb") as f:
    pickle.dump({
        "anchors": anchors_val_np,
        "positives": positives_val_np,
        "negatives": negatives_val_np
    }, f)

print("Saved validation triplets to:", save_path)


#Cell 13(2): Load Validation Triplets

import pickle

# Load triplets
val_triplet_path = "saved_triplets/val_triplets_for_mobilenet.pkl"

#"saved_triplets/val_triplets.pkl"  for  Resnet50
#saved_triplets/val_triplets2.pkl for resnet50 2
#"saved_triplets/val_triplets_for_ResNet101.pkl
#"saved_triplets/val_triplets_for_mobilenet.pkl"  for mobilenet
#"saved_triplets/val_triplets_for_mobilenetV2.pkl"

with open(val_triplet_path, "rb") as f:
    val_triplets = pickle.load(f)

anchors_val = val_triplets["anchors"]
positives_val = val_triplets["positives"]
negatives_val = val_triplets["negatives"]

print("Loaded triplets shapes:")
print("Anchors:", anchors_val.shape)
print("Positives:", positives_val.shape)
print("Negatives:", negatives_val.shape)



#Cell 13(3): Visualize Valid Triplets
import matplotlib.pyplot as plt
import numpy as np

# How many triplets to show
num_samples = 5

# Random indices
indices = np.random.choice(anchors_val.shape[0], size=num_samples, replace=False)

# Plot
fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

for idx, triplet_idx in enumerate(indices):
    anchor_img = anchors_val[triplet_idx]
    positive_img = positives_val[triplet_idx]
    negative_img = negatives_val[triplet_idx]
    
    # Anchor
    axes[idx, 0].imshow(anchor_img)
    axes[idx, 0].set_title("Anchor")
    axes[idx, 0].axis('off')
    
    # Positive
    axes[idx, 1].imshow(positive_img)
    axes[idx, 1].set_title("Positive")
    axes[idx, 1].axis('off')
    
    # Negative
    axes[idx, 2].imshow(negative_img)
    axes[idx, 2].set_title("Negative")
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.show()



#Cell 13(4): generate Train triplets

import pickle
import os
import tensorflow as tf

# ---- GENERATE ----
anchors_train, positives_train, negatives_train = generate_triplets_loss(
    img_map,
    embedding_model,
    split_name="train"  #  Full train part, no trimming
)

# ---- CONVERT TO NUMPY ----
anchors_train_np = tf.stack(anchors_train).numpy()
positives_train_np = tf.stack(positives_train).numpy()
negatives_train_np = tf.stack(negatives_train).numpy()

# ---- SAVE TO FILE ----
save_path = "saved_triplets/Xception.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "wb") as f:
    pickle.dump({
        "anchors": anchors_train_np,
        "positives": positives_train_np,
        "negatives": negatives_train_np
    }, f)

print("Saved all train triplets to:", save_path)
print("Triplet counts:", len(anchors_train_np))



##Cell 13(5): Load Train triplets
import pickle

train_triplet_path = "saved_triplets/train_triplets_for_mobilenet.pkl"


with open(train_triplet_path, "rb") as f:
    train_triplets = pickle.load(f)

anchors_train = train_triplets["anchors"]
positives_train = train_triplets["positives"]
negatives_train = train_triplets["negatives"]

print("Loaded triplets:")
print("Anchors shape:  ", anchors_train.shape)
print("Positives shape:", positives_train.shape)
print("Negatives shape:", negatives_train.shape)



####Cell 13(6):Visualize Train Triplets

import matplotlib.pyplot as plt
import numpy as np

# Number of triplets to show
num_samples = 5

# Randomly pick 10 indices
indices = np.random.choice(anchors_train.shape[0], size=num_samples, replace=False)

# Plot
fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

for idx, triplet_idx in enumerate(indices):
    anchor_img = anchors_train[triplet_idx]
    positive_img = positives_train[triplet_idx]
    negative_img = negatives_train[triplet_idx]
    
    # Anchor
    axes[idx, 0].imshow(anchor_img)
    axes[idx, 0].set_title("Anchor")
    axes[idx, 0].axis('off')
    
    # Positive
    axes[idx, 1].imshow(positive_img)
    axes[idx, 1].set_title("Positive")
    axes[idx, 1].axis('off')
    
    # Negative
    axes[idx, 2].imshow(negative_img)
    axes[idx, 2].set_title("Negative")
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.show()




# Cell 14:  tf.data pipeline for triplet training

def create_triplet_dataset(anchors, positives, negatives, batch_size=32, shuffle=True, img_size=(224, 224)):
    if len(anchors) == 0:
        print("No triplets to create dataset → returning empty dataset.")
        dummy = tf.zeros((0, img_size[0], img_size[1], 3))
        return tf.data.Dataset.from_tensor_slices(((dummy, dummy, dummy), tf.zeros((0,)))).batch(batch_size)

    # Stack tensors
    anchors = tf.stack(anchors)
    positives = tf.stack(positives)
    negatives = tf.stack(negatives)

    # Create tuple-style dataset     
    ds = tf.data.Dataset.from_tensor_slices(((anchors, positives, negatives), tf.zeros((anchors.shape[0],))))  #triplets are then wrapped into a tf.data.Dataset object using from_tensor_sli

    if shuffle:
        ds = ds.shuffle(1024)

    ds = ds.batch(batch_size)
    return ds


# Create train_ds and val_ds

train_ds = create_triplet_dataset(
    anchors_train, positives_train, negatives_train,
    batch_size=32, shuffle=True
)

val_ds = create_triplet_dataset(
    anchors_val, positives_val, negatives_val,
    batch_size=32, shuffle=False
)

# Only count once
train_batches = sum(1 for _ in train_ds)
val_batches = sum(1 for _ in val_ds)

print("Datasets ready!")
print("Train batches:", train_batches)
print("Val batches:", val_batches)



#Cell 15: Imports & Custom Triplet Loss Function
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Custom Triplet Loss (based on cosine similarity)
def custom_triplet_loss(anchor, positive, negative, margin=0.3):
    anchor = tf.math.l2_normalize(anchor, axis=-1)
    positive = tf.math.l2_normalize(positive, axis=-1)
    negative = tf.math.l2_normalize(negative, axis=-1)

    pos_sim = tf.reduce_sum(anchor * positive, axis=-1)
    neg_sim = tf.reduce_sum(anchor * negative, axis=-1)

    loss = tf.maximum(0.0, margin - pos_sim + neg_sim)
    return tf.reduce_mean(loss)

# Wrap into Keras-compatible loss
class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
        return custom_triplet_loss(anchor, positive, negative, self.margin)













