#Cell 1: Load and Crop Objects

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

