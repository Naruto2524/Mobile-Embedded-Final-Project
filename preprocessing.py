import os
import glob
import numpy as np
from PIL import Image
from skimage.feature import hog


def collect_samples(root_dir, subject_ids):
    samples = []

    for sid in subject_ids:
        subj_str = f"{sid:02d}"
        subj_dir = os.path.join(root_dir, subj_str)
        if not os.path.isdir(subj_dir):
            print(f"[WARN] Subject folder not found: {subj_dir}")
            continue

        gesture_dirs = sorted(
            d for d in os.listdir(subj_dir)
            if os.path.isdir(os.path.join(subj_dir, d))
        )

        for gdir in gesture_dirs:
            try:
                gesture_num = int(gdir[:2])
            except ValueError:
                print(f"[WARN] Skipping unexpected folder name: {gdir}")
                continue

            label = gesture_num - 1

            gpath = os.path.join(subj_dir, gdir)
            img_paths = sorted(
                glob.glob(os.path.join(gpath, "*.png"))
            )

            for img_path in img_paths:
                samples.append((img_path, label))

    return samples


def build_features(samples, img_size=128, acc_dim=53):
    hog_list = []
    acc_list = []
    label_list = []

    for path, label in samples:
        img = Image.open(path).convert("L")
        img = img.resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32) / 255.0

        hog_vec = hog(
            arr,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )

        hog_list.append(hog_vec.astype(np.float32))

        acc_vec = np.zeros(acc_dim, dtype=np.float32)
        acc_list.append(acc_vec)

        label_list.append(label)

    hog_feats = np.stack(hog_list, axis=0)
    acc_feats = np.stack(acc_list, axis=0)
    labels = np.array(label_list, dtype=np.int64)

    return hog_feats, acc_feats, labels


def main():
    root_dir = "./leapGestRecog"

    train_subjects = list(range(0, 8))
    test_subjects = list(range(8, 10))

    print("Collecting training samples...")
    train_samples = collect_samples(root_dir, train_subjects)
    print(f"  Found {len(train_samples)} training images.")

    print("Collecting test samples...")
    test_samples = collect_samples(root_dir, test_subjects)
    print(f"  Found {len(test_samples)} test images.")

    if not train_samples or not test_samples:
        raise RuntimeError("No samples found. Check root_dir path and structure.")

    print("Building training features...")
    hog_train, acc_train, labels_train = build_features(train_samples)
    print("  Training shapes:",
          hog_train.shape, acc_train.shape, labels_train.shape)

    print("Building test features...")
    hog_test, acc_test, labels_test = build_features(test_samples)
    print("  Test shapes:",
          hog_test.shape, acc_test.shape, labels_test.shape)

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "hog_train.npy"), hog_train)
    np.save(os.path.join(data_dir, "acc_train.npy"), acc_train)
    np.save(os.path.join(data_dir, "labels_train.npy"), labels_train)

    np.save(os.path.join(data_dir, "hog_test.npy"), hog_test)
    np.save(os.path.join(data_dir, "acc_test.npy"), acc_test)
    np.save(os.path.join(data_dir, "labels_test.npy"), labels_test)

    print(f"Saved .npy files in: {data_dir}")
    print("Update AHL.py with:")
    print(f"  hog_dim = {hog_train.shape[1]}")
    print("  acc_dim = 53 ")
    print("  num_classes = 10")


if __name__ == "__main__":
    main()
