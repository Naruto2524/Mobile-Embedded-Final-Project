import numpy as np
import os

os.makedirs("./data", exist_ok=True)

N_train = 800
N_test  = 200

hog_dim = 1200
acc_dim = 53
num_classes = 14

def make_features(n, dim):
    return np.random.randn(n, dim).astype(np.float32)

def make_labels(n):
    return np.random.randint(0, num_classes, size=(n,), dtype=np.int64)

print("Generating data...")

hog_train = make_features(N_train, hog_dim)
acc_train = make_features(N_train, acc_dim)
y_train   = make_labels(N_train)

hog_test  = make_features(N_test, hog_dim)
acc_test  = make_features(N_test, acc_dim)
y_test    = make_labels(N_test)

print("Saving .npy files...")

np.save("./data/hog_train.npy",   hog_train)
np.save("./data/acc_train.npy",   acc_train)
np.save("./data/labels_train.npy", y_train)

np.save("./data/hog_test.npy",    hog_test)
np.save("./data/acc_test.npy",    acc_test)
np.save("./data/labels_test.npy",  y_test)

print("Done")
print(f"Train shapes: hog={hog_train.shape}, acc={acc_train.shape}, labels={y_train.shape}")
print(f"Test shapes:  hog={hog_test.shape}, acc={acc_test.shape}, labels={y_test.shape}")
