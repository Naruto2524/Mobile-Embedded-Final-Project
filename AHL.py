import os
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader








#  Dataset & Data Loaders




class GestureDataset(Dataset):




    def __init__(self, hog_data, acc_data, labels):
        assert len(hog_data) == len(acc_data) == len(labels)
        self.hog = torch.tensor(hog_data, dtype=torch.float32)
        self.acc = torch.tensor(acc_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)




    def __len__(self):
        return len(self.labels)




    def __getitem__(self, idx):
        return self.hog[idx], self.acc[idx], self.labels[idx]




def smooth_acc(acc, window=5):
    # acc: numpy array of shape (N, acc_dim)
    # Apply moving average smoothing along the sample axis
    kernel = np.ones(window) / window
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=acc)
    return smoothed


def build_loaders(
    data_dir: str,
    batch_size: int = 128,
    hog_dim: int = 8100,
    acc_dim: int = 53,
    num_classes: int = 10,
):




    # Load numpy arrays
    hog_train = np.load(os.path.join(data_dir, "hog_train.npy"))
    acc_train = np.load(os.path.join(data_dir, "acc_train.npy"))
    y_train   = np.load(os.path.join(data_dir, "labels_train.npy"))


    hog_test  = np.load(os.path.join(data_dir, "hog_test.npy"))
    acc_test  = np.load(os.path.join(data_dir, "acc_test.npy"))
    y_test    = np.load(os.path.join(data_dir, "labels_test.npy"))


    assert hog_train.shape[1] == hog_dim, f"Expected HOG dim {hog_dim}, got {hog_train.shape[1]}"
    assert acc_train.shape[1] == acc_dim, f"Expected ACC dim {acc_dim}, got {acc_train.shape[1]}"


    acc_train = smooth_acc(acc_train, window=5)
    acc_test = smooth_acc(acc_test, window=5)


    # Performing PCA
    #pca = PCA(n_components = 512, whiten=True)
    #hog_train = pca.fit_transform(hog_train)
    #hog_test = pca.transform(hog_test)




    # Normalize features
    hog_mean = hog_train.mean(axis=0, keepdims=True)
    hog_std  = hog_train.std(axis=0, keepdims=True) + 1e-6
    acc_mean = acc_train.mean(axis=0, keepdims=True)
    acc_std  = acc_train.std(axis=0, keepdims=True) + 1e-6


    hog_train = (hog_train - hog_mean) / hog_std
    hog_test  = (hog_test  - hog_mean) / hog_std
    acc_train = (acc_train - acc_mean) / acc_std
    acc_test  = (acc_test  - acc_mean) / acc_std




    # Build Dataset objects
    train_dataset = GestureDataset(hog_train, acc_train, y_train)
    test_dataset  = GestureDataset(hog_test,  acc_test,  y_test)


    g = torch.Generator()
    g.manual_seed(42)




    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator = g
    )




    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )




    return train_loader, test_loader, num_classes, hog_dim, acc_dim








#  Adaptive Hidden Layer




class AdaptiveHiddenLayer(nn.Module):




    def __init__(self, input_dim: int, group_size: int, num_groups: int = 2):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size




        # Neuron groups
        self.groups = nn.ModuleList([
            nn.Linear(input_dim, group_size) for _ in range(num_groups)
        ])




        self.lns = nn.ModuleList([
            nn.LayerNorm(group_size) for _ in range(num_groups)
        ])




        # Selector
        self.selector = nn.Linear(input_dim, num_groups)




        # Initialization: use Xavier for all
        for m in self.groups:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0.0)




        nn.init.xavier_uniform_(self.selector.weight)
        nn.init.constant_(self.selector.bias, 0.0)




    def forward(self, x: torch.Tensor):
        # Group outputs with ReLU
        group_outputs = []
        for linear, bn in zip(self.groups, self.lns):
            z = linear(x)
            z = bn(z)
            out = F.relu(z)
            group_outputs.append(out)




        # Stack -> (batch, N, group_size)
        groups_stack = torch.stack(group_outputs, dim=1)




        # Selector: (batch, N) -> softmax
        alpha_logits = self.selector(x)
        alpha = F.softmax(alpha_logits, dim=1)




        # Weighted sum over groups
        alpha_expanded = alpha.unsqueeze(2)           # (batch, N, 1)
        weighted = groups_stack * alpha_expanded      # (batch, N, group_size)
        y = weighted.sum(dim=1)                       # (batch, group_size)




        return y, alpha








#  AHL Gesture Network




class GestureAHLNet(nn.Module):
    """
    Multi-modal gesture network:
      - Visual branch: HOG -> AHL(64) -> AHL(128)
      - Motion branch: ACC -> AHL(64) -> AHL(128)
      - Joint AHL on concatenated 128+128 -> 256
      - Classifier: 256 -> num_classes
    """




    def __init__(self, hog_dim=8100, acc_dim=53, num_classes=10, dropout = 0.1):
        super().__init__()
        # HOG subnetwork
        self.hog_ahl1 = AdaptiveHiddenLayer(hog_dim, 256, num_groups=2)
        self.hog_drop1 = nn.Dropout(p = 0.2)    # Dropout Layer 1 for HOG
        self.hog_ahl2 = AdaptiveHiddenLayer(256,      256, num_groups=2)
        self.hog_drop2 = nn.Dropout(p = 0.2)    # Dropout Layer 2 for HOG




        # Accelerometer subnetwork
        self.acc_ahl1 = AdaptiveHiddenLayer(acc_dim, 128, num_groups=2)
        self.acc_drop1 = nn.Dropout(p = 0.1)    # Dropout Layer 1 for ACC
        self.acc_ahl2 = AdaptiveHiddenLayer(128,      128, num_groups=2)
        self.acc_drop2 = nn.Dropout(p = 0.1)    # Dropout Layer 2 for ACC




        # Joint subnetwork
        self.joint_ahl = AdaptiveHiddenLayer(384, 256, num_groups=2)




        # Dropout
        self.dropout = nn.Dropout(p = 0.1)




        # Final classifier
        self.classifier = nn.Linear(256, num_classes)




    def forward(self, hog_input, acc_input):
        """
        hog_input: (batch, hog_dim)
        acc_input: (batch, acc_dim)
        Returns:
          logits:    (batch, num_classes)
          alpha_list: list of selector distributions from all AHLs
        """
        alpha_list = []




        # HOG branch
        out_hog, alpha = self.hog_ahl1(hog_input)
        alpha_list.append(alpha)
        out_hog = self.hog_drop1(out_hog)
        out_hog, alpha = self.hog_ahl2(out_hog)
        alpha_list.append(alpha)
        out_hog = self.hog_drop2(out_hog)












        # ACC branch
        out_acc, alpha = self.acc_ahl1(acc_input)
        alpha_list.append(alpha)
        out_acc = self.acc_drop1(out_acc)
        out_acc, alpha = self.acc_ahl2(out_acc)
        alpha_list.append(alpha)
        out_acc = self.acc_drop2(out_acc)




        # Fuse
        fused = torch.cat([out_hog, out_acc], dim=1)  # (batch, 256)




        # Joint AHL
        out_joint, alpha = self.joint_ahl(fused)
        alpha_list.append(alpha)




        out_joint = self.dropout(out_joint)




        # Classifier
        logits = self.classifier(out_joint)




        return logits, alpha_list








#  Selection Balancing Regularizer




def selection_balancing_loss(alpha_list, beta: float = 1.0, eps: float = 1e-6):
    """
    SBR: encourages balanced usage of groups across a batch.
    For each AHL:
      p(n) = mean over batch of alpha_n (selector output)
      L_sbr = -beta * sum_n (p(n)+eps) log(p(n)+eps)
    """
    sbr_loss = 0.0
    for alpha in alpha_list:
        # alpha: (batch, N)
        p = alpha.mean(dim=0)  # (N,)
        entropy = -(p + eps) * torch.log(p + eps)
        sbr_loss = sbr_loss + entropy.sum()
    return beta * sbr_loss








#  Training / Evaluation




def train_one_epoch(model, train_loader, optimizer, criterion, device, beta_sbr=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0




    for hog_feats, acc_feats, labels in train_loader:
        hog_feats = hog_feats.to(device)
        acc_feats = acc_feats.to(device)
        labels    = labels.to(device)




        optimizer.zero_grad()




        logits, alpha_list = model(hog_feats, acc_feats)
        cls_loss = criterion(logits, labels)
        sbr_loss = selection_balancing_loss(alpha_list, beta=beta_sbr)
        loss = cls_loss + sbr_loss




        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5.0)
        optimizer.step()




        running_loss += loss.item() * labels.size(0)




        # Accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)




    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc








def evaluate(model, data_loader, criterion, device, beta_sbr=1.0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0




    with torch.no_grad():
        for hog_feats, acc_feats, labels in data_loader:
            hog_feats = hog_feats.to(device)
            acc_feats = acc_feats.to(device)
            labels    = labels.to(device)

            logits, alpha_list = model(hog_feats, acc_feats)
            cls_loss = criterion(logits, labels)
            sbr_loss = selection_balancing_loss(alpha_list, beta=beta_sbr)
            loss = cls_loss + sbr_loss

            running_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def main():
    # Config
    data_dir    = "./data"
    hog_dim     = 8100
    acc_dim     = 53
    num_classes = 10
    batch_size  = 64
    num_epochs  =  25
    init_lr     = 0.001
    weight_decay = 1e-4
    momentum     = 0.9
    beta_sbr     = 0.001


    # Data
    train_loader, test_loader, num_classes, hog_dim, acc_dim = build_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        hog_dim=hog_dim,
        acc_dim=acc_dim,
        num_classes=num_classes,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model, Optimimzer, Scheduler
    model = GestureAHLNet(hog_dim=hog_dim, acc_dim=acc_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        weight_decay=weight_decay
    )
    # LR decay by factor 0.5 every 15 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


    # Training loop
    best_test_acc = 0.0


    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, beta_sbr=beta_sbr
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, beta_sbr=beta_sbr
        )




        scheduler.step()




        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )




        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(data_dir, "best_ahl_model.pth"))




    print(f"Best test accuracy: {best_test_acc*100:.2f}%")




if __name__ == "__main__":
    main()









