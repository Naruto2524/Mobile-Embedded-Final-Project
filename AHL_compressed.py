import os
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_


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
    kernel = np.ones(window) / window
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=acc
    )
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
    y_train = np.load(os.path.join(data_dir, "labels_train.npy"))

    hog_test = np.load(os.path.join(data_dir, "hog_test.npy"))
    acc_test = np.load(os.path.join(data_dir, "acc_test.npy"))
    y_test = np.load(os.path.join(data_dir, "labels_test.npy"))

    assert (
        hog_train.shape[1] == hog_dim
    ), f"Expected HOG dim {hog_dim}, got {hog_train.shape[1]}"
    assert (
        acc_train.shape[1] == acc_dim
    ), f"Expected ACC dim {acc_dim}, got {acc_train.shape[1]}"

    acc_train = smooth_acc(acc_train, window=5)
    acc_test = smooth_acc(acc_test, window=5)

    pca_dim = 512
    pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
    hog_train = pca.fit_transform(hog_train)
    hog_test = pca.transform(hog_test)

    # Normalize features
    hog_mean = hog_train.mean(axis=0, keepdims=True)
    hog_std = hog_train.std(axis=0, keepdims=True) + 1e-6
    acc_mean = acc_train.mean(axis=0, keepdims=True)
    acc_std = acc_train.std(axis=0, keepdims=True) + 1e-6

    hog_train = (hog_train - hog_mean) / hog_std
    hog_test = (hog_test - hog_mean) / hog_std
    acc_train = (acc_train - acc_mean) / acc_std
    acc_test = (acc_test - acc_mean) / acc_std

    # Build Dataset objects
    train_dataset = GestureDataset(hog_train, acc_train, y_train)
    test_dataset = GestureDataset(hog_test, acc_test, y_test)

    g = torch.Generator()
    g.manual_seed(42)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, num_classes, pca_dim, acc_dim


#  Adaptive Hidden Layer with temperature + hard gating
class AdaptiveHiddenLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        group_size: int,
        num_groups: int = 2,
        temperature: float = 1.0,
        hard_gating: bool = False,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.temperature = temperature
        self.hard_gating = hard_gating

        # Neuron groups
        self.groups = nn.ModuleList(
            [nn.Linear(input_dim, group_size) for _ in range(num_groups)]
        )

        self.lns = nn.ModuleList(
            [nn.LayerNorm(group_size) for _ in range(num_groups)]
        )

        # Selector
        self.selector = nn.Linear(input_dim, num_groups)

        # Initialization
        for m in self.groups:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.constant_(m.bias, 0.0)

        nn.init.xavier_uniform_(self.selector.weight)
        nn.init.constant_(self.selector.bias, 0.0)

    def set_temperature(self, tau: float):
        self.temperature = tau

    def enable_hard_gating(self, flag: bool = True):
        self.hard_gating = flag

    def forward(self, x: torch.Tensor):
        """
        x: (batch, input_dim)
        returns:
            y: (batch, group_size)
            alpha: (batch, num_groups)  -- selector probs
        """
        # Group outputs with ReLU
        group_outputs = []
        for linear, ln in zip(self.groups, self.lns):
            z = linear(x)
            z = ln(z)
            out = F.relu(z)
            group_outputs.append(out)

        # Stack -> (batch, N, group_size)
        groups_stack = torch.stack(group_outputs, dim=1)

        # Selector: (batch, N) -> softmax with temperature
        alpha_logits = self.selector(x)
        if self.temperature != 1.0:
            alpha_logits = alpha_logits / self.temperature
        alpha = F.softmax(alpha_logits, dim=1)

        if self.training or not self.hard_gating:
            alpha_expanded = alpha.unsqueeze(2)
            weighted = groups_stack * alpha_expanded
            y = weighted.sum(dim=1)
            return y, alpha

        top1_indices = alpha.argmax(dim=1)
        B = x.size(0)
        y = x.new_zeros(B, self.group_size)

        for g_idx, g in enumerate(self.groups):
            mask = top1_indices == g_idx
            if mask.any():
                x_g = x[mask]
                y_g = g(x_g)
                y[mask] = F.relu(self.lns[g_idx](y_g))

        return y, alpha


#  AHL Gesture Network
class GestureAHLNet(nn.Module):
    """
    Multi-modal gesture network:
      - Visual branch: HOG -> AHL(256) -> AHL(256)
      - Motion branch: ACC -> AHL(128) -> AHL(128)
      - Joint AHL on concatenated 256+128 -> 256
      - Classifier: 256 -> num_classes
    """

    def __init__(
        self,
        hog_dim=512,
        acc_dim=53,
        num_classes=10,
        dropout: float = 0.1,
        selector_temp: float = 1.0,
        hard_gating: bool = False,
    ):
        super().__init__()
        # HOG subnetwork
        self.hog_ahl1 = AdaptiveHiddenLayer(
            hog_dim, 256, num_groups=2,
            temperature=selector_temp, hard_gating=hard_gating
        )
        self.hog_drop1 = nn.Dropout(p=0.2)
        self.hog_ahl2 = AdaptiveHiddenLayer(
            256, 256, num_groups=2,
            temperature=selector_temp, hard_gating=hard_gating
        )
        self.hog_drop2 = nn.Dropout(p=0.2)

        # Accelerometer subnetwork
        self.acc_ahl1 = AdaptiveHiddenLayer(
            acc_dim, 128, num_groups=2,
            temperature=selector_temp, hard_gating=hard_gating
        )
        self.acc_drop1 = nn.Dropout(p=0.1)
        self.acc_ahl2 = AdaptiveHiddenLayer(
            128, 128, num_groups=2,
            temperature=selector_temp, hard_gating=hard_gating
        )
        self.acc_drop2 = nn.Dropout(p=0.1)

        # Joint subnetwork: input is 256 (HOG) + 128 (ACC) = 384
        self.joint_ahl = AdaptiveHiddenLayer(
            384, 256, num_groups=2,
            temperature=selector_temp, hard_gating=hard_gating
        )

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

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
        fused = torch.cat([out_hog, out_acc], dim=1)

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
        p = alpha.mean(dim=0)  # (N,)
        entropy = -(p + eps) * torch.log(p + eps)
        sbr_loss = sbr_loss + entropy.sum()
    return beta * sbr_loss


def set_model_hard_gating(model: nn.Module, flag: bool, temperature: float | None = None):
    for m in model.modules():
        if isinstance(m, AdaptiveHiddenLayer):
            m.enable_hard_gating(flag)
            if temperature is not None:
                m.set_temperature(temperature)


#  Training / Evaluation
def train_one_epoch(model, train_loader, optimizer, criterion, device, beta_sbr=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for hog_feats, acc_feats, labels in train_loader:
        hog_feats = hog_feats.to(device)
        acc_feats = acc_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, alpha_list = model(hog_feats, acc_feats)
        cls_loss = criterion(logits, labels)
        sbr_loss = selection_balancing_loss(alpha_list, beta=beta_sbr)
        loss = cls_loss + sbr_loss

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        # Accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, data_loader, criterion, device, beta_sbr=1.0, hard_gating: bool = False):
    model.eval()
    set_model_hard_gating(model, hard_gating)  # toggle before eval

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for hog_feats, acc_feats, labels in data_loader:
            hog_feats = hog_feats.to(device)
            acc_feats = acc_feats.to(device)
            labels = labels.to(device)

            logits, alpha_list = model(hog_feats, acc_feats)
            cls_loss = criterion(logits, labels)
            sbr_loss = selection_balancing_loss(alpha_list, beta=beta_sbr)
            loss = cls_loss + sbr_loss

            running_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total

    set_model_hard_gating(model, False)
    return avg_loss, acc


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    # Config
    data_dir = "./data"
    hog_dim = 8100
    acc_dim = 53
    num_classes = 10
    batch_size = 64
    num_epochs = 25
    init_lr = 0.001
    weight_decay = 1e-4
    beta_sbr = 0.001

    train_loader, test_loader, num_classes, hog_dim_after_pca, acc_dim = build_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        hog_dim=hog_dim,
        acc_dim=acc_dim,
        num_classes=num_classes,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Effective HOG dim after PCA:", hog_dim_after_pca)

    # Model, Optimizer, Scheduler
    model = GestureAHLNet(
        hog_dim=hog_dim_after_pca,
        acc_dim=acc_dim,
        num_classes=num_classes,
        selector_temp=1.0,
        hard_gating=False,
    ).to(device)
    
    num_params = count_params(model)
    print(f"Total parameters: {num_params/1e6:.3f} M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_test_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, beta_sbr=beta_sbr
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, beta_sbr=beta_sbr, hard_gating=False
        )

        scheduler.step()

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                model.state_dict(),
                os.path.join(data_dir, "best_ahl_model.pth"),
            )

    print(f"Best soft-gating test accuracy: {best_test_acc*100:.2f}%")

    set_model_hard_gating(model, True, temperature=0.5)
    comp_test_loss, comp_test_acc = evaluate(
        model, test_loader, criterion, device, beta_sbr=beta_sbr, hard_gating=True
    )
    print(
        f"[Compressed] Hard-gating test accuracy: {comp_test_acc*100:.2f}% "
        f"(loss {comp_test_loss:.4f})"
    )


if __name__ == "__main__":
    main()
