import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold

# Set the random seed
torch.manual_seed(42)
np.random.seed(42)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# dataset
class DroneDataset(Dataset):
    def __init__(self, high_img_dirs, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for class_id in range(10):
            high_imgs = sorted([
                os.path.join(high_img_dirs[class_id], f)
                for f in os.listdir(high_img_dirs[class_id])
                if f.endswith('.png') or f.endswith('.jpg')
            ], key=natural_sort_key)

            for img_path in high_imgs:
                self.data.append(img_path)
                self.labels.append(class_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        high_img_path = self.data[idx]
        label = self.labels[idx]
        high_img = Image.open(high_img_path).convert('RGB')
        if self.transform:
            high_img = self.transform(high_img)
        return high_img, torch.tensor(label, dtype=torch.long)

# DFP
class DFP(nn.Module):
    def __init__(self, num_classes=10):
        super(DFP, self).__init__()

        def mbconv_block(in_channels, out_channels, kernel_size, stride, expand_ratio):
            hidden_dim = in_channels * expand_ratio
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.silu = nn.SiLU()
        self.blocks = nn.Sequential(
            mbconv_block(32, 16, 3, 1, 1),
            mbconv_block(16, 24, 3, 2, 6),
            mbconv_block(24, 40, 5, 2, 6),
            mbconv_block(40, 80, 3, 2, 6),
            mbconv_block(80, 112, 5, 1, 6),
            mbconv_block(112, 192, 5, 2, 6),
            mbconv_block(192, 320, 3, 1, 6)
        )
        self.conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"DFP-EfficientNet Total Parameters: {total_params}")
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn_head(x)
        x = self.silu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# training function
def train_model(model, train_loader, val_loader, device, num_epochs=50, fold=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    best_acc = 0.0
    best_model_path = f'/media/zyj/zuoshun/DroneRF/四路特征/分类+11特征+展示完整训练+MFP优化+上下文优化7+8+训练50轮+10折+固定范围(-10到10)+不缩放best_model_fold_{fold}.pth'

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_trues = [], []
        for high_imgs, labels in train_loader:
            high_imgs = high_imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(high_imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_trues.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_trues, train_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_running_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for high_imgs, labels in val_loader:
                high_imgs = high_imgs.to(device)
                labels = labels.to(device)

                logits = model(high_imgs)
                loss = criterion(logits, labels)
                val_running_loss += loss.item()
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_acc = accuracy_score(val_trues, val_preds)
        val_f1 = f1_score(val_trues, val_preds, average='macro')

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)  # Save the model that performed best on the validation set

    # Save the training curves for each fold
    if fold is not None:
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold} Loss Curve')

        plt.subplot(1,2,2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Fold {fold} Accuracy Curve')

        plt.tight_layout()
        plt.savefig(f'/media/zyj/zuoshun/DroneRF/四路特征/分类+11特征+展示完整训练+MFP优化+上下文优化7+8+训练50轮+10折+固定范围(-10到10)+不缩放fold_{fold}_train_curves.png')
        plt.close()

    return best_acc, best_f1


# testing function
def test_model(model, test_loader, device):
    model.eval()
    preds, trues = [], []
    attentions = []
    with torch.no_grad():
        for high_imgs, labels in test_loader:
            high_imgs = high_imgs.to(device)
            labels = labels.to(device)

            logits = model(high_imgs)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
            attentions.extend([[] for _ in range(len(labels))])

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    print(f'Test Acc: {acc:.4f}, Test F1: {f1:.4f}')
    print("Confusion Matrix:")
    print(confusion_matrix(trues, preds))
    print("Classification Report:")
    print(classification_report(trues, preds))
    return preds, trues, attentions

def main():
    # data path
    base_path = '/media/zyj/zuoshun/DroneRF/四路特征'
    high_img_dirs = [os.path.join(base_path, f'high_img_{i}') for i in range(10)]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_dataset = DroneDataset(high_img_dirs, transform=transform)
    all_labels = np.array(full_dataset.labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    for trainval_idx, test_idx in sss.split(np.zeros(len(all_labels)), all_labels):
        pass

    trainval_dataset = torch.utils.data.Subset(full_dataset, trainval_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    stratified_labels = np.array(full_dataset.labels)[trainval_idx]
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    test_accs = []
    test_f1s = []

    trainval_indices = list(range(len(trainval_dataset)))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(trainval_indices, stratified_labels)):
        print(f'\nFold {fold + 1}')

        fold_train_indices = [trainval_idx[i] for i in train_idx]
        fold_val_indices = [trainval_idx[i] for i in val_idx]

        fold_train_dataset = torch.utils.data.Subset(full_dataset, fold_train_indices)
        fold_val_dataset = torch.utils.data.Subset(full_dataset, fold_val_indices)

        fold_train_loader = DataLoader(fold_train_dataset, batch_size=32, shuffle=True)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=32, shuffle=False)

        model = DFP().to(device)
        train_model(model, fold_train_loader, fold_val_loader, device, num_epochs=50, fold=fold+1)

        # Load the model that performed best on the validation set
        best_model_path = f'/media/zyj/zuoshun/DroneRF/四路特征/分类+11特征+展示完整训练+MFP优化+上下文优化7+8+训练50轮+10折+固定范围(-10到10)+不缩放best_model_fold_{fold+1}.pth'
        model.load_state_dict(torch.load(best_model_path))

        preds, trues, attentions = test_model(model, test_loader, device)
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='macro')
        print(f'[Fold {fold+1}] Test Acc: {acc:.4f}, F1: {f1:.4f}')
        test_accs.append(acc)
        test_f1s.append(f1)

        cm = confusion_matrix(trues, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f'Fold {fold+1} - Confusion Matrix on Test Set')
        plt.savefig(f'/media/zyj/zuoshun/DroneRF/四路特征/分类+11特征+展示完整训练+MFP优化+上下文优化7+8+训练50轮+10折+固定范围(-10到10)+不缩放test_confusion_matrix_fold_{fold+1}.png')
        plt.close()

        report = classification_report(trues, preds, digits=4)
        with open(f'/media/zyj/zuoshun/DroneRF/四路特征/分类+11特征+展示完整训练+MFP优化+上下文优化7+8+训练50轮+10折+固定范围(-10到10)+不缩放test_classification_report_fold_{fold+1}.txt', 'w') as rf:
            rf.write(f'Fold {fold+1} - Classification Report\n')
            rf.write(report)

    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    mean_f1 = np.mean(test_f1s)
    std_f1 = np.std(test_f1s)

    print(f'\n[10-Fold Test Evaluation]')
    print(f'Average Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'Average Test F1 Score: {mean_f1:.4f} ± {std_f1:.4f}')

    with open('/media/zyj/zuoshun/DroneRF/四路特征/分类+11特征+展示完整训练+MFP优化+上下文优化7+8+训练50轮+10折+固定范围(-10到10)+不缩放final_test_summary.txt', 'w') as f:
        for i in range(10):
            f.write(f'Fold {i+1} - Acc: {test_accs[i]:.4f}, F1: {test_f1s[i]:.4f}\n')
        f.write(f'\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n')
        f.write(f'Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n')

if __name__ == '__main__':
    main()



