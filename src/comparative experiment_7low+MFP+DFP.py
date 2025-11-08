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
import torch.nn.functional as F
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# 数据集
class DroneDataset(Dataset):
    def __init__(self, low_csv_files, low_img_dirs, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.scaler_low = StandardScaler()

        for class_id in range(10):
            # 低频
            df_low = pd.read_csv(low_csv_files[class_id])
            low_features = df_low.iloc[:, :11].values
            labels = df_low.iloc[:, -1].values
            low_scaled = self.scaler_low.fit_transform(low_features)
            low_imgs = sorted([
                os.path.join(low_img_dirs[class_id], f)
                for f in os.listdir(low_img_dirs[class_id])
                if f.endswith('.png') or f.endswith('.jpg')
            ], key=natural_sort_key)

            for i in range(len(low_features)):
                self.data.append((
                    low_scaled[i], low_imgs[i],
                ))
                self.labels.append(class_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        low_scaled, low_img_path = self.data[idx]
        label = self.labels[idx]
        low_img = Image.open(low_img_path).convert('RGB')
        if self.transform:
            low_img = self.transform(low_img)
        return (torch.tensor(low_scaled, dtype=torch.float32),
                low_img,
                torch.tensor(label, dtype=torch.long))


# MFP
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, n_d, n_a, virtual_batch_size=128):
        super().__init__()
        self.output_dim = n_d + n_a
        self.linear = nn.Linear(input_dim, self.output_dim * 2)
        self.bn = nn.BatchNorm1d(self.output_dim * 2)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = F.glu(x, dim=1)
        return x

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, n_a, virtual_batch_size):
        super().__init__()
        self.linear = nn.Linear(n_a, input_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, prior):
        x = self.linear(x)
        x = self.bn(x)
        x = x * prior
        return self.softmax(x)

class MFP(nn.Module):
    def __init__(self, input_dim=11, num_classes=10, n_d=16, n_a=16, n_steps=5, gamma=1.5, virtual_batch_size=128):
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.input_dim = input_dim
        self.gamma = gamma
        self.virtual_batch_size = virtual_batch_size

        self.initial_splitter = FeatureTransformer(
            input_dim, n_d, n_a, virtual_batch_size=virtual_batch_size
        )
        self.feat_transformers = nn.ModuleList([
            FeatureTransformer(input_dim, n_d, n_a, virtual_batch_size=virtual_batch_size)
            for _ in range(n_steps - 1)
        ])

        self.attentive_transformers = nn.ModuleList([
            AttentiveTransformer(input_dim, n_a, virtual_batch_size)
            for _ in range(n_steps)
        ])

        self.fc = nn.Linear(n_d * n_steps, num_classes)
        self.scale = np.sqrt(0.5)
        self.dropout = nn.Dropout(0.1)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"TabNet Total Parameters: {total_params}")

    def forward(self, x):
        batch_size = x.size(0)
        prior = torch.ones(batch_size, self.input_dim, device=x.device)
        agg_mask = torch.zeros_like(prior)
        decision_outputs = []

        x_res = self.initial_splitter(x)
        d = x_res[:, :self.n_d]
        a = x_res[:, self.n_d:]
        decision_outputs.append(d)

        mask = self.attentive_transformers[0](a, prior)
        agg_mask += mask
        prior = prior * (self.gamma - mask)
        x = x * mask

        for step in range(self.n_steps - 1):
            x_res = self.feat_transformers[step](x)
            d = x_res[:, :self.n_d]
            a = x_res[:, self.n_d:]
            decision_outputs.append(d)

            mask = self.attentive_transformers[step + 1](a, prior)
            agg_mask += mask
            prior = prior * (self.gamma - mask)
            x = x * mask

        final_output = torch.cat(decision_outputs, dim=1)
        return self.fc(final_output)


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

# 注意力融合
class AttentionFusion(nn.Module):
    def __init__(self, reg_lambda=0.001):
        super(AttentionFusion, self).__init__()
        self.alpha_logits = nn.Parameter(torch.tensor([0.0, 0.0]))  # 可训练
        self.reg_lambda = reg_lambda

    def forward(self, p_mfp_low, p_dfp_low):
        alpha = torch.softmax(self.alpha_logits, dim=0)  # 确保归一化
        p_fused = (
            alpha[0] * p_mfp_low +
            alpha[1] * p_dfp_low
        )
        entropy = -torch.sum(alpha * torch.log(alpha + 1e-6))
        return p_fused, self.reg_lambda * entropy

# DPSL 模型
class DPSL(nn.Module):
    def __init__(self):
        super(DPSL, self).__init__()
        self.mfp_low = MFP(input_dim=11)
        self.dfp_low = DFP()
        self.fusion = AttentionFusion(reg_lambda=0.001)
        self.final_fc = nn.Linear(10, 10)

    def forward(self, low_feat, low_img):
        p_mfp_low = self.mfp_low(low_feat)
        p_dfp_low = self.dfp_low(low_img)
        p_fused, entropy_loss = self.fusion(p_mfp_low, p_dfp_low)
        logits = self.final_fc(p_fused)
        return logits, p_mfp_low, p_dfp_low, entropy_loss

# 训练函数
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
        for low_feats, low_imgs, labels in train_loader:
            low_feats, low_imgs = low_feats.to(device), low_imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, p_mfp_low, p_dfp_low, entropy_loss = model(
                low_feats, low_imgs
            )
            loss = criterion(logits, labels) + entropy_loss
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
            for low_feats, low_imgs, labels in val_loader:
                low_feats, low_imgs = low_feats.to(device), low_imgs.to(device)
                labels = labels.to(device)
                logits, _, _, _ = model(low_feats, low_imgs)
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
            torch.save(model.state_dict(), best_model_path)  # 保存验证集上表现最好的模型

    # 保存每一折的训练曲线
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


# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    preds, trues = [], []
    attentions = []
    with torch.no_grad():
        for low_feats, low_imgs, labels in test_loader:
            low_feats, low_imgs = low_feats.to(device), low_imgs.to(device)
            labels = labels.to(device)
            logits, _, _, _ = model(low_feats, low_imgs)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
            # 无需计算attention，直接填充空的占位符
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
    # 数据路径
    base_path = '/media/zyj/zuoshun/DroneRF/四路特征'
    low_csv_files = [os.path.join(base_path, f'low_{i}.csv') for i in range(10)]
    low_img_dirs = [os.path.join(base_path, f'low_img_{i}') for i in range(10)]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_dataset = DroneDataset(low_csv_files, low_img_dirs, transform=transform)
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

        model = DPSL().to(device)
        train_model(model, fold_train_loader, fold_val_loader, device, num_epochs=50, fold=fold+1)

        # 加载验证集上表现最好的模型
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



