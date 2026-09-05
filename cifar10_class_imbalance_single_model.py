# -*- coding: utf-8 -*-
"""
Single-Model + CIFAR-10 Long-Tailed Class Imbalance
====================================================

实验目的：
1. CIFAR-10 原始训练集 50,000 中划出 10% (= 5,000) 作为 calibration/validation set，
   且 calibration set 每个类别严格 500 张，保持 class-balanced。
2. 剩余 45,000 张训练数据构造 long-tailed class imbalance。
3. 使用 torchvision.models.resnet18 做 CIFAR-10 分类。
4. 用标准的单模型 mini-batch SGD 在这份不平衡训练集上直接训练(不再是联邦学习)。
5. 每个 epoch 后，在独立、class-balanced 的 calibration set 上计算：
   - 每个 class 的 accuracy
   在 train set 上计算:
   - 每个 class 的 accuracy
   - 每个 class 的 cross-entropy loss
   在 test set 上计算:
   - 每个 class 的 accuracy

注：本文件由 fl_cifar10_class_imbalance_fedavg_claude_v2.py 改造而来，
    只去掉了"Dirichlet 划分到多个 client + FedAvg 聚合"这一层，
    数据切分、long-tail 构造、模型结构、评估函数、结果保存的 npz 字段
    全部保持不变，所以 plot_all_classes_claude_v2.py 不需要改动
    (只有一个坐标轴文字标签因为不再是"Federated Round"而做了更新)。
"""

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime  # 用于生成带日期时间的实验日志文件名

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models


# ============================================================
# 1. 参数设置 + print 参数
# ============================================================

@dataclass
class Config:
    # Random seed
    seed: int = 42

    # Dataset
    data_dir: str = "./data"
    num_classes: int = 10
    calibration_ratio: float = 0.10

    # Long-tail class imbalance
    # IF = max_class_samples / min_class_samples
    imbalance_factor: float = 100

    # Training
    # 注: 不再有 FL 的 num_clients / dirichlet_alpha / clients_per_round / local_epochs，
    # 单模型训练里，一个 round 就是完整过一遍整个不平衡训练集(等价于一个 epoch)。
    num_rounds: int = 500
    batch_size: int = 512

    # Optimization
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # DataLoader
    num_workers: int = 0

    # Model
    model_name: str = "resnet18"

    # Output
    output_dir: str = "./results_single_model_cifar10"


CFG = Config()

# ==================== 实验日志(txt)相关设置 ====================
# 输出目录提前到这里创建(原来在文件后面第13节才创建，那时前面一大堆
# print 已经执行完了，来不及把内容写进日志文件，所以挪到最前面来)。
output_dir = Path(CFG.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# 文件名精确到分钟，例如 experiment_log_20260901_1425.txt
_log_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
LOG_PATH = output_dir / f"experiment_log_{_log_timestamp}.txt"


def append_log(msg=""):
    """
    把内容追加写入本次实验的 txt 日志文件。
    只写文件、不额外 print，因为对应内容原本就已经有 print 语句负责打印到控制台了，
    这里只是把同样的内容再存一份到文件里。
    """
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


print(f"本次实验日志将写入: {LOG_PATH.resolve()}")
# ================================================================


def print_config(cfg: Config):
    print("=" * 70)
    print("Experiment Configuration")
    print("=" * 70)
    for key, value in asdict(cfg).items():
        print(f"{key:25s}: {value}")
    print("=" * 70)


print_config(CFG)

# 把和 print_config 一样的内容也写进 txt 日志
append_log("=" * 70)
append_log("Experiment Configuration")
append_log("=" * 70)
for _key, _value in asdict(CFG).items():
    append_log(f"{_key:25s}: {_value}")
append_log("=" * 70)
append_log("")


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了尽可能保证重复实验一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# 3. CIFAR-10 数据集
# ============================================================

# CIFAR-10 常用 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# 为了让 calibration set 与 train set 使用不同 transform，
# 同一份 CIFAR-10 原始训练集加载两次，只取相同 index。
full_train_aug = datasets.CIFAR10(
    root=CFG.data_dir,
    train=True,
    download=True,
    transform=train_transform,
)

full_train_eval = datasets.CIFAR10(
    root=CFG.data_dir,
    train=True,
    download=False,
    transform=eval_transform,
)

test_set = datasets.CIFAR10(
    root=CFG.data_dir,
    train=False,
    download=True,
    transform=eval_transform,
)

class_names = full_train_eval.classes
targets = np.array(full_train_eval.targets)

print("\nCIFAR-10 loaded.")
print(f"Original training samples: {len(full_train_eval)}")
print(f"Original test samples    : {len(test_set)}")
print(f"Classes                  : {class_names}")


# ============================================================
# 4. 划分 calibration set：从 50,000 中划出 10%
#    严格保持每类 500 张
# ============================================================

def stratified_calibration_split(targets, num_classes, calibration_ratio, seed):
    rng = np.random.default_rng(seed)

    train_indices = []
    calibration_indices = []

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)

        n_cal = int(len(class_indices) * calibration_ratio)

        calibration_indices.extend(class_indices[:n_cal].tolist())
        train_indices.extend(class_indices[n_cal:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(calibration_indices)

    return train_indices, calibration_indices


train_base_indices, calibration_indices = stratified_calibration_split(
    targets=targets,
    num_classes=CFG.num_classes,
    calibration_ratio=CFG.calibration_ratio,
    seed=CFG.seed,
)

print("\nDataset split:")
print(f"Training pool     : {len(train_base_indices)}")
print(f"Calibration set   : {len(calibration_indices)}")
print(f"Expected 10% split: {int(50000 * CFG.calibration_ratio)}")

# 训练集/验证集(calibration)/测试集样本数量写入 txt 日志
append_log("\nDataset split:")
append_log(f"Test set (测试集)         : {len(test_set)}")
append_log(f"Calibration set (验证集)  : {len(calibration_indices)}")
append_log(f"Training pool (训练集)    : {len(train_base_indices)}")
append_log(f"Expected 10% split        : {int(50000 * CFG.calibration_ratio)}")


def count_by_class(indices, targets, num_classes):
    result = np.zeros(num_classes, dtype=int)
    for idx in indices:
        result[targets[idx]] += 1
    return result


base_train_class_counts = count_by_class(
    train_base_indices, targets, CFG.num_classes
)
calibration_class_counts = count_by_class(
    calibration_indices, targets, CFG.num_classes
)

print("\nClass counts before long-tail construction:")
for c in range(CFG.num_classes):
    print(
        f"Class {c:2d} ({class_names[c]:10s}) | "
        f"train pool = {base_train_class_counts[c]:4d} | "
        f"calibration = {calibration_class_counts[c]:4d}"
    )


# ============================================================
# 5. 构造 long-tailed class imbalance
# ============================================================

def build_long_tail_indices(
    base_train_indices,
    targets,
    num_classes,
    imbalance_factor,
    seed,
):
    """
    使用 exponential long-tail：
        N_c = N_max * IF^(-c/(C-1))

    其中：
        IF = N_max / N_min

    为了让 class 0 是 head、class 9 是 tail。
    """
    rng = np.random.default_rng(seed)

    class_to_indices = {}
    for c in range(num_classes):
        indices = np.array(
            [idx for idx in base_train_indices if targets[idx] == c],
            dtype=int,
        )
        rng.shuffle(indices)
        class_to_indices[c] = indices

    max_samples = min(len(class_to_indices[c]) for c in range(num_classes))

    class_counts = []
    for c in range(num_classes):
        fraction = imbalance_factor ** (-c / (num_classes - 1))
        n_c = int(round(max_samples * fraction))
        n_c = max(1, min(n_c, len(class_to_indices[c])))
        class_counts.append(n_c)

    long_tail_indices = []

    for c in range(num_classes):
        selected = class_to_indices[c][:class_counts[c]]
        long_tail_indices.extend(selected.tolist())

    rng.shuffle(long_tail_indices)

    return long_tail_indices, np.array(class_counts, dtype=int)


long_tail_indices, long_tail_counts = build_long_tail_indices(
    base_train_indices=train_base_indices,
    targets=targets,
    num_classes=CFG.num_classes,
    imbalance_factor=CFG.imbalance_factor,
    seed=CFG.seed,
)

print("\nLong-tailed training distribution:")
for c in range(CFG.num_classes):
    print(
        f"Class {c:2d} ({class_names[c]:10s}) | "
        f"{long_tail_counts[c]:4d} samples"
    )

print(
    f"\nActual imbalance factor: "
    f"{long_tail_counts.max() / long_tail_counts.min():.2f}"
)

# long-tail 构造后【训练集】总样本数 + 各类别样本数写入 txt 日志
# (long-tail 只作用于训练数据，测试集从头到尾都是固定的、不受影响)
append_log("\nLong-tailed training distribution:")
append_log(f"Total samples (长尾训练集总数): {int(long_tail_counts.sum())}")
for c in range(CFG.num_classes):
    append_log(
        f"Class {c:2d} ({class_names[c]:10s}) | "
        f"{long_tail_counts[c]:4d} samples"
    )
append_log(
    f"\nActual imbalance factor: "
    f"{long_tail_counts.max() / long_tail_counts.min():.2f}"
)


# ============================================================
# 6. Dataset wrapper
# ============================================================
# 注: 原来这里是"6. 再使用 Dirichlet(alpha) 划分到客户端"，
#     单模型训练不需要划分 client，long_tail_indices 本身就是完整的训练集，
#     直接拿去建 Dataset/DataLoader 即可。

class IndexedDataset(Dataset):
    """
    通过 CIFAR-10 dataset + 全局 index 构造一个 dataset。
    """
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        global_idx = self.indices[local_idx]
        return self.base_dataset[global_idx]


# 训练用的 dataset/loader: 用带数据增强的 full_train_aug，读取的是
# long-tail 构造后的完整训练集 long_tail_indices(不再按 client 切分)。
train_dataset = IndexedDataset(full_train_aug, long_tail_indices)

calibration_dataset = Subset(
    full_train_eval,
    calibration_indices,
)

test_dataset = test_set


def make_loader(dataset, shuffle, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(CFG.num_workers > 0),
    )


train_loader = make_loader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=CFG.batch_size,
)

calibration_loader = make_loader(
    dataset=calibration_dataset,
    shuffle=False,
    batch_size=CFG.batch_size,
)

test_loader = make_loader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=CFG.batch_size,
)

# 用于每个 epoch 评估"训练池"上 class-wise train accuracy/loss 的 loader。
# 内容 = long-tail 构造之后的完整训练集，用 eval_transform（无数据增强）读取，
# 这样才能和 calibration_loader 公平对比。
# 训练集在整个实验过程中不会变，所以只在这里构建一次，不要放进 round 循环里
# 每轮重建 —— 否则等于每轮都重新构造一次 DataLoader，纯属浪费。
train_pool_dataset = IndexedDataset(
    full_train_eval,
    long_tail_indices,
)

train_pool_loader = make_loader(
    dataset=train_pool_dataset,
    shuffle=False,
    batch_size=CFG.batch_size,
)


# ============================================================
# 7. 模型：ResNet-18，但修改 CIFAR-10 输入结构
# ============================================================

def build_model(num_classes=10):
    model = models.resnet18(weights=None)

    # ImageNet ResNet-18 的 stem 对 CIFAR-10 32x32 不太合适：
    # 改成 CIFAR-style 3x3 conv，stride=1，不使用 7x7 + maxpool。
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


model = build_model(CFG.num_classes).to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
# print("\nModel:")
# print(model)
# print(f"Trainable parameters: {num_params:,}")


# ============================================================
# 8. 训练:标准单模型 mini-batch SGD
# ============================================================
# 注: 原来这里是"9. Local training"(每个client各建一个模型、每轮从
#     global weights 重新 load 再本地训练) + "10. FedAvg"(聚合)。
#     单模型训练不需要这两步，模型和 optimizer 只创建一次，
#     之后每个 round 就是在 train_loader 上完整跑一遍(一个 epoch)。

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=CFG.learning_rate,
    momentum=CFG.momentum,
    weight_decay=CFG.weight_decay,
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy


# ============================================================
# 9. Calibration：每个 class 的 accuracy / loss
# ============================================================

@torch.no_grad()
def evaluate_classwise(
    model,
    loader,
    num_classes,
    device,
):
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")

    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    class_loss_sum = np.zeros(num_classes, dtype=np.float64)

    overall_correct = 0
    overall_total = 0
    overall_loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        losses = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        correct = predictions.eq(labels)

        overall_correct += correct.sum().item()
        overall_total += labels.size(0)
        overall_loss_sum += losses.sum().item()

        for c in range(num_classes):
            mask = labels.eq(c)
            n_c = mask.sum().item()

            if n_c > 0:
                class_total[c] += n_c
                class_correct[c] += correct[mask].sum().item()
                class_loss_sum[c] += losses[mask].sum().item()

    class_accuracy = np.divide(
        class_correct,
        class_total,
        out=np.zeros(num_classes, dtype=np.float64),
        where=class_total > 0,
    )

    class_loss = np.divide(
        class_loss_sum,
        class_total,
        out=np.zeros(num_classes, dtype=np.float64),
        where=class_total > 0,
    )

    overall_accuracy = overall_correct / overall_total
    overall_loss = overall_loss_sum / overall_total

    return {
        "class_accuracy": class_accuracy,
        "class_loss": class_loss,
        "overall_accuracy": overall_accuracy,
        "overall_loss": overall_loss,
        "class_total": class_total,
    }


# ============================================================
# 10. 计算 class-wise train accuracy
# ============================================================

@torch.no_grad()
def evaluate_train_classwise(model, loader, num_classes, device):
    """
    计算当前 model 在"实际训练样本"(train_pool_loader)上的
    class-wise accuracy / loss。

    注意：
    - 这里不做 train augmentation，直接用 eval-transform 读取同一批训练 indices，
      避免随机 augmentation 让 generalization gap 不稳定。
    - loader 由外部（round 循环之外）一次性构建好并传入，这里不重复构造
      dataset/DataLoader。
    - 实质上和 evaluate_classwise 是同一个函数，保留这层薄封装只是为了
      调用处 train_metrics = evaluate_train_classwise(...) 语义更清楚。
    """
    return evaluate_classwise(
        model=model,
        loader=loader,
        num_classes=num_classes,
        device=device,
    )


# ============================================================
# 11. 保存实验结果的数据结构
# ============================================================

results = {
    "round": [],

    "calibration_class_accuracy": [],
    "calibration_class_loss": [],

    "train_class_accuracy": [],
    "train_class_loss": [],

    "test_class_accuracy": [],  # 每个 epoch 在测试集上的 class-wise accuracy
    "test_overall_accuracy": [],  # 每个 epoch 在测试集上的整体 accuracy

    "generalization_gap": [],

    "calibration_overall_accuracy": [],
    "calibration_overall_loss": [],
}

# 输出目录 (已经提前到文件靠前的位置创建过了，这里不重复创建)
# output_dir = Path(CFG.output_dir)
# output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 12. Centralized Training (单模型训练)
# ============================================================
# 注: 原来这里是"14. Federated Training"，每轮要:
#     选 client -> 各 client 本地训练(train_one_client) -> FedAvg 聚合。
#     单模型训练直接把这三步换成:在 train_loader 上跑一个 epoch(train_one_epoch)，
#     后面 calibration / train_pool / test 的评估和结果保存完全不变。

print("\n" + "=" * 70)
print("Start Centralized Training")
print("=" * 70)

for round_id in range(1, CFG.num_rounds + 1):
    print(f"\n[Round {round_id}/{CFG.num_rounds}]")

    train_epoch_loss, train_epoch_acc = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
    )

    print(
        f"  Epoch train: "
        f"loss={train_epoch_loss:.4f}, "
        f"acc={train_epoch_acc:.4f}"
    )

    # --------------------------------------------------------
    # 每个 epoch 结束后，进行 class-wise calibration evaluation
    # --------------------------------------------------------
    calibration_metrics = evaluate_classwise(
        model=model,
        loader=calibration_loader,
        num_classes=CFG.num_classes,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # 在训练集(train_pool_loader)上计算 class-wise train accuracy
    # --------------------------------------------------------
    train_metrics = evaluate_train_classwise(
        model=model,
        loader=train_pool_loader,
        num_classes=CFG.num_classes,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # 在测试集(test_loader)上评估 class-wise accuracy
    # test_loader 前面已经建好了，这里直接复用，不用重新构造
    # --------------------------------------------------------
    test_metrics = evaluate_classwise(
        model=model,
        loader=test_loader,
        num_classes=CFG.num_classes,
        device=DEVICE,
    )

    generalization_gap = (
        train_metrics["class_accuracy"]
        - calibration_metrics["class_accuracy"]
    )

    # 保存
    results["round"].append(round_id)
    results["calibration_class_accuracy"].append(
        calibration_metrics["class_accuracy"].copy()
    )
    results["calibration_class_loss"].append(
        calibration_metrics["class_loss"].copy()
    )
    results["train_class_accuracy"].append(
        train_metrics["class_accuracy"].copy()
    )
    results["train_class_loss"].append(
        train_metrics["class_loss"].copy()
    )
    results["generalization_gap"].append(
        generalization_gap.copy()
    )
    results["calibration_overall_accuracy"].append(
        calibration_metrics["overall_accuracy"]
    )
    results["calibration_overall_loss"].append(
        calibration_metrics["overall_loss"]
    )
    # 保存每个 epoch 测试集的 class-wise / overall accuracy
    results["test_class_accuracy"].append(
        test_metrics["class_accuracy"].copy()
    )
    results["test_overall_accuracy"].append(
        test_metrics["overall_accuracy"]
    )

    print(
        f"  Global calibration accuracy: "
        f"{calibration_metrics['overall_accuracy']:.4f}"
    )
    print(
        f"  Global test accuracy       : "
        f"{test_metrics['overall_accuracy']:.4f}"
    )

    print("  Class calibration accuracy:")
    print(
        "    " +
        ", ".join(
            f"{class_names[c]}={calibration_metrics['class_accuracy'][c]:.3f}"
            for c in range(CFG.num_classes)
        )
    )

    print("  Class generalization gap:")
    print(
        "    " +
        ", ".join(
            f"{class_names[c]}={generalization_gap[c]:+.3f}"
            for c in range(CFG.num_classes)
        )
    )


# ============================================================
# 13. 保存结果到 NPZ
# ============================================================

np.savez(
    output_dir / "classwise_metrics.npz",
    rounds=np.array(results["round"]),
    calibration_class_accuracy=np.array(
        results["calibration_class_accuracy"]
    ),
    calibration_class_loss=np.array(
        results["calibration_class_loss"]
    ),
    train_class_accuracy=np.array(
        results["train_class_accuracy"]
    ),
    train_class_loss=np.array(
        results["train_class_loss"]
    ),
    test_class_accuracy=np.array(
        results["test_class_accuracy"]
    ),
    test_overall_accuracy=np.array(
        results["test_overall_accuracy"]
    ),
    generalization_gap=np.array(
        results["generalization_gap"]
    ),
    calibration_overall_accuracy=np.array(
        results["calibration_overall_accuracy"]
    ),
    calibration_overall_loss=np.array(
        results["calibration_overall_loss"]
    ),
    long_tail_class_counts=long_tail_counts,
)

print(
    f"\nMetrics saved to: "
    f"{output_dir / 'classwise_metrics.npz'}"
)

# 收尾信息也写进 txt 日志
append_log(f"\nMetrics saved to: {output_dir / 'classwise_metrics.npz'}")
append_log(f"Experiment log file: {LOG_PATH.resolve()}")
