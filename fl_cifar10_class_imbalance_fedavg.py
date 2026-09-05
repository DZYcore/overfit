# -*- coding: utf-8 -*-
"""
FL + CIFAR-10 Long-Tailed Class Imbalance + Dirichlet + FedAvg
==============================================================

实验目的：
1. CIFAR-10 原始训练集 50,000 中划出 10% (= 5,000) 作为 calibration/validation set，
   且 calibration set 每个类别严格 500 张，保持 class-balanced。
2. 剩余 45,000 张训练数据构造 long-tailed class imbalance。
3. 再使用 Dirichlet(alpha) 将不平衡训练数据划分到 10 个客户端。
4. 使用 torchvision.models.resnet18 做 CIFAR-10 分类。
5. 使用 FedAvg 进行联邦训练。
6. 每轮 FedAvg 后，在独立、class-balanced 的 calibration set 上计算：
   - 每个 class 的 accuracy
   在 train set 上计算:
   - 每个 class 的 accuracy
   - 每个 class 的 cross-entropy loss

"""

import copy
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime  # 新增: 用于生成带日期时间的实验日志文件名

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

    # Federated learning
    num_clients: int = 10

    dirichlet_alpha: float = 30
    clients_per_round: int = 7
    num_rounds: int = 500
    local_epochs: int = 5
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
    output_dir: str = "./results_fl_cifar10"


CFG = Config()

# ==================== 新增: 实验日志(txt)相关设置 ====================
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

# 新增: 把和 print_config 一样的内容也写进 txt 日志
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

# 新增: 训练集/验证集(calibration)/测试集样本数量写入 txt 日志
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

# 新增: long-tail 构造后【训练集】总样本数 + 各类别样本数写入 txt 日志
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
# 6. 再使用 Dirichlet(alpha) 划分到客户端
# ============================================================

def dirichlet_partition(
    indices,
    targets,
    num_clients,
    num_classes,
    alpha,
    seed,
):
    """
    对每个 class c：
        p ~ Dirichlet(alpha, ..., alpha)

    然后把该 class 的样本按 p 分配给不同 clients。

    注意：
    这里的 long-tail class imbalance 已经存在；
    Dirichlet 只是进一步制造 client-level label heterogeneity。
    """
    rng = np.random.default_rng(seed)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.array(
            [idx for idx in indices if targets[idx] == c],
            dtype=int,
        )
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(
            np.full(num_clients, alpha, dtype=float)
        )

        # 根据比例计算每个 client 应获得多少样本
        raw_counts = proportions * len(class_indices)
        counts = np.floor(raw_counts).astype(int)

        # 把 floor 后剩余样本分给小数部分最大的 clients
        remainder = len(class_indices) - counts.sum()

        if remainder > 0:
            fractional = raw_counts - counts
            order = np.argsort(-fractional)
            counts[order[:remainder]] += 1

        assert counts.sum() == len(class_indices)

        start = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                selected = class_indices[start:start + count]
                client_indices[client_id].extend(selected.tolist())
                start += count

    # 每个 client 内再随机打乱
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


client_indices = dirichlet_partition(
    indices=long_tail_indices,
    targets=targets,
    num_clients=CFG.num_clients,
    num_classes=CFG.num_classes,
    alpha=CFG.dirichlet_alpha,
    seed=CFG.seed,
)

print("\nClient distributions after Dirichlet partition:")
for client_id in range(CFG.num_clients):
    counts = count_by_class(
        client_indices[client_id], targets, CFG.num_classes
    )
    print(
        f"Client {client_id:2d}: "
        f"total={len(client_indices[client_id]):5d} | "
        f"class distribution={counts.tolist()}"
    )

# 新增: Dirichlet 划分后，每个 client 的总样本数 + 各类别样本数写入 txt 日志
append_log("\nClient distributions after Dirichlet partition:")
for client_id in range(CFG.num_clients):
    counts = count_by_class(
        client_indices[client_id], targets, CFG.num_classes
    )
    append_log(
        f"Client {client_id:2d}: "
        f"total={len(client_indices[client_id]):5d} | "
        f"class distribution={counts.tolist()}"
    )


# ============================================================
# 7. Dataset wrapper
# ============================================================

class IndexedDataset(Dataset):
    """
    通过 CIFAR-10 dataset + 全局 index 构造一个 client dataset。
    """
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        global_idx = self.indices[local_idx]
        return self.base_dataset[global_idx]


client_datasets = [
    IndexedDataset(full_train_aug, indices)
    for indices in client_indices
]

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


client_loaders = [
    make_loader(
        dataset=dataset,
        shuffle=True,
        batch_size=CFG.batch_size,
    )
    for dataset in client_datasets
]

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

# 用于每轮评估"训练池"上 class-wise train accuracy/loss 的 loader。
# 内容 = 所有 client 实际用于训练的样本(long-tail + Dirichlet 划分之后)，
# 用 eval_transform（无数据增强）读取，这样才能和 calibration_loader 公平对比。
# 划分结果在整个实验过程中不会变，所以只在这里构建一次，
# 不要放进 round 循环里每轮重建 —— 否则等于每轮都重新构造一次 DataLoader
# (num_workers>0 时还会重新拉起 worker 进程)，纯属浪费。
train_pool_dataset = IndexedDataset(
    full_train_eval,
    list(np.concatenate(client_indices)),
)

train_pool_loader = make_loader(
    dataset=train_pool_dataset,
    shuffle=False,
    batch_size=CFG.batch_size,
)


# ============================================================
# 8. 模型：ResNet-18，但修改 CIFAR-10 输入结构
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


global_model = build_model(CFG.num_classes).to(DEVICE)

num_params = sum(p.numel() for p in global_model.parameters())
# print("\nModel:")
# print(global_model)
# print(f"Trainable parameters: {num_params:,}")


# ============================================================
# 9. Local training
# ============================================================

def train_one_client(
    global_state_dict,
    loader,
    local_epochs,
    device,
):
    model = build_model(CFG.num_classes).to(device)
    model.load_state_dict(global_state_dict)

    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CFG.learning_rate,
        momentum=CFG.momentum,
        weight_decay=CFG.weight_decay,
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(local_epochs):
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

    return (
        model.state_dict(),
        total_samples,
        avg_loss,
        accuracy,
    )


# ============================================================
# 10. FedAvg
# ============================================================

def fedavg(client_state_dicts, client_sample_counts):
    """
    按 client 的训练样本数量进行 weighted FedAvg。
    """
    assert len(client_state_dicts) == len(client_sample_counts)

    total_samples = sum(client_sample_counts)

    new_state_dict = {}

    for key in client_state_dicts[0].keys():
        first_tensor = client_state_dicts[0][key]

        # BatchNorm 的 num_batches_tracked 是整型 tensor，
        # 不能直接做浮点加权平均；直接取样本最多 client 对应值即可。
        if not torch.is_floating_point(first_tensor):
            max_idx = int(np.argmax(client_sample_counts))
            new_state_dict[key] = client_state_dicts[max_idx][key].clone()
            continue

        aggregated = torch.zeros_like(first_tensor)

        for state_dict, n_samples in zip(
            client_state_dicts,
            client_sample_counts,
        ):
            weight = n_samples / total_samples
            aggregated += state_dict[key] * weight

        new_state_dict[key] = aggregated

    return new_state_dict


# ============================================================
# 11. Calibration：每个 class 的 accuracy / loss
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
# 12. 计算 class-wise train accuracy
# ============================================================

@torch.no_grad()
def evaluate_train_classwise(model, loader, num_classes, device):
    """
    计算当前 global model 在"实际 FL training samples"(train_pool_loader)上的
    class-wise accuracy / loss。

    注意：
    - 这里不做 train augmentation，直接用 eval-transform 读取同一批训练 indices，
      避免随机 augmentation 让 generalization gap 不稳定。
    - loader 由外部（round 循环之外）一次性构建好并传入，这里不再重新构造
      dataset/DataLoader —— 原来的实现每轮都重建一次，另外原来的 loaders 形参
      也从未在函数体内被用到，两者一并清理掉。
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
# 13. 保存实验结果的数据结构
# ============================================================

results = {
    "round": [],

    "calibration_class_accuracy": [],
    "calibration_class_loss": [],

    "train_class_accuracy": [],
    "train_class_loss": [],

    "test_class_accuracy": [],  # 新增: 每轮在测试集上的 class-wise accuracy
    "test_overall_accuracy": [],  # 新增: 每轮在测试集上的整体 accuracy

    "generalization_gap": [],

    "calibration_overall_accuracy": [],
    "calibration_overall_loss": [],
}

# 输出目录 (已经提前到文件靠前的位置创建过了，这里注释掉，不重复创建)
# output_dir = Path(CFG.output_dir)
# output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 14. Federated Training
# ============================================================

print("\n" + "=" * 70)
print("Start Federated Training")
print("=" * 70)

for round_id in range(1, CFG.num_rounds + 1):
    print(f"\n[Round {round_id}/{CFG.num_rounds}]")

    # 选择参与本轮训练的 clients
    if CFG.clients_per_round >= CFG.num_clients:
        selected_clients = list(range(CFG.num_clients))
    else:
        selected_clients = random.sample(
            range(CFG.num_clients),
            CFG.clients_per_round,
        )

    client_states = []
    client_sample_counts = []

    local_train_losses = []
    local_train_accs = []

    current_global_state = copy.deepcopy(global_model.state_dict())

    for client_id in selected_clients:
        state_dict, n_samples, train_loss, train_acc = train_one_client(
            global_state_dict=current_global_state,
            loader=client_loaders[client_id],
            local_epochs=CFG.local_epochs,
            device=DEVICE,
        )

        client_states.append(state_dict)
        client_sample_counts.append(n_samples)

        local_train_losses.append(train_loss)
        local_train_accs.append(train_acc)

        print(
            f"  Client {client_id:2d}: "
            f"samples={n_samples:5d}, "
            f"local_loss={train_loss:.4f}, "
            f"local_acc={train_acc:.4f}"
        )

    # FedAvg
    aggregated_state = fedavg(
        client_state_dicts=client_states,
        client_sample_counts=client_sample_counts,
    )

    global_model.load_state_dict(aggregated_state)

    # --------------------------------------------------------
    # 每轮 global model 后，进行 class-wise calibration evaluation
    # --------------------------------------------------------
    calibration_metrics = evaluate_classwise(
        model=global_model,
        loader=calibration_loader,
        num_classes=CFG.num_classes,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # 在 FL training samples 上计算 class-wise train accuracy
    # --------------------------------------------------------
    train_metrics = evaluate_train_classwise(
        model=global_model,
        loader=train_pool_loader,
        num_classes=CFG.num_classes,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # 新增: 每轮 global model 后，在测试集(test_loader)上评估 class-wise accuracy
    # test_loader 前面已经建好了，这里直接复用，不用重新构造
    # --------------------------------------------------------
    test_metrics = evaluate_classwise(
        model=global_model,
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
    # 新增: 保存每轮测试集的 class-wise / overall accuracy
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
    # 新增: 打印测试集整体准确率
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
# 15. 保存结果到 NPZ
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

# 新增: 收尾信息也写进 txt 日志
append_log(f"\nMetrics saved to: {output_dir / 'classwise_metrics.npz'}")
append_log(f"Experiment log file: {LOG_PATH.resolve()}")


