# -*- coding: utf-8 -*-
"""
plot_all_classes.py
===================
独立绘图脚本：从已保存的 npz 文件读取训练与验证指标，
为每个类别单独绘制并保存一张图片。
每张图片包含 4 条核心曲线：
1. Train Accuracy (训练集准确率) - 左纵轴
2. Calibration / Validation Accuracy (验证集准确率) - 左纵轴
3. Test Accuracy (测试集准确率) - 左纵轴 [新增]
4. Train Loss (训练集 Loss) - 右纵轴

注: 如果读取的 npz 文件是旧版本脚本跑出来的、没有 test_class_accuracy 字段，
    会自动跳过 Test Accuracy 曲线，不会报错。

注2: 本脚本对应的训练脚本已经从联邦学习改成了单模型训练，
     npz 里的字段结构完全没变，这里唯一的改动是把横轴标签从
     "Federated Round" 改成了 "Training Round"，因为不再有联邦学习了。
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot FL Metrics for Each Class Individually")
    parser.add_argument(
        "--npz_path",
        type=str,
        default="./results_single_model_cifar10/classwise_metrics.npz",
        help="Path to the saved classwise_metrics.npz file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_single_model_cifar10/per_class_plots",
        help="Directory to save the per-class plots",
    )
    return parser.parse_args()



args = parse_args()
npz_path = Path(args.npz_path)

if not npz_path.exists():
    raise FileNotFoundError(f"NPZ file not found at: {npz_path.resolve()}")

# 1. 读取 NPZ 数据
data = np.load(npz_path)
rounds = data["rounds"]
train_acc = data["train_class_accuracy"]
cal_acc = data["calibration_class_accuracy"]

# 兼容性检查：判断是否存在 train_class_loss
if "train_class_loss" in data:
    train_loss = data["train_class_loss"]
else:
    print("[Warning] 'train_class_loss' not found in NPZ! Fallback to 'calibration_class_loss'.")
    train_loss = data["calibration_class_loss"]

# 兼容性检查——判断是否存在 test_class_accuracy
# (用旧版本脚本跑出来的npz里没有这个字段，这里做个检查，没有就跳过测试集曲线，不报错)
if "test_class_accuracy" in data:
    test_acc = data["test_class_accuracy"]
    has_test_acc = True
else:
    print("[Warning] 'test_class_accuracy' not found in NPZ! Test accuracy curve will be skipped.")
    has_test_acc = False

long_tail_counts = data["long_tail_class_counts"]
num_classes = len(long_tail_counts)

# CIFAR-10 类别名称
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 创建输出文件夹
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Starting to generate plots for {num_classes} classes...")

# 2. 为每个类别单独绘制一张图
for c in range(num_classes):
    c_name = class_names[c] if c < len(class_names) else f"class_{c}"
    n_samples = long_tail_counts[c]

    fig, ax_acc = plt.subplots(figsize=(10, 6))
    ax_loss = ax_acc.twinx()  # 右侧 Loss 纵坐标轴

    # --- 左轴：Accuracy 曲线 ---
    l1, = ax_acc.plot(
        rounds, train_acc[:, c],
        color="#1f77b4", linestyle="-", linewidth=2,  markersize=4,
        label="Train Acc"
    )
    l2, = ax_acc.plot(
        rounds, cal_acc[:, c],
        color="#2ca02c", linestyle="--", linewidth=2, markersize=4,
        label="Calib Acc"
    )

    # Test Accuracy 曲线,同样画在左轴(和Train/Calib Acc同一个量纲)
    if has_test_acc:
        l4, = ax_acc.plot(
            rounds, test_acc[:, c],
            color="#ff7f0e", linestyle="-.", linewidth=2, markersize=4,
            label="Test Acc"
        )

    # --- 右轴：Train Loss 曲线 ---
    l3, = ax_loss.plot(
        rounds, train_loss[:, c],
        color="#d62728", linestyle="-", linewidth=2, markersize=4,
        label="Train Loss"
    )

    # --- 坐标轴与细节设置 ---
    ax_acc.set_xlabel("Training epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12, color="black")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.tick_params(axis="y", labelcolor="black")

    ax_loss.set_ylabel("Train Cross-Entropy Loss", fontsize=12, color="#d62728")
    ax_loss.tick_params(axis="y", labelcolor="#d62728")

    plt.title(
        f"Class {c}: {c_name.capitalize()} (Train Samples N = {n_samples})",
        fontsize=14, pad=12
    )

    ax_acc.grid(True, linestyle=":", alpha=0.6)

    # 合并图例到右上角
    lines = [l1, l2, l4, l3] if has_test_acc else [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax_acc.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=10, framealpha=0.9)

    fig.tight_layout()

    # 保存图片
    save_path = output_dir / f"class_{c:02d}_{c_name}.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"  [Saved] Class {c:2d} ({c_name}): {save_path.name}")

print("=" * 60)
print(f"[Success] All {num_classes} class plots saved to: {output_dir.resolve()}")
print("=" * 60)
