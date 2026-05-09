# -*- coding: utf-8 -*-
"""
DeepEmbryo - Görselleştirme Modülü
====================================
PDF İsterleri: Accuracy-Loss grafikleri, Learning Curve, Confusion Matrix.
Tüm grafikler outputs/plots/ altına kaydedilir.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan çalışma
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CLASS_NAMES, PLOT_DIR


def set_plot_style():
    """Profesyonel grafik stili ayarla."""
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    sns.set_palette("husl")


def plot_accuracy_loss(history, save_path=None):
    """
    PDF İsteri 4.1: Accuracy-Loss Grafiği.
    Eğitim ve validasyon setleri için ayrı ayrı çizilmiş, epoch sayısına bağlı
    doğruluk ve kayıp grafikleri. Overfitting kontrolü yapılabilmelidir.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history["loss"]) + 1)
    
    # Loss grafiği
    ax1.plot(epochs, history["loss"], "b-o", markersize=3, label="Eğitim Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-s", markersize=3, label="Validasyon Loss", linewidth=2)
    ax1.set_title("Eğitim ve Validasyon Kayıp (Loss) Grafiği", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Kayıp (Loss)")
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # Overfitting noktasını işaretle
    val_losses = history["val_loss"]
    min_val_idx = np.argmin(val_losses)
    ax1.axvline(x=min_val_idx + 1, color="green", linestyle="--", alpha=0.7,
                label=f"En düşük val_loss (Epoch {min_val_idx + 1})")
    ax1.legend()
    
    # Accuracy grafiği
    ax2.plot(epochs, history["accuracy"], "b-o", markersize=3, label="Eğitim Acc", linewidth=2)
    ax2.plot(epochs, history["val_accuracy"], "r-s", markersize=3, label="Validasyon Acc", linewidth=2)
    ax2.set_title("Eğitim ve Validasyon Doğruluk (Accuracy) Grafiği", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Doğruluk (Accuracy)")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    path = save_path or os.path.join(PLOT_DIR, "accuracy_loss.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  📊 Accuracy-Loss grafiği: {path}")
    return path


def plot_confusion_matrix(cm, present_classes, save_path=None):
    """
    PDF İsteri 4.2: Confusion Matrix (Karışıklık Matrisi).
    Her sınıf için gerçek vs tahmin karşılaştırması.
    """
    set_plot_style()
    
    labels = [CLASS_NAMES[i] for i in present_classes]
    n = len(labels)
    
    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), max(8, n * 0.7)))
    
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar_kws={"label": "Tahmin Sayısı"},
        linewidths=0.5, linecolor="gray"
    )
    ax.set_title("Confusion Matrix (Karışıklık Matrisi)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12)
    ax.set_ylabel("Gerçek Sınıf", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    path = save_path or os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion Matrix: {path}")
    return path


def plot_class_distribution(labels, title="Sınıf Dağılımı", save_path=None):
    """Veri setindeki sınıf dağılımını bar chart olarak gösterir."""
    set_plot_style()
    
    from collections import Counter
    counter = Counter(labels)
    
    names = []
    counts = []
    for i in sorted(counter.keys()):
        names.append(CLASS_NAMES[i] if isinstance(i, int) else str(i))
        counts.append(counter[i])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = sns.color_palette("husl", len(names))
    bars = ax.bar(names, counts, color=colors, edgecolor="black", linewidth=0.5)
    
    # Sayıları barların üstüne yaz
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontweight="bold")
    
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Görüntü Sayısı")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    path = save_path or os.path.join(PLOT_DIR, "class_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  📊 Sınıf dağılımı: {path}")
    return path


def plot_learning_curve(history, save_path=None):
    """
    PDF İsteri 4.1: Öğrenme Eğrisi (Learning Curve).
    Epoch bazlı eğitim/validasyon performansını ve genelleme farkını gösterir.
    Bu grafik, mevcut eğitim sürecinde overfitting kontrolünü netleştirir.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(history["accuracy"]) + 1)
    
    ax.plot(epochs, history["accuracy"], "b-", linewidth=2, label="Eğitim Doğruluğu")
    ax.plot(epochs, history["val_accuracy"], "r-", linewidth=2, label="Validasyon Doğruluğu")
    
    # Fark alanını boyama (overfitting göstergesi)
    train_acc = np.array(history["accuracy"])
    val_acc = np.array(history["val_accuracy"])
    ax.fill_between(epochs, train_acc, val_acc, alpha=0.15, color="red",
                     label="Genelleme Farkı (Gap)")
    
    ax.set_title("Epoch Bazli Ogrenme Egrisi (Overfitting Kontrolu)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dogruluk (Accuracy)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "learning_curve.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  📊 Learning Curve: {path}")
    return path


def plot_per_class_metrics(report_dict, present_classes, save_path=None):
    """
    PDF İsteri: Her sınıf için Precision, Recall, F1-Score bar chart.
    """
    set_plot_style()
    
    labels = [CLASS_NAMES[i] for i in present_classes]
    
    precisions = [report_dict.get(l, {}).get("precision", 0) for l in labels]
    recalls = [report_dict.get(l, {}).get("recall", 0) for l in labels]
    f1s = [report_dict.get(l, {}).get("f1-score", 0) for l in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width, precisions, width, label="Precision", color="#3498db", edgecolor="black", linewidth=0.5)
    ax.bar(x, recalls, width, label="Recall", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, f1s, width, label="F1-Score", color="#e74c3c", edgecolor="black", linewidth=0.5)
    
    ax.set_title("Sınıf Bazında Performans Metrikleri", fontweight="bold")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Skor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "per_class_metrics.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  📊 Per-class metrics: {path}")
    return path


def generate_all_plots(history, cm, report_dict, present_classes, labels_all):
    """Tüm PDF isterlerine uygun grafikleri üretir."""
    print("\n" + "=" * 60)
    print("  GRAFİKLER ÜRETİLİYOR")
    print("=" * 60)
    
    paths = {}
    paths["accuracy_loss"] = plot_accuracy_loss(history)
    paths["confusion_matrix"] = plot_confusion_matrix(cm, present_classes)
    paths["learning_curve"] = plot_learning_curve(history)
    paths["class_distribution"] = plot_class_distribution(labels_all)
    paths["per_class_metrics"] = plot_per_class_metrics(report_dict, present_classes)
    
    print(f"\n  ✅ Toplam {len(paths)} grafik üretildi → {PLOT_DIR}")
    return paths
