# -*- coding: utf-8 -*-
"""
DeepEmbryo - Grad-CAM Modulu
===============================
PDF Isteri 4.3 & 5.1: Gradient-weighted Class Activation Mapping.
Duzlestirilmis InceptionV3 modeli icin optimize edilmistir.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    CLASS_NAMES, GRADCAM_DIR, GRADCAM_LAST_CONV_LAYER,
    IMG_HEIGHT, IMG_WIDTH, LOW_CONFIDENCE_THRESHOLD
)


def make_gradcam_heatmap(model, img_array, last_conv_layer_name=GRADCAM_LAST_CONV_LAYER, pred_index=None):
    """
    Grad-CAM isi haritasi uretir.
    Duzlestirilmis model yapisi icin (model.get_layer() ile dogrudan erisim).
    """
    img_tensor = tf.cast(img_array, tf.float32)
    
    # Duzlestirilmis modelde katmana dogrudan erisim
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        print(f"  [UYARI] '{last_conv_layer_name}' katmani bulunamadi!")
        return np.ones((8, 8)) * 0.5
    
    # Gradient model: conv output + final output
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        print("  [UYARI] Gradyanlar None!")
        return np.ones((8, 8)) * 0.5
    
    # Global Average Pooling -> agirliklar
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Feature map * agirliklar
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU + normalizasyon
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """Orijinal goruntu uzerine Grad-CAM isi haritasini bindirir."""
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (original_img.shape[1], original_img.shape[0]), Image.LANCZOS
        )
    )
    
    jet = mpl_cm.get_cmap("jet")
    jet_heatmap = jet(heatmap_resized / 255.0)[:, :, :3]
    jet_heatmap = (jet_heatmap * 255).astype(np.uint8)
    
    if original_img.max() > 1:
        orig = original_img.astype(np.uint8)
    else:
        orig = (original_img * 255).astype(np.uint8)
    
    superimposed = (jet_heatmap * alpha + orig * (1 - alpha)).astype(np.uint8)
    return superimposed


def _clear_old_gradcam_outputs():
    """Onceki calismalardan kalan Grad-CAM PNG'lerini temizler."""
    for filename in os.listdir(GRADCAM_DIR):
        if filename.lower().endswith(".png") and filename.startswith("gradcam_"):
            try:
                os.remove(os.path.join(GRADCAM_DIR, filename))
            except OSError as e:
                print(f"  [UYARI] Eski Grad-CAM silinemedi ({filename}): {e}")


def generate_gradcam_for_samples(model, X_samples, y_true, y_pred, predictions,
                                  file_paths=None, max_samples=20, clear_existing=True):
    """PDF Isteri 5.1: Her embriyo icin Grad-CAM ciktisi."""
    print("\n" + "=" * 60)
    print("  GRAD-CAM ISI HARITALARI URETILIYOR")
    print("=" * 60)

    if clear_existing:
        _clear_old_gradcam_outputs()
    
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    
    n_samples = min(len(X_samples), max_samples)
    success_count = 0
    heatmaps = []
    
    for i in range(n_samples):
        try:
            img_processed = preprocess_input(X_samples[i:i+1].copy())
            heatmap = make_gradcam_heatmap(model, img_processed)
            heatmaps.append(heatmap)
            overlay = overlay_gradcam(X_samples[i], heatmap)
            
            true_cls = CLASS_NAMES[y_true[i]] if y_true[i] < len(CLASS_NAMES) else f"Sinif_{y_true[i]}"
            pred_cls = CLASS_NAMES[y_pred[i]] if y_pred[i] < len(CLASS_NAMES) else f"Sinif_{y_pred[i]}"
            confidence = float(predictions[i][y_pred[i]])
            is_correct = y_true[i] == y_pred[i]
            is_low_conf = confidence < LOW_CONFIDENCE_THRESHOLD
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            orig_display = X_samples[i].astype(np.uint8) if X_samples[i].max() > 1 else X_samples[i]
            axes[0].imshow(orig_display)
            axes[0].set_title("Orijinal Goruntu", fontweight="bold")
            axes[0].axis("off")
            
            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM Isi Haritasi", fontweight="bold")
            axes[1].axis("off")
            
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (Birlestirilmis)", fontweight="bold")
            axes[2].axis("off")
            
            status = "DOGRU" if is_correct else "YANLIS"
            conf_warning = " - DUSUK GUVENILIRLIK" if is_low_conf else ""
            title = f"Gercek: {true_cls} | Tahmin: {pred_cls} | Guven: {confidence:.2%} | {status}{conf_warning}"
            fig.suptitle(title, fontsize=13, fontweight="bold",
                         color="green" if is_correct else "red")
            
            if is_low_conf:
                fig.text(0.5, 0.02,
                         "UYARI: Bu tahmin dusuk guvenilirliklidir, lutfen manuel kontrol yapiniz.",
                         ha="center", fontsize=11, color="orange", fontweight="bold")
            
            plt.tight_layout()
            save_name = f"gradcam_{i:03d}_{true_cls}_pred_{pred_cls}.png"
            save_path = os.path.join(GRADCAM_DIR, save_name)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            success_count += 1
            
        except Exception as e:
            print(f"  [UYARI] Grad-CAM hatasi (ornek {i}): {e}")
            plt.close("all")
            continue
    
    print(f"  {success_count}/{n_samples} Grad-CAM goruntusi basariyla olusturuldu -> {GRADCAM_DIR}")
    return heatmaps
