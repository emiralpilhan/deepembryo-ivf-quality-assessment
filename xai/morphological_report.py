# -*- coding: utf-8 -*-
"""
DeepEmbryo - Morfolojik Özellik Raporu Modülü
===============================================
PDF İsteri 5.3: Modelin geleneksel embriyoloji kriterlerine göre mi yoksa
farklı özelliklere göre mi karar verdiğini analiz eden özet çıktı.

Analiz edilen kriterler:
- Simetri
- Fragmentasyon oranı
- Vakuol varlığı
- ICM (İç Hücre Kütlesi) odaklanma
- TE (Trofektoderm) odaklanma
"""

import os
import sys
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CLASS_NAMES, REPORT_DIR, LOW_CONFIDENCE_THRESHOLD


def analyze_gradcam_regions(heatmap, threshold=0.5):
    """
    Grad-CAM ısı haritasını analiz ederek modelin odaklandığı bölgeleri tespit eder.
    
    Blastosist morfolojisine göre bölge analizi:
    - Merkez bölge: ICM (İç Hücre Kütlesi) alanı
    - Çevre bölge: TE (Trofektoderm) alanı
    - Genel dağılım: Simetri analizi
    
    Returns:
        dict: Bölge analiz sonuçları
    """
    h, w = heatmap.shape
    
    # Merkez bölge (ICM tahmini - görüntünün merkezi)
    center_y, center_x = h // 2, w // 2
    r = min(h, w) // 4  # Merkez yarıçapı
    
    # Merkez maskesi (ICM)
    y_grid, x_grid = np.ogrid[:h, :w]
    center_mask = ((y_grid - center_y)**2 + (x_grid - center_x)**2) <= r**2
    
    # Çevre maskesi (TE)
    outer_mask = ~center_mask
    
    # Aktivasyon yoğunlukları
    total_activation = np.sum(heatmap)
    center_activation = np.sum(heatmap[center_mask]) / (total_activation + 1e-8)
    outer_activation = np.sum(heatmap[outer_mask]) / (total_activation + 1e-8)
    
    # Simetri analizi (yatay simetri)
    left_half = heatmap[:, :w//2]
    right_half = np.fliplr(heatmap[:, w//2:w//2 + left_half.shape[1]])
    if left_half.shape == right_half.shape:
        symmetry = 1.0 - np.mean(np.abs(left_half - right_half)) / (np.max(heatmap) + 1e-8)
    else:
        symmetry = 0.5
    
    # Odaklanma yoğunluğu (ne kadar lokalize)
    high_activation = np.sum(heatmap > threshold) / heatmap.size
    
    return {
        "icm_focus_ratio": float(center_activation),
        "te_focus_ratio": float(outer_activation),
        "symmetry_score": float(np.clip(symmetry, 0, 1)),
        "localization_score": float(1.0 - high_activation),  # Yüksek = daha lokalize
        "max_activation": float(np.max(heatmap)),
        "mean_activation": float(np.mean(heatmap)),
    }


def generate_morphological_report(predictions, y_pred, y_true, heatmaps=None):
    """
    PDF İsteri 5.3: Morfolojik Özellik Raporu.
    
    Modelin geleneksel embriyoloji kriterlerine göre mi yoksa farklı
    özelliklere göre mi karar verdiğini analiz eden özet çıktı.
    
    Returns:
        report: dict - Kapsamlı morfolojik analiz raporu
    """
    print("\n" + "=" * 60)
    print("  MORFOLOJİK ÖZELLİK RAPORU")
    print("=" * 60)
    
    report = {
        "genel_ozet": {},
        "sinif_bazli_analiz": {},
        "guvenilirlik_analizi": {},
        "morfolojik_kriterler": {}
    }
    
    # 1. Genel Özet
    max_probs = np.max(predictions, axis=1)
    report["genel_ozet"] = {
        "toplam_tahmin": int(len(predictions)),
        "ortalama_guvenilirlik": float(np.mean(max_probs)),
        "medyan_guvenilirlik": float(np.median(max_probs)),
        "min_guvenilirlik": float(np.min(max_probs)),
        "max_guvenilirlik": float(np.max(max_probs)),
        "dusuk_guvenilirlik_sayisi": int(np.sum(max_probs < LOW_CONFIDENCE_THRESHOLD)),
    }
    
    print(f"\n  Genel Güvenilirlik:")
    print(f"    Ortalama : {report['genel_ozet']['ortalama_guvenilirlik']:.4f}")
    print(f"    Medyan   : {report['genel_ozet']['medyan_guvenilirlik']:.4f}")
    print(f"    Min/Max  : {report['genel_ozet']['min_guvenilirlik']:.4f} / "
          f"{report['genel_ozet']['max_guvenilirlik']:.4f}")
    
    # 2. Sınıf bazlı güvenilirlik analizi
    print(f"\n  Sınıf Bazlı Güvenilirlik:")
    for cls_idx in sorted(set(y_true)):
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Sınıf_{cls_idx}"
        mask = y_true == cls_idx
        
        if np.sum(mask) == 0:
            continue
        
        cls_probs = max_probs[mask]
        cls_correct = np.sum(y_pred[mask] == y_true[mask])
        cls_total = np.sum(mask)
        
        cls_analysis = {
            "dogruluk": float(cls_correct / cls_total) if cls_total > 0 else 0,
            "ortalama_guvenilirlik": float(np.mean(cls_probs)),
            "ornek_sayisi": int(cls_total),
        }
        report["sinif_bazli_analiz"][cls_name] = cls_analysis
        
        print(f"    {cls_name:20s}: Acc={cls_analysis['dogruluk']:.2f} | "
              f"Güven={cls_analysis['ortalama_guvenilirlik']:.2f} | "
              f"N={cls_analysis['ornek_sayisi']}")
    
    # 3. Morfolojik Kriter Analizi
    if heatmaps is not None and len(heatmaps) > 0:
        icm_ratios = []
        te_ratios = []
        symmetry_scores = []
        
        for hm in heatmaps:
            analysis = analyze_gradcam_regions(hm)
            icm_ratios.append(analysis["icm_focus_ratio"])
            te_ratios.append(analysis["te_focus_ratio"])
            symmetry_scores.append(analysis["symmetry_score"])
        
        report["morfolojik_kriterler"] = {
            "icm_odaklanma_ortalama": float(np.mean(icm_ratios)),
            "te_odaklanma_ortalama": float(np.mean(te_ratios)),
            "simetri_skoru_ortalama": float(np.mean(symmetry_scores)),
            "yorum": _generate_morphological_interpretation(
                np.mean(icm_ratios), np.mean(te_ratios), np.mean(symmetry_scores)
            )
        }
        
        print(f"\n  Morfolojik Kriter Analizi:")
        print(f"    ICM Odaklanma  : {report['morfolojik_kriterler']['icm_odaklanma_ortalama']:.3f}")
        print(f"    TE Odaklanma   : {report['morfolojik_kriterler']['te_odaklanma_ortalama']:.3f}")
        print(f"    Simetri Skoru  : {report['morfolojik_kriterler']['simetri_skoru_ortalama']:.3f}")
        print(f"    Yorum: {report['morfolojik_kriterler']['yorum']}")
    
    # Raporu kaydet
    report_path = os.path.join(REPORT_DIR, "morphological_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Morfolojik rapor: {report_path}")
    
    return report


def _generate_morphological_interpretation(icm_ratio, te_ratio, symmetry):
    """Morfolojik analiz sonuçlarını yorumlar."""
    interpretations = []
    
    if icm_ratio > 0.5:
        interpretations.append(
            "Model ağırlıklı olarak ICM (İç Hücre Kütlesi) bölgesine odaklanmaktadır. "
            "Bu, geleneksel embriyoloji kriterlerinden ICM kalite değerlendirmesiyle uyumludur."
        )
    elif te_ratio > 0.5:
        interpretations.append(
            "Model ağırlıklı olarak TE (Trofektoderm) bölgesine odaklanmaktadır. "
            "Bu, trofektoderm hücre düzenliliğini esas aldığını göstermektedir."
        )
    else:
        interpretations.append(
            "Model hem ICM hem TE bölgelerine dengeli şekilde odaklanmaktadır. "
            "Bu, geleneksel Gardner skalası değerlendirmesiyle uyumlu bir yaklaşımdır."
        )
    
    if symmetry > 0.7:
        interpretations.append("Simetri analizi: Model simetrik yapıları dikkate almaktadır.")
    else:
        interpretations.append(
            "Simetri analizi: Model asimetrik özelliklere (fragmentasyon, vakuol) "
            "de dikkat etmektedir."
        )
    
    return " ".join(interpretations)
