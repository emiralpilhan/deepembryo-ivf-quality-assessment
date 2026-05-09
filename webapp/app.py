# -*- coding: utf-8 -*-
"""
DeepEmbryo - Flask Web Uygulaması
===================================
PDF İsteri 6: Web tabanlı arayüz.
- Tekli/toplu görsel yükleme
- Grad-CAM overlay gösterimi
- Düşük güvenilirlik uyarıları
- CSV raporlama
- Veritabanı entegrasyonu
"""

import os
import sys
import numpy as np
from PIL import Image
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flask import (
    Flask, render_template, request, jsonify,
    send_file, send_from_directory, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

from config.config import (
    CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, UPLOAD_FOLDER,
    LOW_CONFIDENCE_THRESHOLD, MODEL_DIR, FLASK_HOST, FLASK_PORT,
    RANDOM_SEED, WEB_TTA_AUGMENTS
)
from webapp.database import (
    save_prediction,
    get_all_predictions,
    export_to_csv,
    update_actual_class,
)
from xai.gradcam import make_gradcam_heatmap, overlay_gradcam

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "deepembryo_secret_key_2026"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global model değişkeni
model = None


def load_model_once():
    """Model'i bir kez yükler."""
    global model
    if model is None:
        model_path = os.path.join(MODEL_DIR, "deepembryo_final.h5")
        if os.path.exists(model_path):
            loaded_model = tf.keras.models.load_model(model_path)
            print(f"  ✅ Model yüklendi: {model_path}")
        else:
            loaded_model = None
            # Stage2 veya Stage1 modelini dene
            for alt in ["stage2_best.h5", "stage1_best.h5"]:
                alt_path = os.path.join(MODEL_DIR, alt)
                if os.path.exists(alt_path):
                    loaded_model = tf.keras.models.load_model(alt_path)
                    print(f"  ✅ Model yüklendi: {alt_path}")
                    break

            if loaded_model is None:
                print("  ❌ Eğitilmiş model bulunamadı!")

        if loaded_model is not None:
            output_dim = int(loaded_model.output_shape[-1])
            expected_dim = len(CLASS_NAMES)
            if output_dim != expected_dim:
                print(
                    "  ❌ Model sınıf sayısı uyumsuz: "
                    f"model={output_dim}, config={expected_dim}. "
                    "Dosya adı bazlı yeni etiketleme için modeli yeniden eğitin."
                )
                model = None
            else:
                model = loaded_model
    return model


def predict_with_tta(mdl, img_array, n_augments=WEB_TTA_AUGMENTS):
    """Tek goruntu icin deterministik TTA ortalamasiyla tahmin yapar."""
    img_batch = img_array[np.newaxis, ...].copy()
    preds_sum = mdl.predict(preprocess_input(img_batch), verbose=0)[0]

    if n_augments <= 1:
        return preds_sum

    tta_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.9, 1.1],
        zoom_range=0.08,
        fill_mode="reflect",
    )

    for aug_i in range(n_augments - 1):
        aug_img = tta_gen.random_transform(
            img_array,
            seed=RANDOM_SEED + (aug_i + 1) * 1000
        )
        aug_batch = np.array(aug_img[np.newaxis, ...], dtype=np.float32)
        preds_sum += mdl.predict(preprocess_input(aug_batch), verbose=0)[0]

    return preds_sum / n_augments


def predict_single_image(img_pil, original_filename=None):
    """
    Tek bir görüntü için tahmin yapar.
    
    Returns:
        predicted_class, confidence, all_predictions, warning_message, gradcam_filename
    """
    mdl = load_model_once()
    if mdl is None:
        return None, 0, None, "Model yüklenemedi veya sınıf sayısı uyumsuz!", None
    
    # Görüntüyü hazırla
    img = img_pil.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_processed = preprocess_input(img_array[np.newaxis, ...].copy())

    # Tahmin (WEB_TTA_AUGMENTS=1 ise raw; >1 ise deterministik TTA)
    preds = predict_with_tta(mdl, img_array)
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    predicted_class = CLASS_NAMES[pred_idx]

    gradcam_filename = None
    try:
        heatmap = make_gradcam_heatmap(mdl, img_processed, pred_index=pred_idx)
        overlay = overlay_gradcam(img_array, heatmap)
        safe_name = secure_filename(original_filename or "image.png") or "image.png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        gradcam_filename = f"gradcam_{timestamp}_{safe_name}.png"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        Image.fromarray(overlay).save(gradcam_path)
    except Exception as e:
        print(f"  [UYARI] Web Grad-CAM üretilemedi: {e}")
    
    # Düşük güvenilirlik uyarısı (PDF isteri 5.2)
    warning = None
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        warning = (
            f"⚠️ DÜŞÜK GÜVENİLİRLİK ({confidence:.2%}): "
            f"Bu tahmin düşük güvenilirliktedir, lütfen manuel kontrol yapınız."
        )
    
    return predicted_class, confidence, preds, warning, gradcam_filename


@app.route("/")
def index():
    """Ana sayfa."""
    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Üretilen Grad-CAM görsellerini servis eder."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """Tekli görsel tahmin endpoint'i."""
    if "file" not in request.files:
        flash("Dosya seçilmedi!", "error")
        return redirect(url_for("index"))
    
    file = request.files["file"]
    if file.filename == "":
        flash("Dosya seçilmedi!", "error")
        return redirect(url_for("index"))
    
    try:
        img = Image.open(file.stream)
        predicted_class, confidence, preds, warning, gradcam_filename = predict_single_image(
            img, original_filename=file.filename
        )
        
        if predicted_class is None:
            flash("Tahmin yapılamadı!", "error")
            return redirect(url_for("index"))
        
        # Veritabanına kaydet
        save_prediction(
            filename=file.filename,
            predicted_class=predicted_class,
            confidence=confidence,
            is_low_confidence=(confidence < LOW_CONFIDENCE_THRESHOLD),
            gradcam_path=gradcam_filename
        )
        
        # Sinif olasiliklarini hazirla
        top5_indices = np.argsort(preds)[::-1][:5]
        top5 = [
            {"class": CLASS_NAMES[i], "probability": f"{preds[i]:.2%}"}
            for i in top5_indices
        ]
        
        return render_template(
            "result.html",
            filename=file.filename,
            predicted_class=predicted_class,
            confidence=f"{confidence:.2%}",
            warning=warning,
            top5=top5,
            actual_class=None,
            is_low_confidence=(confidence < LOW_CONFIDENCE_THRESHOLD),
            gradcam_url=url_for("uploaded_file", filename=gradcam_filename) if gradcam_filename else None
        )
    except Exception as e:
        flash(f"Hata: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/batch", methods=["POST"])
def batch_predict():
    """Toplu tahmin endpoint'i (PDF isteri: batch processing)."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    
    results = []
    for file in files:
        try:
            img = Image.open(file.stream)
            predicted_class, confidence, preds, warning, gradcam_filename = predict_single_image(
                img, original_filename=file.filename
            )
            
            if predicted_class is not None:
                save_prediction(
                    filename=file.filename,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    is_low_confidence=(confidence < LOW_CONFIDENCE_THRESHOLD),
                    gradcam_path=gradcam_filename
                )
                results.append({
                    "filename": file.filename,
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence:.2%}",
                    "warning": warning,
                    "gradcam_url": url_for("uploaded_file", filename=gradcam_filename) if gradcam_filename else None
                })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return render_template("batch_results.html", results=results)


@app.route("/history")
def history():
    """Tahmin geçmişi sayfası."""
    predictions = get_all_predictions()
    return render_template(
        "history.html",
        predictions=predictions,
        class_names=CLASS_NAMES
    )


@app.route("/history/<int:prediction_id>/actual-class", methods=["POST"])
def update_history_actual_class(prediction_id):
    """Tahmin kaydının gerçek sınıf bilgisini sonradan günceller."""
    actual_class = (request.form.get("actual_class") or "").strip() or None
    if actual_class is not None and actual_class not in CLASS_NAMES:
        flash("Geçersiz gerçek sınıf seçimi.", "error")
        return redirect(url_for("history"))

    updated = update_actual_class(prediction_id, actual_class)
    if updated:
        flash("Gerçek sınıf bilgisi güncellendi.", "success")
    else:
        flash("Güncellenecek tahmin kaydı bulunamadı.", "error")
    return redirect(url_for("history"))


@app.route("/export/csv")
def export_csv():
    """CSV olarak dışa aktarım (PDF isteri: CSV formatında dışa aktarma)."""
    output_path = os.path.join(UPLOAD_FOLDER, "predictions_export.csv")
    result = export_to_csv(output_path)
    if result:
        return send_file(result, as_attachment=True,
                         download_name="deepembryo_predictions.csv")
    return "Dışa aktarılacak veri bulunamadı.", 404


def run_webapp():
    """Flask uygulamasını başlatır."""
    load_model_once()
    print(f"\n  🌐 Web uygulaması başlatılıyor: http://localhost:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)


if __name__ == "__main__":
    run_webapp()
