from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# ✅ tambahan import untuk base64 fallback (tidak menghapus yg lama)
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = "flood-risk-secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_IMG_FOLDER = os.path.join(BASE_DIR, "static", "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMG_FOLDER, exist_ok=True)

RF_MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.pkl")

# fitur yang dipakai model di project kamu
FEATURES = ["Rainfall_mm", "WaterLevel_m", "SoilMoisture_pct", "Elevation_m"]

def load_rf_model():
    try:
        model = joblib.load(RF_MODEL_PATH)
        print("✅ RF model loaded")
        return model
    except Exception as e:
        print(f"⚠️ Gagal load model: {e}")
        return None

rf_model = load_rf_model()

def standardize_columns(df):
    """biar kalau nama kolom beda sedikit tetap kebaca"""
    colmap = {c.lower(): c for c in df.columns}
    rename = {}

    candidates = {
        "Rainfall_mm": ["rainfall_mm", "rainfall", "curah_hujan", "rain_mm"],
        "WaterLevel_m": ["waterlevel_m", "water_level_m", "waterlevel", "tinggi_air"],
        "SoilMoisture_pct": ["soilmoisture_pct", "soil_moisture_pct", "soilmoisture", "kelembaban_tanah"],
        "Elevation_m": ["elevation_m", "elevation", "ketinggian", "altitude_m"]
    }

    for std, keys in candidates.items():
        for k in keys:
            if k in colmap:
                rename[colmap[k]] = std
                break

    if rename:
        df = df.rename(columns=rename)

    return df

def generate_plot(preds):
    idx = np.arange(len(preds))

    plt.figure(figsize=(9.5, 4.8))
    plt.plot(idx, preds, linewidth=0.9, alpha=0.9)
    plt.scatter(idx, preds, s=6, alpha=0.6)

    plt.title("Grafik Prediksi Risiko Banjir")
    plt.xlabel("Data ke-")
    plt.ylabel("Prediksi (0=Rendah, 1=Sedang, 2=Tinggi)")
    plt.yticks([0, 1, 2], ["Rendah", "Sedang", "Tinggi"])
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"risk_plot_{ts}.png"
    plt.savefig(os.path.join(STATIC_IMG_FOLDER, fname), dpi=160)
    plt.close()
    return fname

# ✅ fungsi baru (fallback) -> bikin plot base64 kalau static gabisa dipakai
def generate_plot_base64(preds):
    idx = np.arange(len(preds))

    plt.figure(figsize=(9.5, 4.8))
    plt.plot(idx, preds, linewidth=0.9, alpha=0.9)
    plt.scatter(idx, preds, s=6, alpha=0.6)

    plt.title("Grafik Prediksi Risiko Banjir")
    plt.xlabel("Data ke-")
    plt.ylabel("Prediksi (0=Rendah, 1=Sedang, 2=Tinggi)")
    plt.yticks([0, 1, 2], ["Rendah", "Sedang", "Tinggi"])
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")

def predict_from_csv(df):
    df = standardize_columns(df)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {missing}")

    # ambil fitur saja, paksa numerik
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if rf_model is None:
        preds = np.zeros(len(X), dtype=int)
    else:
        preds = rf_model.predict(X)

    counts = {
        "low": int((preds == 0).sum()),
        "medium": int((preds == 1).sum()),
        "high": int((preds == 2).sum())
    }

    # ✅ coba simpan ke static dulu, kalau gagal pakai base64
    try:
        plot_file = generate_plot(preds)
        plot_b64 = None
    except Exception:
        plot_file = None
        plot_b64 = generate_plot_base64(preds)

    return plot_file, plot_b64, counts

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("File tidak ditemukan.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("Pilih file CSV dulu.")
        return redirect(url_for("index"))

    if not file.filename.lower().endswith(".csv"):
        flash("Harus file CSV.")
        return redirect(url_for("index"))

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        df = pd.read_csv(save_path)

        # ✅ sekarang balikannya 3 item
        plot_file, plot_b64, counts = predict_from_csv(df)

        plot_url = None
        if plot_file:
            plot_url = url_for("static", filename="images/" + plot_file)

        return render_template(
            "result.html",
            plot_url=plot_url,
            plot_b64=plot_b64,
            counts=counts
        )

    except Exception as e:
        flash(f"Gagal memproses CSV: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
