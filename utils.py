import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ['csv']

def predict_with_rf(df, model, out_folder):
    df2 = df.copy()
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in uploaded CSV")

    X = df2[numeric_cols].fillna(0)

    if model is None:
        preds = np.zeros(len(X))
        summary = "Model tidak ditemukan di server."
    else:
        preds = model.predict(X)
        summary = (
            f"Prediksi (jumlah baris): {len(preds)} | "
            f"Rendah: {(preds==0).sum()} | "
            f"Sedang: {(preds==1).sum()} | "
            f"Tinggi: {(preds==2).sum()}"
        )

    counts = np.bincount(preds.astype(int), minlength=3)
    labels = ["Rendah", "Sedang", "Tinggi"]

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(counts)), counts, color=["green", "yellow", "red"])
    plt.xticks(range(len(counts)), labels)
    plt.ylabel("Jumlah Prediksi")
    plt.xlabel("Kategori Risiko")
    plt.title("Grafik Prediksi Risiko Banjir")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    imgname = f"pred_summary_{timestamp}.png"
    outpath = os.path.join(out_folder, imgname)
    plt.savefig(outpath)
    plt.close()

    return imgname, summary
