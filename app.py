from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import uuid

from predict import predict_with_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/docs")
def docs():
    return render_template("docs.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]
    if file.filename == "":
        return "No file selected"

    # Formdan hangi model seçilmiş?
    model_choice = request.form.get("model", "resnet")  # 'resnet' veya 'cnn'

    # Dosyayı kaydet
    ext = file.filename.rsplit(".", 1)[-1].lower()
    new_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, new_name)
    file.save(save_path)

    # İki modelle de tahmin yap
    resnet_pred, resnet_conf, resnet_top3 = predict_with_model("resnet", save_path)
    cnn_pred, cnn_conf, cnn_top3 = predict_with_model("cnn", save_path)

    
    if model_choice == "cnn":
        used_model = "GameCamNet"
        prediction = cnn_pred
        confidence = cnn_conf
        top3 = cnn_top3

        other_model = "ResNet50"
        other_prediction = resnet_pred
        other_confidence = resnet_conf
        other_top3 = resnet_top3
    else:
        used_model = "ResNet50"
        prediction = resnet_pred
        confidence = resnet_conf
        top3 = resnet_top3

        other_model = "GameCamNet"
        other_prediction = cnn_pred
        other_confidence = cnn_conf
        other_top3 = cnn_top3

    return render_template(
        "result.html",
        image_path=save_path,
        prediction=prediction,
        confidence=confidence,
        top3=top3,
        used_model=used_model,
        other_model=other_model,
        other_prediction=other_prediction,
        other_confidence=other_confidence,
        other_top3=other_top3,   # ← KRİTİK
    )



if __name__ == "__main__":
    app.run(debug=True)
