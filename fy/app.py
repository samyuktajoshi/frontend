from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_dicom  # Import your preprocessing script
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/model.h5")

# Configurations
UPLOAD_FOLDER = 'static/uploaded'
PROCESSED_FOLDER = 'static/processed'
GRAPH_FOLDER = 'static/graphs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['GRAPH_FOLDER'] = GRAPH_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_and_process():
    # Ensure file exists in the request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded ZIP file
    filename = secure_filename(file.filename)
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(zip_path)

    # Extract ZIP file
    extracted_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted")
    os.makedirs(extracted_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    # Process DICOM files
    predictions = []
    for root, _, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                try:
                    # Preprocess DICOM file
                    image_array = preprocess_dicom(dicom_path)

                    # Predict using the model
                    prediction = model.predict(image_array)[0][0]  # Single output
                    predictions.append(prediction)

                except Exception as e:
                    predictions.append(0)  # If error, assign 0 (no hemorrhage)

    # Plot results
    graph_path = os.path.join(app.config['GRAPH_FOLDER'], "prediction_graph.png")
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, marker='o', color='b', label='Prediction Confidence')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    plt.title("Brain Hemorrhage Detection Results")
    plt.xlabel("File Index")
    plt.ylabel("Confidence Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(graph_path)
    plt.close()

    # Check for hemorrhage based on spikes
    hemorrhage_present = any(pred > 0.5 for pred in predictions)
    result_message = "Hemorrhage Detected!" if hemorrhage_present else "No Hemorrhage Detected."

    return render_template(
        "result.html",
        result_message=result_message,
        graph_path=graph_path,
        predictions=predictions
    )

@app.route("/download_graph", methods=["GET"])
def download_graph():
    graph_path = os.path.join(app.config['GRAPH_FOLDER'], "prediction_graph.png")
    return send_file(graph_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
