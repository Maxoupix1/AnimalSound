from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import torchaudio
from utils.audio_utils import get_mel_spectrogram, predict_class
from werkzeug.utils import secure_filename

# Initialisation de Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Charger le modèle PyTorch
MODEL_PATH = "model/efficientnetb0_20_epoch.pt"
model = torch.load(MODEL_PATH)
model.eval()

# Page d'accueil
@app.route('/')
def upload_file():
    return render_template('upload.html')

# Endpoint pour le téléchargement du fichier et prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Aucun fichier fourni", 400

    file = request.files['file']
    if file.filename == '':
        return "Aucun fichier sélectionné", 400

    if file:
        # Sauvegarder le fichier uploadé
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Générer le Mel spectrogramme et faire la prédiction
        mel_spec = get_mel_spectrogram(filepath)
        prediction = predict_class(model, mel_spec)

        # Afficher le résultat
        return render_template('result.html', prediction=prediction)

# Lancer l'application
if __name__ == '__main__':
    app.run(debug=True)
