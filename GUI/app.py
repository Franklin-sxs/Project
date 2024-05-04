from flask import Flask, request, redirect, url_for, render_template, jsonify, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'persona'
app.config['STATIC_FOLDER'] = 'static'
model = load_model('DWICNN.h5')
labels = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
          'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
@app.route('/')
def home():

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        img = Image.open(file.stream)
        filepath = os.path.join(app.config['STATIC_FOLDER'], 'images', 'uploaded_image.jpg')
        img.save(filepath, format='JPEG')  # Save image as JPEG
        img = img.resize((299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        predicted_class = labels[np.argmax(predictions)]
        probabilities = {labels[i]: float(predictions[0][i]) for i in range(len(labels))}

        session['results'] = {
            'predicted_class': predicted_class,
            'probabilities': probabilities
        }

        return redirect(url_for('result'))

@app.route('/result')
def result():
    results = session.get('results', {})
    return render_template('result.html', results=results,image_path='images/uploaded_image.jpg')
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)