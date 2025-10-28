from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('Task2_TensorFlow/mnist_cnn_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json['image']
        
        # Decode base64 image
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        digit = np.argmax(prediction[0])
        confidence = float(prediction[0][digit]) * 100
        
        return jsonify({
            'digit': int(digit),
            'confidence': round(confidence, 2),
            'probabilities': [float(p) * 100 for p in prediction[0]]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)