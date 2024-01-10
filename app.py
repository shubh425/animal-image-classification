from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import tensorflow_hub as hub
from helper_functions import load_and_prep_image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models\efficientnet.h5'
class_names = ['acinonyx-jubatus',
 'ailuropoda-melanoleuca',
 'canis-lupus-familiaris',
 'equus-caballus',
 'felis-catus',
 'gallus-gallus-domesticus',
 'panthera-tigris']

# Load your trained model
model = tf.keras.models.load_model(
       (MODEL_PATH),
       custom_objects={'KerasLayer':hub.KerasLayer})
model.make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
'''from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
model.save('models/model_resnet.h5')
print('Model loaded. Check http://127.0.0.1:5000/')'''


def model_predict(img_path, model):
    img = load_and_prep_image(img_path, img_shape=224,scale=False)

    preds = model.predict(tf.expand_dims(img, axis=0))
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        if len(preds[0])>1:
            # Process your result for human
            pred_class = class_names[preds.argmax()]
        else:
            pred_class = class_names[int(tf.round(preds)[0][0])]       # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = pred_class               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)