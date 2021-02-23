import base64
import numpy as np
import io
import os 
import cv2
import time
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask import jsonify
from flask import Flask
import json as json
from flask_cors import CORS,cross_origin
import mtcnn
from mtcnn.mtcnn import MTCNN
import shutil
import matplotlib.pyplot as plt


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def get_model():
    global model1, model2
    model1 = load_model('C://Users//alexk//Desktop//Modelos_MaskNet//modelo_1.h5')
    print(" * Model 1  loaded!")
    model2 = load_model('C://Users//alexk//Desktop//Modelos_MaskNet//modelo_2_IMKLD.h5')
    print(" * Model 2  loaded!")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255
    return image

def detection(image):
    print("-------Pasando por detector--------")
    image = img_to_array(image)
    detector = MTCNN()
    face_list = detector.detect_faces(image)
    return face_list

def crop_image(image, faces):
    for face in faces:
        x, y, width, height = face['box']
        img2 = image.crop((x, y, x + width + (width * 0.1), y + height + (height * 0.1)))
                    
        return img2 

print(" * Loading Keras model...")
get_model()

@app.route("/MaskNet", methods=["POST"])
@cross_origin()

def predict():
    tic = time.time()
    answer = ""
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    faces = detection(image)
    cropped_image = crop_image(image, faces)
    try:
        processed_image = preprocess_image(cropped_image, target_size=(224, 224))
    except:
        print("No face detected")
        answer = "No se ha detectado una cara. Introduzca una mejor foto"

        response1 = {
            'prediction': {
                'mask': 0,
                'mask_incorrect': 0,
                'no_mask': 0,
            },

            'suggestion': {
                'answer': answer
            }
        }
        return response1
  
    prediction = model1.predict(processed_image).tolist()
    suggestion = model2.predict(processed_image).tolist()
    toc = time.time()
    hours, rem = divmod(toc-tic,3600)
    minutes, seconds = divmod(rem, 60)
    print("---> TIME: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    if ((prediction[0][0] > prediction[0][1]) and (prediction[0][0] > prediction[0][2])):
         answer = "Parece que estas usando la mascarilla de manera adecuada. Felicitaciones!"
    else:
        answer = "Parece que que no tienes mascarilla. Por favor ponte tu mascarilla."

    response1 = {
        'prediction': {
            'mask': prediction[0][0]*100,
            'mask_incorrect': prediction[0][1]*100,
            'no_mask': prediction[0][2]*100
        },

        'suggestion': {
            'answer': answer
        }
    }

    if ((prediction[0][1] > prediction[0][0]) and (prediction[0][1] > prediction[0][2])):

        if ((suggestion[0][0] > suggestion[0][1]) and (suggestion[0][0] > suggestion[0][2])):
            answer = "Parece que tu mascarilla esta sobre tu barbilla. Asegurate de cubrir tu nariz y boca."
        elif ((suggestion[0][1] > suggestion[0][0]) and (suggestion[0][1] > suggestion[0][2])):
            answer = "Parece que tienes tu nariz por fuera. Asegurate de cubrirla adecuadamente."
        else:
            answer = "Parece que tienes tu barbilla por fuera. Asegurate de cubrirla adecuadamente."

        response1 = {
            'prediction': {
                'mask': prediction[0][0]*100,
                'mask_incorrect': prediction[0][1]*100,
                'no_mask': prediction[0][2]*100
            },
            
            'suggestion': {
                'answer': answer,
            }
        }

    return jsonify(response1)

