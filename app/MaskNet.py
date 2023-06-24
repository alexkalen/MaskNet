# Importing the required libraries
import base64
import numpy as np
import io
import time
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import request, jsonify, Flask
from flask_cors import CORS,cross_origin
from mtcnn.mtcnn import MTCNN


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Defining a function to load the models
def get_model():
    """Load the models for mask status and mask position prediction.

    Returns:
        model1: A Keras model for predicting mask status (mask, mask_incorrect, no_mask).
        model2: A Keras model for predicting mask position (chin, nose, mouth).
    """ 
    global model1, model2 # It is not recommended to use global variables, you can return the models instead and assign them to variables outside the function
    model1 = load_model('/Users/albertolandi/Documents/GitHub/MaskNet/app/models/modelo_1.h5')
    print(" * Model 1  loaded!")
    model2 = load_model('/Users/albertolandi/Documents/GitHub/MaskNet/app/models/modelo_2_IMKLD.h5')
    print(" * Model 2  loaded!")

# Defining a function to preprocess the image for prediction
def preprocess_image(image, target_size):
    """Preprocess the image for prediction.

    Args:
        image: A PIL Image object of the cropped face.
        target_size: A tuple of (width, height) for resizing the image.

    Returns:
        image: A numpy array of shape (1, width, height, 3) with normalized pixel values in [0, 1] range.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255 # Normalizing the pixel values to [0, 1] range
    return image

# Defining a function to detect faces in the image using MTCNN
def detection(image):
    """Detect faces in the image using MTCNN.

    Args:
        image: A PIL Image object of the original image.

    Returns:
        face_list: A list of dictionaries containing information about each detected face, such as bounding box coordinates and confidence score.
    """
    print("-------Pasando por detector--------")
    image = img_to_array(image)
    detector = MTCNN()
    face_list = detector.detect_faces(image)
    return face_list

# Defining a function to crop the image around the detected face
def crop_image(image, faces):
    """Crop the image around the detected face.

    Args:
        image: A PIL Image object of the original image.
        faces: A list of dictionaries containing information about each detected face.

    Returns:
        img2: A PIL Image object of the cropped face with some padding around it.
    """
    for face in faces: # This will only crop the first face in the list, you may want to loop over all faces or handle the case when there are multiple faces detected
        x, y, width, height = face['box']
        img2 = image.crop((x, y, x + width + (width * 0.1), y + height + (height * 0.1))) # Adding some padding around the face
                    
        return img2 # This will exit the function after returning the first cropped image

print(" * Loading Keras model...")
get_model() # You can assign the models to variables here and use them later without global variables

# Defining a route for MaskNet prediction using Flask
@app.route("/MaskNet", methods=["POST"])
@cross_origin()

def predict():
    """Predict the mask status and position using Flask and return a JSON response.

    Returns:
        response1: A JSON response containing the prediction probabilities and suggestion for the user.
    
    Raises:
        ValueError: If no face is detected in the image.
    """
    tic = time.time() # Starting a timer to measure the prediction time
    answer = "" # Initializing an empty string for the answer
    message = request.get_json(force=True) # Getting the JSON message from the request
    encoded = message['image'] # Getting the encoded image from the message
    decoded = base64.b64decode(encoded) # Decoding the image from base64 format
    image = Image.open(io.BytesIO(decoded)) # Opening the image using PIL
    faces = detection(image) # Detecting faces in the image using MTCNN
    cropped_image = crop_image(image, faces) # Cropping the image around the detected face
    try:
        processed_image = preprocess_image(cropped_image, target_size=(224, 224)) # Preprocessing the image for prediction
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
        return response1 # Returning a JSON response with zero probabilities and a suggestion to enter a better photo
  
    prediction = model1.predict(processed_image).tolist() # Predicting the mask status using model1
    suggestion = model2.predict(processed_image).tolist() # Predicting the mask position using model2
    toc = time.time() # Stopping the timer
    hours, rem = divmod(toc-tic,3600)
    minutes, seconds = divmod(rem, 60)
    print("---> TIME: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) # Printing the prediction time

    ###### GPT SUGGESTION
    # print(f"---> TIME: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}") # Printing the prediction time

    # labels = {0: 'mask', 1: 'mask_incorrect', 2: 'no_mask'}
    # suggestions = {0: 'chin', 1: 'nose', 2: 'mouth'}
    # messages = {
    #     ('mask', None): "Parece que estas usando la mascarilla de manera adecuada. Felicitaciones!",
    #     ('mask_incorrect', 'chin'): "Parece que tu mascarilla esta sobre tu barbilla. Asegurate de cubrir tu nariz y boca.",
    #     ('mask_incorrect', 'nose'): "Parece que tienes tu nariz por fuera. Asegurate de cubrirla adecuadamente.",
    #     ('mask_incorrect', 'mouth'): "Parece que tienes tu barbilla por fuera. Asegurate de cubrirla adecuadamente.",
    #     ('no_mask', None): "Parece que que no tienes mascarilla. Por favor ponte tu mascarilla."
    # }

    # prediction_label = labels[np.argmax(prediction[0])] # Getting the label with the highest probability for mask status
    # suggestion_label = suggestions[np.argmax(suggestion[0])] # Getting the label with the highest probability for mask position
    # answer = messages[(prediction_label, suggestion_label)] # Getting the corresponding message from the dictionary

    # response1 = {
    #     'prediction': {
    #         label: prob*100 for label, prob in zip(labels.values(), prediction[0]) # Creating a dictionary of labels and probabilities using a list comprehension
    #     },

    #     'suggestion': {
    #         'answer': answer
    #     }
    # }
    #######



    if ((prediction[0][0] > prediction[0][1]) and (prediction[0][0] > prediction[0][2])): # Checking if mask is predicted with highest probability
         answer = "Parece que estas usando la mascarilla de manera adecuada. Felicitaciones!" # Congratulating the user for wearing the mask correctly
    else: # Otherwise
        answer = "Parece que que no tienes mascarilla. Por favor ponte tu mascarilla." # Asking the user to wear a mask

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

    if ((prediction[0][1] > prediction[0][0]) and (prediction[0][1] > prediction[0][2])): # Checking if mask_incorrect is predicted with highest probability

        if ((suggestion[0][0] > suggestion[0][1]) and (suggestion[0][0] > suggestion[0][2])): # Checking if chin is predicted with highest probability
            answer = "Parece que tu mascarilla esta sobre tu barbilla. Asegurate de cubrir tu nariz y boca." # Asking the user to cover their nose and mouth
        elif ((suggestion[0][1] > suggestion[0][0]) and (suggestion[0][1] > suggestion[0][2])): # Checking if nose is predicted with highest probability
            answer = "Parece que tienes tu nariz por fuera. Asegurate de cubrirla adecuadamente." # Asking the user to cover their nose properly
        else: # Otherwise
            answer = "Parece que tienes tu barbilla por fuera. Asegurate de cubrirla adecuadamente." # Asking the user to cover their chin properly

        response1['suggestion']['answer'] = answer # Updating the answer in the response

    return jsonify(response1) # Returning a JSON response with probabilities and suggestion