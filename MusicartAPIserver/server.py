import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify

import librosa
#from librosa.core import convert
import librosa.display

import keras.backend as K

from keras.models import Model, load_model

from keras.preprocessing.image import load_img,img_to_array

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
from pydub import AudioSegment

import pyrebase

app = Flask(__name__)
model = None
SAVED_MODEL_PATH = "model.h5"
config = {
    "apiKey": "AIzaSyCRBmEJcl7TTdCmNmd4dqf_9CYtEReVvjg",
    "authDomain": "musicart-606cb.firebaseapp.com",
    "projectId": "musicart-606cb",
    "storageBucket": "musicart-606cb.appspot.com",
    "messagingSenderId": "612465272524",
    "appId": "1:612465272524:web:1e225655e7adc30e6f4469",
    "measurementId": "G-S135ZNCKJ8",
    "databaseURL":""    
    }

def _load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model("./model.h5")
    
def convert_mp3_to_wav(music_file):
    sound = AudioSegment.from_file(music_file,format = "mp3")
    sound.export("./music_file.wav",format="wav")

def extract_relevant(wav_file,t1,t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*t1:1000*t2]
    wav.export("./extracted.wav",format='wav')
        
def create_melspectrogram(wav_file):
    y,sr = librosa.load(wav_file,duration=3)
    mels = librosa.feature.melspectrogram(y=y,sr=sr)

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
    plt.savefig('./melspectrogram.png')

@app.route("/")
def home():
    return "Musicart API"

@app.route("/predict", methods=["POST"])
def predict():
    
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    
    path_on_cloud = "file.mp3"
    print("Downloading...")
    storage.child(path_on_cloud).download("song.mp3")
    print("Downloaded")
    
    print("Load Model...")
    #load_model
    _load_model()
    print("Model Loaded")
    
    #preprocessing
    print("Converting...")
    convert_mp3_to_wav("./song.mp3")
    print("Extracting...")
    extract_relevant("./music_file.wav",40,50)
    print("Create Mel-Spectrogram...")
    create_melspectrogram("./extracted.wav") 
    image_data = load_img('./melspectrogram.png',color_mode='rgba',target_size=(288,432))
    
    #predict
    print("Input Reshape...")
    image = img_to_array(image_data)
    image = np.reshape(image,(1,288,432,4))
    
    print("Predicting...")
    prediction = model.predict(image/255)
    prediction = prediction.reshape((6,)) 
    class_label = np.argmax(prediction)
    
    class_labels = ['rhythmandblues','country', 'disco' ,'hiphop', 'jazz', 'rock']
    
    result = {"genre" : class_labels[class_label]}
    print(result)
    
    return jsonify(result)

if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 5000))
    #app.run(host='0.0.0.0', port=port)
    
    app.run()
