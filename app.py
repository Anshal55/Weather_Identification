#!/usr/bin/env python
# coding: utf-8

# In[6]:


#flask
from flask import Flask,redirect,url_for,request,render_template

#keras

from tensorflow import keras
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model

#basics
import numpy as np
import pandas as pd
import os


# In[4]:


app = Flask(__name__)


# In[3]:


#load model
model = load_model('Model.h5')
print("Model loaded Now")


# In[7]:
IMAGE_FOLDER = os.getcwd()+ "/static"


#predict function
def predict_img(path):
    classes = ['Cloudy', 'Dawn', 'Rainy', 'Sunny']
    img = load_img(path,target_size=(150,150))
    img_arr = img_to_array(img)
    img_arr_expnd  = np.expand_dims(img_arr,axis=0)
    img = keras.applications.densenet.preprocess_input(img_arr_expnd)

    pred = model.predict(img)
    result = classes[np.argmax(pred)]

    return result


@app.route('/',methods=["GET","POST"])
def index():
    #main page
    return render_template("index.html",data="Hey")


@app.route('/predict',methods=["POST"])
def predict():
    #get image
    if request.method == "POST":
        img = request.files['img']

        if img:
            img_location = os.path.join(
                IMAGE_FOLDER,
                img.filename
            )
            img.save(img_location)

    image = predict_img(img_location)

    return render_template("index.html",data=image,image_loc = img.filename)


# In[11]:


if __name__ == '__main__':
    app.run(debug=True)
