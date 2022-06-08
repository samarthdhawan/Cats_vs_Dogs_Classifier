#!/usr/bin/env python
# coding: utf-8

# In[20]:

import os
import PIL
import tensorflow
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Model
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow import keras


# In[3]:


app = Flask(__name__)


# In[13]:


STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'


# In[21]:


model = VGG16(include_top=False, input_shape=(200, 200, 3))
# mark loaded layers as not trainable
model.trainable = False
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# compile model
opt = keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[22]:


model.load_weights("fourth_try_with_vgg16.h5")


def predict_model(image_path,model):
    img=image.load_img(image_path,target_size=(200,200))
    img_arr=image.img_to_array(img)/255
    img_arr=np.expand_dims(img_arr,axis=0)
    pred=model.predict(img_arr)
    #pred = tensorflow.keras.np_utils.probas_to_classes(y_proba)
    pred="Probability of being a dog is " + str(pred[0][0])
    return pred


# In[16]:


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


# In[ ]:


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(f)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict_model(file_path, model)


        return preds
    return None


# In[ ]:


if __name__ == '__main__':
    app.run(debug=True)

