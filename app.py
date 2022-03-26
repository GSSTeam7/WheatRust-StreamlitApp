import pandas as pd
import numpy as np
import streamlit as st
from os import listdir
from os.path import isfile, join
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.utils import img_to_array

showpred = 0
try:
	model_path = 'model.h5'
except: 
	print("Need to train model")
#Load the pre-trained models
model = load_model(model_path)
st.sidebar.title("About")

st.sidebar.info(
    "This application identifies the crop health in the picture.")


st.title('Wheat Rust Identification')
st.write("Upload an image.")

uploaded_file = st.file_uploader("Upload Image")
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((256, 256))
    image_arr = img_to_array(image)
    image_arr= np.reshape(image_arr, [1,256,256,3])
    
    label = model.predict(image_arr)
    prediction = np.argmax(label)
    if prediction == 0:
    	st.write(f"I think this is a **healthy** (confidence={round(label[prediction],2)})")
    else:
        message = f"I think this has **rust** (confidence={round(sum(label[1:]),2)}), "
        if prediction == 1:
            message+= f"most likely a **leaf rust** (confidence={round(label[1],2)})"
        elif prediction == 2:
            message+= f"most likely a **stem rust** (confidence={round(label[2],2)})"
        st.write(message)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
#st.write("")
#image = Image.open("Data/Test/" + imageselect)
#st.image(image, caption="Let's predict the health of this crop!", use_column_width=True)



