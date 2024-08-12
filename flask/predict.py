import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from PIL import Image

####page 4

#####---------------------------- for perdict gender or age from my models
def load_custom_model(model_select):
    model_path = model_select
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    sys.path.append(model_path)

    get_custom_objects().update({
        'ConvKernalInitializer': ConvKernalInitializer,
        'Swish': Swish,
        'DropConnect': DropConnect
    })

    model = tf.keras.models.load_model(model_path)
    return model

def process_input(image_path):
    image = load_img(image_path, target_size=(224, 224))
    preprocessed_image = img_to_array(image) / 255.0
    return np.expand_dims(preprocessed_image, axis=0)

def predict_age(model, img_path):
    img = process_input(img_path)
    age_prediction = model.predict(img)
    return int(age_prediction[0][0])


def predict_gender(model_layer2, img_path):
    img = process_input(img_path)
    predictions2 = model_layer2.predict(img)
    gender = 'Male' if predictions2[0][0] == 1 else 'Female'
    return gender

def main(left_image_output_path, model_select):
    model = load_custom_model(model_select)
    model.summary()

    model_name = model_select.split('/')[-1]
    input_model_layer = model.output_names[0]
    age_model_layer = model.output_names[0]
    gender_model_layer = model.output_names[1]

    if model_name == 'Age_estimation.h5':
        model_layer1 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(age_model_layer).output)
        model_layer2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(gender_model_layer).output)
        
        predicted_age = predict_age(model_layer1, left_image_output_path)
        
        predicted_text = predicted_age
        
    elif model_name == 'Gender_Prediction.h5':
        model_layer1 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(gender_model_layer).output)
        model_layer2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(age_model_layer).output)
        
        predicted_gender = predict_gender(model_layer2, left_image_output_path)
        
        predicted_text = predicted_gender
    else:
        model_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer(input_model_layer).output)
        predicted = predict_age(model_layer, left_image_output_path)

        predicted_text = predicted
    
    return predicted_text

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)

    left_image_output_path = sys.argv[1]
    model_select = sys.argv[2]
    print(left_image_output_path)
    print(model_select)
    predicted_text = main(left_image_output_path, model_select)
    print('Predicted_text:', predicted_text)
# static/uploads/images/left_opg.png
# static/my_models