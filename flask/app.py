import os
import secrets
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import sys
from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects
import numpy as np
from werkzeug.utils import secure_filename
import keras
from PIL import Image
import cv2
import subprocess
import torch
from yolov5 import utils
from keras.preprocessing.image import load_img, img_to_array
import random
import pandas as pd
from pydantic_settings import BaseSettings

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
# LangChain imports
import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic #v 0.1.15
#v0.2
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.indexes import VectorstoreIndexCreator

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.schema import Document


import time
from google.api_core.exceptions import ResourceExhausted
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Configuration
FOLDER_MY_MODELS = 'static/my_models'
UPLOAD_FOLDER_IMAGES = 'static/uploads/images'
UPLOAD_FOLDER_MODELS = 'static/uploads/models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'h5', 'pth'}

app.config['FOLDER_MY_MODELS'] = FOLDER_MY_MODELS
app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES
app.config['UPLOAD_FOLDER_MODELS'] = UPLOAD_FOLDER_MODELS

# Create upload folders if they don't exist
os.makedirs(FOLDER_MY_MODELS, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MODELS, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_models():
    return [f for f in os.listdir(FOLDER_MY_MODELS) if f.endswith(('.h5', '.pth'))]

def load_custom_model(model_path):
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

def process_bg(images_directory):
    background_data = []
    image_paths = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        try:
            image = load_img(image_path, target_size=(224, 224))
            preprocessed_image = img_to_array(image) / 255.0
            background_data.append(preprocessed_image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return np.array(background_data)

# Define the base path for your images
images_base_path = "images_bg"
# Create background data using the process_input function
background_train = process_bg(images_base_path)
# Convert background data to numpy array
background_train_np = np.array(background_train)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Model file handling
    model_file = request.files.get('model_file2')
    filename = None
    input_model = None
    
    if model_file and allowed_file(model_file.filename):
        filename = model_file.filename
        model_file.save(os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename))
        input_model = os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename)
    
    # Get user input from form
    model_select_input = request.form.get('model_select')  # This should be '0' for Age or '1' for Gender
    session['predict_input'] = model_select_input  # Store this in session for use in /evaluationpage
    
    predict_input_page1 = request.form.get('frompredict')
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')
    
    print('Inputs from form:')
    print(f'predict_input_1: {predict_input_page1}')
    print(f'node0_input1: {node0_input}')
    print(f'node1_input1: {node1_input}')
    
    my_models = get_available_models()
    if model_select_input == '0':  # Age estimation model
        model = os.path.join(app.config['FOLDER_MY_MODELS'], my_models[0]) if my_models else None
    elif model_select_input == '1':  # Sex estimation model
        model = os.path.join(app.config['FOLDER_MY_MODELS'], my_models[1]) if len(my_models) > 1 else None
    else:
        model = None
    
    selected_model = input_model if input_model else model
    print(f'Selected model: {selected_model}')
    image_file = request.files.get('image')
    
    if not selected_model:
        return redirect(url_for('index'))

    if selected_model and image_file and allowed_file(image_file.filename):
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
        image_file.save(image_path)
        
        left_image_filename = 'left_' + image_filename
        left_image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], left_image_filename)
        print(f'Left image path: {left_image_path}')
        
        # Save data in session for the next page
        session['left_image_path'] = left_image_path
        session['selected_model'] = selected_model        

        subprocess.run(['python', 'cut_image.py', image_path, left_image_path])
        
        try:
            model = load_custom_model(selected_model)
            model.summary()
        except Exception as e:
            print(f"Error loading model: {e}")
            return jsonify({'error': str(e)}), 500

        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        print(f'Predicted class: {predicted_class}')
        
        background_train_np_path = 'background_train_np.npy'
        np.save(background_train_np_path, background_train_np)
        
        try:
            result = subprocess.run(
                ['python', 'shap_.py', left_image_path, selected_model, background_train_np_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            shap_image_url = url_for('static', filename='uploads/shap2nd/cropped_shap_image_plot.png')
            
            session['shap_image_url'] = shap_image_url
            session['predict_input'] = model_select_input  # Save the model type (Age or Gender) in the session
            
            return redirect(url_for('evaluationpage'))
        except subprocess.CalledProcessError as e:
            print("Subprocess error:", e.stderr.decode())
            return jsonify({'error': e.stderr.decode()}), 500
        except Exception as e:
            print("General error:", str(e))
            return jsonify({'error': str(e)}), 500
#----------------------------------------------------------------------
@app.route('/evaluationpage', methods=['POST', 'GET'])
def evaluationpage():
    # Retrieve session data
    left_image_path = session.get('left_image_path')
    selected_model = session.get('selected_model')
    
    predict_input = session.get('predict_input', None)  # Retrieve 'predict_input' from the session
    node0_input_page = session.get('node0_input', 'N/A')
    node1_input_page = session.get('node1_input', 'N/A')

    print('Evaluation Inputs:')
    print(f'Predict Input: {predict_input}')
    print(f'Node 0 Input: {node0_input_page}')
    print(f'Node 1 Input: {node1_input_page}')

    # Default values in case predict_input is None
    predict_label = 'None'
    negative_label = 'None'
    positive_label = 'None'

    # Determine labels based on the predict_input value
    if predict_input == '0':  # Age Estimation
        predict_label = 'Age'
        negative_label = 'Younger'
        positive_label = 'Older'
    elif predict_input == '1':  # Gender Estimation
        predict_label = 'Gender'
        negative_label = 'Male'
        positive_label = 'Female'

    predicted_text = ''
    try:
        result = subprocess.run(['python', 'predict.py', left_image_path, selected_model], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        predicted_text = result.stdout.decode('utf-8').strip()
        if 'Predicted_text:' in predicted_text:
            predicted_text = predicted_text.split('Predicted_text:')[1].strip()
        print(f'Prediction Result: {predicted_text}')
    except subprocess.CalledProcessError as e:
        print(f'Error in prediction: {e.stderr.decode()}')

    subprocess.run(['python', 'yolo.py', left_image_path, 'nms_csv.csv'], check=True)
    subprocess.run(['python', 'yolo_shap.py', left_image_path, 'nms_csv.csv', 
                    'save_grayscale_pos.npy', 'save_grayscale_neg.npy', '10'], check=True)
    # SHAP output URLs
    iou_image_plot_neg_url = url_for('static', filename='uploads/iou/iou_image_plot_neg.png')
    iou_image_plot_pos_url = url_for('static', filename='uploads/iou/iou_image_plot_pos.png')

    # Synonyms for LangChain processing
    descriptive_neg = pd.read_csv('result_neg.csv')
    descriptive_pos = pd.read_csv('result_pos.csv')

    synonyms = {
        'Condyle': ['Condyle', 'Condylar process', 'Mandibular condyle'],
        'Posterior border of Ramus': ['Posterior border of Ramus', 'Posterior edge of the mandibular ramus'],
        'Mandibular angle': ['Mandibular angle', 'Angle of the mandible', 'Gonial angle'],
        'Nasal': ['Nasal', 'Nasal bone', 'Nasal region'],
        'Lower Central Incisor': ['Lower Central Incisor', 'Mandibular central incisor', 'Lower front'],
        'Lower Lateral Incisor': ['Lower Lateral Incisor', 'Mandibular lateral incisor'],
        'Lower Canine': ['Lower Canine', 'Mandibular canine', 'Lower cuspid'],
        'Lower First Premolar': ['Lower First Premolar', 'Mandibular first premolar', 'Lower first bicuspid'],
        'Lower Second Premolar': ['Lower Second Premolar', 'Mandibular second premolar', 'Lower second bicuspid'],
        'Lower First Molar': ['Lower First Molar', 'Mandibular first molar'],
        'Lower Second Molar': ['Lower Second Molar', 'Mandibular second molar'],
        'Lower Third Molar': ['Lower Third Molar', 'Mandibular third molar', 'Lower wisdom'],
        'Upper Central Incisor': ['Upper Central Incisor', 'Maxillary central incisor', 'Upper front'],
        'Upper Lateral Incisor': ['Upper Lateral Incisor', 'Maxillary lateral incisor'],
        'Upper Canine': ['Upper Canine', 'Maxillary canine', 'Upper cuspid'],
        'Upper First Premolar': ['Upper First Premolar', 'Maxillary first premolar', 'Upper first bicuspid'],
        'Upper Second Premolar': ['Upper Second Premolar', 'Maxillary second premolar', 'Upper second bicuspid'],
        'Upper First Molar': ['Upper First Molar', 'Maxillary first molar'],
        'Upper Second Molar': ['Upper Second Molar', 'Maxillary second molar'],
        'Upper Third Molar': ['Upper Third Molar', 'Maxillary third molar', 'Upper wisdom']
    }
        


    def map_synonyms(term):
        for standard_term, syn_list in synonyms.items():
            if term in syn_list:
                return standard_term
        return term

    descriptive_neg['name'] = descriptive_neg['name'].apply(map_synonyms)
    descriptive_pos['name'] = descriptive_pos['name'].apply(map_synonyms)

    combined_names = sorted(set(descriptive_neg['name']).union(descriptive_pos['name']))

    # LangChain AI processing
    os.environ['GOOGLE_API_KEY'] = ''
    ref_path = 'test_ref'
    raw_documents = DirectoryLoader(ref_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    # Extract the texts and metadata from the documents
 #   texts = [doc.page_content for doc in documents]  # Get document text content
 #   metadatas = [doc.metadata for doc in documents]  # Get document metadata if available

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore_disk = Chroma(persist_directory="./langchain/chroma_db", embedding_function=gemini_embeddings)

    # Add texts and metadata
#    vectorstore_disk.add_texts(texts=texts, metadatas=metadatas)

    print(f"Number of documents in vectorstore: {vectorstore_disk._collection.count()}")
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, top_p=0.85)
    llm_prompt_template = """
    Based on the provided context, answer the question with a concise summary and include properly formatted references. Use the following structure:
    
    ***References***
    [Author(s)], [Year]. [Title of the article]. [Journal name], [Volume(Issue)].
    
    Context: {context}

    Question: {question}

    If author information is unavailable, include only the year, title, journal, and volume/issue. Respond in the same language as the question. 
    """
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    # Generate AI-driven insights
    results = {}
    for name in combined_names:
        query = f"Summarize the relationship between {name} and predictions using the {predict_label} model."
        results[name] = rag_chain.invoke(query)
        
        #print("name - " + str(name))
        #print("result - " + str(results[name]))

    return render_template(
        'evaluationpage.html',
        iou_image_plot_neg_url=iou_image_plot_neg_url,
        iou_image_plot_pos_url=iou_image_plot_pos_url,
        predict1=predict_label,
        predictresult=predicted_text,
        negative_label=negative_label,
        positive_label=positive_label,
        document_query_result=results
    )




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
