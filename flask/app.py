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

import time
from google.api_core.exceptions import ResourceExhausted

app = Flask(__name__, static_folder='static')  # Include static folder
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Configuration (adjust as needed)
FOLDER_MY_MODELS = 'static/my_models'
UPLOAD_FOLDER_IMAGES = 'static/uploads/images'
UPLOAD_FOLDER_MODELS = 'static/uploads/models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'h5', 'pth'}
FOLDER_SHAP2ND = 'static/uploads/shap2nd/'
FOLDER_PERCENTILE = 'static/uploads/percentile/'
FOLDER_IOU = 'static/uploads/percentile/iou'

app.config['FOLDER_MY_MODELS'] = FOLDER_MY_MODELS
app.config['UPLOAD_FOLDER_IMAGES'] = UPLOAD_FOLDER_IMAGES
app.config['UPLOAD_FOLDER_MODELS'] = UPLOAD_FOLDER_MODELS
app.config['FOLDER_SHAP2ND'] = FOLDER_SHAP2ND
app.config['FOLDER_PERCENTILE'] = FOLDER_PERCENTILE
app.config['FOLDER_IOU'] = FOLDER_IOU

# Create upload folders if they don't exist
os.makedirs(FOLDER_MY_MODELS, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_IMAGES, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MODELS, exist_ok=True)
os.makedirs(FOLDER_SHAP2ND, exist_ok=True)
os.makedirs(FOLDER_PERCENTILE, exist_ok=True)
os.makedirs(FOLDER_IOU, exist_ok=True)

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

# @app.route('/predict',methods=['GET', 'POST'])
# def predict():
#     #model
#     model_file = request.files.get('model_file2')
#     filename = None
#     input_model = None
    
#     if model_file and allowed_file(model_file.filename):
#         filename = model_file.filename
#         model_file.save(os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename))
#         input_model = os.path.join(app.config['UPLOAD_FOLDER_MODELS'], filename)
    
#     model_select_input = request.form.get('model_select')
#     predict_input_page1 = request.form.get('frompredict')
#     node0_input = request.form.get('node0input')
#     node1_input = request.form.get('node1input')
#     #print
#     print('printgetvalue')
#     print(f'predict_input_1: {predict_input_page1}')
#     print(f'node0_input1: {node0_input}')
#     print(f'node1_input1: {node1_input}')
    
#     my_models = get_available_models()
#     if model_select_input == '0':  # Age estimation model
#         model = os.path.join(app.config['FOLDER_MY_MODELS'], my_models[0]) if my_models else None
#     elif model_select_input == '1':  # Sex estimation model
#         model = os.path.join(app.config['FOLDER_MY_MODELS'], my_models[1]) if len(my_models) > 1 else None
#     else:
#         model = None
    
#     selected_model = input_model if input_model else model
#     print(f'selected_model: {selected_model}')
#     image_file = request.files.get('image')
    
#     if not selected_model:
#         #flash("No model selected, image saved", "error")
#         return redirect(url_for('index'))

#     if selected_model and image_file and allowed_file(image_file.filename):
#         # Save the uploaded image file
#         image_filename = image_file.filename
#         image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
#         image_file.save(image_path)
       
#         # Define the output filenames
#         left_image_filename = 'left_' + image_filename
#         left_image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], left_image_filename)
#         print(left_image_path)
        
#         session['left_image_path'] = left_image_path
#         session['selected_model'] = selected_model        

#         # call the cut_image.py script as a subprocess
#         subprocess.run(['python', 'cut_image.py', image_path, left_image_path])
        
#         # Load the model
#         model_path = selected_model
        
#         print(f"Selected model: {selected_model}")
#         print(f"Image filename: {image_filename}")
#         print(f"Image path: {image_path}")
#         print(f"Model path: {model_path}")
        
#         try:
#             model = load_custom_model(model_path)
#             model.summary()
#             # You can proceed with predictions here
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return jsonify({'error': str(e)}), 500

#         #Preprocess the image
#         img = image.load_img(image_path, target_size=(224, 224))  # Adjust target size as per your model's requirement
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0  # Normalize the image if required by your model

#         # Get predictions
#         predictions = model.predict(img_array)
#         # Assuming the model's output is a single class prediction, modify as needed
#         predicted_class = np.argmax(predictions[0])
#         output = f"Prediction from {selected_model}: {predicted_class}"
#         print(f'output: {output}')
        
#         background_train_np_path = 'background_train_np.npy'
#         np.save(background_train_np_path, background_train_np)
        
#         try:
#             # call the shap.py script as a subprocess
#             result = subprocess.run(
#                 ['python', 'shap_.py', left_image_path, selected_model, background_train_np_path],
#                 stdout=subprocess.PIPE,  # Capture standard output
#                 stderr=subprocess.PIPE,  # Capture standard error
#                 check=True
#             )
            
#             shap_image_url = url_for('static', filename='uploads/shap2nd/cropped_shap_image_plot.png')
#             session['shap_image_url'] = shap_image_url
#             session['predict_input'] = predict_input_page1
#             session['node0_input'] = node0_input
#             session['node1_input'] = node1_input
            
#             return render_template('shappage.html', 
#                                 # output=output_data['text'], 
#                                 # shap_values_left_opg_2=session['shap_values_left_opg_2'], 
#                                 shap_image_url=shap_image_url, predict_input=predict_input_page1, node0_input=node0_input, node1_input=node1_input)
#         except subprocess.CalledProcessError as e:
#             print("Subprocess error:", e.stderr.decode())  # Print the error message
#             return jsonify({'error': e.stderr.decode()}), 500
#         except Exception as e:
#             print("General error:", str(e))  # Print any other errors
#             return jsonify({'error': str(e)}), 500

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
    
    model_select_input = request.form.get('model_select')
    predict_input_page1 = request.form.get('frompredict')
    node0_input = request.form.get('node0input')
    node1_input = request.form.get('node1input')
    
    print('printgetvalue')
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
    print(f'selected_model: {selected_model}')
    image_file = request.files.get('image')
    
    if not selected_model:
        return redirect(url_for('index'))

    if selected_model and image_file and allowed_file(image_file.filename):
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], image_filename)
        image_file.save(image_path)
        
        left_image_filename = 'left_' + image_filename
        left_image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], left_image_filename)
        print(left_image_path)
        
        session['left_image_path'] = left_image_path
        session['selected_model'] = selected_model        

        subprocess.run(['python', 'cut_image.py', image_path, left_image_path])
        
        model_path = selected_model
        
        print(f"Selected model: {selected_model}")
        print(f"Image filename: {image_filename}")
        print(f"Image path: {image_path}")
        print(f"Model path: {model_path}")
        
        try:
            model = load_custom_model(model_path)
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
        output = f"Prediction from {selected_model}: {predicted_class}"
        print(f'output: {output}')
        
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
            session['predict_input'] = predict_input_page1
            session['node0_input'] = node0_input
            session['node1_input'] = node1_input
            
            return redirect(url_for('go2'))
        except subprocess.CalledProcessError as e:
            print("Subprocess error:", e.stderr.decode())
            return jsonify({'error': e.stderr.decode()}), 500
        except Exception as e:
            print("General error:", str(e))
            return jsonify({'error': str(e)}), 500
        
@app.route('/go2', methods=['GET', 'POST'])
def go2():
    # Retrieve the necessary session variables
    shap_image_url = session.get('shap_image_url')
    predict_input = session.get('predict_input')
    node0_input = session.get('node0_input')
    node1_input = session.get('node1_input')
    
    if not all([shap_image_url, predict_input, node0_input, node1_input]):
        # Handle the case where session data is missing
        return redirect(url_for('index'))
    
    return render_template('shappage.html', 
                           shap_image_url=shap_image_url, 
                           predict_input=predict_input, 
                           node0_input=node0_input, 
                           node1_input=node1_input)
    
#------------------------------------------------------
#show
@app.route('/shappercentile', methods=['GET', 'POST'])
def shappercentile_page():
    

    predict_input_page = session.get('predict_input_page')
    node0_input_page = session.get('node0_input_page')
    node1_input_page = session.get('node1_input_page')
    
    print('a')
    print(f'predict_input_3: {predict_input_page}')
    print(f'node0_input3: {node0_input_page}')
    print(f'node1_input3: {node1_input_page}')
    
    grayscale_neg_image_url = url_for('static', filename='uploads/percentile/grayscale_image_plot_neg.png') 
    
    grayscale_pos_image_url = url_for('static', filename='uploads/percentile/grayscale_image_plot_pos.png') 
    
    print(grayscale_neg_image_url)
    print(grayscale_pos_image_url)
    return render_template(
        'shappercentile.html', grayscale_neg_image_url=grayscale_neg_image_url, grayscale_pos_image_url=grayscale_pos_image_url, predict_input1=predict_input_page, node0=node0_input_page, node1=node1_input_page)

### this route set 95% which is default 
@app.route('/default_shappercentile',  methods=['POST'])
def default_shappercentile():
    value1 = '95'
    value2 = '95'
    
    shap_values_left_opg_2 = 'shap_values_left.npy'
    
    # Call the grayscale.py script using subprocess.run
    result = subprocess.run(
        ['python', 'grayscale.py', shap_values_left_opg_2, value1, value2],
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard error
        check=True
    )
    
    predict_input_page = request.form.get('predict_input')
    node0_input_page = request.form.get('node0_input')
    node1_input_page = request.form.get('node1_input')
    
    session['predict_input_page'] = predict_input_page
    session['node0_input_page'] = node0_input_page
    session['node1_input_page'] = node1_input_page    
    
        
    print('AAA')
    print(f'predict_input_3: {predict_input_page}')
    print(f'node0_input3: {node0_input_page}')
    print(f'node1_input3: {node1_input_page}')
    
    return redirect(url_for('shappercentile_page')) 

#update
@app.route('/percentile', methods=['POST'])
def call_grayscale():
    data = request.json
    print(data)
    value1 = str(data.get('value1'))
    value2 = str(data.get('value2'))
    
    print(value1, value2)
    shap_values_left_opg_2 = 'shap_values_left.npy'
    
    # Call the grayscale.py script using subprocess.run
    result = subprocess.run(
        ['python', 'grayscale.py', shap_values_left_opg_2, value1, value2],
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard error
        check=True
    )
    return redirect(url_for('shappercentile_page'))

#evaluationpage-----------------------------------------------------------------
# Evaluation Page
            
# show
@app.route('/evaluationpage', methods=['POST', 'GET'])
def evaluationpage():
    left_image_path = session.get('left_image_path')
    selected_model = session.get('selected_model')

    predict_input_page = session.get('predict_input_page')
    node0_input_page = session.get('node0_input_page')
    node1_input_page = session.get('node1_input_page')
    
    print('sssssssssssss')
    print(f'predict_input_4: {predict_input_page}')
    print(f'node0_input4: {node0_input_page}')
    print(f'node1_input4: {node1_input_page}')
    
    # Initialize predicted_text with a default value
    predicted_text = ''

    # Run predict.py script
    result = subprocess.run(['python', 'predict.py', left_image_path, selected_model], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f'Error running predict.py: {result.stderr.decode("utf-8")}')
    else:
        predicted_text = result.stdout.decode('utf-8').strip()
        print('predicted_text:', predicted_text)

    if 'Predicted_text:' in predicted_text:
         predicted_text = predicted_text.split('Predicted_text:')[1].strip()
    
    iou_image_plot_neg_url = url_for('static', filename='uploads/iou/iou_image_plot_neg.png')
    iou_image_plot_pos_url = url_for('static', filename='uploads/iou/iou_image_plot_pos.png')
    
    print(iou_image_plot_neg_url)
    print(iou_image_plot_pos_url)
    
    descriptive_neg = pd.read_csv('result_neg.csv')
    descriptive_pos = pd.read_csv('result_pos.csv')

    tables_neg = descriptive_neg.to_html(classes='table table-striped', index=False)
    tables_pos = descriptive_pos.to_html(classes='table table-striped', index=False)
    
    # Synonyms dictionary
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

    # Function to map synonyms to standard terms
    def map_synonyms(term):
        for standard_term, syn_list in synonyms.items():
            if term in syn_list:
                return standard_term
        return term

    # Map synonyms in the result dataframes
    descriptive_neg['name'] = descriptive_neg['name'].apply(map_synonyms)
    descriptive_pos['name'] = descriptive_pos['name'].apply(map_synonyms)

    descriptive_neg_result = descriptive_neg
    descriptive_pos_result = descriptive_pos

    names_neg = set(descriptive_neg_result['name'])
    names_pos = set(descriptive_pos_result['name'])

    combined_names = sorted(names_neg.union(names_pos))
    names_list = ', '.join(combined_names)
    print(names_list)

    # Document processing and querying
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyCwSGELvgdp5T4yb_zXfeHZ5n7MHqyKAwA'
    ref_path = 'test_ref'

    raw_documents = DirectoryLoader(ref_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    loader = raw_documents.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(loader)

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(documents, embedding=gemini_embeddings, persist_directory="./langchain/chroma_db")

    vectorstore_disk = Chroma(persist_directory="./langchain/chroma_db", embedding_function=gemini_embeddings)
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7, top_p=0.85)

    llm_prompt_template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}

    Respond in the same language as the question.
    """

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    results = {}
    for name in combined_names:
        query = f"summarize relationship between {name} and {predict_input_page} determination with references?"
        result = rag_chain.invoke(query)
        results[name] = result
        print(result)
        
    def invoke_with_retry(chain, query, retries=5, backoff_factor=2):
        for i in range(retries):
            try:
                return chain.invoke(query)
            except ResourceExhausted as e:
                if i < retries - 1:
                    wait_time = backoff_factor ** i
                    print(f"Quota exhausted, retrying in {wait_time} seconds...")
                    time.sleep(20)
                else:
                    raise e
    # Store the result in the session
    session['document_query_result'] = results

    return render_template('evaluationpage.html', 
                           iou_image_plot_neg_url=iou_image_plot_neg_url, 
                           iou_image_plot_pos_url=iou_image_plot_pos_url, 
                           predict1=predict_input_page, 
                           node0_1=node0_input_page, 
                           node1_1=node1_input_page, 
                           predictresult=predicted_text, 
                           tables_descript_neg=[tables_neg], 
                           tables_descript_pos=[tables_pos],
                           document_query_result=results)


# Default Evaluation Page
@app.route('/default_evaluationpage', methods=['POST'])
def default_evaluationpage():
    value_iou = '10'
    left_image_path = session.get('left_image_path')
    csv_path = 'nms_csv.csv'
    grayscale_image_path_pos = 'save_grayscale_pos.npy'
    grayscale_image_path_neg = 'save_grayscale_neg.npy'
     
    process_yolo = subprocess.run(['python', 'yolo.py', left_image_path, csv_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    process_yolo_shap = subprocess.run(['python', 'yolo_shap.py', left_image_path, csv_path, grayscale_image_path_pos, grayscale_image_path_neg, '10'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return redirect(url_for('evaluationpage'))


@app.route('/evaluation', methods=['POST'])
def calculate_iou():
    data = request.json
    print(f'Received data: {data}')
    value_iou = float(data.get('value_iou'))
    print(f'New IOU value: {value_iou}')
    
    predict_input_page = session.get('predict_input_page')
    node0_input_page = session.get('node0_input_page')
    node1_input_page = session.get('node1_input_page')
    
    left_image_path = session.get('left_image_path')
    csv_path = 'nms_csv.csv'
    grayscale_image_path_pos = 'save_grayscale_pos.npy'
    grayscale_image_path_neg = 'save_grayscale_neg.npy'
    
    process_yolo = subprocess.run(['python', 'yolo.py', left_image_path, csv_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    process_yolo_shap = subprocess.run(['python', 'yolo_shap.py', left_image_path, csv_path, grayscale_image_path_pos, grayscale_image_path_neg, str(value_iou)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_yolo_shap.returncode != 0:
        print(f'Error running yolo_shap.py: {process_yolo_shap.stderr.decode("utf-8")}')

    print('before csv')
    descriptive_neg = pd.read_csv('result_neg.csv')
    descriptive_pos = pd.read_csv('result_pos.csv')
    
    tables_neg = descriptive_neg.to_html(classes='table table-striped', index=False)
    tables_pos = descriptive_pos.to_html(classes='table table-striped', index=False)
    print('before api')

    # Synonyms dictionary
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

    # Function to map synonyms to standard terms
    def map_synonyms(term):
        for standard_term, syn_list in synonyms.items():
            if term in syn_list:
                return standard_term
        return term

    # Map synonyms in the result dataframes
    descriptive_neg['name'] = descriptive_neg['name'].apply(map_synonyms)
    descriptive_pos['name'] = descriptive_pos['name'].apply(map_synonyms)

    descriptive_neg_result = descriptive_neg
    descriptive_pos_result = descriptive_pos

    names_neg = set(descriptive_neg_result['name'])
    names_pos = set(descriptive_pos_result['name'])

    combined_names = sorted(names_neg.union(names_pos))
    names_list = ', '.join(combined_names)
    print(names_list)

    # Document processing and querying
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyCwSGELvgdp5T4yb_zXfeHZ5n7MHqyKAwA'
    ref_path = 'test_ref'

    raw_documents = DirectoryLoader(ref_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    loader = raw_documents.load()
    #print(loader) 
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(loader)

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(documents, embedding=gemini_embeddings, persist_directory="./langchain/chroma_db")

    vectorstore_disk = Chroma(persist_directory="./langchain/chroma_db", embedding_function=gemini_embeddings)
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7, top_p=0.85)

    llm_prompt_template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}

    Respond in the same language as the question.
    """

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )
    
    results = {}
    for name in combined_names:
        query = f"summarize relationship between {name} and {predict_input_page} determination with references?"
        result = rag_chain.invoke(query)
        results[name] = result
        print(result)
    
    def invoke_with_retry(chain, query, retries=5, backoff_factor=2):
        for i in range(retries):
            try:
                return chain.invoke(query)
            except ResourceExhausted as e:
                if i < retries - 1:
                    wait_time = backoff_factor ** i
                    print(f"Quota exhausted, retrying in {wait_time} seconds...")
                    time.sleep(20)
                else:
                    raise e
    
    
    return jsonify({
        'tables_descript_neg': tables_neg,
        'tables_descript_pos': tables_pos,
        'document_query_result': results
    })



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)