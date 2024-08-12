



import sys
import json
import pandas as pd
import numpy as np
import shap
from keras.preprocessing.image import load_img, img_to_array 
from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import os, sys

app = {'config': {'FOLDER_PERCENTILE': 'static/uploads/percentile/'}}

def process_input(shap_values_left, value_n, value_p):
    
    print('-----------------GRAYSCALE_PROCESSING...------------------')
    loaded_shap_values = np.load(shap_values_left, allow_pickle=True)
    
    # Assuming data is received as JSON
    shap_values = np.array(loaded_shap_values)
    data = shap_values
    print(data)
   # Process Positive Values
    image_array = data[0]  # Index in the list containing the array
    positive = np.where(image_array >= 0, image_array, 0)
    flattened_array_pos = positive.flatten()
    normalized_array_pos = (flattened_array_pos - np.min(flattened_array_pos)) / (np.max(flattened_array_pos) - np.min(flattened_array_pos))
    normalized_positive = normalized_array_pos.reshape(positive.shape)
    grayscale_image_pos = normalized_positive / 3.0
    grayscale_image_positive = np.mean(grayscale_image_pos, axis=3)
    grayscale_image_positive = grayscale_image_positive.squeeze()

    # Adjust values
    percentile_pos = np.percentile(grayscale_image_positive, value_p)
    grayscale_pos_thresholded = np.copy(grayscale_image_positive)
    grayscale_pos_thresholded[grayscale_pos_thresholded < percentile_pos] = 0

    #print('grayscale_pos_thresholded', grayscale_pos_thresholded)
    
    save_grayscale_pos = 'save_grayscale_pos.npy'
    np.save(save_grayscale_pos, grayscale_pos_thresholded)
    # Filter out zero values
    filtered_array_grayscale_pos_thresholded = grayscale_pos_thresholded[grayscale_pos_thresholded != 0]

# Print the full filtered array without truncation
    np.set_printoptions(threshold=np.inf)
    print('filtered_array_grayscale_pos_thresholded')
    print(filtered_array_grayscale_pos_thresholded)

# Print the thresholded grayscale image array
    # print('grayscale_pos_thresholded')
    # print(grayscale_pos_thresholded)

    # Plot Positive Image
    plt.imshow(grayscale_pos_thresholded, cmap='Reds')
    plt.axis('off')
    
    image_filename = 'grayscale_image_plot_pos.png'  # Include the file extension
    image_path = os.path.join(app['config']['FOLDER_PERCENTILE'], image_filename)
    
    plt.savefig(image_path)
    plt.close()
### NEG-------------------------------------------------------------------------------------------------------
    # Process Negative Values
    image_array = data[0]
    negative = np.where(image_array < 0, image_array, 0)
    negative_aps = np.abs(negative)
    flattened_array_neg = negative_aps.flatten()
    normalized_array_neg = (flattened_array_neg - np.min(flattened_array_neg)) / (np.max(flattened_array_neg) - np.min(flattened_array_neg))
    normalized_neg = normalized_array_neg.reshape(negative_aps.shape)
    grayscale_image_neg = normalized_neg / 3.0
    grayscale_image_negative = np.mean(grayscale_image_neg, axis=3)
    grayscale_image_negative = grayscale_image_negative.squeeze()


    # Adjust values
    percentile_neg = np.percentile(grayscale_image_negative, value_n)
    grayscale_neg_thresholded = np.copy(grayscale_image_negative)
    grayscale_neg_thresholded[grayscale_neg_thresholded < percentile_neg] = 0

    
    #print('grayscale_neg_thresholded:', grayscale_neg_thresholded)
    
    save_grayscale_neg = 'save_grayscale_neg.npy'
    np.save(save_grayscale_neg, grayscale_neg_thresholded)
        # Filter out zero values
    filtered_array_grayscale_neg_thresholded = grayscale_neg_thresholded[grayscale_neg_thresholded != 0]

# Print the full filtered array without truncation
    np.set_printoptions(threshold = np.inf)
    print('filtered_array_grayscale_neg_thresholded')
    print(filtered_array_grayscale_neg_thresholded)
    
    # Plot Negative Image
    plt.imshow(grayscale_neg_thresholded, cmap='Blues')
    plt.axis('off')
    
    image_filename = 'grayscale_image_plot_neg.png'  # Include the file extension
    image_path = os.path.join(app['config']['FOLDER_PERCENTILE'], image_filename)
    
    plt.savefig(image_path)
    plt.close()

if __name__ == '__main__':
    shap_values_left = sys.argv[1]
    value_n = int(sys.argv[2])
    value_p = int(sys.argv[3])

    processed_data = process_input(shap_values_left, value_n, value_p)
    
    
    
    # left_image_filename = 'left_' + image_filename
    # left_image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], left_image_filename)
    # print(left_image_path)
