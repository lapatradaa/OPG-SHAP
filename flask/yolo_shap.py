import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os

app = {'config': {'FOLDER_IOU': 'static/uploads/iou/'}}

def plot_bboxes_on_image_pos(image_path, df, grayscale_image, iou_threshold=0.1):
    selected_bboxes = []
    value_than_IoU = []

    img = Image.open(image_path)
    img_array = np.array(img)
    image_width, image_height = img.size

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(grayscale_image, cmap='Reds', alpha=0.6, extent=[0, image_width, image_height, 0])

    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        confidence, class_label, class_name, IoU = row['confidence'], row['class'], row['name'], row['iou']

        abs_xmin = xmin * image_width
        abs_ymin = ymin * image_height
        abs_width = (xmax - xmin) * image_width
        abs_height = (ymax - ymin) * image_height

        roi_xmin = int(xmin * grayscale_image.shape[1])
        roi_ymin = int(ymin * grayscale_image.shape[0])
        roi_xmax = int(xmax * grayscale_image.shape[1])
        roi_ymax = int(ymax * grayscale_image.shape[0])
        roi = grayscale_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

        nonzero_percentage = np.count_nonzero(roi) / (roi.shape[0] * roi.shape[1])

        if nonzero_percentage > iou_threshold:
            selected_bboxes.append({'xmin': abs_xmin, 'ymin': abs_ymin, 'xmax': abs_xmin + abs_width, 'ymax': abs_ymin + abs_height,
                                    'confidence': confidence, 'class': class_label, 'name': class_name, 'iou': nonzero_percentage})
            value_than_IoU.append(nonzero_percentage)

            rect = patches.Rectangle((abs_xmin, abs_ymin), abs_width, abs_height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            text = f'Class: {class_label}'
            plt.text(abs_xmin, abs_ymin - 10, text, color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    
    image_filename = 'iou_image_plot_pos.png'  # Include the file extension
    image_path = os.path.join(app['config']['FOLDER_IOU'], image_filename)
    plt.savefig(image_path)
    plt.close()
    
    plt.show()
     
    return selected_bboxes, value_than_IoU

def plot_bboxes_on_image_neg(image_path, df, grayscale_image, iou_threshold=0.1):
    selected_bboxes = []
    value_than_IoU = []

    img = Image.open(image_path)
    img_array = np.array(img)
    image_width, image_height = img.size

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(grayscale_image, cmap='Blues', alpha=0.6, extent=[0, image_width, image_height, 0])

    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        confidence, class_label, class_name, IoU = row['confidence'], row['class'], row['name'], row['iou']

        abs_xmin = xmin * image_width
        abs_ymin = ymin * image_height
        abs_width = (xmax - xmin) * image_width
        abs_height = (ymax - ymin) * image_height

        roi_xmin = int(xmin * grayscale_image.shape[1])
        roi_ymin = int(ymin * grayscale_image.shape[0])
        roi_xmax = int(xmax * grayscale_image.shape[1])
        roi_ymax = int(ymax * grayscale_image.shape[0])
        roi = grayscale_image[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

        nonzero_percentage = np.count_nonzero(roi) / (roi.shape[0] * roi.shape[1])

        if nonzero_percentage > iou_threshold:
            selected_bboxes.append({'xmin': abs_xmin, 'ymin': abs_ymin, 'xmax': abs_xmin + abs_width, 'ymax': abs_ymin + abs_height,
                                    'confidence': confidence, 'class': class_label, 'name': class_name, 'iou': nonzero_percentage})
            value_than_IoU.append(nonzero_percentage)

            rect = patches.Rectangle((abs_xmin, abs_ymin), abs_width, abs_height, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            text = f'Class: {class_label}'
            plt.text(abs_xmin, abs_ymin - 10, text, color='black', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    
    image_filename = 'iou_image_plot_neg.png'  # Include the file extension
    image_path = os.path.join(app['config']['FOLDER_IOU'], image_filename)
    
    plt.savefig(image_path)
    plt.close()
    
    plt.show()
    
    return selected_bboxes, value_than_IoU

def calculate_mean_iou(selected_bboxes):
    if not selected_bboxes:
        return 0
    total_iou = sum(bbox['iou'] for bbox in selected_bboxes)
    mean_iou = total_iou / len(selected_bboxes)
    return mean_iou

def get_values_in_bounding_boxes(array, df_bboxes):
    results = []
    for index, row in df_bboxes.iterrows():
        bbox_values = array[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]
        results.append({
            'class': row['class'],
            'name': row['name'],
            'values': bbox_values
        })
    return results

def calculate_mean_values_in_bounding_boxes(values_in_bboxes):
    mean_values = []
    for bbox in values_in_bboxes:
        if bbox['values'].size == 0:
            mean_shap = 0  # or some other placeholder value
        else:
            mean_shap = np.mean(bbox['values']) * 100
        mean_values.append({
            'class': bbox['class'],
            'name': bbox['name'],
            'mean_shap': mean_shap
        })
    return mean_values

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("Usage: python script.py <image_path> <csv_path> <grayscale_pos_path> <grayscale_neg_path> [iou_threshold]")
        sys.exit(1)

    image_path = sys.argv[1]
    csv_path = sys.argv[2]
    grayscale_pos = sys.argv[3]
    grayscale_neg = sys.argv[4]
    iou_threshold = sys.argv[5]
    try:
        iou_threshold = float(iou_threshold) / 100
    except ValueError:
        print("Invalid value for iou_threshold. It must be a numerical value.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    grayscale_pos = np.load(grayscale_pos)
    grayscale_neg = np.load(grayscale_neg)
    r_grayscale_pos = np.random.random((500, 600)) * 0.01  
    r_grayscale_neg = np.random.random((500, 600)) * 0.01  

    selected_bboxes_pos, _ = plot_bboxes_on_image_pos(image_path, df, grayscale_pos, iou_threshold)
    selected_bboxes_neg, _ = plot_bboxes_on_image_neg(image_path, df, grayscale_neg, iou_threshold)

    print('selected_bboxes_pos', selected_bboxes_pos)
    print('selected_bboxes_neg', selected_bboxes_neg)

    mean_iou_pos = calculate_mean_iou(selected_bboxes_pos)
    mean_iou_neg = calculate_mean_iou(selected_bboxes_neg)

    print(f'Mean IoU for positive grayscale image: {mean_iou_pos:.2f}%')
    print(f'Mean IoU for negative grayscale image: {mean_iou_neg:.2f}%')

    # Process positive grayscale image
    df_bboxes_pos = pd.DataFrame(selected_bboxes_pos)
    values_in_bboxes_pos = get_values_in_bounding_boxes(r_grayscale_pos, df_bboxes_pos)
    mean_values_in_bboxes_pos = calculate_mean_values_in_bounding_boxes(values_in_bboxes_pos)
    

    E1_pos = pd.DataFrame(mean_values_in_bboxes_pos)
    E2_pos = pd.DataFrame(selected_bboxes_pos)
    merged_pos = pd.merge(E1_pos, E2_pos, on=['class', 'name'])
    result_pos = merged_pos[['class', 'name', 'mean_shap', 'iou']]
    result_pos['iou'] = result_pos['iou'] * 100  # Convert IoU to percentage for display
    result_pos_sorted = result_pos.sort_values(by='class', ascending=True)
    
    print("Results for Positive Grayscale Image:")
    print(result_pos_sorted)
    csv_path_pos = 'result_pos.csv'
    result_pos_sorted.to_csv(csv_path_pos, index=False)

    # Process negative grayscale image
    df_bboxes_neg = pd.DataFrame(selected_bboxes_neg)
    values_in_bboxes_neg = get_values_in_bounding_boxes(r_grayscale_neg, df_bboxes_neg)
    mean_values_in_bboxes_neg = calculate_mean_values_in_bounding_boxes(values_in_bboxes_neg)

    E1_neg = pd.DataFrame(mean_values_in_bboxes_neg)
    E2_neg = pd.DataFrame(selected_bboxes_neg)
    merged_neg = pd.merge(E1_neg, E2_neg, on=['class', 'name'])
    result_neg = merged_neg[['class', 'name', 'mean_shap', 'iou']]
    result_neg['iou'] = result_neg['iou'] * 100  # Convert IoU to percentage for display
    result_neg_sorted = result_neg.sort_values(by='class', ascending=True)

    print("Results for Negative Grayscale Image:")
    print(result_neg_sorted)
    csv_path_neg = 'result_neg.csv'
    result_neg_sorted.to_csv(csv_path_neg, index=False)
