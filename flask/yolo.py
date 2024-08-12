import sys
import torch
import json
import pandas as pd

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def nms_per_class(df, iou_threshold=0.1):
    df_nms = pd.DataFrame()
    class_ids = df['class'].unique()

    for class_id in class_ids:
        df_class = df[df['class'] == class_id].copy()

        df_class_sorted = df_class.sort_values(by='confidence', ascending=False).reset_index(drop=True)
        retained_indices = []
        iou_values = []

        for i in range(len(df_class_sorted)):
            if i in retained_indices:
                continue

            retained_indices.append(i)
            iou_values.append(1.0)

            for j in range(i + 1, len(df_class_sorted)):
                if j in retained_indices:
                    continue

                boxA = df_class_sorted.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']]
                boxB = df_class_sorted.iloc[j][['xmin', 'ymin', 'xmax', 'ymax']]
                iou_value = compute_iou(boxA, boxB)

                if iou_value > iou_threshold:
                    retained_indices.append(j)
                    iou_values.append(iou_value)

        df_nms_class = df_class_sorted.iloc[retained_indices].reset_index(drop=True)

        df_nms_class['iou'] = iou_values

        df_nms = pd.concat([df_nms, df_nms_class], ignore_index=True)

    return df_nms

def detect(image_path, csv_path):
    # Load the model
    weights_path = 'templates/best.pt'
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    results = yolo_model(image_path)
    # Convert results to a pandas DataFrame
    df_results = results.pandas().xyxyn[0]
    # Apply NMS
    df_nms_filtered = nms_per_class(df_results, iou_threshold=0.5)
    # Save NMS-filtered results to CSV
    df_nms_filtered.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    image_path = sys.argv[1]
    csv_path = sys.argv[2]
    detect(image_path, csv_path)

