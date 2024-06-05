"""
lit_ecology.helpers
===================

This module provides functions to extract and prioritize images from a tar archive based on class predictions.

Functions:
- read_predictions: Reads predictions from a specified file and returns them as a DataFrame.
- extract_priority_class_images: Extracts images of priority classes and saves a merged CSV.
- export_rest: Exports the list of rest images to a JSON file.
- filter_predictions_by_rest_list: Filters predictions by excluding rest images.
"""

import tarfile
import re
import os
import json
import argparse
import shutil
from collections import defaultdict
import pandas as pd

def read_predictions(predictions_file_path):
    """
    Reads image predictions from a file.

    Args:
        predictions_file_path (str): Path to the file containing predictions.

    Returns:
        pd.DataFrame: DataFrame containing filenames, class labels, and predictions.
    """
    data = []
    with open(predictions_file_path, 'r') as file:
        for line in file:
            match = re.match(r"([^/]+)/([^/]+)/([^/]+)\.jpeg------------------ ([^/]+)/([\d.]+)", line)
            if match:
                group, sub_group, filename, class_label, prediction = match.groups()
                full_filename = f"{group}/{sub_group}/{filename}.jpeg"
                data.append({'filename': full_filename, 'class': class_label, 'prediction': float(prediction)})

    df = pd.DataFrame(data).set_index('filename')
    return df

def extract_priority_class_images(output_base_dir, prio, rest):
    """
    Extracts images of priority classes and saves a merged CSV file.

    Args:
        output_base_dir (str): Path to the base output directory.
        prio (pd.DataFrame): DataFrame containing priority class predictions.
        rest (pd.DataFrame): DataFrame containing rest class predictions.

    Returns:
        None
    """
    os.makedirs(output_base_dir, exist_ok=True)
    prio.loc[prio["class"]=="rest","class"]=rest.loc[prio["class"]=="rest","class"]
    prio=prio[prio["class"]==prio["class"]]
    prio.to_csv(os.path.join(output_base_dir,"merged.csv"))
    # print(prio)



def export_rest(rest_images, tar_file_path, predictions, output_base_dir):
    """
    Exports the list of rest images to a JSON file.

    Args:
        rest_images (list): List of rest images.
        tar_file_path (str): Path to the tar file containing the images.
        predictions (pd.DataFrame): DataFrame containing image predictions.
        output_base_dir (str): Path to the base output directory.

    Returns:
        None
    """
    rest_images_json_path = os.path.join(output_base_dir, 'rest_images.json')
    with open(rest_images_json_path, 'w') as json_file:
        json.dump(rest_images, json_file)
    print(f"Saved rest images list to {rest_images_json_path}")

def filter_predictions_by_rest_list(predictions, rest_images_list):
    """
    Filters predictions by excluding rest images.

    Args:
        predictions (dict): Dictionary containing image predictions.
        rest_images_list (list): List of rest images to exclude.

    Returns:
        dict: Filtered predictions excluding rest images.
    """
    filtered_predictions = {k: v for k, v in predictions.items() if k not in rest_images_list}
    return filtered_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images and predictions based on priority classes.")
    parser.add_argument('output_base_dir', type=str, help="Path to the base output directory.")
    parser.add_argument('predictions_priority_file_path', type=str, help="Path to the priority predictions file.")
    parser.add_argument('predictions_rest_file_path', type=str, help="Path to the rest predictions file.")

    args = parser.parse_args()
    os.makedirs(args.output_base_dir, exist_ok=True)

    predictions = read_predictions(args.predictions_priority_file_path)
    predictions_rest = read_predictions(args.predictions_rest_file_path)

    extract_priority_class_images(args.output_base_dir, predictions, predictions_rest)
