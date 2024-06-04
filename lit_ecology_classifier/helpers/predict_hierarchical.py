import tarfile
import re
import os
import json
import argparse
import shutil
from collections import defaultdict
import pandas as pd
import re
"""
Script to extract and prioritize images from a tar archive based on class predictions.

This script performs the following tasks:

1. Reads image predictions from priority and rest prediction files.
2. Merges and filters predictions to identify priority images.
3. Saves the list of rest images for reference.

Key Features:

- Prioritizes images belonging to specific classes.
- Saves a merged CSV of priority predictions.
- Records rest images in a JSON file for further analysis.

Usage:

```bash
python script_name.py output_base_dir predictions_priority_file_path predictions_rest_file_path
```
"""
def read_predictions(predictions_file_path):
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

# Function to extract images from the tar file that belong to priority classes
def extract_priority_class_images( output_base_dir, prio,rest):
    # Create base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    prio.loc[prio["class"]=="rest","class"]=rest.loc[prio["class"]=="rest","class"]
    prio=prio[prio["class"]==prio["class"]]
    prio.to_csv(os.path.join(output_base_dir,"merged.csv"))
    # print(prio)



def export_rest(rest_images,tar_file_path, predictions, output_base_dir):
    # Save the list of rest images to a JSON file
    rest_images_json_path = os.path.join(output_base_dir, 'rest_images.json')
    with open(rest_images_json_path, 'w') as json_file:
        json.dump(rest_images, json_file)
    print(f"Saved rest images list to {rest_images_json_path}")




def filter_predictions_by_rest_list(predictions, rest_images_list):
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
    print(predictions)
    extract_priority_class_images(args.output_base_dir, predictions, predictions_rest)
