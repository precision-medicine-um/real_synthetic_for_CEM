import os
import pandas as pd

import process_images as prim
from parameters import *

# Read the excel file containing image data
inputsheet = pd.read_excel(os.path.join(infodir,input_file), sheet_name=sheetname)

# Define the output JSON file for annotations
output_annotations = f'annotations_{subset}_{dataset}.json'
output_annotation_path = os.path.join(outputdatadir, output_annotations)

# Load existing annotations if the JSON file exists, otherwise create an empty DataFrame
if os.path.exists(output_annotation_path) :
    df = pd.read_json(output_annotation_path)
else :
    df = pd.DataFrame(columns=['filename', 'x1', 'x2', 'y1', 'y2', 'labels', 'path_mask', 'breast box'])

# Process images for each view in viewlist
for view in viewlist :
    df = prim.allprep(inputdatadir, outputdatadir, inputsheet,
                      df, subset,
                      startidx, view, dataset, preprocessing_method)

# Save the updated annotations to the JSON file
df.to_json(os.path.join(outputdatadir, output_annotations))

