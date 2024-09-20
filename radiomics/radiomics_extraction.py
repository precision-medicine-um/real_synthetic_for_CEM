import os
import tqdm
import numpy as np
import pandas as pd
import SimpleITK as sitk
import concurrent.futures

from radiomics import featureextractor


def initialize_feature_extractor():
    """Initializes the radiomics feature extractor with predefined settings."""
    
    params_file = "CEM_extraction.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file, shape2D=True, 
                                                           force2D=True, force2Ddimension=True)
    extractor.addProvenance(False)  # Disable provenance information
    extractor.disableAllFeatures()  # Disable all features initially
    extractor.enableImageTypes(Original={})  # Enable original image type

    # Enable specific feature classes
    feature_classes = ['firstorder', 'shape2D', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    for feature_class in feature_classes:
        extractor.enableFeatureClassByName(feature_class, enabled=True)
    
    return extractor


def generate_features_table(df, extractor, inference_usage=False):
    """
    Extracts radiomics features for both low-energy and recombined images.
    Returns a DataFrame with the extracted features.
    """
    feature_df_low_energy = pd.DataFrame()
    feature_df_recombined = pd.DataFrame()

    # Iterate through the dataset and extract features
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='Processing Rows'):
        try:
            path_low_energy = row["path_low_energy"]
            path_recombined = row["path_recombined"]
            path_mask = row["path_mask"]

            # Extract low-energy features
            feature_vector_low_energy = extractor.execute(path_low_energy, path_mask)
            feature_df_low_energy = pd.concat([feature_df_low_energy, pd.Series(feature_vector_low_energy).to_frame().T], ignore_index=True)

            # Extract recombined features
            feature_vector_recombined = extractor.execute(path_recombined, path_mask)
            feature_df_recombined = pd.concat([feature_df_recombined, pd.Series(feature_vector_recombined).to_frame().T], ignore_index=True)

        except Exception as e:
            print(e, row)

    # Rename columns for low-energy and recombined features
    feature_df_low_energy.columns = [str(col) + "_low_energy" for col in feature_df_low_energy.columns]
    feature_df_recombined.columns = [str(col) + "_recombined" for col in feature_df_recombined.columns]

    # Add outcome column if not in inference mode
    if not inference_usage:
        feature_df_recombined["outcome"] = df["outcome"].values

    # Return the combined DataFrame
    return feature_df_recombined.join(feature_df_low_energy)


def process_row(row, df, extractor, dataset):
    """
    Processes a single row, extracting features for both low-energy and recombined images.
    Returns the extracted features and the mask paths.
    """
    breast = row['filename']
    
    # Determine the dataset format and extract relevant parts of the filename
    if dataset == 'MUMC':
        setname, patientnumber, laterality, view = breast.split('_')
        patientname = f'{setname}_{patientnumber}'
    elif dataset == 'GR':
        patientname, laterality, view = breast.split('_')

    bbox_breast = row['breast_box']
    path_mask = row['path_mask'][:-3] + 'npz'
    pred_mask = np.load(path_mask)['arr_0']

    # Find matching paths in the DataFrame
    image_le_path = next(path for path in df['path_low_energy'] if f'{patientname}_{view}_{laterality}' in path)
    image_recombined_path = next(path for path in df['path_recombined'] if f'{patientname}_{view}_{laterality}' in path)

    image_le = sitk.ReadImage(image_le_path)
    array_le = sitk.GetArrayFromImage(image_le)

    results = []
    
    # Process each mask label
    for mask_label, pred_mask_array in enumerate(pred_mask):
        temp_mask = np.zeros_like(array_le)
        temp_mask[bbox_breast[0]:bbox_breast[1], bbox_breast[2]:bbox_breast[3]] = pred_mask_array
        temp_mask[temp_mask > 0] = 1

        image_mask = sitk.GetImageFromArray(temp_mask)
        save_mha_name = os.path.join(os.path.dirname(path_mask), f'{breast}_{mask_label}.mha')
        sitk.WriteImage(image_mask, save_mha_name)
        
        try:
            # Extract features for low-energy and recombined images
            feature_vector_low_energy = extractor.execute(image_le_path, save_mha_name)
            feature_vector_recombined = extractor.execute(image_recombined_path, save_mha_name)
            results.append((feature_vector_low_energy, feature_vector_recombined, save_mha_name))

        except Exception as e:
            print(e)
            results.append(({}, {}, save_mha_name))

        os.remove(save_mha_name)

    return results


def generate_features_table_predictions_chat(df, df_preds, dataset, extractor, inference_usage=False):
    """
    Generates feature tables for predictions using multi-threading.
    Extracts features for each prediction in df_preds.
    """
    feature_df_low_energy = pd.DataFrame()
    feature_df_recombined = pd.DataFrame()
    image_path_list = []

    # Multi-threaded execution for faster processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_row, row, df, extractor, dataset)
            for _, row in df_preds.iterrows()
        ]

        # Process results as they complete
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing Rows'):
            results = future.result()
            for feature_vector_low_energy, feature_vector_recombined, save_mha_name in results:
                image_path_list.append(save_mha_name)

                feature_df_low_energy = pd.concat([feature_df_low_energy, pd.Series(feature_vector_low_energy).to_frame().T], ignore_index=True)
                feature_df_recombined = pd.concat([feature_df_recombined, pd.Series(feature_vector_recombined).to_frame().T], ignore_index=True)

    # Rename columns for low-energy and recombined features
    feature_df_low_energy.columns = [str(col) + "_low_energy" for col in feature_df_low_energy.columns]
    feature_df_recombined.columns = [str(col) + "_recombined" for col in feature_df_recombined.columns]

    # Add outcome column if not in inference mode
    if not inference_usage:
        feature_df_recombined["outcome"] = df["outcome"].values

    # Combine all features and return
    feature_df_all = feature_df_recombined.join(feature_df_low_energy)
    feature_df_all.insert(0, 'mask path', image_path_list)
    
    return feature_df_all
