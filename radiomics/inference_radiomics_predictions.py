import os
import pickle
import argparse
import numpy as np
import pandas as pd

from utils import preprocessing_test
from radiomics_extraction import initialize_feature_extractor, generate_features_table_predictions_chat


# Main function for generating radiomics predictions
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate radiomics predictions using a trained model.')
    parser.add_argument('--path_excel_sets', default=None, help='Path to Excel file containing image and mask paths.')
    parser.add_argument('--sheetname_test', default=None, help='Sheet name for the test set.')
    parser.add_argument('--path_predictions', default=None, help='Path to JSON file with the predictions.')
    parser.add_argument('--dataset', default=None, help='Dataset name.')
    parser.add_argument('--path_to_load_parameters', default=None, help='Path to load the pre-trained parameters.')
    parser.add_argument('--path_to_load_model', default=None, help='Path to load the trained model.')
    parser.add_argument('--path_dir_to_save', required=True, help='Directory to save the results.')
    args = parser.parse_args()

    # Generate radiomics features if test set is provided
    if args.sheetname_test is not None:
        extractor = initialize_feature_extractor()
        df = pd.read_excel(args.path_excel_sets, sheet_name=args.sheetname_test)  # Load test set
        df_preds = pd.read_json(args.path_predictions)  # Load predictions

        # Generate radiomics features based on the test set and predictions
        df_features_test = generate_features_table_predictions_chat(df, df_preds, args.dataset, extractor, inference_usage=True)
        df_features_test.to_excel(os.path.join(args.path_dir_to_save, "df_features_test.xlsx"))
    
    # Load generated test features from file
    df_features_test = pd.read_excel(os.path.join(args.path_dir_to_save, "df_features_test.xlsx"), index_col=0)

    # Load pre-trained parameters for feature preprocessing
    with open(args.path_to_load_parameters, 'rb') as f:
        mean_std, selector, to_drop, support = pickle.load(f)

    # Extract mask paths and preprocess test features
    path_masks = df_features_test.pop('mask path').tolist()  # Remove mask path column and store values
    decor_dataset = preprocessing_test(df_features_test, mean_std, selector, to_drop)  # Preprocess test features

    # Select only the important features based on the pre-trained model
    filtered_col = np.extract(support, np.array(decor_dataset.columns))
    reduced_features = decor_dataset[filtered_col]
    print("Features processed.")

    # Load the pre-trained model for inference
    with open(args.path_to_load_model, 'rb') as f:
        gsearch = pickle.load(f)
    best_estimator = gsearch.best_estimator_

    predictions_rmics_df = pd.DataFrame()

    # Generate predictions for each test case (image)
    for idx, pathname in enumerate(path_masks):
        temp_proba = best_estimator.predict_proba(reduced_features.iloc[[idx]])  # Predict for single case
        temp_proba_df = pd.DataFrame(temp_proba)  # Convert to DataFrame
        temp_proba_df['mask path'] = pathname  # Add mask path to predictions
        predictions_rmics_df = pd.concat([predictions_rmics_df, temp_proba_df], ignore_index=True)  # Append to results

    # Save all predictions to an Excel file
    predictions_rmics_df.to_excel(os.path.join(args.path_dir_to_save, "df_predictions_rmics_test.xlsx"))
    print(f"Radiomics predictions generated for {len(path_masks)} images.")

if __name__ == '__main__':
    main()
