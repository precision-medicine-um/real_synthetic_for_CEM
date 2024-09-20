import os
import pickle
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from utils import preprocessing_train, preprocessing_test
from radiomics_extraction import initialize_feature_extractor, generate_features_table

# Main function for training a machine learning model with radiomics features
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a machine learning model with radiomics features.')
    parser.add_argument('--path_excel_sets', default=None, help='Path to the Excel file containing paths to images and masks.')
    parser.add_argument('--sheetname_train', default=None, help='Sheet name for the training set.')
    parser.add_argument('--sheetname_test', default=None, help='Sheet name for the test set.')
    parser.add_argument('--num_classes', default=4, help='Number of classes (2 or 4).')
    parser.add_argument('--path_dir_features_to_save', required=True, help='Path to save radiomics features.')
    parser.add_argument('--find_optimal_features', default=False, help='Use RFECV to find optimal features.')
    parser.add_argument('--path_to_save_parameters', default=None, help='Path to save the trained model and predictions.')
    parser.add_argument('--append_save_name', default=None, help='String to append to saved models and predictions.')
    args = parser.parse_args()

    # Generate radiomics features for the test set
    if args.sheetname_test is not None:
        extractor = initialize_feature_extractor()
        df_test = pd.read_excel(args.path_excel_sets, sheet_name=args.sheetname_test)
        df_features_test = generate_features_table(df_test, extractor)
        df_features_test.to_excel(os.path.join(args.path_dir_features_to_save, "df_features_test.xlsx"))
        print("Radiomics features saved for the test set.")
    else:
        raise ValueError("You must provide a path and sheet name for the test set.")

    # Load pre-generated features for training and testing sets
    df_features_train = pd.read_excel(os.path.join(args.path_dir_features_to_save, "df_features_train.xlsx"))
    outcome_train = df_features_train.pop("outcome").tolist()  # Extract target variable
    df_features_train.drop(columns=["Unnamed: 0"], inplace=True)  # Remove unnecessary columns
    
    df_features_test = pd.read_excel(os.path.join(args.path_dir_features_to_save, "df_features_test.xlsx"))
    df_features_test.drop(columns=["Unnamed: 0"], inplace=True)

    # Set the appropriate objective and scoring metric based on the number of classes
    if args.num_classes == str(2):
        outcome_train = [val - 2 if val >= 2 else val for val in outcome_train]
        objective = 'binary:logistic'
        scoring = 'roc_auc'
    else:
        objective = 'multi:softprob'
        scoring = 'roc_auc_ovr'

    print(f"Unique classes in training set: {np.unique(outcome_train)}")

    # Preprocess training and test features
    mean_std, selector, to_drop, decor_dataset_train = preprocessing_train(df_features_train)
    decor_dataset_test = preprocessing_test(df_features_test, mean_std, selector, to_drop)
    print("Features processed.")

    # Initialize the XGBoost classifier
    model = xgb.XGBClassifier(colsample_bytree=1, objective=objective, eval_metric='logloss', nthread=4, seed=27, device='cuda')

    # Feature selection using RFECV or RFE
    if args.find_optimal_features:
        print("Using RFECV to find optimal features.")
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(10), scoring=scoring, min_features_to_select=1)
        rfecv.fit(decor_dataset_train, outcome_train)
        support = rfecv.support_
    else:
        print("Using RFE to select features.")
        rfe = RFE(estimator=model, n_features_to_select=10)
        rfe.fit(decor_dataset_train, outcome_train)
        support = rfe.support_

    # Filter dataset to include only selected features
    filtered_col = np.extract(support, decor_dataset_train.columns)
    reduced_features_train_set = decor_dataset_train[filtered_col]
    reduced_features_test_set = decor_dataset_test[filtered_col]
    print("Reduced feature set.")

    # Define parameter grid for hyperparameter tuning
    param_test_xgb = {
        'max_depth': range(2, 4, 1),
        'min_child_weight': range(1, 6, 2),
        'gamma': [i * 0.1 for i in range(1, 10)],
        'n_estimators': np.linspace(10, 1000, 10, dtype=int).tolist(),
        'learning_rate': [10 ** (-i) for i in range(2, 7)]
    }

    # Hyperparameter tuning with GridSearchCV
    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    gsearch = GridSearchCV(model, param_grid=param_test_xgb, scoring=scoring, n_jobs=6, cv=kfold, verbose=1)
    gsearch.fit(reduced_features_train_set, outcome_train)
    print("Fitted model to features.")

    # Make predictions on training and test sets
    proba_train = gsearch.best_estimator_.predict_proba(reduced_features_train_set)
    proba_test = gsearch.best_estimator_.predict_proba(reduced_features_test_set)
    print("Predictions made.")

    # Save the model, parameters, and predictions if path is provided
    if args.path_to_save_parameters is not None:
        path_to_save = args.path_to_save_parameters

        if args.find_optimal_features:
            filename_rfecv = os.path.join(path_to_save, f'rfecv_radiomics_{args.append_save_name}.pkl')
            pickle.dump(rfecv, open(filename_rfecv, 'wb'))
        else:
            filename_rfe = os.path.join(path_to_save, f'rfe_radiomics_{args.append_save_name}.pkl')
            pickle.dump(rfe, open(filename_rfe, 'wb'))

        filename_gsearch = os.path.join(path_to_save, f'gsearch_radiomics_{args.append_save_name}.pkl')
        pickle.dump(gsearch, open(filename_gsearch, 'wb'))
        filename_parameters = os.path.join(path_to_save, f"parameters_radiomics_{args.append_save_name}.pkl")
        pickle.dump([mean_std, selector, to_drop, support], open(filename_parameters, 'wb'))
        filename_proba_train = os.path.join(path_to_save, f"proba_train_radiomics_{args.append_save_name}.pkl")
        pickle.dump(proba_train, open(filename_proba_train, 'wb'))
        filename_proba_test = os.path.join(path_to_save, f"proba_test_radiomics_{args.append_save_name}.pkl")
        pickle.dump(proba_test, open(filename_proba_test, 'wb'))

if __name__ == '__main__':
    main()
