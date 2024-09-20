#%% Imports

import os
import random
import argparse
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Argument parser to handle input arguments
parser = argparse.ArgumentParser(description='Code for bootstrapping')
parser.add_argument('--path_predictions', default=None,  help='Path to the folder containing the predictions')
parser.add_argument('--file_predictions', default=None, help='Filename of the json containing the predictions')
parser.add_argument('--dataset', default='MUMC')
args = parser.parse_args()

#%% Computing functions

def largest_iou_idx(iou_list, num_preds, region_idx):
    """Find the index of the highest IoU for a given region."""

    ious_region = [iou_list[i*num_preds + region_idx] for i in range(len(iou_list)//num_preds)]
    max_iou_idx = np.argmax(ious_region)

    return max_iou_idx * num_preds + region_idx


def add_patientname_df(target_df, columnname):
    """Extract patient names from file paths in a DataFrame column."""

    patientname_list = [filepath.split('\\')[-1][:-4] for filepath in target_df[columnname]]
    target_df.insert(1, 'patientname', patientname_list)

    return target_df


def unique_predictions(pred_df, target_df, max_predictions_breast):
    """Generate unique predictions by matching patients in the prediction and target DataFrames."""
    pred_all_list = []

    # Loop over each patient
    for target_idx in range(len(target_df)):
        targetname = target_df['patientname'][target_idx]
        targetlabel = target_df['labels'][target_idx]
        num_targets = len(target_df['x1'][target_idx]) if isinstance(target_df['x1'][target_idx], (list, np.ndarray)) else 1

        # Find matching predictions for this patient
        for pred_idx in range(len(pred_df)):
            if pred_df['filename'][pred_idx] == targetname:
                ious, dices, scores, labels = pred_df['iou'][pred_idx], pred_df['dice'][pred_idx], pred_df['scores'][pred_idx], pred_df['labels'][pred_idx]
                xmins, xmaxs, ymins, ymaxs = pred_df['x1'][pred_idx], pred_df['x2'][pred_idx], pred_df['y1'][pred_idx], pred_df['y2'][pred_idx]
                num_preds = len(ious) // num_targets

                # Process predictions for each region
                for region_idx in range(min(num_preds, max_predictions_breast)):
                    max_iou_idx = largest_iou_idx(ious, num_preds, region_idx)
                    target_region_idx = (max_iou_idx - region_idx) // num_preds

                    # Convert score to benign-malignant probabilities
                    score_mal = 0.5 * scores[max_iou_idx] + 0.5
                    score_ben = 1 - score_mal if labels[max_iou_idx] == 1 else 0.5 * scores[max_iou_idx] + 0.5

                    pred_all_list.append([targetname, target_region_idx, targetlabel,
                                          labels[max_iou_idx]-1, scores[max_iou_idx], score_ben, score_mal, dices[max_iou_idx], ious[max_iou_idx],
                                          xmins[max_iou_idx], xmaxs[max_iou_idx], ymins[max_iou_idx], ymaxs[max_iou_idx]])

    pred_all_df = pd.DataFrame(pred_all_list, columns=['targetname', 'targetregion', 'targetlabel',
                                                       'predlabel', 'score', 'scoreben', 'scoremal', 'dice', 'iou',
                                                       'xmin', 'xmax', 'ymin', 'ymax'])
    return pred_all_df


def unique_targets(target_df):
    """Separate target regions into distinct rows per region."""
    targets_all_list = []

    for _, row in target_df.iterrows():
        for idx in range(len(row['x1'])):
            target_row = row.copy()
            target_row['x1'], target_row['x2'] = row['x1'][idx], row['x2'][idx]
            target_row['y1'], target_row['y2'] = row['y1'][idx], row['y2'][idx]
            target_row['targetregion'] = idx
            targets_all_list.append(target_row.values)

    targets_all_df = pd.DataFrame(targets_all_list, columns=target_df.columns.tolist() + ['targetregion'])
    return targets_all_df


def combine_df_rmics(pred_df, rmics_df):
    """Combine DL predictions with radiomics predictions."""
    avg_ben_list, avg_mal_list = [], []

    for idx in range(len(pred_df)):
        ben_score = (pred_df['scoreben'][idx] + rmics_df[0][idx]) / 2
        mal_score = (pred_df['scoremal'][idx] + rmics_df[1][idx]) / 2
        total_score = ben_score + mal_score
        avg_ben_list.append(ben_score / total_score)
        avg_mal_list.append(mal_score / total_score)

    pred_df.insert(7, 'scoreben rmics', rmics_df[0])
    pred_df.insert(8, 'scoremal rmics', rmics_df[1])
    pred_df.insert(9, 'scoreben avg', avg_ben_list)
    pred_df.insert(10, 'scoremal avg', avg_mal_list)

    return pred_df


def prediction_is_detected(malignant_def, row, dl_th=0.5, rmics_th=0.5, avg_th=0.5):
    """Check if a prediction detects a malignant lesion based on different criteria."""
    match malignant_def:
        case 'dl_label':
            return (row['predlabel'] % 2 == 0) == (row['targetlabel'] % 2 == 0)
        case 'dl_score':
            return (row['scoreben'] > dl_th) == (row['targetlabel'] % 2 == 0)
        case 'rmics_score':
            return (row['scoreben rmics'] > rmics_th) == (row['targetlabel'] % 2 == 0)
        case 'minor_vote':
            return (row['scoreben'] > dl_th and row['scoreben rmics'] > rmics_th) == (row['targetlabel'] % 2 == 0)
        case 'major_vote':
            return (row['scoreben'] > dl_th or row['scoreben rmics'] > rmics_th) == (row['targetlabel'] % 2 == 0)
        case 'average':
            return (row['scoreben avg'] > avg_th) == (row['targetlabel'] % 2 == 0)
        case _:
            return False


def detected_regions(combined_df, target_df, malignant_def='dl_label', iou_th=0.1, score_th=0.1):
    """Categorize detected regions into true positives, false positives, etc."""
    true_pos_list, true_pos_target_list, false_pos_list = [], [], []
    false_neg_df = target_df.copy()  # Assume no regions were detected

    for _, row in combined_df.iterrows():
        if row['iou'] > iou_th and row['score'] > score_th and prediction_is_detected(malignant_def, row):
            true_pos_list.append(row.values)
            false_neg_df = false_neg_df.drop(false_neg_df[(false_neg_df['patientname'] == row['targetname'])
                                                          & (false_neg_df['targetregion'] == row['targetregion'])].index)
            true_pos_row = target_df.loc[(target_df['patientname'] == row['targetname']) &
                                         (target_df['targetregion'] == row['targetregion'])]
            true_pos_target_list.append(true_pos_row.values[0])
        else:
            false_pos_list.append(row.values)

    true_pos_df = pd.DataFrame(true_pos_list, columns=combined_df.columns)
    false_pos_df = pd.DataFrame(false_pos_list, columns=combined_df.columns)
    true_pos_target_df = pd.DataFrame(true_pos_target_list, columns=target_df.columns).drop_duplicates(ignore_index=True)

    return true_pos_df, true_pos_target_df, false_pos_df, false_neg_df


def prediction_is_benign(malignant_def, row, dl_th=0.5, rmics_th=0.5, avg_th=0.5):
    """Determine if a lesion is benign based on the specified malignant definition."""
    
    match malignant_def:  # Determine malignancy based on the specified definition
        case 'dl_label':   # Predicted DL label indicates malignancy
            return row['predlabel'] % 2 == 0
        case 'dl_score':    # Predicted DL score indicates malignancy
            return row['scoreben'] > dl_th
        case 'rmics_score': # RMICS score indicates malignancy
            return row['scoreben rmics'] > rmics_th
        case 'minor_vote':  # Either score indicates malignancy
            return (row['scoreben'] > dl_th and row['scoreben rmics'] > rmics_th)
        case 'major_vote':  # Both scores indicate malignancy
            return (row['scoreben'] > dl_th or row['scoreben rmics'] > rmics_th)
        case 'average':     # Average score indicates malignancy
            return row['scoreben avg'] > avg_th
        case _:             # Default case if no match
            return False
        

def classified_regions(combined_df, malignant_def='dl_label', iou_th=0.1, score_th=0.1):
    """Classify regions in combined DataFrame into true positives, false positives, etc."""
    
    true_pos_list, false_pos_list, true_neg_list, false_neg_list = [], [], [], []

    for idx, row in combined_df.iterrows():  # Iterate through each row in the DataFrame
        if row['iou'] > iou_th and row['score'] > score_th:  # Correctly detected region
            if prediction_is_benign(malignant_def, row):  # Classified as benign
                if row['targetlabel'] % 2 == 0:  # Target is benign
                    true_neg_list.append(row.values)  # True negative
                else:  # Target is malignant
                    false_neg_list.append(row.values)  # False negative
            else:  # Classified as malignant
                if row['targetlabel'] % 2 == 0:  # Target is benign
                    false_pos_list.append(row.values)  # False positive
                else:  # Target is malignant
                    true_pos_list.append(row.values)  # True positive
        elif row['score'] > score_th:  # Incorrectly detected region
            row['targetlabel'] = 0  # Classify as benign
            if row['predlabel'] % 2 == 0:  # Classified as benign
                true_neg_list.append(row.values)  # True negative
            else:  # Classified as malignant
                false_pos_list.append(row.values)  # False positive

    # Convert lists to DataFrames
    true_pos_df = pd.DataFrame(true_pos_list, columns=combined_df.columns)
    false_pos_df = pd.DataFrame(false_pos_list, columns=combined_df.columns)
    false_neg_df = pd.DataFrame(false_neg_list, columns=combined_df.columns)
    true_neg_df = pd.DataFrame(true_neg_list, columns=combined_df.columns)

    return true_pos_df, false_pos_df, true_neg_df, false_neg_df


def compute_sensitivity(true_pos, false_neg):
    return len(true_pos) / (len(true_pos) + len(false_neg)) if len(true_pos) + len(false_neg) > 0 else float('NaN')


def compute_precision(true_pos, false_pos):
    return len(true_pos) / (len(true_pos) + len(false_pos)) if len(true_pos) + len(false_pos) > 0 else float('NaN')


def compute_specificity(true_neg, false_pos):
    return len(true_neg) / (len(true_neg) + len(false_pos)) if len(true_neg) + len(false_pos) > 0 else float('NaN')


def compute_f1(true_pos, false_pos, false_neg):
    return 2 * len(true_pos) / (2 * len(true_pos) + len(false_pos) + len(false_neg))


def youden(true_pos, false_pos, false_neg, true_neg):
    return (len(true_pos) / (len(true_pos) + len(false_neg))) + \
           (len(true_neg) / (len(true_neg) + len(false_pos))) - 1


def auc_classification(all_preds, name_list, malignant_def='dl_label', plot_figures=False):
    """Calculate AUC for different classification methods, optionally plotting ROC curves."""
    
    if plot_figures:
        plt.figure()  # Initialize figure for plotting if needed

    auc_list = []

    # Loop through each prediction DataFrame and its corresponding name
    for pred_df, name in zip(all_preds, name_list):
        
        # Convert target labels to binary: 0 for benign, 1 for malignant
        target_labels = [0 if x % 2 == 0 else 1 for x in pred_df['targetlabel']]

        # Determine predicted probabilities based on the definition of malignancy
        match malignant_def:
            case 'dl_label':   # Use the predicted DL label
                pred_probs = [x % 2 * y + (x + 1) % 2 * (1 - y) for x, y in zip(pred_df['predlabel'], pred_df['scoremal'])]
            case 'dl_score':   # Use the predicted DL score
                pred_probs = [x % 2 * y + (x + 1) % 2 * (1 - y) for x, y in zip(pred_df['predlabel'], pred_df['scoremal'])]
            case 'rmics_score':  # Use the predicted rmics score
                pred_probs = [x % 2 * y + (x + 1) % 2 * (1 - y) for x, y in zip(pred_df['predlabel'], pred_df['scoremal rmics'])]
            case 'minor_vote':  # Use the higher of DL or rmics score (OR condition)
                pred_probs = [x % 2 * max(y, z) + (x + 1) % 2 * (1 - max(y, z)) for x, y, z in zip(pred_df['predlabel'], pred_df['scoremal'], pred_df['scoremal rmics'])]
            case 'major_vote':  # Use the lower of DL or rmics score (AND condition)
                pred_probs = [x % 2 * min(y, z) + (x + 1) % 2 * (1 - min(y, z)) for x, y, z in zip(pred_df['predlabel'], pred_df['scoremal'], pred_df['scoremal rmics'])]
            case 'average':  # Use the average of DL and rmics scores
                pred_probs = [x % 2 * y + (x + 1) % 2 * (1 - y) for x, y in zip(pred_df['predlabel'], pred_df['scoremal avg'])]
            case _:  # Default to zero probabilities if no valid case is matched
                pred_probs = [0] * len(target_labels)

        # Calculate AUC if there are at least two unique target labels
        if len(np.unique(target_labels)) > 1:
            auc_val = roc_auc_score(target_labels, pred_probs)
        else:
            auc_val = float('NaN')  # AUC undefined if only one class present
        auc_list.append(auc_val)

        # Plot ROC curve if requested
        if plot_figures:
            fpr, tpr, _ = roc_curve(target_labels, pred_probs)
            plt.plot(fpr, tpr, label=f"{name}, AUC={round(auc_val, 4)}")

    # Finalize and show plot if needed
    if plot_figures:
        plt.legend([f"{x}, AUC={round(y, 4)}" for x, y in zip(name_list, auc_list)])
        plt.show()

    return auc_list

#%% Read targets and predictions

dataset = 'MUMC'

# for MUMC
if dataset == 'MUMC' :
    targetpath = r'' # directory of the target file
    targetfile = '' # name of the target file

    predpath = r'' # directory of the prediction file
    predfile = '' # name of the DL prediction file
    rmicsfile = '' # name of the radiomics prediction file

# for GR
elif dataset == 'GR' :
    targetpath = r'' # directory of the target file
    targetfile = '' # name of the target file

    predpath = r'' # directory of the prediction file
    predfile = '' # name of the DL prediction file
    rmicsfile = '' # name of the radiomics prediction file

# Read in target and prediction files
target_df = pd.read_json(os.path.join(targetpath,targetfile))
pred_df = pd.read_json(os.path.join(predpath,predfile))
rmics_df = pd.read_excel(os.path.join(predpath, rmicsfile))

# Combine all unique predictions
targetout_df = add_patientname_df(target_df, 'filename')
rmicsout_df = add_patientname_df(rmics_df, 'mask path')
predout_df = unique_predictions(pred_df, target_df, max_predictions_breast=10)
targetunique_df = unique_targets(targetout_df)

pred_all_df = combine_df_rmics(predout_df, rmicsout_df)

# Define criteria to assess malignancy
lesion_is_malignant = 'dl_label'

#%% Detection Metrics

def compute_detection_metrics(pred_df, target_df, lesion_is_malignant, iou_th=0.1, score_th=0.1):
    """Compute detection metrics including sensitivity and precision."""
    
    # Detect true positives, false positives, and false negatives
    det_tp_df, det_tp_target_df, det_fp_df, det_fn_df = detected_regions(
        pred_df, target_df, lesion_is_malignant, iou_th=iou_th, score_th=score_th
    )

    # Calculate sensitivity for various categories
    sensitivities = [
        compute_sensitivity(det_tp_target_df, det_fn_df),  # Overall sensitivity
        compute_sensitivity(det_tp_target_df[det_tp_target_df['labels'] % 2 == 0], 
                            det_fn_df[det_fn_df['labels'] % 2 == 0]),  # Benign sensitivity
        compute_sensitivity(det_tp_target_df[det_tp_target_df['labels'] % 2 == 1], 
                            det_fn_df[det_fn_df['labels'] % 2 == 1]),  # Malignant sensitivity
        compute_sensitivity(det_tp_target_df[det_tp_target_df['labels'] == 0], 
                            det_fn_df[det_fn_df['labels'] == 0]),  # Benign mass sensitivity
        compute_sensitivity(det_tp_target_df[det_tp_target_df['labels'] == 1], 
                            det_fn_df[det_fn_df['labels'] == 1]),  # Malignant mass sensitivity
        compute_sensitivity(det_tp_target_df[det_tp_target_df['labels'] == 2], 
                            det_fn_df[det_fn_df['labels'] == 2]),  # Benign calcification sensitivity
        compute_sensitivity(det_tp_target_df[det_tp_target_df['labels'] == 3], 
                            det_fn_df[det_fn_df['labels'] == 3]),  # Malignant calcification sensitivity
    ]

    # Calculate precision for various categories
    precisions = [
        compute_precision(det_tp_df, det_fp_df),  # Overall precision
        compute_precision(det_tp_df[det_tp_df['predlabel'] % 2 == 0], 
                         det_fp_df[det_fp_df['predlabel'] % 2 == 0]),  # Benign precision
        compute_precision(det_tp_df[det_tp_df['predlabel'] % 2 == 1], 
                         det_fp_df[det_fp_df['predlabel'] % 2 == 1]),  # Malignant precision
        compute_precision(det_tp_df[det_tp_df['predlabel'] == 0], 
                         det_fp_df[det_fp_df['predlabel'] == 0]),  # Benign mass precision
        compute_precision(det_tp_df[det_tp_df['predlabel'] == 1], 
                         det_fp_df[det_fp_df['predlabel'] == 1]),  # Malignant mass precision
        compute_precision(det_tp_df[det_tp_df['predlabel'] == 2], 
                         det_fp_df[det_fp_df['predlabel'] == 2]),  # Benign calcification precision
        compute_precision(det_tp_df[det_tp_df['predlabel'] == 3], 
                         det_fp_df[det_fp_df['predlabel'] == 3]),  # Malignant calcification precision
    ]

    return sensitivities, precisions

# Compute metrics
sens_list, prec_list = compute_detection_metrics(pred_all_df, targetunique_df, lesion_is_malignant, iou_th=0.1, score_th=0.1)

#%% Classification Metrics

def compute_classification_metrics(pred_df, dataset, lesion_is_malignant, iou_th=0.1, score_th=0.1, plot_figures=False):
    """Compute classification metrics and AUC values."""
    
    # Classify regions and get true positives, false positives, true negatives, and false negatives
    cl_tp_df, cl_fp_df, cl_tn_df, cl_fn_df = classified_regions(pred_df, lesion_is_malignant, iou_th=iou_th, score_th=score_th)

    # Combine classification results into one DataFrame
    all_class = pd.concat([cl_tp_df, cl_fp_df, cl_tn_df, cl_fn_df], ignore_index=True)
    all_class_mass = all_class[all_class['targetlabel'] <= 1]  # Mass lesions
    all_class_calc = all_class[all_class['targetlabel'] >= 2]  # Calcified lesions

    # Compute AUC based on the dataset type
    if dataset == 'MUMC':
        auc_val = auc_classification(
            [all_class, all_class_mass, all_class_calc], ['all', 'mass', 'calc'],
            lesion_is_malignant, plot_figures
        )
    elif dataset == 'GR':
        auc_val = auc_classification(
            [all_class], ['all'], lesion_is_malignant, plot_figures
        )
        
    return auc_val

# Compute classification metrics and AUC values
auc_list = compute_classification_metrics(pred_all_df, dataset, lesion_is_malignant, iou_th=0.1, score_th=0.1, plot_figures=False)
