import os
import numpy as np
import pandas as pd

from utils import MetricLogger
from collections import defaultdict

import torch
from torchvision.ops import nms
from torchvision.io import write_png


def move_to(obj, device):
    """Recursively move tensors, dicts, or lists to a specific device (GPU/CPU)."""
    
    if torch.is_tensor(obj):
        return obj.to(device)
    
    # Recursively move dict values to the device
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    
    # Recursively move list elements to the device
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    
    # Raise an error for unsupported types
    else:
        raise TypeError("Invalid type for move_to")


def apply_nms(predictions):
    """Apply Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes."""
    predictions_nms = []

    # Loop over all images in the batch
    for pred_idx in range(len(predictions)):

        # Apply NMS to eliminate overlapping boxes
        if len(predictions[pred_idx]['boxes']) > 0:            
            save_pred = nms(
                torch.cat([torch.unsqueeze(box, dim=0) for box in predictions[pred_idx]['boxes']], dim=0),
                torch.cat([torch.unsqueeze(box, dim=0) for box in predictions[pred_idx]['scores']], dim=0),
                0.2
            )

        # Handle case when no boxes are predicted
        else:
            save_pred = []

        print('Len after nms', len(save_pred))

        # Create a dict of all the bounding boxes to keep after NMS
        predictions_nms_idx = defaultdict(list)
        for idx in range(len(predictions[pred_idx]['boxes'])):
            if idx in save_pred:
                for key in predictions[pred_idx].keys():
                    predictions_nms_idx[key].append(predictions[pred_idx][key][idx])

        # Handle cases when no predictions are left after NMS
        if len(save_pred) == 0:
            predictions_nms_idx = predictions[pred_idx].copy()

        predictions_nms.append(predictions_nms_idx)  # Append the final NMS filtered predictions

    return predictions_nms


def compute_iou_box(pred_box, target_box):
    """Compute Intersection over Union (IoU) for bounding boxes."""

    # Calculate area of prediction and target boxes
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])

    # Find the coordinates of the intersection box
    xs = max(target_box[0], pred_box[0])
    ys = max(target_box[1], pred_box[1])
    xe = min(target_box[2], pred_box[2])
    ye = min(target_box[3], pred_box[3])

    # Compute intersection area
    if xe - xs > 0 and ye - ys > 0:
        intersection_area = (xe - xs) * (ye - ys)
    else:
        intersection_area = -1

    # Compute union area
    union_area = target_area + pred_area - intersection_area

    print('Compute IoU', pred_box, pred_area, target_box, target_area, intersection_area, union_area)

    # Return IoU value
    if union_area > 1e-7:
        return torch.tensor(intersection_area / union_area)
    else:
        return torch.tensor(0.0)


def compute_iou_segm(pred_mask, target_mask, pred_box, target_box):
    """Compute IoU for segmentation masks"""

    pred_area = torch.count_nonzero(pred_mask)
    target_area = torch.count_nonzero(target_mask)

    intersection_area = torch.sum(torch.logical_and(pred_mask, target_mask))  # Area of overlap
    union_area = target_area + pred_area - intersection_area  # Union of both areas

    if union_area > 1e-7:
        return torch.tensor(intersection_area / union_area)
    else:
        return torch.tensor(0.0)


def compute_dice_segm(pred_mask, target_mask):
    """Compute Dice coefficient (2*TP / (2*TP + FP + FN))."""

    true_pos = torch.count_nonzero(pred_mask * target_mask)  # Count true positives
    dice = 2 * true_pos / (torch.count_nonzero(pred_mask) + torch.count_nonzero(target_mask))  # Dice score formula

    return torch.tensor(dice)


def evaluateCEM(model, model_name, data_loader, savedir, malignant_only=False, use_segmentation=True, model_device='cpu'):
    """Evaluate the model on the test dataset, applying NMS and calculating metrics."""

    # Create directory for saving results
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        print('Created path', savedir)
    savedirmodel = os.path.join(savedir, 'outcome_to_png')
    if not os.path.exists(savedirmodel):
        os.mkdir(savedirmodel)
        print('Created path', savedirmodel)

    output_outcomes = f'outcomes_{model_name}.json'

    df = pd.DataFrame(columns=['filename', 'x1', 'x2', 'y1', 'y2', 'iou', 'dice', 'scores', 'labels', 'path_mask', 'breast_box'])

    # Set model to evaluation mode
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # Process batches from the test dataset
    batch_idx = 0
    for test_images, targets in metric_logger.log_every(data_loader, 10, header):

        # Move images to the appropriate device (GPU/CPU)
        images_cuda = []
        for image in test_images:
            images_cuda.append(image.to(model_device))

        # Generate predictions with the model
        predictions_cuda = []
        with torch.no_grad():
            out = model(images_cuda)  # Get model output without calculating gradients
            predictions_cuda.append(out)

        # Apply non-max suppression to filter overlapping boxes
        predictions_nms = apply_nms(predictions_cuda[0])

        # Process each prediction in the batch
        for pred_idx, pred in enumerate(predictions_nms):
            target_id = targets[pred_idx]['patient_id']
            target_id = ''.join([chr(v) for v in target_id])  # Convert patient ID to string
            target_breast_box = targets[pred_idx]["breast_box"].tolist()  # Get breast bounding box
            print('Target', target_id, len(targets[pred_idx]['boxes']))
            print('Predictions', len(pred['boxes']))

            x1, y1, x2, y2, final_ious, final_dices, final_scores, final_labels, final_probs = [], [], [], [], [], [], [], []
            final_mask = torch.zeros_like(targets[pred_idx]['masks'][0], dtype=torch.uint8)  # Initialize final mask
            final_mask_multichannel = []
            print('Mask shape', final_mask.shape)

            # Loop over all target boxes
            for target_idx in range(len(targets[pred_idx]['boxes'])):

                target_box = targets[pred_idx]['boxes'][target_idx]
                target_mask = targets[pred_idx]['masks'][target_idx]
                print('Target mask', torch.count_nonzero((target_mask)))

                target_label = targets[pred_idx]['labels'][target_idx]  # Get target label
                if malignant_only and target_label == 1:
                    target_label = 2
                elif malignant_only and target_label == 2:
                    target_label = 4

                # Loop over all predicted boxes
                for box_idx, pred_box in enumerate(pred['boxes']):
                    pred_mask = pred['masks'][box_idx]
                    pred_mask[pred_mask >= 0.5] = 1  # Binarize mask predictions
                    pred_mask[pred_mask < 1] = 0
                    print('Pred mask', torch.count_nonzero(pred_mask))

                    # Compute IoU for segmentation or bounding box
                    if use_segmentation:
                        iou = compute_iou_segm(pred_mask, target_mask, pred_box, target_box)
                    else:
                        iou = compute_iou_box(pred_box, target_box)

                    print(box_idx, 'IoU', iou, 'Score', pred['scores'][box_idx],  'Pred box', pred_box, 'Target box', target_box)

                    dice = compute_dice_segm(pred_mask, target_mask)  # Compute Dice score

                    pred_label = pred['labels'][box_idx]
                    if malignant_only and pred_label == 1:
                        pred_label = 2
                    elif malignant_only and pred_label == 2:
                        pred_label = 4

                   # Save predictions for each box
                    x1.append(int(pred_box[0].item()))
                    y1.append(int(pred_box[1].item()))
                    x2.append(int(pred_box[2].item()))
                    y2.append(int(pred_box[3].item()))
                    final_ious.append(iou.item())
                    final_dices.append(dice.item())
                    final_scores.append(pred['scores'][box_idx].item())
                    final_labels.append(pred_label.item())

                    # Update the final mask
                    if target_idx == 0:
                        final_mask = torch.add(final_mask, torch.mul(pred_mask, (box_idx + 1)).to(torch.uint8))
                        final_mask_multichannel.append(pred_mask.to('cpu').detach().numpy().astype(np.uint8))


                # Handle case where no predictions are made
                if len(pred['boxes']) == 0 and target_idx == 0:
                    final_mask = torch.add(final_mask, torch.unsqueeze(final_mask, 0).to(torch.uint8))

            print('Mask size', final_mask.shape, torch.unique(final_mask, return_counts=True))

            # Save the final mask as PNG and NPZ files
            savename_png = os.path.join(savedirmodel, f'{target_id}.png')
            savename = os.path.join(savedirmodel, f'{target_id}.npz')
            write_png(final_mask.to('cpu'), savename_png)
            np.savez_compressed(savename, final_mask_multichannel)

            # Save results in the DataFrame
            list_row = [target_id, x1, x2, y1, y2, final_ious, final_dices, final_scores, final_labels, savename, target_breast_box]
            print(list_row)
            df.loc[len(df)] = list_row


        # Clear memory and reset GPU cache
        del images_cuda, predictions_cuda, predictions_nms
        torch.cuda.empty_cache()

        batch_idx += 1

    # Save results to a JSON file
    df.to_json(os.path.join(savedir, output_outcomes))

    return metric_logger, df