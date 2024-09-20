from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data_generator import CEMDataset, get_monai_transform
from testparameters import *
from predictionsMaskRCNN import evaluateCEM

import os
import torch
import utils
import torchvision


def get_model_instance_segmentation(num_classes):
    """
    Load a pre-trained Mask R-CNN model and modify the box and mask heads 
    to match the number of classes in the dataset.
    """
    
    # Load a Mask R-CNN model pre-trained on COCO dataset
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Modify the box predictor (for object detection) to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Modify the mask predictor to match the number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # Use 256 hidden units for the mask head
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# Set environment variables for GPU access (based on PCI bus order and device ID).
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = num_device

# Define the device (use GPU if available, otherwise CPU).
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

# Create the test dataset using custom dataset class CEMDataset and appropriate transformations.
dataset_test = CEMDataset(testdir, testfile, get_monai_transform(False), device, use_bbox=True)

# Define the test data loader for batching, no shuffling since this is evaluation.
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=test_bs, shuffle=False, collate_fn=utils.collate_fn)

# Load the instance segmentation model with the specified number of classes.
model = get_model_instance_segmentation(num_classes)

# Load the trained model checkpoint (saved during training).
checkpoint = torch.load(os.path.join(modeldir, modelname))

# Load the model's saved state dictionary (parameters) from the checkpoint.
model.load_state_dict(checkpoint['model_state_dict'])

# Move the model to the defined device (GPU or CPU).
model.to(device)

# Evaluate the model on the test dataset and generate predictions.
# This uses the evaluateCEM function with custom options for segmentation and device settings.
_, pred_df = evaluateCEM(model, modelname, data_loader_test, savedir,
                         malignant_only=False, use_segmentation=True, model_device=device)
