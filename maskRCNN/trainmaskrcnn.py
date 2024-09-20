import os
import utils

import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trainparameters import *
from data_generator import CEMDataset, train_one_epoch, get_monai_transform


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


# Set environment variables for CUDA (GPU) configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = num_device  # Set the visible GPU devices

# Define the device to use (GPU if available, otherwise CPU)
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

# Print the device name and its CUDA status
print(torch.cuda.current_device(), torch.cuda.get_device_name(), device)

# Load training and test datasets with the appropriate transformations
dataset = CEMDataset(traindir, trainfile, get_monai_transform(True), device)
dataset_test = CEMDataset(testdir, testfile, get_monai_transform(False), device)

# Create data loaders for training and validation datasets
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=train_bs, shuffle=True, collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=test_bs, shuffle=False, collate_fn=utils.collate_fn
)

# Initialize the Mask R-CNN model with the specified number of classes
model = get_model_instance_segmentation(num_classes)

# Move the model to the device (GPU/CPU)
model.to(device)

# Check if the model parameters are on the GPU (CUDA)
print('On cuda?', next(model.parameters()).is_cuda)

# Set up the optimizer to update model weights (Adam optimizer is used)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-5)  # Adam optimizer with learning rate 1e-5

# Set up the learning rate scheduler to reduce LR based on validation loss
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=2, min_lr=1e-7, eps=0.0001
)

# Initialize TensorBoard writer to log training metrics
model_writer = SummaryWriter(os.path.join(boarddir, modelname))

# Set the number of epochs for training
num_epochs = 30

# Training loop: iterate over the number of epochs
for epoch in range(num_epochs):
    """
    Train the model for one epoch and evaluate on the validation set.
    Log metrics for TensorBoard and save the model periodically.
    """
    # Train for one epoch and get the loss and validation loss
    _, _, epoch_loss, epoch_loss_val = train_one_epoch(
        model, optimizer, data_loader, data_loader_test, model_writer, device, epoch, print_freq=50
    )
    print('Finished with training one epoch')

    # Update the learning rate based on validation loss
    lr_scheduler.step(epoch_loss_val)

    # Save the model every 2 epochs, or during specific intervals between 5 and 20 epochs
    if epoch % 2 == 0 or (epoch > 5 and epoch < 20):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'val_loss': epoch_loss_val,
        }, os.path.join(modeldir, modelname + '_' + str(epoch) + '.pth'))

    # Clean up to free memory
    del epoch_loss, epoch_loss_val

# End of training
print("That's it!")
