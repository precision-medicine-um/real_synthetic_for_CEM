import os
import sys
import math
import pandas as pd
from PIL import Image

import torch
import monai.transforms as MT
import torchvision.transforms as T

import utils

from monai.utils.enums import GridSampleMode, GridSamplePadMode
from transforms import PILToTensor, ConvertImageDtype, RandomHorizontalFlip, Compose


def get_transform(train):
    """Return a composed list of transformations based on training mode."""

    transform_list = [PILToTensor(), ConvertImageDtype(torch.float)]
    if train:
        transform_list.append(RandomHorizontalFlip(0.5))

    return Compose(transform_list)


# Define MONAI transforms
monai_flip = MT.RandFlipd(["image", "mask"], spatial_axis=1, prob=0.5)
monai_affine = MT.RandAffined(
    ["image", "mask"],
    scale_range=([-0.1, 0.1], [-0.1, 0.1]),
    shear_range=([-0.1, 0.1], [-0.1, 0.1]),
    prob=1.0,
    mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
    padding_mode=GridSamplePadMode.BORDER,
    device=torch.device("cuda:0"),
)

def get_monai_transform(train=True):
    """Return MONAI transform pipeline based on mode (training vs validation)."""

    if train:
        return MT.Compose(transforms=[monai_flip, monai_affine])
    
    return MT.Compose()


class CEMDataset(torch.utils.data.Dataset):
    """Custom dataset class for image and mask loading and transformation."""
    
    def __init__(self, root, filename, transforms, device, use_bbox=False):
        self.root = root
        self.filename = filename
        self.transforms = transforms
        self.device = device
        self.annotations_file = pd.read_json(os.path.join(root, filename))
        self.use_bbox = use_bbox

    def __getitem__(self, idx):
        """Load and transform an image-mask pair with optional bounding box support."""
        
        # Create dictionary with loaded images
        init_dict = self.read_images(idx)

        # Apply transforms, if specified
        transformed_dict = self.transforms(init_dict)

        # Create the target dictionary
        try:
            target = self.create_target(transformed_dict)
            return transformed_dict["image"], target
        
        except:
            print('Used original image without transforms')
            target = self.create_target(init_dict)
            return init_dict["image"], target

    def read_images(self, idx):
        """Read images and masks from file and prepare initial dictionaries."""        
        img_path = self.annotations_file['filename'][idx]
        mask_path = self.annotations_file['path_mask'][idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image_id = torch.tensor([idx])
        patient_id = img_path.split('\\')[-1][:-4]
        label = self.annotations_file['labels'][idx] + 1  # 0 is background
        
        # Optional bounding box information
        if self.use_bbox:
            breast_box = self.annotations_file['breast box'][idx]

        # Transform to torch tensor
        read_transform = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float)])
        mask = read_transform(mask)
        image = read_transform(image)

        # Prepare the output dictionary
        if self.use_bbox:
            output_dict = {"image": torch.tensor(image).to(self.device), "mask": torch.tensor(mask).to(self.device),
                           "image_id": image_id, "patient_id": torch.tensor([ord(ch) for ch in patient_id]),
                           "label": label, "breast_box": breast_box}
        else:
            output_dict = {"image": image.clone().detach().to(self.device), "mask": mask.clone().detach().to(self.device),
                           "image_id": image_id, "patient_id": torch.tensor([ord(ch) for ch in patient_id]),
                           "label": label}

        return output_dict

    def create_target(self, input_dict):
        """Create the target dictionary containing bounding boxes and labels."""
        
        # Convert the mask to tensor
        tr_mask = input_dict["mask"]
        obj_ids = torch.unique(tr_mask)[1:]  # Exclude background (label 0)

        # Generate binary masks for each object
        output_masks = tr_mask == obj_ids[:, None, None]
        output_masks = torch.as_tensor(output_masks, dtype=torch.uint8)

        # Generate bounding boxes for each object
        num_objs = len(obj_ids)
        boxes = []
        for box_label in range(num_objs):
            bbox = MT.CropForeground(channel_indices=box_label).compute_bounding_box(output_masks)
            boxes.append([bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Calculate area of each bounding box
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Prepare labels and other target fields
        labels = torch.as_tensor([input_dict["label"]] * num_objs, dtype=torch.int64)
        if self.use_bbox:
            breast_box = torch.as_tensor(input_dict["breast_box"], dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Create target dictionary
        target = {"boxes": boxes, "labels": labels, "masks": output_masks,
                  "image_id": input_dict["image_id"], "area": area,
                  "iscrowd": iscrowd, "patient_id": input_dict["patient_id"]}
        if self.use_bbox:
            target["breast_box"] = breast_box

        return target

    def __len__(self):
        """Return the total number of samples."""
        return len(self.annotations_file['filename'])


def train_one_epoch(model, optimizer, train_data_loader, val_data_loader, writer, device, epoch, print_freq, scaler=None):
    """Train the model for one epoch with optional gradient scaling and validation."""
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # Set up learning rate scheduler for the first epoch
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    # Loop over all images and targets in training data loader
    for images, targets in metric_logger.log_every(train_data_loader, print_freq, header):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Perform forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Reduce losses for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        # Stop training if loss is infinite
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # Backward pass and optimization
        optimizer.zero_grad()
        if scaler:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        # Update the learning rate
        if lr_scheduler:
            lr_scheduler.step()

        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])        
        writer.add_scalar('Loss/train', metric_logger.loss.value, epoch)
        writer.add_scalar('Learning rate/train', metric_logger.lr.value, epoch)

    # Validate the model and log validation metrics
    metric_logger_val = utils.MetricLogger(delimiter="  ")
    metric_logger_val.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # Loop over all images and targets in validation data loader
    for images_val, targets_val in metric_logger_val.log_every(val_data_loader, print_freq, header):
        images_val = [image.to(device) for image in images_val]
        targets_val = [{k: v.to(device) for k, v in t.items()} for t in targets_val]

        # Perform forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict_val = model(images_val, targets_val)

        # Reduce losses for logging
        loss_dict_reduced_val = utils.reduce_dict(loss_dict_val)
        losses_reduced_val = sum(loss for loss in loss_dict_reduced_val.values())

        # Log metrics to TensorBoard
        metric_logger_val.update(loss=losses_reduced_val, **loss_dict_reduced_val)
        writer.add_scalar('Loss/val', metric_logger_val.loss.value, epoch)

    return metric_logger, metric_logger_val
