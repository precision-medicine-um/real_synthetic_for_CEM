import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from skimage.measure import regionprops
from skimage.filters import threshold_otsu

from utils_readwrite import read_dicom
from process_masks import *


def breast_mask(image):
    """Creates and returns a breast mask using Otsu thresholding."""
    
    # Apply Otsu threshold to separate background and breast tissue
    otsu_val = threshold_otsu(image)
    
    # Create a binary mask where the breast region is below the threshold
    otsu_array = image < otsu_val
    
    # Find contours in the binary mask
    invert_otsu = (np.ones(otsu_array.shape) - otsu_array).astype(np.uint8)
    (contours, _) = cv2.findContours(invert_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find the largest contour, assumed to be the breast
    max_ctr = contours[0]
    for ctr in contours:
        if cv2.contourArea(ctr) > cv2.contourArea(max_ctr):
            max_ctr = ctr
            
    # Create a new mask based on the largest contour (breast region)
    otsu_mask = np.zeros(invert_otsu.shape)
    cv2.fillPoly(otsu_mask, [max_ctr], [1])

    return otsu_mask


def resample_intensities(orig_img, bin_nr=256):
    """Resamples image intensity values into a smaller range of bins."""
    v_count = 0
    filtered = orig_img.copy()

    # Ensure all intensity values are positive
    if np.min(orig_img.flatten()) < 0:
        filtered += np.min(orig_img.flatten())
        
    # Resample intensity values based on step size calculated from min and max values
    resampled = np.zeros_like(filtered)
    max_val_img = np.max(filtered.flatten())
    min_val_img = np.min(filtered.flatten())
    step = (max_val_img - min_val_img) / bin_nr

    # Assign values to new intensity bins
    for st in np.arange(step + min_val_img, max_val_img + step, step):
        resampled[(filtered <= st) & (filtered >= st - step)] = v_count
        v_count += 1
    
    return np.array(resampled, dtype=np.uint16)


def img_to_8bit(img):
    """Convert an image to 8-bit by adjusting intensity range and resampling if necessary."""
    temp_img = img.copy()
    
    # Set lower and upper thresholds based on quantiles
    low_thr = np.quantile(temp_img[temp_img > 0], 0.01)
    high_thr = np.quantile(temp_img[temp_img > 0], 0.99)

    # Adjust image values below and above thresholds
    temp_img[temp_img < low_thr] = low_thr
    temp_img[temp_img > high_thr] = high_thr

    # If too many unique values, resample intensities
    if len(np.unique(temp_img[temp_img > 0])) > 256:
        temp_img_sampled = resample_intensities(temp_img[temp_img > 0])
        temp_img[temp_img > 0] = temp_img_sampled    
    else:
        # Rescale intensities between 0 and 255
        new_img = (temp_img - np.min(temp_img)) / (np.max(temp_img) - np.min(temp_img)) 
        temp_img = (new_img * 255).astype(np.uint8)   
        
    return temp_img


def crop_img(temp_img_re_original, temp_img_le_original):
    """Crop both recombined and low-energy breast images based on detected breast region."""
    
    # Create a breast mask and apply it to both images
    otsu_array = breast_mask(temp_img_le_original)
    temp_img_re = otsu_array * temp_img_re_original
    temp_img_le = otsu_array * temp_img_le_original
    
    # Get bounding box of the breast region
    props = regionprops(np.array(temp_img_re > 0, np.uint8))
    r0, c0, r1, c1 = props[0].bbox
    
    # Crop and convert both images to 8-bit format
    temp_img_re = img_to_8bit(temp_img_re[r0:r1, c0:c1])
    temp_img_le = img_to_8bit(temp_img_le[r0:r1, c0:c1])
    
    return temp_img_re, temp_img_le, [r0, r1, c0, c1]


def preprocessing(imagepath, lename, rcname, dataset, preptype='calcification'):
    """Apply preprocessing steps to breast images: segmentation, cropping, CLAHE filtering, and merging."""
    
    print(dataset)     

    # Load low-energy and recombined images
    image_le = read_dicom(imagepath, lename).pixel_array
    image_rc = read_dicom(imagepath, rcname).pixel_array

    # Crop the images
    img_re, img_le, box_breast = crop_img(image_rc, image_le)

    # Apply CLAHE filtering to enhance image contrast
    clahe_filt = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    clahe_orig = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))

    new_img_le = (img_le - np.min(img_le)) / (np.max(img_le) - np.min(img_le))
    new_img_le = (new_img_le * 255).astype(np.uint8)
    x_le_orig = clahe_orig.apply(new_img_le).astype(np.uint8)
    x_le_filt = clahe_filt.apply(new_img_le).astype(np.uint8)

    new_img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    new_img_re = (new_img_re * 255).astype(np.uint8)
    x_re_orig = clahe_orig.apply(new_img_re).astype(np.uint8)
    x_re_filt = clahe_filt.apply(new_img_re).astype(np.uint8)

    # Merge the filtered images depending on the type of preprocessing
    if preptype == 'mass':
        merged_img = Image.fromarray(cv2.merge((x_le_filt, x_re_orig, x_re_filt)))
    elif preptype == 'calcification':
        merged_img = Image.fromarray(cv2.merge((x_le_filt, x_le_orig, x_re_filt)))
    else:
        raise ValueError('Possible types of preprocessing are "mass" and "calcification".')

    # Convert to RGB and return the result
    img_rgb = merged_img.convert("RGB")
    img_breast = img_rgb

    return img_breast, box_breast


def allprep(inputdir, outputdir, input_sheet, df, subset, startidx, view, dataset, preptype):

    # Initialize list to store patient information as tuple (patient name, laterality, label)
    list_patients = []

    # Processing for the GR dataset
    if dataset == 'GR' :
        # Add patient information based on the selected view (CC or MLO)
        for idx, row in enumerate(input_sheet) :
            patient_label = row[f'Class {view}'] if isinstance(row[f'Class {view}'], int) else int(row[f'Class {view}']) # Ensure label is integer
            list_patients.append((row['Patient Number '], row[f'Breast {view}'], patient_label))

    # Processing for MUMC and synthetic datasets with lesions
    elif dataset == 'MUMC' or dataset == 'synthetic':
        # Add patient information based on the selected view (CC or MLO) and only if the bounding box exists
        for idx, row in enumerate(input_sheet) :
            patient_label = row[f'Class {view}']
            # Ensure label is numeric
            if patient_label == 'benign':
                patient_label = '0'
            if patient_label == 'malignant':
                patient_label = '1'
            list_patients.append((row['PatientID'], row[f'Breast {view}'], patient_label))     

    # Processing for MUMCnormal dataset
    elif dataset == 'MUMCnormal':
        # No delineation exists for normal cases; both left and right breasts are added with label 0
        for idx, row in enumerate(input_sheet) :
            list_patients.append((row['PatientID'], 'L', 0))
            list_patients.append((row['PatientID'], 'R', 0))

    else:
        # If dataset name is not recognized, return None
        print("Invalid dataset name entered.")
        return None

    # Loop through patient directories from the start index
    for idx, patient in enumerate(list_patients[startidx:]) :
        patient_dir, patient_laterality, patient_label = patient

        # Extract a unique patient name
        if dataset == 'GR':
            patient_dir = patient_dir.replace(" ", "")
        else:
            midx = patient_dir.find('MUMC')
            patient_dir = patient_dir[midx:midx+9]

        # Print patient information and prepare path for the current view
        print(idx, startidx + idx, patient_dir, patient_laterality, patient_label)
        view_dir = os.path.join(inputdir, patient_dir, f'{patient_laterality}_{view}')
        print('Viewdir', view_dir)

        # Initialize variables to track found files
        allfound = 0

        # Loop through files in the view directory to find necessary images and masks
        if os.path.exists(view_dir):
            for file in os.listdir(view_dir):
                print('File', file)
                if 'LOW' in file and not 'mask' in file and not 'MASK' in file and '.dcm' in file and not 'CIRCLE' in file:
                    print('LE found')
                    le_name = file
                    allfound += 1
                elif 'REC' in file and not 'mask' in file and '.dcm' in file:
                    print('RC found')
                    rc_name = file   
                    allfound += 1
                elif 'STRUCT' in file and '.mha' in file and dataset == 'MUMC':
                    print('Mask found')
                    mask_name = file
                    allfound += 1
                elif 'STRUCT' in file and '.dcm' in file and dataset == 'GR':
                    print('Mask found')
                    mask_name = file
                    allfound += 1
                elif ('mask' in file or 'MASK' in file) and '.dcm' in file and dataset == 'synthetic':
                    print('Mask found')
                    mask_name = file
                    allfound += 1

            # No mask can be found for an image without a lesion
            if dataset == 'MUMCnormal':
                allfound += 1

        # If all required files are found and not processing normal MUMC cases
        if allfound == 3:
            # Apply preprocessing to the image
            bg, bbox = preprocessing(view_dir, le_name, rc_name, dataset=dataset, preptype=preptype)

            # Save the mask if it's delineated (non-zero mask)
            output_name_mask = os.path.join(outputdir, f'mask_{subset}', f'{patient_dir}_{patient_laterality}_{view}.png')
            print('Out mask', output_name_mask)

            # Locate bounding box and save mask box for different datasets
            if dataset != 'MUMCnormal':            
                if dataset == 'MUMC':
                    x1, x2, y1, y2 = loc_mask_mha(view_dir, mask_name, output_name_mask, bbox)
                elif dataset == 'synthetic':
                    x1, x2, y1, y2 = loc_mask_dicom(view_dir, mask_name, output_name_mask, bbox)
                elif dataset == 'GR':
                    x1, x2, y1, y2 = loc_mask_gr(view_dir, mask_name, le_name, output_name_mask, bbox)

            # Save preprocessed image if mask is valid or no lesion is present
            if x1[0] > 0 or dataset == 'MUMCnormal':
                im = Image.fromarray(bg)
                output_name_image = os.path.join(outputdir, f'colored_{subset}', f'{patient_dir}_{patient_laterality}_{view}.png')
                print('Out image', output_name_image)
                im.save(output_name_image, quality=100)

                # Add information to the dataframe
                list_row = [output_name_image, x1, x2, y1, y2, patient_label, output_name_mask, bbox]
                print(list_row)
                df.loc[len(df)] = list_row

    return df

