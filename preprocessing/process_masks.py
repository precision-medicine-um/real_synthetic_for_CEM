import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from skimage.measure import label
from utils_readwrite import read_dicom, read_mha


def coord2pixels(maskpath, maskname):
    """Convert contour data from a DICOM mask to pixel coordinates."""
    mask = read_dicom(maskpath, maskname)  # Read the DICOM mask file.
    pixel_coords = []

    # Define the physical distance between pixel centers.
    x_spacing, y_spacing = 0.1, 0.1
    
    # Extract contour sequences from the mask.
    ctrs = mask.ROIContourSequence
    for ctr_idx, ctr in enumerate(ctrs):
        print(ctr_idx)
        if len(ctr.ContourSequence) > 0:
            contour_coord = ctr.ContourSequence[0].ContourData
            coord = [(contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]) 
                     for i in range(0, len(contour_coord), 3)]

            # Convert 3D coordinates to 2D pixel coordinates.
            pixel_coords.append([(np.sqrt(x**2 + y**2) / x_spacing, -z / y_spacing) for x, y, z in coord])
        
    return pixel_coords


def poly_to_mask(polygon, width, height):
    """Convert polygon to a binary mask."""

    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)

    return mask


def plot2dcontour(img_arr, contour_arr, figsize=(20, 20)):
    """Display the 2D image with contours overlay."""
    masked_contour_arr = np.ma.masked_where(contour_arr == 0, contour_arr)

    plt.figure(figsize=figsize)
    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    # Show the original image with contours overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr, cmap='gray', interpolation='none')
    plt.imshow(masked_contour_arr, cmap='cool', interpolation='none', alpha=0.7)
    plt.show()


def find_mask_region(array, savemask=False, savepath=''):
    """Find and return bounding boxes of all connected regions in the mask."""
    xmins, xmaxs, ymins, ymaxs = []

    # Label connected regions in the mask.
    labeled_mask = label(array, connectivity=2)
    if np.max(labeled_mask) > 0:
        # Define a bounding box for each unique label
        for label_idx in np.unique(labeled_mask)[1:]:
            unique_labeled_mask = (labeled_mask == label_idx).astype(int)
            label_nz = np.nonzero(unique_labeled_mask)
            xmin, xmax = np.min(label_nz[0]), np.max(label_nz[0])
            ymin, ymax = np.min(label_nz[1]), np.max(label_nz[1])
            
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)

        # Save the labeled mask
        if savemask:
            PIL_mask = Image.fromarray(np.uint8(labeled_mask))
            PIL_mask.save(savepath)

        return xmins, xmaxs, ymins, ymaxs
    
    # Return -1 if no regions found.
    return [-1], [-1], [-1], [-1]


def loc_mask_dicom(inputpath, maskname, outputname, bbox=[0, -1, 0, -1]):
    """Locate mask in a DICOM file and return bounding boxes."""

    mask = read_dicom(inputpath, maskname)
    mask_array = mask.pixel_array[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    return find_mask_region(mask_array, savemask=True, savepath=outputname)


def loc_mask_mha(inputpath, maskname, outputname, bbox=[0, -1, 0, -1]):
    """Locate mask in a MHA file and return bounding boxes."""

    mask_array = read_mha(inputpath, maskname)
    mask_array = mask_array[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    return find_mask_region(mask_array, savemask=True, savepath=outputname)


def loc_mask_gr(inputpath, maskname, imagename, outputname, bbox=[0, -1, 0, -1]):
    """Locate mask in GR dataset and return bounding boxes."""
    image = read_dicom(inputpath, imagename)
    mask_array = np.zeros(image.pixel_array.shape)
     
    # Create a mask form the polygon coordinates
    mask_coords = coord2pixels(inputpath, maskname)  
    for coords in mask_coords:
        coord_mask = poly_to_mask(coords, mask_array.shape[1], mask_array.shape[0])
        mask_array[coord_mask > 0] = 1

    #  Crop the mask using the bounding box.
    mask_array = mask_array[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    return find_mask_region(mask_array, savemask=True, savepath=outputname)

