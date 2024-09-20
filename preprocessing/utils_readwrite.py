import os
import pydicom
import numpy as np
import SimpleITK as sitk

from roifile import ImagejRoi
from skimage.measure import find_contours


def read_dicom(imagepath, imagename, force=False):
    """Read a DICOM file from the specified path."""

    image = pydicom.dcmread(os.path.join(imagepath, imagename), force=force)

    return image


def write_dicom(imagepath, imagename, image):
    """Save the DICOM file to the specified path."""

    image.save_as(os.path.join(imagepath, imagename))


def read_mha(imagepath, imagename):
    """Read an MHA file and return its content as a NumPy array."""

    image = sitk.ReadImage(os.path.join(imagepath, imagename), sitk.sitkInt16)
    
    if 'STRUCT' in imagename:
        array = sitk.GetArrayFromImage(image)  
    elif 'ROI' in imagename:
        array = sitk.GetArrayFromImage(image).astype(np.uint16)

        # Find contours and create overlays
        overlays = [
            ImagejRoi.frompoints(np.round(contour)[:, ::-1]).tobytes()
            for contour in find_contours(array, level=0.9999)
        ]
        for overlay in overlays:
            array = ImagejRoi.frombytes(overlay)

    return array


def write_mha(imagepath, imagename, array):
    """Convert a NumPy array to an MHA file and save it."""
    
    image = sitk.GetImageFromArray(array, sitk.sitkUInt8)
    
    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(imagepath, imagename))
    writer.Execute(image)


def read_raw(imagepath, imagename):
    """Read a raw file and return its content as a NumPy array."""

    filename, file_extension = os.path.splitext(imagename)
    size_substr = filename[filename.find('size') + 5: filename.find('_0')]
    rows, _, cols = map(int, size_substr.split('x'))

    # Read the raw data into a NumPy array
    with open(os.path.join(imagepath, imagename), 'rb') as image:
        array = np.fromfile(image, dtype=np.float32, count=rows * cols)
        array = np.reshape(array, (rows, cols))

    return array

