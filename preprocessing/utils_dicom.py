import pydicom


def edit_dicom_array(dicomimage, newarray) :

    dicomimage.PixelData = newarray.tobytes()

    return dicomimage


def add_dicom_tag(dicomimage, tag, keyword, representation, value) :

    # dicomimage[tag] = pydicom.dataelem.DataElement(keyword, representation, value)

    setattr(dicomimage, tag, pydicom.dataelem.DataElement(keyword, representation, value))

    return dicomimage


def remove_dicom_tag(dicomimage, tag) :

    # del dicomimage.tag

    delattr(dicomimage, tag)

    return dicomimage



