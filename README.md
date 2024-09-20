# The impact of synthetic images on a deep learning model for detection and classification in contrast-enhanced mammography

## Purpose
This is the code used to train and validate a deep learning and a handcrafted radiomics model on real and synthetic contrast-enhanced mammography data. If you use this code, please refer to our work 

Van Camp, A. et al. The impact of synthetic images on a deep learning model for detection and classification in contrast-enhanced mammography. (2024). [Manuscript submitted for publication]

## Installation

Key packages to install are

* [math](https://pypi.org/project/python-math/)
* [matplotlib](https://pypi.org/project/matplotlib/)
* [monai](https://pypi.org/project/monai/)
* [numpy](https://pypi.org/project/numpy/)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [openpyxl](https://pypi.org/project/openpyxl/)
* [pandas](https://pypi.org/project/pandas/)
* [pillow](https://pypi.org/project/pillow/)
* [pycocotools](https://pypi.org/project/pycocotools/)
* [pydicom](https://pypi.org/project/pydicom/)
* [pyradiomics](https://pypi.org/project/pyradiomics/)
* [pytorch](https://pytorch.org/get-started/locally/)
* [roifile](https://pypi.org/project/roifile/)
* [scikit-learn](https://pypi.org/project/scikit-learn/)
* [scikit-image](https://pypi.org/project/scikit-image/)
* [SimpleITK](https://pypi.org/project/SimpleITK/)
* [tqdm](https://pypi.org/project/tqdm/)
* [xgboost](https://pypi.org/project/xgboost/)

The [pyradiomics](https://pypi.org/project/pyradiomics/) used to create the 2D models, may not be compatible with the most recent python versions (validated for version 3.9).

## Usage
The folder [preprocessing](preprocessing) contains all scripts used for preprocessing the data. The main file [preprocess_main.py](preprocessing/preprocess_main.py) reads parameters from [parameters.py](preprocessing/parameters.py).

The folder [maskRCNN](maskRCNN) contains all scripts to setup the deep learning model for detection and classification of lesions. The file [trainmaskrcnn.py](maskRCNN/trainmaskrcnn.py) reads parameters from [trainparameters.py](maskRCNN/trainparameters.py) to train the model. The file [testmaskrcnn.py](maskRCNN/testmaskrcnn.py) reads parameters from [testparameters.py](maskRCNN/testparameters.py) to validate the model.

The folder [radiomics](radiomics) contains all scripts to setup the handcrafted radiomics model for classification of lesions. The file [train_radiomics_model.py](radiomics/train_radiomics_model.py) trains the model. The file [inferenced_radiomics_predictions.py](radiomics/inference_radiomics_predictions.py) validates the model.

The file [combine_predictions.py](combine_predictions.py) computes all results for the predictions of the deep learning model, the radiomics model, and the ensembled model.

## Contact
For more information, questions or bug reporting, please contact a.vancamp@maastrichtuniversity.nl.



