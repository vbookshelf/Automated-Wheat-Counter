
# This prediction code is simulated in this kernel:
# https://www.kaggle.com/vbookshelf/e-33-wheat-flask-app-inference-code

# Handling base64 images is simulated in this kernel:
# https://www.kaggle.com/vbookshelf/tb-my-flask-python-app-workflow

# *** NOTE: This entire file will get imported into __init__.py ***
# ------------------------------------------------------------------

#from the app folder import the app object
from app import app


from flask import request
from flask import jsonify
from flask import Flask
import base64
from PIL import Image
import io


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models # reg_model comes from here


import numpy as np
import cv2



# Special packages

# seg_model comes from here
import segmentation_models_pytorch as smp 
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# this is for the pre-processing
import albumentations as albu 
from albumentations import Compose



# Are we using and special packages?
# ------------------------------------

# Check if these are pre-intalled on PythonAnywhere.
# If not, these packages have to be pip installed onto the server:

# (1) Segmentation Models Pytorch
# https://github.com/qubvel/segmentation_models.pytorch
# $ pip install segmentation-models-pytorch

# (2) Albumentations (used to pre-process the image)
# https://github.com/albumentations-team/albumentations
# $ pip install albumentations

# ----end




# -------------------------------
# Define the Model Architectures
# -------------------------------

# This app uses two models - segmentation and regression.

# (1) seg_model
# --------------

BACKBONE = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation


seg_model = smp.Unet(
    encoder_name=BACKBONE, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=1, 
    activation=ACTIVATION,
)


# Initialize the pre-processing function.
# Both models are resnet34 so the same image pre-processing method applies to both.
preprocessing_fn = smp.encoders.get_preprocessing_fn(BACKBONE, ENCODER_WEIGHTS)



# (2) reg_model
# --------------

reg_model = models.resnet34(pretrained=True)
in_features = reg_model.fc.in_features 

reg_model.fc = nn.Linear(in_features, 1)




# -------------------------------
# Define the helper functions
# -------------------------------


def get_model():
	
    global seg_model
    global reg_model
	
	# Load the saved weights into the architecture.
	# Note that this file (views.py) gets imported into the app.ini file therefore,
	# place the model in the same folder as the app.ini file.
    
    seg_model_path = 'seg_model.pt'
    reg_model_path = 'reg_model.pt'
    seg_model.load_state_dict(torch.load(seg_model_path, map_location=torch.device('cpu')))
    reg_model.load_state_dict(torch.load(reg_model_path, map_location=torch.device('cpu')))

    # send the models to the device
    seg_model.to(device)
    reg_model.to(device)

    # Set the Mode
    seg_model.eval()
    reg_model.eval()

    # Turn gradient calculations off.
    torch.set_grad_enabled(False)
	
    print(" * Models loaded!")
	
	
	
	
	
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
    ]
    return albu.Compose(_transform)
	
	
	
	
def preprocess_image(base64Image):
    
	"""
	Input: base64 image (dataURL with prefix removed)
	Output: image as numpy array --> 512x512x3
	"""
	
	# Decode the base64 image.
	# The result is a PIL image not a numpy array.
	decoded = base64.b64decode(base64Image)
	PIL_image = Image.open(io.BytesIO(decoded))
	
	# convert the PIL image into a numpy array
	# This method uses Keras img_to_array.
	#np_image = img_to_array(PIL_image)
	
	# This method uses numpy.
	## Double check the output shape.
	np_image = np.array(PIL_image)
	
	
	# resize the image to 512x512x3
	np_image = cv2.resize(np_image, (512, 512))
	
	
	# Pre-process the image
	
	# This is a dummy mask so that we can use the same 
	# setup as that in the Kaggle notebook.
	mask = np.zeros((512, 512, 1))
	
	# initialize the pre-processing function.
	# This step was done when the model was initialized.
	# source: from segmentation_models_pytorch.encoders import get_preprocessing_fn
	# preprocessing_fn = get_preprocessing_fn(BACKBONE, pretrained='imagenet')
	
	# call the function that we defined above
	preprocessing = get_preprocessing(preprocessing_fn)
	
	# get the pre-processed image
	sample = preprocessing(image=np_image, mask=mask)
	np_image, dummy_mask = sample['image'], sample['mask']
	
	# convert to a channels first format to suit Pytorch
	np_image = np_image.transpose((2, 0, 1))
	
	# convert to a torch tensor
	torch_image = torch.tensor(np_image, dtype=torch.float)
	
	# add a batch dimension
	torch_image = torch_image.unsqueeze(0)
	
	return torch_image
	

	
def multiply_masks_and_images(torch_images, torch_thresh_masks):
    
	"""
	Trying to do this multiplication with Pytorch tensors
	did not produce the result that I wanted. Therefore, here I am
	converting the tensors to numy, doing the multiplication, and 
	then converting back to pytorch.
	
	"""
	
	# convert from torch tensors to numpy
	np_images = torch_images.cpu().numpy()
	np_thresh_masks = torch_thresh_masks.cpu().numpy()
	
	# reshape
	np_images = np_images.reshape((-1, 512, 512, 3))
	np_thresh_masks = np_thresh_masks.reshape((-1, 512, 512, 1))
	
	
	# multiply the mask by the image
	modified_images = np_thresh_masks * np_images
	
	# change shape to channels first to suit pytorch
	#modified_images = modified_images.transpose((2, 0, 1))
	modified_images = modified_images.reshape((-1, 3, 512, 512))
	
	# convert to torch tensor
	modified_images = torch.tensor(modified_images, dtype=torch.float)
	
	return modified_images
	
	
	
	

# -------------------------------
# RUN THE CODE
# -------------------------------

# Define the device
# ------------------
device = "cpu"


# Load the models
# ----------------

print(" * Loading Pytorch models...")
get_model()




# Define the endpoints
# ---------------------


@app.route('/')
def index():
	return 'Hello world. I am a flask app.'
	
	

@app.route('/test')
def test():
	return 'Testing testing...'



# To access this endpoint navigate to:
# server_ip_address/static/predict.html
# Update the server ip address in the static/predict.html file.
# This endpoint has an html page in the static flask folder.

@app.route("/predict", methods=["POST"])
def predict():
	message = request.get_json(force=True)
	base64Image = message['image']
	#decoded = base64.b64decode(encoded)
	#image = Image.open(io.BytesIO(decoded))
	
	
	processed_image = preprocess_image(base64Image)
	print('Image pre-processing complete.')
	
	# send the masks to the device
	processed_image = processed_image.to(device, dtype=torch.float)
	
	# use seg_model to make a prediction
	seg_pred = seg_model(processed_image)
	print('Seg prediction complete.')
	
	# Threshold the predicted segmentation masks.
	# Remember that these are torch tensors and not numpy matrices. We are using
	# the torch tensor method of changing the datatype from bool to integer.
	thresh_mask = (seg_pred >= 0.7).int()
	
	# do the multiplication with numpy
	seg_output_masks = multiply_masks_and_images(processed_image, thresh_mask)
	
	# send the masks to the device
	seg_output_masks = seg_output_masks.to(device, dtype=torch.float)
	
	# pass the input through the model
	reg_pred = reg_model(seg_output_masks)
	
	# convert the prediction to a number
	pred = reg_pred.item()
	
	# round the prediction
	pred = round(pred, 0)
	
	print('Reg prediction complete.')
	
	
	response = {
	    'prediction': {
	        'wheat_count': pred,
	    }
	}
	return jsonify(response)
	
	
	