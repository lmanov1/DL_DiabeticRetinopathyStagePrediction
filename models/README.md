# diabetic-retinopathy-stage-prediction
Diabetic Retinopathy Stage Prediction DL Project

# Load the saved model
## keras ##
from tensorflow.keras.models import load_model
model = load_model('v1.0_vgg16_model.keras')

## torch ##
# Load the entire model (architecture + parameters)
import torch
model = torch.load("eye_disease_full_model.pth")

#   Models Versioning
## ------------------- ##

Models Versions and Metrics can be found here:
https://docs.google.com/spreadsheets/d/1nHFWxYxLbr_LActM5Ux2sQX9ZV8HJ1R0fv4t9QF0eqg/edit?gid=0#gid=0
