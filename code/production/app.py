
import gradio as gr
import torch 
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
import os
from PIL import Image
from models.train_model import EyeDiseaseClassifier , PretrainedEyeDiseaseClassifier , pretrained_models , num_dr_classes , load_model_from_pth , load_model_from_pkl
from all_defs import DATASET_REPO_ID , MODEL_DATASET_PATH    
from huggingface_hub import HfApi, login , hf_hub_download
sys.path.append('./')

examples = ["0a0780ad3395.jpg","0a262e8b2a5a.jpg","0ad36156ad5d.jpg"]
# Check if CUDA is available and set the device accordingly
# print("Checking for CUDA availability...")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
 
hf_api_token = os.getenv("MY_TOKEN")
login(token=hf_api_token)
api = HfApi()

print(f"Downloading file from Hugging Face Hub: {MODEL_DATASET_PATH}")
pretrained_model_weigths_file = api.hf_hub_download(repo_id=DATASET_REPO_ID, filename=str(MODEL_DATASET_PATH), repo_type="dataset")
print(f"Downloaded file from Hugging Face Hub: {pretrained_model_weigths_file}")                              

# Create another instance of PretrainedEyeDiseaseClassifier
#model = PretrainedEyeDiseaseClassifier(num_classes=num_dr_classes, pretrained_model=pretrained_models[0])
#model.to(device) - don't
#print(model)           
# Load the pretrained weights
#load_model_from_pth(model , pretrained_model_weigths_file)
# no need to load PretrainedEyeDiseaseClassifier because pkl file keeps both model architecture and weigths 

#gr.load("models/timm/efficientnet_b3.ra2_in1k").launch(server_port=7861,share=True)

inf_learner = load_model_from_pkl(pretrained_model_weigths_file)
inf_learner.model.eval()            
print(inf_learner.model)
def classify_img(img):
    pred, idx, probs = inf_learner.predict(img)
    print(f"Prediction: {pred}; Probability: {probs[idx]:.04f}")
    print(inf_learner.dls.vocab)
    return dict(zip(inf_learner.dls.vocab, probs))

image = gr.Image()
label = gr.Label()

iface = gr.Interface(fn=classify_img, inputs=image, outputs=label, examples=examples)
iface.launch()