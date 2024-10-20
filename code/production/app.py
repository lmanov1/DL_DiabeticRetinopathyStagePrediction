
import gradio as gr
import torch 
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
import os
from PIL import Image
from models.train_model import EyeDiseaseClassifier , PretrainedEyeDiseaseClassifier , pretrained_models , num_dr_classes , load_model_from_pth , load_model_from_pkl , MODEL_FORMAT
from all_defs import  classes_mapping  , translate_labels 
from all_defs import REPO_ID , DATASET_REPO_ID , MODEL_DATASET_DIR  , models_upload_to_dataset    
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

model_name_to_use = os.getenv("MODEL_NAME")
print(f"Model name to use: {model_name_to_use}")
model_name, model_extension = os.path.splitext(model_name_to_use)
print(f"Model name: {model_name}, Model extension: {model_extension}")
model_type = model_extension[1:] # remove the dot
if model_type not in MODEL_FORMAT:
    raise ValueError(f"Unsupported model format. Choose one of: {MODEL_FORMAT}")

# pkl_file_item = None
# pkl_file_item = next(item for item in models_upload_to_dataset if str(item).endswith('.pkl'))

# if pkl_file_item is None:
#     raise ValueError(f"Pretrained model file not found in {MODEL_DATASET_DIR}")

# file_path = Path(f"{MODEL_DATASET_DIR}/{pkl_file_item}")
file_path = Path(f"{MODEL_DATASET_DIR}/{model_name_to_use}")
try:
    print(f"Downloading file from Hugging Face Hub: {file_path}")
    pretrained_model_weigths_file =  api.hf_hub_download(
            repo_id=DATASET_REPO_ID, 
            filename=str(file_path), 
            repo_type="dataset")
    print(f"Downloaded file from Hugging Face Hub: {pretrained_model_weigths_file}")

except Exception as e:
    print(f"Error downloading file from Hugging Face Hub: {file_path}")
    print(e)
    pass


print(f"Loading model from {pretrained_model_weigths_file}")


if model_type == 'pth':
    model = EyeDiseaseClassifier(num_classes=num_dr_classes)
    load_model_from_pth(model, pretrained_model_weigths_file)
    inf_learner = model.get_learner()
elif model_type == 'pkl':
    inf_learner = load_model_from_pkl(pretrained_model_weigths_file)
elif model_type == 'keras':
    print("Keras model is not supported")
    pass
else:
    raise ValueError(f"Unsupported model format: {model_type}")

inf_learner.model.eval()
print(inf_learner.model)
def classify_img(img):
    pred, idx, probs = inf_learner.predict(img)
    #print(f"Prediction: {pred}; Probability: {probs[idx]:.04f}")
    translated_labels = translate_labels(inf_learner.dls.vocab)    
    return "## Detected severity of Diabetic Retinopathy ", dict(zip(translated_labels, probs))

image = gr.Image()
label = gr.Label()

iface = gr.Interface(fn=classify_img, inputs=image, outputs=
        [   gr.Markdown("## Please choose a retinal fundus camera image for prediction"), label ], 
        examples=examples)
iface.launch()