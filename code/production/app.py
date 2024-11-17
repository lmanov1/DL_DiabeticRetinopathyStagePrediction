import datetime
import gradio as gr
import torch
import torchvision.models
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
import os
import subprocess
from PIL import Image
from all_defs import  classes_mapping  , translate_labels , print_directory_recursively
from all_defs import REPO_ID , DATASET_REPO_ID , MODEL_DATASET_DIR  , models_upload_to_dataset
from huggingface_hub import HfApi, login , hf_hub_download
sys.path.append('./')
import subprocess
from haggingface_model import *
import fastai

from torch.utils.tensorboard import SummaryWriter
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.callback.progress import ProgressCallback
from TensorboardStats import *

examples = [
    "train19/0/69f43381317b.jpg",
     "train19/1/5347b4c8e9b3.jpg",
     "train19/2/28a4d00927b7.jpg",
     "train19/3/4fef9ed8a4c5.jpg",
     "train19/4/b87f9c59748b.jpg"
]

hf_api_token = os.getenv("MY_TOKEN")
login(token=hf_api_token)
api = HfApi()

model_name_to_use = os.getenv("MODEL_NAME")
print(f"Model name to use: {model_name_to_use}")
model_name, model_extension = os.path.splitext(model_name_to_use)
print(f"Model name: {model_name}, Model extension: {model_extension}")
model_type = model_extension[1:] # remove the dot

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
print(f"model_name = {  model_name }")

#%rm -rf "logs/fine_tune/" + model_name

datetimestr = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir="logs/fine_tune/" + model_name + '/' + datetimestr
print(f"Log directory: {logdir}")
print(f"datetimestr: {datetimestr}")
# Create a SummaryWriter
print("Initializing summary writer")
#%rm -rf "runs/" + model_name + '*'

writer = SummaryWriter(log_dir='runs/' + model_name + '/' + datetimestr)
# Print the current working directory
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

# Define the DataBlock
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # Input: Image, Target: Category
    get_items=get_image_files,  # Get all image files in the path
    #splitter=RandomSplitter(valid_pct=0.9, seed=42),  # Train/valid split
    splitter = None,
    get_y=parent_label,  # Labels from parent folder names
    item_tfms=Resize(460),  # Resize to 460 before batching
    batch_tfms=aug_transforms(size=224)  # Resize to 224 with augmentations
)
print("initialized datablock" , dblock)
# Load the DataLoaders
path = Path('train19')
#print_directory_recursively(current_directory)
dls = dblock.dataloaders(path, bs=8, num_workers=0, with_labels=True)
print("initialized dataloaders" , dls)
# Initialize the Model
model = PretrainedEyeDiseaseClassifier(pretrained_model='efficientnet-b7', num_classes=5)


inf_learner = Learner(
    dls,
    model,
    opt_func=partial(Adam, lr=0.001),
    metrics=[
        accuracy,
        precision_micro, recall_micro, f1score_micro,
        precision_macro, recall_macro, f1score_macro
    ]
    )

early_stopping = EarlyStopping(patience=5)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_type == 'pth':    
    model = torch.load(pretrained_model_weigths_file,map_location=torch.device("cpu"))   
    inf_learner.model = model   

elif model_type == 'pkl':
    inf_learner = load_learner(pretrained_model_weigths_file)
    inf_learner.model = model
    model.class_learner = inf_learner
elif model_type == 'keras':
    print("Keras model is not supported")
    pass
else:
    raise ValueError(f"Unsupported model format: {model_type}")

class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']  
inf_learner.create_opt()
inf_learner.model.eval()

from fastai.vision.all import *
import numpy as np
from PIL import Image

# def preprocess_image(img_path):
#     # Load the image
#     pil_img = Image.open(img_path).convert("RGB")
#     # Convert to fastai PILImage
#     fastai_img = PILImage.create(pil_img)
#     # Resize to 460 before batching 
#     resize_tfm = Resize(460)
#     resized_img = resize_tfm(fastai_img)

#     # Convert the PIL image to a tensor 
#     transform_to_tensor = transforms.ToTensor() 
#     tensor_img = transform_to_tensor(resized_img)

#     #Apply augmentation transformations (resize to 224 with augmentations) 
#     batch_tfms = aug_transforms(size=224)
#    # Create a temporary DataLoader to apply batch transformations 
#     dl = DataLoaders.from_dsets(([tensor_img], [0]), bs=1, after_batch=batch_tfms).test_dl([tensor_img]) 
#     # Fetch the first batch (this applies the transformations) 
#     processed_tensor_img = dl.one_batch()[0] 
#     # Add a batch dimension 
#     processed_tensor_img = processed_tensor_img.unsqueeze(0)
#     return processed_tensor_img
#     return tensor_img


def classify_img(img):
     # 'img' is a NumPy array with shape (height, width, 3)
    try:        
        print(f'received from gradio: {img}') 
        print(f"!!!!!!!!!!!Image type: {type(img)}")       
        
        # Make a prediction using the learner
        pred, idx, probs = inf_learner.predict(img)
        print(f"Prediction: {pred}; Probability: {probs[idx]:.04f}")
        print( f"probs: {probs}" )       
        return "## Detected severity of Diabetic Retinopathy ", dict(zip(class_names, probs))       
        
    except Exception as e:
        print(f"Debug - Error details: {str(e)}")
        traceback.print_exc()
        return f"Error during prediction: {str(e)}", None
  

import traceback
def validate():
    try:               
        img_shape = dls.train.one_batch()[0].shape 
        print(f'Shape of the image: {img_shape}')
    
        preds, targets = inf_learner.get_preds(dl=dls.train) 
        print(f'Predictions: {preds[:5]}, Targets: {targets[:5]}') 
		
        # Show results (sample predictions) 
        print("Learner's show results..")
        inf_learner.show_results() 
        fig_sample_preds = plt.gcf()  # Get the current figure
        writer.add_figure('Learner show results', fig_sample_preds)
        plt.show()  # Ensure the plot is displayed
        #plt.close(fig_sample_preds)
        
        # Interpret the model 
        interp = ClassificationInterpretation.from_learner(inf_learner) 

        # Plot top losses
        print("Plotting top losses...")
        interp.plot_top_losses(k=5)
        interp_tl_fig = plt.gcf()  # Get the current figure
        # Add legend to the plot
        handles, labels = interp_tl_fig.get_axes()[0].get_legend_handles_labels()
        interp_tl_fig.get_axes()[0].legend(handles, labels, loc='upper right')
        writer.add_figure('Top Losses', interp_tl_fig)
        plt.show()  # Ensure the plot is displayed
        #plt.close(interp_tl_fig)
        
        most_confused = interp.most_confused(min_val=2)
        print("Most confused classes:")
        print(most_confused)
        most_confused_text = ""
        for i, (predicted, actual, count) in enumerate(most_confused):    
            predicted = int(predicted)
            actual = int(actual)
            most_confused_text += str(f"{i + 1}: {class_names[predicted]} ({predicted}) vs {class_names[actual]} ({actual}) : ({count} samples)")  
            most_confused_text += "\n" 
        writer.add_text('Most Confused Classes', most_confused_text)
        print('Most Confused Classes\n' + most_confused_text)
        
        conf_preds = np.argmax(preds, axis=1)
        conf_targets = targets.numpy()
        # Confusion matrix
        cm = confusion_matrix(conf_targets, conf_preds)      
        # Plot confusion matrix
        fig_conf_matrix = plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized Confusion Matrix', tensorboard_writer=writer)
        #fig_conf_matrix = plot_confusion_matrix(cm, classes=class_names, tensorboard_writer=writer)

        # Convert targets and predictions to numpy arrays if they aren't already
        class_targets = np.array(targets)
        class_preds = np.argmax(preds, axis=1)  # Convert continuous predictions to discrete class labels
        # Generate the classification report
        print("Classification Report (external to learn)")
        report_text = classification_report(
        y_true=class_targets,
        y_pred=class_preds,
        target_names=class_names,
        digits=3,zero_division=0
        )
        print(report_text)
        writer.add_text('Classification Report', report_text)

        # Classification report
        report = classification_report(class_targets, class_preds, target_names=class_names, output_dict=True,zero_division=0)
        print(report)
        # Log Precision, Recall, F1-score, and Accuracy
        for label, metrics in report.items():
            if label == 'accuracy':
                writer.add_scalar(f'Accuracy', metrics, 1)
            else:
                for metric_name, value in metrics.items():
                    writer.add_scalar(f'{label}/{metric_name.capitalize()}', value, 1)

        # ROC AUC
        print("Plotting ROC AUC...")
        roc_auc_fig = plot_roc_auc(class_names, class_targets, class_preds, tensorboard_writer=writer)        
        # Plot and log sample predictions
        print("Plotting 10 Sample Predictions...")
        fig_sample_preds = plot_sample_predictions(dls.train, class_preds, class_targets, path, tensorboard_writer= writer)  
        	
        
        writer.close()
       
        return [
                    #torchvision_graph_file,
                    report_text,
                    most_confused_text,
                    fig_sample_preds,   #inf_learner.show_results(),
                    interp_tl_fig,    # top losses                 
                    roc_auc_fig,      # ROC AUC   
                    fig_conf_matrix   # confusion matrix
                ]
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        traceback.print_exc()
        return [
             f"Error during analysis: {str(e)}","",None,None,None,None
        ]


# Create the main application
inf_learner.model.eval()
inf_learner.remove_cb(fastai.callback.progress.ProgressCallback)

demo = gr.Blocks()

with demo:
    
    gr.Markdown("# Diabetic Retinopathy Detection System")
    
    with gr.Tabs():
        with gr.Tab("Single Image Prediction"):            
            #type="pil"
            #image_input = gr.Image(type="filepath"),
            markdown_output = gr.Markdown("## Please choose a retinal fundus camera image for prediction")
            label_output = gr.Label()            
                        
            gr.Interface(
                fn=classify_img,
                inputs=gr.Image(type="pil", label="Choose Image to classify"),
                outputs=[markdown_output, label_output],
                examples=examples,
                title="Single Image Prediction",
                cache_examples=False
            )
            
        with gr.Tab("Model Analysis"):          
            #model_graph = gr.Image(label="Model Architecture")
            report = gr.Textbox(label="Classification Report", lines=10)
            confused = gr.Textbox(label="Most Confused Classes", lines=10)
            predictions = gr.Plot(label="Sample Predictions")
            top_losses = gr.Plot(label="Top 5 Losses")
            roc_auc = gr.Plot(label="Area under the ROC curve")
            conf_matrix = gr.Plot(label="Confusion Matrix")
                        
            gr.Interface(
                fn=validate,
                inputs=None,                          
                outputs=[report, confused , predictions, top_losses , roc_auc ,conf_matrix],
                title="Model Validation"
            )

# Start TensorBoard
tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", "tb_logs"])

# Launch the application
demo.launch(share=True)

# Cleanup
tensorboard_process.terminate()