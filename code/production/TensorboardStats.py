from sklearn.metrics import f1_score, recall_score ,accuracy_score ,precision_score
import numpy as np
import matplotlib.pyplot as plt
import itertools
from fastai.vision.core import PILImage
from PIL import Image
from packaging import version
import tensorboard as tb
import torch 
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchviz import make_dot 
import tensorflow as tf 
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Custom function to generate a report after training
def evaluate_model(learner, dl):
    preds, targs = learner.get_preds(dl=dl)
    preds = preds.argmax(dim=1).cpu().numpy()
    targets = targs.cpu().numpy()

    # Accuracy
    acc = accuracy_score(targets, preds)
    print(f"Accuracy: {acc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(targets, preds, zero_division=1))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, preds))

    micro_precision = precision_score(targets, preds, average='micro')
    micro_recall = recall_score(targets, preds, average='micro')
    micro_f1 = f1_score(targets, preds, average='micro')

    macro_precision = precision_score(targets, preds, average='macro')
    macro_recall = recall_score(targets, preds, average='macro')
    macro_f1 = f1_score(targets, preds, average='macro')

    print("\nMicro and Macro Averages:")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'] 

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, tensorboard_writer=None):
    fig_cm, ax = plt.subplots(figsize=(8, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            text = f"{cm[i, j]:.2f}"
        else:
            text = f"{int(cm[i, j])}"
        plt.text(j, i, text, horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()    
    plt.show()
    if tensorboard_writer:
        tensorboard_writer.add_figure(title, fig_cm)
    return fig_cm


def plot_roc_auc(class_names , targets, preds, tensorboard_writer):
        # ROC and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(targets == i, preds == i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    fig_roc, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(class_names)):
        ax.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    if tensorboard_writer:
        tensorboard_writer.add_figure('ROC Curves', fig_roc)   
    return fig_roc



# Function to plot sample predictions
def plot_sample_predictions(dls, preds, targets, img_path='' , 
                            tensorboard_writer=None):
    """
    Plot sample predictions for FastAI vision classification
    
    Parameters:
    dls: FastAI DataLoaders
    preds: Model predictions
    targets: True labels
    img_path: Base path to the images directory (if needed)
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    # Get the first 10 items from validation dataset
    print(f"Number of items in DataLoader: {len(dls.items)}")
    num_items = 10
    if len(dls.items) < 10:
        num_items = len(dls.items)
    items = dls.items[:num_items]
    
    #for i, (item, pred, target) in enumerate(zip(items.itertuples(), preds[:10], targets[:10])):
    for i, (item, pred, target) in enumerate(zip(items, preds[:num_items], targets[:num_items])):
        try:
            print(f"Item: {item}, Prediction: {pred}, Target: {target}")
            # # Construct full image path
            # img_filename = item.id_code            
            # if img_path:
            #     full_path = os.path.join(img_path, img_filename)
            # else:
            #     full_path = itemz
                
            # Load and display image
            img = Image.open(item)
            axes[i].imshow(img)
            axes[i].set_title(f'True: {class_names[target]}\nPred: {class_names[pred]}')
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error processing image {item}: {str(e)}")
            axes[i].text(0.5, 0.5, 'Error loading image', 
                        ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    if tensorboard_writer:      
        tensorboard_writer.add_figure('Sample Predictions', fig)
  
    return fig



    # img = mpimg.imread(torchvision_graph_file)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

