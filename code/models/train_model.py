from fastai.vision.all import *
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score ,roc_auc_score
from fastai.losses import CrossEntropyLossFlat
from code.config import *

from efficientnet_pytorch import EfficientNet
from transformers import AutoModel
import zipfile
import json
from tensorflow.keras.models import load_model
import onnx
import onnxmltools
from onnx2pytorch import ConvertModel
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


MODEL_FORMAT = ['pth', 'pkl']  # pth - pytorch model, pkl - fastai model
num_dr_classes = 5
    # 0 - No DR
    # 1 - Mild
    # 2 - Moderate
    # 3 - Severe
    # 4 - Proliferative DR
#--------------------------------------
class Precision(Metric):
    def __init__(self, average='macro'):
        self.average = average

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def accumulate(self, learn):
        preds = learn.pred.argmax(dim=-1).cpu().numpy()
        targs = learn.y.cpu().numpy()
        self.y_true.extend(targs)
        self.y_pred.extend(preds)

    @property
    def value(self):
        return precision_score(self.y_true, self.y_pred, average=self.average , zero_division=1)

    @property
    def name(self):
        return f"Precision_{self.average}"
#--------------------------------------
class F1Score(Metric):
    def __init__(self, average='macro'):
        self.average = average

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def accumulate(self, learn):
        preds = learn.pred.argmax(dim=-1).cpu().numpy()
        targs = learn.y.cpu().numpy()
        self.y_true.extend(targs)
        self.y_pred.extend(preds)

    @property
    def value(self):
        return f1_score(self.y_true, self.y_pred, average=self.average, zero_division=1)

    @property
    def name(self):
        return f"F1Score_{self.average}"
#--------------------------------------
class Recall(Metric):
    def __init__(self, average='macro'):
        self.average = average

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def accumulate(self, learn):
        preds = learn.pred.argmax(dim=-1).cpu().numpy()
        targs = learn.y.cpu().numpy()
        self.y_true.extend(targs)
        self.y_pred.extend(preds)

    @property
    def value(self):
        return recall_score(self.y_true, self.y_pred, average=self.average , zero_division=1)

    @property
    def name(self):
        return f"Recall_{self.average}"
#--------------------------------------


class CustomModelMethods:
    """CustomModelMethods is a base class that defines methods for training and evaluating a model"""
    def __init__(self, model_name=None , label_weights=None ):
        self.class_learner = None
        self.model_name = model_name
        self.class_weigths = label_weights
        self.recall_macro = Recall(average='macro')
        self.recall_micro = Recall(average='micro')
        self.precision_macro = Precision(average='macro')
        self.precision_micro = Precision(average='micro')
        self.f1_macro = F1Score(average='macro')
        self.f1_micro = F1Score(average='micro')

    def get_learner(self):
        """get_learner initializes a Learner"""
        if self.class_learner is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        return self.class_learner

    def set_learner(self, learner):
        self.class_learner = learner

    def find_learn_rates(self):
        """find_learn_rates uses Learner's lr_find method to find the best learning rates"""
        if self.class_learner is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")

        # performs a "learning rate finder" test. It helps to find the optimal learning rate for your model by training it over a range of exponentially increasing learning rates and observing which one results in the fastest, most stable reduction in the loss.
        print("==> Find best learning rates ...")
        weights_pre_lr_find = L(self.class_learner.model.parameters())
        # lr_min is safest for stability.
        # lr_steep is aggressive, aiming for faster learning.
        # lr_valley and lr_slide balance stability and learning speed.
        # Given no more specific context, Minimum/10 (lr_min/10) is often a good default.
        lr_min, lr_steep, lr_valley, lr_slide = self.class_learner.lr_find(suggest_funcs=(minimum, steep, valley, slide))  # Find optimal learning rate
        weights_post_lr_find = L(self.class_learner.model.parameters())
        test_eq(weights_pre_lr_find, weights_post_lr_find)
        print(f"Learning rates : Minimum/10:\t{lr_min:.2e}\nSteepest point:\t{lr_steep:.2e}\nLongest valley:\t{lr_valley:.2e}\nSlide interval:\t{lr_slide:.2e}")

        print("==> Plot best learning rates ...")
        fig, ax = plt.subplots()
        ax.plot(self.class_learner.recorder.lrs, self.class_learner.recorder.losses)
        # Get the learning rate suggestions
        learning_rates = [lr_min, lr_steep, lr_valley, lr_slide]
        print("Learning rates: ", learning_rates)
        # Interpolate losses for these learning rates
        losses = [np.interp(lr, self.class_learner.recorder.lrs, self.class_learner.recorder.losses) for lr in learning_rates]
        print("Losses: ", losses)

        # Scatter plot for best learning rates
        ax.scatter(x=learning_rates, y=losses, marker='o', color='red')
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        # Annotate the plot with learning rates and their labels
        labels = ['Minimum', 'Steep', 'Valley', 'Slide']
        for lr, label in zip(learning_rates, labels):
            loss = np.interp(lr, self.class_learner.recorder.lrs, self.class_learner.recorder.losses)
            ax.annotate(f'{label}: {lr:.2e}', (lr, loss))
        # Save the plot
        fig.savefig('lr_find_plot.png')
        plt.close(fig)
        return lr_min, lr_steep, lr_valley, lr_slide

    def train_model(self, dls, epochs=10):
        """train_model uses Learner's fine_tune method for training"""
        if self.class_learner is None:
            self.class_learner = Learner(dls, self.model,
                loss_func=CrossEntropyLossFlat(weight=self.class_weigths),
                opt_func=partial(Adam, lr=0.001),
                metrics=[accuracy, self.recall_macro, self.recall_micro , self.precision_macro, self.precision_micro, self.f1_macro, self.f1_micro] )
        if tune_find_lr == True:
            lr_min, lr_steep, lr_valley, lr_slide = self.find_learn_rates()
            lr_rate = lr_min/10
            print("==> Done, going to fine tune a model for ", epochs, " epochs with learning rate " , lr_rate)
            self.class_learner.fine_tune(epochs, base_lr=lr_rate)
        else:
            print("==> Done, going to fine tune a model for ", epochs, " epochs")
            self.class_learner.fine_tune(epochs)
        #print("==> Done, going to fit a model for ", epochs)
        #self.class_learner.fit(epochs)
        fig, ax = plt.subplots(figsize=(8, 8))
        self.class_learner.show_results(max_n=9, ax=ax)
        # Save the plot to a file
        fig.savefig('results.png')
        # Close the plot to free up memory
        plt.close(fig)

    def predict(self, img_path):
        """predict uses Learner's predict method to make predictions"""
        if self.class_learner is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        img = PILImage.create(img_path)
        return self.class_learner.predict(img)

    #Evaluation is for assessing final model performance on unseen data (validation and/or test set).
    def evaluate_model_on(self, learner, test_dls , eval_dls='train'):
        """Uses Learner to generate evaluation metrics, plot the confusion matrix, and display top losses"""
        if learner is None:
             raise ValueError("Model has not been trained yet. Please train the model first.")

        learner.dls = test_dls
        print(f" Evaluate model on {eval_dls} set ... split_data_dataloaders = {split_data_dataloaders}")
        # get_preds - Retrieves predictions for an entire dataset
        # Returns tuples of predictions, targets, and optionally losses
        # Determine which dataloader to use for evaluation
        if split_data_dataloaders == False:    # separate dataloader with train set only, no val split
            preds, targets, losses = learner.get_preds(dl=test_dls.train, with_loss=True)
        else:
            if eval_dls  == 'val':
                preds, targets, losses = learner.get_preds(dl=test_dls.valid, with_loss=True)
        # test is always separate dataloader , while val maybe splitted from train or a separate dataloader
        if eval_dls  == 'test':
                preds, targets, losses = learner.get_preds(dl=test_dls.train, with_loss=True)

        # Convert Predictions to Class Indices
        preds2 = torch.argmax(preds, dim=1)

        # Convert tensors to numpy arrays for use with metrics
        true_labels = targets.numpy()
        pred_labels = preds2.numpy()

        # print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
        # print(f"Loss: {losses.numpy().mean():.4f}")
        print(f"Precision: {precision_score(true_labels, pred_labels, average='macro'):.4f}")
        print(f"Recall: {recall_score(true_labels, pred_labels, average='macro'):.4f}")
        print(f"F1 Score: {f1_score(true_labels, pred_labels, average='macro'):.4f}")

        # Ensure preds is 2D
        probs = F.softmax(preds, dim=1)
        auc = roc_auc_score(targets.numpy(), probs.numpy(), multi_class='ovr')
        print(f'AUC: {auc:.4f}')

        print('\nFull Classification Report:')
        print(classification_report(targets, preds2))
        print('Confusion Matrix:')
        print(confusion_matrix(targets, preds2))

        return preds2, targets , losses


# vgg16: The standard VGG16 model without batch normalization.
# vgg16_bn: This version includes batch normalization layers after each convolutional layer,
# which helps in stabilizing and accelerating the training process.
# resnet18: A smaller version of the ResNet model with 18 layers.
# resnet34: A larger version of the ResNet model with 34 layers.
# dense121: A smaller version of the DenseNet model with 121 layers.
# 224x224 pixel images were used for training all the above models thus the input size is 224x224x3 for all models after resizing and aug_transforms with DataBlocks.
pretrained_models = ['vgg16', 'resnet18', 'vgg16_bn', 'resnet34',
                      'resnet50', 'resnet152', 'efficientnet-b7']
transform_sizes = {
    'vgg16': 224,
    'resnet18': 224,
    'vgg16_bn': 224,
    'resnet34': 224,
    'resnet50': 224,
    'resnet152': 224,
    'efficientnet-b7' : 224
}

# PretrainedEyeDiseaseClassifier defines a pretrained vision model, either resnet18 or vgg16 model.
class PretrainedEyeDiseaseClassifier(nn.Module , CustomModelMethods):
    """Pretrained model for classifying eye diseases which in this project used
       for transfer learning and/or as a reference point for performance evaluation of a (train) model under development.
       It can be any model out of the collection supported by torchvision.models - welcome to replace it and refactor this class!
    """
    def __init__(self, num_classes=5, model_name='vgg16' , label_weights=None):
        nn.Module.__init__(self)
        CustomModelMethods.__init__(self, model_name, label_weights)
        if model_name is not None:
            model_name = model_name.lower()
            if model_name not in pretrained_models:
                raise ValueError(f"Unsupported model name: {model_name}")

            if model_name == 'vgg16':
                self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
                self.model.classifier[6] = nn.Linear(4096, num_classes)
            elif model_name == 'vgg16_bn':
                self.model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
                self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
            elif model_name == 'resnet18':
                self.model = models.resnet18(pretrained=True)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'resnet34':
                self.model = models.resnet34(pretrained=True)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'resnet50':
                self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes)
            elif model_name =='resnet152':
                self.model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes)
            elif model_name == 'efficientnet-b7':
                self.model = EfficientNet.from_pretrained('efficientnet-b7')  # Load pretrained EfficientNet-B7 model
                num_ftrs = self.model._fc.in_features  # Get the number of input features for the final layer
                self.model._fc = nn.Linear(num_ftrs, num_classes)  # Replace final layer with a custom one
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        return self.model(x)

    # def set_num_classes(self, num_classes):
    #     if isinstance(self.model, models.VGG):
    #         self.model.classifier[6] = nn.Linear(4096, num_classes)
    #     elif isinstance(self.model, models.ResNet):
    #         num_ftrs = self.model.fc.in_features
    #         self.model.fc = nn.Linear(num_ftrs, num_classes)


#================================================================================================
# EyeDiseaseClassifier defines CNN model -welcome to change its architecture!
class EyeDiseaseClassifier(nn.Module,CustomModelMethods):
    """CNN model for classifying eye diseases"""
    def __init__(self, num_classes=5 ,model_name=None , label_weights=None):
        nn.Module.__init__(self)
        CustomModelMethods.__init__(self, model_name, label_weights)
        self.model = self # for compatibility with learner used by custom functions in CustomModelMethods
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def set_num_classes(self, num_classes):
        self.fc2 = nn.Linear(512, num_classes)

#================================================================================================
def get_model_name_from_saved(file_path):
    """Get model name from saved model file"""
    # Split base file name from a path pretrained_pkl_path
    base_name = os.path.basename(file_path).lower()
    print(f"Base name of the pretrained model path: {base_name}")
    longest_match = ""
    for model_name in pretrained_models:
        model_name = model_name.lower()
        if model_name in base_name and len(model_name) > len(longest_match):
            longest_match = model_name
    model_name = longest_match
    print(f" Detected model_name: {model_name}")
    return model_name

#================================================================================================
# import importlib
# from tensorflow.keras import Model
# def create_model(config):
#     # Check if 'config' key exists and contains 'layers'
#     if "config" not in config or "layers" not in config["config"]:
#         raise KeyError("The 'config' key or 'layers' key is missing in the configuration file.")

#     # Access layers from config['config']
#     layers = config["config"]["layers"]

#     def create_sequential(layers):
#         model = Sequential()
#         for layer_def in layers:
#             layer_class_name = layer_def["class_name"]
#             layer_config = layer_def["config"]
#             print(f"Creating layer {layer_class_name} with config: {layer_config}")

#             try:
#                 # Dynamically get the layer class from keras.layers
#                 layer_class = getattr(__import__('tensorflow.keras.layers', fromlist=[layer_class_name]), layer_class_name)
#                 # Add the layer to the model
#                 layer = layer_class(**layer_config)
#                 model.add(layer)
#             except AttributeError as e:
#                 print(f"Error creating layer {layer_class_name}: {e}")
#                 raise
#         return model

#     def create_functional(layers):
#         inputs = {}
#         outputs = {}

#         for layer_def in layers:
#             layer_class_name = layer_def["class_name"]
#             layer_config = layer_def["config"]
#             print(f"Creating layer {layer_class_name} with config: {layer_config}")

#             try:
#                 layer_class = getattr(__import__('tensorflow.keras.layers', fromlist=[layer_class_name]), layer_class_name)
#                 if layer_class_name == 'InputLayer':
#                     inputs[layer_def['config']['name']] = Input(**layer_config)
#                 else:
#                     if 'inbound_nodes' in layer_def:
#                         inbound_layers = [inputs[node[0]] for node in layer_def['inbound_nodes'][0]]
#                         if len(inbound_layers) == 1:
#                             outputs[layer_def['config']['name']] = layer_class(**layer_config)(inbound_layers[0])
#                         else:
#                             outputs[layer_def['config']['name']] = layer_class(**layer_config)(inbound_layers)
#                     else:
#                         outputs[layer_def['config']['name']] = layer_class(**layer_config)(list(inputs.values())[0])
#             except AttributeError as e:
#                 print(f"Error creating layer {layer_class_name}: {e}")
#                 raise

#         return Model(inputs=list(inputs.values()), outputs=list(outputs.values()))

#     # Handle different model types
#     if config['class_name'] == 'Sequential':
#         return create_sequential(layers)
#     elif config['class_name'] == 'Functional':
#         return create_functional(layers)
#     else:
#         raise ValueError(f"Unsupported model type: {config['class_name']}")


#============================================================================================
# create model for inference from saved model file
def load_model_from_pth(file_path, export_method = 'weights'):
    """Load a pretrained model from a file"""

    model_name = get_model_name_from_saved(file_path)
    my_model = None
    if model_name not in pretrained_models:
        print(f"Not a pretrained model name: {model_name}")
        my_model = EyeDiseaseClassifier(num_classes=num_dr_classes)
    else:
        print(f"Pretrained model name: {model_name}")
        my_model = PretrainedEyeDiseaseClassifier(
            num_classes=num_dr_classes,
            model_name=model_name)

    print(f"Pretrained export_method: {export_method}")
    if export_method == 'weights':
        model = my_model
        model.load_state_dict(torch.load(file_path))
        model.class_learner = Learner(None, model,
                    loss_func=CrossEntropyLossFlat(weight=model.class_weigths),
                    opt_func=partial(Adam, lr=0.001),
                    metrics=[accuracy, model.recall_macro, model.recall_micro , model.precision_macro, model.precision_micro, model.f1_macro, model.f1_micro] )

    elif export_method == 'huggingface_weights':
        model = AutoModel.from_pretrained(file_path)
    elif export_method == 'full':
        conf = None
        extract_dir = 'extracted_model'
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            print("Extracted files:")
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    print(os.path.join(root, file))

            config_path = os.path.join(extract_dir, 'config.json')
            print(f"Config path: {config_path}")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    conf = json.load(f)                
                    if conf['module'] == 'keras': # neeed to convert to torch format prior to load
                        print("Keras model detected")
                        # Step 1: Load the Keras Model
                        weights_path = os.path.join(extract_dir, 'model.weights.h5')
                        print(f"weights_path path: {weights_path}")
                        try:
                            keras_model = load_model(weights_path)
                        except Exception as e:
                            print(f"Error loading keras model: {e}")
                            raise ValueError(f"Unsupported keras model type without the architecture file. Parsing config.json to reproduce architecture is unsupported.")
                            # keras_model = create_model(conf)
                            # keras_model.load_weights(weights_path)

                        # Step 2: Convert Keras Model to ONNX
                        onnx_model = onnxmltools.convert_keras(keras_model)
                        # Save ONNX model
                        onnxmltools.utils.save_model(onnx_model, 'model.onnx')
                        # Load the ONNX model
                        onnx_model = onnx.load('model.onnx')
                        # Convert ONNX model to PyTorch
                        pytorch_model = ConvertModel(onnx_model)
                        # Initialize your model class with the converted PyTorch model
                        if model_name not in pretrained_models:
                            model = EyeDiseaseClassifier(pytorch_model, num_classes=num_dr_classes)
                        else:
                            model = PretrainedEyeDiseaseClassifier(pytorch_model, num_classes=num_dr_classes)
                        # Step 3: Map Keras Weights to PyTorch
                        keras_weights = keras_model.get_weights()
                        # Ensure the layers are in the same order and then assign weights
                        for idx, layer in enumerate(pytorch_model.layers):
                            layer.weight.data = torch.tensor(keras_weights[idx])

                        # (Optional) Load additional metadata if necessary
                        metadata_path = os.path.join(extract_dir, 'metadata.json')
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            print(metadata)
            else:
                print("No keras config file found.")
                # Try loading with PyTorch
                state_dict = torch.load(file_path, map_location=torch.device('cpu'))
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    print("This is a PyTorch state_dict.")
                    # Load the state_dict into the model
                    my_model.load_state_dict(state_dict)
                    model = my_model
                else:
                    print("This is a PyTorch model file.")
                    # Load the PyTorch model
                    model = torch.load(file_path)
                    model = my_model

        # Remove the extracted_model directory
        shutil.rmtree(extract_dir)

    model.explainability = True
    print(model.get_learner().model)
    model.to(device)
    return model

#================================================================================================
def load_model_from_pkl(file_path):
    """Load a pretrained model from a file"""

    model_name = get_model_name_from_saved(file_path)
    if model_name not in pretrained_models:
        print(f"Not a pretrained model name: {model_name}")
        model = EyeDiseaseClassifier(num_classes=num_dr_classes)
    else:
        print(f"Pretrained model name: {model_name}")
        model = PretrainedEyeDiseaseClassifier(
            num_classes=num_dr_classes,
            model_name=model_name)

    model.set_learner( load_learner(file_path))
    model.explainability = True
    print(model.get_learner().model)
    model.to(device)
    return model
