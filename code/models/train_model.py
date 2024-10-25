# Importing necessary libraries for model training, evaluation, and transfer learning
from fastai.vision.all import *  # useful for building and training vision models and quick Learner setup
import torch  # PyTorch, the core framework for defining models and tensors, building neural networks
import torch.nn as nn  # Neural network layers from PyTorch
import torchvision.models as models  # Contains pre-built, pretrained models like ResNet and VGG.
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix  # Used for calculating accuracy, generating classification reports, and plotting confusion matrices.
import torch.nn.functional as F  # Functional layers, like activation functions  and utilities for the forward pass.


# --- Overall Structure ---
# The file defines three main classes:
# 1. `CustomModelMethods`: Base class providing methods for training, evaluating, and managing models using FastAI Learner.
# 2. `PretrainedEyeDiseaseClassifier`: A transfer learning model using either `vgg16` or `resnet18` for classifying eye diseases.
# 3. `EyeDiseaseClassifier`: A custom Convolutional Neural Network (CNN) architecture for eye disease classification.

# ---- Early Stopping (Optional) ---
# Early stopping can be introduced in the training process to prevent overfitting.
# Example: Add `callbacks=[EarlyStoppingCallback(monitor='accuracy', patience=3)]` in the `Learner` object within `train_model`.

class CustomModelMethods:
    """CustomModelMethods is a base class that defines methods for training and evaluating a model using FastAI"""

    def __init__(self):
        # Initialize without any learner
        self.class_learner = None

    def get_learner(self, dls, criterion, metrics):
        """get_learner initializes a FastAI Learner with a given dataset, criterion, and metrics."""
        if self.class_learner is None:  # If learner doesn't exist, create one
            self.class_learner = Learner(dls, self, loss_func=criterion, metrics=metrics)
        return self.class_learner  # Return the initialized learner


    def train_model(self, dls, epochs=10):
        """train_model uses FastAI Learner's fine_tune method for transfer learning or training from scratch."""
        if self.class_learner is None:  # Create learner if not already initialized
            self.class_learner = Learner(dls, self, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        self.class_learner.fine_tune(epochs)  # Perform fine-tuning over `epochs` epochs

    def evaluate_model(self, dls):
        """Evaluate the model: generates confusion matrix, displays top losses, and calculates accuracy."""

        # Ensure we are using the validation DataLoader
        print("Using validation DataLoader for evaluation.")
        print(f"Validation DataLoader size: {len(dls.valid)}")

        if self.class_learner is None:
            self.class_learner = Learner(dls.valid, self, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

        # Check the dataset sizes during evaluation
        print(f"Validation dataset size: {len(dls.valid_ds)}")
        print(f"Training dataset size: {len(dls.train_ds)}")

        # Print DataLoader sizes
        print(f"Training DataLoader size: {len(dls.train)}")
        print(f"Validation DataLoader size: {len(dls.valid)}")

        # Check the structure of the validation dataset
        print("Validating the structure of the validation dataset...")
        print(f"Sample validation data (first 5 items): {dls.valid_ds.items[:5]}")  # Print the first 5 items

        # Correctly access the validation labels
        labels = [item[1] for item in dls.valid_ds.items]  # Assuming the second element is the label
        print(f"Sample validation labels (first 5): {labels[:5]}")  # Print the first 5 labels

        # ClassificationInterpretation helps generate metrics like confusion matrix and top losses
        interp = ClassificationInterpretation.from_learner(self.class_learner)

        # Get predictions and targets
        preds, targets = self.class_learner.get_preds(ds_idx=1)  # Ensures we use validation set

        # Check for any mismatch in lengths
        if preds.size(0) != targets.size(0):
            print(f"Mismatch detected! Predictions: {preds.size(0)}, Targets: {targets.size(0)}")
            # Truncate predictions or targets to the smaller size
            min_size = min(preds.size(0), targets.size(0))
            preds = preds[:min_size]
            targets = targets[:min_size]
            print(f"Using truncated sizes: {preds.size(0)} for both predictions and targets.")

        # Print the shape of the predictions and targets to diagnose any mismatch
        print(f"Shape of predictions: {preds.shape}")
        print(f"Shape of targets: {targets.shape}")

        # Print the first few predictions and targets for a quick sanity check
        print(f"Sample predictions: {preds[:10]}")
        print(f"Sample targets: {targets[:10]}")

        preds = torch.argmax(preds, dim=1)

        # Check for unique classes in targets and preds
        unique_targets = set(targets.numpy())
        unique_preds = set(preds.numpy())

        if len(unique_targets) == 0 or len(unique_preds) == 0:
            print("Warning: No valid classes present in targets or predictions.")
        else:
            print(f"Accuracy: {accuracy_score(targets, preds):.4f}")
            print('Classification Report:')
            print(classification_report(targets, preds, zero_division=0))  # Adjust zero_division as needed
            print('Confusion Matrix:')
            print(confusion_matrix(targets, preds))


# List of pretrained models we can choose from for transfer learning
pretrained_models = ['vgg16', 'resnet18']


# PretrainedEyeDiseaseClassifier allows the use of pretrained models (VGG16 or ResNet18) for eye disease classification
class PretrainedEyeDiseaseClassifier(nn.Module, CustomModelMethods):
    """A pretrained model for classifying eye diseases using transfer learning. Supports VGG16 or ResNet18."""

    def __init__(self, num_classes=5, pretrained_model='vgg16'):
        nn.Module.__init__(self)  # Initialize PyTorch's nn.Module
        CustomModelMethods.__init__(
            self)  # Initialize methods for training/evaluation from the CustomModelMethods class

        # Choose between VGG16 or ResNet18 pretrained models
        if pretrained_model == 'vgg16':
            self.model = models.vgg16(pretrained=True)  # Load pretrained VGG16 model
            self.model.classifier[6] = nn.Linear(4096, num_classes)  # Replace final layer for `num_classes`
        elif pretrained_model == 'resnet18':
            self.model = models.resnet18(pretrained=True)  # Load pretrained ResNet18 model
            num_ftrs = self.model.fc.in_features  # Get the number of input features for the final layer
            self.model.fc = nn.Linear(num_ftrs, num_classes)  # Replace final layer with a custom one
        else:
            raise ValueError("Unsupported pretrained model. Choose 'vgg16' or 'resnet18'.")

    def forward(self, x):
        """Forward pass for the model, which simply passes input through the selected pretrained model."""
        return self.model(x)

    def set_num_classes(self, num_classes):
        """Dynamically adjust the final layer to accommodate a new number of classes."""
        if isinstance(self.model, models.VGG):
            self.model.classifier[6] = nn.Linear(4096, num_classes)  # Update VGG final layer
        elif isinstance(self.model, models.ResNet):
            num_ftrs = self.model.fc.in_features  # Get features for ResNet final layer
            self.model.fc = nn.Linear(num_ftrs, num_classes)  # Update ResNet final layer


# EyeDiseaseClassifier defines a custom CNN architecture for classifying eye diseases
class EyeDiseaseClassifier(nn.Module, CustomModelMethods):
    """A custom Convolutional Neural Network (CNN) for classifying eye diseases."""

    def __init__(self, num_classes=5):
        nn.Module.__init__(self)  # Initialize nn.Module for PyTorch
        CustomModelMethods.__init__(self)  # Initialize methods from CustomModelMethods for training/evaluation

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # First conv layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Second conv layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Third conv layer

        # Define fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, num_classes)  # Final output layer, number of outputs equals num_classes

        # Add Dropout layer
        self.dropout = nn.Dropout(p=0.5)  # 50% Dropout

    def forward(self, x):
        """Forward pass for the custom CNN architecture."""
        x = F.relu(self.conv1(x))  # Apply ReLU after the first conv layer
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # First pooling layer
        x = F.relu(self.conv2(x))  # ReLU after second conv layer
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Second pooling layer
        x = F.relu(self.conv3(x))  # ReLU after third conv layer
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Third pooling layer
        x = x.view(x.size(0), -1)  # Flatten before feeding into fully connected layers
        x = F.relu(self.fc1(x))  # Apply ReLU after first fully connected layer
        x = self.dropout(x)  # Apply Dropout after fully connected layer
        x = self.fc2(x)  # Output layer
        return x

    def set_num_classes(self, num_classes):
        """Set the final layer dynamically to support different numbers of output classes."""
        self.fc2 = nn.Linear(512, num_classes)  # Adjust final output layer to match num_classes


# --- Usage Notes ---
# 1. Instantiate either `PretrainedEyeDiseaseClassifier` or `EyeDiseaseClassifier` based on model needs.
# 2. Use the `train_model` method from `CustomModelMethods` to fine-tune the model.
# 3. Evaluate the trained model using `evaluate_model` for accuracy, classification report, and confusion matrix.
