# Importing necessary libraries for model training, evaluation, and transfer learning
import joblib
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.vision.all import *  # Useful for building and training vision models and quick Learner setup
import torch  # PyTorch, the core framework for defining models and tensors, building neural networks
import torch.nn as nn  # Neural network layers from PyTorch
import torchvision.models as models  # Contains pre-built, pretrained models like ResNet and VGG.
from fastai.metrics import accuracy

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

import torch.nn.functional as F  # Functional layers, like activation functions and utilities for the forward pass.
from efficientnet_pytorch import EfficientNet
from transformers import AutoModel  # Import for Hugging Face model saving
# Training with Fastai progress tracking
from fastai.callback.progress import ProgressCallback
from fastai.learner import load_learner
from Util.training_Logger import TrainingLogger
import pandas as pd


class CustomModelMethods:
    """CustomModelMethods is a base class that defines methods for training and evaluating a model using FastAI."""

    def __init__(self):
        # Initialize without any learner
        self.class_learner = None

    def get_learner(self, dls, criterion, metrics):
        """Initializes a FastAI Learner with a given dataset, criterion, and metrics."""
        if self.class_learner is None:  # If learner doesn't exist, create one
            self.class_learner = Learner(dls, self, loss_func=criterion, metrics=metrics)
        return self.class_learner  # Return the initialized learner


    def train_model(self, dls, epochs=4, criterion=None, early_stopping=None):
        """
        Train the model using the specified data loaders, loss criterion, and early stopping.

        Parameters:
        - dls: DataLoader object, contains training and validation data.
        - epochs: int, number of training epochs.
        - criterion: loss function, e.g., Focal Loss for handling imbalanced data.
        - early_stopping: EarlyStopping object, used to stop training early if validation loss does not improve.

        This method creates a learner, sets the optimizer, and trains the model using the fit_one_cycle method.
        The training and validation loss, as well as other specified metrics, are recorded and displayed.
        """
        # Initialize the learner with Focal Loss if specified as criterion
        if self.class_learner is None:
            self.class_learner = Learner(
                dls,
                self,
                loss_func=criterion,
                metrics=[Precision(average='macro'), Recall(average='macro')]
            )
            self.class_learner.create_opt()

        print("Starting training...")

        # Train for the full number of epochs with early stopping checks
        for epoch in range(epochs):
            # Train for one cycle of the full epoch using fit_one_cycle(1)
            print(f"Epoch {epoch + 1}/{epochs} - Starting training cycle...")
            self.class_learner.fit_one_cycle(1)  # Complete one cycle for the current epoch

            # Retrieve the training loss
            train_loss = self.class_learner.recorder.losses[epoch] if len(
                self.class_learner.recorder.losses) > epoch else None

            # Retrieve the validation metrics
            val_metrics = self.class_learner.recorder.values[epoch] if len(
                self.class_learner.recorder.values) > epoch else None
            if val_metrics:
                val_loss = val_metrics[0]
                precision_val = val_metrics[1] if len(val_metrics) > 1 else None
                recall_val = val_metrics[2] if len(val_metrics) > 2 else None
            else:
                val_loss = precision_val = recall_val = None

            # Display training and validation metrics for the epoch
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}" if train_loss else 'N/A',
                f"Validation Loss: {val_loss:.4f}" if val_loss else 'N/A',
                f"Precision: {precision_val:.4f}" if precision_val else 'N/A',
                f"Recall: {recall_val:.4f}" if recall_val else 'N/A'
            )

            # Calculate confusion matrix for each epoch
            preds, true_labels = [], []
            for batch in dls.valid:
                inputs, labels = batch
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            cm = confusion_matrix(true_labels, preds)
            print(f"Confusion Matrix for epoch {epoch + 1}:\n{cm}")

            # Early stopping check if early_stopping is provided and val_loss is available
            if early_stopping and val_loss is not None:
                if early_stopping(val_loss):
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                    break

        print("Training completed.")
        print("Recorded training losses:", self.class_learner.recorder.losses)
        print("Recorded validation values:", self.class_learner.recorder.values)

    # def train_model(self, dls, epochs=4, criterion=None, early_stopping=None):
    #     if self.class_learner is None:
    #         self.class_learner = Learner(dls, self, loss_func=criterion, metrics=[Precision(average='macro'), Recall(average='macro')]  )
    #         self.class_learner.create_opt()
    #
    #     print("Starting training...")
    #
    #     # Use fit_one_cycle and observe the detailed debug output
    #     # Now train the model with the custom callback
    #     self.class_learner.fit_one_cycle(epochs)  #  TrainingLogger(), ProgressCallback()  cbs=[TensorBoardCallback(log_dir='my_logs')]
    #     print("Training completed.")
    #
    #     print("Recorded training losses:", self.class_learner.recorder.losses)
    #     print("Recorded validation values:", self.class_learner.recorder.values)
    #
    #     # Display metrics and check recorder contents
    #     for epoch in range(epochs):
    #         if len(self.class_learner.recorder.losses) > epoch:
    #             train_loss = float(self.class_learner.recorder.losses[epoch])
    #         else:
    #             train_loss = None
    #             print(f"No training loss recorded for epoch {epoch + 1}")
    #
    #         if len(self.class_learner.recorder.values) > epoch:
    #             val_metrics = self.class_learner.recorder.values[epoch]
    #             val_loss = val_metrics[0]
    #             accuracy_val = val_metrics[1] if len(val_metrics) > 1 else None
    #         else:
    #             val_loss = accuracy_val = None
    #             print(f"No validation metrics recorded for epoch {epoch + 1}")
    #
    #         if train_loss is not None and val_loss is not None:
    #             print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
    #                   f"Accuracy: {accuracy_val:.4f}" if accuracy_val else "")
    #         else:
    #             print(f"Epoch [{epoch + 1}/{epochs}], Incomplete epoch data.")

    def evaluate_model(self, dls):
        """Evaluates the model: generates confusion matrix, displays top losses, and calculates accuracy."""
        print("Using validation DataLoader for evaluation.")
        print(f"Validation DataLoader size: {len(dls.valid)}")

        if self.class_learner is None:
            self.class_learner = Learner(dls.valid, self, loss_func=CrossEntropyLossFlat(), metrics=[Precision(average='macro'), Recall(average='macro')]  )

        # Check dataset sizes during evaluation
        print(f"Validation dataset size: {len(dls.valid_ds)}")
        print(f"Training dataset size: {len(dls.train_ds)}")

        # Print DataLoader sizes
        print(f"Training DataLoader size: {len(dls.train)}")
        print(f"Validation DataLoader size: {len(dls.valid)}")

        # Validate structure of the validation dataset
        print("Validating the structure of the validation dataset...")
        print(f"Sample validation data (first 5 items): {dls.valid_ds.items[:5]}")  # Print first 5 items

        # Correctly access the validation labels
        labels = [item[1] for item in dls.valid_ds.items]  # Assuming the second element is the label
        print(f"Sample validation labels (first 5): {labels[:5]}")  # Print first 5 labels

        # Generate metrics like confusion matrix and top losses
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

        # Print each label and its corresponding prediction
        for i in range(len(preds)):
            print(f"Validation Image {i + 1}: True Label = {targets[i].item()}, Predicted Label = {preds[i].item()}")

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

            # Micro and Macro metrics
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

    def save_model_(self, filename, mode='weights'):
        """Saves the model weights to the specified file or directory using Hugging Face's save_pretrained method."""

        if mode == 'weights':
            if hasattr(self.class_learner.model, 'save_pretrained'):
                # Using Hugging Face's save_pretrained method for compatible models
                self.class_learner.model.save_pretrained(filename)
                print(f"Hugging Face method - Model weights saved to {filename}.")
            else:
                torch.save(self.class_learner.model.state_dict(), filename)
                print(f"Model weights saved to {filename}.")

        elif mode == 'full':
            # Using joblib to save the entire model
            joblib.dump(self.class_learner, filename)
            print(f"Full model saved to {filename} using joblib.")

        else:
            raise ValueError("Invalid mode. Use 'full' or 'weights'.")

    import joblib
    import torch

    def load_model(self, filename, mode='full'):
        """Loads the model weights or the full model from the specified file."""
        if filename.endswith('.pth') or filename.endswith('.pt'):
            if mode == 'weights':
                self.class_learner.model.load_state_dict(torch.load(filename))
                print(f"Model weights loaded from {filename}.")
            elif mode == 'full':
                # Using joblib to load the entire model
                self.class_learner = joblib.load(filename)
                print(f"Full model loaded from {filename} using joblib.")
            else:
                raise ValueError("Invalid mode. Use 'full' or 'weights'.")
        else:
            raise ValueError("Unsupported file extension. Use .pth, .pt, or .joblib.")


# List of pretrained models we can choose from for transfer learning
pretrained_models = ['vgg16', 'resnet18', 'efficientnet-b7']


class PretrainedEyeDiseaseClassifier(nn.Module, CustomModelMethods):
    """A pretrained model for classifying eye diseases using transfer learning. Supports VGG16, ResNet18, or EfficientNet-B7."""

    def __init__(self, num_classes=5, pretrained_model='vgg16'):
        nn.Module.__init__(self)  # Initialize PyTorch's nn.Module
        CustomModelMethods.__init__(self)  # Initialize methods for training/evaluation from the CustomModelMethods class
        print("Initializing PretrainedEyeDiseaseClassifier...")

        # Initialize num_ftrs as an instance variable
        self.num_ftrs = None

        # Choose between VGG16, ResNet18, or EfficientNet-B7 pretrained models
        if pretrained_model == 'vgg16':
            self.model = models.vgg16(pretrained=True)  # Load pretrained VGG16 model
            self.num_ftrs = 4096  # Set num_ftrs for VGG16
            self.model.classifier[6] = nn.Linear(self.num_ftrs, num_classes)  # Replace final layer for `num_classes`
            print("Using VGG16 model.")
        elif pretrained_model == 'resnet18':
            self.model = models.resnet18(pretrained=True)  # Load pretrained ResNet18 model
            self.num_ftrs = self.model.fc.in_features  # Get the number of input features for the final layer
            self.model.fc = nn.Linear(self.num_ftrs, num_classes)  # Replace final layer with a custom one
            print("Using ResNet18 model.")
        elif pretrained_model == 'efficientnet-b7':
            self.model = EfficientNet.from_pretrained('efficientnet-b7')  # Load pretrained EfficientNet-B7 model
            self.num_ftrs = self.model._fc.in_features   # Get the number of input features for the final layer
            # self.model._fc = nn.Linear(self.num_ftrs, num_classes)  # Replace final layer with a custom one
            self.model._fc = self.create_fc_layers(num_classes, self.num_ftrs) # return
            print("Using EfficientNet-B7 model.")
        else:
            raise ValueError("Unsupported pretrained model. Choose 'vgg16', 'resnet18', or 'efficientnet-b7'.")

    def create_fc_layers(self, num_classes, num_ftrs):
        """Creates custom fully connected layers connected to the pretrained model's output."""
        print("Creating fully connected layers...")
        return nn.Sequential(
            nn.Linear(num_ftrs, 512),  # First fully connected layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, 256),  # Second fully connected layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, num_classes)  # Final layer for classification
        )

    def forward(self, x):
        """Forward pass for the model, which applies the pretrained model's forward pass.
        every batch processed during training or validation calls the forward method,
        and you have a print statement inside that method. so comment it
        """

        ##print("Performing forward pass...")
        x = self.model(x)  # Pass input through the model
        return x  # Return the output from the model

    def set_num_classes(self, num_classes):
        """Dynamically adjust the final layer to accommodate a new number of classes."""
        print("Setting number of classes...")
        if isinstance(self.model, models.VGG):
            self.model.classifier[6] = nn.Linear(self.num_ftrs, num_classes)  # Update VGG final layer
        elif isinstance(self.model, models.ResNet):
            self.model.fc = nn.Linear(self.num_ftrs, num_classes)  # Update ResNet final layer
        elif isinstance(self.model, EfficientNet):
            self.model._fc = nn.Linear(self.num_ftrs, num_classes)  # Update EfficientNet final layer

        print(f"Number of classes updated to {num_classes}.")  # Confirmation message
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
