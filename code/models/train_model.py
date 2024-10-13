from fastai.vision.all import *
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn.functional as F


pretrained_models = ['vgg16', 'resnet18']
class PretrainedEyeDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained_model='vgg16'):
        super(PretrainedEyeDiseaseClassifier, self).__init__()
        if pretrained_model == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif pretrained_model == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError("Unsupported pretrained model. Choose 'vgg16' or 'resnet18'.")        
        self.learn = None        

    def forward(self, x):
        return self.model(x)

    def set_num_classes(self, num_classes):
        if isinstance(self.model, models.VGG):
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif isinstance(self.model, models.ResNet):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

    def train_model(self, dls, epochs=10):
        """train_model initializes a Learner and uses fine_tune for training"""
        if self.learn is None:
            self.learn = Learner(dls, self, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        self.learn.fine_tune(epochs)

    def evaluate_model(self, dls):
        """initializes a Learner to generate evaluation metrics, plot the confusion matrix, and display top losses"""
        learn = Learner(dls, self, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()
        interp.plot_top_losses(k=9, nrows=3)

        preds, targets = learn.get_preds()
        preds = torch.argmax(preds, dim=1)

        # Calculate and print metrics
        # Accuracy, Classification Report, Confusion Matrix
        print(f'Accuracy: {accuracy_score(targets, preds):.4f}')
        print('Classification Report:')
        print(classification_report(targets, preds))
        print('Confusion Matrix:')
        print(confusion_matrix(targets, preds))


# EyeDiseaseClassifier defines CNN model.
class EyeDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(EyeDiseaseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.learn = None

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

    def train_model(self, dls, epochs=10):
        """train_model initializes a Learner and uses fine_tune for training"""
        if self.learn is None:
            self.learn = Learner(dls, self, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        self.learn.fit_one_cycle(epochs)

    def evaluate_model(self, dls):
        """initializes a Learner to generate evaluation metrics, plot the confusion matrix, and display top losses"""
        if self.learn is None:
            self.learn = Learner(dls, self, loss_func=CrossEntropyLossFlat(), metrics=accuracy)                 
        interp = ClassificationInterpretation.from_learner(self.learn)
        interp.plot_confusion_matrix()
        interp.plot_top_losses(k=9, nrows=3)

        preds, targets = self.learn.get_preds()
        preds = torch.argmax(preds, dim=1)

        # Calculate and print metrics
        # Accuracy, Classification Report, Confusion Matrix
        print(f'Accuracy: {accuracy_score(targets, preds):.4f}')
        print('Classification Report:')
        print(classification_report(targets, preds))
        print('Confusion Matrix:')
        print(confusion_matrix(targets, preds))
