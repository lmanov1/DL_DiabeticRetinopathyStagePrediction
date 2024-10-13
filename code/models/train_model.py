from fastai.vision.all import *
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn.functional as F

class CustomModelMethods:
    """CustomModelMethods is a base class that defines methods for training and evaluating a model"""
    def __init__(self):
        self.class_learner = None

    def get_learner(self, dls, criterion, metrics):
        """get_learner initializes a Learner"""
        if self.class_learner is None:
            self.class_learner = Learner(dls, self, loss_func=criterion, metrics=metrics)
        return self.class_learner

    def train_model(self, dls, epochs=10):
        """train_model uses Learner's fine_tune method for training"""
        if self.class_learner is None:
            self.class_learner = Learner(dls, self, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        self.class_learner.fine_tune(epochs)

    def evaluate_model(self, dls):
        """Uses Learner to generate evaluation metrics, plot the confusion matrix, and display top losses"""
        if self.class_learner is None:
            self.class_learner = Learner(dls, self, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        interp = ClassificationInterpretation.from_learner(self.class_learner)
        interp.plot_confusion_matrix()
        interp.plot_top_losses(k=9, nrows=3)
        preds, targets = self.class_learner.get_preds()
        preds = torch.argmax(preds, dim=1)
        # Calculate and print metrics
        print(f'Accuracy: {accuracy_score(targets, preds):.4f}')
        print('Classification Report:')
        print(classification_report(targets, preds))
        print('Confusion Matrix:')
        print(confusion_matrix(targets, preds))


pretrained_models = ['vgg16', 'resnet18']

# PretrainedEyeDiseaseClassifier defines a pretrained vision model, either resnet18 or vgg16 model.
class PretrainedEyeDiseaseClassifier(nn.Module , CustomModelMethods):
    """Pretrained model for classifying eye diseases which in this project used  
       for transfer learning and/or as a reference point for performance evaluation of a (train) model under development.
       It can be any model out of the collection supported by torchvision.models - welcome to replace it and refactor this class!
    """
    def __init__(self, num_classes=5, pretrained_model='vgg16'):        
        nn.Module.__init__(self)
        CustomModelMethods.__init__(self)

        if pretrained_model == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif pretrained_model == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError("Unsupported pretrained model. Choose 'vgg16' or 'resnet18'.")        
              
    def forward(self, x):
        return self.model(x)

    def set_num_classes(self, num_classes):
        if isinstance(self.model, models.VGG):
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif isinstance(self.model, models.ResNet):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)


# EyeDiseaseClassifier defines CNN model -welcome to change its architecture!
class EyeDiseaseClassifier(nn.Module,CustomModelMethods):
    """CNN model for classifying eye diseases"""
    def __init__(self, num_classes=5):        
        nn.Module.__init__(self)
        CustomModelMethods.__init__(self)

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

