import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet50
from torchvision.models import vgg16
from torchvision.models import efficientnet_b7
from torchvision.models import efficientnet_b1
from MDataLoader import CustomDataLoader 



class MModel():
    def __init__(self, device, model_name = 'efficientnet_b1'):
        self.device = device
        if model_name == "efficientnet_b7":
            self.model_name = "efficientnet_b7"
            model = efficientnet_b7(pretrained=True)  
        else: 
            self.model_name = "efficientnet_b1"
            model = efficientnet_b1(pretrained=True)

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 5)  # Adjust output layer for 5 classes
        self.model = model.to(self.device)      

    def train(self, custom_data_loader, num_epochs, learning_rate, l2_reg):
        self.criterion = nn.CrossEntropyLoss(weight=custom_data_loader.class_weights).to(self.device)  
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(custom_data_loader.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # Transfer to GPU
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print statistics
                running_loss += loss.item()
                if batch_idx % 100 == 99:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(custom_data_loader.train_loader)}], Loss: {running_loss/100:.4f}')
                    running_loss = 0.0

  