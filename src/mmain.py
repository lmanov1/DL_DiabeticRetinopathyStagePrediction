import torch
from MDataLoader import CustomDataLoader 
from mmodel import MModel
import optuna
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
import csv
import os
import pandas as pd
import json

model_name = 'efficientnet_b1'
# model_name = 'efficientnet_b7'

if model_name == 'efficientnet_b7':
    resolution = (600, 600)
else:
    resolution = (240, 240)    


dataset_infos = [
        {
                'kaggle_dataset_name': 'benjaminwarner/resized-2015-2019-blindness-detection-images',
                'dwnld_dataset_path': 'data/raw/',
                'images_paths': ['resized train 19'],  
                'label_files': ['labels/trainLabels19.csv'],  # Corresponding CSVs
                'label_column_names': ['diagnosis'], 
                'image_column_names': ['id_code']     
        }
        ]

# dataset_infos = [
#     {
#         'kaggle_dataset_name': 'benjaminwarner/resized-2015-2019-blindness-detection-images',
#         'dwnld_dataset_path': 'data/raw/',
#         'images_paths': ['resized train 19', 'resized train 15', 'resized test 15'],  # List of subfolders
#         'label_files': ['labels/trainLabels19.csv', 'labels/trainLabels15.csv', 'labels/testLabels15.csv'],  # Corresponding CSVs
#         'label_column_names': ['diagnosis', 'level', 'level'], 
#         'image_column_names': ['id_code', 'image', 'image']  
#     }
        
#     # {
#     #     'kaggle_dataset_name': 'dataset_2',
#     #     'dwnld_dataset_path': '/path/to/data2',
#     #     'images_paths': ['images3'],  # Only one folder here
#     #     'label_files': ['labels3.csv']  # Only one corresponding CSV
#     # }
# ]




def objective(trial):  
    # Define hyperparameter ranges
    rotation_angle = trial.suggest_int("rotation_angle", 0, 30)
    brightness = trial.suggest_float("brightness", 0.0, 0.5)
    contrast = trial.suggest_float("contrast", 0.0, 0.5)
    saturation = trial.suggest_float("saturation", 0.0, 0.5)
    hue = trial.suggest_float("hue", 0.0, 0.5)
    batch_size = trial.suggest_int("batch_size", 16, 128)

    
    # Create the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_angle),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    num_epochs = trial.suggest_int('num_epochs', 5, 10)  # Include number of epochs
    l2_reg = trial.suggest_loguniform('l2_reg', 1e-6, 1e-2)


    # Create an instance of MModel
    model_instance = MModel(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_name = model_name)
    custom_data_loader = CustomDataLoader(dataset_infos, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), transform = transform, batch_size=batch_size,do_balance=True)
    model_instance.train(custom_data_loader, num_epochs=num_epochs, learning_rate=learning_rate, l2_reg = l2_reg)

    all_targets, all_predictions = [], []
    for inputs, targets in custom_data_loader.val_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        with torch.no_grad():
            outputs = model_instance.model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    recall_macro = recall_score(all_targets, all_predictions, average='macro', zero_division=1)
    print("Recall by class and macro average:")
    print(classification_report(all_targets, all_predictions, zero_division=1))
    return recall_macro

def evaluate_model(model_instance, custom_data_loader):
    model_instance.model.eval()
    all_targets, all_predictions = [], []
    for inputs, targets in custom_data_loader.test_loader:
        inputs, targets = inputs.to(model_instance.device), targets.to(model_instance.device)
        with torch.no_grad():
            outputs = model_instance.model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return all_targets, all_predictions


def run_with_best_params(params = None):

    # Use the best parameters to re-evaluate the model
    # Create a new objective function call to use the best trial's parameters

    # Extract best hyperparameters
    rotation_angle = params["rotation_angle"]
    brightness = params["brightness"]
    contrast = params["contrast"]
    saturation = params["saturation"]
    hue = params["hue"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    num_epochs = params["num_epochs"]
    l2_reg = params["l2_reg"]

    # Define the transformation pipeline using best hyperparameters
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_angle),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

    # Initialize model and data loader again with the best parameters
    model_instance = MModel(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_name = model_name)
    custom_data_loader = CustomDataLoader(dataset_infos, device=model_instance.device, transform=transform, batch_size=batch_size)
    model_instance.train(custom_data_loader, num_epochs=num_epochs, learning_rate=learning_rate, l2_reg = l2_reg)
    
    # Evaluate the model to get targets and predictions
    all_targets, all_predictions = evaluate_model(model_instance, custom_data_loader)
    
    # Calculate additional metrics for logging
    precision_macro = precision_score(all_targets, all_predictions, average='macro', zero_division=1)
    recall_macro = recall_score(all_targets, all_predictions, average='macro', zero_division=1)
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=1)
    precision = precision_score(all_targets, all_predictions,average='micro', zero_division=1)
    recall = recall_score(all_targets, all_predictions, average='micro', zero_division=1)
    f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=1)
    accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
    cm = confusion_matrix(all_targets, all_predictions)

    # print(f"Best_params: {params}")
    print(f"Evaluation result on test with best params:")
    print(f"precision_macro: {precision_macro}, recall_macro: {recall_macro}, f1_macro: {f1_macro}")
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
    print(f"Confusion matrix: {cm}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{model_name}_{timestamp}.pkl"
    torch.save(model_instance.model.state_dict(), model_filename)
    print("Model saved as: ", model_filename)
    model_filename = f"model_{model_name}_{timestamp}.pth"
    torch.save(model_instance.model.state_dict(),  model_filename)
    print("Model saved as: ", model_filename)


    # Save results 
    results_filename = f"model_{model_name}_{timestamp}.txt"
    with open(results_filename, 'w') as f:
        f.write("Best Hyperparameters:\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"model_filename: {model_filename}\n")   
        f.write("\nEvaluation Metrics:\n")
        f.write(f"Precision (Macro): {precision_macro}\n")
        f.write(f"Recall (Macro): {recall_macro}\n")
        f.write(f"F1 Score (Macro): {f1_macro}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall : {recall}\n")
        f.write(f"F1 Score : {f1}\n")
        f.write(f"Accuracy: {accuracy}\n")        
        f.write("\nConfusion Matrix:\n")
        f.write(json.dumps(cm.tolist()))  # Convert to list for easier readability
        f.write("\n")
        print(f"Results saved to {results_filename}")




def main():
    study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=50)  # Optimize based on precision
    study.optimize(objective, n_trials=10)  # Optimize based on precision
    print("Best hyperparameters: ", study.best_params)
    best_trial = study.best_trial
    print("Best trial value: ", best_trial.value)
    run_with_best_params(study.best_params)

if __name__ == '__main__':
    main()
  