import os
from my_data import Dataloader as KaggleDataLoader
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.utils import resample
from ImgUpscale import MImageUpscale


# Custom dataset class
class CustomImageDataset(Dataset):  
    def __init__(self, dataframe, transform):
        self.df = dataframe  # Accept pandas DataFrame directly
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use the pre-computed image path from the DataFrame
        img_name = self.df.iloc[idx]['image_path']  # Accessing the 'image_path' column directly
        image = Image.open(img_name).convert("RGB")  # Ensure image is in RGB
        label = self.df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        return image, label
    

class CustomDataLoader():
    def __init__(self, dataset_infos, device, transform, batch_size = 32, test_size=0.2, val_size=0.1, do_balance = False, do_upscal = False):
        self.device = device
        combined_df = pd.DataFrame()

        # Define a mapping for standardized column names
        column_mappings = {
            'diagnosis': 'label',  # Standardizing 'diagnosis' column
            'level': 'label',      # 'level' will also map to 'label'
            'id_code': 'image_id', # Standardizing 'id_code' to 'image_id'
            'image': 'image_id'    # Assuming 'image' also refers to the image ID
        }
        
        for info in dataset_infos:
            kaggle_dataset_name = info['kaggle_dataset_name']
            dwnld_dataset_path = info['dwnld_dataset_path']
            image_folders = info['images_paths']  # List of image folder names
            label_files = info['label_files']      # List of corresponding CSV file names
            self.transform = transform
            
            print(f"Downloading dataset {kaggle_dataset_name} into {dwnld_dataset_path}")
            kaggle_downloader = KaggleDataLoader.KaggleDataDownLoader(dwnld_dataset_path, kaggle_dataset_name, None)    
            dataset_dir = kaggle_downloader.dataset_dir

            for label_file, img_folder in zip(label_files, image_folders):
                csv_file_full_path = os.path.join(dataset_dir.strip(''), label_file)
                images_full_path = os.path.join(dataset_dir, img_folder)

                # Load the CSV file
                df = pd.read_csv(csv_file_full_path)
                 # Rename columns according to mapping
                df.rename(columns=column_mappings, inplace=True)



                df['image_path'] = df['image_id'].astype(str).apply(lambda x: os.path.join(images_full_path, f"{x}.jpg"))
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        if do_balance:
            combined_df = self.balance_diabetic_retinopathy_data(combined_df)
        print(combined_df['label'].value_counts())

        if do_upscal:
            Image_Upscale = MImageUpscale(device)
            combined_df = Image_Upscale.process_images(combined_df)

        # Class weights handling - assuming 'label' is the standardized name
        if 'label' not in combined_df.columns:
            raise ValueError("The label column was not found in the combined DataFrame.")
        
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(combined_df['label']),
            y=combined_df['label']
        )
        self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)  # Move to the appropriate device


       # Split the data
        train_df, temp_df = train_test_split(combined_df, test_size=test_size + val_size, stratify=combined_df['label'], random_state=42) # CHANGE: Add stratify
        val_relative_size = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(temp_df, test_size=1 - val_relative_size, stratify=temp_df['label'], random_state=42) # CHANGE: Add stratify

        # Create datasets and loaders
        self.train_dataset = CustomImageDataset(dataframe=train_df, transform=self.transform)
        self.val_dataset = CustomImageDataset(dataframe=val_df, transform=self.transform)
        self.test_dataset = CustomImageDataset(dataframe=test_df, transform=self.transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
       

    def balance_diabetic_retinopathy_data(self, df): 
        class_counts = df['label'].value_counts()
        print("Original class distribution:\n", class_counts) 
        # Determine the number of images in the No DR class (label 0)
        count_no_dr = class_counts.get(0, 0)  # Get the count of No DR class

        # Check if the count exceeds the threshold for case selection
        if count_no_dr > 30000:  # More than 30,000, treat it as case 1
            target_no_dr_count = 30000  # Target for case 1: larger distribution
        else:
            target_no_dr_count = 1200  # Target for case 2: smaller distribution

        # Downsample No DR (label 0) based on calculated target
        majority_class = df[df['label'] == 0]  # Identify the majority class
        if len(majority_class) > target_no_dr_count:  
            majority_downsampled = resample(majority_class,
                                            replace=False,
                                            n_samples=target_no_dr_count,  # Keep specified max sample
                                            random_state=123)  # reproducible results
        else:
            majority_downsampled = majority_class  # No downsampling needed if within limits

        # Start creating a balanced DataFrame with downsampled majority class
        balanced_df = majority_downsampled  

        # Retain all other classes without changes
        for label in class_counts.index: 
            if label != 0:  # Skip majority class
                balanced_df = pd.concat([balanced_df, df[df['label'] == label]])

        print("New class distribution after balancing:\n", balanced_df['label'].value_counts())  
        return balanced_df.reset_index(drop=True)  
