
import numpy as np
import cv2  # Make sure OpenCV is installed
from PIL import Image
import torch
import os
from ESRGAN.RRDBNet_arch import RRDBNet
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToPILImage

class MImageUpscale():
    def __init__(self, device, model_path = "src\\ESRGAN\models\RRDB_ESRGAN_x4.pth"):
        self.device = device
        model = RRDBNet(3, 3, 64, 23, gc=32)  # Model architecture as required

        current_path = os.getcwd()  # Get the current working directory
        model_full_path = os.path.join(current_path, model_path)  # Construct the full path

        model.load_state_dict(torch.load(model_full_path), strict=True)
        model.eval()
        self.model = model

    def upscale_image(self, img_path, output_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = to_tensor(img).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(img_tensor)

        to_pil_image = ToPILImage()
        upscaled_img = to_pil_image(output.squeeze(0))
        upscaled_img.save(output_path)  

    def should_upscale(self, img_path, label, min_resolution=(240, 240)):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                if (width < min_resolution[0]) or (height < min_resolution[1]):
                    return True  # Upscale for resolution

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            
        return False  # No upscaling necessary
      
    def get_image_upscaled_path(self, img_full_name):
        # Create the new path by joining the directory with 'upscaled' and the original file name
        upscaled_dir = os.path.join(os.path.dirname(img_full_name), 'upscaled')
        upscaled_path = os.path.join(upscaled_dir, os.path.basename(img_full_name))

        # Create the 'upscaled' directory if it does not exist
        if not os.path.exists(upscaled_dir):
            os.makedirs(upscaled_dir)
        print(upscaled_path)
        return upscaled_path
    
    def is_blurry(self, image, threshold=100.0):
        gray_image = image.convert('L')
        laplacian = cv2.Laplacian(np.array(gray_image), cv2.CV_64F)
        variance = laplacian.var()
        return variance < threshold  # Returns True if blurry
 

    def process_images(self, df):

        for index, row in df.iterrows():
            img_path = row['image_path']
            label = row['label']

            # Check if the image should be upscaled
            if self.should_upscale(img_path, label):
                img_upscaled_path = self.get_image_upscaled_path(img_path)
                self.upscale_image(img_path, img_upscaled_path)
                df.at[index, 'image_path'] = img_upscaled_path
                
        return df




