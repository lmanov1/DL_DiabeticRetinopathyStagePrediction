import gradio as gr
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from efficientnet_pytorch import EfficientNet  # Import EfficientNet if using this library
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import cv2.dnn_superres  # Importing the dnn_superres module
import pdb
from sklearn.decomposition import PCA
import os





# Set this environment variable in your development environment
# For example, in Unix-based systems: export DEBUG_MODE='true'

DEBUG_MODE = True


def sharpen_image(image_tensor, withotsave=False):
    # Ensure the input tensor is in the format [B, C, H, W]
    assert image_tensor.dim() == 4, "Input tensor must be of shape [B, C, H, W]"

    # Prepare an empty tensor to hold sharpened images
    sharpened_tensor = torch.empty_like(image_tensor)

    # Iterate over each image in the batch (B)
    for i in range(image_tensor.size(0)):
        # Convert the image tensor to NumPy array, squeeze to remove the batch dimension
        image = image_tensor[i].permute(1, 2, 0).numpy()  # Shape: [H, W, C]

        # Convert from float32 to uint8 and scale to [0, 255]
        image = (image * 255).astype(np.uint8)

        # Define the sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        # Apply the sharpening filter
        sharpened = cv2.filter2D(image, -1, kernel)

        # Convert sharpened image back to tensor and scale to [0, 1]
        sharpened_tensor[i] = torch.from_numpy(sharpened.astype(np.float32) / 255.0).permute(2, 0,

                                                                                            1)  # Shape: [C, H, W]
        if withotsave:
            # Save the sharpened image as a file
            # Convert back to uint8 for saving
            sharpened_image = (sharpened_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # Shape: [H, W, C]
            sharpened_image_pil = Image.fromarray(sharpened_image)  # Convert to PIL Image
            sharpened_image_pil.save("sharpened_image.png")  # Save imag

    return sharpened_tensor


def laplacian_sharpen(image_tensor, save=False):
    assert image_tensor.dim() == 4 and image_tensor.size(0) == 1, "Input tensor must be of shape [1, C, H, W]"

    # Convert tensor to NumPy array
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Shape: [H, W, C]
    print(f"Original image shape: {image.shape}")  # Debug: Print shape of the original image
    print(f"Original image dtype: {image.dtype}")  # Debug: Print data type of the original image
    print(f"Original image min value: {image.min()}, max value: {image.max()}")  # Debug: Print min and max values

    # Ensure the image is in uint8 format
    image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
    print(f"Image after scaling to uint8 shape: {image.shape}")  # Debug: Print shape after scaling
    print(f"Image after scaling to uint8 dtype: {image.dtype}")  # Debug: Print data type after scaling
    print(f"Image after scaling to uint8 min value: {image.min()}, max value: {image.max()}")  # Debug: Print min and max values

    # Apply Laplacian sharpening
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    print(f"Laplacian shape: {laplacian.shape}")  # Debug: Print shape of the Laplacian
    print(f"Laplacian dtype: {laplacian.dtype}")  # Debug: Print data type of the Laplacian
    print(f"Laplacian min value: {laplacian.min()}, max value: {laplacian.max()}")  # Debug: Print min and max values

    # Convert the Laplacian result to uint8 for compatibility
    laplacian = cv2.convertScaleAbs(laplacian)  # Convert to uint8
    print(f"Laplacian after convertScaleAbs shape: {laplacian.shape}")  # Debug: Print shape after conversion
    print(f"Laplacian after convertScaleAbs dtype: {laplacian.dtype}")  # Debug: Print data type after conversion
    print(f"Laplacian after convertScaleAbs min value: {laplacian.min()}, max value: {laplacian.max()}")  # Debug: Print min and max values

    # Use addWeighted with compatible types
    sharp = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)
    print(f"Sharpened image shape: {sharp.shape}")  # Debug: Print shape of the sharpened image
    print(f"Sharpened image dtype: {sharp.dtype}")  # Debug: Print data type of the sharpened image
    print(f"Sharpened image min value: {sharp.min()}, max value: {sharp.max()}")  # Debug: Print min and max values

    # Convert back to tensor
    sharp_tensor = torch.from_numpy(sharp.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, C, H, W]

    if save:
        sharpened_image = (sharp_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        sharpened_image_pil = Image.fromarray(sharpened_image)  # Convert to PIL Image
        sharpened_image_pil.save("laplacian_sharpened_image.png")  # Save image

    return sharp_tensor


def high_pass_filter(image_tensor, save=False):
    assert image_tensor.dim() == 4 and image_tensor.size(0) == 1, "Input tensor must be of shape [1, C, H, W]"

    # Squeeze to HWC format
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Shape: [H, W, C]

    # Apply High Pass Filter
    low_pass = cv2.GaussianBlur(image, (21, 21), 0)
    high_pass = cv2.subtract(image, low_pass)

    # Ensure non-negative values and normalization for visualization
    high_pass = np.clip(high_pass, 0, None)
    enhanced_image = np.clip(image + high_pass * 10, 0, 255).astype(np.uint8)  # Drastic increase

    # Normalize to [0, 255] for enhanced image
    enhanced_image = cv2.normalize(enhanced_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Print min and max values for debugging
    print("High Pass Min:", high_pass.min(), "Max:", high_pass.max())
    print("Enhanced Image Min:", enhanced_image.min(), "Max:", enhanced_image.max())

    # Convert back to tensor
    enhanced_tensor = torch.from_numpy(enhanced_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    if save:
        enhanced_image_for_save = (enhanced_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        enhanced_image_pil = Image.fromarray(enhanced_image_for_save)  # Convert to PIL Image
        enhanced_image_pil.save("high_pass_filtered_image.png")  # Save image

    # Show the enhanced image using matplotlib
    plt.imshow(enhanced_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

    return enhanced_tensor

def enhance_contrast(image_tensor, save=False):
    assert image_tensor.dim() == 4 and image_tensor.size(0) == 1, "Input tensor must be of shape [1, C, H, W]"

    # Convert tensor to NumPy array
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Shape: [H, W, C]

    # Ensure the image is in uint8 format
    image = (image * 255).astype(np.uint8)  # Scale to [0, 255]

    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)  # Apply CLAHE to the L channel

    # Merge back the channels
    limg = cv2.merge((cl, a, b))
    contrast_enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Convert back to tensor
    contrast_tensor = torch.from_numpy(contrast_enhanced_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    if save:
        contrast_image_for_save = (contrast_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        contrast_image_pil = Image.fromarray(contrast_image_for_save)  # Convert to PIL Image
        contrast_image_pil.save("contrast_enhanced_image.png")  # Save image

    return contrast_tensor


def adjust_gamma(image_tensor, save=False, gamma=1.0):
    assert image_tensor.dim() == 4 and image_tensor.size(0) == 1, "Input tensor must be of shape [1, C, H, W]"

    # Convert the input tensor to numpy array
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Shape: [H, W, C]

    # Ensure image is in the correct range
    image = np.clip(image, 0, 1)  # Clip to [0, 1]

    # Scale to [0, 255]
    image = (image * 255).astype(np.uint8)

    # Apply gamma adjustment
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)], dtype=np.uint8)
    gamma_adjusted = cv2.LUT(image, table)

    # Convert back to tensor and normalize to [0, 1]
    gamma_tensor = torch.from_numpy(gamma_adjusted.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    if save:
        gamma_image_for_save = (gamma_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gamma_image_pil = Image.fromarray(gamma_image_for_save)  # Convert to PIL Image
        gamma_image_pil.save("gamma_adjusted_image.png")  # Save image

    return gamma_tensor

def denoise_image(image_tensor, save=False):
    assert image_tensor.dim() == 4 and image_tensor.size(0) == 1, "Input tensor must be of shape [1, C, H, W]"

    # Convert tensor to NumPy array
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()  # Shape: [H, W, C]

    # Ensure the image is in uint8 format (scale to [0, 255])
    image = (image * 255).astype(np.uint8)

    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Convert back to tensor
    denoised_tensor = torch.from_numpy(denoised_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    if save:
        denoised_image_for_save = (denoised_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        denoised_image_pil = Image.fromarray(denoised_image_for_save)  # Convert to PIL Image
        denoised_image_pil.save("denoised_image.png")  # Save image

    return denoised_tensor
def load_model(model_path: str, model_type: str, scale: int):
    # Create the super resolution model instance
    sr = cv2.dnn_superres.DnnSuperResImpl_create()  # Using the dnn_superres from cv2
    sr.readModel(model_path)  # Load the model
    sr.setModel(model_type, scale)  # Set the model type and scale
    return sr

def upscale_image(sr, image_path: str):
    # Read and upscale the image
    image = cv2.imread(image_path)
    result = sr.upsample(image)  # Upscale the image
    return result

def save_image(result, output_path: str):
    # Save the upscaled image
    cv2.imwrite(output_path, result)

def perform_pca_on_image(img, n_components=2):
    # Load the image and convert it to a NumPy array

    img_array = np.array(img)

    # Flatten the image array (height * width, color channels)
    img_flattened = img_array.reshape(-1, img_array.shape[-1])  # Shape: (num_pixels, num_channels)

    # Standardize the data
    scaler = StandardScaler()
    img_standardized = scaler.fit_transform(img_flattened)

    # Perform PCA
    pca = PCA(n_components=n_components)
    img_pca = pca.fit_transform(img_standardized)

    return img_pca



def load_pretrained_model(model_path, device='cpu'):
    """
    Load a model from a specified path based on file extension. Supports the following:
    - '.pth': Loads only state_dict (weights).
    - '.pt': Attempts to load the full model directly.
    - '.pkl': Uses torch's pickle-compatible loading.

    Args:
        model_path (str): Path to the model file.
        device (str): 'cuda' or 'cpu', default is 'cpu'.

    Returns:
        torch.nn.Module: The loaded model ready for inference or training.
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')

    # Get file extension
    _, file_extension = os.path.splitext(model_path)

    try:
        if file_extension == '.pth':
            # Load weights (state_dict) only


            model = EfficientNet.from_name('efficientnet-b7')
            model.load_state_dict(torch.load(model_path), strict=False)
            print("Loaded model as weights-only from .pth file.")

        elif file_extension == '.pt':
            # Load full model directly
            model = torch.load(model_path, map_location=device)
            print("Loaded full model from .pt file.")

        elif file_extension == '.pkl':
            # Load using pickle-compatible loading
            model = torch.load(model_path, map_location=device)
            print("Loaded full model from .pkl file.")

        else:
            raise ValueError(f"Unsupported file extension '{file_extension}'")

    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    # Move model to specified device and set to eval mode for inference
    model.to(device)
    model.eval()

    return model



def eval_start():
    # model_path = r"C:\Users\DELL\Documents\GitHub\DL_DiabeticRetinopathyStagePrediction\diabetic_retinopathy_BU17\model_full.pth"
    model_path = r"C:\Users\DELL\Documents\GitHub\DL_DiabeticRetinopathyStagePrediction\code\data\output\model_full.pth"
    pretrained_model = load_pretrained_model(model_path)
    pretrained_model.eval()  # Set model to evaluation mode
    return pretrained_model




# Transformation to preprocess the image
def preprocess_image(img):
    # need to refactor the classes to have separate per pretrained model instead one here
    # Data_preparations, train_model  #each image need to upscale it.

    ## 1
    # read how to use it...each image will be high resolution after that so when reduce to  resize we will get better quality.
    # model_path = "EDSR_x3.pb"  # Path to your model
    # model_type = "edsr"  # Model type
    # scale = 3  # Scale factor
    #
    # sr_model = load_model(model_path, model_type, scale)  # Load the model
    # upscaled_image = upscale_image(sr_model, "input_image.jpg")  # Path to your input image
    # save_image(upscaled_image, "upscaled_image.jpg")  # Path for saving the upscaled image

    ## 2
    print("Preprocessing image of type:", type(img))  # Debug print to check image type
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for EfficientNet input
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])
    img = preprocess(img).unsqueeze(0)  # Add batch dimension

    ## 3
        # Perform PCA on Image to reduce - Jpg
        # Input Tensor Shape: The shape of your image tensor before PCA is (1, 3, 224, 224), which corresponds to:
        # 1: Batch size (one image), 3: Color channels (RGB), 224: Height of the image, 224: Width of the image.
    # print(f"before pca {img.shape}")  # Output shape will be (num_pixels, n_components)
    # img = perform_pca_on_image(img, n_components=3)
    # print(f"after pca {img.shape}")   # Output shape will be (num_pixels, n_components)
    # img = img.fit_transform(img)

    return img
    ## 4
    # Assuming img is your input tensor of shape [1, 3, 224, 224]
    img = sharpen_image(img)  # First sharpen the image
    print(img.shape)  # Check the shape after each processing step

    # Denoising the image before applying sharpness enhancement
    img = denoise_image(img, True)  # Denoise to reduce noise artifacts
    print(img.shape)  # Check the shape after each processing step

    img = laplacian_sharpen(img, True)  # Then apply Laplacian sharpening
    print(img.shape)  # Check the shape after each processing step

    img = high_pass_filter(img, True)  # High pass filtering
    print(img.shape)  # Check the shape after each processing step

    img = enhance_contrast(img, True)  # Contrast enhancement
    print(img.shape)  # Check the shape after each processing step

    img = adjust_gamma(img, True)  # Adjust gamma to correct brightness
    print(img.shape)  # Check the shape after each processing step

    # Ensure the output is within the correct range for the next operation
    img = img.clamp(0, 1)  # Clamp the values to [0, 1]
    print(img.shape)  # Check the shape after each processing step

    return img

# Global Scope:


model = eval_start()


def classify_image(img, img_file):
    # 1. **Image Input Handling**
    if img is None:
        print("No image received. Please upload an image.")
        return "No image uploaded", {}, None

    # Debugging: Check types of inputs
    print("Type of img:", type(img))
    print("Type of img_file:", type(img_file))

    # Get the file name from img_file
    img_name = img_file.name if img_file else "Unknown"

    # If the image was uploaded through the img component and the file component is not set,
    # we can set the img_file.name to the file name from the image input.
    if img_file is None and hasattr(img, 'filename'):
        img_name = img.filename

    # 2. **Image Preprocessing**
    img_tensor = preprocess_image(img)
    print("Image tensor shape:", img_tensor.shape)  # Debugging: Shape check

    # 3. **Model Predictions and Probability Calculation**
    with torch.no_grad():
        preds = model(img_tensor)
        print("Raw predictions (logits):", preds)  # Check raw logits
        probs = torch.softmax(preds, dim=1)  # Softmax for probabilities

    # Dynamic class labels and their descriptions
    class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
    class_aliases = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative DR"]

    # Calculate class probabilities
    class_probs = {class_labels[i]: float(probs[0][i] * 100) for i in range(len(class_labels))}
    print("Class probabilities before normalization:", class_probs)  # Debugging: Check probabilities

    # 4. **Normalization of Probabilities**
    prob_values = np.array(list(class_probs.values()))

    if np.any(prob_values > 0):
        non_zero_probs = prob_values[prob_values > 0]
        total_non_zero = np.sum(non_zero_probs)
        normalized_probs = (non_zero_probs / total_non_zero) * 100
    else:
        normalized_probs = prob_values  # If all are zero

    print("Normalized probabilities:", normalized_probs)  # Debugging

    # Build final_probs with original keys and integer conversion
    final_probs = {key: int(value) for key, value in zip(class_labels, normalized_probs)}
    final_probs = {key: final_probs.get(key, 0) for key in class_labels}

    # 5. **Final Probability Calculation and Adjustment**
    total_int = sum(final_probs.values())
    print("Final class probabilities:", final_probs)  # Debugging
    print("Sum of final probabilities:", total_int)  # Debugging

    if total_int != 100:
        max_class = max(final_probs, key=final_probs.get)
        final_probs[max_class] += 100 - total_int  # Adjust sum to 100

    # 6. **Visualization and Output**
    sorted_class_probs = dict(sorted(final_probs.items(), key=lambda item: item[1], reverse=True))
    print("Sorted class probabilities:", sorted_class_probs)

    # Plotting the histogram
    plt.figure(figsize=(8, 4))
    plt.bar(sorted_class_probs.keys(), sorted_class_probs.values(), color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Probabilities (%)')
    plt.title('Normalized Class Probabilities')
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.grid(axis='y')

    # Create a legend dynamically
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{class_labels[i]}: {class_aliases[i]}",
                          markerfacecolor='skyblue', markersize=10) for i in range(len(class_labels))]
    plt.legend(handles=handles, title="Class Index", bbox_to_anchor=(1, 1), loc='upper left')

    # Save the plot to a file with a specific path
    plt.savefig('class_probabilities.png', bbox_inches='tight')  # Save the histogram as an image file
    plt.close()  # Close the figure to avoid display issues

    # Return the file name and the sorted class probabilities
    return img_name, sorted_class_probs, 'class_probabilities.png'

# Function to classify image and get percentages
# def classify_image(img, img_file):
#     if img is None:
#         print("No image received. Please upload an image.")  # Debug print for None image
#         return "No image uploaded", {}, None
#
#     # Print the type of img and img_file for debugging
#     print("Type of img:", type(img))
#     print("Type of img_file:", type(img_file))
#
#     # Get the file name from img_file, but only if img_file is not None
#     img_name = img_file.name if img_file else "Unknown"
#
#     # If the image was uploaded through the img component and the file component is not set,
#     # we can set the img_file.name to the file name from the image input.
#     if img_file is None and hasattr(img, 'filename'):
#         img_name = img.filename  # Update img_name to reflect the file name of the uploaded image
#
#     # Preprocess the image
#     img_tensor = preprocess_image(img)
#
#     # Get model predictions and convert logits to probabilities
#     with torch.no_grad():
#         preds = model(img_tensor)
#         probs = torch.softmax(preds, dim=1)  # Apply softmax for probability distribution
#
#     # Dynamic class labels and their descriptions
#     class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
#     class_aliases = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative DR"]
#     # Assume class_labels is defined somewhere above this snippet
#     class_probs = {class_labels[i]: float(probs[0][i] * 100) for i in range(len(class_labels))}  # Convert to percentage
#
#     # Create an array of probabilities for normalization
#     prob_values = np.array(list(class_probs.values()))
#
#     # Normalize only non-zero probabilities
#     if np.any(prob_values > 0):
#         non_zero_probs = prob_values[prob_values > 0]
#         total_non_zero = np.sum(non_zero_probs)
#         normalized_probs = (non_zero_probs / total_non_zero) * 100
#     else:
#         normalized_probs = prob_values  # If all are zero
#
#     # Build final_probs with original keys and integer conversion
#     final_probs = {key: int(value) for key, value in zip(class_labels, normalized_probs)}
#     final_probs = {key: final_probs.get(key, 0) for key in class_labels}  # Ensure all keys are present
#
#
#     # class_probs = {class_labels[i]: float(probs[0][i] * 100) for i in range(5)}  # Convert to percentage
#     #
#     # # Normalize the non-zero probabilities
#     # non_zero_probs = {key: value for key, value in class_probs.items() if value > 0}
#     # total_non_zero = sum(non_zero_probs.values())
#     #
#     # if total_non_zero > 0:
#     #     # Normalize each non-zero value
#     #     normalized_probs = {key: (value / total_non_zero) * 100 for key, value in non_zero_probs.items()}
#     # else:
#     #     normalized_probs = class_probs  # If all are zero
#     #
#     # # Add back the zero classes with their original values
#     # final_probs = {key: normalized_probs.get(key, 0) for key in class_labels}
#     #
#     # # Cast to integer
#     # int_probs = {key: int(value) for key, value in final_probs.items()}
#
#     # Ensure the sum is 100
#     total_int = sum(final_probs.values())
#     if total_int != 100:
#         # Find the class with the highest probability to increment
#         max_class = max(final_probs, key=final_probs.get)
#         final_probs[max_class] += 100 - total_int  # Adjust to make the sum 100
#
#     # Sort probabilities in descending order for display
#     sorted_class_probs = dict(sorted(final_probs.items(), key=lambda item: item[1], reverse=True))
#
#     # Print the final probabilities for debugging
#     print("Normalized class probabilities:", sorted_class_probs)
#
#     # Plotting the histogram with legend
#     plt.figure(figsize=(8, 4))
#     plt.bar(sorted_class_probs.keys(), sorted_class_probs.values(), color='skyblue')
#     plt.xlabel('Classes')
#     plt.ylabel('Probabilities (%)')
#     plt.title('Normalized Class Probabilities')
#     plt.xticks(rotation=0)  # Set x-axis labels to horizontal
#     plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
#     plt.grid(axis='y')
#
#     # Create a legend dynamically
#     handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{class_labels[i]}: {class_aliases[i]}",
#                            markerfacecolor='skyblue', markersize=10) for i in range(len(class_labels))]
#     plt.legend(handles=handles, title="Class Index", bbox_to_anchor=(1, 1), loc='upper left')
#
#     # Save the plot to a file
#     # plt.savefig('class_probabilities.png', bbox_inches='tight')
#     #plt.show()
#     plt.close()  # Close the figure to avoid display issues
#
#     # Return the file name and the sorted class probabilities for Gradio
#     return img_name, sorted_class_probs, 'class_probabilities.png'

# Gradio interface setup
iface = gr.Interface(
    fn=classify_image,
    inputs=[gr.Image(type="pil", label="Upload Retina Image"),
            gr.File(label="Upload Image File", elem_id="image_input")],
    outputs=[gr.Textbox(label="File Name", elem_id="out_textbox"),
             gr.JSON(label="Classification Probabilities (%)"),
             gr.Image(label="Class Probabilities Histogram")],
    title="Diabetic Retinopathy Classification",
    description="Upload a retina image to classify its diabetic retinopathy stage and view probabilities for each class (0 to 4).",
    css="""
     #image_input 
     {
      width: 1px; /* Width of the upload image area */
      height: 1px; /* Height of the upload image area */
      overflow: hidden; /* Prevent any overflow */
      visibility: hidden; /* Make it invisible */
        }
     #out_textbox 
     {
      width: 1px; /* Width of the upload image area */
      height: 1px; /* Height of the upload image area */
      overflow: hidden; /* Prevent any overflow */
      visibility: hidden; /* Make it invisible */
        }
    """
)
# Launch the Gradio app with a public link
iface.launch()
