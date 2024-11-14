from torchvision import transforms
from HuggingFace_.app import sharpen_image, denoise_image, laplacian_sharpen, high_pass_filter, enhance_contrast, adjust_gamma
from PIL import Image


def image_manip(img: Image ):
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
    # print("Preprocessing image of type:", type(img))  # Debug print to check image type
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize for EfficientNet input
    #     transforms.ToTensor(),  # Convert to tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    # ])
    # img = preprocess(img).unsqueeze(0)  # Add batch dimension

    ## 3
        # Perform PCA on Image to reduce - Jpg
        # Input Tensor Shape: The shape of your image tensor before PCA is (1, 3, 224, 224), which corresponds to:
        # 1: Batch size (one image), 3: Color channels (RGB), 224: Height of the image, 224: Width of the image.
    # print(f"before pca {img.shape}")  # Output shape will be (num_pixels, n_components)
    # img = perform_pca_on_image(img, n_components=3)
    # print(f"after pca {img.shape}")   # Output shape will be (num_pixels, n_components)
    # img = img.fit_transform(img)


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

# img_tensor = preprocess_image(img)
# print("Image tensor shape:", img_tensor.shape)  # Debugging: Shape check