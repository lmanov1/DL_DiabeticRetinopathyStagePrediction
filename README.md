# DL_DiabeticRetinopathyStagePrediction - Diabetic Retinopathy Stage Prediction With Deep Learning

This repository hosts a final DL project conducted as part of the data scientist certification at BIU.

Contributors:
- Alon Shmueli
- Guy Yarhi
- Einav Lapid
- Lidia Manov

## Project Goals and Background

Diabetic retinopathy is a complication of diabetes mellitus that affects the eyes by damaging the blood vessels in the retina. If not diagnosed and managed in time, it can lead to serious consequences, including vision loss. As diabetes is very common, early, accurate, and efficient Diabetic Retinopathy (DR) stage classification is crucial for improving patient outcomes and preventing vision loss. AI systems make DR screening more accessible, especially in remote or underserved areas. Automation through AI can accurately identify early signs of diabetic retinopathy (DR). This project aims to develop a high-performing deep learning model for DR stage classification, achieving high accuracy, precision, recall, and F1-score. The model's performance will be thoroughly evaluated, and the results will be analyzed for potential improvements.

This project leverages customized pre-trained CNN models (e.g., EfficientNet-B7) and fine-tunes the model for the specific classification task using transfer learning techniques. See the project presentation (details below).

### Prerequisites

- Ensure you have Poetry installed in your environment.
- If you have an NVIDIA GPU card, ensure you have the `nvidia-smi` CLI installed. To detect and install supporting Python system libraries (unsupported by Poetry) like CUDA, run:
    ```bash
    python3 code/Util/check_hardware_and_install.py
    ```

## Environment and Installation

1. Clone the repository and navigate to the cloned directory:
    ```bash
    git clone git@github.com:lmanov1/DL_DiabeticRetinopathyStagePrediction.git
    ```
2. Run `poetry update` (just once) - this will not use the `poetry.lock` of the workspace but will rewrite it. Do not commit your `poetry.lock`. Without an available GPU (and CUDA), TensorFlow, PyTorch, and other libraries leveraging the GPU will automatically use the CPU.
3. Run `poetry shell`.
4. We use Kaggle API to download datasets from Kaggle. To use the Kaggle API, sign up for a Kaggle account at [Kaggle](https://www.kaggle.com). Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the appropriate location for your operating system:

    - Linux: `$XDG_CONFIG_HOME/kaggle/kaggle.json` (defaults to `~/.config/kaggle/kaggle.json`). The path `~/.kaggle/kaggle.json`, used by older versions of the tool, is also still supported.
        ```bash
        chmod 600 ~/.config/kaggle/kaggle.json
        ```
    - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json` - you can check the exact location, sans drive, with `echo %HOMEPATH%`.
    - Other: `~/.kaggle/kaggle.json`.

    - You can define a shell environment variable `KAGGLE_CONFIG_DIR` to change this location to `$KAGGLE_CONFIG_DIR/kaggle.json` (on Windows it will be `%KAGGLE_CONFIG_DIR%\kaggle.json`).

    - You can also choose to export your Kaggle username and token to the environment:
        ```bash
        export KAGGLE_USERNAME=datadinosaur
        export KAGGLE_KEY=xxxxxxxxxxxxxx
        ```
    In addition, you can export any other configuration value that would normally be in the `kaggle.json` in the format `KAGGLE_` (note uppercase).

## Getting Started

After your Poetry environment is initialized locally with `poetry shell`, you can run project logic from the code directory:
```bash
python3 -m /code/main.py
```
However, more advanced model versions are in the notebooks directory (all EfficientNet-B7-based models). Model versions are not kept in this Git repository but can be reproduced by running notebooks. The latest and greatest one - `notebooks/v4.6.4_10epochs-balanced.ipynb`.

## Project Blueprint: Diabetic Retinopathy Severity Classification

### Project Setup

- Dev Environment Setup: Poetry
- Existing code verified on both CPU and GPU environments, Windows, Linux, and Mac.

### Data Acquisition

- Download datasets from Kaggle: 2015 / 2019 Diabetic Retinopathy Detection, APTOS 2019 Blindness Detection (benjaminwarner/resized-2015-2019-blindness-detection-images, mariaherrerot/aptos2019).
- Run `notebooks/DownloadDatasets.ipynb` to define the dataset to download and organize it in a structured format (images grouped per class).

### Data Preprocessing

- Clean and normalize images.
- Implement data augmentation techniques.
- Resize images.
- Split data into training, validation, and test sets.

### Model Definition

- Define the main `EyeDiseaseClassifier` class using a pre-trained model of choice (e.g., EfficientNet-B7) with its pretrained weights.
- Replace the final classifier layers to match the number of classes.
- As the number of classes can be set at model initialization, the model can theoretically be used to train on other eye diseases with relevant datasets. However, the data pipeline should be updated accordingly.

### Training Process

- Use FastAI and PyTorch.
- Load data and create data loaders.
- Use data loaders with a weighted random sampler to reduce class imbalance.
- Use pre- and post-batch data processing steps.
- Train the model with the available dataset(s) or their combination (to compensate for existing class imbalance).

### Model Evaluation

- Evaluate the model on the validation set.
- Calculate metrics like accuracy, precision, recall, F1-score, and ROC-AUC. The most valuable metrics are Recall and F1-score.
- Plot AUC curves, confusion matrix, classification reports ,top losses, smaple predictions for better model evaluation.
- Integrated TensorBoard to keep a history of training and evaluation , all the above, interactive model graph

### Model Validation - available as a feature on Hagging face and in the code (in notebooks)

### Model Saving

- Save the trained models using appropriate file formats (.pth for PyTorch).

### Production - Deploy Model to Hugging Face

#### Deploy Fine-Tuned Model(s) to Hugging Face with a Graphical UI

The UI provides the ability for both inference with a single retina fundus image(s) validation and for model validation with a pre-saved test set. Like on the validation stage, all the same metrics like accuracy, precision, recall, F1-score are calculated, and plots of AUC curves, confusion matrix, top losses, and sample predictions are presented for better model validation.

Hugging Face files are in the `code/production` folder.
- `app.py` - main Gradio application script.
- `update_production.py` - upload relevant files from Git to the production space (`https://huggingface.co/spaces/Lmanov1/timm-efficientnet_b3.ra2_in1k`).

For inference models' persistent storage, we use a dataset repository (`https://huggingface.co/datasets/Lmanov1/BUI17_data`). To run `update_production.py`, you need to log in to Hugging Face with a valid Hugging Face API TOKEN. Currently, the token value is read from the `.env` file, where it should be stored in the format: `MY_TOKEN="PUT YOUR KEY HERE"`. This file is not managed by Git but should be local in the root directory of the project and contain your token.

### Documentation

#### Project Documentation

- Maintain detailed documentation for each step.
- Include README files, code comments, and usage instructions.
- Project presentation file: `presentation/DLStagePredictPresentation.pptx`.

## Update requirements.txt

```bash
poetry export -f requirements.txt --output requirements.txt
```

### Stretch Features

#### LLM Assistant Integration

- Integrate an LLM assistant to act as a virtual doctor.
- Implement functionalities for anamnesis, analysis, and providing recommendations.

#### Additional Disease Classification

- Extend the model to classify other eye diseases like cataracts and glaucoma.
- Adjust the label set and retrain the model accordingly.

### Python Environment

- Python 3.10.12 (must be the active version on your system). If you have another Python version, use `poetry update` instead of `poetry install` during step 5 and do not commit `poetry.lock`.
- Poetry 1.8.3
- Pip 24.2