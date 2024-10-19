from pathlib import Path
import os
#from uuid import uuid4

REPO_ID = "lmanov1/timm-efficientnet_b3.ra2_in1k"
DATASET_REPO_ID = "lmanov1/BUI17_data"
#MODEL_LOCAL_FILE_PATH = Path("../../data/output/trainLabels19_pretrained_export.pth")
MODEL_LOCAL_FILE_PATH = Path("../../data/output/pretrained_trainLabels19_export.pkl")
MODEL_DATASET_DIR = Path(DATASET_REPO_ID)
#MODEL_DATASET_PATH = MODEL_DATASET_DIR / f"{MODEL_LOCAL_FILE_PATH.name}-{uuid4()}.pth"
MODEL_DATASET_PATH = MODEL_DATASET_DIR / f"{MODEL_LOCAL_FILE_PATH.name}"
# origanization : MLSHYAorg