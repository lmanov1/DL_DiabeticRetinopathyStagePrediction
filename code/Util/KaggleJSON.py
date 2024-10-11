import os
import json


def ensure_and_set_kaggle_creds(api_token_path, default_username="your_kaggle_username",
                                default_key="your_kaggle_api_key"):
    """
    Ensures the kaggle.json file exists and sets Kaggle credentials as environment variables.

    :param api_token_path: Path where the kaggle.json file should be created or checked.
    :param default_username: Default username for Kaggle API.
    :param default_key: Default key for Kaggle API.
    """
    kaggle_dir = os.path.dirname(api_token_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)

    # Check if the kaggle.json file exists
    if not os.path.exists(api_token_path):
        # Create the kaggle.json file with default data (Replace with actual API token)
        kaggle_json = {
            "username": default_username,
            "key": default_key
        }
        with open(api_token_path, 'w') as f:
            json.dump(kaggle_json, f)
        print(f"Created kaggle.json at {api_token_path}")
    else:
        print(f"kaggle.json already exists at {api_token_path}")

    # Load the kaggle.json file and set environment variables
    with open(api_token_path, 'r') as f:
        kaggle_creds = json.load(f)

    os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
    os.environ['KAGGLE_KEY'] = kaggle_creds['key']
    print("Kaggle credentials loaded and environment variables set.")


# Usage
# kaggle_json_path = 'data/kaggle.json'
# ensure_and_set_kaggle_creds(kaggle_json_path)